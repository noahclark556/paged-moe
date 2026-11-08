# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Correctness test for flow decode (config.FLOW): one sync per token instead of
one per MoE layer, with expert->slot resolved from a GPU table.

Three properties matter, and together they are why flow can be trusted:

  1. `_run_flow` with every expert computed and every expert resident is
     *bit-identical* to `_run_slab`. Same kernel, same slots - the only thing
     that changed is where the slot numbers came from. Checked directly on one
     layer, because end-to-end residency is never total (prefill hands the slab
     back, and the slab reserve is smaller than the expert count).
  2. Below the router's top-k it becomes a fidelity knob: output may move, but
     logits stay finite and the drop shows up on the counters.
  3. Its preconditions are real. With the slab off, flow declines and the model
     runs the ordinary path with unchanged output.

Run:

    python tests/test_flow.py
"""

import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_stream import CONFIG, build_tiny_model, greedy_generate

from pagedmoe import config as es_config
from pagedmoe import get_stats, load
from pagedmoe.streaming import StreamedSwitchGLU, exact_step


def first_moe_layer(model) -> StreamedSwitchGLU:
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        glu = getattr(mlp, "switch_mlp", None)
        if isinstance(glu, StreamedSwitchGLU):
            return glu
    raise AssertionError("no streamed MoE layer found")


def check_run_flow_matches_run_slab(model):
    """Property 1: same weights, same slots, same bytes out."""
    glu = first_moe_layer(model)
    cache = glu.cache
    k = CONFIG["num_experts_per_tok"]

    # Make a known set of experts resident, then address them both ways.
    ids = list(range(k))
    experts = cache.fetch(glu.layer_key, ids, install="lru")
    slots = {e: ent["__slot__"] for e, ent in experts.items() if "__slot__" in ent}
    assert len(slots) == len(ids), f"experts did not all get slots: {slots}"

    table = cache.flow_table(glu.layer_key)
    assert table is not None, "flow table missing"
    tnp = np.asarray(table)
    for e, slot in slots.items():
        assert tnp[e] == slot, f"table disagrees on expert {e}: {tnp[e]} != {slot}"

    mx.random.seed(11)
    x_flat = mx.random.normal((1, CONFIG["hidden_size"])).astype(mx.bfloat16)
    ind2 = mx.array([ids], dtype=mx.uint32)
    # Rank is only used to order a top-M selection; with topm=0 it is unused,
    # so any finite values exercise the same path production would take.
    rank = mx.array([[4.0, 3.0, 2.0, 1.0]], dtype=mx.float32)

    ref = glu._run_slab(x_flat, np.array([ids]), slots, keep=None)
    got, valid = glu._run_flow(x_flat, ind2, rank, table, 0)
    mx.eval(ref, got, valid)

    assert bool(mx.all(valid)), "every expert was resident but valid says otherwise"
    assert ref.shape == got.shape, f"{ref.shape} != {got.shape}"
    a, b = np.asarray(ref.astype(mx.float32)), np.asarray(got.astype(mx.float32))
    assert np.array_equal(a, b), (
        "flow is not bit-identical to the slab path with all experts resident; "
        f"max |delta| = {np.abs(a - b).max()}"
    )
    print(f"OK: _run_flow == _run_slab bit-for-bit on {a.shape} output")


def check_flow_admit_installs(model):
    """flow_admit must leave the experts resident, in the table, and recent.

    Routing it through prefetch left every read in staging (only fetch
    promotes), so the resident set froze and flow computed almost nothing.
    """
    import time

    glu = first_moe_layer(model)
    cache = glu.cache
    n = CONFIG["num_experts"]
    ids = [n - 1, n - 2]
    for f in cache.flow_admit([(glu.layer_key, ids)]):
        f.result()
    deadline = time.time() + 5
    keys = [(glu.layer_key, e) for e in ids]
    while not all(k in cache._lru for k in keys) and time.time() < deadline:
        time.sleep(0.01)
    assert all(k in cache._lru for k in keys), "flow_admit left reads uninstalled"
    table = np.asarray(cache.flow_table(glu.layer_key))
    assert all(table[e] >= 0 for e in ids), f"table missing admitted experts: {table}"

    other = (glu.layer_key, 0)
    cache.fetch(glu.layer_key, [0], install="lru")
    assert list(cache._lru)[-1] == other
    hits = cache.hits
    cache.flow_admit([(glu.layer_key, [ids[0]])])
    assert cache.hits == hits + 1, "a resident id should count as a hit"
    assert list(cache._lru)[-1] == keys[0], "a resident id should become most recent"
    print("OK: flow_admit installs, fills the table, and refreshes recency")


def check_critical_gate(model):
    """Flow runs only while the cache holds what the router weights heavily.

    Qwen3-235B passed the hit-rate gate at 0.90 and flow then dropped enough
    top-weighted experts to garble agent output. The critical-miss EMA has to
    keep flow off there, hand back when it climbs mid-turn, and only re-enter
    with room to spare.
    """
    glu = first_moe_layer(model)
    ring, cache = glu.ring, glu.cache
    ring.flow_drop()
    # The gate counts per token, summed over layers: 0.05/layer here.
    es_config.FLOW_MAX_CRIT = 0.05 * len(ring.layers)

    def step(crit, n=10):
        for _ in range(n):
            cache.hits += 100
            cache.crit_misses += crit
            cache.crit_steps += 100
            ring.flow_token_start()

    step(20, 4)
    assert not ring.flow_ready(), "warm by hits but 0.2 critical misses/layer"
    step(0, 8)
    assert ring.flow_ready(), "critical misses fell under the bar, flow stayed off"
    step(30, 3)
    assert not ring.flow_ready(), "critical misses climbed, flow did not hand back"
    step(0, 4)
    assert not ring.flow_ready(), "re-entered under the bar but above half of it"
    step(0, 3)
    assert ring.flow_ready(), "never re-entered after the cache recovered"
    ring.flow_drop()
    print("OK: critical-miss gate keeps flow off, hands back, re-enters at half")


def check_flow_drafts(tmp: Path, prompt):
    """Self-drafts run on the flow graph; plain decode never does.

    DRAFT_FLOW builds the tables with FLOW off, so it has to leave the
    ordinary path alone, and drafts on it still only ever reach the output
    through a verify pass.
    """
    from pagedmoe import streaming
    from pagedmoe import verify_decode as vd
    from tests.test_verify_decode import _Force, _run, greedy_given_prefix

    calls = {"n": 0}
    orig = StreamedSwitchGLU._draft_flow

    def counted(self, *a):
        calls["n"] += 1
        return orig(self, *a)

    StreamedSwitchGLU._draft_flow = counted
    old_self = es_config.SELF_DRAFT
    try:
        es_config.DRAFT_FLOW = False
        model_o, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
        assert not streaming.draft_flow_on(model_o), "DRAFT_FLOW=0 still built flow tables"
        del model_o
        es_config.DRAFT_FLOW = True

        model, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
        assert streaming.draft_flow_on(model), "flow drafts not available with the slab on"
        ref = greedy_generate(model, prompt, 16)
        assert calls["n"] == 0, f"plain decode went through the draft path {calls['n']} times"

        es_config.SELF_DRAFT = "1"
        with _Force(lambda toks, k: []):
            got, _, _ = _run(model, prompt, 16)
        st = dict(vd.STATS)
        assert st["self"] > 0 and calls["n"] > 0, f"self drafts did not use flow: {st} {calls}"
        greedy_given_prefix(model, prompt, got)
        moved = "same" if got == ref else "moved within rounding"
        print(f"OK: flow self-drafts ({st['self']} drafts, {calls['n']} flow layers), output greedy ({moved})")
        del model
    finally:
        StreamedSwitchGLU._draft_flow = orig
        es_config.SELF_DRAFT = old_self
        es_config.DRAFT_FLOW = True


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build_tiny_model(tmp)

        rng = np.random.default_rng(3)
        prompt = rng.integers(0, CONFIG["vocab_size"], size=50).tolist()
        n_new = 20

        print("== reference: streamed, per-layer path ==")
        model_ref, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
        ref = greedy_generate(model_ref, prompt, n_new)
        del model_ref

        print("== flow-backed self drafts ==")
        check_flow_drafts(tmp, prompt)

        es_config.FLOW = True
        es_config.FLOW_TOPM = 0
        try:
            print("== flow stays off at PRUNE 0 (the exact setting) ==")
            model_z, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
            got_z = greedy_generate(model_z, prompt, n_new)
            assert got_z == ref, f"FLOW at PRUNE 0 changed output\n  got={got_z}"
            assert get_stats(model_z)["cache"]["pruned_frac"] == 0.0
            print("OK: output identical, nothing skipped")
            del model_z

            # Epsilon keeps every slot in the weight test, so what follows
            # measures flow itself rather than PRUNE.
            es_config.PRUNE = 1e-9

            print("== flow waits for a warm cache ==")
            es_config.FLOW_WARM_HIT = 1.01
            model_w, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
            got_w = greedy_generate(model_w, prompt, n_new)
            assert got_w == ref, f"cold gate still let flow run\n  got={got_w}"
            print("OK: never warm enough, per-layer path, output unchanged")
            del model_w
            es_config.FLOW_WARM_HIT = 0.9

            print("== flow: all experts, all resident, one layer ==")
            model_u, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
            check_run_flow_matches_run_slab(model_u)
            check_flow_admit_installs(model_u)
            check_critical_gate(model_u)
            es_config.FLOW_MAX_CRIT = 5.0
            del model_u

            print("== flow: end to end, all experts ==")
            model_f, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
            _, logits = greedy_generate(model_f, prompt, n_new, want_logits=True)
            stats = get_stats(model_f)
            assert stats.get("flow"), "flow did not engage"
            assert np.isfinite(logits).all(), "flow produced non-finite logits"
            base_drop = stats["cache"]["pruned_frac"]
            print(f"OK: flow on, logits finite, pruned_frac={base_drop}")
            del model_f

            print("== exact_step: full mixture under PRUNE, WAIT_ABOVE and flow ==")
            es_config.PRUNE = 0.9
            es_config.WAIT_ABOVE = 0.3
            # Everything is resident here, so KEEP_FREE would put back all PRUNE drops.
            es_config.KEEP_FREE = False
            model_x, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
            with exact_step():
                got_x = greedy_generate(model_x, prompt, n_new)
            assert got_x == ref, f"exact_step did not give the exact output\n  got={got_x}"
            assert get_stats(model_x)["cache"]["pruned_frac"] == 0.0
            got_p = greedy_generate(model_x, prompt, n_new)
            assert get_stats(model_x)["cache"]["pruned_frac"] > 0.0, (
                "PRUNE 0.9 outside exact_step skipped nothing, so the check above "
                "proved nothing"
            )
            moved = "same" if got_p == ref else "moved"
            print(f"OK: exact output under PRUNE 0.9 (outside it, output {moved})")
            del model_x
            es_config.PRUNE = 1e-9
            es_config.WAIT_ABOVE = 0.0
            es_config.KEEP_FREE = True

            print("== flow: top-M below the router's k is a fidelity knob ==")
            es_config.FLOW_TOPM = 2  # router top-k is 4 here
            model_m, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
            _, logits_m = greedy_generate(model_m, prompt, n_new, want_logits=True)
            stats_m = get_stats(model_m)
            assert np.isfinite(logits_m).all(), "top-M produced non-finite logits"
            assert stats_m["cache"]["pruned_frac"] > base_drop, (
                "halving the computed experts dropped no extra slots: "
                f"{stats_m['cache']['pruned_frac']} vs {base_drop}"
            )
            print(f"OK: pruned_frac={stats_m['cache']['pruned_frac']} (was {base_drop})")
            del model_m

            print("== flow declines when the slab is off ==")
            es_config.FLOW_TOPM = 0
            es_config.SLAB_BYTES = 0
            model_n, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
            assert not get_stats(model_n).get("flow"), (
                "flow claimed to be on without a slab"
            )
            got_n = greedy_generate(model_n, prompt, n_new)
            assert got_n == ref, f"no-slab fallback changed output\n  got={got_n}"
            print("OK: fell back to the per-layer path, output unchanged")
            del model_n
        finally:
            es_config.FLOW = False
            es_config.FLOW_TOPM = 0
            es_config.SLAB_BYTES = None
            es_config.PRUNE = 0.0
            es_config.WAIT_ABOVE = 0.0
            es_config.KEEP_FREE = True
            es_config.FLOW_WARM_HIT = 0.9
            es_config.FLOW_MAX_CRIT = 5.0

    print("\nOK: flow decode tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
