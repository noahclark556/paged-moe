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

from test_stream import CONFIG, build_tiny_model, greedy_generate

from expert_stream import config as es_config
from expert_stream import get_stats, load
from expert_stream.streaming import StreamedSwitchGLU


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

        es_config.FLOW = True
        es_config.FLOW_TOPM = 0
        try:
            print("== flow: all experts, all resident, one layer ==")
            model_u, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
            check_run_flow_matches_run_slab(model_u)
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

    print("\nOK: flow decode tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
