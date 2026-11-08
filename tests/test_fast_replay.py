# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Fast replay on hybrid caches (fast_replay.py).

What has to hold:

  1. On every hybrid layout here, a fast replay after a pass leaves the cache
     where feeding only the kept tokens would: same attention offsets, same
     recurrent state (within kernel rounding), same side caches.
  2. In the decode loop the output stays the greedy pick and fast replays run
     after the per-load checks pass.
  3. A check that does not match turns it off for the model, and the cache is
     the full replay's.
  4. A replay that fails part way restores the snapshot and falls back to the
     full replay, so the cache still ends right.

Run:
    python tests/test_fast_replay.py
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models import jamba, lfm2_moe, nemotron_h, qwen3_5_moe  # noqa: E402
from mlx_lm.models.cache import ArraysCache, make_prompt_cache  # noqa: E402

from tests.test_lookup_hybrid import ARGS as NEXT_ARGS  # noqa: E402
from tests.test_lookup_hybrid import build as build_next  # noqa: E402
from tests.test_lookup_hybrid import check_greedy  # noqa: E402

from pagedmoe import config as es_config  # noqa: E402
from pagedmoe import fast_replay  # noqa: E402
from pagedmoe import verify_decode as vd  # noqa: E402
from pagedmoe.lookup_decode import _snapshot  # noqa: E402


def _sharpen(model):
    model.set_dtype(mx.float32)
    model.update(nn.utils.tree_map(lambda p: p * 3.0 if p.ndim == 2 else p, model.parameters()))
    mx.eval(model.parameters())
    return model


def build_qwen35():
    mx.random.seed(7)
    text = {k: v for k, v in NEXT_ARGS.items() if k != "model_type"}
    text.update(model_type="qwen3_5_moe_text", layer_types=None)
    args = qwen3_5_moe.ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": text})
    return _sharpen(qwen3_5_moe.Model(args)), NEXT_ARGS["vocab_size"]


def build_lfm2():
    mx.random.seed(8)
    args = lfm2_moe.ModelArgs.from_dict(dict(
        model_type="lfm2_moe", vocab_size=96, hidden_size=64, intermediate_size=64,
        moe_intermediate_size=32, num_hidden_layers=4, num_experts=4, num_experts_per_tok=2,
        norm_topk_prob=True, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=1024, use_expert_bias=False, num_dense_layers=1,
        norm_eps=1e-5, conv_bias=False, conv_L_cache=3, full_attn_idxs=[1, 3],
        layer_types=["conv", "full_attention", "conv", "full_attention"],
    ))
    return _sharpen(lfm2_moe.Model(args)), 96


def build_jamba():
    mx.random.seed(9)
    args = jamba.ModelArgs.from_dict(dict(
        model_type="jamba", hidden_size=64, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, attn_layer_offset=1, attn_layer_period=2,
        expert_layer_offset=1, expert_layer_period=2, mamba_d_conv=4, mamba_d_state=8,
        mamba_expand=2, num_experts=4, num_experts_per_tok=2, rms_norm_eps=1e-6,
        max_position_embeddings=1024, vocab_size=96,
    ))
    return _sharpen(jamba.Model(args)), 96


def build_nemotron():
    mx.random.seed(10)
    args = nemotron_h.ModelArgs.from_dict(dict(
        model_type="nemotron_h", vocab_size=96, hidden_size=64, intermediate_size=64,
        num_hidden_layers=4, max_position_embeddings=1024, num_attention_heads=4,
        num_key_value_heads=2, attention_bias=False, mamba_num_heads=4, mamba_head_dim=16,
        mamba_proj_bias=False, ssm_state_size=32, conv_kernel=4, n_groups=1, mlp_bias=False,
        layer_norm_epsilon=1e-5, use_bias=False, use_conv_bias=True,
        hybrid_override_pattern="M*M-",
    ))
    return _sharpen(nemotron_h.Model(args)), 96


BUILDERS = {
    "qwen3_next": lambda: (build_next(), NEXT_ARGS["vocab_size"]),
    "qwen3_5_moe": build_qwen35,
    "lfm2_moe": build_lfm2,
    "jamba": build_jamba,
    "nemotron_h": build_nemotron,
}


def _states(cache):
    out = []
    for c in cache:
        if isinstance(c, ArraysCache):
            out.append(("arrays", list(c.cache)))
        else:
            out.append(("kv", int(c.offset)))
    return out


def _gap(a, b) -> float:
    if a is None or b is None:
        assert a is None and b is None, "one side has state the other does not"
        return 0.0
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    return float(mx.max(mx.abs(a32 - b32)).item()) / (float(mx.max(mx.abs(b32)).item()) + 1e-6)


def check_exact(name, model, vocab):
    rng = np.random.default_rng(11)
    prompt = rng.integers(0, vocab, size=20).tolist()
    feed = rng.integers(0, vocab, size=6).tolist()
    old = (es_config.FAST_REPLAY_CHECKS, es_config.FAST_REPLAY)
    es_config.FAST_REPLAY_CHECKS, es_config.FAST_REPLAY = 0, True
    try:
        for n in (1, 3, 5):
            keep = feed[:n]
            want = make_prompt_cache(model)
            mx.eval(model(mx.array([prompt]), cache=want))
            mx.eval(model(mx.array([keep]), cache=want))

            cache = make_prompt_cache(model)
            mx.eval(model(mx.array([prompt]), cache=cache))
            snap = _snapshot(cache)
            assert snap is not None, f"{name}: hybrid cache did not snapshot"
            rep = fast_replay.for_model(model)
            with rep.recording(len(feed)):
                mx.eval(model(mx.array([feed]), cache=cache))
            full = lambda toks: mx.eval(model(mx.array([toks]), cache=cache))  # noqa: E731
            fast = rep.replay(cache, snap, len(feed), keep, full)
            assert fast, f"{name}: fast replay did not run ({rep.why})"
            worst = 0.0
            for (ka, a), (kb, b) in zip(_states(cache), _states(want)):
                assert ka == kb
                if ka == "kv":
                    assert a == b, f"{name}: attention offset {a}, want {b}"
                else:
                    assert len(a) == len(b)
                    worst = max([worst] + [_gap(x, y) for x, y in zip(a, b)])
            assert worst < 1e-2, f"{name}: recurrent state off by {worst:.2e} keeping {n}"
            # The next token after the replay must see the same model state.
            nxt = mx.array([[feed[-1]]])
            gap = _gap(model(nxt, cache=cache), model(nxt, cache=want))
            assert gap < 1e-2, f"{name}: next-step logits off by {gap:.2e}"
    finally:
        es_config.FAST_REPLAY_CHECKS, es_config.FAST_REPLAY = old
    roles = sorted(set(rep.role.values()))
    print(f"OK: {name} fast replay matches feeding only the kept tokens (worst {worst:.1e}, roles {roles})")


class _Force:
    def __init__(self, oracle):
        self.oracle = oracle

    def __enter__(self):
        self.saved = (vd._Governor.ready, vd._Governor.want, vd._Index.query)
        oracle = self.oracle
        vd._Governor.ready = lambda self, span: True
        vd._Governor.want = lambda self, span, key: True

        def query(index, k):
            index.last_n = 8
            return oracle(list(index.toks), k)

        vd._Index.query = query
        return self

    def __exit__(self, *exc):
        vd._Governor.ready, vd._Governor.want, vd._Index.query = self.saved


def _decode(model, vocab, seed=12, n=40):
    rng = np.random.default_rng(seed)
    prompt = rng.integers(0, vocab, size=24).tolist()
    from tests.test_lookup_hybrid import plain

    ref = plain(model, prompt, n + 8)

    def oracle(toks, k):
        g = len(toks) - len(prompt)
        out = list(ref[g:g + k]) if g >= 0 else []
        if not out:
            return [int(toks[-1])] * k
        bad = g % 4
        if bad < len(out):
            out[bad] = (out[bad] + 1) % vocab
        return out

    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([prompt[:-1]]), cache=cache))
    old = es_config.VERIFY_SPANS
    es_config.VERIFY_SPANS = "all"
    try:
        with _Force(oracle):
            got = [int(t) for t, _ in vd.generate_step(
                mx.array(prompt[-1:], dtype=mx.uint32), model, max_tokens=n,
                prompt_cache=cache, history=prompt[:-1],
            )]
    finally:
        es_config.VERIFY_SPANS = old
    return prompt, got, cache, dict(vd.STATS)


def check_decode():
    old = es_config.FAST_REPLAY
    es_config.FAST_REPLAY = False
    try:
        _p, ref, ref_cache, _ = _decode(build_next(), NEXT_ARGS["vocab_size"])
    finally:
        es_config.FAST_REPLAY = old
    model = build_next()
    prompt, got, cache, st = _decode(model, NEXT_ARGS["vocab_size"])
    check_greedy(model, prompt, got)
    assert st["replays"] > es_config.FAST_REPLAY_CHECKS, f"too few replays to test: {st}"
    assert st["fast_replays"] == st["replays"] - es_config.FAST_REPLAY_CHECKS, st
    assert got == ref, f"output differs from full replays\n  {got}\n  {ref}"
    assert int(cache[-1].offset) == int(ref_cache[-1].offset) == len(prompt) + len(got)
    worst = max(_gap(a, b) for c, r in zip(cache, ref_cache) if isinstance(c, ArraysCache)
                for a, b in zip(c.cache, r.cache))
    assert worst < 1e-2, f"final recurrent state off by {worst:.2e}"
    print(f"OK: decode output and cache match full replays, {st['fast_replays']} of {st['replays']} replays fast")


def check_streamed_reference():
    """On a streamed model the check reuses the pass's MoE rows, so MoE outputs
    that move between forwards (WAIT_ABOVE) do not fail it."""
    model = build_next()
    model._pagedmoe_info = {"mode": "streamed"}
    moe_cls = type(model.layers[0].mlp)
    hooked = moe_cls.__call__
    orig = getattr(hooked, "__wrapped__", hooked)
    calls = [0]

    def jitter(self, x):
        calls[0] += 1
        y = orig(self, x)
        return y * (1.0 + 0.01 * (calls[0] % 3))

    moe_cls.__call__ = jitter
    fast_replay._PATCHED.discard(moe_cls)
    try:
        prompt, got, cache, st = _decode(model, NEXT_ARGS["vocab_size"], seed=15)
    finally:
        moe_cls.__call__ = hooked
        fast_replay._PATCHED.add(moe_cls)
    rep = fast_replay.for_model(model)
    assert not rep.off, f"streamed check failed: {rep.why}"
    assert rep.checks == es_config.FAST_REPLAY_CHECKS and st["fast_replays"] > 0, st
    assert int(cache[-1].offset) == len(prompt) + len(got)
    print(f"OK: streamed models check against the pass's own MoE rows ({st['fast_replays']} fast)")


def check_mismatch_falls_back():
    model = build_next()
    old = es_config.FAST_REPLAY_TOL
    es_config.FAST_REPLAY_TOL = -1.0
    try:
        prompt, got, cache, st = _decode(model, NEXT_ARGS["vocab_size"], seed=13)
    finally:
        es_config.FAST_REPLAY_TOL = old
    rep = fast_replay.for_model(model)
    assert rep.off and "check" in rep.why, f"a failed check did not turn it off: {rep.why}"
    assert st["fast_replays"] == 0 and st["replays"] > 0, st
    check_greedy(model, prompt, got)
    print("OK: a failed check turns fast replay off, output still greedy")


def check_abort_falls_back():
    model = build_next()
    orig = fast_replay._rows
    calls = [0]

    def rows(x, T, keep):
        calls[0] += 1
        if calls[0] == 3:
            raise fast_replay._Abort("forced")
        return orig(x, T, keep)

    old = es_config.FAST_REPLAY_CHECKS
    es_config.FAST_REPLAY_CHECKS = 0
    fast_replay._rows = rows
    try:
        prompt, got, cache, st = _decode(model, NEXT_ARGS["vocab_size"], seed=14)
    finally:
        fast_replay._rows = orig
        es_config.FAST_REPLAY_CHECKS = old
    rep = fast_replay.for_model(model)
    assert rep.off and rep.why == "forced", rep.why
    assert st["replays"] > 0, st
    check_greedy(model, prompt, got)
    assert int(cache[-1].offset) == len(prompt) + len(got)
    print("OK: a replay that fails part way falls back to the full replay, output still greedy")


def main():
    for name, b in BUILDERS.items():
        model, vocab = b()
        check_exact(name, model, vocab)
    check_decode()
    check_streamed_reference()
    check_mismatch_falls_back()
    check_abort_falls_back()
    print("\nOK: fast replay tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
