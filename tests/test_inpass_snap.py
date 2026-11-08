# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""In-pass mid-snap: the recurrent state read inside one prefill pass must be
the state a stop at the anchor would have left, and splitting the pass must
not change what the pass computes.

    python tests/test_inpass_snap.py

Runs on a tiny qwen3_next, and on a tiny qwen4_exp when the Flash port is
installed (~/mlx-models/qwen38-coder-flash/qwen4_exp.py).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import ArraysCache  # noqa: E402

from tests.test_lookup_hybrid import ARGS  # noqa: E402

from pagedmoe import config as es_config  # noqa: E402
from pagedmoe import hybrid_ckpt, inpass_snap  # noqa: E402

KEY = ("tiny", None, None)
FLASH_PORT = Path.home() / "mlx-models/qwen38-coder-flash/qwen4_exp.py"


def _qwen3_next():
    from mlx_lm.models import qwen3_next

    mx.random.seed(0)
    model = qwen3_next.Model(qwen3_next.ModelArgs(**{**ARGS, "head_dim": 32}))
    mx.eval(model.parameters())
    return model, ARGS["vocab_size"]


def _qwen4_exp():
    if not FLASH_PORT.exists():
        return None
    # Not registered in sys.modules, same as the engine's loader.
    spec = importlib.util.spec_from_file_location("qwen4_exp", str(FLASH_PORT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = dict(
        hidden_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, vocab_size=512, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=64, shared_expert_intermediate_size=64,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, hc_count=2, hc_lowrank=16, indexer_n_heads=2,
        indexer_head_dim=16, indexer_budget=64, indexer_compress_ratio=4,
        ngram_vocab_size_base=1000, make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4, heads_per_ngram=2, ple_embed_dim=128, ple_layer_ids=[2],
    )
    mx.random.seed(0)
    model = mod.Model(mod.ModelArgs(model_type="qwen4_exp", text_config=text))
    mx.eval(model.parameters())
    return model, 512


def _arrays(cache):
    return {i: list(c.cache) for i, c in enumerate(cache) if isinstance(c, ArraysCache)}


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b))


# The anchor state comes out of the chunk's projections over more rows than a
# stop at the anchor runs them on. When that picks a different matmul kernel
# the activations round differently (bf16), same as moving a chunk boundary.
_ROUNDING = 1e-2


def _anchor_diff(slots, want) -> float:
    """0.0 when every slot is bit-identical to `want`, else the worst relative diff."""
    worst = 0.0
    for i, ref in want.items():
        for x, y in zip(slots[i], ref):
            if _same(x, y):
                continue
            assert x is not None and y is not None and x.shape == y.shape, "slot layout differs"
            x32, y32 = x.astype(mx.float32), y.astype(mx.float32)
            d = float(mx.abs(x32 - y32).max()) / (float(mx.abs(y32).max()) + 1e-12)
            worst = max(worst, d)
    return worst


def check(model, vocab: int, name: str):
    inpass_snap._off = None
    hybrid_ckpt.clear()
    rng = np.random.default_rng(7)
    a = rng.integers(0, vocab, size=120).tolist()
    head, rel = 40, 70  # resumed prefix, then one chunk with the anchor inside it
    ids = mx.array([a[head:]])
    assert inpass_snap.supports(model, model.make_cache()), f"{name}: not supported"

    # Stop at the anchor: the reference state.
    stop = model.make_cache()
    mx.eval(model(mx.array([a[:head]]), cache=stop))
    mx.eval(model(mx.array([a[head : head + rel]]), cache=stop))
    want = _arrays(stop)

    # One pass, plain and with the in-pass capture.
    plain = model.make_cache()
    mx.eval(model(mx.array([a[:head]]), cache=plain))
    lp = model(ids, cache=plain)
    mx.eval(lp)

    live = model.make_cache()
    mx.eval(model(mx.array([a[:head]]), cache=live))
    out = {}
    slots = inpass_snap.run(model, live, ids, rel, lambda: out.setdefault("l", model(ids, cache=live)))
    assert slots is not None, f"{name}: capture failed ({inpass_snap._off})"
    mx.eval(out["l"])

    assert bool(mx.array_equal(out["l"], lp)), f"{name}: split changed the pass's logits"
    for i, now in _arrays(live).items():
        for s, (x, y) in enumerate(zip(now, _arrays(plain)[i])):
            assert _same(x, y), f"{name}: split changed layer {i} slot {s} at the end of the pass"
    d_anchor = _anchor_diff(slots, want)
    assert d_anchor <= _ROUNDING, f"{name}: anchor state off by {d_anchor:g} relative"

    # Rebuild the cache at the anchor and continue a branch from it.
    p = head + rel
    ckpt = hybrid_ckpt.from_slots(KEY, a[:p], live, slots)
    assert ckpt is not None, f"{name}: from_slots refused"
    entry = hybrid_ckpt.restore(live, ckpt)
    assert entry is not None, f"{name}: restore refused"
    branch = mx.array([rng.integers(0, vocab, size=12).tolist()])
    la = model(branch, cache=entry)
    lb = model(branch, cache=stop)
    d = float(mx.abs(la - lb).max())
    how = "bit-identical to" if d_anchor == 0.0 else f"within {d_anchor:.2g} of"
    print(f"OK: {name} split pass unchanged; anchor state {how} a stop there; "
          f"branch logits vs two-phase max |diff| = {d:g}")
    return d


def check_multi(model, vocab: int, name: str):
    """Several anchors in one pass: the pass is unchanged and each anchor's
    state is the state a stop there would have left."""
    inpass_snap._off = None
    rng = np.random.default_rng(12)
    a = rng.integers(0, vocab, size=120).tolist()
    head, rels = 20, [15, 40, 77]
    ids = mx.array([a[head:]])

    plain = model.make_cache()
    mx.eval(model(mx.array([a[:head]]), cache=plain))
    lp = model(ids, cache=plain)
    mx.eval(lp)

    live = model.make_cache()
    mx.eval(model(mx.array([a[:head]]), cache=live))
    out = {}
    got = inpass_snap.run_multi(model, live, ids, rels, lambda: out.setdefault("l", model(ids, cache=live)))
    assert got is not None and len(got) == len(rels), f"{name}: capture failed ({inpass_snap._off})"
    mx.eval(out["l"])
    assert bool(mx.array_equal(out["l"], lp)), f"{name}: split changed the pass's logits"
    for i, now in _arrays(live).items():
        for s, (x, y) in enumerate(zip(now, _arrays(plain)[i])):
            assert _same(x, y), f"{name}: split changed layer {i} slot {s} at the end of the pass"
    worst = 0.0
    for rel, slots in zip(rels, got):
        stop = model.make_cache()
        mx.eval(model(mx.array([a[:head]]), cache=stop))
        mx.eval(model(mx.array([a[head : head + rel]]), cache=stop))
        d = _anchor_diff(slots, _arrays(stop))
        assert d <= _ROUNDING, f"{name}: anchor {rel} off by {d:g} relative"
        worst = max(worst, d)
    print(f"OK: {name} three anchors in one pass, logits unchanged, worst anchor diff {worst:g}")


def check_refuses_unknown_slot(model, vocab: int):
    """A recurrent slot no rule explains turns the capture off, not wrong."""
    inpass_snap._off = None
    rng = np.random.default_rng(8)
    a = rng.integers(0, vocab, size=60).tolist()
    live = model.make_cache()
    idx = next(i for i, c in enumerate(live) if isinstance(c, ArraysCache))
    ids = mx.array([a])

    def forward():
        model(ids, cache=live)
        live[idx][0] = live[idx][0] + 1  # a write no rule produces

    assert inpass_snap.run(model, live, ids, 30, forward) is None, "unknown write was accepted"
    assert inpass_snap._off is not None and not inpass_snap.supports(model, live)
    inpass_snap._off = None
    print("OK: an unexplained recurrent write disables the in-pass snapshot")


def check_prefill_driver(model, vocab: int):
    """server._prefill_remainder hands back the anchor state wherever the
    anchor falls: inside a chunk, on a chunk boundary, or at the end."""
    from pagedmoe import server

    rng = np.random.default_rng(9)
    a = rng.integers(0, vocab, size=90).tolist()
    old = es_config.ADAPTIVE_PREFILL
    es_config.ADAPTIVE_PREFILL = False
    try:
        for snap_at, where, exact in (
            (25, "inside a chunk", False),
            (32, "on a chunk boundary", True),
            (89, "at the end", True),
        ):
            inpass_snap._off = None
            stop = model.make_cache()
            for lo in range(0, snap_at, 32):  # the driver's chunks, cut at the anchor
                mx.eval(model(mx.array([a[lo : min(lo + 32, snap_at)]]), cache=stop))
            want = _arrays(stop)
            live = model.make_cache()
            out: dict = {}
            done = server._prefill_remainder(
                model, a, live, n_model_layers=len(live), draft_model=None, draft_cache=None,
                step_cap=32, progress=lambda *_: None, total=len(a), processed=0, target=89,
                snap_at=snap_at, snap_out=out,
            )
            assert done == 89, f"{where}: prefill stopped at {done}"
            slots = out.get("slots")
            assert slots, f"{where}: no state handed back ({inpass_snap._off})"
            diff = _anchor_diff(slots, want)
            if exact:
                assert diff == 0.0, f"{where}: live state off by {diff:g}"
            else:
                assert diff <= _ROUNDING, f"{where}: anchor state off by {diff:g} relative"
        # Disk checkpoints in the same run: one inside a chunk next to the
        # mid-snap, one on a boundary, none past the target.
        inpass_snap._off = None
        live = model.make_cache()
        out, ck = {}, {}
        done = server._prefill_remainder(
            model, a, live, n_model_layers=len(live), draft_model=None, draft_cache=None,
            step_cap=32, progress=lambda *_: None, total=len(a), processed=0, target=89,
            snap_at=25, snap_out=out, ckpt_at=[10, 64, 95], ckpt_out=ck,
        )
        assert done == 89 and out.get("slots"), "mid-snap lost next to disk checkpoints"
        assert sorted(ck) == [10, 64], f"checkpoints at {sorted(ck)}"
        for p, exact in ((10, False), (64, True)):
            stop = model.make_cache()
            for lo in range(0, p, 32):
                mx.eval(model(mx.array([a[lo : min(lo + 32, p)]]), cache=stop))
            diff = _anchor_diff(ck[p], _arrays(stop))
            assert (diff == 0.0) if exact else (diff <= _ROUNDING), f"checkpoint {p} off by {diff:g}"
        ref = model.make_cache()
        server._prefill_remainder(
            model, a, ref, n_model_layers=len(ref), draft_model=None, draft_cache=None,
            step_cap=32, progress=lambda *_: None, total=len(a), processed=0, target=89,
        )
        for i, now in _arrays(live).items():
            for s, (x, y) in enumerate(zip(now, _arrays(ref)[i])):
                assert _same(x, y), f"checkpoints changed layer {i} slot {s} at the end of the prefill"
    finally:
        es_config.ADAPTIVE_PREFILL = old
    print("OK: the prefill driver captures the anchor inside a chunk, on a boundary and at the end")


def main():
    old_min = es_config.HYBRID_CKPT_MIN_TOKENS
    es_config.HYBRID_CKPT_MIN_TOKENS = 1
    try:
        model, vocab = _qwen3_next()
        check(model, vocab, "qwen3_next")
        check_multi(model, vocab, "qwen3_next")
        check_refuses_unknown_slot(model, vocab)
        check_prefill_driver(model, vocab)
        got = _qwen4_exp()
        if got is None:
            print(f"skip qwen4_exp: no port at {FLASH_PORT}")
        else:
            check(*got, "qwen4_exp")
            check_multi(*got, "qwen4_exp")
    finally:
        es_config.HYBRID_CKPT_MIN_TOKENS = old_min
        hybrid_ckpt.clear()
        inpass_snap._off = None
    print("\nOK: in-pass snapshot tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
