#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Can this layer's experts be named before its attention runs?

The expert read cannot start until the router runs, and the router runs on
the post-attention residual. Attention is therefore disk-idle. This compares
the real top-k with a gate evaluated on the pre-attention residual. Exact
prefill. No server change.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PAGED_MOE_SIDECAR", "0")
os.environ.setdefault("PAGED_MOE_PRUNE", "0")
os.environ.setdefault("PAGED_MOE_FLOW", "0")

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from pagedmoe import load  # noqa: E402
from pagedmoe.streaming import cache_only, exact_step  # noqa: E402

PRED: list[np.ndarray] = []
REAL: list[np.ndarray] = []


def _topk(gate, x, k: int) -> np.ndarray:
    logits = gate(x.astype(mx.float32))
    idx = mx.argpartition(-logits, k - 1, axis=-1)[..., :k]
    mx.eval(idx)
    return np.array(idx, copy=True)


def _patch(model) -> None:
    layer = model.model.layers[0]
    dec_cls = type(layer)
    moe_cls = type(layer.mlp)
    dec_orig = dec_cls.__call__
    moe_orig = moe_cls.__call__

    def dec_call(self, h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx):
        if h.shape[-2] > 20:
            # Mix the pre-attention residual the way the MoE block will mix
            # the post-attention one. The mix has no cache.
            mixed = self.mlp_hyper_connection(h)
            x = mixed[0] if isinstance(mixed, tuple) else mixed
            PRED.append(_topk(self.mlp.gate, x, self.mlp.top_k))
        return dec_orig(self, h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx)

    def moe_call(self, x):
        if x.shape[-2] > 20:
            REAL.append(_topk(self.gate, x, self.top_k))
        return moe_orig(self, x)

    dec_cls.__call__ = dec_call
    moe_cls.__call__ = moe_call


def _scores(pred: np.ndarray, real: np.ndarray) -> tuple[float, float, float, float]:
    # pred/real: [tokens, k] possibly with a leading 1.
    p = pred.reshape(-1, pred.shape[-1])
    r = real.reshape(-1, real.shape[-1])
    n = min(len(p), len(r))
    p, r = p[:n], r[:n]
    inter = 0
    for i in range(n):
        inter += len(set(p[i].tolist()) & set(r[i].tolist()))
    token_recall = inter / max(1, n * r.shape[-1])
    pu, ru = set(p.reshape(-1).tolist()), set(r.reshape(-1).tolist())
    union_recall = len(pu & ru) / max(1, len(ru))
    union_prec = len(pu & ru) / max(1, len(pu))
    overhead = len(pu - ru) / max(1, len(ru))
    return token_recall, union_recall, union_prec, overhead


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "mlx-models/qwen38-coder-flash")
    prefix_n, chunk_n = 1024, 512
    model, tok = load(model_path, mode="streamed", verbose=False)
    _patch(model)
    text = "\n".join(
        p.read_text(errors="replace")[:40_000]
        for p in sorted((ROOT / "pagedmoe").glob("*.py"))[:8]
    )
    ids = list(tok.encode(text))
    while len(ids) < prefix_n + chunk_n + 4:
        ids = ids + ids
    cache = make_prompt_cache(model)
    with exact_step(), cache_only():
        mx.eval(model(mx.array([ids[:prefix_n]]), cache=cache))
    PRED.clear()
    REAL.clear()
    t0 = time.perf_counter()
    with exact_step(), cache_only():
        mx.eval(model(mx.array([ids[prefix_n : prefix_n + chunk_n]]), cache=cache))
    wall = time.perf_counter() - t0
    n = min(len(PRED), len(REAL))
    print(f"layers {n}  chunk {chunk_n}  wall {wall:.2f}s (probe gates included)", flush=True)
    toks, urec, upre, overs = [], [], [], []
    for i in range(n):
        tr, ur, up, ov = _scores(PRED[i], REAL[i])
        toks.append(tr)
        urec.append(ur)
        upre.append(up)
        overs.append(ov)
    print(
        f"token top-k recall mean {np.mean(toks):.3f}  "
        f"union recall mean {np.mean(urec):.3f} min {np.min(urec):.3f}  "
        f"union precision mean {np.mean(upre):.3f}  "
        f"extra experts / true union mean {np.mean(overs):.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
