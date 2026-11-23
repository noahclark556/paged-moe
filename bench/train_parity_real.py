# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Paged-training parity on a real checkpoint.

Same LoRA weights, same batch:
  resident fused       (reference)
  resident layered     must match the reference
  streamed layered     (forced, small cache) must match the reference

    python bench/train_parity_real.py ~/mlx-models/Qwen3.6-35B-A3B-4bit --cache-gb 4
"""

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map

from pagedmoe.train import env as train_env

train_env.apply()
train_env.index_ops_stop_gradient()
train_env.exact_numerics()

from pagedmoe import load  # noqa: E402
from pagedmoe.train import experts as train_experts  # noqa: E402
from pagedmoe.train import linear_attn  # noqa: E402
from pagedmoe.train import lora as train_lora  # noqa: E402
from pagedmoe.train.executor import Batch, FusedExecutor, LayeredExecutor  # noqa: E402

TEXT = (
    "Mixture-of-experts models route each token to a few experts out of many. "
    "Training one on a laptop means the experts that do not fit in memory have to "
    "come from the SSD, and the backward pass needs them again in reverse order. "
)


def batches(tok, T, n):
    ids = tok.encode(TEXT * 40)
    out = []
    for j in range(n):
        seq = ids[j * 7 : j * 7 + 2 * (T + 1)]
        arr = np.asarray(seq, np.int32).reshape(2, T + 1)
        out.append(Batch(tokens=mx.array(arr[:, :-1]), targets=mx.array(arr[:, 1:]),
                         mask=mx.ones((2, T), dtype=mx.float32)))
    return out


def prep(model, streamed):
    train_lora.apply(model, train_lora.LoRAConfig(rank=8, alpha=16), streamed=streamed)
    mx.random.seed(3)
    model.update(tree_map(lambda a: mx.random.normal(a.shape) * 0.02, model.trainable_parameters()))
    mx.eval(model.parameters())
    model.train()


def loss_for(bs):
    norm = float(sum(b.n_tokens for b in bs))

    def f(lp, b):
        nll = -(lp * b.mask).sum() / norm
        return nll, {}

    return f


def flat(g):
    return {k: np.asarray(v.astype(mx.float32)) for k, v in tree_flatten(g)}


def diff(a, b):
    """'max rel / cosine' between two flat grad dicts."""
    worst = 0.0
    for k in a:
        s = max(1e-8, float(np.abs(a[k]).max()))
        worst = max(worst, float(np.abs(a[k] - b[k]).max()) / s)
    va = np.concatenate([a[k].ravel() for k in sorted(a)])
    vb = np.concatenate([b[k].ravel() for k in sorted(a)])
    cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))
    return f"{worst:.2e} / cos {cos:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--cache-gb", type=float, default=4.0)
    ap.add_argument("--loop-delta", action="store_true", help="keep mlx-lm's per-token gated delta")
    args = ap.parse_args()

    model, tok = load(args.model, mode="resident", verbose=False)
    if not args.loop_delta:
        linear_attn.install()
    prep(model, False)
    bs = batches(tok, args.T, args.micro)
    lf = loss_for(bs)

    t0 = time.perf_counter()
    v_ref, g_ref, _, _ = FusedExecutor(model, checkpoint=True).step(bs, lf)
    t_f = time.perf_counter() - t0
    ref = flat(g_ref)
    print(f"resident fused+ckpt  loss {v_ref:.6f}  {t_f:.1f}s  peak {mx.get_peak_memory() / 1e9:.1f} GB")

    mx.reset_peak_memory()
    lay = LayeredExecutor(model)
    t0 = time.perf_counter()
    v, g, _, _ = lay.step(bs, lf)
    t_l = time.perf_counter() - t0
    print(f"resident layered     loss {v:.6f}  {t_l:.1f}s  peak {mx.get_peak_memory() / 1e9:.1f} GB  "
          f"grad diff {diff(ref, flat(g))}  timing {lay.timing}")
    del model, lay, g, g_ref
    gc.collect()
    mx.clear_cache()

    mx.reset_peak_memory()
    model, _ = load(args.model, mode="streamed", cache_gb=args.cache_gb, verbose=False)
    train_experts.install(model)
    prep(model, True)
    cache = model._pagedmoe_cache
    lay = LayeredExecutor(model)
    for k in range(2):
        r0 = cache.bytes_read
        w0 = cache.disk_wait_s
        train_experts.COUNTERS.reset()
        t0 = time.perf_counter()
        v, g, _, _ = lay.step(bs, lf)
        dt = time.perf_counter() - t0
        print(f"streamed layered #{k} loss {v:.6f}  {dt:.1f}s  read {(cache.bytes_read - r0) / 1e9:.1f} GB  "
              f"disk wait {cache.disk_wait_s - w0:.1f}s  peak {mx.get_peak_memory() / 1e9:.1f} GB  "
              f"grad diff {diff(ref, flat(g))}  experts {train_experts.COUNTERS.snapshot()}")
    print(f"timing {lay.timing}")


if __name__ == "__main__":
    main()
