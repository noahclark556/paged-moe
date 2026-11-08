# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prefill MoE compute: per-expert loop (streaming._run) vs one sorted gather_qmm.

Synthetic layer at a checkpoint's real shapes, all experts resident in one
stacked buffer (what the slab looks like), random top-k routing. Times only
compute, and checks the two paths agree bit for bit.

  python bench/prefill_gather_probe.py --d 2560 --i 640 --e 512 --k 10 --bits 6 --tokens 300 1300 2600
"""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2560)
    ap.add_argument("--i", type=int, default=640)
    ap.add_argument("--e", type=int, default=512)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--gs", type=int, default=64)
    ap.add_argument("--tokens", type=int, nargs="+", default=[300, 1300, 2600])
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    mx.random.seed(0)
    D, I, E = args.d, args.i, args.e

    def qw(out_f, in_f):
        w = mx.random.normal((E, out_f, in_f)).astype(mx.bfloat16) * 0.02
        return mx.quantize(w, group_size=args.gs, bits=args.bits)

    W = {"up": qw(I, D), "gate": qw(I, D), "down": qw(D, I)}
    mx.eval(W)

    def qmm(x, p, sl=None, rhs=None, lhs=None, sorted_=False):
        w, s, b = W[p]
        if sl is not None:
            return mx.quantized_matmul(
                x, mx.squeeze(w[sl : sl + 1], 0), mx.squeeze(s[sl : sl + 1], 0),
                mx.squeeze(b[sl : sl + 1], 0), transpose=True, group_size=args.gs, bits=args.bits)
        return mx.gather_qmm(x, w, s, b, lhs_indices=lhs, rhs_indices=rhs, transpose=True,
                             group_size=args.gs, bits=args.bits, sorted_indices=sorted_)

    def per_expert(x, inds):
        N, K = inds.shape
        flat = inds.reshape(-1)
        order = np.argsort(flat, kind="stable")
        se = flat[order]
        xs = x[mx.array(order // K)]
        ys = []
        for e in np.unique(se):
            lo = np.searchsorted(se, e, "left")
            hi = np.searchsorted(se, e, "right")
            xe = xs[lo:hi]
            h = nn.silu(qmm(xe, "gate", sl=int(e))) * qmm(xe, "up", sl=int(e))
            ys.append(qmm(h, "down", sl=int(e)))
        y = mx.concatenate(ys, axis=0)
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        return y[mx.array(inv)].reshape(N, K, -1)

    def gathered(x, inds):
        N, K = inds.shape
        flat = inds.reshape(-1)
        order = np.argsort(flat, kind="stable")
        rhs = mx.array(flat[order].astype(np.uint32))
        xs = x[mx.array(order // K)][:, None, :]
        h = nn.silu(qmm(xs, "gate", rhs=rhs, sorted_=True)) * qmm(xs, "up", rhs=rhs, sorted_=True)
        y = qmm(h, "down", rhs=rhs, sorted_=True).squeeze(1)
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        return y[mx.array(inv)].reshape(N, K, -1)

    rng = np.random.default_rng(0)
    for n in args.tokens:
        x = (mx.random.normal((n, D)) * 0.5).astype(mx.bfloat16)
        inds = np.stack([rng.choice(E, args.k, replace=False) for _ in range(n)]).astype(np.int64)
        mx.eval(x)
        res = {}
        outs = {}
        for name, fn in (("per_expert", per_expert), ("gather_sorted", gathered)):
            ts = []
            for _ in range(args.reps):
                t = time.perf_counter()
                y = fn(x, inds)
                mx.eval(y)
                ts.append(time.perf_counter() - t)
            res[name] = sorted(ts)[len(ts) // 2]
            outs[name] = y
        same = bool(mx.array_equal(outs["per_expert"], outs["gather_sorted"]).item())
        diff = float(mx.abs(outs["per_expert"].astype(mx.float32) - outs["gather_sorted"].astype(mx.float32)).max().item())
        print(
            f"N={n:5} experts={len(np.unique(inds)):4}  per-expert {res['per_expert'] * 1e3:7.1f} ms  "
            f"gather {res['gather_sorted'] * 1e3:7.1f} ms  ({res['per_expert'] / res['gather_sorted']:4.1f}x)  "
            f"bit-identical={same} max|d|={diff:.2e}  x48 layers: {res['per_expert'] * 48:5.2f}s vs {res['gather_sorted'] * 48:5.2f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
