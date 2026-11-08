# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does batching the per-expert matmuls into mx.gather_qmm actually pay?

The streamed path runs one quantized_matmul per expert per projection, because
cache entries are per-expert arrays. Batching them would require the cache to
hold experts in stacked blocks (a real refactor: LRU, accounting and eviction
all become block-shaped), so measure the ceiling before paying for it.

Default shapes are Qwen3-Coder-Next's: hidden 2048, moe_intermediate 512,
top-10 of 512 experts, 48 MoE layers, 4-bit / group 64.

    python bench/gather_qmm_probe.py

**The answer is shape-dependent - always pass the shapes of the model you
care about.** Coder-Next's 1.6 MB experts made this a null result; Qwen3-235B's
are 10.6 MB across 94 layers, which is a different regime entirely:

    python bench/gather_qmm_probe.py --hidden 4096 --inter 1536 \
        --topk 8 --layers 94
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

HIDDEN = 2048
INTER = 512
TOPK = 10
LAYERS = 48
BITS = 4
GROUP = 64
REPS = 20


def make_stack(experts: int, out_dim: int, in_dim: int):
    w = mx.random.normal((experts, out_dim, in_dim)).astype(mx.float16)
    return mx.quantize(w, group_size=GROUP, bits=BITS)


def timed(fn, *, warmup=3, reps=REPS) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / reps


def run(rows: int) -> None:
    """rows = activations per expert (decode: 1; prefill chunk: many)."""
    gate = make_stack(TOPK, INTER, HIDDEN)
    up = make_stack(TOPK, INTER, HIDDEN)
    down = make_stack(TOPK, HIDDEN, INTER)
    x = mx.random.normal((TOPK, rows, HIDDEN)).astype(mx.float16)

    def per_expert():
        outs = []
        for e in range(TOPK):
            xe = x[e]
            a = mx.quantized_matmul(
                xe, gate[0][e], gate[1][e], gate[2][e],
                transpose=True, group_size=GROUP, bits=BITS,
            )
            b = mx.quantized_matmul(
                xe, up[0][e], up[1][e], up[2][e],
                transpose=True, group_size=GROUP, bits=BITS,
            )
            outs.append(
                mx.quantized_matmul(
                    mx.sigmoid(a) * b, down[0][e], down[1][e], down[2][e],
                    transpose=True, group_size=GROUP, bits=BITS,
                )
            )
        return mx.concatenate(outs, axis=0)

    # One expert per batch element: indices must match x's batch dims exactly,
    # or they broadcast and every expert runs against every block.
    idx = mx.arange(TOPK)

    def batched():
        a = mx.gather_qmm(
            x, gate[0], gate[1], gate[2], rhs_indices=idx,
            transpose=True, group_size=GROUP, bits=BITS,
        )
        b = mx.gather_qmm(
            x, up[0], up[1], up[2], rhs_indices=idx,
            transpose=True, group_size=GROUP, bits=BITS,
        )
        return mx.gather_qmm(
            mx.sigmoid(a) * b, down[0], down[1], down[2], rhs_indices=idx,
            transpose=True, group_size=GROUP, bits=BITS,
        )

    # What the streamed path can actually do: cache entries are per-expert, so
    # the stack is paid on every call. Small at decode (10 experts ~ 1.6 MB).
    per_expert_views = [
        {
            "gate": (gate[0][e], gate[1][e], gate[2][e]),
            "up": (up[0][e], up[1][e], up[2][e]),
            "down": (down[0][e], down[1][e], down[2][e]),
        }
        for e in range(TOPK)
    ]
    mx.eval([t for d in per_expert_views for trio in d.values() for t in trio])

    def stack_then_batched():
        def stack(name, i):
            return mx.stack([d[name][i] for d in per_expert_views])

        a = mx.gather_qmm(
            x, stack("gate", 0), stack("gate", 1), stack("gate", 2), rhs_indices=idx,
            transpose=True, group_size=GROUP, bits=BITS,
        )
        b = mx.gather_qmm(
            x, stack("up", 0), stack("up", 1), stack("up", 2), rhs_indices=idx,
            transpose=True, group_size=GROUP, bits=BITS,
        )
        return mx.gather_qmm(
            mx.sigmoid(a) * b, stack("down", 0), stack("down", 1), stack("down", 2),
            rhs_indices=idx, transpose=True, group_size=GROUP, bits=BITS,
        )

    loose = timed(per_expert)
    tight = timed(batched)
    stacked = timed(stack_then_batched)
    print(
        f"rows/expert={rows:<6} stack+gather {stacked * 1e3:7.3f} ms "
        f"(vs per-expert {loose / stacked:5.2f}x, stack cost "
        f"{(stacked - tight) * 1e3:6.3f} ms)"
    )
    tokens = rows  # decode: rows==1 token; prefill: rows per expert
    print(
        f"rows/expert={rows:<6} per-expert {loose * 1e3:7.3f} ms   "
        f"gather_qmm {tight * 1e3:7.3f} ms   speedup {loose / tight:5.2f}x   "
        f"whole-model delta {(loose - tight) * LAYERS * 1e3:7.1f} ms"
        + (f"  ({tokens} tok)" if tokens > 1 else "")
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--inter", type=int, default=INTER)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--bits", type=int, default=BITS)
    ap.add_argument("--group", type=int, default=GROUP)
    ap.add_argument("--rows", default="1,4,64,640")
    args = ap.parse_args()
    HIDDEN, INTER, TOPK = args.hidden, args.inter, args.topk
    LAYERS, BITS, GROUP = args.layers, args.bits, args.group

    per_expert_mb = (INTER * HIDDEN * 3 * BITS / 8) / 1e6
    print(
        f"top-{TOPK}, hidden={HIDDEN}, inter={INTER}, {BITS}-bit/g{GROUP}, "
        f"{LAYERS} MoE layers, ~{per_expert_mb:.1f} MB/expert"
    )
    for rows in (int(r) for r in args.rows.split(",") if r.strip()):
        run(rows)
