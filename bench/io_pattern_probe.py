#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Isolated throughput of the exact decode-miss read pattern.

Reads N random experts of random MoE layers through the same FilePool
(F_NOCACHE) + slice-pool machinery the ExpertCache uses, with no GPU, no
main-thread contention, and no demand/speculation mix - just the raw pattern.

If this measures well above the ~6.4 GB/s the engine sustains during decode,
the ceiling is contention inside the process (GIL / scheduling), not the
drive. If it measures the same, the drive's scattered-read ceiling is the
wall and only layout changes (packed arena) or fewer bytes can help.

Usage:
  python bench/io_pattern_probe.py --model ~/mlx-models/qwen3-235-4bit \
      --experts 600 [--threads 16] [--slice-mb 2] [--batch 40]
"""

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from pagedmoe.safetensors_index import FilePool, read_headers
from pagedmoe.streaming import _PROJS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--experts", type=int, default=600)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--slice-mb", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=40,
                    help="experts issued concurrently (~ one decode burst)")
    args = ap.parse_args()

    slice_bytes = int(args.slice_mb * (1 << 20))
    index = read_headers(args.model)

    # All stacked expert component tensors: name -> TensorLoc, grouped by layer.
    layers = {}
    for name, loc in index.items():
        parts = name.split(".")
        if len(parts) >= 3 and parts[-2] in _PROJS and "switch_mlp" in name:
            lk = name.rsplit(".", 2)[0]
            layers.setdefault(lk, {})[f"{parts[-2]}.{parts[-1]}"] = loc

    layer_keys = sorted(layers)
    n_experts = next(iter(layers.values()))["gate_proj.weight"].shape[0]
    per_expert = sum(
        loc.nbytes // loc.shape[0] for loc in next(iter(layers.values())).values()
    )
    print(f"{len(layer_keys)} MoE layers, {n_experts} experts, "
          f"{per_expert / 1e6:.1f} MB/expert")

    pool = FilePool(nocache=True)
    expert_pool = ThreadPoolExecutor(max_workers=args.threads)
    slice_pool = ThreadPoolExecutor(max_workers=args.threads)

    def read_expert(lk, eid):
        comps = layers[lk]
        jobs = []
        bufs = []
        for name, loc in comps.items():
            stride = loc.nbytes // loc.shape[0]
            off = loc.offset + eid * stride
            buf = np.empty(stride, dtype=np.uint8)
            bufs.append(buf)
            for start in range(0, stride, slice_bytes):
                end = min(start + slice_bytes, stride)
                jobs.append(
                    slice_pool.submit(pool.read_into, loc.file, off + start,
                                      buf[start:end])
                )
        for j in jobs:
            j.result()
        return sum(b.nbytes for b in bufs)

    rng = random.Random(7)
    picks = [
        (rng.choice(layer_keys), rng.randrange(n_experts))
        for _ in range(args.experts)
    ]

    # Warm-up (file opens)
    read_expert(*picks[0])

    total = 0
    t0 = time.perf_counter()
    for i in range(0, len(picks), args.batch):
        futs = [
            expert_pool.submit(read_expert, lk, e)
            for lk, e in picks[i : i + args.batch]
        ]
        for f in futs:
            total += f.result()
    dt = time.perf_counter() - t0
    print(f"{total / 1e9:.2f} GB in {dt:.2f}s = {total / dt / 1e9:.2f} GB/s "
          f"(threads={args.threads}, slice={args.slice_mb} MB, "
          f"batch={args.batch})")


if __name__ == "__main__":
    main()
