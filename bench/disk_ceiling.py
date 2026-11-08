#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ground truth: what does this SSD actually deliver, cold, per access shape?

Every other probe in this directory divides bytes by seconds and calls the
result bandwidth. That is only meaningful if the bytes came off the drive,
and on macOS they often do not: F_NOCACHE stops a read from *populating* the
page cache but does not stop it from being *served* from pages another
process (or an earlier buffered read) already cached. Re-reading the same
ranges across a sweep therefore produces a rising, entirely fictional curve -
which is how bench/readpath_probe.py first reported 14 GB/s and then 30 GB/s
for the same code.

This establishes the real ceiling by construction:

  * every timed read targets a byte range no earlier read in this process has
    touched, drawn from a shuffled global plan
  * an optional --verify pass re-reads one earlier range and reports the
    speedup, which is a direct measurement of how much caching is in play
  * request size and queue depth are swept independently, because they are
    the two things the read path can actually choose

Read `cold` as the number to design against, and treat any figure above it
elsewhere as cache.

Usage:
  PAGED_MOE_HOOK=0 ~/.mlx-env/bin/python bench/disk_ceiling.py \
      --file ~/mlx-models/qwen3-235-4bit/model-00010-of-00026.safetensors
"""

from __future__ import annotations

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from pagedmoe.safetensors_index import FilePool


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="checkpoint dir; all model*.safetensors are treated "
                         "as one address space so distinct ranges never run out")
    ap.add_argument("--sizes-mb", default="0.197,1,2,3.146,4,10.62,16,32")
    ap.add_argument("--depths", default="1,4,8,16,32,64")
    ap.add_argument("--per-run-mb", type=float, default=768.0,
                    help="bytes each (size, depth) cell reads - all distinct")
    ap.add_argument("--verify", action="store_true",
                    help="re-read the first cell to quantify cache effects")
    args = ap.parse_args()

    model = Path(args.model).expanduser()
    files = sorted(str(p) for p in model.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors in {model}")
    sizes = [int(float(x) * (1 << 20)) for x in args.sizes_mb.split(",")]
    depths = [int(x) for x in args.depths.split(",")]
    per_run = int(args.per_run_mb * (1 << 20))

    align = 16384
    block = max(sizes)
    # One global pool of distinct, non-overlapping, shuffled (file, offset)
    # blocks across the whole checkpoint. Handing these out once guarantees
    # no timed read ever revisits a range this process already pulled.
    blocks: list[tuple[str, int]] = []
    total_size = 0
    for f in files:
        n = os.path.getsize(f)
        total_size += n
        pos = 0
        while pos + block <= n:
            blocks.append((f, pos))
            pos += block
    rng = random.Random(11)
    rng.shuffle(blocks)

    print(f"{model}")
    print(f"  {len(files)} files, {total_size/1e9:.1f} GB, "
          f"{len(blocks)} distinct {block>>20} MiB blocks")

    pool = FilePool(nocache=True)
    cursor = {"pos": 0}

    def take(n_blocks: int) -> list[tuple[str, int]]:
        i = cursor["pos"]
        cursor["pos"] = i + n_blocks
        if cursor["pos"] > len(blocks):
            raise RuntimeError(
                f"ran out of distinct ranges after {i} blocks "
                f"({i*block/1e9:.1f} GB); lower --per-run-mb"
            )
        return blocks[i : i + n_blocks]

    first_cell: list[tuple[str, int, int]] | None = None

    print(f"\n{'req MB':>8} " + "".join(f"{f'qd{d}':>9}" for d in depths)
          + "   (GB/s, cold)")
    for sz in sizes:
        row = []
        for depth in depths:
            n_reads = max(depth, per_run // sz)
            per_block = max(1, block // sz)
            bases = take(-(-n_reads // per_block))
            # Each read gets a distinct offset inside a block no other cell
            # was handed, so nothing in this sweep is ever re-read.
            jobs = []
            for i in range(n_reads):
                f, base = bases[i // per_block]
                off = (base + (i % per_block) * sz) // align * align
                jobs.append((f, off, sz))
            if first_cell is None:
                first_cell = list(jobs)

            bufs = [np.empty(sz, dtype=np.uint8) for _ in range(depth)]
            ex = ThreadPoolExecutor(max_workers=depth)

            def one(job, slot):
                f, off, n = job
                pool.read_into(f, off, bufs[slot])
                return n

            t0 = time.perf_counter()
            total = 0
            for i in range(0, len(jobs), depth):
                chunk = jobs[i : i + depth]
                futs = [ex.submit(one, j, k) for k, j in enumerate(chunk)]
                for f in futs:
                    total += f.result()
            dt = time.perf_counter() - t0
            ex.shutdown(wait=True)
            row.append(total / dt / 1e9)
        print(f"{sz/(1<<20):8.3f} " + "".join(f"{g:8.2f} " for g in row))

    if args.verify and first_cell:
        depth = depths[-1]
        sz = first_cell[0][2]
        bufs = [np.empty(sz, dtype=np.uint8) for _ in range(depth)]
        ex = ThreadPoolExecutor(max_workers=depth)

        def one(job, slot):
            f, off, n = job
            pool.read_into(f, off, bufs[slot])
            return n

        t0 = time.perf_counter()
        total = 0
        for i in range(0, len(first_cell), depth):
            chunk = first_cell[i : i + depth]
            futs = [ex.submit(one, j, k) for k, j in enumerate(chunk)]
            for f in futs:
                total += f.result()
        dt = time.perf_counter() - t0
        ex.shutdown(wait=True)
        print(f"\n  re-read of an already-read range: {total/dt/1e9:.2f} GB/s")
        print("  (if this is far above the cold numbers, caching is in play "
              "and any probe that re-reads ranges is measuring RAM)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
