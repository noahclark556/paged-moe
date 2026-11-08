#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Who steals the GIL from the reader threads during decode?

bench/disk_ceiling.py says this SSD delivers ~14.5 GB/s for the decode-miss
access shape. bench/decode_profile.py says decode achieves 7.54 GB/s on that
same shape. bench/readpath_probe.py shows the whole gap is explained by
main-thread GIL occupancy - but not *which* main-thread work is responsible,
and the two candidates need opposite fixes:

  python glue   list comprehensions, np.unique, per-layer lock traffic,
                prefetch bookkeeping. Fix: do less of it.
  mx.eval       if MLX holds the GIL while blocking on GPU completion, then
                every router sync starves the readers for the duration of a
                GPU drain - and it does so at exactly the wrong moment,
                because the compute window is the window prefetch is supposed
                to hide reads inside. Fix: none available in python; the
                reads have to stop needing the GIL per request.

This measures read bandwidth from worker threads under four main-thread
behaviours, all else identical:

  idle        main thread sleeps                      (upper bound)
  python      main thread spins in pure python        (known GIL hog)
  mx_eval     main thread builds + evals a GPU graph  (the question)
  mx_sleep    main thread evals, then sleeps the same wall time as mx_eval
              spent, to separate "holds the GIL" from "takes wall time"

mx_eval and mx_sleep doing the same wall time but different bandwidth is the
signature of eval holding the GIL. If they match, eval releases it and the
gap is python glue.

Reads target cold, distinct, non-overlapping ranges (see disk_ceiling.py for
why that matters).

Usage:
  PAGED_MOE_HOOK=0 ~/.mlx-env/bin/python bench/gil_probe.py \
      --model ~/mlx-models/qwen3-235-4bit
"""

from __future__ import annotations

import argparse
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import mlx.core as mx

from pagedmoe.safetensors_index import FilePool


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--req-mb", type=float, default=3.146,
                    help="request size; 3.146 MB is one 235B expert weight")
    ap.add_argument("--seconds", type=float, default=2.5,
                    help="how long each cell reads for")
    ap.add_argument("--gpu-ms", type=float, default=1.5,
                    help="target GPU time per eval (one layer's worth)")
    args = ap.parse_args()

    model = Path(args.model).expanduser()
    files = sorted(str(p) for p in model.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors in {model}")
    req = int(args.req_mb * (1 << 20))
    align = 16384

    # Global pool of distinct cold ranges, shuffled once.
    ranges: list[tuple[str, int]] = []
    for f in files:
        n = os.path.getsize(f)
        pos = 0
        while pos + req <= n:
            ranges.append((f, pos // align * align))
            pos += req
    random.Random(5).shuffle(ranges)
    print(f"{len(ranges)} distinct {args.req_mb} MB ranges "
          f"({len(ranges)*req/1e9:.0f} GB) across {len(files)} files")

    cursor = {"pos": 0, "wrapped": 0}
    lock = threading.Lock()

    def next_range():
        # Wrap rather than stop. Running out mid-cell would idle the readers
        # and report the shortfall as low bandwidth - which is exactly what an
        # earlier version of this probe did to its last cell. F_NOCACHE keeps
        # a second pass honest (bench/disk_ceiling.py --verify measures the
        # re-read at cold speed), and the wrap count is printed so a cell that
        # leaned on it is visible.
        with lock:
            i = cursor["pos"]
            cursor["pos"] = i + 1
            if i >= len(ranges):
                cursor["wrapped"] += 1
                cursor["pos"] = 1
                i = 0
            return ranges[i]

    pool = FilePool(nocache=True)

    # Size a GPU graph to roughly --gpu-ms. Shapes in the ballpark of one
    # MoE layer's gather_qmm at 235B dimensions, so the GIL/stream behaviour
    # is representative rather than a microbenchmark artifact.
    a = mx.random.normal((2048, 4096)).astype(mx.float16)
    b = mx.random.normal((4096, 2048)).astype(mx.float16)
    mx.eval(a, b)

    def gpu_once(reps: int):
        y = a
        for _ in range(reps):
            y = (y @ b) @ a
        mx.eval(y)

    reps = 1
    for _ in range(12):
        gpu_once(reps)
        t0 = time.perf_counter()
        gpu_once(reps)
        dt = (time.perf_counter() - t0) * 1e3
        if dt >= args.gpu_ms:
            break
        reps += 1
    gpu_once(reps)
    t0 = time.perf_counter()
    for _ in range(10):
        gpu_once(reps)
    eval_ms = (time.perf_counter() - t0) * 1e3 / 10
    print(f"gpu graph: {reps} rep(s) = {eval_ms:.2f} ms/eval\n")

    stop = threading.Event()

    def reader():
        buf = np.empty(req, dtype=np.uint8)
        total = 0
        while not stop.is_set():
            r = next_range()
            if r is None:
                break
            pool.read_into(r[0], r[1], buf)
            total += req
        return total

    def measure(main_fn) -> tuple[float, float]:
        """Returns (read GB/s, main-thread iterations/s)."""
        stop.clear()
        ex = ThreadPoolExecutor(max_workers=args.threads)
        futs = [ex.submit(reader) for _ in range(args.threads)]
        iters = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.seconds:
            main_fn()
            iters += 1
        stop.set()
        total = sum(f.result() for f in futs)
        dt = time.perf_counter() - t0
        ex.shutdown(wait=True)
        return total / dt / 1e9, iters / dt

    def main_idle():
        time.sleep(0.002)

    def main_python():
        x = 0
        t_end = time.perf_counter() + 0.002
        while time.perf_counter() < t_end:
            for _ in range(200):
                x += 1

    def main_eval():
        gpu_once(reps)

    def main_eval_sleep():
        time.sleep(eval_ms / 1e3)

    cells = (
        ("idle", main_idle),
        ("python", main_python),
        ("mx_eval", main_eval),
        ("mx_sleep", main_eval_sleep),
    )
    print(f"{'main thread':<12}{'read GB/s':>12}{'main it/s':>12}{'wraps':>8}")
    out = {}
    for name, fn in cells:
        before = cursor["wrapped"]
        gbs, its = measure(fn)
        out[name] = gbs
        print(f"{name:<12}{gbs:12.2f}{its:12.0f}"
              f"{cursor['wrapped'] - before:8d}")

    print("\ndecode_profile measures 7.54 GB/s end to end.")
    print(f"  python spin costs   {1 - out['python']/out['idle']:6.1%} of read bandwidth")
    print(f"  mx.eval costs       {1 - out['mx_eval']/out['idle']:6.1%}")
    print(f"  sleeping the same   {1 - out['mx_sleep']/out['idle']:6.1%} "
          "(wall time alone, GIL free)")
    if out["mx_eval"] < out["mx_sleep"] * 0.85:
        print("\n  => mx.eval HOLDS THE GIL while waiting on the GPU. Reader "
              "threads are starved for the whole GPU drain, which is exactly "
              "the window prefetch needs. Reads must stop needing one GIL "
              "acquisition per request.")
    else:
        print("\n  => mx.eval releases the GIL; the gap is python glue on the "
              "main thread, so the fix is to do less of it per layer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
