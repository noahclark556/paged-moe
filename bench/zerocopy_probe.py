#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Can a prefill run read land in MLX memory without the host copy?

`bench/prefill_cost.py` says ~40% of an incremental prefill is `_materialize`:
`mx.array(np_buf)` over every byte read. `slab.py` already avoids that for
decode by preading into `np.asarray(mx_array)`, so the question for prefill is
narrower than "can we do it at all":

  1. is a run-sized MLX buffer's numpy view still writeable and contiguous
  2. does slicing expert i out of a `[count, *shape]` MLX buffer stay a view, or
     does reshape/view/astype quietly reintroduce the copy this is meant to
     remove
  3. what does the allocation itself cost, since a run buffer is per-run

(2) is the one that decides the design. If a row slice copies, per-expert entries
have to go and the run has to be handed to `gather_qmm` stacked.

    python bench/zerocopy_probe.py
"""

import argparse
import time

import mlx.core as mx
import numpy as np


def timed(fn, reps: int) -> float:
    """Best-of, to cut scheduler noise. Returns ms."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    # Defaults are one GLM-4.7 weight component for a 43-expert run:
    # 43 x (5120 x 1536 / 2 bytes at 4-bit) ~= 169 MB.
    ap.add_argument("--experts", type=int, default=43)
    ap.add_argument("--expert-bytes", type=int, default=5120 * 1536 // 2)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    n = args.experts * args.expert_bytes
    print(f"span: {args.experts} experts x {args.expert_bytes / 1e6:.1f} MB "
          f"= {n / 1e6:.0f} MB\n")

    src = np.random.randint(0, 255, size=n, dtype=np.uint8)

    # --- 1. writeable contiguous view?
    buf = mx.zeros((n,), dtype=mx.uint8)
    mx.eval(buf)
    view = np.asarray(buf)
    print(f"view writeable={view.flags.writeable} "
          f"contiguous={view.flags.c_contiguous}")
    if not (view.flags.writeable and view.flags.c_contiguous):
        print("FAIL: MLX does not expose a writeable buffer; zero-copy is out")
        return 1

    # --- 2. the copy we are trying to delete
    def baseline():
        host = np.empty(n, dtype=np.uint8)
        host[:] = src              # stands in for the pread
        arr = mx.array(host)       # <- _materialize
        mx.eval(arr)
        return arr

    t_base = timed(baseline, args.reps)

    # --- 3. write straight into MLX memory, then slice rows out
    rows = (args.experts, args.expert_bytes)

    def zerocopy():
        b = mx.zeros((n,), dtype=mx.uint8)
        mx.eval(b)
        v = np.asarray(b)
        v[:] = src                 # stands in for the pread
        stacked = b.reshape(rows)
        per = [stacked[i] for i in range(args.experts)]
        mx.eval(*per)
        return per

    t_zero = timed(zerocopy, args.reps)

    # --- 4. allocation alone, since a run buffer is allocated per run
    def alloc_only():
        b = mx.zeros((n,), dtype=mx.uint8)
        mx.eval(b)
        return b

    t_alloc = timed(alloc_only, args.reps)

    # --- 5. write-only, to separate the pread stand-in from everything else
    def write_only():
        v = np.asarray(buf)
        v[:] = src

    t_write = timed(write_only, args.reps)

    gb = n / 1e9
    print(f"\n{'baseline (pread + mx.array copy)':<40} {t_base:7.1f} ms  "
          f"{gb / (t_base / 1e3):5.1f} GB/s")
    print(f"{'zero-copy (pread into MLX + slice)':<40} {t_zero:7.1f} ms  "
          f"{gb / (t_zero / 1e3):5.1f} GB/s")
    print(f"{'  of which: mx.zeros + eval':<40} {t_alloc:7.1f} ms")
    print(f"{'  of which: the write itself':<40} {t_write:7.1f} ms")
    print(f"\nspeedup {t_base / max(t_zero, 1e-9):.2f}x  "
          f"saved {t_base - t_zero:.1f} ms per {gb * 1e3:.0f} MB span")

    # --- 5b. The actual question.
    # _materialize copies per expert per component: a 43-expert run over 9
    # components is ~387 mx.array() calls of a few MB each. The alternative is
    # one call per component span (9) and MLX slices for the rows. Same bytes,
    # 43x fewer calls. Production measures the copy at 11.2 GB/s against the
    # 31.6 GB/s a single big copy gets, and per-call overhead is the only
    # candidate for that gap.
    span_np = np.asarray(buf).reshape(rows)

    def per_expert_copy():
        out = []
        for i in range(args.experts):
            out.append(mx.array(span_np[i]))
        mx.eval(*out)
        return out

    def per_span_copy():
        whole = mx.array(np.asarray(buf))
        stacked_ = whole.reshape(rows)
        out = [stacked_[i] for i in range(args.experts)]
        mx.eval(*out)
        return out

    t_per_expert = timed(per_expert_copy, args.reps)
    t_per_span = timed(per_span_copy, args.reps)
    print(f"\n{'copy per expert (what _materialize does)':<40} "
          f"{t_per_expert:7.1f} ms  {gb / (t_per_expert / 1e3):5.1f} GB/s")
    print(f"{'copy per span, slice rows':<40} "
          f"{t_per_span:7.1f} ms  {gb / (t_per_span / 1e3):5.1f} GB/s")
    print(f"{'':<40} -> {t_per_expert / max(t_per_span, 1e-9):.2f}x fewer ms "
          f"for the same bytes")
    a = per_expert_copy()
    b_ = per_span_copy()
    same = all(
        np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b_)
    )
    print(f"{'':<40} identical bytes: {same}")

    # --- 6. correctness: does a row slice hold the right bytes
    b = mx.zeros((n,), dtype=mx.uint8)
    mx.eval(b)
    np.asarray(b)[:] = src
    stacked = b.reshape(rows)
    for i in (0, args.experts // 2, args.experts - 1):
        got = np.asarray(stacked[i])
        want = src[i * args.expert_bytes : (i + 1) * args.expert_bytes]
        if not np.array_equal(got, want):
            print(f"FAIL: row {i} bytes differ")
            return 1
    print(f"rows 0/{args.experts // 2}/{args.experts - 1} match source bytes")

    # --- 7. does a dtype view on a row still avoid a copy (4-bit weights are
    # stored as uint32, and bf16 components go through .view)
    as_u32 = stacked.reshape(args.experts, -1)
    def view_rows():
        per = [as_u32[i].view(mx.uint32) for i in range(args.experts)]
        mx.eval(*per)
    t_view = timed(view_rows, args.reps)
    print(f"\n{'row + .view(uint32)':<40} {t_view:7.1f} ms  "
          f"{gb / (t_view / 1e3):5.1f} GB/s")
    if t_view > 0.5 * t_base:
        print("WARNING: .view on a row costs about as much as the copy - the "
              "stacked run must be handed to gather_qmm whole, not per expert")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
