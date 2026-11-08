# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What does one decode-path sync actually cost, and can we avoid the copy?

decode_profile.py attributes ~150 ms/token of a 235B decode to "sync" (94
per-layer router->CPU drains) and ~65 ms/token to numpy->Metal copies, while
actual GPU kernels account for ~11 ms. Both of those are implementation
overhead rather than physics, so before optimizing either one we need to know
what the floor is:

  1. bare mx.eval round-trip on a tiny array (Metal commit + completion)
  2. the same with realistic work queued behind it
  3. np.asarray() of a GPU-computed array - the exact pattern
     StreamedSwitchGLU uses to get router indices to the CPU
  4. whether 94 sequential drains cost 94x one drain (i.e. no amortization)
  5. whether MLX exposes writable memory we can pread() into directly,
     which would remove the numpy->Metal copy entirely

Run: python bench/sync_probe.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np

import mlx.core as mx


def timeit(fn, n=200, warmup=20):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return {
        "mean_ms": statistics.mean(samples),
        "p50_ms": samples[len(samples) // 2],
        "p90_ms": samples[int(len(samples) * 0.9)],
    }


def show(label, r, note=""):
    print(
        f"{label:<44} mean {r['mean_ms']:7.3f} ms  p50 {r['p50_ms']:7.3f}  "
        f"p90 {r['p90_ms']:7.3f}  {note}",
        flush=True,
    )


def main():
    print(f"mlx {mx.__version__}\n", flush=True)

    small = mx.array([1, 2, 3, 4, 5, 6, 7, 8], mx.uint32)
    mx.eval(small)

    # 1. Bare round trip: nothing to compute, just commit + wait.
    show("mx.eval(already-evaluated tiny)", timeit(lambda: mx.eval(small)))

    def fresh_eval():
        mx.eval(small + 1)

    show("mx.eval(tiny add)", timeit(fresh_eval), "= commit+drain floor")

    # 2. Drain with realistic per-layer work queued behind it. A 235B decode
    #    layer is ~4096-dim attention projections; emulate one block's worth.
    w = mx.random.normal((4096, 4096)).astype(mx.float16)
    x = mx.random.normal((1, 4096)).astype(mx.float16)
    mx.eval(w, x)

    def work_then_drain():
        y = x @ w
        mx.eval(y)

    show("matmul 1x4096x4096 + eval", timeit(work_then_drain, n=100))

    # 3. The actual pattern: router logits -> top-k -> indices to CPU.
    gate = mx.random.normal((4096, 128)).astype(mx.float16)
    mx.eval(gate)

    def router_sync():
        logits = x @ gate
        inds = mx.argpartition(logits, kth=-8, axis=-1)[..., -8:]
        return np.asarray(inds)

    show("router matvec + np.asarray(indices)", timeit(router_sync, n=100),
         "<- per MoE layer")

    # 4. 94 of them back to back: does anything amortize?
    def full_token():
        for _ in range(94):
            router_sync()

    r = timeit(full_token, n=5, warmup=2)
    per = r["mean_ms"] / 94
    show("94x router_sync (one token's syncs)", r, f"= {per:.3f} ms/layer")

    # 4b. Same count of drains, but batched into ONE eval: the upper bound on
    #     what batching syncs could ever buy.
    def batched_94():
        outs = []
        for _ in range(94):
            logits = x @ gate
            outs.append(mx.argpartition(logits, kth=-8, axis=-1)[..., -8:])
        mx.eval(outs)
        return [np.asarray(o) for o in outs]

    r2 = timeit(batched_94, n=5, warmup=2)
    show("94x router, ONE eval + 94 asarray", r2,
         f"= {r2['mean_ms'] / 94:.3f} ms/layer")

    # 5. Zero-copy: can we pread() straight into memory MLX will use?
    print("\n--- zero-copy probes ---", flush=True)
    buf_mx = mx.zeros((1024,), mx.uint8)
    mx.eval(buf_mx)
    for name, fn in [
        ("np.asarray(mx) writeable", lambda: np.asarray(buf_mx).flags.writeable),
        ("memoryview(mx).readonly", lambda: memoryview(buf_mx).readonly),
        ("has __array_interface__", lambda: hasattr(buf_mx, "__array_interface__")),
        ("has __dlpack__", lambda: hasattr(buf_mx, "__dlpack__")),
        ("has __buffer__", lambda: hasattr(buf_mx, "__buffer__")),
    ]:
        try:
            print(f"  {name:<34} {fn()}", flush=True)
        except Exception as e:
            print(f"  {name:<34} ERROR {type(e).__name__}: {e}", flush=True)

    # Does writing through a numpy view of an mx array actually land?
    try:
        view = np.asarray(buf_mx)
        view[0] = 42
        mx.eval(buf_mx)
        print(f"  write-through numpy view -> mx[0] = {int(buf_mx[0])} "
              f"(42 means ZERO-COPY WORKS)", flush=True)
    except Exception as e:
        print(f"  write-through numpy view       ERROR {type(e).__name__}: {e}",
              flush=True)

    # Cost of the copy we currently pay, per 10.6 MB expert.
    nb = 10_600_000
    host = np.empty(nb, dtype=np.uint8)
    r3 = timeit(lambda: mx.eval(mx.array(host)), n=30, warmup=5)
    gbs = nb / (r3["mean_ms"] / 1e3) / 1e9
    show("\nmx.array(10.6 MB numpy) + eval", r3, f"= {gbs:.1f} GB/s copy")


if __name__ == "__main__":
    main()
