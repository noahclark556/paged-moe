# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Can we pread() expert weights straight into a Metal-backed slab and run
gather_qmm on the slots?

This is the load-bearing experiment for the slab cache. Today the decode path
pays twice for the same bytes:

  * ~65 ms/token copying numpy read buffers into Metal (`mx.array(np_arr)`)
  * ~112 ms/token running 3 quantized_matmul calls *per expert* because cache
    entries are separate per-expert arrays. gather_qmm on pre-stacked experts
    is 2.4x faster at 235B shapes, but stacking per call costs exactly what it
    saves (see gather_qmm_probe.py).

Both vanish if experts *live* in one contiguous slab: reads land in their final
location, and gather_qmm addresses experts by slot index. That requires writing
into MLX-owned memory from a worker thread through a numpy view, which is
outside MLX's contract - so verify numerically before building on it.

Checks, at Qwen3-235B shapes:
  1. gather_qmm over pread-filled slots == reference on the original arrays
     (bit-identical, not close-enough)
  2. slab (pread -> gather_qmm) vs today's (pread -> copy -> per-expert qmm)

    python bench/slab_probe.py
"""

from __future__ import annotations

import os
import tempfile
import time

import numpy as np

import mlx.core as mx

HIDDEN = 4096
INTER = 1536
TOPK = 8
SLOTS = 64
BITS = 4
GROUP = 64
LAYERS = 94
REPS = 20


def timed(fn, *, warmup=3, reps=REPS):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / reps


def main():
    print(f"mlx {mx.__version__}  |  {SLOTS} slots, hidden={HIDDEN}, "
          f"inter={INTER}, {BITS}-bit/g{GROUP}\n", flush=True)

    # Reference experts, as a checkpoint would hold them (stacked, quantized).
    w = mx.random.normal((SLOTS, INTER, HIDDEN)).astype(mx.float16)
    wq, scales, biases = mx.quantize(w, group_size=GROUP, bits=BITS)
    mx.eval(wq, scales, biases)
    print(f"packed weight {tuple(wq.shape)} {wq.dtype}, "
          f"scales {tuple(scales.shape)} {scales.dtype}", flush=True)

    per_expert_bytes = wq[0].nbytes
    print(f"per-expert packed weight: {per_expert_bytes/1e6:.2f} MB", flush=True)

    # Write them out so we can pread() like the real loader does.
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "experts.bin")
    with open(path, "wb") as f:
        f.write(np.asarray(wq).tobytes())
    fd = os.open(path, os.O_RDONLY)

    try:
        # ---- the hack: allocate a slab, pread into a writeable numpy view ----
        slab = mx.zeros((SLOTS, INTER, HIDDEN * BITS // 32), dtype=mx.uint32)
        mx.eval(slab)
        flat = np.asarray(slab).reshape(-1).view(np.uint8)
        print(f"slab numpy view writeable={flat.flags.writeable}", flush=True)

        stride = per_expert_bytes
        t0 = time.perf_counter()
        for slot in range(SLOTS):
            got = os.preadv(fd, [flat[slot * stride : (slot + 1) * stride]],
                            slot * stride)
            assert got == stride, got
        read_ms = (time.perf_counter() - t0) * 1e3
        print(f"pread {SLOTS} experts into slab: {read_ms:.2f} ms "
              f"({SLOTS*stride/1e9/(read_ms/1e3):.1f} GB/s, page-cached)",
              flush=True)

        # ---- check 1: does gather_qmm see the pread'd bytes correctly? ----
        x = mx.random.normal((TOPK, 1, HIDDEN)).astype(mx.float16)
        idx = mx.array([3, 17, 40, 5, 62, 0, 29, 51], mx.uint32)
        mx.eval(x, idx)

        ref = mx.gather_qmm(x, wq, scales, biases, rhs_indices=idx,
                            transpose=True, group_size=GROUP, bits=BITS)
        got = mx.gather_qmm(x, slab, scales, biases, rhs_indices=idx,
                            transpose=True, group_size=GROUP, bits=BITS)
        mx.eval(ref, got)
        identical = bool(mx.all(ref == got).item())
        maxdiff = float(mx.abs(ref - got).max().item())
        print(f"\nCHECK 1 gather_qmm(slab) == gather_qmm(original): "
              f"{'IDENTICAL' if identical else f'DIFFERS (max {maxdiff})'}",
              flush=True)

        # Per-expert path on slab slices must agree too (stage-1 compute).
        per_ref = mx.concatenate([
            mx.quantized_matmul(
                x[i], slab[int(idx[i].item())], scales[int(idx[i].item())],
                biases[int(idx[i].item())], transpose=True,
                group_size=GROUP, bits=BITS,
            )
            for i in range(TOPK)
        ], axis=0)
        mx.eval(per_ref)
        same2 = bool(mx.all(per_ref == ref.reshape(per_ref.shape)).item())
        print(f"CHECK 2 per-expert(slab slices) == gather_qmm(slab): "
              f"{'IDENTICAL' if same2 else 'DIFFERS'}", flush=True)

        # ---- check 3: is it actually faster end to end? ----
        print("\n--- per-layer compute (decode, 1 row/expert) ---", flush=True)

        def slab_gather():
            return mx.gather_qmm(x, slab, scales, biases, rhs_indices=idx,
                                 transpose=True, group_size=GROUP, bits=BITS)

        def per_expert_slices():
            outs = []
            for i in range(TOPK):
                s = int(idx[i].item())
                outs.append(mx.quantized_matmul(
                    x[i], slab[s], scales[s], biases[s], transpose=True,
                    group_size=GROUP, bits=BITS))
            return mx.concatenate(outs, axis=0)

        # Today's path also pays a numpy->Metal copy per missed expert.
        host = np.asarray(wq[0]).copy()

        def copy_then_qmm():
            outs = []
            for i in range(TOPK):
                s = int(idx[i].item())
                w_dev = mx.array(host)  # the copy we want to delete
                outs.append(mx.quantized_matmul(
                    x[i], w_dev, scales[s], biases[s], transpose=True,
                    group_size=GROUP, bits=BITS))
            return mx.concatenate(outs, axis=0)

        g = timed(slab_gather)
        p = timed(per_expert_slices)
        c = timed(copy_then_qmm)
        for label, sec in (("gather_qmm on slab", g),
                           ("per-expert on slab slices", p),
                           ("per-expert + mx.array copy", c)):
            print(f"  {label:<30} {sec*1e3:7.3f} ms/layer  "
                  f"-> {sec*LAYERS*1e3:7.1f} ms/token", flush=True)
        print(f"\n  slab+gather vs per-expert:      {p/g:5.2f}x  "
              f"({(p-g)*LAYERS*1e3:6.1f} ms/token)", flush=True)
        print(f"  slab+gather vs copy+per-expert: {c/g:5.2f}x  "
              f"({(c-g)*LAYERS*1e3:6.1f} ms/token)", flush=True)
    finally:
        os.close(fd)
        os.remove(path)
        os.rmdir(tmp)


if __name__ == "__main__":
    main()
