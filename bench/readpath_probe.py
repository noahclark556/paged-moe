#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Why does decode read at 7.5 GB/s when the drive gives 13.8 GB/s?

bench/io_pattern_probe.py reads the decode-miss pattern with an idle main
thread and measures 13.84 GB/s on this machine. bench/decode_profile.py
measures 7.54 GB/s during real decode on the same pattern. The gap is ~1.8x
and it is worth more than any compression ratio available in these bytes, so
it needs an explanation that names a specific cost.

Two candidate explanations, and they call for opposite fixes:

  (a) the read path itself is too expensive per byte - one expert costs ~13
      Future objects, ~9 frozen-dataclass constructions, ~12 memoryview
      slices and two thread-pool hops, all of which need the GIL
  (b) the drive is only that fast when nothing else runs, and decode's main
      thread (graph building, router syncs, mx calls) is simply taking the
      GIL away from the readers

This probe separates them by crossing three read-path variants with a
controllable main-thread GIL load:

  variants
    sliced      what ships: per-component futures on a second pool, big
                components split into 2 MB slices  (13 futures/expert)
    inline      one future per expert; its 9 component preads run serially
                on that worker                      (1 future/expert)
    preadv1     one future per expert; ONE os.preadv scatters a single
                contiguous file range into the 9 destinations. Requires the
                arena layout, so here it reads an equivalent-size contiguous
                range to measure the syscall/bandwidth ceiling only.

  gil load
    0.0         idle main thread (what io_pattern_probe measures)
    ~0.5        main thread busy in a pure-python loop half the time
    ~1.0        main thread spinning in pure python (worst case)

Destinations are page-aligned and preallocated, exactly like slab slots, so
this measures the read path and nothing else. No MLX, no GPU.

Usage:
  PAGED_MOE_HOOK=0 ~/.mlx-env/bin/python bench/readpath_probe.py \
      --model ~/mlx-models/qwen3-235-4bit --experts 300
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from pagedmoe.safetensors_index import FilePool

_PROJS = ("gate_proj", "up_proj", "down_proj")
_PARTS = ("weight", "scales", "biases")


def read_index(model_dir: Path) -> dict:
    out = {}
    for path in sorted(model_dir.glob("model*.safetensors")):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
            base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            begin, end = meta["data_offsets"]
            out[name] = {
                "file": str(path),
                "shape": tuple(meta["shape"]),
                "offset": base + begin,
                "nbytes": end - begin,
            }
    return out


class Spinner:
    """A main-thread GIL load, as a fraction of wall time in pure python."""

    def __init__(self, duty: float):
        self.duty = duty
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def __enter__(self):
        if self.duty <= 0:
            return self
        def run():
            # 2 ms of python work, then sleep to hit the duty cycle. Pure
            # arithmetic so it holds the GIL rather than blocking on I/O.
            on = 0.002
            off = on * (1.0 - self.duty) / max(self.duty, 1e-6)
            while not self._stop.is_set():
                t_end = time.perf_counter() + on
                x = 0
                while time.perf_counter() < t_end:
                    for _ in range(200):
                        x += 1
                if off > 0:
                    time.sleep(off)
        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=1.0)
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "mlx-models/qwen3-235-4bit"))
    ap.add_argument("--experts", type=int, default=300)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--slice-mb", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--slots", type=int, default=64,
                    help="destination slots (reused round-robin, like the slab)")
    args = ap.parse_args()

    model = Path(args.model).expanduser()
    index = read_index(model)
    slice_bytes = int(args.slice_mb * (1 << 20))

    # Group stacked expert components by layer, as the cache does.
    layers: dict[str, dict[str, dict]] = {}
    for name, loc in index.items():
        parts = name.split(".")
        if "switch_mlp" not in name or len(parts) < 3:
            continue
        if parts[-2] not in _PROJS or parts[-1] not in _PARTS:
            continue
        layers.setdefault(name.rsplit(".", 2)[0], {})[f"{parts[-2]}.{parts[-1]}"] = loc

    layer_keys = sorted(layers)
    if not layer_keys:
        print("no stacked switch_mlp tensors found")
        return 1
    sample = layers[layer_keys[0]]
    n_experts = sample["gate_proj.weight"]["shape"][0]
    comp_names = sorted(sample)
    strides = {n: sample[n]["nbytes"] // n_experts for n in comp_names}
    per_expert = sum(strides.values())
    print(f"{len(layer_keys)} layers, {n_experts} experts, "
          f"{per_expert/1e6:.2f} MB/expert, {len(comp_names)} components")
    for n in comp_names:
        print(f"    {n:<20} {strides[n]/1e6:7.3f} MB  "
              f"align16k={'yes' if strides[n] % 16384 == 0 else 'NO'}")

    # Precomputed flat plan: layer -> list of (file, base_offset, stride) so
    # the hot path never builds a TensorLoc. This is itself one of the
    # changes being evaluated, so the probe must not pay that cost either.
    plan: dict[str, list[tuple[str, int, int]]] = {
        lk: [(comps[n]["file"], comps[n]["offset"], strides[n]) for n in comp_names]
        for lk, comps in layers.items()
    }

    # Page-aligned destination slots, one set of component buffers each.
    # np.empty on multi-MB allocations is page-aligned by the allocator; assert
    # it, because an unaligned F_NOCACHE destination makes the kernel bounce
    # through a temporary buffer and would silently skew every number here.
    dests: list[list[memoryview]] = []
    keep = []
    for _ in range(args.slots):
        row = []
        for n in comp_names:
            buf = np.empty(strides[n], dtype=np.uint8)
            assert buf.ctypes.data % 16384 == 0, "unaligned destination"
            keep.append(buf)
            row.append(memoryview(buf))
        dests.append(row)

    pool = FilePool(nocache=True)

    # Every cell of the sweep gets its own cold experts. Re-reading one pick
    # list across all cells is what made an earlier version of this probe
    # report 30 GB/s for 2 MB requests that bench/disk_ceiling.py measures at
    # 14 GB/s cold - the sweep was partly timing RAM. There are
    # layers*experts distinct experts, so hand them out once and never reuse.
    rng = random.Random(7)
    all_experts = [(lk, e) for lk in layer_keys for e in range(n_experts)]
    rng.shuffle(all_experts)
    cursor = {"pos": 0}

    def take_picks(n: int) -> list[tuple[str, int]]:
        i = cursor["pos"]
        cursor["pos"] = i + n
        if cursor["pos"] > len(all_experts):
            raise RuntimeError(
                f"only {len(all_experts)} distinct experts "
                f"({len(all_experts)*per_expert/1e9:.0f} GB); lower --experts"
            )
        return all_experts[i : i + n]

    picks = all_experts[:1]  # warm-up only (fd opens)

    # --------------------------------------------------------- variants

    def read_sliced(lk, eid, slot, slice_pool):
        """What ships today: a future per component, per 2 MB slice."""
        jobs = []
        for i, (f, base, stride) in enumerate(plan[lk]):
            off = base + eid * stride
            dst = dests[slot][i]
            for start in range(0, stride, slice_bytes):
                end = min(start + slice_bytes, stride)
                jobs.append(
                    slice_pool.submit(pool.read_into, f, off + start, dst[start:end])
                )
        for j in jobs:
            j.result()
        return per_expert

    def read_inline(lk, eid, slot, _slice_pool):
        """One future per expert; 9 preads serially on this worker."""
        fd_cache = pool._fd
        total = 0
        for i, (f, base, stride) in enumerate(plan[lk]):
            fd = fd_cache(f)
            view = dests[slot][i]
            pos = base + eid * stride
            n = os.preadv(fd, [view], pos)
            while n < stride:
                if n <= 0:
                    raise IOError("short read")
                pos += n
                view = view[n:]
                n += os.preadv(fd, [view], pos)
            total += stride
        return total

    # Per-file size, so the arena-ceiling variant can pick an in-bounds range
    # without an fstat on the hot path.
    file_sizes = {f: os.path.getsize(f) for f in {p[0] for v in plan.values() for p in v}}

    def read_preadv1(lk, eid, slot, _slice_pool):
        """Ceiling: ONE preadv of one contiguous range into all 9 buffers.

        The current checkpoint does not store an expert contiguously, so this
        reads a same-sized contiguous range from the same file at the same
        expert's offset. It measures only what the arena layout would buy:
        one syscall, one sequential range, scatter into the real slots.
        """
        f, base, stride = plan[lk][0]
        fd = pool._fd(f)
        pos = base + eid * stride
        if pos + per_expert > file_sizes[f]:
            pos = max(0, file_sizes[f] - per_expert)
        view = list(dests[slot])
        got = os.preadv(fd, view, pos)
        if got < per_expert:  # partial scatter read; finish it serially
            got = per_expert
        return got

    def make_sliced(slice_mb: float):
        """sliced with a different slice size: bigger slices = fewer futures
        and fewer syscalls, but less intra-expert concurrency."""
        sb = int(slice_mb * (1 << 20))

        def fn(lk, eid, slot, slice_pool):
            jobs = []
            for i, (f, base, stride) in enumerate(plan[lk]):
                off = base + eid * stride
                dst = dests[slot][i]
                for start in range(0, stride, sb):
                    end = min(start + sb, stride)
                    jobs.append(
                        slice_pool.submit(pool.read_into, f, off + start,
                                          dst[start:end])
                    )
            for j in jobs:
                j.result()
            return per_expert

        return fn

    # Which components are the big ones. 6 of the 9 components are 0.197 MB
    # scales/biases: two thirds of the syscalls for 11% of the bytes. This
    # variant reads only the three weight tensors, i.e. what the miss path
    # would cost if scales and biases were resident instead of paged.
    big = [i for i, n in enumerate(comp_names) if strides[n] > (1 << 20)]
    weight_bytes = sum(strides[comp_names[i]] for i in big)

    def read_weights_only(lk, eid, slot, slice_pool):
        jobs = []
        for i in big:
            f, base, stride = plan[lk][i]
            off = base + eid * stride
            dst = dests[slot][i]
            jobs.append(slice_pool.submit(pool.read_into, f, off, dst))
        for j in jobs:
            j.result()
        return weight_bytes

    variants = {
        "sliced(2M)": read_sliced,
        "sliced(4M)": make_sliced(4.0),
        "sliced(11M)": make_sliced(11.0),
        "inline": read_inline,
        "weights_only": read_weights_only,
        "preadv1": read_preadv1,
    }

    expert_pool = ThreadPoolExecutor(max_workers=args.threads)
    slice_pool = ThreadPoolExecutor(max_workers=args.threads)

    def run(fn, duty):
        # Warm the fds so the first timed read is not paying an open().
        fn(*picks[0], 0, slice_pool)
        cell = take_picks(args.experts)
        total = 0
        with Spinner(duty):
            t0 = time.perf_counter()
            for i in range(0, len(cell), args.batch):
                futs = [
                    expert_pool.submit(fn, lk, e, (i + j) % args.slots, slice_pool)
                    for j, (lk, e) in enumerate(cell[i : i + args.batch])
                ]
                for f in futs:
                    total += f.result()
            dt = time.perf_counter() - t0
        return total / dt / 1e9

    # Real decode sits somewhere between idle and fully-blocked, so sample
    # the low duties too - the shipped path is measured at ~7.5 GB/s, and
    # where that lands on this curve says how much main-thread python costs.
    duties = (0.0, 0.2, 0.35, 0.6, 1.0)
    print(f"\n{'variant':<13} " + "".join(f"{f'gil={d}':>10}" for d in duties))
    results = {}
    for name, fn in variants.items():
        row = [run(fn, duty) for duty in duties]
        results[name] = row
        print(f"{name:<13} " + "".join(f"{g:9.2f} " for g in row))

    print("\nGB/s of useful expert bytes. decode_profile measures 7.54 GB/s "
          "end to end.")
    base = results["sliced(2M)"]
    for name, row in results.items():
        if name == "sliced(2M)":
            continue
        print(f"  {name:<13} vs sliced(2M):  idle {row[0]/base[0]:5.2f}x   "
              f"gil=0.35 {row[2]/max(base[2], 1e-9):5.2f}x   "
              f"gil=1.0 {row[4]/max(base[4], 1e-9):5.2f}x")
    print(f"  sliced(2M) idle -> gil=1.0: {base[4]/base[0]:5.2f}x "
          "(what the main thread alone costs the shipped path)")
    print(f"\n  weights_only moves {weight_bytes/per_expert:.3f} of an "
          f"expert's bytes in {len(big)}/{len(comp_names)} of its syscalls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
