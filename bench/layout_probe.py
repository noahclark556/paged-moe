# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What does the scattered per-expert read pattern cost, versus a contiguous one?

bench/drive_idle.py measures decode moving 2.65 GB/token at 9.6 GB/s while the
drive is busy, against the ~14.4 GB/s bench/disk_ceiling.py gets. The gap is
request shape, not bandwidth: safetensors stores each *tensor* contiguously, so
one expert is scattered across 9 tensors (3 projections x weight/scales/biases)
at 9 unrelated file offsets. Six of those nine are 0.197 MB scales/biases - two
thirds of the IOPS for a ninth of the bytes.

This probe reads the real offsets from a real checkpoint and times the patterns
against each other at matched queue depth on cold, distinct ranges:

  scattered  what the engine does today: 9 preads per expert
  grouped    same 9 reads, but coalesced where the ranges happen to touch
  big-only   only the 3 weight reads (i.e. scales/biases resident in RAM)
  contiguous 1 pread of the whole 10.6 MB (what an offline relayout buys)

"contiguous" reads an equally-sized cold span from the same files rather than a
relayout that does not exist yet - the drive does not care what the bytes mean,
and this is meant to size the prize before paying for it.

Usage:
  python bench/layout_probe.py --model ~/mlx-models/qwen3-235-4bit
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import json
import os
import re
import threading
import time
from collections import defaultdict

F_NOCACHE = 48


def read_header(path: str) -> dict:
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        return json.loads(fh.read(n)), 8 + n


def expert_extents(model_dir: str):
    """[(layer, expert) -> [(path, offset, nbytes), ...]] from real metadata."""
    per: dict[tuple, list] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(model_dir, "model*.safetensors"))):
        hdr, base = read_header(path)
        for k, meta in hdr.items():
            if k == "__metadata__" or "switch_mlp" not in k:
                continue
            m = re.search(r"layers\.(\d+)", k)
            shape = meta.get("shape") or []
            if m is None or not shape or shape[0] < 8:
                continue
            layer, n_exp = int(m.group(1)), shape[0]
            st, en = meta["data_offsets"]
            stride = (en - st) // n_exp
            for e in range(n_exp):
                per[(layer, e)].append((path, base + st + e * stride, stride))
    return per


def open_fds(paths) -> dict:
    fds = {}
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        fcntl.fcntl(fd, F_NOCACHE, 1)
        fds[p] = fd
    return fds


def run(jobs, fds, threads: int) -> tuple[float, int]:
    """Issue `jobs` [(path, off, size)] across `threads`, return (seconds, bytes)."""
    idx = [0]
    lock = threading.Lock()
    total = [0]

    def worker():
        buf = bytearray(16 << 20)
        mv = memoryview(buf)
        got = 0
        while True:
            with lock:
                i = idx[0]
                idx[0] += 1
            if i >= len(jobs):
                break
            path, off, size = jobs[i]
            dst = mv[:size]
            fd, o, n = fds[path], off, 0
            while n < size:
                r = os.preadv(fd, [dst[n:]], o + n)
                if r <= 0:
                    raise OSError(f"short read at {o + n}")
                n += r
            got += size
        with lock:
            total[0] += got

    ts = [threading.Thread(target=worker) for _ in range(threads)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return time.perf_counter() - t0, total[0]


def coalesce(parts, gap: int = 0):
    """Merge ranges in the same file that are adjacent (or within `gap`)."""
    out = []
    for path, group in _by_path(parts):
        group.sort(key=lambda x: x[1])
        cur_off, cur_end = group[0][1], group[0][1] + group[0][2]
        for _, off, size in group[1:]:
            if off <= cur_end + gap:
                cur_end = max(cur_end, off + size)
            else:
                out.append((path, cur_off, cur_end - cur_off))
                cur_off, cur_end = off, off + size
        out.append((path, cur_off, cur_end - cur_off))
    return out


def _by_path(parts):
    d = defaultdict(list)
    for p in parts:
        d[p[0]].append(p)
    return d.items()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--experts", type=int, default=1200, help="experts per pattern")
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()

    model = os.path.expanduser(args.model)
    print(f"reading metadata from {model} ...")
    per = expert_extents(model)
    keys = sorted(per)
    if not keys:
        raise SystemExit("no switch_mlp expert tensors found")
    sizes = [s for _, _, s in per[keys[0]]]
    per_expert = sum(sizes)
    print(
        f"{len(keys)} experts, {len(sizes)} components each, "
        f"{per_expert / 1e6:.2f} MB/expert"
    )
    print("  component sizes (MB): " + ", ".join(f"{s / 1e6:.3f}" for s in sorted(sizes)))

    fds = open_fds({p for k in keys for p, _, _ in per[k]})

    # Distinct experts per pattern so nothing is warm from a previous pattern.
    # Strided so every pattern spans the whole checkpoint rather than one shard.
    cursor = [0]

    def take(n):
        out = [keys[(cursor[0] + i * 7919) % len(keys)] for i in range(n)]
        cursor[0] += n * 7919
        return out

    patterns = {}

    sel = take(args.experts)
    patterns["scattered  (today: 9 reads/expert)"] = [
        part for k in sel for part in per[k]
    ]

    sel = take(args.experts)
    patterns["grouped    (coalesce adjacent)"] = [
        r for k in sel for r in coalesce(per[k])
    ]

    sel = take(args.experts)
    patterns["big-only   (scales/biases resident)"] = [
        p for k in sel for p in per[k] if p[2] > 1 << 20
    ]

    # Contiguous: same total bytes, one request per expert. Walks distinct
    # spans across every shard - wrapping inside one shard re-reads it and
    # measures the page cache (that read 28 GB/s, well past what the drive
    # can do, which is how the bug announced itself).
    step = per_expert
    contig, need = [], args.experts
    for path in sorted(fds):
        room = os.path.getsize(path)
        off = 0
        while off + step <= room and need > 0:
            contig.append((path, off, step))
            off += step
            need -= 1
        if need == 0:
            break
    if need:
        raise SystemExit(
            f"checkpoint too small for {args.experts} distinct contiguous spans"
        )
    patterns["contiguous (1 read/expert, relayout)"] = contig

    print(f"\nqueue depth {args.threads}, {args.experts} experts per pattern\n")
    print(f"{'pattern':38s} {'reads':>7s} {'MB':>9s} {'avg KB':>8s} {'GB/s':>7s}")
    for name, jobs in patterns.items():
        dt, got = run(jobs, fds, args.threads)
        print(
            f"{name:38s} {len(jobs):7d} {got / 1e6:9.1f} "
            f"{got / len(jobs) / 1e3:8.1f} {got / dt / 1e9:7.2f}"
        )

    for fd in fds.values():
        os.close(fd)


if __name__ == "__main__":
    main()
