#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does the file offset alignment of an F_NOCACHE pread change what it costs?

disk_ceiling.py and gil_probe.py round every offset down to 16 KB. The engine
reads expert components at their real safetensors offsets, which sit wherever
the header and the tensors before them left them. This reads the same kind of
ranges (real expert components, 1 MB capped preads, page-aligned
destinations) three ways and reports useful GB/s, CPU time and file cache
growth for each:

  asis      what ships: offset as stored, destination page aligned
  aligned   offset rounded down to 16 KB, length rounded up, the component
            sits `offset % 16384` bytes into the destination
  coalign   offset as stored, destination shifted by the same `offset % 16384`

Every pass draws experts no other pass touched, and the modes rotate so drift
lands on all of them.

Usage:
  PAGED_MOE_HOOK=0 ~/.mlx-env/bin/python bench/align_probe.py \
      --model ~/mlx-models/qwen3-235-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from pagedmoe import pagecache
from pagedmoe.safetensors_index import MAX_IO_BYTES, FilePool

PAGE = 16384
_PROJS = ("gate_proj", "up_proj", "down_proj")
_PARTS = ("weight", "scales", "biases")


def expert_components(model_dir: Path) -> list[list[tuple[str, int, int]]]:
    """[(file, offset, nbytes)] per (layer, expert), 9 components each."""
    tensors = {}
    for path in sorted(model_dir.glob("model*.safetensors")):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            b, e = meta["data_offsets"]
            tensors[name] = (str(path), base + b, e - b, meta["shape"])
    layers: dict[str, dict] = {}
    for name, loc in tensors.items():
        if ".switch_mlp." not in name:
            continue
        prefix, rest = name.split(".switch_mlp.", 1)
        layers.setdefault(prefix, {})[rest] = loc
    out = []
    for prefix in sorted(layers):
        parts = layers[prefix]
        keys = [f"{p}.{q}" for p in _PROJS for q in _PARTS]
        if not all(k in parts for k in keys):
            continue
        n_exp = parts[keys[0]][3][0]
        for e in range(n_exp):
            comps = []
            for k in keys:
                path, off, nbytes, shape = parts[k]
                per = nbytes // shape[0]
                comps.append((path, off + e * per, per))
            out.append(comps)
    return out


_SIZES: dict[str, int] = {}


def plan(comp: tuple[str, int, int], mode: str) -> tuple[str, int, int, int]:
    """(file, read_offset, read_len, dest_shift)."""
    path, off, n = comp
    if mode == "asis":
        return path, off, n, 0
    head = off % PAGE
    if mode == "aligned":
        start = off - head
        size = _SIZES.setdefault(path, os.path.getsize(path))
        length = min(-(-(head + n) // PAGE) * PAGE, size - start)
        return path, start, length, 0
    return path, off, n, head  # coalign


def run_pass(pool, experts, mode, threads):
    jobs = [plan(c, mode) for ex in experts for c in ex]
    useful = sum(c[2] for ex in experts for c in ex)
    read = sum(j[2] for j in jobs)
    biggest = max(j[2] + j[3] for j in jobs)
    bufs = [np.empty(biggest + PAGE, dtype=np.uint8) for _ in range(threads)]
    for b in bufs:
        assert b.ctypes.data % PAGE == 0, "unaligned destination"

    def one(i_job):
        i, (path, off, n, shift) = i_job
        buf = bufs[i % threads]
        fd = pool._fd(path)
        view = memoryview(buf)[shift: shift + n]
        pos = off
        while len(view):
            got = os.preadv(fd, [view[:MAX_IO_BYTES]], pos)
            if got <= 0:
                raise IOError(f"short read at {pos} in {path}")
            pos += got
            view = view[got:]

    fc0 = pagecache.file_cache_bytes() or 0
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        # Round-robin by thread slot so no two in-flight jobs share a buffer.
        for start in range(0, len(jobs), threads):
            list(ex.map(one, enumerate(jobs[start:start + threads], start)))
    dt = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    fc1 = pagecache.file_cache_bytes() or 0
    return {
        "useful_gb_s": useful / dt / 1e9,
        "read_gb": read / 1e9,
        "useful_gb": useful / 1e9,
        "sys_s": r1.ru_stime - r0.ru_stime,
        "user_s": r1.ru_utime - r0.ru_utime,
        "filecache_gb": (fc1 - fc0) / 1e9,
        "wall_s": dt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--experts", type=int, default=400,
                    help="experts per pass (9 components each)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--modes", default="asis,aligned,coalign")
    args = ap.parse_args()

    model = Path(args.model).expanduser()
    files = sorted(str(p) for p in model.glob("model*.safetensors"))
    experts = expert_components(model)
    heads = [c[1] % PAGE for ex in experts for c in ex]
    print(f"{model.name}: {len(experts)} experts, "
          f"{sum(h == 0 for h in heads)}/{len(heads)} components 16K aligned")
    modes = args.modes.split(",")
    need = args.experts * args.rounds * len(modes)
    if need > len(experts):
        raise SystemExit(f"need {need} distinct experts, have {len(experts)}")
    rng = random.Random(7)
    rng.shuffle(experts)
    pagecache.drop(files)

    pool = FilePool(nocache=True)
    totals = {m: [] for m in modes}
    cursor = 0
    for r in range(args.rounds):
        order = modes[r % len(modes):] + modes[:r % len(modes)]
        for mode in order:
            batch = experts[cursor:cursor + args.experts]
            cursor += args.experts
            res = run_pass(pool, batch, mode, args.threads)
            totals[mode].append(res)
            print(f"  round {r} {mode:8s} {res['useful_gb_s']:6.2f} GB/s  "
                  f"sys {res['sys_s']:5.2f}s user {res['user_s']:5.2f}s  "
                  f"filecache {res['filecache_gb']:+6.2f} GB  "
                  f"read {res['read_gb']:.2f}/{res['useful_gb']:.2f} GB", flush=True)
            pagecache.drop(files)
    print("\nmean over rounds")
    for mode in modes:
        rs = totals[mode]
        print(f"  {mode:8s} {np.mean([x['useful_gb_s'] for x in rs]):6.2f} GB/s  "
              f"sys {np.mean([x['sys_s'] for x in rs]):5.2f}s  "
              f"filecache {np.mean([x['filecache_gb'] for x in rs]):+6.2f} GB")
    pool.close()


if __name__ == "__main__":
    main()
