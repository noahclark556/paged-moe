#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does decompression beat the SSD? Settles whether compression can pay here.

bench/entropy_probe.py found real lossless headroom in MLX-quantized expert
bytes: 0.79 of stored size using only contexts a decoder has available
(order-1 on the 4-bit codes, bias predicted from scale). 21% fewer bytes per
miss is a big number and it is genuinely there.

It is also not sufficient, and this is the probe that says so. Compression
pays on the miss path only if

    decompress_time(output_bytes) < bytes_saved / drive_bandwidth

bench/disk_ceiling.py measures this machine's drive at ~14.5 GB/s for the
decode access shape. So saving 21% of an expert buys 0.21 * 10.62 MB /
14.5 GB/s = 154 us, and the decompressor has to reconstruct the full
10.62 MB inside that - a required throughput of ~69 GB/s per expert stream,
which is where the whole idea lives or dies.

This measures what real codecs actually do on real expert bytes, in
decompressed GB/s, single-threaded and scaled by core count, then prints the
break-even bandwidth each one would need. The comparison is deliberately
generous to compression: it ignores that the decompressor's cores are the
same cores the reader threads and the main thread need, and that its output
writes compete for the same memory bandwidth the GPU is using.

Usage:
  PAGED_MOE_HOOK=0 ~/.mlx-env/bin/python bench/decompress_ceiling.py \
      --model ~/mlx-models/qwen3-235-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import time
import zlib
from pathlib import Path

import numpy as np

CODECS = {}

try:
    import zstandard as zstd

    CODECS["zstd-1"] = (
        lambda b: zstd.ZstdCompressor(level=1).compress(b),
        lambda b: zstd.ZstdDecompressor().decompress(b),
    )
    CODECS["zstd-9"] = (
        lambda b: zstd.ZstdCompressor(level=9).compress(b),
        lambda b: zstd.ZstdDecompressor().decompress(b),
    )
except Exception:
    pass

try:
    import lz4.block as lz4b

    CODECS["lz4"] = (
        lambda b: lz4b.compress(b, store_size=True),
        lambda b: lz4b.decompress(b),
    )
except Exception:
    pass

CODECS["zlib-6"] = (lambda b: zlib.compress(b, 6), lambda b: zlib.decompress(b))


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
                "file": str(path), "dtype": meta["dtype"],
                "shape": tuple(meta["shape"]),
                "offset": base + begin, "nbytes": end - begin,
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "mlx-models/qwen3-235-4bit"))
    ap.add_argument("--layer", type=int, default=47)
    ap.add_argument("--drive-gbs", type=float, default=14.5,
                    help="measured by bench/disk_ceiling.py")
    ap.add_argument("--cores", type=int, default=max(1, (os.cpu_count() or 8) - 4),
                    help="cores a decompressor could realistically use while "
                         "the main thread and 16 readers also run")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    model = Path(args.model).expanduser()
    index = read_index(model)

    # One expert's worth of real bytes, all nine components, as the miss path
    # would read them.
    blobs = []
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for part in ("weight", "scales", "biases"):
            key = f"model.layers.{args.layer}.mlp.switch_mlp.{proj}.{part}"
            if key not in index:
                continue
            loc = index[key]
            stride = loc["nbytes"] // loc["shape"][0]
            with open(loc["file"], "rb") as f:
                f.seek(loc["offset"] + 3 * stride)  # an arbitrary expert
                blobs.append(f.read(stride))
    raw = b"".join(blobs)
    n_raw = len(raw)
    print(f"one expert, layer {args.layer}: {n_raw/1e6:.2f} MB across "
          f"{len(blobs)} components")
    print(f"drive {args.drive_gbs} GB/s (bench/disk_ceiling.py), "
          f"{args.cores} cores assumed available\n")

    print(f"{'codec':<10}{'ratio':>8}{'1 core':>10}{f'{args.cores} cores':>10}"
          f"{'saved us':>10}{'decomp us':>11}{'verdict':>10}")
    for name, (enc, dec) in CODECS.items():
        comp = enc(raw)
        assert dec(comp) == raw, f"{name} roundtrip failed"
        dec(comp)  # warm
        t0 = time.perf_counter()
        for _ in range(args.reps):
            dec(comp)
        dt = (time.perf_counter() - t0) / args.reps
        gbs = n_raw / dt / 1e9
        ratio = len(comp) / n_raw
        # Time the drive would have spent on the bytes we did not read.
        saved_us = (n_raw - len(comp)) / (args.drive_gbs * 1e9) * 1e6
        # Decompress time if the work spreads perfectly over `cores`.
        dec_us = dt / args.cores * 1e6
        verdict = "WIN" if dec_us < saved_us else "loss"
        print(f"{name:<10}{ratio:8.3f}{gbs:10.2f}{gbs*args.cores:10.2f}"
              f"{saved_us:10.0f}{dec_us:11.0f}{verdict:>10}")

    print("\n'1 core'/'N cores' are decompressed GB/s. 'saved us' is the drive "
          "time the\nunread bytes would have cost; 'decomp us' is the "
          "reconstruction cost at N cores.")
    best_ratio = min(len(enc(raw)) / n_raw for enc, _ in CODECS.values())
    need = n_raw / ((n_raw - n_raw * best_ratio) / (args.drive_gbs * 1e9)) / 1e9
    print(f"\nBreak-even: at the best ratio here ({best_ratio:.3f}) a "
          f"decompressor must sustain\n  {need:.0f} GB/s of output to be free. "
          f"At the 0.79 entropy-probe ceiling it must sustain\n  "
          f"{n_raw / ((n_raw - n_raw*0.79) / (args.drive_gbs*1e9)) / 1e9:.0f}"
          f" GB/s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
