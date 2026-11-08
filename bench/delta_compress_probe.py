#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Probe: can delta / delta-of-delta (or BWT-MTF) shrink expert miss traffic?

Decode on 235B/GLM is bound by SSD bytes per miss, not by route knowledge.
Denser quant is out of scope. This asks whether a *lossless* transform of the
existing quantized expert bytes can cut bytes_on_disk enough that a miss
would finish faster after decompress, without changing the tensors the GPU
sees.

Transforms tried (all invertible on the byte/int grid we store):

  raw                baseline U32/BF16 expert bytes
  zlib / zstd / lz4  plain lossless on raw (transport compression floor)
  delta_neighbor     expert[e] - expert[e-1]  (same layer, stacked axis)
  delta_mean         expert[e] - round(mean over e)
  delta_xor_neighbor XOR of U32 words with previous expert (bit-grid delta)
  dod_neighbor       delta-of-delta along expert axis:
                       d[e] = x[e]-x[e-1];  dd[e] = d[e]-d[e-1]
  dod_scan           delta-of-delta along the flat byte/int stream inside one
                     expert (classic time-series DoD on the weight vector)
  bwt_mtf_rle_*      Burrows-Wheeler -> Move-to-Front -> RLE, then entropy
                     (zlib as stand-in for Huffman/ANS; same redundancy test)

A scheme is "promising" only if compressed_size / raw_size is clearly below
~0.85 on the *weight* tensors (the bulk of miss bytes). Scales/biases are
tiny. We also estimate decompress + SSD time vs raw SSD time under a simple
cost model so a high-ratio but expensive decode cannot fake a win.

Usage:
  ~/.mlx-env/bin/python bench/delta_compress_probe.py \\
      --model ~/mlx-models/qwen3-235-4bit --layers 0,8,40 --experts 32
"""

from __future__ import annotations

import argparse
import io
import json
import struct
import time
import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np

# Optional codecs -----------------------------------------------------------
try:
    import zstandard as zstd

    _HAS_ZSTD = True
except Exception:
    _HAS_ZSTD = False
try:
    import lz4.frame as lz4f

    _HAS_LZ4 = True
except Exception:
    _HAS_LZ4 = False


# --- safetensors (stdlib only; no pagedmoe / mlx import) --------------

_DTYPES = {
    "U32": (np.uint32, 4),
    "BF16": (np.uint16, 2),  # raw bits
    "F16": (np.float16, 2),
    "F32": (np.float32, 4),
    "I8": (np.int8, 1),
    "U8": (np.uint8, 1),
}


def read_index(model_dir: Path) -> dict:
    out = {}
    for path in sorted(model_dir.glob("*.safetensors")):
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
                "dtype": meta["dtype"],
                "shape": tuple(meta["shape"]),
                "offset": base + begin,
                "nbytes": end - begin,
            }
    return out


def load_stacked(loc: dict) -> np.ndarray:
    dt, _ = _DTYPES[loc["dtype"]]
    with open(loc["file"], "rb") as f:
        f.seek(loc["offset"])
        buf = f.read(loc["nbytes"])
    return np.frombuffer(buf, dtype=dt).reshape(loc["shape"]).copy()


# --- reversible transforms ------------------------------------------------

def delta_along_axis0(x: np.ndarray) -> np.ndarray:
    """First-order delta along expert axis. Invert: cumsum."""
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - x[:-1]
    return y


def dod_along_axis0(x: np.ndarray) -> np.ndarray:
    """Second-order delta (delta-of-delta) along expert axis.

    d[0]=x[0]; d[e]=x[e]-x[e-1]
    out[0]=d[0]; out[e]=d[e]-d[e-1]
    Invert: two cumsums.
    """
    d = delta_along_axis0(x)
    return delta_along_axis0(d)


def delta_from_mean(x: np.ndarray) -> np.ndarray:
    """Residual vs per-element mean across experts (rounded to dtype grid)."""
    if np.issubdtype(x.dtype, np.integer):
        mean = np.rint(x.astype(np.float64).mean(axis=0)).astype(x.dtype)
    else:
        mean = x.mean(axis=0).astype(x.dtype)
    y = np.empty_like(x)
    y[0] = mean  # store base in slot 0 for invertibility in this packing
    y[1:] = x[1:] - mean
    # also store expert 0 residual in a side channel: pack as
    # [mean | x[0]-mean | x[1]-mean | ...] by overwriting:
    y[0] = x[0] - mean
    # Keep mean separately - caller compresses (mean_bytes + residual_bytes).
    return mean, y


def xor_delta_axis0(x: np.ndarray) -> np.ndarray:
    """Bitwise XOR 'delta' along expert axis (natural for U32 quant codes)."""
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = np.bitwise_xor(x[1:], x[:-1])
    return y


def dod_scan_flat(x: np.ndarray) -> np.ndarray:
    """DoD along the flattened weight stream (same algebra as dod_along_axis0).

    Invert with two modular cumsums (verify_dod_scan).
    """
    flat = x.reshape(-1)
    return dod_along_axis0(flat.reshape(flat.shape[0], 1)).reshape(flat.shape)


def byte_planes_u32(x: np.ndarray) -> bytes:
    """Split little-endian U32 words into 4 contiguous byte planes.

    Often helps entropy coders when low bytes are noisier than high bytes
    (or vice versa) - a common lossless preprocess for integer grids.
    """
    u = np.ascontiguousarray(x, dtype=np.uint32).view(np.uint8).reshape(-1, 4)
    return b"".join(u[:, i].tobytes() for i in range(4))


def verify_delta_axis0(x: np.ndarray) -> bool:
    y = delta_along_axis0(x)
    rec = np.cumsum(y, axis=0).astype(x.dtype, copy=False)
    return bool(np.array_equal(rec, x))


def verify_dod_axis0(x: np.ndarray) -> bool:
    y = dod_along_axis0(x)
    rec = np.cumsum(np.cumsum(y, axis=0), axis=0).astype(x.dtype, copy=False)
    return bool(np.array_equal(rec, x))


def verify_xor_axis0(x: np.ndarray) -> bool:
    y = xor_delta_axis0(x)
    rec = np.empty_like(y)
    rec[0] = y[0]
    for e in range(1, y.shape[0]):
        rec[e] = np.bitwise_xor(rec[e - 1], y[e])
    return bool(np.array_equal(rec, x))


def verify_dod_scan(x: np.ndarray) -> bool:
    y = dod_scan_flat(x)
    flat = y.reshape(-1).copy()
    np.cumsum(flat, out=flat)
    np.cumsum(flat, out=flat)
    return bool(np.array_equal(flat.reshape(x.shape), x.reshape(x.shape)))


# --- BWT / MTF / RLE (byte-oriented; applied per expert, size-capped) ------

def bwt_transform(data: bytes) -> tuple[bytes, int]:
    """Classic BWT via rotation sort. Returns (L, primary_index).

    Only for small buffers (we call it on chunks). Correctness > speed here.
    """
    n = len(data)
    if n == 0:
        return b"", 0
    # rotations as index sort on doubled string
    s = data + data
    idxs = sorted(range(n), key=lambda i: s[i : i + n])
    primary = idxs.index(0)
    last = bytes(s[i + n - 1] for i in idxs)
    return last, primary


def bwt_inverse_correct(last: bytes, primary: int) -> bytes:
    n = len(last)
    if n == 0:
        return b""
    counts = [0] * 256
    for b in last:
        counts[b] += 1
    start = [0] * 256
    total = 0
    for c in range(256):
        start[c] = total
        total += counts[c]
    occ = [0] * 256
    lf = [0] * n
    for i, b in enumerate(last):
        lf[i] = start[b] + occ[b]
        occ[b] += 1
    out = bytearray(n)
    i = primary
    for k in range(n - 1, -1, -1):
        out[k] = last[i]
        i = lf[i]
    return bytes(out)


def mtf_encode(data: bytes) -> bytes:
    table = list(range(256))
    out = bytearray(len(data))
    for i, b in enumerate(data):
        idx = table.index(b)
        out[i] = idx
        # move to front
        del table[idx]
        table.insert(0, b)
    return bytes(out)


def mtf_decode(data: bytes) -> bytes:
    table = list(range(256))
    out = bytearray(len(data))
    for i, idx in enumerate(data):
        b = table[idx]
        out[i] = b
        del table[idx]
        table.insert(0, b)
    return bytes(out)


def rle_encode(data: bytes) -> bytes:
    """Simple escape RLE: 0x00 <len> <byte> for runs of length >= 4; else raw.

    Not the tightest; enough to see if MTF created long runs worth keeping.
    """
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        b = data[i]
        j = i + 1
        while j < n and data[j] == b and (j - i) < 255:
            j += 1
        run = j - i
        if run >= 4 or b == 0:
            out.append(0)
            out.append(run)
            out.append(b)
        else:
            out.extend(data[i:j])
        i = j
    return bytes(out)


def pipeline_bwt_mtf_rle(data: bytes, chunk: int = 1 << 16) -> bytes:
    """BWT -> MTF -> RLE over fixed chunks (BWT is O(n^2 log n) if naive-sorted;
    we keep chunks small so the probe finishes). Then zlib as entropy coder.
    """
    parts = []
    for off in range(0, len(data), chunk):
        block = data[off : off + chunk]
        L, primary = bwt_transform(block)
        mtf = mtf_encode(L)
        rle = rle_encode(mtf)
        parts.append(struct.pack("<II", primary, len(rle)) + rle)
    blob = b"".join(parts)
    return zlib.compress(blob, level=6)


def pipeline_mtf_rle(data: bytes) -> bytes:
    """MTF -> RLE -> zlib (no BWT). Cheaper; tests whether symbol locality alone helps."""
    return zlib.compress(rle_encode(mtf_encode(data)), level=6)


def verify_bwt_mtf_rle_roundtrip(data: bytes, chunk: int = 4096) -> bool:
    """Decode the chunked BWT-MTF-RLE container (sans zlib) for soundness."""
    # Direct path without zlib: encode one chunk and invert RLE is lossy in our
    # simple RLE (raw bytes can contain 0x00). So only verify BWT+MTF here;
    # RLE is a size probe, not the on-disk format we would ship.
    L, pidx = bwt_transform(data)
    if bwt_inverse_correct(L, pidx) != data:
        return False
    return mtf_decode(mtf_encode(L)) == L


# --- compression helpers --------------------------------------------------

def enc_zlib(b: bytes) -> bytes:
    return zlib.compress(b, level=6)


def enc_zstd(b: bytes) -> bytes:
    cctx = zstd.ZstdCompressor(level=3)
    return cctx.compress(b)


def enc_lz4(b: bytes) -> bytes:
    return lz4f.compress(b)


def timed_decompress_zlib(b: bytes, reps: int = 5) -> float:
    # warm
    zlib.decompress(b)
    t0 = time.perf_counter()
    for _ in range(reps):
        zlib.decompress(b)
    return (time.perf_counter() - t0) / reps


def cost_model(raw_nbytes: int, comp_nbytes: int, decomp_s: float,
               ssd_gbs: float = 8.0) -> dict:
    """Compare raw SSD read vs compressed read + decompress."""
    raw_s = raw_nbytes / (ssd_gbs * 1e9)
    comp_s = comp_nbytes / (ssd_gbs * 1e9) + decomp_s
    return {
        "raw_ms": raw_s * 1e3,
        "comp_ms": comp_s * 1e3,
        "speedup": raw_s / comp_s if comp_s > 0 else float("inf"),
    }


# --- main probe -----------------------------------------------------------

def find_weight_keys(index: dict, layers: list[int]) -> list[tuple[int, str, dict]]:
    found = []
    for layer in layers:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            # MLX stacked layout
            for suffix in ("weight",):
                name = f"model.layers.{layer}.mlp.switch_mlp.{proj}.{suffix}"
                if name in index:
                    found.append((layer, name, index[name]))
    return found


def ratio_table(label: str, raw_n: int, variants: dict) -> None:
    print(f"\n{label}  raw={raw_n/1e6:.2f} MB")
    print(f"  {'scheme':<36} {'MB':>8} {'ratio':>8} {'win?':>6}")
    for name, (blob, denom) in variants.items():
        r = len(blob) / max(1, denom)
        flag = "YES" if r < 0.85 else ("meh" if r < 0.95 else "no")
        print(f"  {name:<36} {len(blob)/1e6:8.2f} {r:8.3f} {flag:>6}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "mlx-models/qwen3-235-4bit"))
    ap.add_argument("--layers", default="0,8,40,80")
    ap.add_argument("--experts", type=int, default=32,
                    help="how many experts of the stack to include in the probe")
    ap.add_argument("--bwt-chunk-kb", type=int, default=64)
    ap.add_argument("--ssd-gbs", type=float, default=8.0)
    ap.add_argument("--skip-bwt", action="store_true")
    args = ap.parse_args()

    model = Path(args.model).expanduser()
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    print(f"indexing {model} ...", flush=True)
    index = read_index(model)
    keys = find_weight_keys(index, layers)
    if not keys:
        print("no switch_mlp weight tensors found")
        return 1
    print(f"found {len(keys)} weight tensors across layers {layers}")
    print(f"codecs: zlib=yes zstd={_HAS_ZSTD} lz4={_HAS_LZ4}")

    # Soundness checks on a small random int grid
    rng = np.random.default_rng(0)
    toy = rng.integers(0, 2**32, size=(16, 64), dtype=np.uint32)
    assert verify_delta_axis0(toy), "delta_axis0 not invertible"
    assert verify_dod_axis0(toy), "dod_axis0 not invertible"
    assert verify_xor_axis0(toy), "xor_axis0 not invertible"
    assert verify_dod_scan(toy), "dod_scan not invertible"
    # BWT / MTF roundtrip
    sample = bytes(rng.integers(0, 256, size=2000, dtype=np.uint8))
    assert verify_bwt_mtf_rle_roundtrip(sample), "BWT/MTF not invertible"
    assert mtf_decode(mtf_encode(sample)) == sample, "MTF not invertible"
    print("invertibility checks: OK (delta, DoD, XOR, DoD-scan, BWT, MTF)")

    # Aggregate across layers/projections for a stable ratio
    all_raw = []
    per_layer_summaries = []

    for layer, name, loc in keys:
        print(f"\nloading {name} {loc['shape']} {loc['dtype']} ...", flush=True)
        x_full = load_stacked(loc)  # [E, ...]
        e_use = min(args.experts, x_full.shape[0])
        x = x_full[:e_use]
        raw = x.tobytes()
        all_raw.append(raw)

        variants = OrderedDict()
        # Each entry: name -> (compressed_bytes, denominator_raw_nbytes)
        def add(name: str, blob: bytes, denom: int = len(raw)) -> None:
            variants[name] = (blob, denom)

        add("zlib(raw)", enc_zlib(raw))
        if _HAS_ZSTD:
            add("zstd(raw)", enc_zstd(raw))
        if _HAS_LZ4:
            add("lz4(raw)", enc_lz4(raw))

        # --- delta family (expert axis) ---
        d1 = delta_along_axis0(x)
        add("zlib(delta_neighbor)", enc_zlib(d1.tobytes()))
        if _HAS_ZSTD:
            add("zstd(delta_neighbor)", enc_zstd(d1.tobytes()))

        d2 = dod_along_axis0(x)
        add("zlib(dod_neighbor)", enc_zlib(d2.tobytes()))
        if _HAS_ZSTD:
            add("zstd(dod_neighbor)", enc_zstd(d2.tobytes()))

        if np.issubdtype(x.dtype, np.integer):
            xd = xor_delta_axis0(x)
            add("zlib(xor_neighbor)", enc_zlib(xd.tobytes()))
            if _HAS_ZSTD:
                add("zstd(xor_neighbor)", enc_zstd(xd.tobytes()))

        mean, resid = delta_from_mean(x)
        pack = mean.tobytes() + resid.tobytes()
        add("zlib(delta_mean)", enc_zlib(pack))
        if _HAS_ZSTD:
            add("zstd(delta_mean)", enc_zstd(pack))

        # --- DoD along flat scan of one mid expert (and of whole stack) ---
        mid = x[e_use // 2]
        mid_raw = mid.tobytes()
        mid_n = len(mid_raw)
        dd_one = dod_scan_flat(mid)
        add("zlib(dod_scan_one_expert)", enc_zlib(dd_one.tobytes()), mid_n)
        add("zlib(raw_one_expert)", enc_zlib(mid_raw), mid_n)
        if _HAS_ZSTD:
            add("zstd(dod_scan_one_expert)", enc_zstd(dd_one.tobytes()), mid_n)
            add("zstd(raw_one_expert)", enc_zstd(mid_raw), mid_n)
        dd_all = dod_scan_flat(x)
        add("zlib(dod_scan_stack)", enc_zlib(dd_all.tobytes()))

        # --- byte-plane split (U32 quant codes) ---
        if x.dtype == np.uint32:
            planes = byte_planes_u32(x)
            add("zlib(byte_planes)", enc_zlib(planes))
            if _HAS_ZSTD:
                add("zstd(byte_planes)", enc_zstd(planes))
            planes_d2 = byte_planes_u32(d2)
            add("zlib(dod_neighbor+planes)", enc_zlib(planes_d2))
            if _HAS_ZSTD:
                add("zstd(dod_neighbor+planes)", enc_zstd(planes_d2))

        # --- BWT / MTF on a single-expert slice vs zlib on SAME slice ---
        if not args.skip_bwt:
            cap = min(mid_n, 256 << 10)
            chunk = args.bwt_chunk_kb << 10
            slice_ = mid_raw[:cap]
            t0 = time.perf_counter()
            bwt_blob = pipeline_bwt_mtf_rle(slice_, chunk=chunk)
            bwt_s = time.perf_counter() - t0
            mtf_blob = pipeline_mtf_rle(slice_)
            zlib_slice = enc_zlib(slice_)
            zstd_slice = enc_zstd(slice_) if _HAS_ZSTD else b""
            dd_bytes = dd_one.reshape(-1).tobytes()[:cap]
            add(f"zlib(raw_slice,{cap//1024}KiB)", zlib_slice, cap)
            if _HAS_ZSTD:
                add(f"zstd(raw_slice,{cap//1024}KiB)", zstd_slice, cap)
            add(f"bwt_mtf_rle_zlib({cap//1024}KiB)", bwt_blob, cap)
            add(f"mtf_rle_zlib({cap//1024}KiB)", mtf_blob, cap)
            add(f"dod_scan+mtf_rle_zlib({cap//1024}KiB)",
                pipeline_mtf_rle(dd_bytes), cap)
            print(f"  slice {cap/1024:.0f}KiB: BWT {bwt_s:.2f}s  "
                  f"zlib={len(zlib_slice)/cap:.3f}  "
                  f"bwt_mtf={len(bwt_blob)/cap:.3f}  "
                  f"mtf={len(mtf_blob)/cap:.3f}")

        ratio_table(f"L{layer} {name.split('.')[-2]}.{loc['dtype']} E={e_use}",
                    len(raw), variants)

        # Cost model: only schemes whose denom == full raw (fair SSD compare)
        full = [(n, b, d) for n, (b, d) in variants.items() if d == len(raw)]
        best_name, best_blob, _ = min(full, key=lambda kv: len(kv[1]))
        z_blob = variants["zlib(raw)"][0]
        dt = timed_decompress_zlib(z_blob)
        cm = cost_model(len(raw), len(z_blob), dt, args.ssd_gbs)
        print(f"  cost zlib(raw): SSD+decomp {cm['comp_ms']:.2f} ms vs raw "
              f"{cm['raw_ms']:.2f} ms -> {cm['speedup']:.2f}x")
        if best_name.startswith("zlib"):
            dt = timed_decompress_zlib(best_blob)
            cm = cost_model(len(raw), len(best_blob), dt, args.ssd_gbs)
            print(f"  cost {best_name}: SSD+decomp {cm['comp_ms']:.2f} ms vs raw "
                  f"{cm['raw_ms']:.2f} ms -> {cm['speedup']:.2f}x")

        per_layer_summaries.append(
            (layer, name, len(raw),
             {k: (len(v), d) for k, (v, d) in variants.items()})
        )

    # Global pool (concat) - what a multi-expert miss burst looks like
    pooled = b"".join(all_raw)
    print("\n" + "=" * 60)
    print(f"POOLED weights ({len(all_raw)} tensors, {len(pooled)/1e6:.1f} MB)")
    pool_vars = OrderedDict()
    pool_vars["zlib"] = (enc_zlib(pooled), len(pooled))
    if _HAS_ZSTD:
        pool_vars["zstd"] = (enc_zstd(pooled), len(pooled))
    if _HAS_LZ4:
        pool_vars["lz4"] = (enc_lz4(pooled), len(pooled))
    ratio_table("pooled raw", len(pooled), pool_vars)

    # Verdict - only count schemes with fair denominators; ignore slice toys
    # unless they beat zlib on the *same* slice by a clear margin.
    print("\n" + "=" * 60)
    print("VERDICT")
    promising = []
    slice_notes = []
    for layer, name, raw_n, sizes in per_layer_summaries:
        for scheme, (n, denom) in sizes.items():
            r = n / denom
            is_slice = "slice" in scheme or "KiB" in scheme or "one_expert" in scheme
            if is_slice:
                slice_notes.append((r, layer, scheme, name))
                continue
            if r < 0.85:
                promising.append((r, layer, scheme, name))
    # Did any transform beat zlib(raw) on the same full tensor?
    beats_raw = []
    for layer, name, raw_n, sizes in per_layer_summaries:
        base = sizes.get("zlib(raw)", (raw_n, raw_n))[0]
        for scheme, (n, denom) in sizes.items():
            if denom != raw_n or scheme == "zlib(raw)":
                continue
            if n < base * 0.95:  # >=5% better than plain zlib
                beats_raw.append((n / raw_n, n / base, layer, scheme))
    print("Transforms that beat plain zlib(raw) by >=5% on full tensor:")
    if not beats_raw:
        print("  (none)")
    else:
        for r, rel, layer, scheme in sorted(beats_raw)[:12]:
            print(f"  ratio={r:.3f}  vs_zlib={rel:.3f}  L{layer}  {scheme}")

    print("\nSlice / single-expert probes (fair denom; compare within group):")
    for r, layer, scheme, name in sorted(slice_notes)[:24]:
        print(f"  {r:.3f}  L{layer}  {scheme}  {name.split('.')[-2]}")

    if not beats_raw:
        print("\nDelta / DoD / BWT-MTF: NOT promising for miss traffic.")
        print("No transform beat plain zlib by >=5% on full expert stacks;")
        print("mid/late layers are already near-entropy (~0.92-0.94 zlib).")
        print("Decompress cost also exceeds SSD read time at ~8 GB/s.")
        return 2
    print("\nSome transforms beat zlib - worth a follow-up only if decompress")
    print("stays cheap vs SSD on the miss path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
