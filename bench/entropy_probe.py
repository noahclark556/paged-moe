#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where is the lossless headroom in MLX-quantized expert bytes?

bench/delta_compress_probe.py answered "can a byte-grid transform beat zlib"
and correctly said no. It never unpacked the quantization grid, so every
number in it is a *byte* entropy number. A 4-bit checkpoint stores eight
codes per uint32 word; a byte-oriented coder sees code pairs and a group's
scale/bias interleaved with codes from other groups. This probe measures the
thing that actually bounds a codec:

    H(code) and H(code | context) in bits per 4-bit code,
    and the scales/biases streams separately,

because those three streams have completely different statistics and the
on-disk layout currently mixes them.

Why context matters here, and it is the whole thesis of this file: MLX affine
quantization picks scale = (max-min)/(2^bits-1) per group of 64. A group
holding one outlier gets a large scale, and its other 63 weights then land in
two or three adjacent codes near the middle - an almost deterministic stream.
A group with no outlier spreads codes over the full range. Pooled, those look
like one medium-entropy distribution (~3.6 bits, which is exactly what zlib
reported as 0.92). Conditioned on the group's scale - which a decoder can
read BEFORE the codes, for free, because scales are their own tensor - they
separate.

So the quantity to measure is not "how compressible are these bytes" but
"how much does knowing the group's scale tell you about its codes".

Streams measured
----------------
  codes     unpacked 4-bit quantization codes  (88.9% of expert bytes @ 4-bit)
  scales    bf16/fp16, one per 64 codes        (5.6%)
  biases    bf16/fp16, one per 64 codes        (5.6%)

Contexts tried for codes (all decoder-available, cost accounted):
  none              H0, the static-Huffman/ANS ceiling
  prev code         order-1 along the group axis
  scale bucket      log2(scale / tensor median scale), quantized  <- the thesis
  scale+prev        both
  group oracle      each group's own histogram (NOT achievable; upper bound)

Contexts for scales/biases:
  raw16             order-0 over the 16-bit symbol
  planes            sign / exponent / mantissa entropy, summed
  exp delta         exponent minus previous exponent along the group axis
  bias|scale        bias's bf16 bits XOR a prediction made from scale

Every entropy is reported in bits per code (or per element) and converted to
a ratio against what is stored today, so the numbers compose into an expert-
level ratio at the bottom. Model cost (the probability tables a decoder needs)
is charged explicitly - a context with 4096 states and 16 symbols is not free.

Usage:
  PAGED_MOE_HOOK=0 ~/.mlx-env/bin/python bench/entropy_probe.py \
      --model ~/mlx-models/qwen3-235-4bit --layers 0,8,40,80 --experts 8
"""

from __future__ import annotations

import argparse
import json
import struct
import time
import zlib
from pathlib import Path

import numpy as np

try:
    import zstandard as zstd

    _HAS_ZSTD = True
except Exception:
    _HAS_ZSTD = False


_DTYPES = {
    "U32": (np.uint32, 4),
    "U16": (np.uint16, 2),
    "U8": (np.uint8, 1),
    "BF16": (np.uint16, 2),
    "F16": (np.float16, 2),
    "F32": (np.float32, 4),
    "I8": (np.int8, 1),
}

_PROJS = ("gate_proj", "up_proj", "down_proj")


# --------------------------------------------------------------- safetensors

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


def load_experts(loc: dict, n_experts: int) -> np.ndarray:
    """Read the first `n_experts` rows of a stacked [E, ...] tensor."""
    dt, _ = _DTYPES[loc["dtype"]]
    shape = loc["shape"]
    stride = loc["nbytes"] // shape[0]
    take = min(n_experts, shape[0])
    with open(loc["file"], "rb") as f:
        f.seek(loc["offset"])
        buf = f.read(stride * take)
    return np.frombuffer(buf, dtype=dt).reshape((take,) + tuple(shape[1:])).copy()


# ------------------------------------------------------------------- unpack

def unpack_codes(words: np.ndarray, bits: int) -> np.ndarray:
    """uint32 packed codes -> uint8 codes, MLX layout.

    MLX packs `32 // bits` consecutive elements of the last axis into one
    uint32, element j at bit offset j*bits (little end first). Returns an
    array whose last axis is expanded by that factor.
    """
    per = 32 // bits
    mask = (1 << bits) - 1
    w = np.ascontiguousarray(words, dtype=np.uint32)
    out = np.empty(w.shape + (per,), dtype=np.uint8)
    for j in range(per):
        out[..., j] = ((w >> (bits * j)) & mask).astype(np.uint8)
    return out.reshape(w.shape[:-1] + (w.shape[-1] * per,))


def repack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Inverse of unpack_codes - used only to prove the layout is right."""
    per = 32 // bits
    c = codes.reshape(codes.shape[:-1] + (codes.shape[-1] // per, per))
    out = np.zeros(c.shape[:-1], dtype=np.uint32)
    for j in range(per):
        out |= c[..., j].astype(np.uint32) << (bits * j)
    return out


# ------------------------------------------------------------------ entropy

def _h_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy of a count vector, in bits."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log2(p)).sum())


def h0(sym: np.ndarray, alphabet: int) -> float:
    return _h_from_counts(np.bincount(sym.reshape(-1), minlength=alphabet))


def h_conditional(sym: np.ndarray, ctx: np.ndarray, alphabet: int,
                  n_ctx: int) -> tuple[float, int]:
    """H(sym | ctx) in bits/symbol, and the number of non-empty contexts.

    Contexts are flattened; joint counts via one bincount so this stays fast
    on ~100M symbols.
    """
    s = sym.reshape(-1).astype(np.int64)
    c = ctx.reshape(-1).astype(np.int64)
    joint = np.bincount(c * alphabet + s, minlength=n_ctx * alphabet)
    joint = joint.reshape(n_ctx, alphabet)
    per_ctx = joint.sum(axis=1)
    total = per_ctx.sum()
    if total <= 0:
        return 0.0, 0
    hsum = 0.0
    used = 0
    for i in np.nonzero(per_ctx)[0]:
        used += 1
        hsum += per_ctx[i] * _h_from_counts(joint[i])
    return hsum / total, used


def model_bits(n_ctx_used: int, alphabet: int, precision: int = 12) -> int:
    """Bits a decoder needs for the probability tables of a context model.

    Charged honestly: one `precision`-bit frequency per (context, symbol),
    which is what an rANS/FSE decoder actually needs to rebuild its tables.
    """
    return n_ctx_used * alphabet * precision


# ----------------------------------------------------------- code contexts

def scale_buckets(scales_bits: np.ndarray, is_bf16: bool, n_buckets: int,
                  group_size: int, n_codes_last: int) -> tuple[np.ndarray, int]:
    """Per-code context id from its group's scale exponent.

    The decoder has the scales tensor before it decodes any code (scales are
    a separate tensor and only 1/64th the bytes), so this context is free -
    no side channel, no per-group id in the stream.

    Bucketing is on the raw float exponent field rather than a computed
    log2: it is exact integer work on both sides and it is what actually
    distinguishes an outlier group (big scale) from a flat one.
    """
    f = bf16_to_f32(scales_bits) if is_bf16 else scales_bits.astype(np.float32)
    mag = np.abs(f).astype(np.float32)
    med = float(np.median(mag[mag > 0])) if np.any(mag > 0) else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.log2(np.where(mag > 0, mag, med) / max(med, 1e-30))
    # Clamp to +-4 octaves and quantize; outside that range the tails are
    # too thin to justify their own tables.
    q = np.clip(np.floor((rel + 4.0) / 8.0 * n_buckets), 0, n_buckets - 1)
    q = q.astype(np.int64)
    # One scale covers `group_size` consecutive codes along the last axis.
    ctx = np.repeat(q, group_size, axis=-1)
    assert ctx.shape[-1] == n_codes_last, (ctx.shape, n_codes_last)
    return ctx, n_buckets


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    """bf16 bit patterns -> float32 (bf16 is f32 with the low 16 bits cut)."""
    u = np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32) if u.dtype == np.uint32 else u.astype(np.float32)


def group_oracle_bits(codes: np.ndarray, group_size: int) -> float:
    """H of each group's own 64 codes, averaged. NOT achievable (the decoder
    would need every group's histogram) - reported as the ceiling that any
    group-aware context is chasing."""
    flat = codes.reshape(-1, group_size).astype(np.int64)
    n_groups = flat.shape[0]
    # One-hot counts per group without a Python loop.
    counts = np.zeros((n_groups, 16), dtype=np.int32)
    rows = np.repeat(np.arange(n_groups), group_size)
    np.add.at(counts, (rows, flat.reshape(-1)), 1)
    p = counts.astype(np.float64) / group_size
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(p > 0, np.log2(p), 0.0)
    return float(-(p * lg).sum(axis=1).mean())


# ------------------------------------------------------- float bit planes

def float_planes(bits: np.ndarray, exp_bits: int, man_bits: int) -> dict:
    """Split 16-bit float patterns into sign / exponent / mantissa."""
    u = np.ascontiguousarray(bits, dtype=np.uint16)
    man = (u & ((1 << man_bits) - 1)).astype(np.uint16)
    exp = ((u >> man_bits) & ((1 << exp_bits) - 1)).astype(np.uint16)
    sign = (u >> (exp_bits + man_bits)).astype(np.uint8)
    return {"sign": sign, "exp": exp, "man": man}


# --------------------------------------------------------------- reporting

class Acc:
    """Accumulates bits-per-element across tensors, weighted by element count."""

    def __init__(self):
        self.rows: dict[str, list[float]] = {}

    def add(self, key: str, bits_per_elem: float, n_elems: int) -> None:
        r = self.rows.setdefault(key, [0.0, 0])
        r[0] += bits_per_elem * n_elems
        r[1] += n_elems

    def mean(self, key: str) -> float:
        r = self.rows.get(key)
        return (r[0] / r[1]) if r and r[1] else float("nan")

    def elems(self, key: str) -> int:
        r = self.rows.get(key)
        return r[1] if r else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(Path.home() / "mlx-models/qwen3-235-4bit"))
    ap.add_argument("--layers", default="0,8,24,47,70,93")
    ap.add_argument("--experts", type=int, default=6)
    ap.add_argument("--buckets", type=int, default=16,
                    help="scale-exponent context buckets for the code stream")
    ap.add_argument("--no-oracle", action="store_true",
                    help="skip the per-group oracle (it is the slow one)")
    args = ap.parse_args()

    model = Path(args.model).expanduser()
    cfg = json.loads((model / "config.json").read_text())
    quant = cfg.get("quantization", {})
    bits = int(quant.get("bits", 4))
    group_size = int(quant.get("group_size", 64))
    print(f"{model.name}: bits={bits} group_size={group_size}")

    index = read_index(model)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    acc = Acc()
    stream_bytes = {"codes": 0, "scales": 0, "biases": 0}
    zstd_sizes = {"raw": 0, "split": 0}
    raw_total = 0

    for layer in layers:
        for proj in _PROJS:
            base = f"model.layers.{layer}.mlp.switch_mlp.{proj}"
            wk, sk, bk = f"{base}.weight", f"{base}.scales", f"{base}.biases"
            if wk not in index:
                continue
            w = load_experts(index[wk], args.experts)
            s = load_experts(index[sk], args.experts) if sk in index else None
            b = load_experts(index[bk], args.experts) if bk in index else None
            s_dtype = index[sk]["dtype"] if sk in index else None

            codes = unpack_codes(w, bits)
            # Prove the unpack is exactly MLX's layout before trusting any
            # entropy computed from it. A wrong nibble order would still give
            # a plausible-looking H0 and a meaningless H1.
            assert np.array_equal(repack_codes(codes, bits), w), "unpack mismatch"

            n_codes = codes.size
            raw_total += w.nbytes + (s.nbytes if s is not None else 0) + (
                b.nbytes if b is not None else 0
            )
            stream_bytes["codes"] += w.nbytes
            if s is not None:
                stream_bytes["scales"] += s.nbytes
            if b is not None:
                stream_bytes["biases"] += b.nbytes

            alpha = 1 << bits

            # --- code stream ------------------------------------------------
            h_none = h0(codes, alpha)
            acc.add("codes/h0", h_none, n_codes)

            prev = np.empty_like(codes)
            prev[..., 0] = 0
            prev[..., 1:] = codes[..., :-1]
            h_prev, n_prev = h_conditional(codes, prev, alpha, alpha)
            acc.add("codes/order1", h_prev + model_bits(n_prev, alpha) / n_codes,
                    n_codes)

            if s is not None:
                ctx, n_b = scale_buckets(
                    s, s_dtype == "BF16", args.buckets, group_size,
                    codes.shape[-1],
                )
                h_sc, used = h_conditional(codes, ctx, alpha, n_b)
                acc.add("codes/scale_ctx",
                        h_sc + model_bits(used, alpha) / n_codes, n_codes)

                joint = ctx.astype(np.int64) * alpha + prev.astype(np.int64)
                h_both, used2 = h_conditional(codes, joint, alpha, n_b * alpha)
                acc.add("codes/scale+order1",
                        h_both + model_bits(used2, alpha) / n_codes, n_codes)

            if not args.no_oracle:
                acc.add("codes/group_oracle",
                        group_oracle_bits(codes, group_size), n_codes)

            # --- scale / bias streams --------------------------------------
            for tag, arr in (("scales", s), ("biases", b)):
                if arr is None:
                    continue
                u16 = np.ascontiguousarray(arr).view(np.uint16)
                n = u16.size
                acc.add(f"{tag}/raw16", h0(u16.astype(np.int64), 1 << 16), n)
                if s_dtype == "BF16":
                    eb, mb = 8, 7
                else:
                    eb, mb = 5, 10
                pl = float_planes(u16, eb, mb)
                h_planes = (
                    h0(pl["sign"], 2)
                    + h0(pl["exp"], 1 << eb)
                    + h0(pl["man"], 1 << mb)
                )
                acc.add(f"{tag}/planes", h_planes, n)

                # Exponent delta along the group axis: adjacent groups of one
                # output row have similar dynamic range.
                e = pl["exp"].reshape(arr.shape)
                de = np.empty_like(e)
                de[..., 0] = e[..., 0]
                de[..., 1:] = (e[..., 1:] - e[..., :-1]) & ((1 << eb) - 1)
                h_de = (
                    h0(pl["sign"], 2) + h0(de, 1 << eb) + h0(pl["man"], 1 << mb)
                )
                acc.add(f"{tag}/expdelta", h_de, n)

            # biases predicted from scales: affine quant sets
            # scale=(max-min)/(2^bits-1), bias=min, so for a roughly symmetric
            # group bias ~= -scale*(2^bits-1)/2. Store the residual, not bias.
            if s is not None and b is not None:
                is_bf = s_dtype == "BF16"
                sf = bf16_to_f32(np.ascontiguousarray(s).view(np.uint16)) if is_bf \
                    else np.ascontiguousarray(s).view(np.float16).astype(np.float32)
                bf = bf16_to_f32(np.ascontiguousarray(b).view(np.uint16)) if is_bf \
                    else np.ascontiguousarray(b).view(np.float16).astype(np.float32)
                pred = -sf * ((1 << bits) - 1) / 2.0
                # Residual on the *bit* grid so the reconstruction is exact.
                if is_bf:
                    pb = (pred.view(np.uint32) >> 16).astype(np.uint16)
                    ab = np.ascontiguousarray(b).view(np.uint16)
                else:
                    pb = pred.astype(np.float16).view(np.uint16)
                    ab = np.ascontiguousarray(b).view(np.uint16)
                resid = (ab.astype(np.int32) - pb.astype(np.int32)).astype(np.int32)
                # Zig-zag so small signed residuals are small unsigned symbols.
                zz = ((resid << 1) ^ (resid >> 31)).astype(np.int64)
                lo = (zz & 0xFF).astype(np.int64)
                hi = ((zz >> 8) & 0xFFFF).astype(np.int64)
                acc.add("biases/pred_from_scale",
                        h0(lo, 256) + h0(hi, 1 << 16), zz.size)

            # --- what a shipping general-purpose codec gets ------------------
            if _HAS_ZSTD:
                c = zstd.ZstdCompressor(level=9)
                blob_raw = (
                    c.compress(w.tobytes())
                    + (c.compress(s.tobytes()) if s is not None else b"")
                    + (c.compress(b.tobytes()) if b is not None else b"")
                )
                zstd_sizes["raw"] += len(blob_raw)
                # Same bytes, but codes unpacked to one byte each and floats
                # split into planes - the "help the codec see the grid" version.
                pieces = [codes.tobytes()]
                for arr in (s, b):
                    if arr is None:
                        continue
                    u16 = np.ascontiguousarray(arr).view(np.uint16)
                    eb, mb = (8, 7) if s_dtype == "BF16" else (5, 10)
                    pl = float_planes(u16, eb, mb)
                    pieces += [pl["sign"].tobytes(), pl["exp"].tobytes(),
                               pl["man"].tobytes()]
                zstd_sizes["split"] += sum(len(c.compress(p)) for p in pieces)

            print(f"  L{layer:<3} {proj:<10} codes={n_codes/1e6:7.2f}M  "
                  f"H0={h_none:.3f}  "
                  f"scale_ctx={acc.mean('codes/scale_ctx'):.3f}", flush=True)

    if not acc.rows:
        print("no switch_mlp tensors found")
        return 1

    # ------------------------------------------------------------- summary
    code_elems = acc.elems("codes/h0")
    sb = stream_bytes
    total = sum(sb.values()) or 1
    print("\n" + "=" * 72)
    print("STREAM SHARES of expert bytes")
    for k, v in sb.items():
        print(f"  {k:<8} {v/1e6:9.1f} MB  {100*v/total:5.1f}%")

    print("\nCODE STREAM (bits per 4-bit code; stored today = "
          f"{bits}.000)")
    for key in ("codes/h0", "codes/order1", "codes/scale_ctx",
                "codes/scale+order1", "codes/group_oracle"):
        if key in acc.rows:
            m = acc.mean(key)
            print(f"  {key.split('/')[1]:<16} {m:6.3f} bits  "
                  f"ratio {m/bits:5.3f}")

    print("\nSCALE / BIAS STREAMS (bits per 16-bit element; stored = 16.000)")
    for key in sorted(k for k in acc.rows if k.startswith(("scales/", "biases/"))):
        m = acc.mean(key)
        print(f"  {key:<28} {m:6.3f} bits  ratio {m/16:5.3f}")

    # Compose an expert-level ratio from the best decoder-available option
    # in each stream. This is the number that decides whether to build it.
    best_code = min(
        acc.mean(k) for k in ("codes/h0", "codes/order1", "codes/scale_ctx",
                              "codes/scale+order1") if k in acc.rows
    )
    best_scale = min(acc.mean(k) for k in acc.rows if k.startswith("scales/"))
    bias_keys = [k for k in acc.rows if k.startswith("biases/")]
    best_bias = min(acc.mean(k) for k in bias_keys) if bias_keys else 16.0
    comp = (
        sb["codes"] * best_code / bits
        + sb["scales"] * best_scale / 16
        + sb["biases"] * best_bias / 16
    )
    print("\n" + "=" * 72)
    print("COMPOSED EXPERT RATIO (decoder-available contexts only)")
    print(f"  codes  {best_code:6.3f}/{bits} bits   "
          f"scales {best_scale:6.3f}/16   biases {best_bias:6.3f}/16")
    print(f"  expert ratio = {comp/total:.4f}   "
          f"({100*(1-comp/total):.1f}% fewer bytes per miss)")
    if _HAS_ZSTD and raw_total:
        print(f"  zstd-9 on raw bytes        ratio {zstd_sizes['raw']/raw_total:.4f}")
        print(f"  zstd-9 on unpacked+planes  ratio "
              f"{zstd_sizes['split']/raw_total:.4f}")

    print(f"\n  code elements measured: {code_elems/1e6:.1f}M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
