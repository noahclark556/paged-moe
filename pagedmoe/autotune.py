# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Read a checkpoint and turn a speed setting into engine knobs.

The engine already picks resident or streamed by size and sizes its cache
from this Mac. The one thing it can't pick for you is fidelity: how much of
the router's mixture to drop when an expert would need a disk read. That's
`--speed`:

  1 (or "exact")  nothing dropped, bit-identical output
  0.5             PRUNE 0.7, WAIT_ABOVE 0.25
  0               PRUNE 0.95, WAIT_ABOVE 0.5, one-pass decode
  auto            a per-model pick from total size, active size and bit width

Positions in between are spaced by how much of the mixture each setting keeps
on this model, once the engine has recorded that (prune_curve.py). Before that
they follow the fixed mapping above.

Nothing in here imports the engine, so it can run before pagedmoe.config
reads the environment.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path

# Whole checkpoint plus this much headroom has to fit in this share of RAM
# before it runs resident (no prune needed).
_RESIDENT_RAM_FRACTION = 0.82
_RESIDENT_HEADROOM_GB = 6.0
# Few active params means each dropped expert is a bigger share of the output.
_SMALL_ACTIVE_B = 12.0
# Same bars prune_curve.position() uses before a table counts.
_CURVE_MIN_ROWS = 4096
_CURVE_MIN_SPAN = 0.05
_CURVE_POINTS = 101

_ROUTED_RE = re.compile(r"\.(?:experts|switch_mlp|block_sparse_moe\.experts)\.")
_SHARED_RE = re.compile(r"shared_expert")
_LOOKUP_RE = re.compile(r"embed")
_DTYPE_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


@dataclass
class Checkpoint:
    path: str
    model_type: str
    total_b: float | None
    active_b: float | None
    moe: bool
    bits: int | None
    disk_gb: float
    ram_gb: float

    @property
    def fits_resident(self) -> bool:
        return self.disk_gb + _RESIDENT_HEADROOM_GB <= self.ram_gb * _RESIDENT_RAM_FRACTION

    def describe(self) -> str:
        parts = []
        if self.total_b:
            parts.append(f"{self.total_b:g}B")
        if self.moe:
            parts.append(f"MoE, {self.active_b:g}B active" if self.active_b else "MoE")
        if self.bits:
            parts.append(f"{self.bits}-bit")
        parts.append(f"{self.disk_gb:.0f} GB on disk")
        where = "fits in memory" if self.fits_resident else "streams"
        return f"{', '.join(parts)}, {where} on this {self.ram_gb:.0f} GB Mac"


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _ram_gb() -> float:
    try:
        fake = float(os.environ.get("PAGED_MOE_SIMULATE_RAM_GB") or 0)
    except ValueError:
        fake = 0.0
    if fake > 0:
        return fake
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1 << 30)
    except (ValueError, OSError):
        return 48.0


def _expert_counts(cfg: dict) -> tuple[int, int]:
    def first(*keys):
        for k in keys:
            try:
                v = int(cfg.get(k) or 0)
            except (TypeError, ValueError):
                continue
            if v > 0:
                return v
        return 0

    experts = first("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts", "num_routed_experts")
    top_k = first("num_experts_per_tok", "moe_topk", "num_experts_per_token", "moe_k", "top_k_experts")
    return experts, top_k


def _scan_params(model_dir: Path, quant: dict) -> tuple[float, float, float] | None:
    """(total, routed expert, embedding) params from the safetensors headers.

    Quantized weights are packed U32, so the bit width comes back per tensor
    from its scales. Mixed-precision checkpoints count right that way.
    """
    files = sorted(glob.glob(str(model_dir / "*.safetensors")))
    index = _read_json(model_dir / "model.safetensors.index.json").get("weight_map")
    if isinstance(index, dict) and index:
        named = set(index.values())
        files = [f for f in files if os.path.basename(f) in named]
    if not files:
        return None
    tensors: dict[str, tuple[str, list[int]]] = {}
    for path in files:
        try:
            with open(path, "rb") as f:
                (size,) = struct.unpack("<Q", f.read(8))
                if not 0 < size < 256 << 20:
                    return None
                header = json.loads(f.read(size))
        except (OSError, ValueError, struct.error):
            return None
        for name, meta in header.items():
            if name != "__metadata__":
                tensors[name] = (meta["dtype"], list(meta["shape"]))

    bits_default = int(quant.get("bits") or 4)
    group_default = int(quant.get("group_size") or 64)
    total = routed = lookup = 0.0
    for name, (dtype, shape) in tensors.items():
        if re.search(r"\.(scales|biases)$", name):
            owner = tensors.get(re.sub(r"\.(scales|biases)$", ".weight", name))
            if owner and owner[0] == "U32":
                continue
        elems = 1
        for d in shape:
            elems *= d
        if dtype == "U32":
            base = name.removesuffix(".weight")
            override = quant.get(base) if isinstance(quant.get(base), dict) else {}
            group = int(override.get("group_size") or group_default)
            scales = tensors.get(f"{base}.scales")
            packed = shape[-1] if shape else 1
            if scales and scales[1]:
                bits = packed * 32 / (scales[1][-1] * group)
            else:
                bits = int(override.get("bits") or bits_default)
            params = elems * 32 / (bits if bits > 0 else bits_default)
        elif dtype in _DTYPE_BYTES:
            params = float(elems)
        else:
            continue
        total += params
        if _ROUTED_RE.search(name) and not _SHARED_RE.search(name):
            routed += params
        elif _LOOKUP_RE.search(name):
            lookup += params
    return (total, routed, lookup) if total > 0 else None


def inspect(model_path: str) -> Checkpoint:
    """Sizes and shape of a local MLX checkpoint, from its config and headers."""
    model_dir = Path(os.path.expanduser(model_path))
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"no config.json in {model_dir}")
    config = _read_json(model_dir / "config.json")
    lang = dict(config)
    for key in ("text_config", "llm_config", "language_config"):
        if isinstance(config.get(key), dict):
            lang.update(config[key])
            break
    quant = config.get("quantization") or config.get("quantization_config") or {}
    quant = quant if isinstance(quant, dict) else {}
    bits = int(quant["bits"]) if str(quant.get("bits", "")).isdigit() else None

    disk = 0
    for f in glob.glob(str(model_dir / "*.safetensors")):
        try:
            disk += os.stat(f).st_size
        except OSError:
            pass

    experts, top_k = _expert_counts(lang)
    scan = _scan_params(model_dir, quant)
    moe = bool(scan and scan[1] > 0) or experts >= 4

    def round_b(n: float) -> float:
        return max(0.1, round(n / 1e8) / 10)

    total_b = active_b = None
    if scan:
        total, routed, lookup = scan
        total_b = round_b(total)
        if moe and experts and top_k and routed:
            active_b = round_b(total - lookup - routed + routed * top_k / experts)
        else:
            active_b = total_b
    elif disk:
        total_b = round(disk * 8 / ((bits + 0.5) if bits else 16) / 1e9)
        active_b = round(total_b * top_k / experts) if moe and experts and top_k else total_b

    return Checkpoint(
        path=str(model_dir),
        model_type=str(lang.get("model_type") or ""),
        total_b=total_b,
        active_b=active_b,
        moe=moe,
        bits=bits,
        disk_gb=disk / 1e9,
        ram_gb=_ram_gb(),
    )


def suggested_prune(ck: Checkpoint) -> float:
    """PRUNE for --speed auto. Bigger models take more pruning; low-bit and
    small-active ones less, since they have less fidelity to spare (Kimi K2 at
    2-bit and PRUNE 0.85 loops into garbage)."""
    if ck.fits_resident:
        return 0.0
    t = ck.total_b or 80
    prune = 0.85 if t >= 200 else 0.7 if t >= 70 else 0.55 if t >= 30 else 0.4
    if ck.active_b is not None and ck.active_b < _SMALL_ACTIVE_B:
        prune = min(prune, 0.7)
    if ck.bits is not None and ck.bits <= 2:
        return min(prune, 0.5)
    if ck.bits is not None and ck.bits <= 3:
        return min(prune, 0.7)
    return prune


def _round_half_up(v: float, places: int) -> float:
    k = 10**places
    return math.floor(v * k + 0.5) / k


def _fixed_profile(x: float) -> tuple[float, float, bool]:
    """(PRUNE, WAIT_ABOVE, one-pass) at position x on the fixed scale."""
    x = min(1.0, max(0.0, x))
    prune = 0.95 - 0.25 * (x / 0.5) if x <= 0.5 else 0.7 - 0.7 * ((x - 0.5) / 0.5)
    return _round_half_up(prune, 2), _round_half_up(0.5 * (1 - x), 2), x < 0.25


def _curve_path(model_path: str) -> Path:
    real = os.path.realpath(os.path.expanduser(model_path))
    key = hashlib.sha1(real.encode()).hexdigest()[:16]
    root = os.environ.get("PAGED_MOE_SIDECAR_DIR") or str(Path.home() / ".paged_moe" / "sidecar")
    return Path(root).expanduser() / "prune_curves" / f"{key}.json"


def _mass_at(table: dict, prune: float, wait: float) -> float:
    def cell(xs, v):
        if v <= xs[0]:
            return 0, 0, 0.0
        if v >= xs[-1]:
            return len(xs) - 1, len(xs) - 1, 0.0
        i = max(j for j in range(len(xs) - 1) if xs[j] <= v)
        return i, i + 1, (v - xs[i]) / (xs[i + 1] - xs[i])

    i0, i1, fi = cell(table["prune"], prune)
    j0, j1, fj = cell(table["wait_above"], wait)
    m = table["mass"]
    a = m[i0][j0] * (1 - fj) + m[i0][j1] * fj
    b = m[i1][j0] * (1 - fj) + m[i1][j1] * fj
    return a * (1 - fi) + b * fi


def speed_curve(model_path: str) -> list[float] | None:
    """Kept-mass position at each fixed position i/100, or None without a
    usable table. Rises strictly from 0 to 1."""
    table = _read_json(_curve_path(model_path))
    try:
        if table.get("version") != 1 or int(table.get("rows", 0)) < _CURVE_MIN_ROWS:
            return None
        if len(table["mass"]) != len(table["prune"]):
            return None
        kept: list[float] = []
        for i in range(_CURVE_POINTS):
            p, w, _ = _fixed_profile(i / (_CURVE_POINTS - 1))
            kept.append(max(_mass_at(table, p, w), kept[-1] if kept else float("-inf")))
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    lo, hi = kept[0], kept[-1]
    if hi - lo < _CURVE_MIN_SPAN:
        return None
    x = [(m - lo) / (hi - lo) for m in kept]
    for i in range(1, len(x)):
        x[i] = max(x[i], x[i - 1] + 1e-4)
    return [_round_half_up(v / x[-1], 4) for v in x]


def _invert(curve: list[float], v: float) -> float:
    v = min(1.0, max(0.0, v))
    i = 0
    while i < len(curve) - 2 and curve[i + 1] < v:
        i += 1
    span = curve[i + 1] - curve[i]
    f = (v - curve[i]) / span if span > 0 else 0.0
    return (i + min(1.0, max(0.0, f))) / (len(curve) - 1)


def parse_speed(raw: str) -> float | str:
    """1.0 for exact, a float in [0, 1], or "auto". Raises ValueError."""
    s = raw.strip().lower()
    if s in ("exact", "max", "1", "1.0"):
        return 1.0
    if s == "auto":
        return "auto"
    v = float(s)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"--speed takes 0 to 1, exact or auto, not {raw!r}")
    return v


def speed_env(ck: Checkpoint, speed: float | str) -> dict[str, str]:
    """PRUNE / WAIT_ABOVE / FLOW for a speed setting on this checkpoint."""
    if speed == "auto":
        prune = suggested_prune(ck)
        wait = 0.2 if prune > 0 else 0.0
        one_pass = False
    elif float(speed) >= 1.0:
        prune, wait, one_pass = 0.0, 0.0, False
    else:
        curve = speed_curve(ck.path)
        x = _invert(curve, float(speed)) if curve else float(speed)
        prune, wait, one_pass = _fixed_profile(x)
    return {
        "PAGED_MOE_PRUNE": f"{prune:g}",
        "PAGED_MOE_WAIT_ABOVE": f"{wait:g}",
        "PAGED_MOE_FLOW": "1" if one_pass else "0",
    }
