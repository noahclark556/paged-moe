# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""User-facing PagedMoE config (``~/paged-moe-config.yaml``).

Sits next to ``~/mlx-config.yaml``. Data (sidecar weights, etc.) stays under
``~/.paged_moe/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - pyyaml comes with mlx-lm
    yaml = None


def default_root() -> Path:
    """Data directory (sidecar, caches). Not where the user config lives."""
    override = os.environ.get("PAGED_MOE_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".paged_moe"


def preferred_config_path() -> Path:
    """Where new installs write the user config (never the legacy nested path)."""
    override = os.environ.get("PAGED_MOE_CONFIG", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / "paged-moe-config.yaml"


def config_path() -> Path:
    """Resolve the active user config path.

    Prefer ``~/paged-moe-config.yaml`` (or ``PAGED_MOE_CONFIG``). If that file
    is missing, fall back to legacy ``~/.paged_moe/config.yaml`` when present
    so existing installs keep working.
    """
    preferred = preferred_config_path()
    if preferred.is_file():
        return preferred
    if not os.environ.get("PAGED_MOE_CONFIG", "").strip():
        legacy = default_root() / "config.yaml"
        if legacy.is_file():
            return legacy
    return preferred


def load_user_config() -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        return {}
    if yaml is None:
        return {}
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _models_dir(models_dir: str | Path | None = None) -> Path:
    if models_dir is not None and str(models_dir).strip():
        return Path(models_dir).expanduser()
    override = os.environ.get("PAGED_MOE_MODELS_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / "mlx-models"


def sample_config_text(models_dir: str | Path | None = None) -> str:
    """Shipped defaults for the tested MoE checkpoints.

    Paths are absolute under the chosen models directory (default
    ``~/mlx-models/``). Engine knobs match the measured profiles from live use.
    """
    root = _models_dir(models_dir)
    next_path = root / "qwen3-next"
    b235_path = root / "qwen3-235-4bit"
    glm47_path = root / "glm-47-4bit"
    c480_path = root / "qwen3-coder-480-4bit"
    ds32_path = root / "deepseek-v32-4bit"
    kimi_path = root / "Kimi-K2-Instruct-4bit"
    return f"""\
# =============================================================================
# PagedMoE config  (~/paged-moe-config.yaml)
# =============================================================================
# Tell PagedMoE which MoE checkpoints to stream. Sidecar data lives under
# ~/.paged_moe/ automatically - you usually don't set those paths.
# Edit model paths anytime; the installer only seeds this file when missing
# or when models: is still empty.
#
# You can also opt a checkpoint in by dropping a .paged_moe (or paged_moe.yaml)
# file in its directory, or by setting PAGED_MOE=1 for every load.
#
# -----------------------------------------------------------------------------
# Machine (optional)
# -----------------------------------------------------------------------------
# Auto-detect is the default. Capacity knobs (expert cache ceiling, reader
# threads, headroom) scale from the measured 48 GB M5 Pro baseline. Fidelity
# knobs (prune / wait / sidecar) stay per-model below.
#
# machine:
#   auto: true
#   # Optional overrides (any unset field is detected):
#   # ram_gb: 16
#   # perf_cores: 8
#   # disk_gb_s: 6.0
#   # storage_gb: 512
#   # profile: mba-m2-2023-16gb   # named class; see `paged-moe machine catalog`
#
# paged-moe machine          # show what this Mac resolved to
# paged-moe machine probe    # measure SSD + GPU cores
# paged-moe ladder           # model fit table for this Mac
#
# -----------------------------------------------------------------------------
# Top-level switches
# -----------------------------------------------------------------------------
# enable_all: true  -> stream every load (tiny models that fit in RAM still stay
#                     fully resident). false = only paths listed below.
# debug: true       -> print [paged-moe] stream / passthrough lines to stderr.
#
# defaults:         -> optional settings applied to every matched model first;
#                     per-model env: below wins on conflicts.
#
# models:           -> list of {{ path, env? }}. path is the checkpoint folder.
#
# -----------------------------------------------------------------------------
# Common settings (all optional - omit to keep the defaults)
# -----------------------------------------------------------------------------
# EXPERT_STREAM_PRUNE          0-1. Drop weak experts when they'd need a disk
#                              read. Higher = fewer reads / faster decode, a
#                              little less fidelity. 0.7 suits most big MoEs;
#                              GLM-4.7 prefers 0.8.
#
# EXPERT_STREAM_WAIT_ABOVE     0-1. Only wait on a disk miss if that expert is
#                              at least this important in the mixture. 0.2 is
#                              a solid default.
#
# EXPERT_STREAM_KV_FP16_CTX    Keep the KV cache in fp16 below this context
#                              length, otherwise 8-bit. "0" = always 8-bit
#                              (saves memory on models with a heavy KV).
#
# EXPERT_STREAM_SIDECAR        "1" turns on the sidecar; "0" leaves it off.
#
#                              The sidecar is a small helper (not the big MoE)
#                              that watches which experts the router picks as
#                              you generate, and learns to prefetch the next
#                              ones from disk before the GPU stalls. It
#                              improves with normal use - day one is already
#                              fine; after a few minutes hit-rate and speed
#                              usually climb. Weights are saved under
#                              ~/.paged_moe/sidecar/. For an optional offline
#                              warm-start, see paged-moe-pretrain.
#                              Status lines stay off unless you set
#                              EXPERT_STREAM_SIDECAR_DEBUG=1 (or PAGED_MOE_DEBUG=1).
#
# EXPERT_STREAM_SIDECAR_MIN_PRECISION
#                              How confident the sidecar must be before it acts
#                              (~0.22 is balanced).
#
# EXPERT_STREAM_SIDECAR_LOCK_HITS
#                              Freeze sidecar weights after this many good hits.
#                              "0" = keep learning (recommended).
#
# EXPERT_STREAM_NATIVE_READ     "1" prefers the C expert-read pool (one job per
#                              expert, GIL released for the latch). Same bytes
#                              as the Python pool; falls back automatically if
#                              the extension cannot load or compile. "0" forces
#                              Python. Leave on for large streamed MoEs.
#
# EXPERT_STREAM_EARLY_DECODE    "1" prefetches experts that show up in the first
#                              decode tokens after prefill (quality-safe; wrong
#                              guesses only waste a read). On for large MoEs.
# EXPERT_STREAM_EARLY_DECODE_TRAIN
#                              Listed "0" on purpose. Cold start auto-trains
#                              until a high cover/prec bar, then freezes train
#                              and keeps prefetch-only. Set "1" only if you want
#                              continuous online learning. Pool also holds
#                              actuation when cover/prec fall off; counts decay
#                              each train episode so stale domains fade.
#
#
#
# Only stream models listed under models: (recommended).
enable_all: false

# Set true to see stream / passthrough decisions.
debug: false

# Auto-detect this Mac and size the expert cache from unified memory.
# Uncomment to pin a named class or override a field.
machine:
  auto: true
  # ram_gb: 16
  # perf_cores: 8
  # disk_gb_s: 6.0
  # profile: mba-m2-2023-16gb

# Optional shared defaults for every matched model:
# defaults:
#   EXPERT_STREAM_SIDECAR: "1"

models:

  # --- Qwen3-Coder-Next (non-thinking) ---------------------------------------
  # Already fast day one; sidecar helps further with use.
  - path: {next_path}
    env:
      EXPERT_STREAM_SIDECAR: "1"
      EXPERT_STREAM_SIDECAR_LOCK_HITS: "0"
      EXPERT_STREAM_SIDECAR_HEAD_WRAP: "1"
      EXPERT_STREAM_SIDECAR_HEAD_PREFILL: "0"
      EXPERT_STREAM_SIDECAR_HEAD_RESIDENCY: "1"
      EXPERT_STREAM_SIDECAR_HEAD_PRUNE: "1"
      # C pread pool when available (same bytes; silent Python fallback).
      EXPERT_STREAM_NATIVE_READ: "1"
      # Prefetch after prefill. TRAIN off = cold-bootstrap then freeze.
      EXPERT_STREAM_EARLY_DECODE: "1"
      EXPERT_STREAM_EARLY_DECODE_TRAIN: "0"

  # --- Qwen3-235B-A22B 4-bit -------------------------------------------------
  # Large MoE - prune + wait + sidecar keep decode practical on a laptop.
  - path: {b235_path}
    env:
      EXPERT_STREAM_PRUNE: "0.7"
      EXPERT_STREAM_WAIT_ABOVE: "0.2"
      EXPERT_STREAM_SIDECAR: "1"
      EXPERT_STREAM_SIDECAR_MIN_PRECISION: "0.22"
      EXPERT_STREAM_SIDECAR_LOCK_HITS: "0"
      EXPERT_STREAM_SIDECAR_HEAD_WRAP: "1"
      EXPERT_STREAM_SIDECAR_HEAD_PREFILL: "0"
      EXPERT_STREAM_SIDECAR_HEAD_RESIDENCY: "1"
      EXPERT_STREAM_SIDECAR_HEAD_PRUNE: "1"
      EXPERT_STREAM_NATIVE_READ: "1"
      EXPERT_STREAM_EARLY_DECODE: "1"
      EXPERT_STREAM_EARLY_DECODE_TRAIN: "0"

  # --- GLM-4.7 4-bit ---------------------------------------------------------
  # Heavier KV - use a harder prune and always-8-bit KV so the cache still fits.
  - path: {glm47_path}
    env:
      EXPERT_STREAM_PRUNE: "0.8"
      EXPERT_STREAM_WAIT_ABOVE: "0.2"
      EXPERT_STREAM_KV_FP16_CTX: "0"
      EXPERT_STREAM_SIDECAR: "1"
      EXPERT_STREAM_SIDECAR_MIN_PRECISION: "0.22"
      EXPERT_STREAM_SIDECAR_LOCK_HITS: "0"
      EXPERT_STREAM_SIDECAR_HEAD_WRAP: "1"
      EXPERT_STREAM_SIDECAR_HEAD_PREFILL: "0"
      EXPERT_STREAM_SIDECAR_HEAD_RESIDENCY: "1"
      EXPERT_STREAM_SIDECAR_HEAD_PRUNE: "1"
      EXPERT_STREAM_NATIVE_READ: "1"
      EXPERT_STREAM_EARLY_DECODE: "1"
      EXPERT_STREAM_EARLY_DECODE_TRAIN: "0"

  # --- Qwen3-Coder-480B 4-bit ------------------------------------------------
  # Very large checkpoint - same tuning as 235B; streaming is what makes it fit.
  - path: {c480_path}
    env:
      EXPERT_STREAM_PRUNE: "0.7"
      EXPERT_STREAM_WAIT_ABOVE: "0.2"
      EXPERT_STREAM_SIDECAR: "1"
      EXPERT_STREAM_SIDECAR_MIN_PRECISION: "0.22"
      EXPERT_STREAM_SIDECAR_LOCK_HITS: "0"
      EXPERT_STREAM_SIDECAR_HEAD_WRAP: "1"
      EXPERT_STREAM_SIDECAR_HEAD_PREFILL: "0"
      EXPERT_STREAM_SIDECAR_HEAD_RESIDENCY: "1"
      EXPERT_STREAM_SIDECAR_HEAD_PRUNE: "1"
      EXPERT_STREAM_NATIVE_READ: "1"
      EXPERT_STREAM_EARLY_DECODE: "1"
      EXPERT_STREAM_EARLY_DECODE_TRAIN: "0"

  # --- DeepSeek-V3.2 4-bit ---------------------------------------------------
  # ~378 GB MoE. Must be listed (or PAGED_MOE=1) - passthrough OOMs hard.
  # DSA prefill: fused + adaptive so pe_scores fit Metal while MoE still runs
  # once per step. Prune/wait match the MoEGate recipe used for GLM-4.7.
  - path: {ds32_path}
    env:
      EXPERT_STREAM_PRUNE: "0.8"
      EXPERT_STREAM_WAIT_ABOVE: "0.2"
      EXPERT_STREAM_FUSED_PREFILL: "1"
      EXPERT_STREAM_ADAPTIVE_PREFILL: "1"
      EXPERT_STREAM_ADAPTIVE_PREFILL_MODE: "dsa"
      EXPERT_STREAM_PREFILL_CHUNK: "16384"
      EXPERT_STREAM_ADAPTIVE_PREFILL_MAX: "16384"
      EXPERT_STREAM_SIDECAR: "1"
      EXPERT_STREAM_SIDECAR_MIN_PRECISION: "0.22"
      EXPERT_STREAM_SIDECAR_LOCK_HITS: "0"
      EXPERT_STREAM_SIDECAR_HEAD_WRAP: "1"
      EXPERT_STREAM_SIDECAR_HEAD_PREFILL: "0"
      EXPERT_STREAM_SIDECAR_HEAD_RESIDENCY: "1"
      EXPERT_STREAM_SIDECAR_HEAD_PRUNE: "1"
      EXPERT_STREAM_NATIVE_READ: "1"
      EXPERT_STREAM_EARLY_DECODE: "1"
      EXPERT_STREAM_EARLY_DECODE_TRAIN: "0"

  # --- Kimi-K2-Instruct 4-bit ------------------------------------------------
  # ~430-580 GB. mlx-lm remaps kimi_k2 -> deepseek_v3 (plain MLA, not DSA).
  # Needs trust_remote_code (server default) + tiktoken. Keep prune/wait like
  # GLM / DeepSeek. Engine forces fp16 KV for this arch (8-bit MLA pe_scores
  # crash in mlx-lm). Disk-bound on 48 GB - keep the checkpoint internal.
  - path: {kimi_path}
    env:
      EXPERT_STREAM_PRUNE: "0.8"
      EXPERT_STREAM_WAIT_ABOVE: "0.2"
      EXPERT_STREAM_FUSED_PREFILL: "1"
      EXPERT_STREAM_ADAPTIVE_PREFILL: "1"
      EXPERT_STREAM_PREFILL_CHUNK: "16384"
      EXPERT_STREAM_ADAPTIVE_PREFILL_MAX: "16384"
      EXPERT_STREAM_SIDECAR: "1"
      EXPERT_STREAM_SIDECAR_MIN_PRECISION: "0.22"
      EXPERT_STREAM_SIDECAR_LOCK_HITS: "0"
      EXPERT_STREAM_SIDECAR_HEAD_WRAP: "1"
      EXPERT_STREAM_SIDECAR_HEAD_PREFILL: "0"
      EXPERT_STREAM_SIDECAR_HEAD_RESIDENCY: "1"
      EXPERT_STREAM_SIDECAR_HEAD_PRUNE: "1"
      EXPERT_STREAM_NATIVE_READ: "1"
      EXPERT_STREAM_EARLY_DECODE: "1"
      EXPERT_STREAM_EARLY_DECODE_TRAIN: "0"
"""


def ensure_sample_config(models_dir: str | Path | None = None) -> Path:
    root = default_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "sidecar").mkdir(parents=True, exist_ok=True)
    install_pretrained_sidecars()
    path = preferred_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = sample_config_text(models_dir)
    if not path.exists():
        path.write_text(text)
        return path
    # Refresh if the file still has an empty models list (fresh sample / old seed).
    if yaml is not None:
        try:
            raw = yaml.safe_load(path.read_text()) or {}
        except Exception:
            raw = {}
        if isinstance(raw, dict) and not (raw.get("models") or []):
            path.write_text(text)
    return path


def pretrained_sidecar_dir() -> Path:
    """Bundled warm-start weights shipped with the package (may be empty)."""
    return Path(__file__).resolve().parent / "pretrained_sidecar"


def install_pretrained_sidecars(*, overwrite: bool = False) -> list[str]:
    """Copy shipped sidecar ``.npz`` files into ``~/.paged_moe/sidecar/``.

    Default is seed-only: existing files (user's warmer weights) are left alone.
    Returns the list of basenames newly written.
    """
    src_root = pretrained_sidecar_dir()
    if not src_root.is_dir():
        return []
    files = sorted(
        p for p in src_root.iterdir() if p.is_file() and p.suffix == ".npz"
    )
    if not files:
        return []
    dest_root = default_root() / "sidecar"
    dest_root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for src in files:
        dest = dest_root / src.name
        if dest.exists() and not overwrite:
            continue
        try:
            dest.write_bytes(src.read_bytes())
            written.append(src.name)
        except OSError:
            continue
    return written
