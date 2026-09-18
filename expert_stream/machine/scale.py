# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Affine envelope math from the measured 48 GB M5 Pro baseline.

Capacity knobs only. Do not put PRUNE / WAIT / sidecar / prefill-chunk here.

Formulas (ram_gb relative to 48):

    budget       = ram_gb * 0.68
    max_cache    = 26 * (ram_gb / 48)          # ceiling; load still subtracts backbone
    prefill_cache = clamp(4 * ram/48, 2, 8)
    headroom     = clamp(6 * ram/48, 2, 8)     # flat KV+scratch when RESERVE_CTX is 0
    activation   = clamp(3 * ram/48, 1.5, 4)
    staging_cap  = clamp(6 * ram/48, 1, 12)
    read_threads = clamp(perf_cores, 2, 32)
    kv_fp16_ctx  = 0 if ram<=18 else 4096 if ram<=32 else 8192

48 GB is a snap-to-exact so the measured machine does not drift.
"""

from __future__ import annotations

from .types import Envelope, Hardware

# Measured on MacBook Pro M5 Pro, 48 GB unified (development host).
BASELINE_RAM_GB = 48.0
BASELINE_PERF_CORES = 16
BASELINE_DISK_GB_S = 12.0
BASELINE_RAM_FRACTION = 0.68
BASELINE_MAX_CACHE_GB = 26.0
BASELINE_PREFILL_CACHE_GB = 4.0
BASELINE_HEADROOM_GB = 6.0
BASELINE_ACTIVATION_GB = 3.0
BASELINE_STAGING_CAP_GB = 6.0
BASELINE_KV_FP16_CTX = 8192
BASELINE_READ_THREADS = 16

BASELINE = Envelope(
    ram_gb=BASELINE_RAM_GB,
    ram_fraction=BASELINE_RAM_FRACTION,
    ram_budget_gb=round(BASELINE_RAM_GB * BASELINE_RAM_FRACTION, 2),
    max_cache_gb=BASELINE_MAX_CACHE_GB,
    prefill_cache_gb=BASELINE_PREFILL_CACHE_GB,
    read_threads=BASELINE_READ_THREADS,
    headroom_gb=BASELINE_HEADROOM_GB,
    activation_gb=BASELINE_ACTIVATION_GB,
    staging_cap_gb=BASELINE_STAGING_CAP_GB,
    kv_fp16_ctx=BASELINE_KV_FP16_CTX,
    disk_gb_s=BASELINE_DISK_GB_S,
    near_baseline=True,
)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _round1(v: float) -> float:
    return round(float(v), 1)


def is_baseline_hardware(hw: Hardware) -> bool:
    """True when RAM matches the measured 48 GB class (do not retune knobs).

    Performance-core counts differ across 48 GB SKUs (this host reports 6P+12E
    via sysctl while READ_THREADS was measured at 16). RAM is the capacity
    signal; do not drop the 48 GB machine off baseline because of core layout.
    """
    return abs(float(hw.ram_gb) - BASELINE_RAM_GB) <= 1.5


def derive_envelope(
    hw: Hardware,
    *,
    host_disk_gb_s: float | None = None,
    ram_fraction: float | None = None,
    overrides: dict | None = None,
) -> Envelope:
    """Map hardware to capacity knobs. ``overrides`` are explicit yaml/env caps."""
    ov = overrides or {}
    ram = float(hw.ram_gb)
    frac = float(
        ov["ram_fraction"]
        if ov.get("ram_fraction") is not None
        else (ram_fraction if ram_fraction is not None else BASELINE_RAM_FRACTION)
    )
    budget = ram * frac
    scale = ram / BASELINE_RAM_GB
    near = is_baseline_hardware(hw) and abs(frac - BASELINE_RAM_FRACTION) < 1e-9

    if near:
        max_cache = BASELINE_MAX_CACHE_GB
        prefill = BASELINE_PREFILL_CACHE_GB
        headroom = BASELINE_HEADROOM_GB
        activation = BASELINE_ACTIVATION_GB
        staging = BASELINE_STAGING_CAP_GB
        kv_ctx = BASELINE_KV_FP16_CTX
    else:
        max_cache = _round1(max(2.0, BASELINE_MAX_CACHE_GB * scale))
        prefill = _round1(_clamp(BASELINE_PREFILL_CACHE_GB * scale, 2.0, 8.0))
        headroom = _round1(_clamp(BASELINE_HEADROOM_GB * scale, 2.0, 8.0))
        activation = _round1(_clamp(BASELINE_ACTIVATION_GB * scale, 1.5, 4.0))
        staging = _round1(_clamp(BASELINE_STAGING_CAP_GB * scale, 1.0, 12.0))
        if ram <= 18:
            kv_ctx = 0
        elif ram <= 32:
            kv_ctx = 4096
        else:
            kv_ctx = BASELINE_KV_FP16_CTX

    threads = int(hw.perf_cores)
    threads = max(2, min(32, threads))
    if near:
        threads = BASELINE_READ_THREADS

    # Explicit yaml/profile knobs win over the formula.
    if ov.get("max_cache_gb") is not None:
        max_cache = float(ov["max_cache_gb"])
    if ov.get("prefill_cache_gb") is not None:
        prefill = float(ov["prefill_cache_gb"])
    if ov.get("headroom_gb") is not None:
        headroom = float(ov["headroom_gb"])
    if ov.get("activation_gb") is not None:
        activation = float(ov["activation_gb"])
    if ov.get("staging_cap_gb") is not None:
        staging = float(ov["staging_cap_gb"])
    if ov.get("kv_fp16_ctx") is not None:
        kv_ctx = int(ov["kv_fp16_ctx"])
    if ov.get("read_threads") is not None:
        threads = max(1, int(ov["read_threads"]))

    extra = {}
    if isinstance(ov.get("env"), dict):
        extra = {str(k): str(v) for k, v in ov["env"].items()}

    target_disk = hw.disk_gb_s
    host_disk = host_disk_gb_s
    delay = 0.0
    if ov.get("read_delay_us_per_mb") is not None:
        delay = float(ov["read_delay_us_per_mb"])
    elif target_disk and host_disk:
        delay = read_delay_us_per_mb(float(target_disk), float(host_disk))

    env = Envelope(
        ram_gb=ram,
        ram_fraction=frac,
        ram_budget_gb=round(budget, 2),
        max_cache_gb=float(max_cache),
        prefill_cache_gb=float(prefill),
        read_threads=int(threads),
        headroom_gb=float(headroom),
        activation_gb=float(activation),
        staging_cap_gb=float(staging),
        kv_fp16_ctx=int(kv_ctx),
        read_delay_us_per_mb=round(float(delay), 2),
        near_baseline=near,
        disk_gb_s=target_disk,
        extra_env=extra,
    )
    return env


def read_delay_us_per_mb(target_gb_s: float, host_gb_s: float) -> float:
    """Sleep per MB so a fast host SSD stands in for a slower target.

    Cannot emulate a *faster* disk than the host; returns 0 in that case.
    """
    if target_gb_s <= 0 or host_gb_s <= 0:
        return 0.0
    if target_gb_s >= host_gb_s:
        return 0.0
    host_us = 1e6 / (host_gb_s * 1024)
    tgt_us = 1e6 / (target_gb_s * 1024)
    return max(0.0, tgt_us - host_us)
