# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Affine envelope math from the measured 48 GB M5 Pro baseline.

Capacity knobs only. Do not put PRUNE / WAIT / sidecar / prefill-chunk here.

Formulas (ram_gb relative to 48):

    budget       = ram_gb * ram_fraction(ram_gb)
    ram_fraction = 0.68 up to 48 GB. Above that the budget stays 4.8 GiB under
                   Metal's working set: the host's real one, or 0.75 of RAM for
                   catalog and simulated Macs (96 GB -> 0.70, 128 GB -> 0.7125).
                   loader.py checks it again at load
    max_cache    = 26 * (budget / 32.64)       # ceiling; load still subtracts backbone
    prefill_cache = clamp(4 * ram/48, 2, 8)
    headroom     = clamp(6 * ram/48, 2, 8)     # flat KV+scratch when RESERVE_CTX is 0
    activation   = clamp(3 * ram/48, 1.5, 4)
    staging_cap  = clamp(6 * ram/48, 1, 12)
    read_threads = clamp(perf_cores, 2, 32), never under 16 from the 48 GB class up
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
# What the 48 GB budget leaves for macOS and apps (48 * 0.32).
BASELINE_OS_RESERVE_GB = BASELINE_RAM_GB * (1.0 - BASELINE_RAM_FRACTION)
# Share of each GB past 48 that also stays out of the budget.
LARGE_RESERVE_PER_GB = 0.12
LARGE_RAM_FRACTION_CAP = 0.82
# The budget is nearly all Metal memory, and going past Metal's recommended
# working set is what aborts a command buffer. 0.68 on 48 GB sits 4.8 GiB
# under the 48 GB M5 Pro's 37.44 GiB, so a raised budget keeps that gap. 0.75 of RAM
# is only the estimate for Macs that are not the host.
LARGE_WORKING_SET_FRACTION = 0.75
WORKING_SET_MARGIN_GB = 4.8

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


def auto_ram_fraction(
    ram_gb: float,
    working_set_gb: float | None = None,
    margin_gb: float = WORKING_SET_MARGIN_GB,
) -> float:
    """Default budget share of RAM for this machine size.

    Up to 48 GB this is exactly 0.68. Above that it is the lower of two
    bounds: macOS keeps the 48 GB reserve plus 12% of the extra RAM, and the
    budget stays WORKING_SET_MARGIN_GB under Metal's working set. That is the
    host's real figure when known, else 0.75 of RAM (96 GB -> 0.70, 128 GB ->
    0.7125).
    """
    ram = float(ram_gb)
    if ram <= BASELINE_RAM_GB + 1.5:
        return BASELINE_RAM_FRACTION
    reserve = BASELINE_OS_RESERVE_GB + LARGE_RESERVE_PER_GB * (ram - BASELINE_RAM_GB)
    by_os = 1.0 - reserve / ram
    ws = float(working_set_gb) if working_set_gb else LARGE_WORKING_SET_FRACTION * ram
    by_metal = (ws - float(margin_gb)) / ram
    frac = min(LARGE_RAM_FRACTION_CAP, by_os, by_metal)
    return round(max(BASELINE_RAM_FRACTION, frac), 4)


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
    auto_frac = auto_ram_fraction(ram, hw.gpu_working_set_gb)
    explicit = ov.get("ram_fraction") is not None or ram_fraction is not None
    frac = float(
        ov["ram_fraction"]
        if ov.get("ram_fraction") is not None
        else (ram_fraction if ram_fraction is not None else auto_frac)
    )
    budget = ram * frac
    scale = ram / BASELINE_RAM_GB
    # The cache ceiling grows with the default budget. A hand-set fraction
    # keeps the plain RAM-scaled ceiling it always had.
    cache_scale = scale if explicit else scale * (auto_frac / BASELINE_RAM_FRACTION)
    near = is_baseline_hardware(hw) and abs(frac - BASELINE_RAM_FRACTION) < 1e-9
    # The raised share is for the expert cache, which shrinks under memory
    # pressure. Whether a whole model loads resident stays on the 48 GB share.
    resident_frac = (
        BASELINE_RAM_FRACTION if not explicit and frac > BASELINE_RAM_FRACTION else None
    )

    if near:
        max_cache = BASELINE_MAX_CACHE_GB
        prefill = BASELINE_PREFILL_CACHE_GB
        headroom = BASELINE_HEADROOM_GB
        activation = BASELINE_ACTIVATION_GB
        staging = BASELINE_STAGING_CAP_GB
        kv_ctx = BASELINE_KV_FP16_CTX
    else:
        max_cache = _round1(max(2.0, BASELINE_MAX_CACHE_GB * cache_scale))
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
    elif ram >= BASELINE_RAM_GB - 1.5:
        # Readers wait on the SSD, not the CPU, and the drive wants ~qd16.
        # perflevel0 undercounts M5 chips (6 "Super" of 18 non-efficiency
        # cores), which put a 96 GB M5 at 6 readers.
        threads = max(BASELINE_READ_THREADS, threads)

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
        resident_fraction=resident_frac,
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
