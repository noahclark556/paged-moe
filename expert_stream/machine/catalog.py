# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Named Mac classes. Typical published specs, not a substitute for detect.

``source`` on each row:
  apple-spec  RAM SKUs / cores / bandwidth from Apple's published configs
  typical-ssd sequential read stand-in (NAND count varies a lot by SSD size)
  measured    this development host
"""

from __future__ import annotations

from .types import Hardware

# Typical cold sequential internal SSD (GB/s). 256 GB SKUs are often much
# slower; detect.probe_disk_gb_s is the truth when available.
_SSD_AIR = 3.0
_SSD_PRO = 6.0
_SSD_M4_PRO = 7.5
_SSD_M5_PRO = 12.0


def _h(**kwargs) -> Hardware:
    kwargs.setdefault("source", "catalog")
    return Hardware(**kwargs)


CATALOG: dict[str, Hardware] = {
    # --- 16 GB class (most common weaker Macs) --------------------------------
    "mba-m2-2023-16gb": _h(
        ram_gb=16,
        perf_cores=4,
        efficiency_cores=4,
        chip="Apple M2",
        hw_model="Mac14,2",
        gpu_cores=10,
        mem_bw_gb_s=100,
        disk_gb_s=_SSD_AIR,
        year=2023,
        form="air",
        notes="MacBook Air M2 16 GB. Tightest common MLX box. 4 performance cores.",
    ),
    "mbp-m2-pro-2023-16gb": _h(
        ram_gb=16,
        perf_cores=8,
        efficiency_cores=4,
        chip="Apple M2 Pro",
        hw_model="Mac14,9",
        gpu_cores=19,
        mem_bw_gb_s=200,
        disk_gb_s=_SSD_PRO,
        year=2023,
        form="pro",
        notes="14/16-inch MacBook Pro M2 Pro 16 GB. Same RAM as Air, 8P cores + 200 GB/s.",
    ),
    "mba-m3-2024-16gb": _h(
        ram_gb=16,
        perf_cores=4,
        efficiency_cores=4,
        chip="Apple M3",
        hw_model="Mac15,12",
        gpu_cores=10,
        mem_bw_gb_s=100,
        disk_gb_s=_SSD_AIR,
        year=2024,
        form="air",
        notes="MacBook Air M3 16 GB.",
    ),
    "mbp-m4-2024-16gb": _h(
        ram_gb=16,
        perf_cores=4,
        efficiency_cores=6,
        chip="Apple M4",
        hw_model="Mac16,1",
        gpu_cores=10,
        mem_bw_gb_s=120,
        disk_gb_s=_SSD_AIR,
        year=2024,
        form="pro",
        notes="14-inch MacBook Pro M4 16 GB (10-core CPU, 4P+6E).",
    ),
    "mba-m4-2025-16gb": _h(
        ram_gb=16,
        perf_cores=4,
        efficiency_cores=6,
        chip="Apple M4",
        gpu_cores=10,
        mem_bw_gb_s=120,
        disk_gb_s=_SSD_AIR,
        year=2025,
        form="air",
        notes="MacBook Air M4 16 GB.",
    ),
    # --- 18-36 GB -------------------------------------------------------------
    "mbp-m3-pro-2023-18gb": _h(
        ram_gb=18,
        perf_cores=6,
        efficiency_cores=5,
        chip="Apple M3 Pro",
        hw_model="Mac15,3",
        gpu_cores=18,
        mem_bw_gb_s=150,
        disk_gb_s=_SSD_PRO,
        year=2023,
        form="pro",
        notes="M3 Pro base unified memory is 18 GB, not 16.",
    ),
    "mbp-m2-pro-2023-32gb": _h(
        ram_gb=32,
        perf_cores=8,
        efficiency_cores=4,
        chip="Apple M2 Pro",
        gpu_cores=19,
        mem_bw_gb_s=200,
        disk_gb_s=_SSD_PRO,
        year=2023,
        form="pro",
    ),
    "mbp-m3-pro-2023-36gb": _h(
        ram_gb=36,
        perf_cores=6,
        efficiency_cores=6,
        chip="Apple M3 Pro",
        gpu_cores=18,
        mem_bw_gb_s=150,
        disk_gb_s=_SSD_PRO,
        year=2023,
        form="pro",
    ),
    "mbp-m4-pro-2024-24gb": _h(
        ram_gb=24,
        perf_cores=8,
        efficiency_cores=4,
        chip="Apple M4 Pro",
        hw_model="Mac16,6",
        gpu_cores=16,
        mem_bw_gb_s=273,
        disk_gb_s=_SSD_M4_PRO,
        year=2024,
        form="pro",
        notes="M4 Pro base unified memory is 24 GB.",
    ),
    # --- 48 GB class (baseline + peers) --------------------------------------
    "mbp-m4-pro-2024-48gb": _h(
        ram_gb=48,
        perf_cores=10,
        efficiency_cores=4,
        chip="Apple M4 Pro",
        gpu_cores=20,
        mem_bw_gb_s=273,
        disk_gb_s=_SSD_M4_PRO,
        year=2024,
        form="pro",
    ),
    "mbp-m4-max-2024-48gb": _h(
        ram_gb=48,
        perf_cores=12,
        efficiency_cores=4,
        chip="Apple M4 Max",
        gpu_cores=40,
        mem_bw_gb_s=546,
        disk_gb_s=_SSD_M4_PRO,
        year=2024,
        form="max",
    ),
    "host-m5-pro-48gb": _h(
        ram_gb=48,
        perf_cores=16,
        efficiency_cores=None,
        chip="Apple M5 Pro",
        hw_model="Mac17,9",
        gpu_cores=None,
        mem_bw_gb_s=546,
        disk_gb_s=_SSD_M5_PRO,
        year=2025,
        form="pro",
        source="measured",
        notes=(
            "Development host (Mac17,9). Engine defaults in config.py were "
            "measured here. sysctl P-core count is not the READ_THREADS "
            "default; keep 16 on this RAM class."
        ),
    ),
    # --- scale-up -------------------------------------------------------------
    "mbp-m4-max-2024-64gb": _h(
        ram_gb=64,
        perf_cores=12,
        efficiency_cores=4,
        chip="Apple M4 Max",
        gpu_cores=40,
        mem_bw_gb_s=546,
        disk_gb_s=_SSD_M4_PRO,
        year=2024,
        form="max",
    ),
    "mbp-m5-pro-64gb": _h(
        ram_gb=64,
        perf_cores=16,
        chip="Apple M5 Pro",
        mem_bw_gb_s=546,
        disk_gb_s=_SSD_M5_PRO,
        year=2025,
        form="pro",
        notes="Same chip class as the host, more unified memory. Envelope scales up. GPU tok/s not re-measured.",
    ),
    "mbp-m5-max-128gb": _h(
        ram_gb=128,
        perf_cores=16,
        chip="Apple M5 Max",
        mem_bw_gb_s=546,
        disk_gb_s=_SSD_M5_PRO,
        year=2025,
        form="max",
        notes="Scale-up estimate. Not run on this hardware. GPU/disk may be faster than the host.",
    ),
}


def catalog_entry(profile_id: str) -> Hardware:
    if profile_id not in CATALOG:
        known = ", ".join(sorted(CATALOG))
        raise KeyError(f"unknown catalog id {profile_id!r}. known: {known}")
    return CATALOG[profile_id]


def list_catalog() -> list[str]:
    return list(CATALOG.keys())


def match_catalog(hw: Hardware) -> str | None:
    """Best named class for detected hardware (RAM + chip family)."""
    chip = (hw.chip or "").lower()
    ram = float(hw.ram_gb)
    best = None
    best_score = -1
    for pid, row in CATALOG.items():
        score = 0
        if abs(row.ram_gb - ram) <= 1.0:
            score += 3
        elif abs(row.ram_gb - ram) <= 4.0:
            score += 1
        row_chip = (row.chip or "").lower()
        if chip and row_chip and (row_chip in chip or chip in row_chip):
            score += 3
        if hw.form and row.form == hw.form:
            score += 1
        if score > best_score:
            best_score = score
            best = pid
    if best_score < 3:
        return None
    return best
