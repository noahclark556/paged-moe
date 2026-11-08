# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPU memory held outside the engine at load (mempressure.gpu_held_excess).

Covered here:
  - normal readings (a few GB of WindowServer / editor) change nothing
  - a reading past the threshold is charged above the normal allowance
  - the threshold scales with RAM on big machines
  - the simulate env stands in for ioreg
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pagedmoe import mempressure  # noqa: E402

GB = 10**9
RAM48 = 48 * (1 << 30)


def test_normal_readings_are_ignored():
    for held in (None, 0, 1 * GB, 4 * GB, 7.9 * GB):
        assert mempressure.gpu_held_excess(held, RAM48) == 0


def test_leak_sized_reading_is_charged_above_normal():
    assert mempressure.gpu_held_excess(30.6 * GB, RAM48) == int(30.6 * GB) - mempressure.GPU_NORMAL_BYTES


def test_threshold_scales_with_ram():
    ram192 = 192 * (1 << 30)
    assert mempressure.gpu_held_excess(20 * GB, ram192) == 0
    assert mempressure.gpu_held_excess(40 * GB, ram192) > 0


def test_simulate_env(monkeypatch):
    monkeypatch.setenv("PAGED_MOE_SIMULATE_GPU_HELD_GB", "12.5")
    assert mempressure.gpu_in_use_bytes() == int(12.5 * GB)
    monkeypatch.setenv("PAGED_MOE_SIMULATE_GPU_HELD_GB", "junk")
    assert mempressure.gpu_in_use_bytes() is None


def _reset_working_set(monkeypatch):
    monkeypatch.setattr(mempressure, "_working_set", None, raising=False)
    monkeypatch.setattr(mempressure, "_working_set_read", False, raising=False)


def test_working_set_simulate_env(monkeypatch):
    _reset_working_set(monkeypatch)
    monkeypatch.setenv("PAGED_MOE_SIMULATE_GPU_WORKING_SET_GB", "40.2")
    assert mempressure.gpu_working_set_bytes() == int(40.2 * GB)
    # Cached: a second call does not re-read even if the env changes.
    monkeypatch.setenv("PAGED_MOE_SIMULATE_GPU_WORKING_SET_GB", "10")
    assert mempressure.gpu_working_set_bytes() == int(40.2 * GB)


def test_working_set_bad_env_is_none(monkeypatch):
    _reset_working_set(monkeypatch)
    monkeypatch.setenv("PAGED_MOE_SIMULATE_GPU_WORKING_SET_GB", "junk")
    assert mempressure.gpu_working_set_bytes() is None


def _ws(monkeypatch, gb):
    _reset_working_set(monkeypatch)
    monkeypatch.setenv("PAGED_MOE_SIMULATE_GPU_WORKING_SET_GB", str(gb))


def test_held_aware_budget_clamps_to_the_working_set(monkeypatch):
    # 10 GB held by another GPU app on a 48 GB / 40 GB-working-set Mac: load,
    # but with the cache sized to what is actually resident, not the RAM budget.
    from pagedmoe import config, loader

    _ws(monkeypatch, 40)
    monkeypatch.setattr(config, "STAGING_CAP_GB", 6.0, raising=False)
    backbone = 7 * GB
    headroom = 5 * GB
    held = 10 * GB
    excess = mempressure.gpu_held_excess(held, RAM48)
    out = loader.held_aware_budget(
        35 * GB,
        gpu_held=held,
        held_excess=excess,
        backbone_bytes=backbone,
        headroom=headroom,
        ram=RAM48,
        name="m",
    )
    ceiling = min(40 * GB, RAM48) - (2 << 30)
    scratch = 6 * (1 << 30)
    expected = min(35 * GB - excess, ceiling - scratch - held)
    assert out == expected
    assert out > backbone + headroom + GB  # above the floor, so it loads


def test_held_aware_budget_refuses_when_floor_will_not_fit(monkeypatch):
    # A ~30 GB ghost on the same Mac: even a 1 GB cache cannot fit, so refuse
    # instead of loading into an abort that strands more memory.
    from pagedmoe import config, loader
    from pagedmoe.machine.oom import EnvelopeOOM
    import pytest

    _ws(monkeypatch, 40)
    monkeypatch.setattr(config, "STAGING_CAP_GB", 6.0, raising=False)
    held = 30 * GB
    excess = mempressure.gpu_held_excess(held, RAM48)
    with pytest.raises(EnvelopeOOM):
        loader.held_aware_budget(
            35 * GB,
            gpu_held=held,
            held_excess=excess,
            backbone_bytes=7 * GB,
            headroom=5 * GB,
            ram=RAM48,
            name="m",
        )
