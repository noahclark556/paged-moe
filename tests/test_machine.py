# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Machine envelope provider: scale, fit, OOM. No Metal required."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pagedmoe.machine.catalog import CATALOG
from pagedmoe.machine.ladder import LADDER, fit_model
from pagedmoe.machine.oom import EnvelopeOOM, enforce_fit
from pagedmoe.machine.provider import MachineProvider, reset_provider
from pagedmoe.machine.scale import (
    BASELINE,
    BASELINE_MAX_CACHE_GB,
    derive_envelope,
    read_delay_us_per_mb,
)
from pagedmoe.machine.types import Hardware


def test_baseline_48gb_is_exact():
    hw = Hardware(ram_gb=48, perf_cores=16, disk_gb_s=12.0, source="test")
    env = derive_envelope(hw)
    assert env.near_baseline is True
    assert env.max_cache_gb == BASELINE_MAX_CACHE_GB
    assert env.prefill_cache_gb == BASELINE.prefill_cache_gb
    assert env.headroom_gb == BASELINE.headroom_gb
    assert env.read_threads == 16
    assert env.kv_fp16_ctx == 8192
    # This host's sysctl P-core count is 6; RAM class still keeps measured knobs.
    env6 = derive_envelope(Hardware(ram_gb=48, perf_cores=6, source="test"))
    assert env6.near_baseline is True
    assert env6.read_threads == 16


def test_m2_16gb_scales_cache():
    hw = Hardware(ram_gb=16, perf_cores=8, disk_gb_s=6.0, source="test")
    env = derive_envelope(hw, host_disk_gb_s=12.0)
    assert env.near_baseline is False
    assert abs(env.max_cache_gb - 8.7) < 0.15  # 26 * 16/48
    assert env.prefill_cache_gb == 2.0
    assert env.headroom_gb == 2.0
    assert env.read_threads == 8
    assert env.kv_fp16_ctx == 0
    assert env.read_delay_us_per_mb > 0


def test_m2_air_uses_four_perf_cores():
    env = derive_envelope(CATALOG["mba-m2-2023-16gb"], host_disk_gb_s=12.0)
    assert env.read_threads == 4
    assert abs(env.max_cache_gb - 8.7) < 0.15


def test_128gb_scales_up():
    env = derive_envelope(CATALOG["mbp-m5-max-128gb"])
    assert env.max_cache_gb > 60
    assert env.max_cache_gb == round(26 * (128 / 48), 1)
    assert env.prefill_cache_gb == 8.0  # capped
    assert env.headroom_gb == 8.0  # capped


def test_faster_disk_has_no_delay():
    assert read_delay_us_per_mb(12.0, 6.0) == 0.0
    assert read_delay_us_per_mb(6.0, 12.0) > 0.0


def test_coder_next_fits_16gb():
    env = derive_envelope(CATALOG["mbp-m2-pro-2023-16gb"])
    row = next(m for m in LADDER if m["id"] == "coder-next-6bit")
    r = fit_model(row, env)
    assert r.status == "streamed"
    assert r.gain_gb > 40


def test_glm_ooms_16gb():
    env = derive_envelope(CATALOG["mba-m2-2023-16gb"])
    row = next(m for m in LADDER if m["id"] == "glm-47-4bit")
    r = fit_model(row, env)
    assert r.status == "oom"


def test_kimi_fits_48gb():
    env = derive_envelope(CATALOG["host-m5-pro-48gb"])
    row = next(m for m in LADDER if m["id"] == "kimi-k2-4bit")
    r = fit_model(row, env)
    assert r.status == "streamed"
    assert r.gain_gb > 500
    assert r.speed_basis == "measured"


def test_olmoe_resident_16gb():
    env = derive_envelope(CATALOG["mba-m2-2023-16gb"])
    row = next(m for m in LADDER if m["id"] == "olmoe-1b-7b-4bit")
    r = fit_model(row, env)
    assert r.status == "resident"


def test_synthetic_oom_raises():
    env = derive_envelope(Hardware(ram_gb=8, perf_cores=4, source="test"))
    try:
        enforce_fit(
            backbone_bytes=int(12e9),
            headroom_bytes=int(env.headroom_gb * (1 << 30)),
            budget_bytes=int(env.ram_fraction * 8 * (1 << 30)),
            name="synthetic",
            ram_gb=8,
        )
    except EnvelopeOOM:
        return
    raise AssertionError("expected EnvelopeOOM")


def test_yaml_ram_override():
    import tempfile
    from pathlib import Path

    prev_cfg = os.environ.get("PAGED_MOE_CONFIG")
    prev_home = os.environ.get("PAGED_MOE_HOME")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cfg = root / "paged-moe-config.yaml"
        cfg.write_text("machine:\n  auto: true\n  ram_gb: 16\n  perf_cores: 8\n")
        os.environ["PAGED_MOE_CONFIG"] = str(cfg)
        os.environ["PAGED_MOE_HOME"] = str(root / "pm")
        try:
            reset_provider()
            p = MachineProvider.detect()
            assert abs(p.hardware.ram_gb - 16) < 0.01
            assert abs(p.envelope.max_cache_gb - 8.7) < 0.15
        finally:
            reset_provider()
            if prev_cfg is None:
                os.environ.pop("PAGED_MOE_CONFIG", None)
            else:
                os.environ["PAGED_MOE_CONFIG"] = prev_cfg
            if prev_home is None:
                os.environ.pop("PAGED_MOE_HOME", None)
            else:
                os.environ["PAGED_MOE_HOME"] = prev_home


def test_explicit_env_wins_over_derived():
    prev = os.environ.get("PAGED_MOE_MAX_CACHE_GB")
    os.environ["PAGED_MOE_MAX_CACHE_GB"] = "11"
    try:
        reset_provider()
        p = MachineProvider.from_catalog("mba-m2-2023-16gb", simulate=True)
        p.apply(only_unset=True)
        assert os.environ["PAGED_MOE_MAX_CACHE_GB"] == "11"
    finally:
        if prev is None:
            os.environ.pop("PAGED_MOE_MAX_CACHE_GB", None)
        else:
            os.environ["PAGED_MOE_MAX_CACHE_GB"] = prev
        reset_provider()


def test_detect_honors_simulate_ram():
    prev = os.environ.get("PAGED_MOE_SIMULATE_RAM_GB")
    os.environ["PAGED_MOE_SIMULATE_RAM_GB"] = "16"
    try:
        reset_provider()
        p = MachineProvider.detect()
        assert abs(float(p.hardware.ram_gb) - 16) < 0.05
        assert p.simulate
        assert p.envelope.ram_budget_gb < 12
    finally:
        if prev is None:
            os.environ.pop("PAGED_MOE_SIMULATE_RAM_GB", None)
        else:
            os.environ["PAGED_MOE_SIMULATE_RAM_GB"] = prev
        reset_provider()


def test_total_ram_bytes_honors_simulate():
    prev = os.environ.get("PAGED_MOE_SIMULATE_RAM_GB")
    os.environ["PAGED_MOE_SIMULATE_RAM_GB"] = "16"
    try:
        from pagedmoe import config as es_config

        assert abs(es_config.total_ram_bytes() / (1 << 30) - 16) < 0.01
    finally:
        if prev is None:
            os.environ.pop("PAGED_MOE_SIMULATE_RAM_GB", None)
        else:
            os.environ["PAGED_MOE_SIMULATE_RAM_GB"] = prev


def main() -> None:
    tests = [
        test_baseline_48gb_is_exact,
        test_m2_16gb_scales_cache,
        test_m2_air_uses_four_perf_cores,
        test_128gb_scales_up,
        test_faster_disk_has_no_delay,
        test_coder_next_fits_16gb,
        test_glm_ooms_16gb,
        test_kimi_fits_48gb,
        test_olmoe_resident_16gb,
        test_synthetic_oom_raises,
        test_yaml_ram_override,
        test_explicit_env_wins_over_derived,
        test_detect_honors_simulate_ram,
        test_total_ram_bytes_honors_simulate,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    if failed:
        raise SystemExit(1)
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
