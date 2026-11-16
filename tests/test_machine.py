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
    auto_ram_fraction,
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
    assert env.ram_fraction == auto_ram_fraction(128) == 0.7125
    assert env.max_cache_gb == round(26 * (128 / 48) * (0.7125 / 0.68), 1)
    assert env.prefill_cache_gb == 8.0  # capped
    assert env.headroom_gb == 8.0  # capped


def _old_envelope(ram: float) -> tuple:
    """derive_envelope before the large-Mac fraction (flat 0.68, RAM-scaled cap)."""
    s = ram / 48
    clamp = lambda v, lo, hi: max(lo, min(hi, v))  # noqa: E731
    return (
        0.68,
        round(max(2.0, 26 * s), 1),
        round(clamp(4 * s, 2.0, 8.0), 1),
        round(clamp(6 * s, 2.0, 8.0), 1),
        round(clamp(3 * s, 1.5, 4.0), 1),
        round(clamp(6 * s, 1.0, 12.0), 1),
    )


def test_48gb_and_below_unchanged():
    for ram in (8, 16, 18, 24, 32, 36, 46.5, 49.5):
        env = derive_envelope(Hardware(ram_gb=ram, perf_cores=8, source="test"))
        got = (
            env.ram_fraction,
            env.max_cache_gb,
            env.prefill_cache_gb,
            env.headroom_gb,
            env.activation_gb,
            env.staging_cap_gb,
        )
        if env.near_baseline:
            assert env.ram_fraction == 0.68 and env.max_cache_gb == BASELINE_MAX_CACHE_GB
        else:
            assert got == _old_envelope(ram), (ram, got, _old_envelope(ram))
        assert env.resident_fraction is None, ram
        assert "PAGED_MOE_RESIDENT_RAM_FRACTION" not in env.to_env()


def test_large_mac_fraction_curve():
    # 64 GB: 0.75 * 64 - 4.8 = 43.2 GiB is under the 0.68 budget, so no raise.
    env = derive_envelope(Hardware(ram_gb=64, perf_cores=12, source="test"))
    assert env.ram_fraction == 0.68 and env.resident_fraction is None
    want = {96: 0.70, 128: 0.7125, 192: 0.725, 256: 0.7312, 512: 0.7406}
    prev = 0.68
    for ram, frac in want.items():
        got = auto_ram_fraction(ram)
        assert abs(got - frac) < 0.0006, (ram, got)
        assert got >= prev
        prev = got
        env = derive_envelope(Hardware(ram_gb=ram, perf_cores=12, source="test"))
        assert env.ram_fraction == got
        # Same gap under a 0.75 Metal working set that 0.68 has on 48 GB.
        assert ram * 0.75 - ram * got >= 4.8 - 0.01
        # macOS keeps at least the 48 GB reserve in absolute terms.
        assert ram * (1 - got) >= 48 * 0.32 - 0.01
        # The extra share never decides a resident load.
        assert env.resident_fraction == 0.68
        assert env.to_env()["PAGED_MOE_RESIDENT_RAM_FRACTION"] == "0.68"
        # Cache ceiling tracks the budget the same way 26 GB tracks 32.64.
        assert abs(env.max_cache_gb - 26 * env.ram_budget_gb / 32.64) < 0.1


def test_real_working_set_replaces_estimate():
    # Unknown (catalog, simulated): 0.75 of RAM estimate.
    assert auto_ram_fraction(96, None) == 0.70
    assert auto_ram_fraction(96, 72.0) == 0.70
    # A host that reports 0.78 of RAM (like this 48 GB one) gets more.
    assert auto_ram_fraction(96, 74.88) == 0.73
    # A small one never goes under 0.68.
    assert auto_ram_fraction(96, 60.0) == 0.68
    # A huge one is still held by the macOS reserve.
    assert auto_ram_fraction(96, 200.0) == 0.78
    env = derive_envelope(
        Hardware(ram_gb=96, perf_cores=12, gpu_working_set_gb=74.88, source="test")
    )
    assert env.ram_fraction == 0.73 and env.resident_fraction == 0.68
    # 48 GB ignores it entirely.
    env = derive_envelope(
        Hardware(ram_gb=48, perf_cores=6, gpu_working_set_gb=40.0, source="test")
    )
    assert env.near_baseline and env.ram_fraction == 0.68


def test_explicit_fraction_keeps_old_behavior():
    env = derive_envelope(Hardware(ram_gb=96, perf_cores=12, source="test"), ram_fraction=0.68)
    assert env.ram_fraction == 0.68
    assert env.resident_fraction is None
    env = derive_envelope(
        Hardware(ram_gb=96, perf_cores=12, source="test"), overrides={"ram_fraction": 0.7}
    )
    assert env.ram_fraction == 0.7
    assert env.resident_fraction is None


def test_96gb_resident_call_unchanged():
    # A checkpoint that only fits under the raised share stays streamed, with
    # a bigger cache than before.
    hw = Hardware(ram_gb=96, perf_cores=12, source="test")
    new = derive_envelope(hw)
    old = derive_envelope(hw, ram_fraction=0.68)
    row = {
        # 62 + 8.6 headroom fits the 0.70 budget (72.2 GB) but not 0.68 (70.1).
        "id": "synthetic-62",
        "name": "synthetic 62 GB",
        "checkpoint_gb": 62.0,
        "backbone_gb": 4.0,
        "kv_kb_token": 0,
    }
    r_old, r_new = fit_model(row, old), fit_model(row, new)
    assert r_old.status == r_new.status == "streamed"
    assert r_old.cache_gb == old.max_cache_gb == 52.0
    assert r_new.cache_gb == new.max_cache_gb == 53.5
    small = dict(row, id="synthetic-55", name="synthetic 55 GB", checkpoint_gb=55.0)
    assert fit_model(small, old).status == fit_model(small, new).status == "resident"


def test_loader_caps_raised_budget_on_real_working_set():
    from pagedmoe import config as cfg
    from pagedmoe import loader, mempressure

    gib = 1 << 30
    ram = 96 * gib
    keep = (cfg.RAM_FRACTION, cfg.RESIDENT_RAM_FRACTION, mempressure._working_set_read,
            mempressure._working_set, os.environ.get("PAGED_MOE_SIMULATE_GPU_WORKING_SET_GB"))
    try:
        cfg.RAM_FRACTION, cfg.RESIDENT_RAM_FRACTION = 0.70, 0.68
        budget = int(0.70 * ram)
        for ws_gb, want in (
            (72 * gib / 1e9, int(0.70 * ram)),  # 0.75 working set: budget already fits
            (71 * gib / 1e9, int(71 * gib - 4.8 * gib)),  # smaller one pulls it in
            (60 * gib / 1e9, int(0.68 * ram)),  # never below the old budget
        ):
            os.environ["PAGED_MOE_SIMULATE_GPU_WORKING_SET_GB"] = str(ws_gb)
            mempressure._working_set_read = False
            got = loader.raised_budget(budget, ram)
            assert abs(got - want) < 2e6, (ws_gb, got, want)
    finally:
        cfg.RAM_FRACTION, cfg.RESIDENT_RAM_FRACTION = keep[0], keep[1]
        mempressure._working_set_read, mempressure._working_set = keep[2], keep[3]
        if keep[4] is None:
            os.environ.pop("PAGED_MOE_SIMULATE_GPU_WORKING_SET_GB", None)
        else:
            os.environ["PAGED_MOE_SIMULATE_GPU_WORKING_SET_GB"] = keep[4]


def test_hand_set_fraction_drops_resident_env():
    saved = dict(os.environ)
    try:
        for k in [k for k in os.environ if k.startswith("PAGED_MOE_")]:
            os.environ.pop(k)
        os.environ["PAGED_MOE_RAM_FRACTION"] = "0.7"
        reset_provider()
        hw = Hardware(ram_gb=96, perf_cores=12, source="test")
        p = MachineProvider.from_hardware(hw)
        assert p.envelope.ram_fraction == 0.7
        assert p.envelope.max_cache_gb == 52.0  # old RAM-scaled ceiling
        p.apply_to_environ(only_unset=True)
        assert os.environ["PAGED_MOE_RAM_FRACTION"] == "0.7"
        assert "PAGED_MOE_RESIDENT_RAM_FRACTION" not in os.environ
        os.environ.pop("PAGED_MOE_RAM_FRACTION")
        os.environ.pop("PAGED_MOE_MAX_CACHE_GB", None)
        p = MachineProvider.from_hardware(hw)
        assert p.envelope.max_cache_gb == 53.5
        p.apply_to_environ(only_unset=True)
        assert os.environ["PAGED_MOE_RAM_FRACTION"] == "0.7"
        assert os.environ["PAGED_MOE_RESIDENT_RAM_FRACTION"] == "0.68"
    finally:
        os.environ.clear()
        os.environ.update(saved)
        reset_provider()


def test_read_threads_floor_from_48gb_up():
    # M5 sysctl reports 6 perflevel0 cores. Bigger RAM must not mean fewer readers.
    for ram in (64, 96, 128, 192, 256, 512):
        env = derive_envelope(Hardware(ram_gb=ram, perf_cores=6, source="test"))
        assert env.near_baseline is False
        assert env.read_threads == 16, (ram, env.read_threads)
    # 48 GB with a custom fraction skips the baseline snap; still 16.
    env = derive_envelope(Hardware(ram_gb=48, perf_cores=6, source="test"), ram_fraction=0.6)
    assert env.near_baseline is False
    assert env.read_threads == 16
    # More cores still raise it, up to the cap.
    assert derive_envelope(Hardware(ram_gb=192, perf_cores=24, source="test")).read_threads == 24
    assert derive_envelope(Hardware(ram_gb=512, perf_cores=40, source="test")).read_threads == 32
    # Explicit override still wins.
    env = derive_envelope(
        Hardware(ram_gb=256, perf_cores=6, source="test"), overrides={"read_threads": 8}
    )
    assert env.read_threads == 8
    # Below the 48 GB class nothing changes.
    assert derive_envelope(Hardware(ram_gb=36, perf_cores=6, source="test")).read_threads == 6
    assert derive_envelope(Hardware(ram_gb=16, perf_cores=4, source="test")).read_threads == 4


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
        assert p.hardware.gpu_working_set_gb is None  # the host's does not apply
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
        test_48gb_and_below_unchanged,
        test_large_mac_fraction_curve,
        test_real_working_set_replaces_estimate,
        test_explicit_fraction_keeps_old_behavior,
        test_96gb_resident_call_unchanged,
        test_loader_caps_raised_budget_on_real_working_set,
        test_hand_set_fraction_drops_resident_env,
        test_read_threads_floor_from_48gb_up,
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
