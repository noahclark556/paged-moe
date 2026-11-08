# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
from pagedmoe import config, dev_mitm, mlx_hook, release
from pagedmoe.machine import oom


def test_source_checkout_is_not_sealed():
    assert release.SEALED is False


def test_source_checkout_reads_dev_switches(monkeypatch):
    monkeypatch.setenv("PAGED_MOE_SIMULATE_RAM_GB", "16")
    monkeypatch.setenv("PAGED_MOE_MITM", "1")
    monkeypatch.setenv("PAGED_MOE_HOOK", "0")
    assert release.env("PAGED_MOE_SIMULATE_RAM_GB") == "16"
    assert config.total_ram_bytes() == 16 << 30
    assert dev_mitm._env_on() is True
    assert mlx_hook._hook_disabled() is True


def test_release_ignores_dev_switches(monkeypatch):
    monkeypatch.setattr(release, "SEALED", True)
    monkeypatch.setenv("PAGED_MOE_SIMULATE_RAM_GB", "16")
    monkeypatch.setenv("PAGED_MOE_ENVELOPE_STRICT", "1")
    monkeypatch.setenv("PAGED_MOE_MITM", "1")
    monkeypatch.setenv("PAGED_MOE_HOOK", "1")
    monkeypatch.setenv("PAGED_MOE_ENABLE_ALL", "1")
    monkeypatch.setenv("PAGED_MOE_PRUNE", "0.5")
    assert release.env("PAGED_MOE_SIMULATE_RAM_GB") == ""
    assert config.total_ram_bytes() != 16 << 30
    assert oom.envelope_strict() is False
    assert dev_mitm._env_on() is False
    assert mlx_hook._hook_disabled() is True
    assert release.env("PAGED_MOE_PRUNE") == "0.5"
