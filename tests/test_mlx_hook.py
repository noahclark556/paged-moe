# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""mlx_lm drop-in hook: matching + env merge (no Metal)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from expert_stream import config
from expert_stream import mlx_hook
from expert_stream.user_config import ensure_sample_config, preferred_config_path


def test_default_sidecar_dir_is_paged_moe(tmp_path, monkeypatch):
    monkeypatch.delenv("EXPERT_STREAM_SIDECAR_DIR", raising=False)
    # Re-read default helper (module already bound SIDECAR_DIR at import).
    assert "paged_moe" in config._default_sidecar_dir().replace("\\", "/")
    assert "global_agent" not in config._default_sidecar_dir()


def test_sync_from_environ_updates_sidecar_dir(monkeypatch):
    monkeypatch.setenv("EXPERT_STREAM_SIDECAR_DIR", "/tmp/es-sidecar-test")
    config.sync_from_environ()
    assert config.SIDECAR_DIR == "/tmp/es-sidecar-test"


def test_marker_file_opts_in(tmp_path, monkeypatch):
    monkeypatch.setenv("PAGED_MOE_HOME", str(tmp_path / "pm"))
    monkeypatch.setenv("PAGED_MOE_CONFIG", str(tmp_path / "paged-moe-config.yaml"))
    monkeypatch.delenv("PAGED_MOE", raising=False)
    monkeypatch.delenv("EXPERT_STREAM_HOOK", raising=False)
    ensure_sample_config()
    model = tmp_path / "fake-moe"
    model.mkdir()
    assert mlx_hook.should_stream(str(model)) is False
    (model / ".paged_moe").write_text("")
    assert mlx_hook.should_stream(str(model)) is True


def test_config_yaml_path_match(tmp_path, monkeypatch):
    cfg = tmp_path / "paged-moe-config.yaml"
    monkeypatch.setenv("PAGED_MOE_CONFIG", str(cfg))
    monkeypatch.setenv("PAGED_MOE_HOME", str(tmp_path / "pm"))
    monkeypatch.delenv("PAGED_MOE", raising=False)
    model = tmp_path / "big-moe"
    model.mkdir()
    cfg.write_text(
        "enable_all: false\n"
        "models:\n"
        f"  - path: {model}\n"
        "    env:\n"
        '      EXPERT_STREAM_SIDECAR: "1"\n'
    )
    assert mlx_hook.should_stream(str(model)) is True
    env = mlx_hook._merged_env(str(model))
    assert env.get("EXPERT_STREAM_SIDECAR") == "1"


def test_preferred_config_is_home_level(tmp_path, monkeypatch):
    monkeypatch.delenv("PAGED_MOE_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    path = preferred_config_path()
    assert path == tmp_path / "paged-moe-config.yaml"
    assert ".paged_moe" not in str(path)


def test_legacy_config_still_read(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PAGED_MOE_CONFIG", raising=False)
    monkeypatch.delenv("PAGED_MOE_HOME", raising=False)
    legacy_root = tmp_path / ".paged_moe"
    legacy_root.mkdir()
    model = tmp_path / "legacy-moe"
    model.mkdir()
    (legacy_root / "config.yaml").write_text(
        "enable_all: false\n"
        "models:\n"
        f"  - path: {model}\n"
    )
    assert mlx_hook.should_stream(str(model)) is True


def test_hook_disabled_env(monkeypatch):
    monkeypatch.setenv("PAGED_MOE_HOOK", "0")
    assert mlx_hook.install() is False
