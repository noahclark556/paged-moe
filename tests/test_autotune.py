# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""--speed and checkpoint inspection (no Metal)."""

from __future__ import annotations

import json
import os
import struct
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pagedmoe import autotune, launch_env


def _write_checkpoint(root: Path, experts: int = 8, top_k: int = 2) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fake_moe",
                "num_experts": experts,
                "num_experts_per_tok": top_k,
                "quantization": {"bits": 4, "group_size": 64},
            }
        )
    )
    header: dict = {}
    off = 0

    def add(name, dtype, shape, nbytes):
        nonlocal off
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + nbytes]}
        off += nbytes

    add("model.embed_tokens.weight", "F16", [100, 16], 3200)
    add("model.layers.0.self_attn.q_proj.weight", "F16", [16, 16], 512)
    for e in range(experts):
        base = f"model.layers.0.mlp.experts.{e}.down_proj"
        # 4-bit: 8 values per U32, one scale per 64 values.
        add(f"{base}.weight", "U32", [64, 8], 2048)
        add(f"{base}.scales", "F16", [64, 1], 128)
        add(f"{base}.biases", "F16", [64, 1], 128)
    raw = json.dumps(header).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * off)
    return root


def test_scan_counts_quantized_and_routed(tmp_path):
    ck_dir = _write_checkpoint(tmp_path / "m")
    total, routed, lookup = autotune._scan_params(ck_dir, {"bits": 4, "group_size": 64})
    assert routed == 8 * 64 * 64
    assert lookup == 1600
    assert total == routed + lookup + 256


def test_inspect_reads_moe_shape(tmp_path):
    ck = autotune.inspect(str(_write_checkpoint(tmp_path / "m")))
    assert ck.moe and ck.bits == 4 and ck.model_type == "fake_moe"
    # RAM comes from the host, and CI runners have about 7 GB.
    assert replace(ck, ram_gb=48.0).fits_resident
    assert not replace(ck, ram_gb=48.0, disk_gb=200.0).fits_resident


def test_inspect_needs_a_local_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        autotune.inspect(str(tmp_path / "missing"))


def test_fixed_scale_end_points():
    assert autotune._fixed_profile(1.0) == (0.0, 0.0, False)
    assert autotune._fixed_profile(0.5) == (0.7, 0.25, False)
    assert autotune._fixed_profile(0.0) == (0.95, 0.5, True)
    # Rounds half up like the slider it mirrors (0.825 -> 0.83).
    assert autotune._fixed_profile(0.25)[0] == 0.83


def test_parse_speed():
    assert autotune.parse_speed("exact") == 1.0
    assert autotune.parse_speed("auto") == "auto"
    assert autotune.parse_speed("0.4") == 0.4
    for bad in ("1.5", "-0.1", "fast"):
        with pytest.raises(ValueError):
            autotune.parse_speed(bad)


def _streamed(total_b, active_b, bits):
    return autotune.Checkpoint("x", "", total_b, active_b, True, bits, 500.0, 48.0)


def test_suggested_prune_caps():
    assert autotune.suggested_prune(_streamed(480, 35, 4)) == 0.85
    assert autotune.suggested_prune(_streamed(229, 7, 6)) == 0.7
    assert autotune.suggested_prune(_streamed(235, 22, 3)) == 0.7
    assert autotune.suggested_prune(_streamed(1000, 32, 2)) == 0.5
    assert autotune.suggested_prune(replace(_streamed(30, 3, 4), disk_gb=16)) == 0.0


def test_speed_one_is_exact():
    env = autotune.speed_env(_streamed(480, 35, 4), 1.0)
    assert env == {"PAGED_MOE_PRUNE": "0", "PAGED_MOE_WAIT_ABOVE": "0", "PAGED_MOE_FLOW": "0"}


def test_curve_spaces_positions_by_kept_mass(tmp_path, monkeypatch):
    p = [round(0.05 * i, 2) for i in range(20)]
    w = [round(0.025 * i, 3) for i in range(21)]
    mass = [[1 - 0.6 * a**1.5 - 0.3 * b for b in w] for a in p]
    table = tmp_path / "t.json"
    table.write_text(json.dumps({"version": 1, "rows": 5000, "prune": p, "wait_above": w, "mass": mass}))
    monkeypatch.setattr(autotune, "_curve_path", lambda _: table)
    curve = autotune.speed_curve("x")
    assert curve[0] == 0.0 and curve[-1] == 1.0
    assert all(b > a for a, b in zip(curve, curve[1:]))
    # Same numbers the app slider produces from this table.
    assert autotune.speed_env(_streamed(200, 20, 4), 0.5)["PAGED_MOE_PRUNE"] == "0.61"


def test_curve_needs_enough_rows(tmp_path, monkeypatch):
    table = tmp_path / "t.json"
    table.write_text(json.dumps({"version": 1, "rows": 10, "prune": [0, 1], "wait_above": [0, 1], "mass": [[1, 1], [0, 0]]}))
    monkeypatch.setattr(autotune, "_curve_path", lambda _: table)
    assert autotune.speed_curve("x") is None


def test_launch_env_speed_beats_config_and_env(tmp_path, monkeypatch):
    model = _write_checkpoint(tmp_path / "m")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f'models:\n  - path: {model}\n    env:\n      PAGED_MOE_PRUNE: "0.7"\n')
    monkeypatch.setenv("PAGED_MOE_CONFIG", str(cfg))
    monkeypatch.delenv("PAGED_MOE_HOOK", raising=False)
    monkeypatch.setenv("PAGED_MOE_WAIT_ABOVE", "0.3")
    for k in ("PAGED_MOE_PRUNE", "PAGED_MOE_FLOW"):
        monkeypatch.delenv(k, raising=False)
    argv = ["--model", str(model), "--speed", "1", "--port", "1"]
    launch_env.apply(argv, quiet=True)
    assert argv == ["--model", str(model), "--port", "1"]
    assert os.environ["PAGED_MOE_PRUNE"] == "0"
    assert os.environ["PAGED_MOE_WAIT_ABOVE"] == "0"


def test_launch_env_hook_off_without_speed_sets_nothing(tmp_path, monkeypatch, capsys):
    model = _write_checkpoint(tmp_path / "m")
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f'models:\n  - path: {model}\n    env:\n      PAGED_MOE_PRUNE: "0.7"\n')
    monkeypatch.setenv("PAGED_MOE_CONFIG", str(cfg))
    monkeypatch.setenv("PAGED_MOE_HOOK", "0")
    monkeypatch.delenv("PAGED_MOE_PRUNE", raising=False)
    assert launch_env.apply(["--model", str(model)]) == {}
    assert "PAGED_MOE_PRUNE" not in os.environ
    assert capsys.readouterr().err == ""
