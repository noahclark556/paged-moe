# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
import os
import tempfile

import numpy as np

from pagedmoe import gpu_crumbs, gpu_trail


def test_report_names_the_last_work_and_layer():
    gpu_trail.mark("prefill chunk 148 tok at kv=21455")
    gpu_trail.step("decode first step (1 tok left from prefill)")
    gpu_trail.layer("model.layers.16.mlp", False)
    gpu_trail.layer("model.layers.17.mlp", False)
    lines = gpu_trail.report()
    work = [ln for ln in lines if "ago" in ln and "last MoE" not in ln]
    assert "decode first step" in work[-1]
    assert any("layer fetch gaps" in ln and "L17 +" in ln for ln in lines)
    assert any(ln.endswith("last MoE layer fetched: model.layers.17.mlp (decode)") for ln in lines)


def test_steps_do_not_push_rare_work_out():
    gpu_trail.mark("tool-warm prefetch enter, 48 layer(s)")
    for n in range(200):
        gpu_trail.step(f"decode plain step, token {n}")
    lines = gpu_trail.report()
    assert any("tool-warm prefetch enter" in ln for ln in lines)
    assert not any("token 100," in ln or ln.endswith("token 100") for ln in lines)
    assert any(ln.endswith("token 199") for ln in lines)


def test_probe_lines_and_a_failing_probe():
    gpu_trail.probe("t_ok", lambda: "cache: 3 of 9 reads still running")
    gpu_trail.probe("t_bad", lambda: 1 / 0)
    try:
        lines = gpu_trail.report()
        assert any("cache: 3 of 9 reads still running" in ln for ln in lines)
        assert any("t_bad: probe failed (ZeroDivisionError" in ln for ln in lines)
    finally:
        gpu_trail._probes.pop("t_ok", None)
        gpu_trail._probes.pop("t_bad", None)


def test_write_only_when_changed_and_rotate_keeps_the_last_one():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "gpu-trail.txt")
        gpu_trail.step("decode plain step, token 3")
        gpu_trail.write(path)
        first = os.stat(path).st_mtime_ns
        gpu_trail.write(path)
        assert os.stat(path).st_mtime_ns == first
        os.remove(path)
        gpu_trail.write(path, force=True)
        with open(path) as f:
            assert "decode plain step, token 3" in f.read()
        gpu_trail.rotate(path)
        assert not os.path.exists(path)
        assert os.path.exists(f"{path}.prev")


def _fake_crumbs(done: int, labels: list[str]):
    """Crumb state as if `labels` were queued and the GPU wrote `done`."""
    gpu_crumbs.ON = True
    gpu_crumbs._view = np.array([done, 0, 0, 0], dtype=np.uint32)
    gpu_crumbs._queued.clear()
    for i, label in enumerate(labels, start=1):
        gpu_crumbs._queued.append((i, 100.0 + i, label))


def _restore_crumbs():
    gpu_crumbs.ON = False
    gpu_crumbs._view = None
    gpu_crumbs._queued.clear()


def test_crumbs_name_the_gap_the_gpu_stopped_in():
    _fake_crumbs(2, ["model.layers.46 output (experts done)",
                     "model.layers.47 input (attention done)",
                     "model.layers.47 output (experts done)",
                     "token 211 sampled"])
    try:
        lines = gpu_crumbs.report(now=200.0)
        assert "finished #2 model.layers.47 input (attention done)" in lines[-3]
        assert "STUCK before #3 model.layers.47 output (experts done)" in lines[-2]
        assert "2 crumb(s) not finished" in lines[-2]
        assert lines[-1].endswith("then #4 token 211 sampled")
    finally:
        _restore_crumbs()


def test_crumbs_all_finished_points_past_the_last_one():
    _fake_crumbs(2, ["model.layers.47 output (experts done)", "token 211 sampled"])
    try:
        lines = gpu_crumbs.report(now=200.0)
        assert "finished #2 token 211 sampled" in lines[-2]
        assert "finished every crumb queued" in lines[-1]
    finally:
        _restore_crumbs()


def test_crumbs_none_finished_and_off():
    _fake_crumbs(0, ["model.layers.0 input (attention done)"])
    try:
        assert "has not finished any" in gpu_crumbs.report(now=200.0)[0]
    finally:
        _restore_crumbs()
    assert gpu_crumbs.report() == []
    gpu_crumbs.crumb(object(), "ignored while off")
    assert not gpu_crumbs._queued


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all ok")
