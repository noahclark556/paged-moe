# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Watchdog: stall exit, pressure exit, leftover sweep."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from pagedmoe import activity, mempressure, watchdog


@pytest.fixture(autouse=True)
def _quiet_activity():
    activity._active = 0
    activity._decoding = 0
    activity._stopping.clear()
    yield
    activity._active = 0
    activity._decoding = 0
    activity._stopping.clear()


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _dog(clock, level, exits, **kw):
    kw.setdefault("stall_s", 60)
    kw.setdefault("pressure_s", 30)
    kw.setdefault("heartbeat_s", 0)
    kw.setdefault("thermal", lambda: "ok")
    return watchdog.Watchdog(
        exit=lambda code, drain: exits.append(code), now=clock, level=level, **kw
    )


def test_stall_exits_70_only_while_a_request_is_open(monkeypatch):
    exits: list[int] = []
    clock = _Clock()
    dog = _dog(clock, lambda: mempressure.NORMAL, exits)
    monkeypatch.setattr(watchdog, "_dump_stacks", lambda: None)
    sampled: list[int] = []
    monkeypatch.setattr(watchdog, "_native_sample", lambda: sampled.append(1) or "/tmp/stall.txt")
    oslog: list[float] = []
    monkeypatch.setattr(watchdog, "_os_gpu_log", lambda s: oslog.append(s) or ("/tmp/os.txt", ["AGX fault"]))
    # Idle engine, old touch: not a stall.
    activity._last = time.monotonic() - 10_000
    dog.check()
    assert exits == [] and sampled == []
    activity.enter()
    activity._last = time.monotonic() - 10_000
    dog.check()
    assert exits == [watchdog.EXIT_STALL]
    assert sampled == [1]
    assert len(oslog) == 1 and oslog[0] > 10_000


def test_drained_exit_waits_for_decode_to_leave(monkeypatch):
    import threading

    exited: list[float] = []
    monkeypatch.setattr(watchdog.os, "_exit", lambda code: exited.append(time.monotonic()))
    activity.decode_enter()

    def _decode():
        while not activity.stopping():
            time.sleep(0.01)
        time.sleep(0.2)
        activity.decode_leave()

    threading.Thread(target=_decode, daemon=True).start()
    t0 = time.monotonic()
    watchdog._exit(watchdog.EXIT_PRESSURE, True)
    assert activity.stopping() and activity.decoding() == 0
    assert 0.15 < exited[0] - t0 < 3.0


def test_heartbeat_reports_growth_since_the_first_beat(monkeypatch, capsys):
    clock = _Clock()
    dog = _dog(clock, lambda: mempressure.NORMAL, [], heartbeat_s=300)
    mem = iter([(20.0, 1.0), (23.5, 2.0)])
    monkeypatch.setattr(watchdog, "_mx_numbers", lambda: next(mem))
    monkeypatch.setattr(watchdog, "_gpu_stats", lambda: "device 70%")
    monkeypatch.setattr(watchdog, "_thermal", lambda: "ok")
    monkeypatch.setattr(mempressure, "compressor_gb", lambda: 0.5)
    for _ in range(2):
        clock.t += 301
        dog.check()
    lines = [l for l in capsys.readouterr().out.splitlines() if "alive:" in l]
    assert len(lines) == 2
    assert "mx active 20.0 GB (+0.0 since first)" in lines[0]
    assert "mx active 23.5 GB (+3.5 since first), cache 2.0 GB (+1.0)" in lines[1]
    assert "gpu device 70%, thermal ok" in lines[1]


def test_native_sample_is_off_when_disabled(monkeypatch):
    monkeypatch.setattr(watchdog.config, "WATCHDOG_SAMPLE_S", 0)
    assert watchdog._native_sample() is None


def test_os_gpu_log_is_off_when_disabled(monkeypatch):
    monkeypatch.setattr(watchdog.config, "WATCHDOG_OSLOG", 0)
    assert watchdog._os_gpu_log(60) == (None, [])


def test_early_warning_prints_the_trail_once_before_the_exit(monkeypatch, capsys):
    exits: list[int] = []
    dog = _dog(_Clock(), lambda: mempressure.NORMAL, exits, stall_s=120)
    monkeypatch.setattr(watchdog, "_gpu_stats", lambda: "device 100%")
    watchdog.gpu_trail.step("decode plain step, token 210")
    activity.enter()
    activity._last = time.monotonic() - 40
    dog.check()
    dog.check()
    out = capsys.readouterr().out
    assert exits == []
    assert out.count("no progress for 40s (exit at 120s). gpu: device 100%") == 1
    assert "decode plain step, token 210" in out
    activity.touch()
    dog.check()
    assert dog._warned is False


@pytest.mark.skipif(sys.platform != "darwin" or not os.path.exists("/usr/bin/sample"),
                    reason="macOS sample only")
def test_native_sample_writes_this_process(monkeypatch, tmp_path):
    monkeypatch.setattr(watchdog.config, "WATCHDOG_SAMPLE_S", 1)
    monkeypatch.setattr(watchdog.config, "INCIDENT_DIR", str(tmp_path / "inc"))
    out = watchdog._native_sample()
    if out is None:
        pytest.skip("sample could not attach in this environment")
    assert out.startswith(str(tmp_path / "inc"))
    assert os.path.getsize(out) > 0


def test_progress_resets_the_stall_clock(monkeypatch):
    exits: list[int] = []
    dog = _dog(_Clock(), lambda: mempressure.NORMAL, exits)
    activity.enter()
    activity._last = time.monotonic() - 59
    dog.check()
    activity.touch()
    dog.check()
    assert exits == []


def test_critical_pressure_exits_71_after_sustained_window():
    exits: list[int] = []
    clock = _Clock()
    lv = {"v": mempressure.CRITICAL}
    dog = _dog(clock, lambda: lv["v"], exits)
    activity.enter()
    activity.touch()
    dog.check()  # starts the window
    clock.t += 20
    dog.check()
    assert exits == []
    clock.t += 15
    dog.check()
    assert exits == [watchdog.EXIT_PRESSURE]


def test_pressure_window_needs_an_open_request_and_resets():
    exits: list[int] = []
    clock = _Clock()
    lv = {"v": mempressure.CRITICAL}
    dog = _dog(clock, lambda: lv["v"], exits)
    # No request open: critical alone never exits.
    for _ in range(5):
        clock.t += 30
        dog.check()
    assert exits == []
    activity.enter()
    activity.touch()
    dog.check()
    clock.t += 20
    lv["v"] = mempressure.NORMAL
    dog.check()  # window resets
    lv["v"] = mempressure.CRITICAL
    clock.t += 20
    dog.check()
    assert exits == []


def test_parse_ps_filters_self_and_unrelated():
    me = 4242
    text = (
        f"{me} 1 /x/python -m pagedmoe.server --model a\n"
        "500 1 /x/python -m pagedmoe.server --model b\n"
        "501 1 /x/.venv/bin/python3.12 /y/pagedmoe/server.py --port 1\n"
        "502 1 /x/python3.12 -I -B -u /t/ga-launch-a/pagedmoe_launch.py pagedmoe.server /t/e --port 2\n"
        "503 1 /x/python3.12 -I /t/ga-launch-b/pagedmoe_launch.py pagedmoe.pretrain_sidecar /t/e\n"
        "600 500 /x/python bench/decode_profile.py\n"
        "650 1 /bin/zsh -c python -m pagedmoe.server; echo done\n"
        "700 123 node ga server\n"
        "bad line\n"
    )
    rows = watchdog.parse_ps(text, me)
    assert rows == [
        (500, 1, "/x/python -m pagedmoe.server --model b"),
        (501, 1, "/x/.venv/bin/python3.12 /y/pagedmoe/server.py --port 1"),
        (502, 1, "/x/python3.12 -I -B -u /t/ga-launch-a/pagedmoe_launch.py pagedmoe.server /t/e --port 2"),
    ]


def test_sweep_ends_orphan_and_leaves_owned_process_alone():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", "pagedmoe.server"]
    )
    try:
        owned = f"{child.pid} {os.getpid()} python -m pagedmoe.server --port 1\n"
        assert watchdog.sweep_leftovers(ps_text=owned) == []
        assert child.poll() is None
        orphan = f"{child.pid} 1 python -m pagedmoe.server --port 1\n"
        assert watchdog.sweep_leftovers(ps_text=orphan) == [child.pid]
        child.wait(timeout=10)
        assert child.returncode is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_sweep_kill_false_only_reports():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", "pagedmoe.server"]
    )
    try:
        orphan = f"{child.pid} 1 python -m pagedmoe.server --port 1\n"
        assert watchdog.sweep_leftovers(kill=False, ps_text=orphan) == []
        assert child.poll() is None
    finally:
        child.kill()
        child.wait(timeout=5)


def test_thermal_warning_paces_decode_and_clears():
    exits: list[int] = []
    clock = _Clock()
    state = ["ok"]
    dog = _dog(clock, lambda: mempressure.NORMAL, exits, thermal=lambda: state[0], pace_ms=30)
    activity.set_pace(0.0)
    activity.enter()
    activity.touch()
    dog.check()
    assert activity.pace() == 0.0
    state[0] = "warning level 2"
    clock.t += 5
    dog.check()
    assert activity.pace() == 0.0, "pmset is read at most every 15 s"
    clock.t += 15
    activity.touch()
    dog.check()
    assert activity.pace() == pytest.approx(0.03)
    activity.leave()
    dog.check()
    assert activity.pace() == 0.0, "no pacing once the request is done"
    assert exits == []


def test_plain_decode_path_touches_every_token(monkeypatch):
    """Resident and dense decode take mlx-lm's own loop, with no expert fetch."""
    import types

    import mlx_lm.generate  # noqa: F401  (real module loads first)

    from pagedmoe import adaptive_prefill, config, server

    closed = []
    seen = []

    def fake_gs(prompt, model, **kw):
        try:
            for i in range(5):
                seen.append(activity._last)
                yield i, None
        finally:
            closed.append(True)

    fake = types.ModuleType("mlx_lm.generate")
    fake.generate_step = fake_gs
    fake.speculative_generate_step = fake_gs
    fake.generation_stream = None
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake)
    monkeypatch.setattr(server, "_adaptive_generate_installed", False)
    monkeypatch.setattr(adaptive_prefill, "enabled", lambda: False)
    monkeypatch.setattr(config, "LOOKUP", False)
    monkeypatch.setattr(config, "VERIFY", False)
    server._install_adaptive_prefill_generate()

    activity.set_pace(0.0)
    activity.enter()
    activity._last = time.monotonic() - 500
    out = []
    gen = fake.generate_step([1, 2, 3], object())
    for tok, _ in gen:
        out.append(tok)
        assert activity.stalled_for() < 1.0, "a decoded token is progress"
        activity._last = time.monotonic() - 500
        if tok == 2:
            break
    gen.close()
    activity.leave()
    assert out == [0, 1, 2]
    assert closed == [True], "stopping decode closes the inner loop at once"
    assert activity.decoding() == 0
