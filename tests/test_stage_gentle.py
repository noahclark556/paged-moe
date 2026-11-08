# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gentle host stages (stage_gentle.py): pause and disk class only while a
stage runs, boost ends both, nothing raises.

Run:
    python tests/test_stage_gentle.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pagedmoe import config, stage_gentle


def _paced_ms() -> float:
    t = time.perf_counter()
    stage_gentle.pace()
    return (time.perf_counter() - t) * 1000


def test_pace_only_while_staging():
    old = (config.STAGE_PACE_MS, config.STAGE_IOPOL, config.STAGE_YIELD_S)
    config.STAGE_PACE_MS, config.STAGE_IOPOL, config.STAGE_YIELD_S = 20.0, "utility", 0.0
    try:
        assert _paced_ms() < 5, "paused with no stage running"
        stage_gentle.begin()
        assert stage_gentle.active()
        assert _paced_ms() >= 15, "no pause during a stage"
        assert stage_gentle.boost(), "boost did not report the running stage"
        assert not stage_gentle.active()
        assert _paced_ms() < 5, "still paused after boost"
        assert not stage_gentle.end(), "end after boost reported a stage"
    finally:
        stage_gentle.end()
        config.STAGE_PACE_MS, config.STAGE_IOPOL, config.STAGE_YIELD_S = old


def test_waits_while_the_user_is_active():
    old = (config.STAGE_PACE_MS, config.STAGE_YIELD_S, stage_gentle.user_idle_seconds)
    config.STAGE_PACE_MS, config.STAGE_YIELD_S = 0.0, 2.0
    clock = {"idle": 0.1}
    stage_gentle.user_idle_seconds = lambda: clock["idle"]
    try:
        stage_gentle.wait_for_pause()  # no stage: never waits
        stage_gentle.begin()
        clock["idle"] = 5.0
        assert _paced_ms() < 20, "waited with the user idle"
        clock["idle"] = 0.1
        import threading

        threading.Timer(0.4, lambda: clock.update(idle=3.0)).start()
        t = time.perf_counter()
        stage_gentle.wait_for_pause()
        waited = time.perf_counter() - t
        assert 0.3 <= waited < 1.5, f"did not hold until the user paused ({waited:.2f}s)"
        clock["idle"] = 0.1
        threading.Timer(0.3, stage_gentle.boost).start()
        t = time.perf_counter()
        stage_gentle.wait_for_pause()
        assert time.perf_counter() - t < 1.0, "boost did not release a waiting stage"
        stage_gentle.begin()
        stage_gentle.user_idle_seconds = lambda: None
        assert _paced_ms() < 20, "an unreadable idle clock blocked the stage"
    finally:
        stage_gentle.end()
        config.STAGE_PACE_MS, config.STAGE_YIELD_S, stage_gentle.user_idle_seconds = old


def test_idle_clock_reads():
    idle = stage_gentle.user_idle_seconds()
    assert idle is None or idle >= 0


def test_bad_policy_is_harmless():
    old = (config.STAGE_IOPOL, config.STAGE_YIELD_S)
    config.STAGE_IOPOL, config.STAGE_YIELD_S = "nonsense", 0.0
    try:
        stage_gentle.begin()
        assert stage_gentle.active()
    finally:
        stage_gentle.end()
        config.STAGE_IOPOL, config.STAGE_YIELD_S = old
    assert not stage_gentle.active()


if __name__ == "__main__":
    test_pace_only_while_staging()
    test_waits_while_the_user_is_active()
    test_idle_clock_reads()
    test_bad_policy_is_harmless()
    print("OK: stage_gentle")
