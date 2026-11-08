# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decode's read wait keeps the GPU warm without changing what it returns."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future

from pagedmoe import cache as cache_mod


def _later(futs, delay_s):
    def run():
        for i, f in enumerate(futs):
            time.sleep(delay_s)
            f.set_result(i)

    t = threading.Thread(target=run)
    t.start()
    return t


def test_completed_warm_yields_every_future_and_pings(monkeypatch):
    pings = []
    monkeypatch.setattr(cache_mod, "_gpu_ping", lambda: pings.append(1))
    futs = [Future() for _ in range(4)]
    t = _later(futs, 0.01)
    got = list(cache_mod._completed_warm(futs, 0.001))
    t.join()
    assert sorted(f.result() for f in got) == [0, 1, 2, 3]
    assert len(got) == 4
    assert pings


def test_completed_warm_survives_a_failing_ping(monkeypatch):
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("no gpu")

    monkeypatch.setattr(cache_mod, "_gpu_ping", boom)
    futs = [Future() for _ in range(3)]
    t = _later(futs, 0.01)
    got = list(cache_mod._completed_warm(futs, 0.001))
    t.join()
    assert len(got) == 3
    # The first failure turns pinging off for the rest of the wait.
    assert len(calls) == 1


def test_completed_warm_with_everything_done():
    futs = [Future() for _ in range(2)]
    for i, f in enumerate(futs):
        f.set_result(i)
    assert len(list(cache_mod._completed_warm(futs, 0.001))) == 2
