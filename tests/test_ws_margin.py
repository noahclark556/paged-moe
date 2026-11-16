# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Learned working-set margin: bounds, steps, persistence, loader budget."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pagedmoe import config as cfg
from pagedmoe import loader, mempressure, ws_margin

GIB = 1 << 30
WS = 72 * GIB


@pytest.fixture
def side(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "SIDECAR_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "WS_MARGIN_LEARN", True)
    yield tmp_path
    ws_margin.end()


def _write(model, ws, margin):
    p = ws_margin.path_for(model, ws)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"version": 1, "margin_gb": margin, "sessions": []}))


def test_next_margin_bounds_and_steps():
    nm = ws_margin.next_margin
    assert nm(4.8, 1.0, [], True) == 3.8  # one step per session
    assert nm(3.8, 1.0, [], True) == 2.8
    assert nm(2.5, 0.5, [], True) == 2.0  # never under MIN_GB
    assert nm(4.8, 3.1, [], True) == 3.8  # step, not straight to needed
    assert nm(3.8, 3.1, [], True) == 3.1
    assert nm(3.8, 1.0, [3.5, 2.0], True) == 3.5  # worst recent session holds it
    assert nm(3.0, 1.0, [], False) == 3.0  # session does not count yet
    assert nm(2.0, 3.4, [], False) == 3.4  # needing more grows right away
    assert nm(3.0, 9.0, [], True) == 4.8  # never above the static margin


def test_margin_for_reads_and_clamps(side):
    m = "/models/a"
    assert ws_margin.margin_for(m, WS) == 4.8
    assert ws_margin.margin_for(m, None) == 4.8
    _write(m, WS, 3.0)
    assert ws_margin.margin_for(m, WS) == 3.0
    assert ws_margin.margin_for(m, WS + GIB) == 4.8  # another working set starts fresh
    assert ws_margin.margin_for("/models/b", WS) == 4.8
    _write(m, WS, 0.5)
    assert ws_margin.margin_for(m, WS) == 2.0
    _write(m, WS, 9.0)
    assert ws_margin.margin_for(m, WS) == 4.8
    cfg.WS_MARGIN_LEARN = False
    _write(m, WS, 3.0)
    assert ws_margin.margin_for(m, WS) == 4.8


def _settle(monkeypatch, *, peak, budget, in_use, active, margin, decoded=500,
            prefilled=True, warned=False):
    monkeypatch.setattr(mempressure, "gpu_in_use_bytes", lambda: in_use)
    monkeypatch.setattr(mempressure, "warned_since", lambda t: warned)
    monkeypatch.setattr(mempressure, "start", lambda: True)
    monkeypatch.setattr(ws_margin, "_active", lambda: active)
    ws_margin.begin("/models/a", WS, budget, margin)
    s = ws_margin._session
    s["decoded"], s["prefilled"] = decoded, prefilled
    ws_margin._settle(s, peak)
    return s


def test_moving_active_skips_the_sample(side, monkeypatch):
    budget = 60 * GIB
    reads = iter([40 * GIB, 30 * GIB])
    monkeypatch.setattr(mempressure, "gpu_in_use_bytes", lambda: 41 * GIB)
    monkeypatch.setattr(mempressure, "warned_since", lambda t: False)
    monkeypatch.setattr(mempressure, "start", lambda: True)
    monkeypatch.setattr(ws_margin, "_active", lambda: next(reads))
    ws_margin.begin("/models/a", WS, budget, 4.8)
    s = ws_margin._session
    s["decoded"], s["prefilled"] = 500, True
    ws_margin._settle(s, budget)
    assert s["needed"] == float("-inf")
    assert not ws_margin.path_for("/models/a", WS).exists()


def test_quiet_session_shrinks_one_step(side, monkeypatch):
    budget = 60 * GIB
    # Peak 3 GiB under budget, 0.5 GiB of other GPU use: needs 1.5 + 0 -> MIN.
    _settle(monkeypatch, peak=budget - 3 * GIB, budget=budget,
            in_use=40 * GIB + GIB // 2, active=40 * GIB, margin=4.8)
    assert ws_margin.margin_for("/models/a", WS) == 3.8
    data = json.loads(ws_margin.path_for("/models/a", WS).read_text())
    assert data["sessions"][-1]["needed_gb"] == pytest.approx(-0.5, abs=0.01)


def test_session_that_does_not_count_holds(side, monkeypatch):
    budget = 60 * GIB
    for kw in ({"decoded": 10}, {"prefilled": False}, {"warned": True}):
        _settle(monkeypatch, peak=budget - 3 * GIB, budget=budget,
                in_use=40 * GIB, active=40 * GIB, margin=4.8, **kw)
        assert ws_margin.margin_for("/models/a", WS) == 4.8, kw


def test_tight_session_grows_back(side, monkeypatch):
    _write("/models/a", WS, 2.0)
    budget = 62 * GIB
    # Peak 0.5 GiB over budget and 1 GiB of others: needs 3.5.
    _settle(monkeypatch, peak=budget + GIB // 2, budget=budget,
            in_use=41 * GIB, active=40 * GIB, margin=2.0, warned=True)
    assert ws_margin.margin_for("/models/a", WS) == 3.5


def test_pooled_buffers_do_not_count_as_others(side, monkeypatch):
    budget = 60 * GIB
    # in_use below active (pool returned): others clamps to 0, not negative.
    _settle(monkeypatch, peak=budget - 3 * GIB, budget=budget,
            in_use=38 * GIB, active=40 * GIB, margin=4.8)
    data = json.loads(ws_margin.path_for("/models/a", WS).read_text())
    assert data["sessions"][-1]["needed_gb"] == pytest.approx(-1.0, abs=0.01)


def test_loader_uses_learned_margin(side, monkeypatch):
    ram = 96 * GIB
    monkeypatch.setattr(cfg, "RAM_FRACTION", 0.70)
    monkeypatch.setattr(cfg, "RESIDENT_RAM_FRACTION", 0.68)
    monkeypatch.setattr(mempressure, "gpu_working_set_bytes", lambda: WS)
    budget = int(0.70 * ram)
    m = "/models/a"

    got, margin, lift = loader.learned_budget(budget, ram, m)
    assert (margin, lift) == (4.8, 0)
    assert got == loader.raised_budget(budget, ram)

    _write(m, WS, 2.0)
    got, margin, lift = loader.learned_budget(budget, ram, m)
    assert margin == 2.0
    # 72 GiB working set - 2 GiB = 70 GiB, inside the OS bound (0.78 at 96).
    assert abs(got - (WS - 2 * GIB)) < 2e6
    assert lift == got - loader.raised_budget(budget, ram)
    assert lift > 2 * GIB


def test_loader_learned_margin_keeps_os_bound(side, monkeypatch):
    ram = 96 * GIB
    big_ws = int(90 * GIB)  # e.g. a raised wired limit
    monkeypatch.setattr(cfg, "RAM_FRACTION", 0.78)
    monkeypatch.setattr(cfg, "RESIDENT_RAM_FRACTION", 0.68)
    monkeypatch.setattr(mempressure, "gpu_working_set_bytes", lambda: big_ws)
    _write("/models/a", big_ws, 2.0)
    got, _, _ = loader.learned_budget(int(0.78 * ram), ram, "/models/a")
    assert got <= int(0.78 * ram) + 1


def test_loader_without_working_set_falls_back(side, monkeypatch):
    ram = 96 * GIB
    monkeypatch.setattr(cfg, "RAM_FRACTION", 0.70)
    monkeypatch.setattr(cfg, "RESIDENT_RAM_FRACTION", 0.68)
    monkeypatch.setattr(mempressure, "gpu_working_set_bytes", lambda: None)
    got, margin, lift = loader.learned_budget(int(0.70 * ram), ram, "/models/a")
    assert got == int(0.68 * ram)
    assert (margin, lift) == (4.8, 0)


def test_turn_without_session_is_free(side):
    ws_margin.end()
    ws_margin.note_turn(100, True)
    assert not (side / "ws_margin").exists()
