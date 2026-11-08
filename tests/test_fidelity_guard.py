# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Degenerate-decode guard in fidelity.py."""

from __future__ import annotations

import pytest

from pagedmoe import config, fidelity


@pytest.fixture(autouse=True)
def _approx(monkeypatch):
    monkeypatch.setattr(config, "PRUNE", 0.7)
    monkeypatch.setattr(config, "WAIT_ABOVE", 0.2)
    monkeypatch.setattr(config, "FIDELITY_GUARD", True)
    monkeypatch.setattr(config, "FIDELITY_GUARD_RUN", 12)
    monkeypatch.setattr(config, "FIDELITY_GUARD_WINDOW", 32)
    monkeypatch.setattr(config, "FIDELITY_GUARD_DISTINCT", 3)
    monkeypatch.setattr(config, "FIDELITY_GUARD_TOKENS", 10)
    monkeypatch.setattr(config, "FIDELITY_GUARD_STICKY", 2)
    fidelity.begin_turn()
    yield
    fidelity.end_turn()


def _feed(tokens):
    for t in tokens:
        fidelity.observe(t)


def test_varied_text_never_trips():
    _feed(range(1000, 1400))
    assert not fidelity.rescuing()
    assert fidelity.guard_stats() == (0, 0)


def test_two_token_loop_trips_then_releases():
    _feed([7, 8] * 16)
    assert fidelity.rescuing()
    _feed(range(2000, 2010))
    assert not fidelity.rescuing()
    assert fidelity.guard_stats() == (1, 10)


def test_repeat_run_trips_before_window_fills():
    _feed([5] * 12)
    assert fidelity.rescuing()


def test_second_trip_holds_rest_of_turn():
    _feed([7, 8] * 16)
    _feed(range(2000, 2010))
    _feed([7, 8] * 16)
    _feed(range(3000, 3500))
    assert fidelity.rescuing()
    fidelity.end_turn()
    fidelity.begin_turn()
    assert not fidelity.rescuing()


def test_exact_config_does_nothing(monkeypatch):
    monkeypatch.setattr(config, "PRUNE", 0.0)
    monkeypatch.setattr(config, "WAIT_ABOVE", 0.0)
    _feed([7, 8] * 64)
    assert not fidelity.rescuing()


def test_off_switch(monkeypatch):
    monkeypatch.setattr(config, "FIDELITY_GUARD", False)
    _feed([5] * 64)
    assert not fidelity.rescuing()
