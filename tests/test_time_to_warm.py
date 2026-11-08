# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Time-to-warm greppable [ttw] log."""

from __future__ import annotations

from unittest.mock import MagicMock

from pagedmoe.time_to_warm import TimeToWarm


def test_ttw_emits_start_prefetch_warm(capsys):
    ttw = TimeToWarm(window=2)
    cache = MagicMock()
    cache.hits = 10
    cache.misses = 10
    cache.bytes_read = 0
    cache.demand_miss_bytes = 1_000_000_000
    cache.disk_wait_s = 1.0

    ttw.on_first_decode(cache, early=True)
    ttw.note_prefetch(12.5, plan_n=40, actuated=True)

    cache.hits = 18
    cache.misses = 12
    cache.demand_miss_bytes = 1_500_000_000
    cache.disk_wait_s = 1.4
    ttw.end_token(cache)
    out = capsys.readouterr().out
    assert "[ttw] start turn=1 early=1 window=2" in out
    assert "[ttw] prefetch turn=1 ms=12.5 plan=40" in out
    assert "[ttw] warm" not in out  # need 2 tokens

    ttw.end_token(cache)
    out2 = capsys.readouterr().out
    assert "[ttw] warm turn=1" in out2
    assert "tok=2/2" in out2
    assert "early=1" in out2
    assert "act=1" in out2
    assert "pref_ms=12.5" in out2
    assert "hit=0.80" in out2  # 8/(8+2)
    assert "miss_gb=0.50" in out2


def test_ttw_works_with_early_off(capsys):
    ttw = TimeToWarm(window=1)
    cache = MagicMock()
    cache.hits = 0
    cache.misses = 0
    cache.bytes_read = 0
    cache.demand_miss_bytes = 0
    cache.disk_wait_s = 0.0
    ttw.on_first_decode(cache, early=False)
    ttw.end_token(cache)
    out = capsys.readouterr().out
    assert "early=0" in out
    assert "act=0" in out
    assert "[ttw] warm" in out


def test_ttw_reset_starts_new_turn(capsys):
    ttw = TimeToWarm(window=1)
    cache = MagicMock()
    cache.hits = cache.misses = 0
    cache.bytes_read = 0
    cache.demand_miss_bytes = 0
    cache.disk_wait_s = 0.0
    ttw.on_first_decode(cache, early=False)
    ttw.end_token(cache)
    ttw.reset()
    ttw.on_first_decode(cache, early=True)
    out = capsys.readouterr().out
    assert "turn=2" in out
