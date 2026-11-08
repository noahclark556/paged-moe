# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Early-decode warm pool: train + prefetch, quality-safe."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from pagedmoe import config
from pagedmoe.early_decode import EarlyDecodeWarm


def test_early_warm_disabled_is_noop():
    saved = config.EARLY_DECODE, config.EARLY_DECODE_TRAIN
    try:
        config.EARLY_DECODE = False
        config.EARLY_DECODE_TRAIN = False
        # Construction still allowed when flags are off only if caller attaches;
        # patch_model skips attach. Direct construct with flags off should not
        # start a bg thread that matters.
        w = EarlyDecodeWarm(
            "test-model",
            [f"L{i}" for i in range(4)],
            n_experts=8,
            expert_bytes=1 << 20,
            slot_id="test_early_off",
        )
        assert w.enabled is False
        assert w.train is False
        cache = MagicMock()
        w.maybe_warm(cache)
        cache.prefetch_many.assert_not_called()
        w.close()
    finally:
        config.EARLY_DECODE, config.EARLY_DECODE_TRAIN = saved


def test_train_accumulates_and_builds_plan(tmp_path, monkeypatch):
    saved = (
        config.EARLY_DECODE,
        config.EARLY_DECODE_TRAIN,
        config.EARLY_DECODE_TOKENS,
        config.EARLY_DECODE_GB,
        config.SIDECAR_DIR,
    )
    try:
        monkeypatch.setattr(config, "SIDECAR_DIR", str(tmp_path))
        config.EARLY_DECODE = True
        config.EARLY_DECODE_TRAIN = True
        config.EARLY_DECODE_TOKENS = 4
        config.EARLY_DECODE_GB = 0.01  # ~10 MB
        keys = [f"L{i}" for i in range(3)]
        w = EarlyDecodeWarm(
            "test-model",
            keys,
            n_experts=16,
            expert_bytes=2 << 20,
            slot_id="test_early_train",
        )
        # Simulate 4 early tokens demanding the same few experts.
        for _ in range(4):
            w.drop_token()
            w.note_demand("L0", [1, 2])
            w.note_demand("L1", [3])
            w.note_demand("L2", [1])
            w.end_token()
        w._rebuild_plan()
        assert w._plan_experts > 0
        assert any(k == "L0" for k, _ in w._plan)
        # Prefetch path only.
        cache = MagicMock()
        w._warmed = False
        w.maybe_warm(cache)
        cache.prefetch_many.assert_called_once()
        batch = cache.prefetch_many.call_args[0][0]
        assert isinstance(batch, list)
        # Second call in same turn is a no-op.
        w.maybe_warm(cache)
        assert cache.prefetch_many.call_count == 1
        w._dirty = True
        w._persist()
        assert w._path.is_file(), w._path
        w.close()
    finally:
        (
            config.EARLY_DECODE,
            config.EARLY_DECODE_TRAIN,
            config.EARLY_DECODE_TOKENS,
            config.EARLY_DECODE_GB,
            config.SIDECAR_DIR,
        ) = saved


def test_train_stops_after_window(tmp_path, monkeypatch):
    saved = (
        config.EARLY_DECODE,
        config.EARLY_DECODE_TRAIN,
        config.EARLY_DECODE_TOKENS,
        config.SIDECAR_DIR,
    )
    try:
        monkeypatch.setattr(config, "SIDECAR_DIR", str(tmp_path))
        config.EARLY_DECODE = False
        config.EARLY_DECODE_TRAIN = True
        config.EARLY_DECODE_TOKENS = 2
        w = EarlyDecodeWarm(
            "m",
            ["L0"],
            n_experts=4,
            expert_bytes=1024,
            slot_id="test_window",
        )
        for i in range(5):
            w.note_demand("L0", [i % 4])
            w.end_token()
        # Only first 2 tokens should have counted.
        assert w._tokens_in_window == 2
        assert w._episodes == 1
        assert w._scored_episodes == 1
        assert float(w._counts.sum()) > 0
        w.close()
    finally:
        (
            config.EARLY_DECODE,
            config.EARLY_DECODE_TRAIN,
            config.EARLY_DECODE_TOKENS,
            config.SIDECAR_DIR,
        ) = saved


def test_episode_log_cover_climbs(tmp_path, monkeypatch, capsys):
    """After an empty plan episode, training should raise cover on the next."""
    saved = (
        config.EARLY_DECODE,
        config.EARLY_DECODE_TRAIN,
        config.EARLY_DECODE_TOKENS,
        config.EARLY_DECODE_GB,
        config.SIDECAR_DIR,
    )
    try:
        monkeypatch.setattr(config, "SIDECAR_DIR", str(tmp_path))
        config.EARLY_DECODE = True
        config.EARLY_DECODE_TRAIN = True
        config.EARLY_DECODE_TOKENS = 2
        config.EARLY_DECODE_GB = 1.0
        keys = ["L0", "L1"]
        w = EarlyDecodeWarm(
            "m",
            keys,
            n_experts=8,
            expert_bytes=1 << 20,
            slot_id="test_cover",
        )
        # Episode 1: no plan yet -> cover 0, but counts fill.
        w.drop_token()
        for _ in range(2):
            w.note_demand("L0", [1, 2])
            w.note_demand("L1", [3])
            w.end_token()
        assert w._scored_episodes == 1
        assert w._cover_ema == 0.0
        # Episode 2: plan exists from rebuild; same demand -> cover > 0.
        cache = MagicMock()
        w.drop_token()
        w.maybe_warm(cache)
        assert w._did_prefetch or w._plan_experts > 0
        for _ in range(2):
            w.note_demand("L0", [1, 2])
            w.note_demand("L1", [3])
            w.end_token()
        assert w._scored_episodes == 2
        assert w._cover_ema > 0.0
        w.close()
    finally:
        (
            config.EARLY_DECODE,
            config.EARLY_DECODE_TRAIN,
            config.EARLY_DECODE_TOKENS,
            config.EARLY_DECODE_GB,
            config.SIDECAR_DIR,
        ) = saved


def test_prefetch_failure_does_not_raise():
    saved = config.EARLY_DECODE, config.EARLY_DECODE_TRAIN
    try:
        config.EARLY_DECODE = True
        config.EARLY_DECODE_TRAIN = False
        w = EarlyDecodeWarm(
            "m",
            ["L0", "L1"],
            n_experts=8,
            expert_bytes=1 << 20,
            slot_id="test_pf_fail",
        )
        w._plan = [("L0", [1, 2])]
        w._plan_experts = 2
        cache = MagicMock()
        cache.prefetch_many.side_effect = RuntimeError("boom")
        w.maybe_warm(cache)  # must not raise
        w.close()
    finally:
        config.EARLY_DECODE, config.EARLY_DECODE_TRAIN = saved


def test_governor_holds_when_emas_below_bar(tmp_path, monkeypatch):
    """Weak cover/prec after warmup must stop prefetch; train still runs."""
    saved = (
        config.EARLY_DECODE,
        config.EARLY_DECODE_TRAIN,
        config.EARLY_DECODE_MIN_COVER,
        config.EARLY_DECODE_MIN_PREC,
        config.EARLY_DECODE_MIN_EPISODES,
        config.EARLY_DECODE_HOLD_STREAK,
        config.EARLY_DECODE_CLEAR_STREAK,
        config.EARLY_DECODE_DECAY,
        config.SIDECAR_DIR,
    )
    try:
        monkeypatch.setattr(config, "SIDECAR_DIR", str(tmp_path))
        config.EARLY_DECODE = True
        config.EARLY_DECODE_TRAIN = True
        config.EARLY_DECODE_MIN_COVER = 0.5
        config.EARLY_DECODE_MIN_PREC = 0.5
        config.EARLY_DECODE_MIN_EPISODES = 2
        config.EARLY_DECODE_HOLD_STREAK = 2
        config.EARLY_DECODE_CLEAR_STREAK = 2
        config.EARLY_DECODE_DECAY = 1.0
        w = EarlyDecodeWarm(
            "m",
            ["L0"],
            n_experts=8,
            expert_bytes=1 << 20,
            slot_id="test_hold",
        )
        w._plan = [("L0", [0, 1, 2, 3])]
        w._plan_experts = 4
        w._scored_episodes = 2
        w._cover_ema = 0.05
        w._prec_ema = 0.05
        w._actuating = True
        w._bad_streak = 0
        w._update_governor()
        assert w._actuating is True  # first bad: streak only
        assert w._bad_streak == 1
        w._update_governor()
        assert w._actuating is False
        assert w._bad_streak == 0

        cache = MagicMock()
        w._warmed = False
        w.maybe_warm(cache)
        cache.prefetch_many.assert_not_called()
        # Plan still snapshotted for scoring.
        assert len(w._plan_set) == 4

        # Recovery: clear streak then actuate again.
        w._cover_ema = 0.8
        w._prec_ema = 0.8
        w._update_governor()
        assert w._actuating is False
        assert w._good_streak == 1
        w._update_governor()
        assert w._actuating is True
        w._warmed = False
        w.maybe_warm(cache)
        cache.prefetch_many.assert_called_once()
        w.close()
    finally:
        (
            config.EARLY_DECODE,
            config.EARLY_DECODE_TRAIN,
            config.EARLY_DECODE_MIN_COVER,
            config.EARLY_DECODE_MIN_PREC,
            config.EARLY_DECODE_MIN_EPISODES,
            config.EARLY_DECODE_HOLD_STREAK,
            config.EARLY_DECODE_CLEAR_STREAK,
            config.EARLY_DECODE_DECAY,
            config.SIDECAR_DIR,
        ) = saved


def test_count_decay_softens_history(tmp_path, monkeypatch):
    saved = (
        config.EARLY_DECODE,
        config.EARLY_DECODE_TRAIN,
        config.EARLY_DECODE_DECAY,
        config.SIDECAR_DIR,
    )
    try:
        monkeypatch.setattr(config, "SIDECAR_DIR", str(tmp_path))
        config.EARLY_DECODE = False
        config.EARLY_DECODE_TRAIN = True
        config.EARLY_DECODE_DECAY = 0.5
        w = EarlyDecodeWarm(
            "m",
            ["L0"],
            n_experts=4,
            expert_bytes=1024,
            slot_id="test_decay",
        )
        with w._lock:
            w._counts[0, 1] = 10.0
        w._apply_decay()
        assert float(w._counts[0, 1]) == 5.0
        w.close()
    finally:
        (
            config.EARLY_DECODE,
            config.EARLY_DECODE_TRAIN,
            config.EARLY_DECODE_DECAY,
            config.SIDECAR_DIR,
        ) = saved


def test_cold_bootstrap_when_enable_without_train(tmp_path, monkeypatch):
    """EARLY_DECODE on + TRAIN off + no weights -> temporary bootstrap train."""
    saved = (
        config.EARLY_DECODE,
        config.EARLY_DECODE_TRAIN,
        config.SIDECAR_DIR,
    )
    try:
        monkeypatch.setattr(config, "SIDECAR_DIR", str(tmp_path))
        config.EARLY_DECODE = True
        config.EARLY_DECODE_TRAIN = False
        w = EarlyDecodeWarm(
            "m",
            ["L0"],
            n_experts=8,
            expert_bytes=1 << 20,
            slot_id="test_boot_cold",
        )
        assert w._explicit_train is False
        assert w._bootstrap is True
        assert w.train is True
        w.close()
    finally:
        (
            config.EARLY_DECODE,
            config.EARLY_DECODE_TRAIN,
            config.SIDECAR_DIR,
        ) = saved


def test_existing_weights_skip_bootstrap(tmp_path, monkeypatch):
    saved = (
        config.EARLY_DECODE,
        config.EARLY_DECODE_TRAIN,
        config.SIDECAR_DIR,
    )
    try:
        monkeypatch.setattr(config, "SIDECAR_DIR", str(tmp_path))
        config.EARLY_DECODE = True
        config.EARLY_DECODE_TRAIN = True
        w = EarlyDecodeWarm(
            "m",
            ["L0"],
            n_experts=8,
            expert_bytes=1 << 20,
            slot_id="test_boot_warm",
        )
        with w._lock:
            w._counts[0, 1] = 5.0
        w._episodes = 3
        w._dirty = True
        w._persist()
        w.close()

        config.EARLY_DECODE_TRAIN = False
        w2 = EarlyDecodeWarm(
            "m",
            ["L0"],
            n_experts=8,
            expert_bytes=1 << 20,
            slot_id="test_boot_warm",
        )
        assert w2._bootstrap is False
        assert w2.train is False
        assert w2.enabled is True
        assert float(w2._counts[0, 1]) == 5.0
        w2.close()
    finally:
        (
            config.EARLY_DECODE,
            config.EARLY_DECODE_TRAIN,
            config.SIDECAR_DIR,
        ) = saved


def test_bootstrap_finishes_at_high_bars(tmp_path, monkeypatch):
    saved = (
        config.EARLY_DECODE,
        config.EARLY_DECODE_TRAIN,
        config.EARLY_DECODE_BOOTSTRAP_COVER,
        config.EARLY_DECODE_BOOTSTRAP_PREC,
        config.EARLY_DECODE_BOOTSTRAP_MIN_EPISODES,
        config.EARLY_DECODE_BOOTSTRAP_MAX_EPISODES,
        config.SIDECAR_DIR,
    )
    try:
        monkeypatch.setattr(config, "SIDECAR_DIR", str(tmp_path))
        config.EARLY_DECODE = True
        config.EARLY_DECODE_TRAIN = False
        config.EARLY_DECODE_BOOTSTRAP_COVER = 0.4
        config.EARLY_DECODE_BOOTSTRAP_PREC = 0.2
        config.EARLY_DECODE_BOOTSTRAP_MIN_EPISODES = 3
        config.EARLY_DECODE_BOOTSTRAP_MAX_EPISODES = 100
        w = EarlyDecodeWarm(
            "m",
            ["L0"],
            n_experts=8,
            expert_bytes=1 << 20,
            slot_id="test_boot_done",
        )
        assert w._bootstrap is True
        w._scored_episodes = 3
        w._cover_ema = 0.45
        w._prec_ema = 0.30
        assert w._bootstrap_ready() is True
        w._finish_bootstrap("bars")
        assert w._bootstrap is False
        assert w._bootstrap_complete is True
        assert w.train is False
        assert w.enabled is True
        w.close()

        # Reload: no re-bootstrap.
        w2 = EarlyDecodeWarm(
            "m",
            ["L0"],
            n_experts=8,
            expert_bytes=1 << 20,
            slot_id="test_boot_done",
        )
        assert w2._bootstrap_complete is True
        assert w2._bootstrap is False
        assert w2.train is False
        w2.close()
    finally:
        (
            config.EARLY_DECODE,
            config.EARLY_DECODE_TRAIN,
            config.EARLY_DECODE_BOOTSTRAP_COVER,
            config.EARLY_DECODE_BOOTSTRAP_PREC,
            config.EARLY_DECODE_BOOTSTRAP_MIN_EPISODES,
            config.EARLY_DECODE_BOOTSTRAP_MAX_EPISODES,
            config.SIDECAR_DIR,
        ) = saved


def test_explicit_train_never_auto_stops(tmp_path, monkeypatch):
    saved = (
        config.EARLY_DECODE,
        config.EARLY_DECODE_TRAIN,
        config.EARLY_DECODE_BOOTSTRAP_MIN_EPISODES,
        config.SIDECAR_DIR,
    )
    try:
        monkeypatch.setattr(config, "SIDECAR_DIR", str(tmp_path))
        config.EARLY_DECODE = True
        config.EARLY_DECODE_TRAIN = True
        config.EARLY_DECODE_BOOTSTRAP_MIN_EPISODES = 1
        w = EarlyDecodeWarm(
            "m",
            ["L0"],
            n_experts=4,
            expert_bytes=1024,
            slot_id="test_explicit",
        )
        assert w._bootstrap is False
        assert w._explicit_train is True
        assert w.train is True
        w._scored_episodes = 100
        w._cover_ema = 0.99
        w._prec_ema = 0.99
        # Would be ready if bootstrapping, but explicit train ignores it.
        assert w._bootstrap_ready() is True
        assert w.train is True
        w.close()
    finally:
        (
            config.EARLY_DECODE,
            config.EARLY_DECODE_TRAIN,
            config.EARLY_DECODE_BOOTSTRAP_MIN_EPISODES,
            config.SIDECAR_DIR,
        ) = saved
