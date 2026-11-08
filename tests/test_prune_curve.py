# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
import json

import numpy as np
import pytest

from pagedmoe import config, prune_curve


@pytest.fixture(autouse=True)
def _dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SIDECAR_DIR", str(tmp_path))
    monkeypatch.setattr(config, "PRUNE_CURVE", 1)
    prune_curve.reset()
    yield
    prune_curve.reset()


def _rows(n=6000, k=8, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.random((n, k)).astype(np.float32) ** 3
    w /= w.sum(axis=-1, keepdims=True)
    res = rng.random((n, k)) < 0.1
    return w, res


def test_table_ends():
    w, res = _rows()
    t = prune_curve.table(w, res)
    assert t.shape == (len(prune_curve.PRUNE_GRID), len(prune_curve.WAIT_GRID))
    assert t[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert t[-1, -1] < t[0, 0]
    assert np.all(np.diff(t, axis=0) <= 1e-9)
    assert np.all(np.diff(t, axis=1) <= 1e-9)


def test_position_needs_a_table(tmp_path):
    assert prune_curve.position(str(tmp_path / "m"), 0.7, 0.25) is None


def test_position_from_written_table(tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    w, res = _rows()
    prune_curve._write(str(model), w, res)
    data = json.loads(prune_curve.path_for(str(model)).read_text())
    assert data["rows"] == 6000
    assert prune_curve.position(str(model), 0.0, 0.0) == pytest.approx(1.0)
    assert prune_curve.position(str(model), 0.95, 0.5) == pytest.approx(0.0)
    mid = prune_curve.position(str(model), 0.7, 0.25)
    assert 0.0 < mid < 1.0

    prune_curve._write(str(model), *_rows(seed=1))
    assert json.loads(prune_curve.path_for(str(model)).read_text())["rows"] == 12000


def test_flush_needs_enough_rows(tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    w, res = _rows(n=100)
    prune_curve.note(w, res)
    prune_curve.flush(str(model))
    assert not prune_curve.path_for(str(model)).exists()


def test_off_records_nothing(monkeypatch):
    monkeypatch.setattr(config, "PRUNE_CURVE", 0)
    w, res = _rows(n=10)
    prune_curve.note(w, res)
    assert prune_curve._n == 0
