# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How much of the router's mixture a decode keeps, by PRUNE and WAIT_ABOVE.

PRUNE keeps experts within a ratio of the row's winner, so one value means
very different things on different models. GLM-5.2 (8 of 256, flat sigmoid
weights, a cache that holds a few percent of 418 GB) keeps ~0.45 of the mass
at PRUNE 0.7 / WAIT_ABOVE 0.25; MiniMax-M3 keeps 0.50-0.63 at 0.8 / 0.35.

Approximate decode rows are recorded with the experts that were free (no
disk read) when the layer ran, and after a generation the mass a token would
keep is worked out over a grid of (PRUNE, WAIT_ABOVE): free experts always
stay (KEEP_FREE), the rest need ratio >= PRUNE and weight >= WAIT_ABOVE. The
table goes to SIDECAR_DIR/prune_curves/<sha1(realpath)[:16]>.json, merged
with what was there. A host app can map a speed slider through it. Recording is two
list appends per layer; the table is built off the decode thread.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from . import config

PRUNE_GRID = [round(0.05 * i, 2) for i in range(20)]
WAIT_GRID = [round(0.025 * i, 3) for i in range(21)]
# Rows held between flushes, rows a table needs, rows one table is built from.
_MAX_ROWS = 32768
_MIN_ROWS = 2048
_SAMPLE_ROWS = 8192
# Older rows count for at most this many when merging, so the table follows
# a cache size or setting that changed.
_KEEP_ROWS = 100_000

_lock = threading.Lock()
_w: list = []
_res: list = []
_n = 0


def note(w: np.ndarray, res: np.ndarray) -> None:
    """One layer's decode rows: unit-mass weights and which slots were free."""
    global _n
    if not config.PRUNE_CURVE or _n >= _MAX_ROWS:
        return
    with _lock:
        _w.append(w)
        _res.append(res)
        _n += int(w.shape[0])


def path_for(model_path: str) -> Path:
    real = os.path.realpath(os.path.expanduser(model_path))
    key = hashlib.sha1(real.encode()).hexdigest()[:16]
    return Path(config.SIDECAR_DIR).expanduser() / "prune_curves" / f"{key}.json"


def table(w: np.ndarray, res: np.ndarray) -> np.ndarray:
    """Mean kept mass, [len(PRUNE_GRID), len(WAIT_GRID)]."""
    w = w.astype(np.float32)
    res = res.astype(bool)
    top = w.max(axis=-1, keepdims=True)
    waits = np.asarray(WAIT_GRID, np.float32)[:, None, None]
    out = np.empty((len(PRUNE_GRID), len(WAIT_GRID)), np.float64)
    for i, r in enumerate(PRUNE_GRID):
        sel = w >= r * top
        kept = res[None] | (sel[None] & (w[None] >= waits))
        out[i] = (w[None] * kept).sum(axis=-1).mean(axis=-1)
    return out


def _merge(path: Path, real: str, mass: np.ndarray, rows: int) -> None:
    old_rows = 0
    try:
        old = json.loads(path.read_text())
        same = (
            old.get("version") == 1
            and old.get("prune") == PRUNE_GRID
            and old.get("wait_above") == WAIT_GRID
        )
        if same:
            old_rows = min(int(old.get("rows", 0)), _KEEP_ROWS)
            if old_rows > 0:
                prev = np.asarray(old["mass"], np.float64)
                mass = (prev * old_rows + mass * rows) / (old_rows + rows)
    except (OSError, ValueError, KeyError, TypeError):
        old_rows = 0
    data = {
        "version": 1,
        "model": real,
        "rows": old_rows + rows,
        "updated": int(time.time()),
        "prune": PRUNE_GRID,
        "wait_above": WAIT_GRID,
        "mass": np.round(mass, 4).tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)


def _write(model_path: str, w: np.ndarray, res: np.ndarray) -> None:
    try:
        if w.shape[0] > _SAMPLE_ROWS:
            pick = np.random.default_rng().choice(w.shape[0], _SAMPLE_ROWS, replace=False)
            w, res = w[pick], res[pick]
        real = os.path.realpath(os.path.expanduser(model_path))
        _merge(path_for(model_path), real, table(w, res), int(w.shape[0]))
    except Exception as e:
        print(f"[paged-moe] prune curve not saved: {e!r}", flush=True)


def flush(model_path: str | None) -> None:
    """End of a generation: fold what was recorded into the model's table."""
    global _w, _res, _n
    if not model_path or _n < _MIN_ROWS:
        return
    with _lock:
        ws, rs = _w, _res
        _w, _res, _n = [], [], 0
    try:
        K = ws[0].shape[-1]
        pairs = [(a, b) for a, b in zip(ws, rs) if a.shape[-1] == K and a.shape == b.shape]
        w = np.concatenate([a for a, _ in pairs]).reshape(-1, K)
        res = np.concatenate([b for _, b in pairs]).reshape(-1, K)
    except (ValueError, IndexError):
        return
    threading.Thread(target=_write, args=(model_path, w, res), daemon=True).start()


def reset() -> None:
    global _w, _res, _n
    with _lock:
        _w, _res, _n = [], [], 0
    _loaded.clear()


# A table needs this many rows, and this much kept mass between the fastest
# setting and exact, before position() answers. A host that reads the table
# itself should use the same bars so both agree on the slider.
_POS_ROWS = 4096
_POS_SPAN = 0.05
_loaded: dict[str, tuple[float, np.ndarray | None]] = {}


def _load(model_path: str) -> np.ndarray | None:
    path = path_for(model_path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _loaded.get(str(path))
    if hit is not None and hit[0] == mtime:
        return hit[1]
    mass = None
    try:
        data = json.loads(path.read_text())
        if (
            data.get("version") == 1
            and int(data.get("rows", 0)) >= _POS_ROWS
            and data.get("prune") == PRUNE_GRID
            and data.get("wait_above") == WAIT_GRID
        ):
            mass = np.asarray(data["mass"], np.float64)
            if mass.shape != (len(PRUNE_GRID), len(WAIT_GRID)):
                mass = None
    except (OSError, ValueError, KeyError, TypeError):
        mass = None
    _loaded[str(path)] = (mtime, mass)
    return mass


def _at(mass: np.ndarray, prune: float, wait_above: float) -> float:
    def cell(xs, v):
        if v <= xs[0]:
            return 0, 0, 0.0
        if v >= xs[-1]:
            return len(xs) - 1, len(xs) - 1, 0.0
        i = int(np.searchsorted(xs, v, side="right")) - 1
        return i, i + 1, (v - xs[i]) / (xs[i + 1] - xs[i])

    i0, i1, fi = cell(PRUNE_GRID, prune)
    j0, j1, fj = cell(WAIT_GRID, wait_above)
    a = mass[i0, j0] * (1 - fj) + mass[i0, j1] * fj
    b = mass[i1, j0] * (1 - fj) + mass[i1, j1] * fj
    return float(a * (1 - fi) + b * fi)


def position(model_path: str | None, prune: float, wait_above: float) -> float | None:
    """Where (prune, wait_above) sits on a speed slider, 0 fastest to 1 exact:
    the share of what the fastest setting drops that this one keeps. None
    without a usable table."""
    if not model_path:
        return None
    mass = _load(model_path)
    if mass is None:
        return None
    lo = _at(mass, PRUNE_GRID[-1], WAIT_GRID[-1])
    hi = _at(mass, 0.0, 0.0)
    if hi - lo < _POS_SPAN:
        return None
    return min(1.0, max(0.0, (_at(mass, prune, wait_above) - lo) / (hi - lo)))
