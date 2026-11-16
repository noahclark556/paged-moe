# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Working-set margin learned per model on big Macs.

Above 48 GB the machine layer raises the RAM budget but keeps it
WORKING_SET_MARGIN_GB (4.8 GiB) under Metal's recommended working set. That
gap is what 0.68 leaves on the 48 GB M5 Pro, not something measured on the
Mac in question. This measures how close a model really gets and lets the
next load of that model use a smaller gap.

At the end of each turn (off the decode thread):

    over   = process Metal peak - the budget the load planned for
    others = GPU memory in use outside this process (ioreg)
    needed = over + others + SLACK_GB

MLX's recycled-buffer pool is left out of the peak on purpose: pooled buffers
do not count as GPU memory in use (checked with ioreg on a 4 GiB alloc/free).

The margin only moves between MIN_GB and the static 4.8 GiB, so learning can
give back part of the gap but never makes a load smaller than it was without
it. It grows right away when a turn needs more than the margin the load used,
and shrinks at most STEP_GB per session, never under the worst of the last
few sessions. A session only counts toward shrinking once it has prefilled,
decoded MIN_DECODE tokens, and the kernel has not warned about memory.

Files: SIDECAR_DIR/ws_margin/<sha1(realpath + working set)[:16]>.json. A
different Mac or wired-limit setting gets a fresh file.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

from . import config

MIN_GB = 2.0
SLACK_GB = 2.0
STEP_GB = 1.0
MIN_DECODE = 128
KEEP_SESSIONS = 5
STABLE_BYTES = 512 << 20

_GIB = 1 << 30
_lock = threading.Lock()
_session: dict | None = None
_busy = False


def static_gb() -> float:
    from .machine.scale import WORKING_SET_MARGIN_GB

    return float(WORKING_SET_MARGIN_GB)


def path_for(model_path: str, ws: int) -> Path:
    real = os.path.realpath(os.path.expanduser(str(model_path)))
    key = hashlib.sha1(f"{real}|{int(ws)}".encode()).hexdigest()[:16]
    return Path(config.SIDECAR_DIR).expanduser() / "ws_margin" / f"{key}.json"


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) and data.get("version") == 1 else {}
    except (OSError, ValueError):
        return {}


def _clamp(m: float) -> float:
    return round(min(static_gb(), max(MIN_GB, float(m))), 2)


def margin_for(model_path: str, ws: int | None) -> float:
    """Margin in GiB for this model on this Mac. Static when nothing is learned."""
    if not config.WS_MARGIN_LEARN or not ws:
        return static_gb()
    m = _read(path_for(model_path, ws)).get("margin_gb")
    if not isinstance(m, (int, float)):
        return static_gb()
    return _clamp(m)


def begin(model_path: str, ws: int, budget: int, margin_gb: float) -> None:
    """Start measuring a streamed load that ran on a raised budget."""
    global _session
    from . import mempressure

    mempressure.start()
    with _lock:
        _session = {
            "id": f"{os.getpid()}-{int(time.time())}",
            "path": path_for(model_path, ws),
            "model": os.path.realpath(os.path.expanduser(str(model_path))),
            "ws": int(ws),
            "budget": int(budget),
            "margin": float(margin_gb),
            "t0": time.monotonic(),
            "decoded": 0,
            "prefilled": False,
            "needed": float("-inf"),
        }


def end() -> None:
    global _session
    with _lock:
        _session = None


def needed_gb(peak: int, budget: int, others: int) -> float:
    return (int(peak) - int(budget) + max(0, int(others))) / _GIB + SLACK_GB


def next_margin(used: float, needed: float, recent: list[float], counts: bool) -> float:
    """Margin the next load should use.

    `used` is what this session loaded with, `needed` the most it has needed
    so far, `recent` the needed figures of earlier sessions.
    """
    if needed > used:
        return _clamp(needed)
    if not counts:
        return _clamp(used)
    floor = max([needed, *recent])
    return _clamp(max(floor, used - STEP_GB))


def _active() -> int:
    import mlx.core as mx

    return int(mx.get_active_memory())


def note_turn(decode_tokens: int, prefilled: bool) -> None:
    """End of a turn. Cheap here; the ioreg read and the write run on a thread."""
    global _busy
    s = _session
    if s is None or not config.WS_MARGIN_LEARN:
        return
    from .cache import note_peak

    peak = note_peak()
    with _lock:
        if _session is not s:
            return
        s["decoded"] += max(0, int(decode_tokens))
        s["prefilled"] = s["prefilled"] or bool(prefilled)
        if _busy:
            return
        _busy = True
    threading.Thread(target=_settle, args=(s, peak), name="ws-margin", daemon=True).start()


def _settle(s: dict, peak: int) -> None:
    global _busy
    try:
        from . import mempressure

        before = _active()
        in_use = mempressure.gpu_in_use_bytes()
        after = _active()
        # ioreg takes tens of ms; if our own use moved meanwhile, the gap
        # between the two is not other apps. Try again next turn.
        if in_use is None or abs(after - before) > STABLE_BYTES:
            return
        others = max(0, in_use - max(before, after))
        need = needed_gb(peak, s["budget"], others)
        with _lock:
            if need > s["needed"]:
                s["needed"] = need
                s["seen"] = {
                    "peak_gb": round(peak / _GIB, 2),
                    "budget_gb": round(s["budget"] / _GIB, 2),
                    "others_gb": round(others / _GIB, 2),
                }
            need = s["needed"]
            counts = (
                s["prefilled"]
                and s["decoded"] >= MIN_DECODE
                and not mempressure.warned_since(s["t0"])
            )
        _save(s, need, counts)
    except Exception as e:
        print(f"[paged-moe] working-set margin not saved: {e!r}", flush=True)
    finally:
        with _lock:
            _busy = False


def _save(s: dict, need: float, counts: bool) -> None:
    path: Path = s["path"]
    old = _read(path)
    sessions = [x for x in old.get("sessions", []) if isinstance(x, dict)]
    recent = [
        float(x["needed_gb"])
        for x in sessions
        if x.get("id") != s["id"] and isinstance(x.get("needed_gb"), (int, float))
    ][-KEEP_SESSIONS:]
    was = old.get("margin_gb")
    prev = float(was) if isinstance(was, (int, float)) else s["margin"]
    new = next_margin(s["margin"], need, recent, counts)
    if counts or need > s["margin"]:
        sessions = [x for x in sessions if x.get("id") != s["id"]]
        sessions.append({"id": s["id"], "needed_gb": round(need, 2), **s.get("seen", {})})
        sessions = sessions[-KEEP_SESSIONS:]
    data = {
        "version": 1,
        "model": s["model"],
        "working_set_gb": round(s["ws"] / _GIB, 2),
        "margin_gb": new,
        "updated": int(time.time()),
        "sessions": sessions,
    }
    if prev == new and old.get("sessions", []) == sessions:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)
    if prev != new:
        print(
            f"[paged-moe] working-set margin for {Path(s['model']).name}: "
            f"{prev:g} -> {new:g} GiB (needed {need:.2f}), used from the next load",
            flush=True,
        )
