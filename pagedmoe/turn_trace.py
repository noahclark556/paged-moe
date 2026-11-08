# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One JSON line per chat request: where its wall time went.

Set ``PAGED_MOE_TRACE=/path/turns.jsonl`` to turn it on. Off costs nothing
past one ``if``. The generation worker runs one request at a time, but HTTP
requests overlap: a client that aborts a stream sends its next request while
the worker is still unwinding the old one. So each HTTP thread owns its
record, records queue in arrival order, and the worker takes the oldest one
when it starts on a request (``bind_model``, called from the prompt-cache
fetch).

Times are seconds from the request's arrival. Expert cache counters are
snapshotted at phase boundaries, so ``decode`` bytes and disk wait are the
decode phase only, not the whole process.
"""

from __future__ import annotations

import json
import os
import threading
import time

_PATH = os.environ.get("PAGED_MOE_TRACE", "").strip()
_lock = threading.Lock()
_http = threading.local()
_queue: list = []
_cur: dict | None = None
_model = None

_COUNTERS = (
    "hits",
    "misses",
    "bytes_read",
    "demand_miss_bytes",
    "disk_wait_s",
    "materialize_s",
    "spec_wasted_bytes",
    "prefill_layers",
    "skipped_waits",
    "closed_ok",
    "closed_replay",
    "stitch_forwards",
    "full_stitch_tokens",
)


def enabled() -> bool:
    return bool(_PATH)


def _now(rec: dict) -> float:
    return time.perf_counter() - rec["_t0"]


def _snap() -> dict:
    c = getattr(_model, "_pagedmoe_cache", None) if _model is not None else None
    if c is None:
        return {}
    out = {}
    for k in _COUNTERS:
        try:
            out[k] = float(getattr(c, k, 0) or 0)
        except (TypeError, ValueError):
            out[k] = 0.0
    slab = getattr(c, "slab", None)
    out["slab_slots"] = float(getattr(slab, "slots", 0) or 0) if slab is not None else 0.0
    return out


def _delta(a: dict, b: dict) -> dict:
    d = {k: round(b.get(k, 0.0) - a.get(k, 0.0), 4) for k in _COUNTERS}
    d["read_gb"] = round(d.pop("bytes_read") / 1e9, 3)
    d["demand_gb"] = round(d.pop("demand_miss_bytes") / 1e9, 3)
    d["spec_wasted_gb"] = round(d.pop("spec_wasted_bytes") / 1e9, 3)
    tot = d["hits"] + d["misses"]
    d["hit_rate"] = round(d["hits"] / tot, 4) if tot else None
    return d


def begin(body: dict, model=None) -> None:
    """HTTP thread, before the request is tokenized."""
    global _model
    if not _PATH:
        return
    if model is not None:
        _model = model
    msgs = body.get("messages") or []
    rec = {
        "t": round(time.time(), 3),
        # Name only: the client may send a checkpoint path.
        "model": os.path.basename(str(body.get("model") or "").rstrip("/")) or None,
        "n_messages": len(msgs),
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens"),
        "warm": bool(body.get("warm_prefix") or body.get("warmPrefix")),
        "tools": len(body.get("tools") or []),
        "temperature": body.get("temperature"),
        "prefill": [],
        "events": [],
        "_t0": time.perf_counter(),
        "_marks": {"begin": _snap()},
    }
    _http.rec = rec
    with _lock:
        _queue.append(rec)


def bind_model(model) -> None:
    """Generation worker, at the start of a request: take the oldest queued record."""
    global _cur, _model
    if not _PATH:
        return
    if model is not None:
        _model = model
    with _lock:
        _cur = _queue.pop(0) if _queue else None
    if _cur is not None:
        _cur["queued_s"] = round(_now(_cur), 4)
        _cur["_marks"]["begin"] = _snap()


def note(**kw) -> None:
    if _cur is not None:
        _cur.update(kw)


def event(name: str, **kw) -> None:
    """Something that happened mid-request (cold miss, slab yield, OOM)."""
    if _cur is not None:
        _cur["events"].append({"at": round(_now(_cur), 3), "name": name, **kw})


def mark(name: str) -> None:
    """First time only: offset of a phase boundary plus a counter snapshot."""
    if _cur is None or name in _cur:
        return
    _cur[name] = round(_now(_cur), 4)
    _cur["_marks"][name] = _snap()


def prefill_chunk(tokens: int, seconds: float, read_gb: float, blocked: float, copy: float, kv: int) -> None:
    if _cur is not None:
        _cur["prefill"].append(
            {
                "tok": int(tokens),
                "kv": int(kv),
                "s": round(seconds, 3),
                "gb": round(read_gb, 3),
                "blocked": round(blocked, 3),
                "copy": round(copy, 3),
            }
        )


def end(error: str | None = None) -> None:
    """HTTP thread, after the response finished (or failed).

    A record the worker never started is dropped from the queue here. One the
    worker is still unwinding (client went away mid-stream) is written as it
    stands; its late notes are lost.
    """
    global _cur
    rec = getattr(_http, "rec", None)
    if rec is None:
        return
    _http.rec = None
    with _lock:
        if rec in _queue:
            _queue.remove(rec)
            rec["never_started"] = True
    if _cur is rec:
        _cur = None
    rec["end"] = round(_now(rec), 4)
    end_snap = _snap()
    if error:
        rec["error"] = error
    marks = rec.pop("_marks")
    rec.pop("_t0", None)
    b = marks.get("begin", {})
    if "first_token" in marks:
        rec["phase_prefill"] = _delta(b, marks["first_token"])
        last = marks.get("decode_end", end_snap)
        rec["phase_decode"] = _delta(marks["first_token"], last)
    else:
        rec["phase_prefill"] = _delta(b, end_snap)
    rec["slab_slots"] = [b.get("slab_slots"), end_snap.get("slab_slots")]
    c = getattr(_model, "_pagedmoe_cache", None) if _model is not None else None
    rec["stitch_full"] = bool(getattr(c, "stitched", False)) if c is not None else False
    try:
        from . import config

        # Read at the end: live_config can move them mid-session.
        rec["prune"] = float(config.PRUNE)
        rec["wait_above"] = float(config.WAIT_ABOVE)
    except Exception:
        pass
    try:
        import mlx.core as mx

        from .cache import note_peak

        rec["mx_active_gb"] = round(mx.get_active_memory() / 1e9, 3)
        rec["mx_peak_gb"] = round(note_peak() / 1e9, 3)
    except Exception:
        pass
    line = json.dumps(rec, separators=(",", ":"), default=str)
    with _lock:
        try:
            os.makedirs(os.path.dirname(_PATH) or ".", exist_ok=True)
            with open(_PATH, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
