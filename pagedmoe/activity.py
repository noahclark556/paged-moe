# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Is the engine busy? Background work waits for a quiet moment.

Requests enter/leave around the whole HTTP handler, decode tokens touch it.
Anything heavy on the CPU (sidecar saves, json, zlib) competes with the read
path for the GIL, so it runs only once nothing has happened for a while.
"""

from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_active = 0
_last = time.monotonic()


def enter() -> None:
    global _active, _last
    with _lock:
        _active += 1
        _last = time.monotonic()


def leave() -> None:
    global _active, _last
    with _lock:
        _active = max(0, _active - 1)
        _last = time.monotonic()


def touch() -> None:
    global _last
    _last = time.monotonic()


_pace = 0.0


def set_pace(seconds: float) -> None:
    """Pause after each decoded token. The watchdog sets it from the thermal state."""
    global _pace
    _pace = max(0.0, float(seconds))


def pace() -> float:
    return _pace


def idle_for() -> float:
    """Seconds since the last request or token, 0 while a request is open."""
    with _lock:
        if _active:
            return 0.0
        return time.monotonic() - _last


def open_requests() -> int:
    return _active


# Shutdown. MLX streams belong to the thread that made them, so a signal
# handler cannot synchronize the generation thread's GPU work itself. It sets
# stop(), decode ends at its next token and the generation thread drains its
# own stream, and the handler waits for decoding() to reach 0 before it exits.
# decoding() counts open generate_step calls (prefill included).
_stopping = threading.Event()
_decoding = 0


def stop() -> None:
    _stopping.set()


def stopping() -> bool:
    return _stopping.is_set()


def decode_enter() -> None:
    global _decoding
    with _lock:
        _decoding += 1


def decode_leave() -> None:
    global _decoding
    with _lock:
        _decoding = max(0, _decoding - 1)


def decoding() -> int:
    return _decoding


def stalled_for() -> float:
    """Seconds a request has been open with nothing touching the engine.

    0 when no request is open. The watchdog reads this; every MoE layer's
    fetch touches, so a hung Metal wait shows up as this number climbing.
    """
    with _lock:
        if not _active:
            return 0.0
        return time.monotonic() - _last
