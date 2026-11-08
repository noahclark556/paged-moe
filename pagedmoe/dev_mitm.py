# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hook for a dev-only activation debugger.

The debugger itself is not part of this repo. With ``PAGED_MOE_MITM=0`` (the
default) or the module missing, every entry point here is a no-op: one
import try, no wrap, no hot-path work. A release build ignores the variable
(see release.py).
"""

from __future__ import annotations

from typing import Any, Optional

from . import release

_TRUE = {"1", "true", "yes", "on"}

_mod = None
_tried = False


def _env_on() -> bool:
    raw = (release.env("PAGED_MOE_MITM") or "").strip().lower()
    return raw in _TRUE


def _load():
    """Import private.activation_mitm once. None when off or unavailable."""
    global _mod, _tried
    if _tried:
        return _mod
    _tried = True
    if not _env_on():
        _mod = None
        return None
    try:
        from private import activation_mitm as m  # type: ignore

        _mod = m if m.enabled() else None
    except ImportError:
        _mod = None
    return _mod


def maybe_install(model: Any, ring: Any = None) -> Optional[Any]:
    m = _load()
    if m is None:
        return None
    return m.install(model, ring)


def begin_response(model: Any, **meta: Any) -> None:
    m = _load()
    if m is not None:
        m.begin_response(model, **meta)


def end_response(model: Any) -> None:
    m = _load()
    if m is not None:
        m.end_response(model)


def active() -> bool:
    return _load() is not None
