# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Settings a running server can take without a restart.

A restart reloads the backbone, rebuilds the slab cold and drops the RAM
prompt cache, so a settings change used to cost the next turn a cold start.
Only settings every reader looks up per call belong here. Anything read once
at load (FLOW's table, the sidecar heads, EARLY_DECODE, cache sizing) still
needs a restart, and `apply` reports it back so the caller restarts instead.

POST /config  {"env": {"PAGED_MOE_PRUNE": "0.5", ...}}
  -> {"applied": {...}, "rejected": {...}, "live": [...]}
GET  /config  -> {"live": [...], "values": {...}}

Both live settings change logits, so KV computed under the old values must
not be served under the new ones. A restart got that for free by dropping
every cache. Here each applied change bumps an epoch: the next request drops
the in-memory prompt KV (`begin_request`), and a request the change landed in
the middle of stores none of its KV (`kv_stale`).
"""

from __future__ import annotations

import os
import threading

from . import config

# env name -> (config attribute, lowest, highest)
_LIVE: dict[str, tuple[str, float, float]] = {
    "PAGED_MOE_PRUNE": ("PRUNE", 0.0, 1.0),
    "PAGED_MOE_WAIT_ABOVE": ("WAIT_ABOVE", 0.0, 1.0),
    "PAGED_MOE_FIDELITY_TOOL_SCALE": ("FIDELITY_TOOL_SCALE", 0.0, 1.0),
    "PAGED_MOE_FIDELITY_TURN_START": ("FIDELITY_TURN_START", 0.0, 4096.0),
}

_lock = threading.Lock()
_epoch = 0
_begun = 0
# Epoch the request on this (generation) thread started under.
_req = threading.local()


def epoch() -> int:
    return _epoch


def begin_request() -> bool:
    """Call at the start of each request, on the generation thread, before
    any cached KV is looked up. True when settings changed since the last
    request: the caller drops what was cached under the old ones."""
    global _begun
    now = _epoch
    _req.epoch = now
    changed = now != _begun
    _begun = now
    return changed


def kv_stale() -> bool:
    """Settings changed after this thread's request started. Its KV mixes
    both, so it must not be stored anywhere."""
    seen = getattr(_req, "epoch", None)
    return seen is not None and seen != _epoch


def live_keys() -> list[str]:
    return sorted(_LIVE)


def values() -> dict[str, float]:
    return {name: float(getattr(config, attr)) for name, (attr, _, _) in _LIVE.items()}


def _why_not(name: str) -> str | None:
    if name not in _LIVE:
        return "read at load"
    # A prune head on rebases to the new PRUNE / WAIT_ABOVE on its next token
    # (heads/prune.py _sync_base) and earns live use again from there.
    return None


def apply(env: dict) -> dict:
    """Apply what can change live. All or nothing: if any key needs a
    restart, nothing is applied, so the caller never runs a mix of old and
    new settings."""
    parsed: dict[str, tuple[str, float]] = {}
    rejected: dict[str, str] = {}
    for name, raw in (env or {}).items():
        why = _why_not(str(name))
        if why is not None:
            rejected[str(name)] = why
            continue
        attr, lo, hi = _LIVE[name]
        if raw is None or str(raw).strip() == "":
            value = 0.0
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                rejected[name] = "not a number"
                continue
        if not lo <= value <= hi:
            rejected[name] = f"outside [{lo}, {hi}]"
            continue
        parsed[name] = (attr, value)
    if rejected:
        return {"applied": {}, "rejected": rejected, "live": live_keys()}
    global _epoch
    applied = {}
    with _lock:
        for name, (attr, value) in parsed.items():
            old = float(getattr(config, attr))
            setattr(config, attr, value)
            os.environ[name] = repr(value)
            if old != value:
                applied[name] = value
        if applied:
            _epoch += 1
    if applied:
        print(
            "[paged-moe] live settings: "
            + ", ".join(f"{k.removeprefix('PAGED_MOE_')}={v:g}" for k, v in applied.items()),
            flush=True,
        )
    return {"applied": applied, "rejected": {}, "live": live_keys()}
