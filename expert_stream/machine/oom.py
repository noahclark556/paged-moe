# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Software RAM envelope so a sandbox OOM fires on a larger host.

Metal still runs on the host GPU. We cannot make macOS jetsam a 16 GB process
on a 48 GB Mac. What we can do is:

  1. Size cache/headroom from the *simulated* RAM (loader already did this
     via EXPERT_STREAM_SIMULATE_RAM_GB).
  2. Raise EnvelopeOOM when backbone + min cache + headroom exceeds the
     fake budget, instead of allocating into the host's extra RAM.
  3. Cap mlx.set_memory_limit at the simulated budget so Metal's soft
     ceiling matches the envelope.

Larger-than-host machines cannot be OOM-tested here; those rows are estimates.
"""

from __future__ import annotations

import os

from .types import Envelope


class EnvelopeOOM(MemoryError):
    """This checkpoint does not fit the (possibly simulated) RAM envelope."""


def envelope_strict() -> bool:
    raw = os.environ.get("EXPERT_STREAM_ENVELOPE_STRICT", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    return bool(os.environ.get("EXPERT_STREAM_SIMULATE_RAM_GB", "").strip())


def enforce_fit(
    *,
    backbone_bytes: int,
    headroom_bytes: int,
    budget_bytes: int,
    min_cache_bytes: int = 1 << 30,
    name: str = "model",
    ram_gb: float | None = None,
) -> None:
    """Raise EnvelopeOOM when streamed load would have a sub-1 GB expert cache."""
    room = int(budget_bytes) - int(backbone_bytes) - int(headroom_bytes)
    if room >= int(min_cache_bytes):
        return
    budget_gb = budget_bytes / 1e9
    backbone_gb = backbone_bytes / 1e9
    headroom_gb = headroom_bytes / 1e9
    ram_s = f"{ram_gb:g} GB " if ram_gb else ""
    raise EnvelopeOOM(
        f"{name} does not fit this {ram_s}envelope: backbone {backbone_gb:.1f} GB "
        f"+ headroom {headroom_gb:.1f} GB leaves {room / 1e9:.1f} GB for the expert "
        f"cache (budget {budget_gb:.1f} GB). This would OOM on the target Mac."
    )


def install_metal_cap(envelope: Envelope) -> None:
    """Soft-cap mlx at the envelope budget. Best-effort; missing mlx is fine."""
    budget = int(envelope.ram_budget_gb * 1e9)
    extra = int(envelope.headroom_gb * (1 << 30))
    try:
        import mlx.core as mx
    except Exception:
        return
    try:
        if hasattr(mx, "set_memory_limit"):
            mx.set_memory_limit(budget + extra)
    except Exception:
        pass
