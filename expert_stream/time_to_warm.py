# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Greppable ``[ttw]`` timer for early-decode A/B.

Wall time from first decode through the early window. Works with early on
or off; ``grep '[ttw]'`` the server log to compare.
"""

from __future__ import annotations

import time


def _ttw_print(msg: str) -> None:
    # Always emit - not gated on SIDECAR_DEBUG - so A/B tails stay reliable.
    print(f"[ttw] {msg}", flush=True)


class TimeToWarm:
    """Per-turn timer: first decode -> early window complete."""

    def __init__(self, window: int):
        self.window = max(1, int(window))
        self._turn = 0
        self.reset()

    def reset(self) -> None:
        """Prefill / turn boundary: next decode starts a fresh measurement."""
        self._t0: float | None = None
        self._tokens = 0
        self._done = False
        self._snap: dict | None = None
        self.pref_ms = 0.0
        self.early = 0
        self.act = 0
        self.plan_n = 0

    def on_first_decode(self, cache, *, early: bool) -> None:
        """Call once at leave_prefill / first MoE layer of first decode token."""
        if self._t0 is not None:
            return
        self._turn += 1
        self._t0 = time.perf_counter()
        self.early = 1 if early else 0
        self._snap = {
            "hits": int(getattr(cache, "hits", 0) or 0),
            "misses": int(getattr(cache, "misses", 0) or 0),
            "bytes": int(getattr(cache, "bytes_read", 0) or 0),
            "demand": int(getattr(cache, "demand_miss_bytes", 0) or 0),
            "disk": float(getattr(cache, "disk_wait_s", 0.0) or 0.0),
        }
        _ttw_print(
            f"start turn={self._turn} early={self.early} window={self.window}"
        )

    def note_prefetch(self, ms: float, plan_n: int, *, actuated: bool) -> None:
        self.pref_ms = float(ms)
        self.plan_n = int(plan_n)
        self.act = 1 if actuated else 0
        if actuated:
            _ttw_print(
                f"prefetch turn={self._turn} ms={self.pref_ms:.1f} plan={self.plan_n}"
            )

    def end_token(self, cache) -> None:
        if self._t0 is None or self._done:
            return
        self._tokens += 1
        if self._tokens < self.window:
            return
        self._done = True
        elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        # Compare warm ms with early=0 vs early=1 on the same prompt. Hit rate
        # alone lies when the window is short.
        snap = self._snap or {}
        hits = int(getattr(cache, "hits", 0) or 0) - int(snap.get("hits", 0))
        misses = int(getattr(cache, "misses", 0) or 0) - int(snap.get("misses", 0))
        total = hits + misses
        hit = (hits / total) if total else 0.0
        miss_gb = (
            int(getattr(cache, "demand_miss_bytes", 0) or 0)
            - int(snap.get("demand", 0))
        ) / 1e9
        disk_s = float(getattr(cache, "disk_wait_s", 0.0) or 0.0) - float(
            snap.get("disk", 0.0)
        )
        _ttw_print(
            f"warm turn={self._turn} ms={elapsed_ms:.1f} "
            f"tok={self._tokens}/{self.window} "
            f"hit={hit:.2f} miss_gb={miss_gb:.2f} disk_s={disk_s:.2f} "
            f"early={self.early} act={self.act} pref_ms={self.pref_ms:.1f} "
            f"plan={self.plan_n}"
        )
