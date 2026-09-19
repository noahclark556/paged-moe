# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sidecar byte accounting for speculative prefetch heads.

Public package ships ByteLedger only. Actuation gating beyond head enable
flags is left to the caller (wrap defaults off).
"""

from __future__ import annotations


class ByteLedger:
    """Per-head speculative byte accounting (the unit that actually costs).

    On bandwidth-bound decode, value is bytes saved minus bytes spent:

      issued   speculative bytes requested
      used     of those, bytes a layer later computed with
      wasted   issued - used (bandwidth taken from demand misses)

    `net_bytes` is signed on purpose: deficit means stop actuating. `gain`
    cannot say that. Decisions use the *window*, not lifetime totals - else
    early learning debt or a long good history both distort the call. Call
    `reset_window` when the head's configuration changes.
    """

    __slots__ = (
        "expert_bytes",
        "issued",
        "issued_bytes",
        "used",
        "used_bytes",
        "w_issued",
        "w_issued_bytes",
        "w_used",
        "w_used_bytes",
    )

    def __init__(self, expert_bytes: int = 0):
        self.expert_bytes = int(expert_bytes)
        self.issued_bytes = 0
        self.used_bytes = 0
        self.issued = 0
        self.used = 0
        self.w_issued_bytes = 0
        self.w_used_bytes = 0
        self.w_issued = 0
        self.w_used = 0

    def note_issued(self, n: int, expert_bytes: int | None = None) -> None:
        per = int(expert_bytes if expert_bytes is not None else self.expert_bytes)
        n, nb = int(n), int(n) * per
        self.issued += n
        self.issued_bytes += nb
        self.w_issued += n
        self.w_issued_bytes += nb

    def note_used(self, n: int, expert_bytes: int | None = None) -> None:
        per = int(expert_bytes if expert_bytes is not None else self.expert_bytes)
        n, nb = int(n), int(n) * per
        self.used += n
        self.used_bytes += nb
        self.w_used += n
        self.w_used_bytes += nb

    def reset_window(self) -> None:
        """Start a fresh judgement window; lifetime totals are untouched."""
        self.w_issued_bytes = 0
        self.w_used_bytes = 0
        self.w_issued = 0
        self.w_used = 0

    @property
    def window_issued(self) -> int:
        return self.w_issued

    @property
    def window_precision(self) -> float:
        return (self.w_used / self.w_issued) if self.w_issued else 0.0

    @property
    def window_net_bytes(self) -> int:
        wasted = max(0, self.w_issued_bytes - self.w_used_bytes)
        return self.w_used_bytes - wasted

    @property
    def wasted_bytes(self) -> int:
        return max(0, self.issued_bytes - self.used_bytes)

    @property
    def precision(self) -> float:
        return (self.used / self.issued) if self.issued else 0.0

    @property
    def net_bytes(self) -> int:
        """Useful bytes minus wasted bytes.

        A correct speculative read costs the same bytes the demand miss would
        have cost, so it is not a saving in *bytes* - it is a saving in
        *latency*, because the read overlaps work instead of blocking a layer.
        Charging it at par against waste is therefore conservative, which is
        what we want from a guard: a head at net zero is not worth its risk.
        """
        return self.used_bytes - self.wasted_bytes

    def issued_mb_str(self) -> str:
        """Compact tick-log form: issued MB, precision, net MB."""
        return (
            f"{self.issued_bytes / 1e6:.0f}mb/p{self.precision:.2f}"
            f"/net{self.net_bytes / 1e6:+.0f}mb"
        )

    def stats(self) -> dict:
        return {
            "issued": self.issued,
            "used": self.used,
            "precision": round(self.precision, 4),
            "issued_mb": round(self.issued_bytes / 1e6, 1),
            "wasted_mb": round(self.wasted_bytes / 1e6, 1),
            "net_mb": round(self.net_bytes / 1e6, 1),
            "window_issued": self.w_issued,
            "window_precision": round(self.window_precision, 4),
            "window_net_mb": round(self.window_net_bytes / 1e6, 1),
        }
