# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prune-policy head: adaptive prune / wait thresholds (quality-affecting)."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from ... import config
from ..slot import _log


class PrunePolicyHead:
    """Propose per-token prune / wait_above from router-weight stats.

    Shadow trains online; live values only apply after promote. Floor is the
    env-configured PRUNE / WAIT_ABOVE so we never go more aggressive than the
    shipped baseline until the head proves safer denser pruning.
    """

    def __init__(self, slot_id: str, root: Path):
        self.enabled = bool(config.SIDECAR_HEAD_PRUNE)
        self.base_prune = float(config.PRUNE)
        self.base_wait = float(config.WAIT_ABOVE)
        self.max_prune = float(config.SIDECAR_PRUNE_MAX)
        self.feat_dim = 8
        # Tiny linear maps: features -> delta prune / delta wait in [0,1].
        scale = 0.01
        self.live_w = np.random.randn(self.feat_dim).astype(np.float32) * scale
        self.shadow_w = self.live_w.copy()
        self.live_b = np.float32(0.0)
        self.shadow_b = np.float32(0.0)
        self.prefetch_enabled = False  # here: "live adaptive on"
        self.locked = False
        self.train_steps = 0
        self.tokens_seen = 0
        self._shadow_agree = deque(maxlen=64)
        # Mixture mass the shadow policy would have kept, and what the shipped
        # baseline threshold keeps on the same tokens. Agreement with a
        # regression target says nothing about output quality; retained mass
        # does - but only relative to the baseline, since the shipped PRUNE is
        # itself aggressive (0.7 keeps well under half the mass on GLM and was
        # measured fine). An absolute bar would just permanently veto the head.
        self._kept_mass = deque(maxlen=64)
        self._base_mass = deque(maxlen=64)
        self.min_mass = float(config.SIDECAR_PRUNE_MIN_MASS)
        self._path = root / f"{slot_id}_prune.npz"
        self.mode = "fallback"
        self._load()

    @staticmethod
    def _features(wnp: np.ndarray) -> np.ndarray:
        # wnp: (N, K) router weights for selected experts.
        flat = wnp.reshape(-1, wnp.shape[-1])
        row = flat[-1]
        mx = float(row.max()) + 1e-8
        top2 = np.sort(row)[-2:]
        mass2 = float(top2.sum())
        # Entropy proxy on the K selected slots.
        p = row / (row.sum() + 1e-8)
        ent = float(-(p * np.log(p + 1e-8)).sum())
        return np.array(
            [
                mx,
                mass2,
                ent,
                float(row.mean()),
                float(row.std()),
                float((row >= 0.5 * mx).sum()) / max(1, row.size),
                float((row >= 0.2).sum()) / max(1, row.size),
                1.0,
            ],
            dtype=np.float32,
        )

    def propose(self, wnp: np.ndarray | None) -> tuple[float, float]:
        """Return (prune, wait_above) to use this token."""
        if not self.enabled or wnp is None:
            return self.base_prune, self.base_wait
        if not self.prefetch_enabled and not self.locked:
            return self.base_prune, self.base_wait
        x = self._features(wnp)
        raw = float(x @ self.live_w + self.live_b)
        # Map to [base, max]: only ever *more* aggressive than baseline when live.
        t = self.base_prune
        if self.base_prune > 0:
            span = max(0.0, self.max_prune - self.base_prune)
            t = self.base_prune + span * (1.0 / (1.0 + np.exp(-raw)))
        wait = self.base_wait
        if self.base_wait > 0:
            # Slight wait bump when mass is peaked (high top2).
            wait = min(0.95, self.base_wait + 0.15 * (1.0 / (1.0 + np.exp(-raw))))
        self.mode = "prefetch"
        return float(t), float(wait)

    def observe(self, wnp: np.ndarray, used_prune: float) -> None:
        """Train shadow toward a mass-safe target prune."""
        if not self.enabled or self.locked or wnp is None:
            return
        self.tokens_seen += 1
        x = self._features(wnp)
        row = wnp.reshape(-1, wnp.shape[-1])[-1]
        mx = float(row.max()) + 1e-8
        # Target: highest t such that kept mass >= 0.85 of selected mass.
        # Approximate by scanning a few candidate thresholds.
        target = self.base_prune
        total = float(row.sum()) + 1e-8
        for cand in (0.5, 0.6, 0.7, 0.8, 0.9, self.max_prune):
            if cand < self.base_prune:
                continue
            kept = float(row[row >= cand * mx].sum())
            if kept / total >= 0.85:
                target = cand
        # Regression on sigmoid-space vs base->max span.
        span = max(1e-6, self.max_prune - self.base_prune)
        y = (target - self.base_prune) / span
        y = min(1.0, max(0.0, y))
        pred = 1.0 / (1.0 + np.exp(-float(x @ self.shadow_w + self.shadow_b)))
        err = pred - y
        lr = float(config.SIDECAR_LR)
        self.shadow_w -= lr * err * pred * (1 - pred) * x
        self.shadow_b -= np.float32(lr * err * pred * (1 - pred))
        self.train_steps += 1
        self._shadow_agree.append(1.0 - abs(pred - y))
        # What the shadow policy would actually have done to this token, and
        # what the shipped threshold does, for a like-for-like comparison.
        thresh = self.base_prune + span * pred
        self._kept_mass.append(float(row[row >= thresh * mx].sum()) / total)
        self._base_mass.append(
            float(row[row >= self.base_prune * mx].sum()) / total
        )

        if (
            not self.prefetch_enabled
            and len(self._shadow_agree) >= 24
            and (sum(self._shadow_agree) / len(self._shadow_agree)) >= 0.7
            and self.mass_ratio() >= self.min_mass
        ):
            self.live_w = self.shadow_w.copy()
            self.live_b = self.shadow_b
            self.prefetch_enabled = True
            self.mode = "prefetch"
            _log(
                f"prune promote agree={sum(self._shadow_agree)/len(self._shadow_agree):.2f} "
                f"mass={self.mean_mass():.3f}/{self.mean_base_mass():.3f} "
                f"ratio={self.mass_ratio():.3f} "
                f"base={self.base_prune:.2f} max={self.max_prune:.2f}"
            )
        elif self.prefetch_enabled and len(self._shadow_agree) >= 24:
            agree = sum(self._shadow_agree) / len(self._shadow_agree)
            ratio = self.mass_ratio()
            if agree < 0.45 or ratio < self.min_mass * 0.98:
                self.prefetch_enabled = False
                self.mode = "train"
                _log(f"prune demote agree={agree:.2f} mass_ratio={ratio:.3f}")

    def mean_mass(self) -> float:
        if not self._kept_mass:
            return 0.0
        return float(sum(self._kept_mass) / len(self._kept_mass))

    def mean_base_mass(self) -> float:
        if not self._base_mass:
            return 0.0
        return float(sum(self._base_mass) / len(self._base_mass))

    def mass_ratio(self) -> float:
        """Shadow policy's retained mass as a fraction of the baseline's."""
        base = self.mean_base_mass()
        if base <= 0.0:
            return 0.0
        return self.mean_mass() / base

    def stats(self) -> dict:
        return {
            "mass": round(self.mean_mass(), 4),
            "base_mass": round(self.mean_base_mass(), 4),
            "mass_ratio": round(self.mass_ratio(), 4),
            "min_mass_ratio": self.min_mass,
            "enabled": self.enabled,
            "mode": self.mode,
            "live": self.prefetch_enabled or self.locked,
            "tokens_seen": self.tokens_seen,
            "train_steps": self.train_steps,
            "base_prune": self.base_prune,
            "max_prune": self.max_prune,
            "agree": round(
                float(sum(self._shadow_agree) / len(self._shadow_agree))
                if self._shadow_agree
                else 0.0,
                4,
            ),
        }

    def persist(self) -> None:
        if not self.enabled:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self._path,
                live_w=self.live_w,
                shadow_w=self.shadow_w,
                live_b=np.array([self.live_b]),
                shadow_b=np.array([self.shadow_b]),
                live=np.array([int(self.prefetch_enabled)]),
            )
        except Exception as e:
            _log(f"prune save failed: {e!r}")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = np.load(self._path)
            self.live_w = np.asarray(data["live_w"], dtype=np.float32)
            self.shadow_w = np.asarray(data["shadow_w"], dtype=np.float32)
            self.live_b = np.float32(data["live_b"][0])
            self.shadow_b = np.float32(data["shadow_b"][0])
            # Always re-earn live adaptive prune.
            self.prefetch_enabled = False
        except Exception as e:
            _log(f"prune load failed: {e!r}")
