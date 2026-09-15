# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Residency head: bias eviction toward cold experts (quality-safe).

Hooks into ExpertCache eviction: when choosing a victim, skip keys likely to be
wanted again within the next H tokens. A wrong guess only changes which expert
is re-read later - never a logit.

The reuse window is measured in *tokens*. The previous version appended one
window per `note_demand` call, i.e. one per MoE layer, so on any model with more
MoE layers than the horizon a key's own next appearance always fell outside the
window and no score ever rose above its initial value. Every score stayed at
0.15 against a 0.35 bar, which is why `protect=0` no matter how long it ran.

Scores decay lazily on read (by tokens elapsed since last use) so there is no
periodic sweep over the table, and protection is rate-limited so a cache of
uniformly hot keys still makes eviction progress.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ... import config
from ..slot import _log

_INIT = 0.15


class ResidencyHead:
    def __init__(self, slot_id: str, n_experts: int, root: Path, n_layers: int = 0):
        self.enabled = bool(config.SIDECAR_HEAD_RESIDENCY)
        self.horizon = max(4, int(config.SIDECAR_RESIDENCY_HORIZON))
        self.min_score = float(config.SIDECAR_RESIDENCY_MIN_SCORE)
        self.max_protect = float(config.SIDECAR_RESIDENCY_MAX_PROTECT)
        self.n_experts = max(1, int(n_experts))
        # Flat (layer * n_experts + expert) tables rather than a dict of tuples:
        # a token touches every MoE layer, so the update has to be vectorized or
        # it costs more than everything else in the sidecar put together.
        self.n_layers = max(1, int(n_layers or 1))
        self._grown = n_layers <= 0
        self._ema = np.zeros(self.n_layers * self.n_experts, dtype=np.float32)
        self._last = np.full(self.n_layers * self.n_experts, -1, dtype=np.int32)
        self._cur: list[tuple[int, np.ndarray]] = []
        self._keys = 0
        self._t = 0
        self._protect_hits = 0
        self._scans = 0
        self._tuned_at_scans = 0
        self._tuned_at_hits = 0
        self._reuses = 0
        self._observed = 0
        self._decay = 0.9
        self._path = root / f"{slot_id}_residency.npz"
        self._load()
        self.mode = "train" if self.enabled else "off"

    def _ensure(self, layer_idx: int) -> None:
        if layer_idx < self.n_layers:
            return
        want = layer_idx + 1
        size = want * self.n_experts
        ema = np.zeros(size, dtype=np.float32)
        last = np.full(size, -1, dtype=np.int32)
        ema[: self._ema.size] = self._ema
        last[: self._last.size] = self._last
        self._ema, self._last, self.n_layers = ema, last, want

    # -------------------------------------------------------------- observing

    def note_demand(self, layer_idx: int, expert_ids) -> None:
        """Buffer this layer's demand; scoring happens once per token."""
        if not self.enabled or layer_idx < 0:
            return
        if layer_idx >= self.n_layers:
            self._ensure(int(layer_idx))
        ids = np.asarray(expert_ids, dtype=np.int64)
        if ids.size:
            # Offset here so end_token is a single concatenate.
            np.clip(ids, 0, self.n_experts - 1, out=ids)
            ids += int(layer_idx) * self.n_experts
            self._cur.append(ids)

    def end_token(self) -> None:
        """Fold one token of demand into the reuse model (fully vectorized)."""
        if not self.enabled or not self._cur:
            return
        t = self._t
        self._t += 1
        # Each (layer, expert) can appear at most once per token - note_demand
        # is called once per MoE layer - so no dedupe is needed.
        flat = (
            self._cur[0] if len(self._cur) == 1 else np.concatenate(self._cur)
        )
        self._cur = []
        last = self._last[flat]
        seen = last >= 0
        prev = self._ema[flat]
        reused = seen & ((t - last) <= self.horizon)
        decayed = prev * self._decay + reused * (1.0 - self._decay)
        self._ema[flat] = np.where(seen, decayed, np.float32(_INIT))
        self._last[flat] = t
        self._keys += int((~seen).sum())
        self._reuses += int(reused.sum())
        self._observed += int(flat.size)

    def drop_token(self) -> None:
        self._cur.clear()

    # --------------------------------------------------------------- scoring

    def protect_score(self, layer_key: str, expert_id: int, key_to_idx: dict) -> float:
        if not self.enabled:
            return 0.0
        idx = key_to_idx.get(layer_key)
        if idx is None or idx >= self.n_layers:
            return 0.0
        flat = idx * self.n_experts + int(expert_id)
        if flat < 0 or flat >= self._ema.size:
            return 0.0
        last = int(self._last[flat])
        if last < 0:
            return 0.0
        score = float(self._ema[flat])
        # Age the score by how long ago the key was last wanted, so a formerly
        # hot expert stops being protected once the model moves on.
        age = self._t - last
        if age > self.horizon:
            score *= self._decay ** min(32.0, age / self.horizon)
        return score

    def should_protect(self, layer_key: str, expert_id: int, key_to_idx: dict) -> bool:
        if not self.enabled or not self._live():
            return False
        # Rate limit: never protect more than max_protect of scanned candidates.
        if self._scans > 64 and self._protect_hits > self.max_protect * self._scans:
            return False
        return self.protect_score(layer_key, expert_id, key_to_idx) >= self.min_score

    def _live(self) -> bool:
        return self._keys >= 32 and self._t >= self.horizon

    def on_protect(self) -> None:
        self._protect_hits += 1
        self.mode = "prefetch"

    def on_scan(self) -> None:
        self._scans += 1
        if self._scans % 4096 == 0:
            self._retune()

    def _retune(self) -> None:
        """Keep protection selective rather than rate-cap-bound.

        Sitting exactly at max_protect means the bar is too low and we are
        effectively protecting whatever the scan happens to reach first, which is
        no better than LRU. Aim for about half the cap.
        """
        window = self._scans - self._tuned_at_scans
        if window <= 0:
            return
        rate = (self._protect_hits - self._tuned_at_hits) / window
        target = self.max_protect * 0.5
        if rate > self.max_protect * 0.9:
            self.min_score = min(0.95, self.min_score + 0.02)
        elif rate < target * 0.5:
            self.min_score = max(0.05, self.min_score - 0.02)
        self._tuned_at_scans = self._scans
        self._tuned_at_hits = self._protect_hits

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "ema_keys": self._keys,
            "protect_hits": self._protect_hits,
            "scans": self._scans,
            "protect_rate": (
                round(self._protect_hits / self._scans, 4) if self._scans else 0.0
            ),
            "reuse_rate": (
                round(self._reuses / self._observed, 4) if self._observed else 0.0
            ),
            "tokens": self._t,
            "horizon": self.horizon,
            "min_score": self.min_score,
        }

    # ------------------------------------------------------------ persistence

    def persist(self) -> None:
        if not self.enabled or not self._keys:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self._path,
                ema=self._ema,
                last=self._last,
                geom=np.array(
                    [self.n_layers, self.n_experts, self._t, self._keys], dtype=np.int64
                ),
            )
        except Exception as e:
            _log(f"residency save failed: {e!r}")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = np.load(self._path, allow_pickle=True)
            if "geom" not in data.files:
                return  # pre-v4 dict layout; not worth migrating
            n_layers, n_experts, t, keys = (
                int(v) for v in np.asarray(data["geom"]).tolist()
            )
            if n_experts != self.n_experts:
                return
            self.n_layers = max(self.n_layers, n_layers)
            self._ema = np.zeros(self.n_layers * self.n_experts, dtype=np.float32)
            self._last = np.full(self.n_layers * self.n_experts, -1, dtype=np.int32)
            ema = np.asarray(data["ema"], dtype=np.float32)
            last = np.asarray(data["last"], dtype=np.int32)
            self._ema[: ema.size] = ema
            self._last[: last.size] = last
            self._keys = keys
            # Token indices are relative to the saved run; rebase so ages stay
            # finite instead of appearing millions of tokens stale.
            self._t = int(t)
        except Exception as e:
            _log(f"residency load failed: {e!r}")
