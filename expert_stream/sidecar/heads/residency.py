# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Residency head: bias eviction toward cold experts (quality-safe).

Hooks into ExpertCache eviction: when choosing a victim, skip keys likely to
be wanted again soon. A wrong guess only changes which expert is re-read
later - never a logit.

Off by default. Protects only a distinguishable hot slice (quantile bar +
spread + min_hits + rate cap), and only while the net-time governor is
actuating. LFRU already captures most of what is causally available; this is
a small nudge on top.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ... import config
from ..slot import _log

_INIT = 0.15
# Minimum spread (95th minus 10th percentile of live reuse scores) before any
# key is protected. Below this the population is indistinguishable and LRU's
# recency order is the better selector - which, measured, it usually is.
_MIN_SPREAD = 0.15
# Slack below the top quantile, so the whole top cluster qualifies rather than
# whichever member of it happens to hold the maximum this refresh.
_BAR_EPS = 0.01


class ResidencyHead:
    def __init__(
        self,
        slot_id: str,
        n_experts: int,
        root: Path,
        n_layers: int = 0,
        governor=None,
    ):
        self.enabled = bool(config.SIDECAR_HEAD_RESIDENCY)
        self.horizon = max(4, int(config.SIDECAR_RESIDENCY_HORIZON))
        self.min_score = float(config.SIDECAR_RESIDENCY_MIN_SCORE)
        self.max_protect = float(config.SIDECAR_RESIDENCY_MAX_PROTECT)
        self.min_hits = max(2, int(config.SIDECAR_RESIDENCY_MIN_HITS))
        self.governor = governor
        self.n_experts = max(1, int(n_experts))
        # Flat (layer * n_experts + expert) tables rather than a dict of tuples:
        # a token touches every MoE layer, so the update has to be vectorized or
        # it costs more than everything else in the sidecar put together.
        self.n_layers = max(1, int(n_layers or 1))
        self._grown = n_layers <= 0
        self._ema = np.zeros(self.n_layers * self.n_experts, dtype=np.float32)
        self._last = np.full(self.n_layers * self.n_experts, -1, dtype=np.int32)
        # Hit counts gate on *evidence*: a key seen twice whose reuse rate is
        # 1.0 has told us much less than one seen two hundred times at 0.9.
        self._hits = np.zeros(self.n_layers * self.n_experts, dtype=np.int32)
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
        self._bar = 0.0
        self._flat_bars = 0
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
        hits = np.zeros(size, dtype=np.int32)
        ema[: self._ema.size] = self._ema
        last[: self._last.size] = self._last
        hits[: self._hits.size] = self._hits
        self._ema, self._last, self._hits = ema, last, hits
        self.n_layers = want

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
        np.add.at(self._hits, flat, 1)
        self._keys += int((~seen).sum())
        self._reuses += int(reused.sum())
        self._observed += int(flat.size)
        # Running quantile of live scores, so the protect bar tracks the
        # distribution instead of a fixed constant the scores have all
        # saturated past. Sampled - a full sort of a 12k-entry table every
        # token would cost more than the misses it saves.
        if t % 32 == 0:
            self._refresh_bar()

    def _refresh_bar(self) -> None:
        """Locate the top cluster of live scores, if there is one at all.

        Two quantiles, and the gap between them is the important part. A high
        score is not evidence that a key deserves protection - almost every
        score is high, because almost every just-used expert is used again
        soon. What matters is whether this key stands out from the others
        competing for the same cache line.

        When the scores have no spread there is nothing to tell apart, and the
        right policy is to protect nothing and let LRU do its job: LRU is the
        floor this head must never fall below, and in the absence of a signal
        the floor is also the ceiling. That case is not hypothetical - it is
        the shipped head's normal operating condition, and acting on it anyway
        is what turned a "quality-safe" optimisation into +55% misses.
        """
        live = self._hits >= self.min_hits
        n = int(live.sum())
        if n < 64:
            return
        scores = self._ema[live]
        if scores.size > 4096:  # sample; the shape is what matters, not detail
            scores = scores[:: scores.size // 4096]
        hi = float(np.quantile(scores, 0.95))
        lo = float(np.quantile(scores, 0.10))
        if hi - lo < _MIN_SPREAD:
            self._bar = 0.0
            self._flat_bars += 1
            return
        # Blend so the bar cannot whipsaw between refreshes.
        bar = hi - _BAR_EPS
        self._bar = 0.7 * self._bar + 0.3 * bar if self._bar else bar

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
        # Age by time since last demand so formerly-hot keys stop being protected.
        age = self._t - last
        if age > self.horizon:
            score *= self._decay ** min(32.0, age / self.horizon)
        return score

    def should_protect(self, layer_key: str, expert_id: int, key_to_idx: dict) -> bool:
        if not self.enabled or not self._live():
            return False
        # Hard cap, not a target. Hitting max_protect means we stopped selecting
        # and are only perturbing LRU's victim order (measured worse than LRU).
        if self._scans > 64 and self._protect_hits > self.max_protect * self._scans:
            return False
        idx = key_to_idx.get(layer_key)
        if idx is None or idx >= self.n_layers:
            return False
        flat = idx * self.n_experts + int(expert_id)
        if flat < 0 or flat >= self._hits.size:
            return False
        # Too few observations: reuse rate is noise; prefer LRU recency.
        if int(self._hits[flat]) < self.min_hits:
            return False
        return self.protect_score(layer_key, expert_id, key_to_idx) >= max(
            self.min_score, self._bar
        )

    def _live(self) -> bool:
        if self._keys < 32 or self._t < self.horizon or self._bar <= 0.0:
            return False
        # No bandwidth of its own, but it changes who gets re-read, so same
        # net-time gate as the bandwidth-spending heads.
        return self.governor is None or self.governor.actuating

    def on_protect(self) -> None:
        self._protect_hits += 1
        self.mode = "prefetch"

    def on_scan(self) -> None:
        self._scans += 1
        if self._scans % 4096 == 0:
            self._retune()

    def _retune(self) -> None:
        """Raise the bar when protection approaches the rate cap.

        One-way ratchet: sitting near max_protect was the failure mode that
        lost to LRU. A low protect rate is fine - protecting nothing is LRU,
        the floor this head guarantees.
        """
        window = self._scans - self._tuned_at_scans
        if window <= 0:
            return
        rate = (self._protect_hits - self._tuned_at_hits) / window
        if rate > self.max_protect * 0.75:
            self.min_score = min(0.95, self.min_score + 0.05)
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
            "bar": round(self._bar, 4),
            "min_hits": self.min_hits,
            # How often the score distribution had no usable spread.
            # High is normal: the head correctly declining to act.
            "flat_refreshes": self._flat_bars,
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
                hits=self._hits,
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
            size = self.n_layers * self.n_experts
            self._ema = np.zeros(size, dtype=np.float32)
            self._last = np.full(size, -1, dtype=np.int32)
            self._hits = np.zeros(size, dtype=np.int32)
            ema = np.asarray(data["ema"], dtype=np.float32)
            last = np.asarray(data["last"], dtype=np.int32)
            self._ema[: ema.size] = ema
            self._last[: last.size] = last
            if "hits" in data.files:
                hits = np.asarray(data["hits"], dtype=np.int32)
                self._hits[: hits.size] = hits
            else:
                # Older slots have no hit counts. Leaving them at zero means
                # the evidence gate blocks everything until the key is seen
                # again, which is the right way to fail: no protection is LRU.
                pass
            self._keys = keys
            # Token indices are relative to the saved run; rebase so ages stay
            # finite instead of appearing millions of tokens stale.
            self._t = int(t)
        except Exception as e:
            _log(f"residency load failed: {e!r}")
