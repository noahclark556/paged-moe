# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Wrap head: next-token early-layer expert prefetch (quality-safe).

Actuation is *incremental*. The engine's own prefetch ring already replays the
previous token's experts for the next token, so this head issues only the
experts its model predicts that the previous token did not already cover. Two
consequences, both important:

  - Coverage can never regress. Whatever the head does or does not believe, the
    baseline reads still happen; the head only ever adds.
  - Its value is directly measurable. `gain` is the recall those added experts
    contributed, `waste` the share of them nothing wanted. Promotion gates on
    gain rather than on absolute recall, which is what stops a head from going
    live merely by re-deriving the baseline (high recall, zero value) and from
    being blocked when it adds real coverage on top of a low baseline.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ... import config
from ..features import (
    EXTRA_DEMAND,
    EXTRA_HAS_HIDDEN,
    EXTRA_STEP,
    HIST_DECAY,
    FeatureSpec,
    auto_feat_dim,
    auto_wrap,
    same_layer_ceiling,
)
from ..slot import ExpertHeadSlot, _log


class WrapHead:
    def __init__(
        self,
        slot_id: str,
        n_layers: int,
        n_experts: int,
        layer_keys: list[str],
        root: Path,
        *,
        wrap: int | None = None,
        topk: int | None = None,
        feat_dim: int | None = None,
    ):
        self.enabled = bool(config.SIDECAR_HEAD_WRAP)
        self.layer_keys = layer_keys
        self.wrap = max(1, min(int(wrap or auto_wrap(n_layers)), n_layers))
        self.auto_topk = int(config.SIDECAR_TOPK) <= 0 and topk is None
        self.topk = max(1, int(topk or config.SIDECAR_TOPK or 3))
        self.min_score = float(config.SIDECAR_MIN_SCORE)
        self.min_recall = float(config.SIDECAR_MIN_RECALL)
        self.min_precision = float(config.SIDECAR_MIN_PRECISION)
        self.min_gain = float(config.SIDECAR_MIN_GAIN)
        self.feat_dim = int(feat_dim or auto_feat_dim(n_layers, n_experts))
        self.history_len = max(1, int(config.SIDECAR_HISTORY))
        self.spec = FeatureSpec.build(self.feat_dim, n_layers, self.wrap, n_experts)
        path = root / f"{slot_id}_wrap.npz"
        self.slot = ExpertHeadSlot.create(
            "wrap", slot_id, n_layers, n_experts, self.wrap, self.feat_dim, path=path
        )
        # Decayed history block, updated in place: one scaled add per token
        # instead of re-hashing the whole window (that rebuild was ~90% of the
        # sidecar's per-token cost).
        self._acc = np.zeros(self.spec.hist_dim, dtype=np.float32)
        self._prev_features: np.ndarray | None = None
        self._prev_predict: dict[int, list[int]] | None = None
        self._prev_demand: dict[int, list[int]] | None = None
        self._issued = 0
        self._used = 0
        self._extra_issued = 0
        self._extra_used = 0
        # Work-shedding strides, raised by the service when the sidecar's own
        # inline cost starts showing up in token latency (see _budget_check).
        self.score_stride = 1
        self.train_stride = 1
        self.mode = "fallback"

        if self.slot.locked and int(config.SIDECAR_LOCK_HITS) <= 0:
            self.slot.unlock()
            _log(f"wrap unlock slot={slot_id} - auto-lock disabled")

    # ------------------------------------------------------------------ budget

    @property
    def extra_budget(self) -> int:
        cfg = int(config.SIDECAR_EXTRA_BUDGET)
        return cfg if cfg > 0 else self.wrap * self.topk

    def _retune_topk(self) -> None:
        """Track the router's fan-out, then trade it off against wasted reads.

        The ceiling comes from the observed experts-per-layer (assuming top-3 of
        8 is wrong on any model that prunes). Within that, top-k walks down while
        most of the extra reads go unused and back up when they are paying, so
        the cost/benefit point is found per model instead of configured.
        """
        if not self.auto_topk:
            return
        mean = self.slot.mean_demand()
        if mean <= 0:
            return
        ceiling = max(
            1,
            min(
                self.slot.n_experts,
                int(math.ceil(mean * float(config.SIDECAR_TOPK_FACTOR))),
            ),
        )
        if len(self.slot.wastes) < 16:
            self.topk = ceiling
            return
        waste = self.slot.mean_waste()
        max_waste = float(config.SIDECAR_MAX_WASTE)
        if waste > max_waste and self.topk > 1:
            self.topk -= 1
        elif waste < max_waste * 0.9 and self.topk < ceiling:
            self.topk += 1
        else:
            self.topk = min(self.topk, ceiling)

    # ------------------------------------------------------------- token hook

    def observe_end_token(
        self,
        demand: dict[int, list[int]],
        history: list[dict[int, list[int]]],
        cache,
        *,
        hidden_sketch: np.ndarray | None = None,
    ) -> str:
        if not self.enabled:
            return "off"
        self.slot.tokens_seen += 1

        if self._prev_predict is not None:
            self._score(demand)
            self._gate()

        if (
            self._prev_features is not None
            and not self.slot.locked
            and self.slot.tokens_seen % self.train_stride == 0
        ):
            self.slot.train_shadow(
                self._prev_features, demand, float(config.SIDECAR_LR)
            )
            if self.mode == "fallback":
                self.mode = "train"
        if self.slot.tokens_seen % 64 == 0:
            self._retune_topk()

        self._acc *= np.float32(HIST_DECAY)
        self._acc += self.spec.demand_counts(demand)
        features = self.spec.compose(
            self._acc,
            demand,
            hidden_sketch=hidden_sketch,
            extra={
                EXTRA_HAS_HIDDEN: 1.0 if hidden_sketch is not None else 0.0,
                EXTRA_DEMAND: min(4.0, self.slot.mean_demand() / 8.0),
                EXTRA_STEP: math.log1p(self.slot.tokens_seen) / 16.0,
            },
        )
        predicted = self.slot.predict_topk(
            features, self.topk, min_score=self.min_score
        )
        self._prev_predict = predicted
        self._prev_features = features
        self._prev_demand = dict(demand)

        if self.slot.prefetch_enabled or self.slot.locked:
            self._issue(predicted, demand, cache)
            if self.mode != "locked":
                self.mode = "prefetch"
        elif self.mode != "train":
            self.mode = "fallback"
        return self.mode

    def _issue(
        self,
        predicted: dict[int, list[int]],
        baseline: dict[int, list[int]],
        cache,
    ) -> None:
        """Queue only what the engine's last-token replay does not already cover."""
        budget = self.extra_budget
        batch = []
        for i, ids in predicted.items():
            if i >= len(self.layer_keys) or budget <= 0:
                break
            already = set(baseline.get(i, ()))
            extra = [e for e in ids if e not in already][:budget]
            if extra:
                budget -= len(extra)
                self._extra_issued += len(extra)
                batch.append((self.layer_keys[i], extra))
        if batch:
            cache.prefetch_many(batch)

    # ----------------------------------------------------------------- scoring

    def _score(self, demand: dict[int, list[int]]) -> None:
        slot = self.slot
        recall, precision = slot.score_prediction(self._prev_predict, demand)
        slot.recalls.append(recall)
        slot.precisions.append(precision)
        base = self._prev_demand or {}
        if self._prev_demand is not None:
            slot.ceilings.append(same_layer_ceiling(base, demand, self.wrap))
        gain, waste = slot.score_margin(self._prev_predict, base, demand)
        slot.gains.append(gain)
        slot.wastes.append(waste)
        for i, ids in self._prev_predict.items():
            want = demand.get(i, ())
            self._issued += len(ids)
            hits = set(ids).intersection(want)
            self._used += len(hits)
            self._extra_used += len(hits - set(base.get(i, ())))
        if (
            self._prev_features is not None
            and not slot.locked
            and slot.tokens_seen % self.score_stride == 0
        ):
            sp = slot.predict_topk(
                self._prev_features,
                self.topk,
                use_shadow=True,
                min_score=self.min_score,
            )
            sr, spr = slot.score_prediction(sp, demand)
            slot.shadow_recalls.append(sr)
            slot.shadow_precisions.append(spr)
            sg, _ = slot.score_margin(sp, base, demand)
            slot.shadow_gains.append(sg)

    def _gate(self) -> None:
        slot = self.slot
        if slot.promote_if_better(
            self.min_recall, self.min_precision, min_gain=self.min_gain
        ):
            lift = slot.lift_vs_ceiling(shadow=True)
            lift_s = f"{lift:+.2f}" if lift is not None else "n/a"
            _log(
                f"wrap promote shadow_r={slot.mean_shadow_recall():.2f} "
                f"shadow_p={slot.mean_shadow_precision():.2f} "
                f"sh_gain={slot.mean_shadow_gain():+.3f} sh_lift={lift_s} "
                f"topk={self.topk}"
            )
            # Drop the stale live window so demote can't fire on pre-promote junk.
            slot.recalls.clear()
            slot.precisions.clear()
            slot.gains.clear()
            slot.wastes.clear()
            self.mode = "prefetch"
        if slot.maybe_lock(
            self.min_recall, self.min_precision, config.SIDECAR_LOCK_HITS
        ):
            _log(f"wrap lock recall={slot.mean_recall():.2f}")
            self.mode = "locked"
        elif not slot.prefetch_enabled:
            self.mode = "train"
        elif (
            not slot.locked
            and len(slot.gains) >= 24
            # Demote on value, not on raw accuracy: a head whose extra reads stop
            # paying for themselves is worse than no head at all.
            and (
                slot.mean_gain() < self.min_gain * 0.5
                or slot.mean_precision() < self.min_precision * 0.6
            )
        ):
            slot.demote()
            self.mode = "train"
            _log(
                f"wrap demote gain={slot.mean_gain():+.3f} "
                f"p={slot.mean_precision():.2f} - back to residual+router"
            )

    # ------------------------------------------------------------------- stats

    def stats(self) -> dict:
        slot = self.slot
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "locked": slot.locked,
            "prefetch_enabled": slot.prefetch_enabled,
            "wrap": self.wrap,
            "topk": self.topk,
            "feat_dim": self.feat_dim,
            "tokens_seen": slot.tokens_seen,
            "train_steps": slot.train_steps,
            "mean_recall": round(slot.mean_recall(), 4),
            "mean_precision": round(slot.mean_precision(), 4),
            "mean_ceiling": round(slot.mean_ceiling(), 4),
            "mean_gain": round(slot.mean_gain(), 4),
            "mean_waste": round(slot.mean_waste(), 4),
            "extra_issued": self._extra_issued,
            "extra_used": self._extra_used,
            "extra_hit_rate": (
                round(self._extra_used / self._extra_issued, 4)
                if self._extra_issued
                else 0.0
            ),
            "lift_vs_ceiling": (
                round(slot.lift_vs_ceiling() or 0.0, 4) if slot.ceilings else None
            ),
            "shadow_lift_vs_ceiling": (
                round(slot.lift_vs_ceiling(shadow=True) or 0.0, 4)
                if slot.ceilings
                else None
            ),
        }

    def persist(self) -> None:
        self.slot.save()
