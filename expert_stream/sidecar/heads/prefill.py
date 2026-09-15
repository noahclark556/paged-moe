# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prefill head: predict the experts a chunk will lean on (quality-safe).

The original formulation predicted a chunk's expert *union* and was close to
worthless: a few hundred tokens route across nearly every expert, so the union
is almost the whole set. Copying the previous chunk's union scored ~1.00 recall
("nothing to learn") while a 12-per-layer prediction scored ~0.01 ("hopeless
target"). Neither number said anything about prefetch value.

This head targets the chunk's HOT set instead - per layer, the experts whose
token count is a meaningful fraction of that layer's busiest expert. Those are
the experts worth having resident before the chunk runs, they are a small
fraction of the union, and predicting them is a real problem with a real
baseline.

Actuation is prefetch + post-leave-prefill warm only; prefill always computes
the full mixture, so a prompt's hidden states stay exact.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ... import config
from ..features import (
    EXTRA_CHUNK,
    EXTRA_HAS_HIDDEN,
    FeatureSpec,
    auto_feat_dim,
    auto_prefill_layers,
    hot_set,
    union_ceiling,
)
from ..slot import ExpertHeadSlot, _log


class PrefillUnionHead:
    def __init__(
        self,
        slot_id: str,
        n_layers: int,
        n_experts: int,
        layer_keys: list[str],
        root: Path,
        *,
        n_heads: int | None = None,
        topk: int | None = None,
        feat_dim: int | None = None,
    ):
        self.enabled = bool(config.SIDECAR_HEAD_PREFILL)
        self.layer_keys = layer_keys
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.n_heads = max(
            1, min(n_layers, int(n_heads or auto_prefill_layers(n_layers)))
        )
        self.auto_topk = int(config.SIDECAR_PREFILL_TOPK) <= 0 and topk is None
        self.topk = max(1, int(topk or config.SIDECAR_PREFILL_TOPK or 12))
        self.hot_frac = float(config.SIDECAR_PREFILL_HOT_FRAC)
        self.min_score = float(config.SIDECAR_MIN_SCORE)
        self.min_recall = float(config.SIDECAR_MIN_RECALL) * 0.75
        self.min_precision = float(config.SIDECAR_MIN_PRECISION) * 0.75
        self.feat_dim = int(feat_dim or auto_feat_dim(n_layers, n_experts))
        self.spec = FeatureSpec.build(
            self.feat_dim, n_layers, min(4, self.n_heads), n_experts
        )
        path = root / f"{slot_id}_prefill.npz"
        self.slot = ExpertHeadSlot.create(
            "prefill",
            slot_id,
            n_layers,
            n_experts,
            self.n_heads,
            self.feat_dim,
            path=path,
        )
        self._counts: dict[int, dict[int, int]] = {}
        self._union_seen = 0
        self._chunk_tokens = 0
        self._prev_hot: dict[int, list[int]] | None = None
        self._prev_features: np.ndarray | None = None
        self._chunk_active = False
        self._warm_pending = False
        self._chunks = 0
        self._union_frac = 0.0
        self._copy_issued = 0
        self._model_issued = 0
        self.mode = "fallback"

    # ------------------------------------------------------------ chunk hooks

    def begin_chunk(self, history: list[dict[int, list[int]]], cache) -> None:
        if not self.enabled:
            return
        self._counts = {}
        self._union_seen = 0
        self._chunk_tokens = 0
        self._chunk_active = True
        self._prev_features = self._features(history)
        batch = self._plan(self._prev_features)
        if batch:
            cache.prefetch_many(batch)
            self._warm_pending = False

    def _features(self, history: list[dict[int, list[int]]]) -> np.ndarray:
        return self.spec.build_vector(
            history,
            extra={
                EXTRA_HAS_HIDDEN: 0.0,
                EXTRA_CHUNK: math.log1p(self._chunks) / 8.0,
            },
        )

    def _plan(self, features: np.ndarray) -> list[tuple[str, list[int]]]:
        """Copy-baseline warm ∪ model prediction, deduped, per layer.

        The copy set is free and effective, so it ships whenever we have one -
        the model only has to add to it. Model additions are counted separately
        so its marginal contribution stays visible.
        """
        plan: dict[int, list[int]] = {}
        if self._prev_hot:
            for i, ids in self._prev_hot.items():
                if i < len(self.layer_keys):
                    plan[i] = list(ids)
                    self._copy_issued += len(ids)
        if self.slot.prefetch_enabled or self.slot.locked:
            pred = self.slot.predict_topk(
                features, self.topk, min_score=self.min_score
            )
            for i, ids in pred.items():
                if i >= len(self.layer_keys):
                    continue
                have = set(plan.get(i, ()))
                extra = [e for e in ids if e not in have]
                if extra:
                    plan.setdefault(i, []).extend(extra)
                    self._model_issued += len(extra)
            self.mode = "prefetch"
        elif plan:
            self.mode = "copy-warm"
        return [(self.layer_keys[i], ids) for i, ids in plan.items() if ids]

    def note_demand(self, layer_idx: int, expert_ids: list[int]) -> None:
        if not self.enabled or not self._chunk_active:
            return
        if layer_idx < 0 or layer_idx >= self.n_heads:
            return
        bucket = self._counts.setdefault(layer_idx, {})
        for e in expert_ids:
            e = int(e)
            bucket[e] = bucket.get(e, 0) + 1
        if layer_idx == 0:
            self._chunk_tokens += 1

    def end_chunk(self, history: list[dict[int, list[int]]]) -> None:
        if not self.enabled or not self._chunk_active:
            return
        self._chunk_active = False
        self._chunks += 1
        hot = hot_set(self._counts, self.hot_frac)
        if not hot:
            return
        target = {i: hot.get(i, []) for i in range(self.n_heads)}
        # How degenerate the old union target was, for the record.
        union = sum(len(v) for v in self._counts.values())
        hot_n = sum(len(v) for v in hot.values())
        self._union_frac = (
            hot_n / union if union else 0.0
        )
        self.slot.tokens_seen += 1
        if self.auto_topk and hot:
            mean_hot = hot_n / max(1, len(hot))
            self.topk = max(1, min(self.n_experts, int(math.ceil(mean_hot))))

        if self._prev_features is not None and not self.slot.locked:
            pred = self.slot.predict_topk(
                self._prev_features, self.topk, min_score=self.min_score
            )
            recall, precision = self.slot.score_prediction(pred, target)
            self.slot.recalls.append(recall)
            self.slot.precisions.append(precision)
            base = self._prev_hot or {}
            if self._prev_hot is not None:
                self.slot.ceilings.append(
                    union_ceiling(
                        {k: set(v) for k, v in base.items()},
                        {k: set(v) for k, v in target.items()},
                    )
                )
                gain, waste = self.slot.score_margin(pred, base, target)
                self.slot.gains.append(gain)
                self.slot.wastes.append(waste)
            sp = self.slot.predict_topk(
                self._prev_features,
                self.topk,
                use_shadow=True,
                min_score=self.min_score,
            )
            sr, spr = self.slot.score_prediction(sp, target)
            self.slot.shadow_recalls.append(sr)
            self.slot.shadow_precisions.append(spr)
            if self._prev_hot is not None:
                sg, _ = self.slot.score_margin(sp, base, target)
                self.slot.shadow_gains.append(sg)
            self.slot.train_shadow(
                self._prev_features, target, float(config.SIDECAR_LR)
            )
            if self.slot.promote_if_better(
                self.min_recall,
                self.min_precision,
                min_gain=float(config.SIDECAR_MIN_GAIN),
            ):
                lift = self.slot.lift_vs_ceiling(shadow=True)
                lift_s = f"{lift:+.2f}" if lift is not None else "n/a"
                _log(
                    f"prefill promote shadow_r={self.slot.mean_shadow_recall():.2f} "
                    f"shadow_p={self.slot.mean_shadow_precision():.2f} "
                    f"sh_gain={self.slot.mean_shadow_gain():+.3f} plift={lift_s} "
                    f"hot_topk={self.topk}"
                )
                self.mode = "prefetch"
            elif not self.slot.prefetch_enabled:
                self.mode = "train"
        self._prev_hot = {k: list(v) for k, v in target.items() if v}
        self._warm_pending = True

    def abort_chunk(self) -> None:
        self._chunk_active = False
        self._counts = {}

    def maybe_warm_after_leave_prefill(
        self, history: list[dict[int, list[int]]], cache
    ) -> None:
        """After a big prefill rebuilt the slab, warm the experts decode will want."""
        if not self.enabled or not self._warm_pending:
            return
        batch = self._plan(self._features(history))
        if batch:
            cache.prefetch_many(batch)
        self._warm_pending = False

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "prefetch_enabled": self.slot.prefetch_enabled,
            "chunks": self._chunks,
            "tokens_seen": self.slot.tokens_seen,
            "train_steps": self.slot.train_steps,
            "mean_recall": round(self.slot.mean_recall(), 4),
            "mean_precision": round(self.slot.mean_precision(), 4),
            "mean_ceiling": round(self.slot.mean_ceiling(), 4),
            "mean_gain": round(self.slot.mean_gain(), 4),
            "hot_share": round(self._union_frac, 4),
            "copy_issued": self._copy_issued,
            "model_issued": self._model_issued,
            "n_heads": self.n_heads,
            "topk": self.topk,
        }

    def persist(self) -> None:
        if self.enabled:
            self.slot.save()
