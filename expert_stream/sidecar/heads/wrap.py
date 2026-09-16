# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Wrap head: next-token early-layer expert prefetch (quality-safe).

Actuation is incremental: the engine's prefetch ring already replays the
previous token's experts, so this head issues only what its model predicts
the replay missed. Coverage cannot regress; a wrong extra only wastes a read.

Issuance is probability-thresholded, byte-budgeted against demand-miss
traffic, and idle-gated via cache.read_slack(). Promotion is owned by the
net-time governor (governor.py), not by recall/gain.
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
from ..governor import ByteLedger
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
        governor=None,
    ):
        self.enabled = bool(config.SIDECAR_HEAD_WRAP)
        self.governor = governor
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
        # Exactly what was queued last token, for byte-accurate scoring.
        self._prev_issued: dict[int, list[int]] | None = None
        self._issued = 0
        self._used = 0
        self._extra_issued = 0
        self._extra_used = 0
        # Byte-denominated accounting. `gain`/`waste` describe the model;
        # this describes the trade, and it is what the governor's decision has
        # to be reconcilable with.
        self.ledger = ByteLedger()
        self._min_prob = float(config.SIDECAR_MIN_PROB)
        self._byte_frac = float(config.SIDECAR_SPEC_BYTE_FRAC)
        # EMA of demand-miss bytes per token, read from the cache's counters.
        # The per-token speculative allowance is a share of this, so a model
        # that is not disk-bound gets essentially no allowance - there is no
        # stall to hide and nothing to win.
        self._miss_bytes_ema = 0.0
        self._prev_miss_bytes = 0
        self._expert_bytes = 0
        self._skipped_no_slack = 0
        self._skipped_no_budget = 0
        self._batches = 0
        # Free evidence gathered while the governor has actuation switched off:
        # what the head would have issued, and how much of it was wanted.
        self._counterfactual: dict[int, list[int]] | None = None
        self._cf_precision = 0.0
        self._cf_issued = 0
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
        """Size the candidate set to the router's fan-out.

        `topk` is now only how many candidates are *considered*; what gets
        issued is decided by probability and the byte budget in `_issue`. So
        this just tracks the observed experts-per-layer and no longer trades
        against waste - that trade moved to the place that can see bytes.
        """
        if not self.auto_topk:
            return
        mean = self.slot.mean_demand()
        if mean <= 0:
            return
        self.topk = max(
            1,
            min(
                self.slot.n_experts,
                math.ceil(mean * float(config.SIDECAR_TOPK_FACTOR)),
            ),
        )

    def _retune_threshold(self) -> None:
        """Walk the issuance probability bar toward the target precision.

        The bar, not the count, is the cost knob now. Raising it makes the head
        quieter and more selective; lowering it lets more through. Target is a
        *precision* on issued reads, which is the same thing as "most of the
        bandwidth I spend gets used" - the property the old waste ceiling was
        reaching for but measured against the wrong set.
        """
        led = self.ledger
        want = float(config.SIDECAR_TARGET_PRECISION)
        if led.window_issued >= 64:
            p = led.window_precision
        elif self._cf_issued >= 64:
            # Not actuating. The counterfactual measures the same quantity on
            # the same decision rule, so the bar can keep tuning while the head
            # is switched off - which is the difference between a head that
            # sits at whatever threshold it was demoted with and one that comes
            # back better than it left.
            p = self._cf_precision
        else:
            return
        if p < want:
            # Asymmetric on purpose: tighten fast, loosen slowly. Being too
            # quiet costs an opportunity; being too loud costs demand-path
            # bandwidth, and only one of those can make the model slower than
            # not running the head at all.
            self._min_prob = min(0.9, self._min_prob * 1.15 + 0.002)
        elif p > min(0.98, want + 0.1):
            self._min_prob = max(
                float(config.SIDECAR_MIN_PROB) * 0.25, self._min_prob * 0.92
            )

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
            self._note_counterfactual(demand)
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
            self._retune_threshold()

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
        predicted, probs = self.slot.predict_scored(
            features, self.topk, min_score=self.min_score
        )
        self._prev_predict = predicted
        self._prev_features = features
        self._prev_demand = dict(demand)

        if self._actuating():
            self._counterfactual = None
            self._issue(probs, demand, cache)
            if self.mode != "locked":
                self.mode = "prefetch"
        elif self.slot.prefetch_enabled or self.slot.locked:
            # Switched off by the governor, but still measurable for free:
            # remember what we *would* have issued so the next token can score
            # it. Costs one dict of ints and no disk.
            self._counterfactual = self._would_issue(probs, demand)
            self.mode = "hold"
        elif self.mode != "train":
            self._counterfactual = None
            self.mode = "fallback"
        return self.mode

    def _actuating(self) -> bool:
        """Both the model and the governor have to say yes.

        The model's own promote gate decides whether it predicts anything
        useful at all; the governor decides whether acting on it is faster on
        this machine, with this model, on this workload. Neither alone is
        sufficient - a head can be accurate and still be a bad trade, which is
        exactly the failure this file exists to prevent.
        """
        if not (self.slot.prefetch_enabled or self.slot.locked):
            return False
        gov = self.governor
        return True if gov is None else bool(gov.actuating)

    def _would_issue(
        self,
        probs: dict[int, list[tuple[int, float]]],
        baseline: dict[int, list[int]],
    ) -> dict[int, list[int]] | None:
        """The issue set this token would have produced, without issuing it.

        Same probability threshold and same incremental filter as `_issue`;
        only the capacity gates are left out, since they are about the disk's
        state rather than the model's opinion. Scoring this against the demand
        that follows is a real precision measurement obtained for free, and it
        is what lets a head that has trained its way to usefulness ask the
        governor for another look (see NetGovernor.request_probe).
        """
        out: dict[int, list[int]] = {}
        for i, scored in probs.items():
            if i >= len(self.layer_keys):
                continue
            already = set(baseline.get(i, ()))
            ids = [e for e, p in scored if p >= self._min_prob and e not in already]
            if ids:
                out[i] = ids
        return out or None

    def _note_counterfactual(self, demand: dict[int, list[int]]) -> None:
        """Score the previous token's would-be issue set and maybe ask to retry."""
        cf = self._counterfactual
        self._counterfactual = None
        if not cf:
            return
        issued = used = 0
        for i, ids in cf.items():
            want = demand.get(i, ())
            issued += len(ids)
            used += len(set(ids).intersection(want))
        if not issued:
            return
        p = used / issued
        self._cf_issued += issued
        self._cf_precision = (
            p if self._cf_precision == 0.0 else self._cf_precision * 0.95 + p * 0.05
        )
        gov = self.governor
        if gov is None or self._cf_issued < 512:
            return
        # Only worth another look if the head would now be spending its
        # bandwidth well. This is the same bar the live threshold servos
        # toward, so a request means "I would pass the test I failed".
        if self._cf_precision >= float(
            config.SIDECAR_TARGET_PRECISION
        ) and gov.request_probe(
            f"wrap counterfactual p={self._cf_precision:.2f} "
            f"over {self._cf_issued} would-be reads"
        ):
            self._cf_issued = 0

    def _token_budget_bytes(self, cache) -> int:
        """Speculative bytes allowed this token.

        A share of the demand-miss bytes this model reads per token, so the
        allowance is proportional to how much disk pressure there is to hide
        behind. Derived from the cache's own counters rather than configured,
        because the same fraction means completely different things on a model
        whose experts are 1.6 MB and one whose experts are 10.6 MB.
        """
        total = int(getattr(cache, "demand_miss_bytes", 0))
        delta = max(0, total - self._prev_miss_bytes)
        self._prev_miss_bytes = total
        if delta:
            self._miss_bytes_ema = (
                float(delta)
                if self._miss_bytes_ema == 0.0
                else self._miss_bytes_ema * 0.9 + float(delta) * 0.1
            )
        return int(self._miss_bytes_ema * self._byte_frac)

    def _issue(
        self,
        probs: dict[int, list[tuple[int, float]]],
        baseline: dict[int, list[int]],
        cache,
    ) -> None:
        """Queue high-confidence extras into idle capacity, within budget.

        Gates, cheapest refusal first:
        1. Read slack - if readers are busy on blocking work, stand down.
        2. Byte budget - at most a fixed share of per-token demand-miss bytes.
        3. Per-expert probability - fixed top-k spends even when guessing;
           thresholding keeps an unsure head silent.
        """
        slack = cache.read_slack()
        if slack <= 0:
            self._skipped_no_slack += 1
            return
        budget_bytes = self._token_budget_bytes(cache)
        if budget_bytes <= 0:
            self._skipped_no_budget += 1
            return
        per = self._expert_bytes or self._expert_bytes_of(cache)
        if per <= 0:
            return
        allowed = min(slack, budget_bytes // per)
        if allowed <= 0:
            self._skipped_no_budget += 1
            return

        # Rank candidates across all wrap layers by probability, so the byte
        # budget goes to the head's best guesses wherever they are rather than
        # being spent by whichever layer is enumerated first.
        cands: list[tuple[float, int, int]] = []
        for i, scored in probs.items():
            if i >= len(self.layer_keys):
                continue
            already = set(baseline.get(i, ()))
            for eid, p in scored:
                if p < self._min_prob or eid in already:
                    continue
                cands.append((p, i, eid))
        if not cands:
            return
        cands.sort(reverse=True)
        del cands[allowed:]

        batch: dict[int, list[int]] = {}
        for _p, i, eid in cands:
            batch.setdefault(i, []).append(eid)
        n = sum(len(v) for v in batch.values())
        self._extra_issued += n
        self.ledger.note_issued(n, per)
        self._batches += 1
        self._prev_issued = batch
        cache.prefetch_many([(self.layer_keys[i], ids) for i, ids in batch.items()])

    def _expert_bytes_of(self, cache) -> int:
        try:
            self._expert_bytes = int(cache.expert_nbytes(self.layer_keys[0]))
        except Exception:  # noqa: BLE001 - no expert size, no byte budget, no issuing
            self._expert_bytes = 0
        self.ledger.expert_bytes = self._expert_bytes
        return self._expert_bytes

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
        # Score the trade against what was actually *issued*, not against the
        # model's whole top-k. Issuance is a thresholded, budgeted subset of the
        # prediction, so "how good is the model" and "was the bandwidth worth
        # spending" are different questions; conflating them is what let a head
        # wasting 44% of its reads report gain +0.26.
        if self._prev_issued:
            used = 0
            for i, ids in self._prev_issued.items():
                used += len(set(ids).intersection(demand.get(i, ())))
            self.ledger.note_used(used)
            self._prev_issued = None
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
            # Drop the stale live window so demote cannot fire on pre-promote junk.
            slot.recalls.clear()
            slot.precisions.clear()
            slot.gains.clear()
            slot.wastes.clear()
            # New weights, new trade: reset the byte window too.
            self.ledger.reset_window()
            self.mode = "prefetch"
        if slot.maybe_lock(
            self.min_recall, self.min_precision, config.SIDECAR_LOCK_HITS
        ):
            _log(f"wrap lock recall={slot.mean_recall():.2f}")
            self.mode = "locked"
        elif not slot.prefetch_enabled:
            self.mode = "train"
        elif not slot.locked and self._in_deficit():
            slot.demote()
            self.mode = "train"
            _log(
                f"wrap demote net={self.ledger.window_net_bytes / 1e6:+.0f}mb "
                f"p={self.ledger.window_precision:.2f} "
                f"gain={slot.mean_gain():+.3f} - back to residual+router"
            )
            self.ledger.reset_window()

    def _in_deficit(self) -> bool:
        """Has actuation spent more bandwidth than it earned?

        Demotion is on bytes, not recall. High gain with mostly-unused reads
        still loses; bytes are the unit that shows it.
        """
        led = self.ledger
        if led.window_issued < 256:
            return False
        return led.window_net_bytes < 0

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
            # The trade, in the unit that costs. `bytes` is what decides
            # whether this head should be running; the recall numbers above
            # only say whether the model is any good.
            "bytes": self.ledger.stats(),
            "min_prob": round(self._min_prob, 4),
            "spec_byte_frac": self._byte_frac,
            "skipped_no_slack": self._skipped_no_slack,
            "skipped_no_budget": self._skipped_no_budget,
            "batches": self._batches,
            # What the head would be achieving if it were let back on, measured
            # while it is off and costing nothing.
            "counterfactual_precision": round(self._cf_precision, 4),
            "counterfactual_issued": self._cf_issued,
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
