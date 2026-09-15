# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-head online expert sidecar façade.

Heads (each independently flag-gated; disabled = no weights / no hot-path work):
  * wrap - next-token early-layer prefetch (quality-safe)
  * prefill - chunk hot-expert prefetch / post-leave warm (quality-safe)
  * residency - eviction protect scores (quality-safe)
  * prune - adaptive prune/wait (quality-affecting; shadow->promote)

When EXPERT_STREAM_SIDECAR=0 this module is never constructed and the decode
path pays a single None-check per hook.

Two cross-cutting concerns live here rather than in the heads:

  * Geometry. Everything (feature width, wrap depth, prefill breadth, top-k)
    scales from the model's own MoE shape unless an env var pins it, so a 16-
    expert 24-layer model and a 160-expert 89-layer model both get sane sizes
    without anyone tuning them.
  * Cost. The sidecar runs inline between tokens, so its own time *is* token
    latency. It measures itself against a wall-clock budget and sheds work
    (shadow scoring, then training, then top-k, then actuation) rather than
    quietly taxing a model it cannot help.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .. import config
from .features import (
    SLOT_VERSION,
    auto_feat_dim,
    auto_prefill_layers,
    auto_wrap,
    migrate_legacy_slot_files,
    model_slot_id,
    project_hidden,
    sketch_matrix,
)
from .heads import PrefillUnionHead, PrunePolicyHead, ResidencyHead, WrapHead
from .slot import _log


class ExpertSidecar:
    def __init__(
        self,
        model_path: str,
        layer_keys: list[str],
        n_experts: int,
        *,
        slot_id: str | None = None,
    ):
        self.n_layers = len(layer_keys)
        self.n_experts = int(n_experts)
        self.layer_keys = list(layer_keys)
        self.key_to_idx = {k: i for i, k in enumerate(layer_keys)}
        self.model_path = model_path
        root = Path(config.SIDECAR_DIR)
        root.mkdir(parents=True, exist_ok=True)
        self._root = root

        if slot_id is None:
            new_id = model_slot_id(model_path, self.n_layers, self.n_experts)
            self.slot_id, migrated = migrate_legacy_slot_files(
                root, model_path, self.n_layers, self.n_experts, new_id=new_id
            )
            if migrated:
                _log(
                    f"migrated path-slot -> content-slot {self.slot_id} "
                    f"({len(migrated)} files)"
                )
        else:
            self.slot_id = slot_id

        # ---- resolved geometry (env pins win; 0/unset means scale it) --------
        self.feat_dim = int(config.SIDECAR_FEAT_DIM) or auto_feat_dim(
            self.n_layers, self.n_experts
        )
        self.wrap_n = int(config.SIDECAR_WRAP) or auto_wrap(self.n_layers)
        prefill_layers = int(config.SIDECAR_PREFILL_LAYERS) or auto_prefill_layers(
            self.n_layers
        )

        self.wrap = WrapHead(
            self.slot_id,
            self.n_layers,
            self.n_experts,
            self.layer_keys,
            root,
            wrap=self.wrap_n,
            feat_dim=self.feat_dim,
        )
        self.prefill = PrefillUnionHead(
            self.slot_id,
            self.n_layers,
            self.n_experts,
            self.layer_keys,
            root,
            n_heads=prefill_layers,
            feat_dim=self.feat_dim,
        )
        self.residency = ResidencyHead(
            self.slot_id, self.n_experts, root, n_layers=self.n_layers
        )
        self.prune = PrunePolicyHead(self.slot_id, root)

        self._token_demand: dict[int, list[int]] = {}
        self._history: deque = deque(maxlen=max(1, int(config.SIDECAR_HISTORY)))
        self._mode = "fallback"
        self._log_every = max(1, int(config.SIDECAR_LOG_EVERY))
        self._tokens_since_log = 0
        self._pretrain = False
        self._pretrain_samples: list = []
        # Demand trajectories for densified wrap fit (sliding-window expand).
        self._pretrain_traj: list[dict[int, list[int]]] = []
        self._pretrain_trajectories: list[list[dict[int, list[int]]]] = []
        # Live bank: accumulate short decode stretches during normal use so
        # `paged-moe-pretrain fit` / live bank warm without a curated corpus.
        self._bank_traj: list[dict[int, list[int]]] = []
        self._bank_dirty = False
        self._save_lock = threading.Lock()
        self._left_prefill_warmed = False

        # ---- hidden-state sketch --------------------------------------------
        self._sketch_dim = (
            int(config.SIDECAR_SKETCH_DIM) if bool(config.SIDECAR_HIDDEN) else 0
        )
        self._proj: np.ndarray | None = None
        self._hidden_dim = 0
        self._hidden_ok = self._sketch_dim > 0
        self._hidden_hits = 0
        self._sketch: np.ndarray | None = None

        # ---- cost accounting ------------------------------------------------
        self._us_ema = 0.0
        self._gap_ema = 0.0
        self._last_end = 0.0
        self._shed = 0
        self._cost_samples = 0
        self._over_streak = 0
        self._under_streak = 0
        self._save_every = max(0, int(config.SIDECAR_SAVE_EVERY))
        self._tokens_since_save = 0

        # Back-compat attributes used by pretrain / tests.
        self.slot = self.wrap.slot
        self.min_score = self.wrap.min_score

        heads = [
            name
            for name, head in (
                ("wrap", self.wrap),
                ("prefill", self.prefill),
                ("residency", self.residency),
                ("prune", self.prune),
            )
            if head.enabled
        ]
        _log(
            f"enabled v{SLOT_VERSION} slot={self.slot_id} layers={self.n_layers} "
            f"experts={self.n_experts} heads={','.join(heads) or 'none'} "
            f"path={root}"
        )
        _log(
            f"geometry feat={self.feat_dim} sketch={self._sketch_dim} "
            f"wrap={self.wrap.wrap} topk={self.wrap.topk}"
            f"{'(auto)' if self.wrap.auto_topk else ''} "
            f"prefill_layers={self.prefill.n_heads} "
            f"budget={config.SIDECAR_TIME_BUDGET_MS:.1f}ms/"
            f"{config.SIDECAR_MAX_OVERHEAD * 100:.0f}%"
        )
        if self.wrap.slot.train_steps:
            _log(
                f"resumed steps={self.wrap.slot.train_steps} "
                f"tokens={self.wrap.slot.tokens_seen} "
                f"live={int(self.wrap.slot.prefetch_enabled)} "
                f"r={self.wrap.slot.mean_recall():.2f} "
                f"gain={self.wrap.slot.mean_gain():+.3f}"
            )

    @property
    def topk(self) -> int:
        """Live top-k (the wrap head retunes it to the router's real fan-out)."""
        return self.wrap.topk

    # ---------------------------------------------------------- decode hooks

    def note_demand(self, layer_key: str, expert_ids: list[int]) -> None:
        idx = self.key_to_idx.get(layer_key)
        if idx is None:
            return
        ids = [int(e) for e in expert_ids]
        self._token_demand[idx] = ids
        self.residency.note_demand(idx, ids)

    def note_hidden(self, hidden) -> None:
        """Accept the last MoE layer's hidden state for this token.

        Cheap and best-effort: any failure permanently disables the block rather
        than risking the decode path. The vector is projected once per token.
        """
        if not self._hidden_ok or hidden is None:
            return
        try:
            vec = np.asarray(hidden, dtype=np.float32).reshape(-1)
            if vec.size == 0:
                return
            if self._proj is None or vec.size != self._hidden_dim:
                self._hidden_dim = int(vec.size)
                self._proj = sketch_matrix(
                    self.slot_id, self._hidden_dim, self._sketch_dim
                )
                _log(f"hidden sketch {self._hidden_dim}->{self._sketch_dim}")
            self._sketch = project_hidden(vec, self._proj)
            self._hidden_hits += 1
        except Exception as e:
            self._hidden_ok = False
            self._sketch = None
            _log(f"hidden sketch disabled: {e!r}")

    def end_token(self, cache) -> str:
        demand = self._token_demand
        self._token_demand = {}
        if not demand:
            return self._mode
        t0 = time.perf_counter()
        if self._last_end:
            gap = t0 - self._last_end
            if gap < 5.0:  # ignore turn boundaries / stalls
                self._gap_ema = (
                    gap if self._gap_ema == 0.0 else self._gap_ema * 0.9 + gap * 0.1
                )

        history = list(self._history)
        if self._pretrain:
            self._collect_pretrain(demand, history)
            self._last_end = time.perf_counter()
            return self._mode

        self.residency.end_token()

        # First decode layer after a prefill: warm the rebuilt slab once.
        if not self._left_prefill_warmed:
            self.prefill.maybe_warm_after_leave_prefill(history, cache)
            self._left_prefill_warmed = True

        mode = self.wrap.observe_end_token(
            demand, history, cache, hidden_sketch=self._sketch
        )
        self._sketch = None
        self._history.append(dict(demand))
        self._mode = mode
        self._bank_note(demand)

        dt = time.perf_counter() - t0
        self._us_ema = (
            dt if self._us_ema == 0.0 else self._us_ema * 0.95 + dt * 0.05
        )
        self._last_end = time.perf_counter()
        self._budget_check()

        self._tokens_since_log += 1
        if self._tokens_since_log >= self._log_every:
            self._tokens_since_log = 0
            self._tick_log()
        self._tokens_since_save += 1
        if self._save_every and self._tokens_since_save >= self._save_every:
            self._tokens_since_save = 0
            try:
                self._persist()
            except Exception:
                pass
        return self._mode

    def drop_token(self) -> None:
        self._token_demand = {}
        self._sketch = None
        self.residency.drop_token()
        self.prefill.abort_chunk()
        self._left_prefill_warmed = False

    # ------------------------------------------------------------- cost guard

    @property
    def overhead(self) -> float:
        """Share of a token's wall time spent inside the sidecar."""
        total = self._gap_ema + self._us_ema
        return float(self._us_ema / total) if total > 0 else 0.0

    @property
    def decode_tok_s(self) -> float | None:
        """EMA decode rate from inter-token gap + sidecar cost (live, not bench).

        ``_gap_ema`` is wall time between end_token calls (model+disk); ``_us_ema``
        is time spent inside the sidecar. Together they are one decode token.
        Gaps ≥5s are ignored so turn boundaries do not crush the EMA.
        """
        total = self._gap_ema + self._us_ema
        if total <= 0.0:
            return None
        return float(1.0 / total)

    def _budget_check(self) -> None:
        """Shed work when the sidecar shows up in token latency.

        Ordered cheapest-loss-first: shadow scoring only slows learning, then
        training stride, then a smaller top-k, and only as a last resort stop
        actuating. A model the sidecar cannot help must not pay for it.

        Requires a warm EMA and a sustained streak before acting - a couple of
        slow tokens during a turn boundary must not cost the head its state.
        """
        self._cost_samples += 1
        if self._cost_samples < 64 or self._gap_ema <= 0.0:
            return
        budget = float(config.SIDECAR_TIME_BUDGET_MS) / 1000.0
        max_ovh = float(config.SIDECAR_MAX_OVERHEAD)
        over = self._us_ema > budget or self.overhead > max_ovh
        if over:
            self._under_streak = 0
            self._over_streak += 1
            if self._over_streak < 32 or self._shed >= 4:
                return
            self._over_streak = 0
            self._shed += 1
            self._apply_shed()
            _log(
                f"budget shed level={self._shed} us={self._us_ema * 1e6:.0f} "
                f"ovh={self.overhead * 100:.1f}% (budget {budget * 1e3:.1f}ms/"
                f"{max_ovh * 100:.0f}%)"
            )
            return
        self._over_streak = 0
        # Recover one level at a time once we are comfortably inside budget.
        if self._shed <= 0:
            return
        if self._us_ema < budget * 0.5 and self.overhead < max_ovh * 0.5:
            self._under_streak += 1
            if self._under_streak >= 256:
                self._under_streak = 0
                self._shed -= 1
                self._apply_shed()
                _log(f"budget recover level={self._shed} us={self._us_ema * 1e6:.0f}")

    def _apply_shed(self) -> None:
        level = self._shed
        self.wrap.score_stride = (1, 4, 4, 8, 8)[level]
        self.wrap.train_stride = (1, 1, 2, 4, 8)[level]
        if level >= 3 and self.wrap.topk > 1:
            self.wrap.auto_topk = False
            self.wrap.topk = max(1, self.wrap.topk - 1)
        if level >= 4 and self.wrap.slot.prefetch_enabled:
            self.wrap.slot.demote()
            self.wrap.mode = "train"

    # ---------------------------------------------------------- prefill hooks

    def begin_prefill_chunk(self, cache) -> None:
        self._left_prefill_warmed = False
        self.prefill.begin_chunk(list(self._history), cache)

    def note_prefill_demand(self, layer_key: str, expert_ids: list[int]) -> None:
        idx = self.key_to_idx.get(layer_key)
        if idx is None:
            return
        self.prefill.note_demand(idx, [int(e) for e in expert_ids])

    def end_prefill_chunk(self) -> None:
        self.prefill.end_chunk(list(self._history))

    # ------------------------------------------------------ prune / residency

    def prune_params(self, wnp) -> tuple[float, float]:
        """Adaptive (prune, wait_above); falls back to config when head off/cold."""
        return self.prune.propose(wnp)

    def observe_prune(self, wnp, used_prune: float) -> None:
        self.prune.observe(wnp, used_prune)

    def should_protect(self, layer_key: str, expert_id: int) -> bool:
        return self.residency.should_protect(layer_key, expert_id, self.key_to_idx)

    def on_protect(self) -> None:
        self.residency.on_protect()

    def on_scan(self) -> None:
        self.residency.on_scan()

    def on_residency_protect(self) -> None:
        self.on_protect()

    def on_residency_scan(self) -> None:
        self.on_scan()

    # ---------------------------------------------------------- pretrain API

    def begin_pretrain(self) -> None:
        self._pretrain = True
        self._pretrain_samples.clear()
        self._pretrain_traj = []
        self._pretrain_trajectories = []
        self.wrap.slot.unlock()
        self._prev_features = None
        self._mode = "pretrain"
        _log(f"pretrain collect slot={self.slot_id}")

    def end_pretrain_prompt(self) -> None:
        """Close one corpus prompt's demand trajectory (call between prompts)."""
        if len(self._pretrain_traj) >= 2:
            self._pretrain_trajectories.append(self._pretrain_traj)
        self._pretrain_traj = []
        self._prev_features = None

    def _collect_pretrain(
        self, demand: dict[int, list[int]], history: list[dict[int, list[int]]]
    ) -> None:
        frame = {int(k): [int(e) for e in v] for k, v in demand.items()}
        self._pretrain_traj.append(frame)
        # Keep the online step too so weights move during collect; dense fit
        # later reuses the expanded bank.
        features = self.wrap.spec.build_vector(history + [frame])
        if getattr(self, "_prev_features", None) is not None:
            self._pretrain_samples.append((self._prev_features, dict(frame)))
            self.wrap.slot.train_shadow(
                self._prev_features, frame, float(config.SIDECAR_LR)
            )
        self._prev_features = features
        self._history.append(dict(frame))
        self._mode = "pretrain"

    def _bank_note(self, demand: dict[int, list[int]]) -> None:
        """Append one live token to the wrap bank (memory only on the hot path).

        Disk flush happens in ``_persist`` / close - never inline here. A failed
        or slow rewrite of the bank must not hitch or abort decode.
        """
        frame = {int(k): [int(e) for e in v] for k, v in demand.items()}
        self._bank_traj.append(frame)
        self._bank_dirty = True
        # Bound the in-memory stretch so a long turn cannot grow without limit;
        # overflow drops the oldest half (still enough for sliding windows).
        if len(self._bank_traj) > 256:
            self._bank_traj = self._bank_traj[-128:]

    def _flush_bank_traj(self) -> None:
        if len(self._bank_traj) < 2:
            self._bank_traj = []
            self._bank_dirty = False
            return
        try:
            from .bank import bank_path, load_bank, merge_trajectories, save_bank

            path = bank_path(self._root, self.slot_id)
            existing, meta = load_bank(path)
            merged = merge_trajectories(existing, [list(self._bank_traj)])
            save_bank(
                path,
                merged,
                meta={
                    **meta,
                    "slot": self.slot_id,
                    "n_layers": self.n_layers,
                    "n_experts": self.n_experts,
                    "feat_dim": self.feat_dim,
                    "wrap": self.wrap.wrap,
                },
            )
        except Exception as e:
            _log(f"wrap bank flush failed: {e!r}")
        finally:
            self._bank_traj = []
            self._bank_dirty = False

    def fit_pretrain(
        self,
        epochs: int = 4,
        lr: float | None = None,
        *,
        multi_scale: bool = True,
        from_bank: bool = False,
    ) -> dict:
        """Dense wrap fit from collected trajectories (and/or the on-disk bank).

        Sliding-window expansion turns a short decode into many samples so a
        tiny corpus + heavy epochs is enough. ``from_bank`` skips the need to
        have just run collect - useful after normal agent use has filled the bank.
        """
        from .bank import (
            bank_path,
            expand_trajectories,
            load_bank,
            merge_trajectories,
            save_bank,
        )

        lr = float(config.SIDECAR_LR if lr is None else lr)
        self.end_pretrain_prompt()

        trajectories = list(self._pretrain_trajectories)
        path = bank_path(self._root, self.slot_id)
        # from_bank: prefer the on-disk bank. If memory also holds trajs (e.g.
        # just finished collect that already wrote the bank), do not append the
        # same list again - that used to double every sample.
        if from_bank:
            existing, _ = load_bank(path)
            if not trajectories:
                trajectories = existing
            elif existing and len(existing) >= len(trajectories):
                trajectories = existing
            elif existing:
                trajectories = merge_trajectories(existing, trajectories)
        elif not trajectories:
            existing, _ = load_bank(path)
            if existing:
                trajectories = existing

        samples = expand_trajectories(
            trajectories,
            self.wrap.spec,
            history_len=max(1, int(config.SIDECAR_HISTORY)),
            multi_scale=multi_scale,
        )
        # Fall back to the sparse online pairs if somehow no traj landed.
        if len(samples) < 8 and self._pretrain_samples:
            samples = list(self._pretrain_samples)
        n = len(samples)
        if n < 8:
            raise RuntimeError(f"pretrain needs more samples (have {n})")

        # Persist for --fit-only. Merge with any live-collected bank when this
        # call came from a fresh collect (from_bank already included disk).
        existing, prev_meta = load_bank(path)
        to_save = (
            trajectories
            if from_bank
            else merge_trajectories(existing, trajectories)
        )
        save_bank(
            path,
            to_save,
            meta={
                **(prev_meta or {}),
                "slot": self.slot_id,
                "n_layers": self.n_layers,
                "n_experts": self.n_experts,
                "feat_dim": self.feat_dim,
                "wrap": self.wrap.wrap,
                "samples_expanded": n,
            },
        )

        cut = max(1, int(n * 0.8))
        # Shuffle before the split so holdout isn't just "the last prompt."
        order_all = np.random.permutation(n)
        train_idx = order_all[:cut]
        hold_idx = order_all[cut:]
        train = [samples[int(i)] for i in train_idx]
        hold = [samples[int(i)] for i in hold_idx] or train[-max(1, len(train) // 5) :]

        slot = self.wrap.slot
        slot.locked = False
        for ep in range(max(1, epochs)):
            order = np.random.permutation(len(train))
            for j in order:
                features, demand = train[int(j)]
                slot.train_shadow(features, demand, lr)
            _log(
                f"pretrain epoch={ep + 1}/{epochs} samples={len(train)} "
                f"(expanded from {len(trajectories)} traj)"
            )
        slot.live = slot.shadow.copy()
        slot.live_b = slot.shadow_b.copy()
        hit = need = issued = 0
        for features, demand in hold:
            pred = slot.predict_topk(
                features, self.wrap.topk, min_score=self.min_score
            )
            for i in range(slot.n_heads):
                want = set(demand.get(i, ()))
                got = pred.get(i, ())
                issued += len(got)
                if want:
                    need += len(want)
                    hit += len(want.intersection(got))
        recall = (hit / need) if need else 0.0
        precision = (hit / issued) if issued else 0.0
        # Leave unlocked so online promote still gates actuation.
        slot.prefetch_enabled = False
        slot.locked = False
        self._pretrain = False
        # Drop in-memory trajs so a later from_bank=True cannot re-merge them.
        self._pretrain_trajectories = []
        self._pretrain_traj = []
        self._pretrain_samples = []
        self._persist()
        report = {
            "samples": n,
            "trajectories": len(trajectories),
            "train": len(train),
            "holdout": len(hold),
            "holdout_recall": round(recall, 4),
            "holdout_precision": round(precision, 4),
            "train_steps": slot.train_steps,
            "slot": self.slot_id,
            "bank": str(path),
            "path": str(slot.path) if slot.path else None,
        }
        _log(
            f"pretrain done samples={n} traj={len(trajectories)} "
            f"holdout_recall={recall:.2f} holdout_precision={precision:.2f}"
        )
        return report

    def fit_from_bank(self, epochs: int = 8, lr: float | None = None) -> dict:
        """Dense wrap fit using only the on-disk bank (no MoE generate)."""
        self._pretrain = True
        self._pretrain_trajectories = []
        self._pretrain_traj = []
        return self.fit_pretrain(epochs=epochs, lr=lr, from_bank=True)

    @classmethod
    def from_bank(cls, path: Path) -> "ExpertSidecar":
        """Rebuild a wrap-capable sidecar from a saved bank (no model load).

        Does **not** preload trajectories into ``_pretrain_trajectories`` - call
        ``fit_pretrain(from_bank=True)`` (or ``fit_from_bank``) so the bank is
        read once. Preloading + ``from_bank=True`` used to duplicate every sample.
        """
        from .bank import load_bank

        path = Path(path)
        trajectories, meta = load_bank(path)
        if not trajectories:
            raise RuntimeError(f"empty wrap bank: {path}")
        n_layers = int(meta.get("n_layers") or 0)
        n_experts = int(meta.get("n_experts") or 0)
        if n_layers <= 0 or n_experts <= 0:
            raise RuntimeError(f"bank missing geometry meta: {path}")
        slot = str(meta.get("slot") or path.stem.replace("_wrap_bank", ""))
        keys = [f"moe.{i}" for i in range(n_layers)]
        sc = cls("from-bank", keys, n_experts, slot_id=slot)
        sc._root = path.parent
        sc._pretrain = True
        return sc

    def unlock(self) -> None:
        self.wrap.slot.unlock()
        self._mode = "train"
        self._persist()
        _log(f"unlock slot={self.slot_id}")

    # ---------------------------------------------------------- persist / stats

    def _persist(self) -> None:
        with self._save_lock:
            if self._bank_dirty or len(self._bank_traj) >= 2:
                try:
                    self._flush_bank_traj()
                except Exception:
                    pass
            self.wrap.persist()
            self.prefill.persist()
            self.residency.persist()
            self.prune.persist()

    def _tick_log(self) -> None:
        w, p, r, pr = self.wrap, self.prefill, self.residency, self.prune

        def _lift(slot, *, shadow: bool = False) -> str:
            v = slot.lift_vs_ceiling(shadow=shadow)
            return "n/a" if v is None else f"{v:+.2f}"

        # gain / waste are the numbers that decide whether this is worth
        # running: recall the head added on top of the engine's own last-token
        # replay, and how much of what it added nothing wanted.
        parts = [
            f"tick mode={self._mode} slot={self.slot_id}",
            f"wrap_r={w.slot.mean_recall():.2f}/{w.slot.mean_precision():.2f}",
            f"ceil={w.slot.mean_ceiling():.2f}",
            f"lift={_lift(w.slot)} sh_lift={_lift(w.slot, shadow=True)}",
            f"gain={w.slot.mean_gain():+.3f}/{w.slot.mean_shadow_gain():+.3f}",
            f"waste={w.slot.mean_waste():.2f}",
            f"topk={w.topk}",
        ]
        if p.enabled:
            parts.append(
                f"prefill_r={p.slot.mean_recall():.2f}/{p.slot.mean_precision():.2f} "
                f"pceil={p.slot.mean_ceiling():.2f} plift={_lift(p.slot)} "
                f"hot={p._union_frac:.2f}"
            )
        if r.enabled:
            rs = r.stats()
            parts.append(
                f"residency={rs['ema_keys']}k/{rs['protect_hits']}p"
                f"@{rs['protect_rate']:.2f} bar={r.min_score:.2f} "
                f"reuse={rs['reuse_rate']:.2f}"
            )
        if pr.enabled:
            parts.append(
                f"prune_live={int(pr.prefetch_enabled)} "
                f"agree={pr.stats()['agree']:.2f} "
                f"mass={pr.mean_mass():.2f}/{pr.mean_base_mass():.2f}"
            )
        tps = self.decode_tok_s
        parts.append(
            f"tok/s={tps:.2f}" if tps is not None else "tok/s=n/a"
        )
        parts.append(
            f"us={self._us_ema * 1e6:.0f} ovh={self.overhead * 100:.1f}%"
            f"{f' shed={self._shed}' if self._shed else ''}"
        )
        parts.append(f"locked={int(w.slot.locked)}")
        _log(" ".join(parts))

    def close(self) -> None:
        try:
            self._persist()
        except Exception:
            pass
        try:
            w = self.wrap
            tps = self.decode_tok_s
            tps_s = f"tok/s={tps:.2f}" if tps is not None else "tok/s=n/a"
            _log(
                f"closed slot={self.slot_id} tokens={w.slot.tokens_seen} "
                f"steps={w.slot.train_steps} r={w.slot.mean_recall():.2f} "
                f"gain={w.slot.mean_gain():+.3f} "
                f"extra={w._extra_used}/{w._extra_issued} "
                f"{tps_s} us={self._us_ema * 1e6:.0f}"
            )
        except Exception:
            pass

    def stats(self) -> dict:
        return {
            "slot": self.slot_id,
            "version": SLOT_VERSION,
            "mode": self._mode,
            "geometry": {
                "feat_dim": self.feat_dim,
                "sketch_dim": self._sketch_dim,
                "wrap": self.wrap.wrap,
                "prefill_layers": self.prefill.n_heads,
                "topk": self.wrap.topk,
                "hidden_dim": self._hidden_dim,
            },
            "cost": {
                "tok_s": (
                    None
                    if self.decode_tok_s is None
                    else round(self.decode_tok_s, 2)
                ),
                "us_per_token": round(self._us_ema * 1e6, 1),
                "overhead": round(self.overhead, 4),
                "shed_level": self._shed,
            },
            "wrap": self.wrap.stats(),
            "prefill": self.prefill.stats(),
            "residency": self.residency.stats(),
            "prune": self.prune.stats(),
            # Flat fields for older log consumers / loader info.
            "locked": self.wrap.slot.locked,
            "prefetch_enabled": self.wrap.slot.prefetch_enabled,
            "tokens_seen": self.wrap.slot.tokens_seen,
            "train_steps": self.wrap.slot.train_steps,
            "mean_recall": round(self.wrap.slot.mean_recall(), 4),
            "mean_precision": round(self.wrap.slot.mean_precision(), 4),
            "mean_ceiling": round(self.wrap.slot.mean_ceiling(), 4),
            "mean_gain": round(self.wrap.slot.mean_gain(), 4),
            "wrap_layers": self.wrap.wrap,
            "topk": self.wrap.topk,
        }


def unlock_slots(slot_dir: Path | None = None) -> list[str]:
    """Clear lock bits on every persisted head slot (no model load)."""
    root = slot_dir or Path(config.SIDECAR_DIR)
    if not root.is_dir():
        return []
    unlocked: list[str] = []
    for npz in sorted(root.glob("*.npz")):
        try:
            data = np.load(npz, allow_pickle=True)
            if "meta" not in data.files:
                continue
            meta = list(data["meta"])
            if len(meta) < 3 or str(meta[1]) != "1":
                continue
            meta[1] = "0"
            meta[2] = "0"
            tmp = npz.with_suffix(".unlock.tmp.npz")
            payload = {k: data[k] for k in data.files if k != "meta"}
            np.savez_compressed(tmp, meta=np.array(meta, dtype=object), **payload)
            os.replace(tmp, npz)
            unlocked.append(npz.name)
            _log(f"unlock file={npz.name}")
        except Exception as e:
            _log(f"unlock failed ({npz.name}): {e!r}")
    return unlocked
