# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared live/shadow slot: batched linear heads with promote, demote, lock, npz.

Layout (v4): weights are one contiguous (n_heads, feat_dim, n_experts) tensor
plus an (n_heads, n_experts) bias, so a token's whole prediction is a single
BLAS call instead of a Python loop over layer-heads. The bias alone learns each
layer's expert popularity, which is most of the achievable accuracy in the first
few hundred tokens.

Training is listwise softmax cross-entropy against the observed demand set,
with optional AdaGrad (per-parameter step) and weight decay. Only the shadow
copy trains; `live` changes solely through promote, so an actuating head never
drifts mid-flight.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import config
from .features import SLOT_VERSION


def _debug_enabled() -> bool:
    """Sidecar chatter is opt-in (``EXPERT_STREAM_SIDECAR_DEBUG=1`` or ``PAGED_MOE_DEBUG=1``)."""
    for key in ("EXPERT_STREAM_SIDECAR_DEBUG", "PAGED_MOE_DEBUG"):
        raw = (os.environ.get(key) or "").strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True
    return False


def _log(msg: str) -> None:
    if not _debug_enabled():
        return
    print(f"[sidecar] {msg}", flush=True)


def _mean(d) -> float:
    return float(sum(d) / len(d)) if d else 0.0


@dataclass
class ExpertHeadSlot:
    """Per-head adapter: live/shadow weights over experts, one row per layer-head."""

    name: str
    slot_id: str
    n_layers: int
    n_experts: int
    n_heads: int
    feat_dim: int
    live: np.ndarray = field(default=None)  # (H, F, E)
    shadow: np.ndarray = field(default=None)
    live_b: np.ndarray = field(default=None)  # (H, E)
    shadow_b: np.ndarray = field(default=None)
    g2: np.ndarray = field(default=None)  # AdaGrad accumulator for shadow
    g2_b: np.ndarray = field(default=None)
    freq: np.ndarray = field(default=None)  # (H, E) EMA demand frequency prior
    locked: bool = False
    lock_streak: int = 0
    tokens_seen: int = 0
    train_steps: int = 0
    recalls: deque = field(default_factory=lambda: deque(maxlen=64))
    precisions: deque = field(default_factory=lambda: deque(maxlen=64))
    ceilings: deque = field(default_factory=lambda: deque(maxlen=64))
    shadow_recalls: deque = field(default_factory=lambda: deque(maxlen=64))
    shadow_precisions: deque = field(default_factory=lambda: deque(maxlen=64))
    # Marginal value of the experts this head adds on top of the engine's own
    # last-token heuristic: the only number that says whether it is earning its
    # reads. gains are recall added; wastes are added-but-unused fraction.
    gains: deque = field(default_factory=lambda: deque(maxlen=64))
    wastes: deque = field(default_factory=lambda: deque(maxlen=64))
    shadow_gains: deque = field(default_factory=lambda: deque(maxlen=64))
    demand_sizes: deque = field(default_factory=lambda: deque(maxlen=256))
    prefetch_enabled: bool = False
    path: Path | None = None
    _grad: np.ndarray | None = None
    _scratch: np.ndarray | None = None

    @classmethod
    def create(
        cls,
        name: str,
        slot_id: str,
        n_layers: int,
        n_experts: int,
        n_heads: int,
        feat_dim: int,
        path: Path | None = None,
    ) -> "ExpertHeadSlot":
        n_heads = max(1, min(int(n_heads), int(n_layers)))
        n_experts = max(1, int(n_experts))
        feat_dim = max(8, int(feat_dim))
        scale = np.float32(0.1 / np.sqrt(feat_dim))
        live = (
            np.random.randn(n_heads, feat_dim, n_experts).astype(np.float32) * scale
        )
        slot = cls(
            name=name,
            slot_id=slot_id,
            n_layers=int(n_layers),
            n_experts=n_experts,
            n_heads=n_heads,
            feat_dim=feat_dim,
            live=live,
            shadow=live.copy(),
            live_b=np.zeros((n_heads, n_experts), dtype=np.float32),
            shadow_b=np.zeros((n_heads, n_experts), dtype=np.float32),
            g2=np.zeros((n_heads, feat_dim, n_experts), dtype=np.float32),
            g2_b=np.zeros((n_heads, n_experts), dtype=np.float32),
            freq=np.zeros((n_heads, n_experts), dtype=np.float32),
            path=path,
        )
        if path is not None:
            slot.load(path)
        return slot

    # --------------------------------------------------------------- inference

    def _scores(self, features: np.ndarray, *, use_shadow: bool) -> np.ndarray:
        """(H, E) probabilities, frequency-prior blended while the head is cold."""
        W = self.shadow if use_shadow else self.live
        b = self.shadow_b if use_shadow else self.live_b
        logits = np.tensordot(
            np.asarray(features, dtype=np.float32), W, axes=([0], [1])
        )
        logits += b
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True) + 1e-8
        pw = self.prior_weight()
        if pw > 0.0:
            total = self.freq.sum(axis=1, keepdims=True)
            if float(total.max()) > 0.0:
                prior = self.freq / (total + 1e-8)
                probs = (1.0 - pw) * probs + pw * prior
        return probs

    def prior_weight(self) -> float:
        """Blend weight for the popularity prior - decays as the head learns."""
        w0 = float(config.SIDECAR_PRIOR_W)
        if w0 <= 0.0:
            return 0.0
        decay = max(1, int(config.SIDECAR_PRIOR_DECAY))
        return float(max(0.0, w0 * (1.0 - min(1.0, self.train_steps / decay))))

    def predict_topk(
        self,
        features: np.ndarray,
        k: int,
        *,
        use_shadow: bool = False,
        min_score: float = 0.0,
        max_heads: int | None = None,
    ) -> dict[int, list[int]]:
        k = max(1, min(int(k), self.n_experts))
        limit = self.n_heads if max_heads is None else min(self.n_heads, int(max_heads))
        if limit <= 0:
            return {}
        probs = self._scores(features, use_shadow=use_shadow)[:limit]
        if k >= self.n_experts:
            order = np.argsort(-probs, axis=1)
        else:
            part = np.argpartition(-probs, k - 1, axis=1)[:, :k]
            rows = np.arange(part.shape[0])[:, None]
            order = part[rows, np.argsort(-probs[rows, part], axis=1)]
        out: dict[int, list[int]] = {}
        rows = np.arange(order.shape[0])[:, None]
        keepmask = probs[rows, order] >= float(min_score)
        keepmask[:, 0] = True  # always keep the head's single best guess
        for i in range(order.shape[0]):
            out[i] = [int(e) for e in order[i][keepmask[i]]]
        return out

    # ---------------------------------------------------------------- training

    def train_shadow(
        self, features: np.ndarray, demand: dict[int, list[int]], lr: float
    ) -> None:
        if self.locked or not demand:
            return
        x = np.asarray(features, dtype=np.float32)
        target = np.zeros((self.n_heads, self.n_experts), dtype=np.float32)
        active = np.zeros(self.n_heads, dtype=bool)
        for i, ids in demand.items():
            hi = int(i)
            if hi < 0 or hi >= self.n_heads or len(ids) == 0:
                continue
            arr = np.asarray(ids, dtype=np.int64)
            arr = arr[(arr >= 0) & (arr < self.n_experts)]
            if arr.size == 0:
                continue
            target[hi, arr] = np.float32(1.0 / arr.size)
            active[hi] = True
            self.demand_sizes.append(int(arr.size))
        if not active.any():
            return
        # Frequency prior tracks the marginal demand distribution; it is what
        # makes a cold head useful before the weights mean anything.
        self.freq *= np.float32(0.999)
        self.freq += target > 0

        logits = np.tensordot(x, self.shadow, axes=([0], [1])) + self.shadow_b
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True) + 1e-8
        delta = probs - target
        delta[~active] = 0.0

        # Preallocated scratch: the gradient is (H, F, E) and a naive expression
        # allocates half a dozen of them per token.
        if self._grad is None:
            self._grad = np.empty_like(self.shadow)
            self._scratch = np.empty_like(self.shadow)
        grad, scratch = self._grad, self._scratch
        np.multiply(x[None, :, None], delta[:, None, :], out=grad)
        lr = np.float32(lr)
        if bool(config.SIDECAR_ADAGRAD) and self.g2 is not None:
            np.multiply(grad, grad, out=scratch)
            self.g2 += scratch
            np.sqrt(self.g2, out=scratch)
            scratch += np.float32(1e-6)
            np.divide(grad, scratch, out=grad)
            grad *= lr
            self.shadow -= grad
            self.g2_b += delta * delta
            self.shadow_b -= lr * delta / (np.sqrt(self.g2_b) + np.float32(1e-6))
        else:
            grad *= lr
            self.shadow -= grad
            self.shadow_b -= lr * delta
        wd = np.float32(config.SIDECAR_WD)
        if wd > 0:
            self.shadow *= np.float32(1.0) - lr * wd
        self.train_steps += 1

    # ----------------------------------------------------------------- scoring

    def score_prediction(
        self, predicted: dict[int, list[int]], demand: dict[int, list[int]]
    ) -> tuple[float, float]:
        hit = need = issued = 0
        for i in range(self.n_heads):
            want = set(demand.get(i, ()))
            got = list(predicted.get(i, ()))
            issued += len(got)
            if not want:
                continue
            need += len(want)
            hit += len(want.intersection(got))
        recall = (hit / need) if need else 0.0
        precision = (hit / issued) if issued else 0.0
        return recall, precision

    def score_margin(
        self,
        predicted: dict[int, list[int]],
        baseline: dict[int, list[int]],
        demand: dict[int, list[int]],
    ) -> tuple[float, float]:
        """(recall added beyond baseline, share of added guesses that went unused).

        The engine already prefetches the previous token's experts, so only the
        difference is chargeable to this head - both for credit and for cost.
        """
        hit = need = issued = 0
        for i in range(self.n_heads):
            want = set(demand.get(i, ()))
            extra = set(predicted.get(i, ())) - set(baseline.get(i, ()))
            issued += len(extra)
            if not want:
                continue
            need += len(want)
            hit += len(want.intersection(extra))
        gain = (hit / need) if need else 0.0
        waste = (1.0 - hit / issued) if issued else 0.0
        return gain, waste

    def mean_recall(self) -> float:
        return _mean(self.recalls)

    def mean_precision(self) -> float:
        return _mean(self.precisions)

    def mean_ceiling(self) -> float:
        return _mean(self.ceilings)

    def mean_shadow_recall(self) -> float:
        return _mean(self.shadow_recalls)

    def mean_shadow_precision(self) -> float:
        return _mean(self.shadow_precisions)

    def mean_gain(self) -> float:
        return _mean(self.gains)

    def mean_waste(self) -> float:
        return _mean(self.wastes)

    def mean_shadow_gain(self) -> float:
        return _mean(self.shadow_gains)

    def mean_demand(self) -> float:
        return _mean(self.demand_sizes)

    def lift_vs_ceiling(self, *, shadow: bool = False) -> float | None:
        """Recall minus the copy-the-last-token baseline. None until sampled."""
        if not self.ceilings:
            return None
        recall = self.mean_shadow_recall() if shadow else self.mean_recall()
        return recall - self.mean_ceiling()

    # --------------------------------------------------------- promote / lock

    def promote_if_better(
        self,
        min_recall: float,
        min_precision: float,
        *,
        min_gain: float | None = None,
    ) -> bool:
        """Go live (or refresh live weights) when the shadow has earned it.

        Gate on *marginal* recall gain when we have that measurement: an
        absolute recall bar promotes heads that only re-derive the baseline and
        blocks heads that add real coverage on top of a low baseline.
        """
        if self.locked:
            return False
        if len(self.shadow_recalls) < max(8, config.SIDECAR_WINDOW // 2):
            return False
        sr, sp = self.mean_shadow_recall(), self.mean_shadow_precision()
        if sp < min_precision:
            return False
        if min_gain is not None and self.shadow_gains:
            if self.mean_shadow_gain() < float(min_gain):
                return False
        elif sr < min_recall:
            return False
        if not self.prefetch_enabled:
            self.live = self.shadow.copy()
            self.live_b = self.shadow_b.copy()
            self.prefetch_enabled = True
            return True
        better_gain = (
            self.shadow_gains
            and self.gains
            and self.mean_shadow_gain() >= self.mean_gain() + 0.01
        )
        if better_gain or (sr >= self.mean_recall() + 0.03 and sp >= self.mean_precision()):
            self.live = self.shadow.copy()
            self.live_b = self.shadow_b.copy()
        return False

    def maybe_lock(self, min_recall: float, min_precision: float, need_hits: int) -> bool:
        if self.locked or need_hits <= 0:
            return False
        if self.tokens_seen < int(config.SIDECAR_LOCK_MIN_TOKENS):
            self.lock_streak = 0
            return False
        recall_bar = max(
            min_recall, self.mean_ceiling() + float(config.SIDECAR_LOCK_OVER_CEILING)
        )
        good = (
            self.prefetch_enabled
            and self.mean_recall() >= recall_bar
            and self.mean_precision() >= min_precision
        )
        self.lock_streak = self.lock_streak + 1 if good else 0
        if self.lock_streak >= need_hits:
            self.locked = True
            self.live = self.shadow.copy()
            self.live_b = self.shadow_b.copy()
            return True
        return False

    def demote(self) -> None:
        self.prefetch_enabled = False
        self.lock_streak = 0

    def unlock(self) -> None:
        self.locked = False
        self.prefetch_enabled = False
        self.lock_streak = 0

    # ------------------------------------------------------------- persistence

    def save(self, path: Path | None = None) -> None:
        path = path or self.path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(
            f".{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.npz"
        )
        meta = np.array(
            [
                self.slot_id,
                str(int(self.locked)),
                str(int(self.prefetch_enabled)),
                str(self.tokens_seen),
                str(self.train_steps),
                str(self.n_layers),
                str(self.n_experts),
                str(self.n_heads),
                str(self.feat_dim),
                str(SLOT_VERSION),
                self.name,
            ],
            dtype=object,
        )
        try:
            np.savez_compressed(
                tmp,
                meta=meta,
                live=self.live,
                shadow=self.shadow,
                live_b=self.live_b,
                shadow_b=self.shadow_b,
                g2=self.g2,
                g2_b=self.g2_b,
                freq=self.freq,
                # Rolling quality so a restart resumes with its earned state
                # instead of re-proving itself from zero.
                windows=np.array(
                    [
                        self.mean_recall(),
                        self.mean_precision(),
                        self.mean_ceiling(),
                        self.mean_gain(),
                        self.mean_waste(),
                        self.mean_demand(),
                    ],
                    dtype=np.float32,
                ),
            )
            os.replace(tmp, path.with_suffix(".npz"))
        except Exception as e:  # never let persistence break generation
            _log(f"{self.name} save failed: {e!r}")
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def load(self, path: Path) -> None:
        npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
        if not npz_path.exists():
            return
        try:
            data = np.load(npz_path, allow_pickle=True)
            meta = data["meta"]
            version = int(meta[9]) if len(meta) > 9 else 1
            if version != SLOT_VERSION:
                _log(
                    f"{self.name} slot {self.slot_id} v{version} != v{SLOT_VERSION} - fresh"
                )
                return
            n_layers, n_experts, feat_dim = int(meta[5]), int(meta[6]), int(meta[8])
            if (
                n_layers != self.n_layers
                or n_experts != self.n_experts
                or feat_dim != self.feat_dim
            ):
                _log(f"{self.name} slot geometry mismatch - fresh")
                return
            live = np.asarray(data["live"], dtype=np.float32)
            shadow = np.asarray(data["shadow"], dtype=np.float32)
            heads = min(self.n_heads, int(live.shape[0]))
            self.live[:heads] = live[:heads]
            self.shadow[:heads] = shadow[:heads]
            self.live_b[:heads] = np.asarray(data["live_b"], dtype=np.float32)[:heads]
            self.shadow_b[:heads] = np.asarray(data["shadow_b"], dtype=np.float32)[
                :heads
            ]
            if "g2" in data.files:
                self.g2[:heads] = np.asarray(data["g2"], dtype=np.float32)[:heads]
                self.g2_b[:heads] = np.asarray(data["g2_b"], dtype=np.float32)[:heads]
            if "freq" in data.files:
                self.freq[:heads] = np.asarray(data["freq"], dtype=np.float32)[:heads]
            self.locked = str(meta[1]) == "1"
            # A head that had earned actuation keeps it; otherwise a restart
            # silently costs every session its first few hundred tokens.
            self.prefetch_enabled = self.locked or str(meta[2]) == "1"
            self.tokens_seen = int(meta[3])
            self.train_steps = int(meta[4])
            if "windows" in data.files:
                w = np.asarray(data["windows"], dtype=np.float32).tolist()
                seed = max(4, config.SIDECAR_WINDOW // 4)
                for dq, val in (
                    (self.recalls, w[0]),
                    (self.precisions, w[1]),
                    (self.ceilings, w[2]),
                    (self.gains, w[3]),
                    (self.wastes, w[4]),
                ):
                    if val > 0:
                        dq.extend([float(val)] * seed)
                if len(w) > 5 and w[5] > 0:
                    self.demand_sizes.append(float(w[5]))
            self.path = npz_path
        except Exception as e:
            _log(f"{self.name} load failed ({npz_path.name}): {e!r} - fresh")


# Back-compat alias + factory matching the old ModelSlot.create signature.
class ModelSlot(ExpertHeadSlot):
    @classmethod
    def create(  # type: ignore[override]
        cls,
        slot_id: str,
        n_layers: int,
        n_experts: int,
        wrap: int,
        feat_dim: int,
        path: Path | None = None,
        name: str = "wrap",
    ) -> "ModelSlot":
        return ExpertHeadSlot.create(  # type: ignore[return-value]
            name, slot_id, n_layers, n_experts, wrap, feat_dim, path=path
        )
