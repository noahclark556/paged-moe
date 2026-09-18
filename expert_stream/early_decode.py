# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Early-decode warm pool: prefetch experts that show up right after prefill.

Frequency table over real router demand in the first K decode tokens.
Prefetch-only (quality-safe); not a sidecar head. Wrong guesses waste a
read; fetch() still uses real router ids.

EARLY_DECODE on + TRAIN off (shipped): cold-starts until a high cover/prec
bar, then freezes train. Explicit TRAIN=1 keeps online learning. Counts
decay; actuation holds when cover/prec fall off.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from . import config
from .sidecar.features import migrate_legacy_slot_files, model_slot_id
from .sidecar.slot import _log


def _pool_path(root: Path, slot_id: str) -> Path:
    return Path(root) / f"{slot_id}_early_warm.npz"


class EarlyDecodeWarm:
    """Per-model early-decode warm pool attached to PrefetchRing."""

    def __init__(
        self,
        model_path: str,
        layer_keys: list[str],
        n_experts: int,
        *,
        expert_bytes: int = 0,
        slot_id: str | None = None,
    ):
        self.enabled = bool(config.EARLY_DECODE)
        self._explicit_train = bool(config.EARLY_DECODE_TRAIN)
        self.layer_keys = list(layer_keys)
        self.n_layers = len(self.layer_keys)
        self.n_experts = int(n_experts)
        self.key_to_idx = {k: i for i, k in enumerate(self.layer_keys)}
        self.window = max(1, int(config.EARLY_DECODE_TOKENS))
        self.budget_bytes = max(0, int(float(config.EARLY_DECODE_GB) * (1 << 30)))
        self.expert_bytes = max(0, int(expert_bytes))
        self.min_cover = float(config.EARLY_DECODE_MIN_COVER)
        self.min_prec = float(config.EARLY_DECODE_MIN_PREC)
        self.min_episodes = max(1, int(config.EARLY_DECODE_MIN_EPISODES))
        self.hold_need = max(1, int(config.EARLY_DECODE_HOLD_STREAK))
        self.clear_need = max(1, int(config.EARLY_DECODE_CLEAR_STREAK))
        decay = float(config.EARLY_DECODE_DECAY)
        # Clamp to (0, 1]; 1 = no decay. Values <=0 treated as no decay.
        self.decay = decay if 0.0 < decay <= 1.0 else 1.0
        self.bootstrap_cover = float(config.EARLY_DECODE_BOOTSTRAP_COVER)
        self.bootstrap_prec = float(config.EARLY_DECODE_BOOTSTRAP_PREC)
        self.bootstrap_min_ep = max(1, int(config.EARLY_DECODE_BOOTSTRAP_MIN_EPISODES))
        self.bootstrap_max_ep = max(
            self.bootstrap_min_ep, int(config.EARLY_DECODE_BOOTSTRAP_MAX_EPISODES)
        )
        self.bootstrap_plateau_n = max(2, int(config.EARLY_DECODE_BOOTSTRAP_PLATEAU_N))
        self.bootstrap_plateau_eps = float(config.EARLY_DECODE_BOOTSTRAP_PLATEAU_EPS)

        root = Path(config.SIDECAR_DIR)
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self._root = root

        if slot_id is None:
            new_id = model_slot_id(model_path, self.n_layers, self.n_experts)
            self.slot_id, migrated = migrate_legacy_slot_files(
                root, model_path, self.n_layers, self.n_experts, new_id=new_id
            )
            if migrated:
                _log(
                    f"early-warm migrated path-slot -> content-slot {self.slot_id}"
                )
        else:
            self.slot_id = slot_id

        self._path = _pool_path(root, self.slot_id)
        # EMA counts[layer, expert]. Hot path only adds; rebuild is background.
        self._counts = np.zeros((self.n_layers, self.n_experts), dtype=np.float64)
        self._episodes = 0
        self._tokens_seen = 0
        self._lock = threading.Lock()
        self._token_demand: dict[int, list[int]] = {}
        self._tokens_in_window = 0
        self._warmed = False
        self._plan: list[tuple[str, list[int]]] = []
        self._plan_experts = 0
        self._dirty = False
        self._stop = False
        self._bg: threading.Thread | None = None

        # Per-window scoring (episode = one early window after prefill).
        self._plan_set: set[tuple[int, int]] = set()
        self._window_demand: set[tuple[int, int]] = set()
        self._did_prefetch = False
        self._cover_ema = 0.0
        self._prec_ema = 0.0
        self._scored_episodes = 0
        self._sum_cover = 0.0
        self._sum_prec = 0.0

        # Actuation governor: prefetch only while EMAs clear the bars.
        # Warmup episodes always allow actuation so the pool can learn.
        self._actuating = True
        self._bad_streak = 0
        self._good_streak = 0

        # Cold bootstrap: enable+!train+no weights -> train until high bars.
        self._bootstrap_complete = False
        self._cover_hist: deque[float] = deque(maxlen=self.bootstrap_plateau_n)

        self._load()
        cold = self._is_cold()
        self._bootstrap = bool(
            self.enabled and not self._explicit_train and cold
        )
        self.train = bool(self._explicit_train or self._bootstrap)
        self._rebuild_plan()
        if self.train:
            self._bg = threading.Thread(
                target=self._bg_loop, name="early-decode-warm", daemon=True
            )
            self._bg.start()

        if self.enabled or self.train:
            mode = (
                "explicit-train"
                if self._explicit_train
                else ("bootstrap" if self._bootstrap else "prefetch-only")
            )
            _log(
                f"early-decode warm slot={self.slot_id} "
                f"enabled={int(self.enabled)} train={int(self.train)} "
                f"mode={mode} "
                f"window={self.window} budget_gb={self.budget_bytes / (1 << 30):.2f} "
                f"plan_experts={self._plan_experts} episodes={self._episodes} "
                f"act={int(self._actuating)} decay={self.decay:.3f} "
                f"min_cover={self.min_cover:.2f} min_prec={self.min_prec:.2f}"
                + (
                    f" boot_cover={self.bootstrap_cover:.2f}"
                    if self._bootstrap
                    else ""
                )
            )

    # ------------------------------------------------------------- lifecycle

    def drop_token(self) -> None:
        """Prefill / turn boundary: next decode starts a new early window."""
        self._token_demand = {}
        self._tokens_in_window = 0
        self._warmed = False
        self._plan_set = set()
        self._window_demand = set()
        self._did_prefetch = False

    def note_demand(self, layer_key: str, expert_ids: list[int]) -> None:
        # Score coverage whenever we care about the pool (use and/or train).
        if not self.train and not self.enabled:
            return
        if self._tokens_in_window >= self.window:
            return
        idx = self.key_to_idx.get(layer_key)
        if idx is None or not expert_ids:
            return
        if self.train:
            bucket = self._token_demand.setdefault(idx, [])
            bucket.extend(int(e) for e in expert_ids)
        for e in expert_ids:
            ei = int(e)
            if 0 <= ei < self.n_experts:
                self._window_demand.add((idx, ei))

    def end_token(self) -> None:
        """Close one decode token: fold demand into counts if still in window."""
        demand = self._token_demand
        self._token_demand = {}
        if self._tokens_in_window >= self.window:
            return

        scored = False
        if self.train and demand:
            with self._lock:
                for li, ids in demand.items():
                    if li < 0 or li >= self.n_layers:
                        continue
                    for e in ids:
                        if 0 <= e < self.n_experts:
                            self._counts[li, e] += 1.0
                self._tokens_in_window += 1
                self._tokens_seen += 1
                self._dirty = True
                scored = True
                if self._tokens_in_window >= self.window:
                    self._episodes += 1
        elif (self.enabled or self.train) and self._window_demand:
            # Still advance the window so we can score an episode.
            self._tokens_in_window += 1
            scored = True

        if scored and self._tokens_in_window >= self.window:
            self._log_episode()
            if self.train:
                try:
                    self._apply_decay()
                    self._rebuild_plan()
                except Exception:  # noqa: BLE001
                    pass

    def maybe_warm(self, cache) -> dict:
        """Prefetch the warm plan once per turn (first decode after prefill).

        Prefetch-only: never changes which experts a layer computes.
        Held when the governor says the plan is not earning its keep.

        Returns a small dict for the [ttw] timer: pref_ms, plan_n, actuated.
        """
        out = {"pref_ms": 0.0, "plan_n": 0, "actuated": False}
        if not self.enabled or self._warmed:
            return out
        self._warmed = True
        plan = self._plan
        # Always snapshot the plan for scoring, even when held, so cover/prec
        # measure plan quality (not "we skipped so cover=0").
        self._plan_set = {
            (self.key_to_idx[k], int(e))
            for k, ids in plan
            if k in self.key_to_idx
            for e in ids
        }
        plan_n = sum(len(ids) for _, ids in plan) if plan else 0
        out["plan_n"] = plan_n
        if not self._actuating or not plan or self.budget_bytes <= 0:
            return out
        try:
            # Prefetch only. If this ever feeds ids into fetch() compute, the
            # bit-identical default is gone and the whole feature is wrong.
            t0 = time.perf_counter()
            cache.prefetch_many(plan)
            out["pref_ms"] = (time.perf_counter() - t0) * 1000.0
            out["actuated"] = True
            self._did_prefetch = True
        except Exception as e:  # noqa: BLE001 - never fail decode for warm
            _log(f"early-decode warm prefetch skipped: {type(e).__name__}: {e}")
        return out

    def set_expert_bytes(self, expert_bytes: int) -> None:
        """Update per-expert size once the slab exists (load-time)."""
        self.expert_bytes = max(0, int(expert_bytes))
        self._rebuild_plan()

    def close(self) -> None:
        self._stop = True
        try:
            self._persist()
        except Exception:
            pass

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "train": self.train,
            "explicit_train": self._explicit_train,
            "bootstrap": self._bootstrap,
            "bootstrap_complete": self._bootstrap_complete,
            "slot": self.slot_id,
            "window": self.window,
            "budget_gb": round(self.budget_bytes / (1 << 30), 3),
            "plan_experts": self._plan_experts,
            "episodes": self._episodes,
            "tokens_seen": self._tokens_seen,
            "tokens_in_window": self._tokens_in_window,
            "cover_ema": round(self._cover_ema, 4),
            "prec_ema": round(self._prec_ema, 4),
            "scored_episodes": self._scored_episodes,
            "actuating": self._actuating,
            "decay": self.decay,
        }

    # ------------------------------------------------------------- scoring / gate

    def _is_cold(self) -> bool:
        """True when there is no usable warm-pool state on disk / in memory."""
        if self._bootstrap_complete:
            return False
        if self._episodes > 0 or self._scored_episodes > 0:
            return False
        if float(self._counts.sum()) > 0.0:
            return False
        return True

    def _bootstrap_ready(self) -> bool:
        """High-bar / plateau / max-ep stop for cold-start auto-train."""
        # Bars are high on purpose. Stopping early left a mediocre plan that
        # looked "done" and then just sat there wasting reads.
        n = self._scored_episodes
        if n < self.bootstrap_min_ep:
            return False
        if n >= self.bootstrap_max_ep:
            return True
        hit_bars = (
            self._cover_ema >= self.bootstrap_cover
            and self._prec_ema >= self.bootstrap_prec
        )
        if hit_bars:
            return True
        # Near the cover bar and flat -> treat as max gains for this traffic.
        near = self._cover_ema >= self.bootstrap_cover * 0.9
        if near and len(self._cover_hist) >= self.bootstrap_plateau_n:
            span = max(self._cover_hist) - min(self._cover_hist)
            if span <= self.bootstrap_plateau_eps:
                return True
        return False

    def _finish_bootstrap(self, reason: str) -> None:
        if not self._bootstrap:
            return
        self._bootstrap = False
        self._bootstrap_complete = True
        self.train = False
        self._stop = True
        self._dirty = True
        try:
            self._rebuild_plan()
            self._persist()
        except Exception:  # noqa: BLE001
            pass
        _log(
            f"early-warm bootstrap done ({reason}) "
            f"cover_ema={self._cover_ema:.2f} prec_ema={self._prec_ema:.2f} "
            f"ep={self._scored_episodes} -> prefetch-only"
        )

    def _bars_ok(self) -> bool:
        return (
            self._cover_ema >= self.min_cover
            and self._prec_ema >= self.min_prec
        )

    def _update_governor(self) -> None:
        """Promote / hold actuation from cover/prec EMAs. Train always continues."""
        if self._scored_episodes < self.min_episodes:
            # Warmup: keep actuating so the pool can earn its keep.
            if not self._actuating:
                self._actuating = True
                _log("early-warm act=1 (warmup)")
            self._bad_streak = 0
            self._good_streak = 0
            return

        ok = self._bars_ok()
        if self._actuating:
            if ok:
                self._bad_streak = 0
            else:
                self._bad_streak += 1
                self._good_streak = 0
                if self._bad_streak >= self.hold_need:
                    self._actuating = False
                    self._bad_streak = 0
                    _log(
                        f"early-warm act=0 hold "
                        f"cover_ema={self._cover_ema:.2f}<{self.min_cover:.2f} "
                        f"or prec_ema={self._prec_ema:.2f}<{self.min_prec:.2f}"
                    )
        else:
            if ok:
                self._good_streak += 1
                self._bad_streak = 0
                if self._good_streak >= self.clear_need:
                    self._actuating = True
                    self._good_streak = 0
                    _log(
                        f"early-warm act=1 clear "
                        f"cover_ema={self._cover_ema:.2f} "
                        f"prec_ema={self._prec_ema:.2f}"
                    )
            else:
                self._good_streak = 0

    def _apply_decay(self) -> None:
        if self.decay >= 1.0:
            return
        with self._lock:
            self._counts *= self.decay
            self._dirty = True

    def _log_episode(self) -> None:
        """Emit one line after an early window so episode progress is visible."""
        demand = self._window_demand
        plan = self._plan_set
        dem_n = len(demand)
        plan_n = len(plan) if plan else self._plan_experts
        hit = len(demand & plan) if plan else 0
        cover = (hit / dem_n) if dem_n else 0.0
        prec = (hit / plan_n) if plan_n else 0.0

        self._scored_episodes += 1
        n = self._scored_episodes
        # EMA toward recent episodes so the climb shows up in the status line.
        alpha = 0.2
        if n == 1:
            self._cover_ema = cover
            self._prec_ema = prec
        else:
            self._cover_ema = (1.0 - alpha) * self._cover_ema + alpha * cover
            self._prec_ema = (1.0 - alpha) * self._prec_ema + alpha * prec
        self._sum_cover += cover
        self._sum_prec += prec
        mean_cover = self._sum_cover / n
        mean_prec = self._sum_prec / n

        self._cover_hist.append(self._cover_ema)
        self._update_governor()

        if self._bootstrap and self._bootstrap_ready():
            if self._scored_episodes >= self.bootstrap_max_ep:
                reason = "max-ep"
            elif (
                self._cover_ema >= self.bootstrap_cover
                and self._prec_ema >= self.bootstrap_prec
            ):
                reason = "bars"
            else:
                reason = "plateau"
            self._finish_bootstrap(reason)

        pref = "1" if self._did_prefetch else "0"
        boot = (
            "boot"
            if self._bootstrap
            else ("done" if self._bootstrap_complete and not self._explicit_train else "-")
        )
        _log(
            f"early-warm ep={self._episodes or n} "
            f"cover={cover:.2f}/{self._cover_ema:.2f} "
            f"prec={prec:.2f}/{self._prec_ema:.2f} "
            f"mean_cover={mean_cover:.2f} mean_prec={mean_prec:.2f} "
            f"plan={plan_n} dem={dem_n} hit={hit} "
            f"pref={pref} act={int(self._actuating)} {boot} "
            f"tok={self._tokens_in_window}/{self.window}"
        )
        # Persist score EMAs with the counts (best-effort; bg also persists).
        self._dirty = True

    # ------------------------------------------------------------- persist

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            data = np.load(self._path, allow_pickle=False)
            counts = np.asarray(data["counts"], dtype=np.float64)
            if counts.shape != self._counts.shape:
                _log(
                    f"early-decode warm shape mismatch "
                    f"{counts.shape} != {self._counts.shape}; starting fresh"
                )
                return
            self._counts = counts
            self._episodes = int(data["episodes"]) if "episodes" in data else 0
            self._tokens_seen = (
                int(data["tokens_seen"]) if "tokens_seen" in data else 0
            )
            if "cover_ema" in data:
                self._cover_ema = float(data["cover_ema"])
            if "prec_ema" in data:
                self._prec_ema = float(data["prec_ema"])
            if "scored_episodes" in data:
                self._scored_episodes = int(data["scored_episodes"])
            if "sum_cover" in data:
                self._sum_cover = float(data["sum_cover"])
            if "sum_prec" in data:
                self._sum_prec = float(data["sum_prec"])
            if "actuating" in data:
                self._actuating = bool(int(data["actuating"]))
            if "bad_streak" in data:
                self._bad_streak = int(data["bad_streak"])
            if "good_streak" in data:
                self._good_streak = int(data["good_streak"])
            if "bootstrap_complete" in data:
                self._bootstrap_complete = bool(int(data["bootstrap_complete"]))
            # Re-apply gate from loaded EMAs (bars / warmup may have changed).
            if self._scored_episodes >= self.min_episodes and not self._bars_ok():
                self._actuating = False
        except Exception as e:  # noqa: BLE001
            _log(f"early-decode warm load failed: {type(e).__name__}: {e}")

    def _persist(self) -> None:
        with self._lock:
            if not self._dirty and self._path.is_file():
                return
            counts = self._counts.copy()
            episodes = self._episodes
            tokens_seen = self._tokens_seen
            cover_ema = self._cover_ema
            prec_ema = self._prec_ema
            scored = self._scored_episodes
            sum_cover = self._sum_cover
            sum_prec = self._sum_prec
            actuating = int(self._actuating)
            bad_streak = self._bad_streak
            good_streak = self._good_streak
            bootstrap_complete = int(self._bootstrap_complete)
            self._dirty = False
        tmp = self._path.parent / (self._path.stem + ".tmp.npz")
        try:
            np.savez_compressed(
                tmp,
                counts=counts,
                episodes=np.asarray(episodes, dtype=np.int64),
                tokens_seen=np.asarray(tokens_seen, dtype=np.int64),
                n_layers=np.asarray(self.n_layers, dtype=np.int64),
                n_experts=np.asarray(self.n_experts, dtype=np.int64),
                window=np.asarray(self.window, dtype=np.int64),
                cover_ema=np.asarray(cover_ema, dtype=np.float64),
                prec_ema=np.asarray(prec_ema, dtype=np.float64),
                scored_episodes=np.asarray(scored, dtype=np.int64),
                sum_cover=np.asarray(sum_cover, dtype=np.float64),
                sum_prec=np.asarray(sum_prec, dtype=np.float64),
                actuating=np.asarray(actuating, dtype=np.int64),
                bad_streak=np.asarray(bad_streak, dtype=np.int64),
                good_streak=np.asarray(good_streak, dtype=np.int64),
                bootstrap_complete=np.asarray(bootstrap_complete, dtype=np.int64),
            )
            os.replace(tmp, self._path)
        except Exception as e:  # noqa: BLE001
            _log(f"early-decode warm persist failed: {type(e).__name__}: {e}")
            try:
                if tmp.is_file():
                    tmp.unlink()
            except OSError:
                pass
            stray = Path(str(tmp) + ".npz")
            try:
                if stray.is_file():
                    stray.unlink()
            except OSError:
                pass

    def _rebuild_plan(self) -> None:
        """Rank (layer, expert) by count; pack until budget_bytes."""
        with self._lock:
            counts = self._counts.copy()
        if self.budget_bytes <= 0 or self.expert_bytes <= 0:
            per_layer = max(1, min(8, self.n_experts // 16 or 1))
            plan: list[tuple[str, list[int]]] = []
            total = 0
            for li, key in enumerate(self.layer_keys):
                row = counts[li]
                if float(row.sum()) <= 0.0:
                    continue
                k = min(per_layer, self.n_experts)
                ids = np.argpartition(row, -k)[-k:]
                ids = ids[np.argsort(-row[ids])]
                kept = [int(e) for e in ids if row[e] > 0]
                if kept:
                    plan.append((key, kept))
                    total += len(kept)
            self._plan = plan
            self._plan_experts = total
            return

        flat = counts.ravel()
        if float(flat.sum()) <= 0.0:
            self._plan = []
            self._plan_experts = 0
            return
        order = np.argsort(-flat)
        per_layer: dict[int, list[int]] = {}
        used = 0
        for idx in order:
            score = flat[idx]
            if score <= 0.0:
                break
            if used + self.expert_bytes > self.budget_bytes:
                break
            li = int(idx) // self.n_experts
            ei = int(idx) % self.n_experts
            per_layer.setdefault(li, []).append(ei)
            used += self.expert_bytes
        plan = [
            (self.layer_keys[li], ids)
            for li, ids in sorted(per_layer.items())
            if 0 <= li < self.n_layers and ids
        ]
        self._plan = plan
        self._plan_experts = sum(len(ids) for _, ids in plan)

    def _bg_loop(self) -> None:
        """Persist + rebuild plan off the decode hot path."""
        while not self._stop:
            time.sleep(5.0)
            try:
                if self._dirty:
                    self._rebuild_plan()
                    self._persist()
            except Exception as e:  # noqa: BLE001
                _log(f"early-decode warm bg: {type(e).__name__}: {e}")
