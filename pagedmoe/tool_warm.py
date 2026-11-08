# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prefetch experts that tool-call bodies keep asking for.

A file write is a long strict span. The cache at the start of it still holds
whatever the reasoning span routed. This pool counts real router demand inside
tool calls (prefetch only: a layer still computes fetch(router ids)) and
issues one read burst once the body has started, so the rest of the file
hits those experts instead of discovering them one token at a time.

Wrong guesses waste a read. A precision EMA stops the burst when the plan
is not what the body uses.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from . import config, gpu_trail, streaming
from .sidecar.features import migrate_legacy_slot_files, model_slot_id


def _say(msg: str) -> None:
    print(f"[tool-warm] {msg}", flush=True)


def _pool_path(root: Path, slot_id: str) -> Path:
    return Path(root) / f"{slot_id}_tool_warm.npz"


class ToolWarm:
    """Per-model tool-body expert pool attached to PrefetchRing."""

    def __init__(
        self,
        model_path: str,
        layer_keys: list[str],
        n_experts: int,
        *,
        expert_bytes: int = 0,
        slot_id: str | None = None,
    ):
        self.enabled = bool(config.TOOL_WARM)
        self.layer_keys = list(layer_keys)
        self.n_layers = len(self.layer_keys)
        self.n_experts = int(n_experts)
        self.key_to_idx = {k: i for i, k in enumerate(self.layer_keys)}
        self.expert_bytes = max(0, int(expert_bytes))
        self.budget_bytes = max(0, int(float(config.TOOL_WARM_GB) * (1 << 30)))
        self.skip = max(0, int(config.TOOL_WARM_SKIP))
        self.min_tokens = max(1, int(config.TOOL_WARM_MIN_TOKENS))
        decay = float(config.TOOL_WARM_DECAY)
        self.decay = decay if 0.0 < decay <= 1.0 else 1.0

        root = Path(config.SIDECAR_DIR)
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        if slot_id is None:
            new_id = model_slot_id(model_path, self.n_layers, self.n_experts)
            self.slot_id, _migrated = migrate_legacy_slot_files(
                root, model_path, self.n_layers, self.n_experts, new_id=new_id
            )
        else:
            self.slot_id = slot_id
        self._path = _pool_path(root, self.slot_id)
        self._counts = np.zeros((self.n_layers, self.n_experts), dtype=np.float64)
        self._episodes = 0
        self._plan: list[tuple[str, list[int]]] = []
        self._plan_set: set[tuple[int, int]] = set()
        self._plan_experts = 0
        self._open = False
        self._span_tokens = 0
        self._refreshed = False
        self._token_demand: dict[int, list[int]] = {}
        self._span_demand: set[tuple[int, int]] = set()
        self._prefetched: set[tuple[int, int]] = set()
        self._did_prefetch = False
        self._actuating = True
        self._prec_ema = 0.0
        self._cover_ema = 0.0
        self._scored = 0
        self._dirty = False
        self._load()
        self._rebuild()
        if self.enabled:
            _say(
                f"slot={self.slot_id} budget_gb={self.budget_bytes / (1 << 30):.2f} "
                f"plan={self._plan_experts} episodes={self._episodes} "
                f"act={int(self._actuating)} prec={self._prec_ema:.2f}"
            )

    def set_expert_bytes(self, expert_bytes: int) -> None:
        self.expert_bytes = max(0, int(expert_bytes))
        self._rebuild()

    def drop_token(self) -> None:
        """Prefill boundary. A half-open span is closed so its counts survive."""
        if self._open:
            self._close_span()
        self._token_demand = {}

    def end_turn(self) -> None:
        if self._open:
            self._close_span()
        if self._dirty:
            self._persist()

    def note_demand(self, layer_key: str, expert_ids: list[int]) -> None:
        if not self.enabled or not self._open or streaming._DRAFT_STEP is not None:
            return
        if self._span_tokens < self.skip:
            return
        idx = self.key_to_idx.get(layer_key)
        if idx is None or not expert_ids:
            return
        bucket = self._token_demand.setdefault(idx, [])
        for e in expert_ids:
            ei = int(e)
            if 0 <= ei < self.n_experts:
                bucket.append(ei)
                self._span_demand.add((idx, ei))

    def end_token(self, cache=None) -> None:
        if not self.enabled or not self._open or streaming._DRAFT_STEP is not None:
            self._token_demand = {}
            return
        demand = self._token_demand
        self._token_demand = {}
        if self._span_tokens >= self.skip and demand:
            for li, ids in demand.items():
                row = self._counts[li]
                for e in ids:
                    row[e] += 1.0
            self._dirty = True
        self._span_tokens += 1
        if (
            not self._refreshed
            and self._actuating
            and self._span_tokens == self.skip + self.min_tokens
            and cache is not None
        ):
            self._refreshed = True
            self._rebuild()
            self._prefetch(cache, why="body")

    def maybe_warm(self, cache) -> None:
        """Once at the start of a tool span, and again is a no-op until it ends."""
        if not self.enabled:
            return
        from . import fidelity

        inside = fidelity.in_tool()
        if inside and not self._open:
            self._open = True
            self._span_tokens = 0
            self._refreshed = False
            self._token_demand = {}
            self._span_demand = set()
            self._prefetched = set()
            self._did_prefetch = False
            if self._plan and self._actuating:
                self._prefetch(cache, why="enter")
            return
        if not inside and self._open:
            self._close_span()

    def _prefetch(self, cache, *, why: str) -> None:
        plan = self._plan
        if not plan or not self._actuating:
            return
        try:
            t0 = time.perf_counter()
            gpu_trail.mark(f"tool-warm prefetch {why}, {len(plan)} layer(s)")
            cache.prefetch_many(plan)
            for key, ids in plan:
                li = self.key_to_idx.get(key)
                if li is None:
                    continue
                for e in ids:
                    self._prefetched.add((li, int(e)))
            self._did_prefetch = True
            ms = (time.perf_counter() - t0) * 1000.0
            _say(f"prefetch {why} experts={self._plan_experts} ms={ms:.0f}")
        except Exception as e:  # noqa: BLE001 - never fail decode for a prefetch
            _say(f"prefetch skipped: {type(e).__name__}: {e}")

    def _close_span(self) -> None:
        self._open = False
        long = self._span_tokens >= self.skip + self.min_tokens
        self._span_tokens = 0
        self._token_demand = {}
        if not long:
            self._span_demand = set()
            return
        self._episodes += 1
        if self.decay < 1.0:
            self._counts *= self.decay
        self._score()
        self._rebuild()
        self._dirty = True
        self._span_demand = set()

    def _score(self) -> None:
        plan = self._prefetched
        dem = self._span_demand
        if not self._did_prefetch or not plan or not dem:
            return
        hit = len(dem & plan)
        prec = hit / len(plan)
        cover = hit / len(dem)
        self._scored += 1
        a = 0.3
        if self._scored == 1:
            self._prec_ema, self._cover_ema = prec, cover
        else:
            self._prec_ema = (1.0 - a) * self._prec_ema + a * prec
            self._cover_ema = (1.0 - a) * self._cover_ema + a * cover
        if self._scored >= 2 and self._prec_ema < 0.20:
            self._actuating = False
        elif self._prec_ema >= 0.35:
            self._actuating = True
        _say(
            f"ep={self._episodes} cover={cover:.2f}/{self._cover_ema:.2f} "
            f"prec={prec:.2f}/{self._prec_ema:.2f} plan={len(plan)} dem={len(dem)} "
            f"act={int(self._actuating)}"
        )

    def _rebuild(self) -> None:
        if self.expert_bytes <= 0 or self.budget_bytes <= 0:
            self._plan = []
            self._plan_set = set()
            self._plan_experts = 0
            return
        flat = self._counts.ravel()
        if float(flat.sum()) <= 0.0:
            self._plan = []
            self._plan_set = set()
            self._plan_experts = 0
            return
        order = np.argsort(-flat)
        per_layer: dict[int, list[int]] = {}
        used = 0
        n = 0
        for idx in order:
            if flat[idx] <= 0.0:
                break
            if used + self.expert_bytes > self.budget_bytes:
                break
            li = int(idx) // self.n_experts
            ei = int(idx) % self.n_experts
            per_layer.setdefault(li, []).append(ei)
            used += self.expert_bytes
            n += 1
        self._plan = [
            (self.layer_keys[li], ids)
            for li, ids in sorted(per_layer.items())
            if 0 <= li < self.n_layers and ids
        ]
        self._plan_set = {(li, e) for li, ids in per_layer.items() for e in ids}
        self._plan_experts = n

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            data = np.load(self._path, allow_pickle=False)
            counts = np.asarray(data["counts"], dtype=np.float64)
            if counts.shape != self._counts.shape:
                _say(
                    f"shape {counts.shape} != {self._counts.shape}; starting fresh"
                )
                return
            self._counts = counts
            self._episodes = int(data["episodes"]) if "episodes" in data else 0
            if "prec_ema" in data:
                self._prec_ema = float(data["prec_ema"])
            if "cover_ema" in data:
                self._cover_ema = float(data["cover_ema"])
            if "scored" in data:
                self._scored = int(data["scored"])
            if "actuating" in data:
                self._actuating = bool(int(data["actuating"]))
        except Exception as e:  # noqa: BLE001
            _say(f"load failed: {type(e).__name__}: {e}")

    def _persist(self) -> None:
        if not self._dirty and self._path.is_file():
            return
        counts = self._counts.copy()
        tmp = self._path.parent / (self._path.stem + ".tmp.npz")
        try:
            np.savez_compressed(
                tmp,
                counts=counts,
                episodes=np.asarray(self._episodes, dtype=np.int64),
                prec_ema=np.asarray(self._prec_ema, dtype=np.float64),
                cover_ema=np.asarray(self._cover_ema, dtype=np.float64),
                scored=np.asarray(self._scored, dtype=np.int64),
                actuating=np.asarray(int(self._actuating), dtype=np.int64),
            )
            os.replace(tmp, self._path)
            self._dirty = False
        except Exception as e:  # noqa: BLE001
            _say(f"persist failed: {type(e).__name__}: {e}")
            try:
                if tmp.is_file():
                    tmp.unlink()
            except OSError:
                pass
