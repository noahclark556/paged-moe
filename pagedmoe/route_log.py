# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Routing record by KV position, for verify passes that copy context.

Every forward already brings each MoE layer's router ids to the host. This
keeps them, per absolute position, strongest expert first. When verify decode
drafts a copy of text that is already in the context, the rows of the pass
feed the same tokens the copied positions did, so the experts those positions
routed to are a guess at what the pass will read. Router choice in early
layers follows the token closely; middle layers depend more on context, which
the precision gate below measures instead of assuming.

`arm` starts the reads a pass will want before it runs, every layer at once,
instead of one layer at a time as the pass reaches it. Timing only: a layer
still computes fetch(router ids).

How alike a copy routes depends on depth (bench/route_replay_probe.py, top-k
overlap of a text and its copy): Flash 0.96 at layer 0, 0.70 over the first
quarter, 0.24 over the last; MiniMax-M3 0.71 / 0.35 / 0.28. So trust is per
layer. Every pass scores each layer before anything is read (the experts the
plan names that nothing holds, against what the pass then demanded), and a
layer gets reads only once its own precision clears VERIFY_REPLAY_MIN_PREC.
Measuring costs no bytes; a layer that guesses badly never reads.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from . import config

# (start position, token ids mx array or None) of the forward in progress.
# Set by the model wrapper, read by StreamedSwitchGLU. None outside a forward,
# so stray calls (route prediction runs routers directly) record nothing.
_FWD: tuple | None = None
# Set once a forward's tokens were written (first MoE layer of that forward).
_FWD_TOKENS_DONE = False

_PATCHED: set[type] = set()


def _cache_offset(cache) -> int | None:
    """Absolute position of the first row this forward writes."""
    if not cache:
        return None
    for c in cache:
        off = getattr(c, "offset", None)
        if isinstance(off, (int, np.integer)):
            return int(off)
    return None


def install(model, log: "RouteLog") -> bool:
    """Wrap the model's forward so MoE layers know the positions of their rows."""
    cls = type(model)
    model._pagedmoe_routes = log
    if cls in _PATCHED:
        return True
    orig = cls.__call__

    def __call__(self, inputs, *args, **kw):  # noqa: N807 - mirrors the wrapped signature
        global _FWD, _FWD_TOKENS_DONE
        log_ = getattr(self, "_pagedmoe_routes", None)
        prev = (_FWD, _FWD_TOKENS_DONE)
        cache = kw.get("cache", args[0] if args else None)
        start = None
        if (
            log_ is not None
            and isinstance(inputs, mx.array)
            and inputs.ndim == 2
            and int(inputs.shape[0]) == 1
        ):
            start = _cache_offset(cache)
        _FWD = (start, inputs) if start is not None else None
        _FWD_TOKENS_DONE = False
        try:
            return orig(self, inputs, *args, **kw)
        finally:
            _FWD, _FWD_TOKENS_DONE = prev

    __call__.__wrapped__ = orig
    cls.__call__ = __call__
    _PATCHED.add(cls)
    return True


def uninstall() -> None:
    """Restore stock forwards (tests)."""
    for cls in list(_PATCHED):
        orig = getattr(cls.__call__, "__wrapped__", None)
        if orig is not None:
            cls.__call__ = orig
        _PATCHED.discard(cls)


class RouteLog:
    """Router ids per (position, layer), a ring of VERIFY_REPLAY_TOKENS positions."""

    # Scored passes before the plan is trusted.
    MIN_PASSES = 4
    DECAY = 0.85

    def __init__(self, layer_keys: list[str], cap: int | None = None):
        self.layer_keys = list(layer_keys)
        self.li = {lk: i for i, lk in enumerate(self.layer_keys)}
        self.cap = max(256, int(cap if cap is not None else config.VERIFY_REPLAY_TOKENS))
        # Sized by the first layer recorded (the router's top-k).
        self.k = 0
        self.ids: np.ndarray | None = None
        self.tok = np.full(self.cap, -1, dtype=np.int64)
        self.pos = np.full(self.cap, -1, dtype=np.int64)
        # Kept experts per row, per layer, as decode actually computed them.
        # Prefill records the router's full top-k; this says how many of those
        # a plain-fidelity row keeps.
        self.kept: np.ndarray | None = None
        # Per-layer scoring of what plans named (decayed counts).
        n_layers = len(self.layer_keys)
        self.armed: dict[int, set[int]] = {}
        self.named = np.zeros(n_layers)
        self.used = np.zeros(n_layers)
        self.scored = np.zeros(n_layers, dtype=np.int64)
        self.stats = {"planned": 0, "named": 0, "used": 0, "issued": 0, "passes": 0}

    # ------------------------------------------------------------------ record
    def record(self, layer_key: str, start: int, inds: np.ndarray, weights=None) -> None:
        """Rows of one layer at positions start..start+N-1, strongest first."""
        li = self.li.get(layer_key)
        if li is None or inds.ndim != 2:
            return
        n, k = inds.shape
        if n <= 0 or k <= 0:
            return
        if self.ids is None:
            self.k = int(k)
            self.ids = np.full((self.cap, len(self.layer_keys), self.k), -1, dtype=np.int16)
            self.kept = np.full(len(self.layer_keys), float(self.k), dtype=np.float64)
        k = min(k, self.k)
        if weights is not None and weights.shape == inds.shape:
            order = np.argsort(-weights, axis=1, kind="stable")
            inds = np.take_along_axis(inds, order, axis=1)
        if n > self.cap:
            inds = inds[-self.cap :]
            start += n - self.cap
            n = self.cap
        rows = (start + np.arange(n)) % self.cap
        self.ids[rows, li, :k] = inds[:, :k]
        if k < self.k:
            self.ids[rows, li, k:] = -1

    def record_tokens(self, start: int, tokens) -> None:
        toks = np.asarray(tokens).reshape(-1).astype(np.int64)
        n = len(toks)
        if n <= 0:
            return
        if n > self.cap:
            toks = toks[-self.cap :]
            start += n - self.cap
            n = self.cap
        p = start + np.arange(n)
        rows = p % self.cap
        self.tok[rows] = toks
        self.pos[rows] = p

    def note_kept(self, layer_key: str, kept_per_row: float) -> None:
        li = self.li.get(layer_key)
        if li is not None and self.kept is not None:
            self.kept[li] += 0.1 * (float(kept_per_row) - self.kept[li])

    # -------------------------------------------------------------------- plan
    def plan(self, src: int, tokens: list[int]) -> list | None:
        """Per row, per layer: the experts the copied positions routed to.

        `src` is the context position whose token `tokens[0]` copies; row j
        copies position src + j. A row whose position is not on record (or
        holds another token) is None. None when no row is on record.
        """
        if self.ids is None or self.kept is None:
            return None
        out: list = []
        any_hit = False
        m = np.clip(np.floor(self.kept + 0.5).astype(int), 1, self.k)
        for j, t in enumerate(tokens):
            p = int(src) + j
            r = p % self.cap
            if p < 0 or self.pos[r] != p or self.tok[r] != int(t):
                out.append(None)
                continue
            row = self.ids[r]
            per = []
            for li in range(len(self.layer_keys)):
                v = row[li, : m[li]]
                per.append(v[v >= 0])
            out.append(per)
            any_hit = True
        if not any_hit:
            return None
        self.stats["planned"] += 1
        return out

    def union(self, plan: list, rows: int) -> list[set[int]]:
        """Per layer, every expert any of the first `rows` rows names."""
        sets: list[set[int]] = [set() for _ in self.layer_keys]
        for per in plan[:rows]:
            if per is None:
                continue
            for li, v in enumerate(per):
                if len(v):
                    sets[li].update(int(e) for e in v)
        return sets

    # --------------------------------------------------------------------- arm
    def trusted(self) -> np.ndarray:
        """Per layer: has the plan named that layer's cold experts well enough?"""
        prec = np.divide(self.used, self.named, out=np.zeros_like(self.used), where=self.named > 0)
        return (self.scored >= self.MIN_PASSES) & (prec >= float(config.VERIFY_REPLAY_MIN_PREC))

    def arm(self, cache, plan: list, rows: int) -> int:
        """Score the first `rows` rows against the pass; read ahead on trusted layers.

        Prefetch only. Returns how many reads were started.
        """
        sets = self.union(plan, rows)
        trust = self.trusted()
        clock = int(getattr(cache, "_prefetch_clock", 0))
        batch = []
        self.armed = {}
        named = 0
        for li, ids in enumerate(sets):
            if not ids:
                continue
            lk = self.layer_keys[li]
            want = sorted(ids - cache.free_ids(lk, ids))
            if not want:
                continue
            self.armed[li] = set(want)
            named += len(want)
            if trust[li]:
                batch.append((lk, want, clock + li + 1))
        self.stats["named"] += named
        if not batch:
            return 0
        cache.prefetch_many(batch)
        n = sum(len(b[1]) for b in batch)
        self.stats["issued"] += n
        return n

    def _book(self, li: int, named: int, hit: int) -> None:
        self.named[li] = self.named[li] * self.DECAY + named
        self.used[li] = self.used[li] * self.DECAY + hit
        self.scored[li] += 1

    def score(self, layer_key: str, demanded) -> None:
        """A pass layer's real demand, against what the plan named for it."""
        if not self.armed:
            return
        li = self.li.get(layer_key)
        got = self.armed.pop(li, None) if li is not None else None
        if not got:
            return
        hit = len(got.intersection(int(e) for e in demanded))
        self._book(li, len(got), hit)
        self.stats["used"] += hit

    def end_pass(self) -> None:
        """Close a pass's books; a layer the pass never reached named nothing right."""
        if not self.armed and not self.stats["named"]:
            return
        for li, got in self.armed.items():
            self._book(li, len(got), 0)
        self.armed = {}
        self.stats["passes"] += 1

    def precision(self) -> float | None:
        total = float(self.named.sum())
        return float(self.used.sum()) / total if total > 0 else None

    def describe(self) -> str:
        p = self.precision()
        st = self.stats
        return (
            f" replay={st['used']}/{st['named']}"
            + (f"@{p:.2f}" if p is not None else "")
            + f" trusted={int(self.trusted().sum())}/{len(self.layer_keys)}"
            + (f" read={st['issued']}" if st["issued"] else "")
        )


def record_forward(log: RouteLog, layer_key: str, inds: np.ndarray, weights=None) -> None:
    """Called by a streamed layer once its router ids are on the host."""
    global _FWD_TOKENS_DONE
    fwd = _FWD
    if fwd is None:
        return
    start, tokens = fwd
    if inds.ndim != 2 or tokens is None or int(tokens.shape[-1]) != int(inds.shape[0]):
        return
    if not _FWD_TOKENS_DONE:
        _FWD_TOKENS_DONE = True
        log.record_tokens(start, np.asarray(tokens))
    log.record(layer_key, start, inds, weights)
