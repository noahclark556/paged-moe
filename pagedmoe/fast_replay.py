# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Partial-accept replay on hybrid caches without re-reading the model.

A recurrent cache cannot be trimmed, so when a verify pass keeps only some of
its rows the old path restores the pre-pass state and runs the kept tokens
through the whole model again. On a banked model that second forward reads the
experts a second time and costs most of a pass.

Here the pass records what every attention and MoE block returned. The replay
still runs the model over the kept tokens, so embeddings, norms, the recurrent
mixers and any state the model writes at the top level are rebuilt exactly as
before, but each of those blocks hands back its recorded rows instead of
computing them. Those rows are what the full replay would compute: position i
of a causal pass only sees positions up to i. Attention caches keep the rows of
the pass they already hold and drop the rest.

The blocks are found at run time, not by name: each direct child of a decoder
layer that is handed a trimmable cache is attention, one with experts under it
is MoE, and one with linear weights and no cache is a dense MLP. A dense MLP is
most of a dense model's weights, so replaying it from its rows is what makes a
partial accept cheap there. Nothing is assumed about a model this has not seen. The first few
replays of every load are also run the old way and compared, and any mismatch,
or any call pattern the recording cannot account for, puts the model back on
the full replay for good.
"""

from __future__ import annotations

import contextlib
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache

from . import config

try:
    from mlx_lm.models.cache import _BaseCache
except ImportError:  # pragma: no cover
    _BaseCache = object

_ATTN, _MOE, _DENSE, _REC, _PLAIN, _BAD = "attn", "moe", "dense", "rec", "plain", "bad"
_EXPERT_HINTS = ("switch", "expert", "moe")
_PATCHED: set[type] = set()
_ACTIVE: "_Replayer | None" = None


class _Abort(Exception):
    pass


def _decoder_layers(model) -> list | None:
    for path in (("layers",), ("model", "layers"), ("language_model", "model", "layers"),
                 ("language_model", "layers")):
        obj = model
        for a in path:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if isinstance(obj, (list, tuple)) and obj:
            return list(obj)
    return None


def _has_experts(mod) -> bool:
    for _, m in mod.named_modules():
        name = type(m).__name__.lower()
        if any(h in name for h in _EXPERT_HINTS):
            return True
    return False


def _has_linear(mod) -> bool:
    return any(isinstance(m, (nn.Linear, nn.QuantizedLinear)) for _, m in mod.named_modules())


def _hook(cls: type) -> None:
    if cls in _PATCHED:
        return
    orig = cls.__call__

    def __call__(self, *args, **kwargs):
        rep = _ACTIVE
        if rep is None:
            return orig(self, *args, **kwargs)
        return rep._call(self, orig, args, kwargs)

    __call__.__wrapped__ = orig
    cls.__call__ = __call__
    _PATCHED.add(cls)


def _caches_in(args, kwargs):
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, _BaseCache):
            yield a


def _rows(x, T: int, keep: int):
    if isinstance(x, mx.array):
        if x.ndim < 2 or int(x.shape[1]) != T:
            raise _Abort(f"recorded output shaped {tuple(x.shape)}, pass was {T} rows")
        return x[:, :keep]
    if isinstance(x, tuple):
        return tuple(_rows(v, T, keep) for v in x)
    if isinstance(x, list):
        return [_rows(v, T, keep) for v in x]
    if x is None:
        return None
    raise _Abort(f"recorded output of type {type(x).__name__}")


def _peak(x) -> float:
    return float(mx.max(mx.abs(x)).item())


def _close(a, b, before) -> bool:
    """Whether fast state `a` matches full-replay state `b`.

    Measured against how far the kept tokens moved the state from `before`.
    A single kept token runs mlx's one-token kernels in the full replay, and
    those round differently from row 0 of a batched pass; on a 6-bit 35B model
    that reaches a quarter of the move, while a replay missing a token is off
    by more than the whole move.
    """
    if a is None or b is None:
        return a is b
    if not isinstance(a, mx.array) or not isinstance(b, mx.array):
        return False
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if not mx.issubdtype(a.dtype, mx.floating):
        return bool(mx.array_equal(a, b).item())
    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    if isinstance(before, mx.array) and before.shape == b.shape:
        moved = _peak(b32 - before.astype(mx.float32))
    else:
        moved = _peak(b32)
    scale = max(moved, 0.05 * _peak(b32)) + 1e-6
    return _peak(a32 - b32) / scale <= float(config.FAST_REPLAY_TOL)


class _Replayer:
    def __init__(self, model):
        self.model = model
        self.off = False
        self.why = ""
        self.checks = 0
        self.check_s = 0.0
        self.fast = 0
        self.role: dict[int, str] = {}
        self.moe: set[int] = set()
        self.dense: set[int] = set()
        self.loose: set[int] = set()
        self.mode = None
        self.outs: dict[int, list] = {}
        self.pos: dict[int, int] = {}
        self.T = 0
        self.keep = 0
        layers = _decoder_layers(model)
        if not layers:
            self._disable("no decoder layer list")
            return
        for layer in layers:
            if not isinstance(layer, nn.Module):
                self._disable("decoder layers are not modules")
                return
            for _, child in layer.children().items():
                for mod in child if isinstance(child, (list, tuple)) else [child]:
                    if not isinstance(mod, nn.Module):
                        continue
                    if _has_experts(mod):
                        self.moe.add(id(mod))
                    elif _has_linear(mod):
                        self.dense.add(id(mod))
                    _hook(type(mod))

    def _disable(self, why: str) -> None:
        if not self.off:
            print(f"[verify] fast replay off for this model: {why}", flush=True)
        self.off = True
        self.why = why

    @staticmethod
    def _loose(args, kwargs) -> list:
        return [c for c in _caches_in(args, kwargs)
                if not isinstance(c, ArraysCache) and not c.is_trimmable()]

    def _classify(self, mod, args, kwargs) -> str:
        kinds = set()
        loose = self._loose(args, kwargs)
        for c in _caches_in(args, kwargs):
            if isinstance(c, ArraysCache):
                kinds.add(_REC)
            elif c.is_trimmable():
                kinds.add(_ATTN)
        if loose and kinds != {_ATTN}:
            return _BAD
        if len(kinds) > 1:
            return _BAD
        if _REC in kinds:
            return _BAD if id(mod) in self.moe else _REC
        if _ATTN in kinds:
            return _ATTN
        if id(mod) in self.moe:
            return _MOE
        return _DENSE if id(mod) in self.dense else _PLAIN

    def _call(self, mod, orig, args, kwargs):
        if self.mode == "record":
            key = id(mod)
            role = self.role.get(key)
            if role is None:
                role = self.role[key] = self._classify(mod, args, kwargs)
                if role == _BAD:
                    self._disable(f"{type(mod).__name__} takes a cache this cannot replay")
            if role == _ATTN:
                self.loose.update(id(c) for c in self._loose(args, kwargs))
            out = orig(mod, *args, **kwargs)
            if role in (_ATTN, _MOE, _DENSE):
                self.outs.setdefault(key, []).append(out)
            return out
        if self.mode in ("replay", "check"):
            key = id(mod)
            role = self.role.get(key)
            if role is None:
                raise _Abort(f"{type(mod).__name__} was not called during the pass")
            # The check reference reuses MoE rows only; dense blocks run for real.
            if role == _MOE or (role in (_ATTN, _DENSE) and self.mode == "replay"):
                got = self.outs.get(key) or []
                i = self.pos.get(key, 0)
                if i >= len(got):
                    raise _Abort(f"{type(mod).__name__} called more often than in the pass")
                self.pos[key] = i + 1
                return _rows(got[i], self.T, self.keep)
        return orig(mod, *args, **kwargs)

    @contextlib.contextmanager
    def recording(self, T: int):
        """Wrap the pass forward. A no-op once the model is off fast replay."""
        global _ACTIVE
        self.outs = {}
        self.loose = set()
        if self.off or not config.FAST_REPLAY or _ACTIVE is not None:
            yield
            return
        self.T = int(T)
        self.mode = "record"
        _ACTIVE = self
        try:
            yield
        finally:
            _ACTIVE = None
            self.mode = None

    def replay(self, cache, snap, fed: int, keep: list[int], full) -> bool:
        """Leave `cache` as if only `keep` had been fed. True when done fast.

        `full(tokens)` runs the model over `tokens` the old way. On any doubt
        this restores the snapshot and calls it, so the cache always ends right.
        """
        n = len(keep)
        trimmed = [0] * len(cache)
        if self.off or not config.FAST_REPLAY or not self.outs or self.T != fed or n <= 0 or n >= fed:
            self._full(cache, snap, fed, trimmed, keep, full)
            return False
        sides = []
        for c, (kind, s) in zip(cache, snap):
            if kind != "arrays":
                for side, before in s:
                    sides.append((side, before, side.state))
            elif c.lengths is not None or c.left_padding is not None:
                self._disable("batched recurrent cache")
                self._full(cache, snap, fed, trimmed, keep, full)
                return False
        if not self.loose <= {id(side) for side, _b, _a in sides}:
            self._disable("attention takes a side cache the snapshot does not hold")
            self._full(cache, snap, fed, trimmed, keep, full)
            return False
        try:
            drop = fed - n
            for i, (c, (kind, s)) in enumerate(zip(cache, snap)):
                if kind == "arrays":
                    c.cache = list(s)
                    continue
                trimmed[i] = c.trim(drop)
                if trimmed[i] != drop:
                    raise _Abort("an attention cache would not trim")
            for side, before, after in sides:
                side.state = self._side_rows(before, after, fed, n)
            self._skeleton(cache, keep)
        except Exception as e:
            self._disable(f"{e}" if isinstance(e, _Abort) else f"{e!r}")
            self._full(cache, snap, fed, trimmed, keep, full)
            return False
        if self.checks < int(config.FAST_REPLAY_CHECKS):
            return self._check(cache, snap, sides, fed, keep, full)
        self.fast += 1
        return True

    @staticmethod
    def _side_rows(before, after, fed: int, n: int):
        b = 0 if before is None else int(before.shape[1])
        if not isinstance(after, mx.array) or after.ndim < 2 or int(after.shape[1]) != b + fed:
            raise _Abort("attention side cache is not one row per token")
        return after[:, : b + n]

    def _skeleton(self, cache, keep: list[int], mode: str = "replay") -> None:
        global _ACTIVE
        self.pos = {}
        self.keep = len(keep)
        self.mode = mode
        _ACTIVE = self
        try:
            self.model(mx.array([keep], dtype=mx.uint32), cache=cache)
        finally:
            _ACTIVE = None
            self.mode = None
        for key, got in self.outs.items():
            if self.role.get(key) in (_ATTN, _DENSE) and mode == "check":
                continue
            if self.pos.get(key, 0) != len(got):
                raise _Abort("a recorded block was not called back")
        mx.eval([c.state for c in cache])

    def _reference(self, cache, keep: list[int], full) -> None:
        """The replay to check against.

        A resident model runs it for real. A streamed one reuses the pass's MoE
        rows: under WAIT_ABOVE a second forward sees experts the pass just
        loaded and computes other outputs, so it would not be a fixed reference.
        Attention, side caches, mixers and top-level writes still run for real.
        """
        info = getattr(self.model, "_pagedmoe_info", None) or {}
        if info.get("mode") != "streamed":
            full(keep)
            return
        self._skeleton(cache, keep, mode="check")

    @staticmethod
    def _full(cache, snap, fed: int, trimmed: list[int], keep: list[int], full) -> None:
        for c, (kind, s), done in zip(cache, snap, trimmed):
            if kind == "arrays":
                c.cache = list(s)
                continue
            left = fed - done
            got = c.trim(left) if left else 0
            if got != left:
                raise RuntimeError(f"verify decode could not rewind: asked {left}, trimmed {got}")
            for side, state in s:
                side.state = state
        full(keep)

    def take_check_s(self) -> float:
        """Seconds the last replay spent checking itself, which drafting should not pay for."""
        s, self.check_s = self.check_s, 0.0
        return s

    def _check(self, cache, snap, sides, fed: int, keep: list[int], full) -> bool:
        t0 = time.perf_counter()
        try:
            return self._check_inner(cache, snap, sides, fed, keep, full)
        finally:
            self.check_s += time.perf_counter() - t0

    def _check_inner(self, cache, snap, sides, fed: int, keep: list[int], full) -> bool:
        n = len(keep)
        fast = [list(c.cache) if kind == "arrays" else None for c, (kind, _s) in zip(cache, snap)]
        fast_side = [side.state for side, _b, _a in sides]
        offsets = [getattr(c, "offset", None) for c in cache]
        for c, (kind, s) in zip(cache, snap):
            if kind == "arrays":
                c.cache = list(s)
            elif c.trim(n) != n:
                self._disable("an attention cache would not trim for the check")
                return False
        for side, before, _after in sides:
            side.state = before
        try:
            self._reference(cache, keep, full)
        except Exception as e:
            self._disable(f"replay check could not run: {e}")
            base = [o - n if o is not None else None for o in offsets]
            for c, (kind, s), b in zip(cache, snap, base):
                if kind == "arrays":
                    c.cache = list(s)
                    continue
                over = int(c.offset) - b
                if over and c.trim(over) != over:
                    raise RuntimeError("verify decode could not rewind after a failed replay check")
                for side, state in s:
                    side.state = state
            full(keep)
            return False
        bad = None
        if not all(getattr(c, "offset", None) == o for c, o in zip(cache, offsets)):
            bad = "attention offsets"
        for i, (c, f, (_kind, s)) in enumerate(zip(cache, fast, snap)):
            if bad or f is None:
                continue
            if len(f) != len(c.cache):
                bad = f"layer {i} slot count"
            for j, (a, b, s0) in enumerate(zip(f, c.cache, s)):
                if not bad and not _close(a, b, s0):
                    bad = f"layer {i} slot {j}"
        for f, (side, _b, _a) in zip(fast_side, sides):
            if not bad and not _close(f, side.state, None):
                bad = "an attention side cache"
        if bad:
            self._disable(f"replay check did not match the full replay ({bad}, {n} of {fed} kept)")
            return False
        self.checks += 1
        if self.checks == int(config.FAST_REPLAY_CHECKS):
            print(f"[verify] fast replay checked {self.checks} times, on for this model", flush=True)
        return False


def for_model(model) -> _Replayer:
    rep = getattr(model, "_pagedmoe_fast_replay", None)
    if rep is None:
        rep = _Replayer(model)
        try:
            model._pagedmoe_fast_replay = rep
        except Exception:
            pass
    return rep
