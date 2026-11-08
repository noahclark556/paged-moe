# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recurrent state at a position inside one prefill pass.

A hybrid turn stores its prompt-cache entry at the end of the last message,
a few tokens short of the end of the prompt (the generation prompt follows).
Stopping prefill there and running those tokens as a second chunk costs a
second pass over the experts they route to. On Flash that was ~0.5 s and
~2 GB of reads per turn: the first pass streams the whole expert mass through
the slab and keeps little of it.

This takes the same snapshot inside the single pass. Attention KV trims back
to the anchor afterwards (`hybrid_ckpt.restore`). What cannot be recovered
afterwards is each recurrent layer's state at the anchor, so for the one
chunk that contains it:

- `gated_delta_update` runs as two calls split at the anchor. The kernel
  walks tokens in order with the state in float32 and reads and writes it as
  float32, so the outputs and the final state do not change, and the state
  between the two calls is the state after the anchor token.
- every Conv1d in a recurrent layer has its input tapped. The conv state
  after the anchor is the `n` input rows starting at the anchor, which are
  the rows a stop at the anchor would have kept.
- qwen4_exp's n-gram context is the ids just before the anchor.

Each captured slot is checked against what the model wrote at the end of the
chunk. A slot that does not match a known rule fails the capture, and the
caller stops at the anchor as before.
"""

from __future__ import annotations

import threading

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache

_tl = threading.local()
_off: str | None = None


def disable(reason: str) -> None:
    """Stop taking in-pass snapshots for the rest of this process.

    The turn that fails has already passed its anchor and goes without a
    mid-snap; every later turn stops at the anchor instead.
    """
    global _off
    if _off is None:
        _off = reason
        print(f"[prefix]   in-pass mid-snap off: {reason}", flush=True)


def _layers(model):
    from .prefill_fused import _decoder_layers

    return _decoder_layers(model) or []


def _gdu_scope(model):
    """The globals the recurrent layer resolves `gated_delta_update` from.

    Taken from the layer's own `__call__`: a model file loaded from the
    checkpoint directory is not always registered in sys.modules.
    """
    scopes = []
    for layer in _layers(model):
        lin = getattr(layer, "linear_attn", None)
        if lin is None:
            continue
        g = getattr(type(lin).__call__, "__globals__", None)
        if not isinstance(g, dict) or not callable(g.get("gated_delta_update")):
            return None
        if all(g is not s for s in scopes):
            scopes.append(g)
    return scopes[0] if len(scopes) == 1 else None


def supports(model, cache) -> bool:
    """True when this model's recurrent layers are ones `run` can read."""
    if _off is not None or model is None:
        return False
    try:
        if _gdu_scope(model) is None:
            return False
        layers = list(_layers(model))
    except Exception:
        return False
    if not layers or len(layers) > len(cache):
        return False
    for layer, c in zip(layers, cache):
        if isinstance(c, ArraysCache) and getattr(layer, "linear_attn", None) is None:
            return False
    return any(isinstance(c, ArraysCache) for c in cache)


class _Capture:
    def __init__(self, model, cache, ids, rels: list[int]):
        self.model = model
        self.cache = cache
        self.ids = ids
        self.rels = rels
        self.n = int(ids.shape[1])
        self.gdu_calls: list = []
        self.taps: dict[int, list] = {}
        self.tap_layer: dict[int, int] = {}
        self.before = {}
        self.error = None


def _split_gdu(orig):
    def gdu(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
        cap = getattr(_tl, "cap", None)
        T = int(q.shape[1])
        if cap is None or cap.error or T != cap.n:
            if cap is not None and not cap.error:
                cap.error = f"recurrent call over {T} tokens, chunk is {cap.n}"
            return orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=use_kernel)

        def part(x, lo, hi):
            return None if x is None else x[:, lo:hi]

        ys, mids = [], []
        lo = 0
        for hi in [*cap.rels, T]:
            y, state = orig(
                q[:, lo:hi], k[:, lo:hi], v[:, lo:hi], a[:, lo:hi], b[:, lo:hi],
                A_log, dt_bias, state, part(mask, lo, hi), use_kernel=use_kernel,
            )
            ys.append(y)
            mids.append(state)
            lo = hi
        end = mids.pop()
        cap.gdu_calls.append((mids, end))
        return mx.concatenate(ys, axis=1), end

    return gdu


def _tap_conv(orig):
    def call(self, x):
        cap = getattr(_tl, "cap", None)
        rec = None if cap is None else cap.taps.get(id(self))
        if rec is not None:
            n_state = int(x.shape[1]) - cap.n
            if n_state <= 0:
                cap.error = "conv input carries no state rows"
            else:
                # Copied out and evaluated here: a lazy slice would keep the
                # whole chunk's conv input alive until the end of the pass.
                ats = [mx.contiguous(x[:, r : r + n_state]) for r in cap.rels]
                tail = mx.contiguous(x[:, -n_state:])
                mx.eval(*ats, tail)
                rec.append((ats, tail))
        return orig(self, x)

    return call


def run(model, cache, ids, rel: int, forward):
    """Call `forward()` (one model call over `ids`) and return the recurrent
    state each ArraysCache held after `rel` of those tokens, as
    {cache index: [slot values]}, or None if any slot could not be read.
    """
    got = run_multi(model, cache, ids, [rel], forward)
    return None if got is None else got[0]


def run_multi(model, cache, ids, rels: list[int], forward, quiet: bool = False):
    """`run` for several positions in the same pass: one state dict per entry
    of `rels`, in order, or None. Each split adds one more recurrent call over
    the same tokens, not another pass over the experts.

    `quiet` reports a failed capture without turning the mid-snap off; the
    caller then goes without the extra states.
    """
    scope = _gdu_scope(model)
    rels = sorted(set(int(r) for r in rels))
    if scope is None or not rels or not (0 < rels[0] and rels[-1] < int(ids.shape[1])):
        forward()
        return None
    cap = _Capture(model, cache, ids, rels)
    layers = _layers(model)
    for i, (layer, c) in enumerate(zip(layers, cache)):
        if not isinstance(c, ArraysCache):
            continue
        cap.before[i] = list(c.cache)
        for _name, m in layer.named_modules():
            if isinstance(m, nn.Conv1d):
                cap.taps[id(m)] = []
                cap.tap_layer[id(m)] = i
    orig_gdu = scope["gated_delta_update"]
    orig_conv = nn.Conv1d.__call__
    scope["gated_delta_update"] = _split_gdu(orig_gdu)
    nn.Conv1d.__call__ = _tap_conv(orig_conv)
    _tl.cap = cap
    try:
        forward()
    finally:
        _tl.cap = None
        nn.Conv1d.__call__ = orig_conv
        scope["gated_delta_update"] = orig_gdu
    if cap.error:
        if quiet:
            print(f"[prefix]   in-pass checkpoints not taken: {cap.error}", flush=True)
        else:
            disable(cap.error)
        return None
    try:
        return _resolve(cap, layers)
    except Exception as e:
        if quiet:
            print(f"[prefix]   in-pass checkpoints not taken: {e!r}", flush=True)
        else:
            disable(repr(e))
        return None


def _resolve(cap: _Capture, layers):
    outs: list[dict] = [{} for _ in cap.rels]
    gdu = iter(cap.gdu_calls)
    ids = cap.ids
    for i, c in enumerate(cap.cache[: len(layers)]):
        if not isinstance(c, ArraysCache):
            continue
        mids_end = next(gdu, None)
        if mids_end is None:
            raise ValueError(f"layer {i}: no recurrent update recorded")
        mids, end = mids_end
        convs = [r for m_id, recs in cap.taps.items() if cap.tap_layer[m_id] == i for r in recs]
        before = cap.before[i]
        per_rel: list[list] = [[] for _ in cap.rels]
        used_gdu = False
        for s, now in enumerate(c.cache):
            prev = before[s] if s < len(before) else None
            if now is None or now is prev:
                # Untouched by this chunk, so it already holds the anchor's value.
                for slots in per_rel:
                    slots.append(now)
                continue
            if now is end:
                for slots, mid in zip(per_rel, mids):
                    slots.append(mid)
                used_gdu = True
                continue
            if mx.issubdtype(now.dtype, mx.integer) and now.ndim == 2:
                ctx = int(now.shape[1])
                if not (ctx <= cap.rels[0] and ctx <= cap.n):
                    raise ValueError(f"layer {i} slot {s}: anchor inside the id context")
                if not bool(mx.array_equal(ids[:, -ctx:].astype(now.dtype), now)):
                    raise ValueError(f"layer {i} slot {s}: id context is not the last ids")
                for slots, r in zip(per_rel, cap.rels):
                    slots.append(mx.contiguous(ids[:, r - ctx : r].astype(now.dtype)))
                continue
            hit = None
            for ats, tail in convs:
                if tail.shape == now.shape and bool(mx.array_equal(tail, now)):
                    hit = ats
                    break
            if hit is None:
                raise ValueError(f"layer {i} slot {s}: no rule for shape {tuple(now.shape)}")
            for slots, at in zip(per_rel, hit):
                slots.append(at)
        if not used_gdu:
            raise ValueError(f"layer {i}: recurrent state slot not found")
        for out, slots in zip(outs, per_rel):
            out[i] = slots
    if next(gdu, None) is not None:
        raise ValueError("more recurrent updates than recurrent layers")
    mx.eval([v for out in outs for slots in out.values() for v in slots if v is not None])
    return outs
