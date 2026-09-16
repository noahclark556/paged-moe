# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Layer-fused prefill: chunk attention, stream the expert mass once.

Prefill here is bound by expert bytes read, not FLOPs. A wide prompt chunk
hits essentially every expert of every layer, so one chunk = one pass over
the expert mass. mlx-lm chunks the whole model, so small attention-sized
steps multiply that pass.

Only attention wants the small chunk (DSA builds a dense pe_scores matrix
against Metal's max buffer). The MoE half has no such bound. So inside one
decoder layer:

    for each attention sub-chunk:  r_i = attn(ln1(x_i), mask_i, cache)
    r = concat(r_i);  h = x + r
    return h + mlp(ln2(h))          # once, over all tokens

Same arithmetic per token as stock; experts are read once per prompt chunk
instead of once per attention chunk. Sub-chunk size is ATTN_SUB_CHUNK under
the Metal bound (smaller C = less score work on a materialized mask).
"""

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx

from . import adaptive_prefill, config

_installed: set[type] = set()
_active = False


def enabled() -> bool:
    return bool(config.FUSED_PREFILL)


def active() -> bool:
    """True once a model's decoder layers sub-chunk their own attention.

    When attention is chunked inside the layer, the model-level chunk no
    longer has to respect the score-matrix bound.
    """
    return _active


# ----------------------------------------------------------------- helpers

_LAYER_ATTRS = ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")

# Decode (1 token) and speculative verify batches stay on the stock path.
_MIN_SPLIT_TOKENS = 8


def layer_is_fusable(layer: Any) -> bool:
    """Standard pre-norm block: attn -> residual -> mlp -> residual."""
    return all(hasattr(layer, a) for a in _LAYER_ATTRS)


def _cache_offset(cache: Any) -> int:
    """Current KV length of one layer's cache (0 when there is none yet)."""
    if cache is None:
        return 0
    off = getattr(cache, "offset", None)
    if off is None:
        # CacheList (DeepSeek: [kv, indexer]) proxies by index, not attribute.
        try:
            off = getattr(cache[0], "offset", None)
        except (TypeError, IndexError, AttributeError):
            off = None
    try:
        return max(0, int(off))
    except (TypeError, ValueError):
        return 0


def _slice_mask(mask: Any, start: int, end: int, keys: int):
    """Restrict the full chunk's mask to this sub-chunk's rows and live keys.

    A string mask ("causal") is position-independent: mlx aligns queries to
    the end of the keys, which is what a sub-chunk at a non-zero offset wants.
    """
    if mask is None or isinstance(mask, str):
        return mask
    if mask.ndim <= 2:
        return mask[start:end, :keys]
    return mask[..., start:end, :keys]


def _cache_state(cache: Any):
    try:
        return getattr(cache, "state", None)
    except Exception:
        return None


def attention_plan(owner: Any, total: int, offset: int) -> list[tuple[int, int]]:
    """Sub-chunk boundaries for `total` new tokens starting at KV `offset`."""
    plan: list[tuple[int, int]] = []
    start = 0
    while start < total:
        n = adaptive_prefill.attention_sub_chunk(owner, offset + start)
        n = max(1, min(n, total - start))
        plan.append((start, start + n))
        start += n
    return plan


# ----------------------------------------------------------------- forward


def _fused_layer_call(orig):
    def __call__(self, x, mask=None, cache=None, **kw):
        if kw or x.ndim != 3 or x.shape[1] <= _MIN_SPLIT_TOKENS:
            return orig(self, x, mask, cache, **kw)

        total = int(x.shape[1])
        offset = _cache_offset(cache)
        plan = attention_plan(self, total, offset)
        if len(plan) <= 1:
            return orig(self, x, mask, cache)

        parts = []
        for start, end in plan:
            m = _slice_mask(mask, start, end, offset + end)
            r = self.self_attn(self.input_layernorm(x[:, start:end]), m, cache)
            # Eval now so score matrices from earlier sub-chunks can free.
            state = _cache_state(cache)
            mx.eval(r) if state is None else mx.eval(r, state)
            parts.append(r)
        r = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        del parts

        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))

    __call__.__wrapped__ = orig
    return __call__


# ----------------------------------------------------------------- install


def install(model: Any) -> bool:
    """Patch this model's decoder layers to sub-chunk their own attention."""
    global _active
    if not enabled():
        return False
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if not layers:
        return False

    patched = 0
    total = 0
    first = None
    for layer in layers:
        if layer is None:
            continue
        total += 1
        cls = type(layer)
        if cls in _installed:
            patched += 1
            first = first or layer
            continue
        if not layer_is_fusable(layer):
            continue
        cls.__call__ = _fused_layer_call(cls.__call__)
        _installed.add(cls)
        patched += 1
        first = first or layer

    if not patched:
        print(
            "[paged-moe] fused prefill skipped (non-standard decoder layer)",
            flush=True,
        )
        return False
    _active = True
    sub = adaptive_prefill.attention_sub_chunk(first, 0)
    print(
        f"[paged-moe] fused prefill ON ({patched}/{total} layers): attention "
        f"sub-chunks (<= {sub} tokens at cold KV), experts streamed once per "
        "prompt chunk",
        flush=True,
    )
    return True


def uninstall() -> None:
    """Restore stock layer __call__ (tests; never needed in production)."""
    global _active
    for cls in list(_installed):
        orig = getattr(cls.__call__, "__wrapped__", None)
        if orig is not None:
            cls.__call__ = orig
        _installed.discard(cls)
    _active = False
