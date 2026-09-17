# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Layer-fused prefill: chunk *attention*, stream the expert mass once.

Prefill on this engine is bound by expert **bytes read**, not FLOPs. A prompt
chunk of more than a few hundred tokens routes to essentially every expert of
every layer, so one chunk costs one full pass over the checkpoint's expert mass
(~368 GB on DeepSeek-V3.2). mlx-lm chunks the *whole model*, so a 14k prompt at
4096-token steps pays that pass four times - 1.4 TB off the SSD for a prompt
whose weights are 368 GB.

Only attention wants the small chunk. DSA-family attention materializes a dense
``pe_scores [heads, chunk, keys]``, and that product against Metal's max single
buffer is what caps the chunk; fused-SDPA models cap on their own score
intermediates. The MoE half has no such bound - it is per-token work with a
fixed weight mass.

So stop running them at the same granularity. Inside one decoder layer:

    for each attention sub-chunk:  r_i = attn(ln1(x_i), mask_i, cache)
    r = concat(r_i);  h = x + r
    return h + mlp(ln2(h))          <- once, over *all* tokens

Consecutive sub-chunks see a KV cache that grows exactly as consecutive model
calls would, and RoPE positions come from the same ``cache.offset``, so each
token's attention is the same arithmetic on the same inputs and the cache ends
in the same state. What changes is only *how many times the experts are read*:
once per prompt instead of once per attention chunk.

Decoupling the two granularities also inverts the sizing. Under mlx-lm's scheme
a small chunk is ruinous (each one re-reads the expert mass), so the chunk was
pushed as large as Metal would allow. Here the expert cost is fixed, and a
smaller attention sub-chunk is *cheaper*: a sub-chunk of C rows against a KV of
`offset` computes C·(offset+C) score pairs, of which its own C²/2 upper triangle
is causally dead weight a materialized mask cannot skip. Overhead therefore
scales with C/T, and the sub-chunk is sized to `config.ATTN_SUB_CHUNK` with the
Metal bound as a ceiling rather than a target.

Quality: same operations per token, same order within a token. Sub-chunk
boundaries shift SDPA's reduction *blocking* the same way any other
prefill_step_size does - logit-near-identical, never expert-substituting.
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

    The outer prefill loop reads this: when attention is chunked inside the
    layer, the model-level chunk no longer has to respect the score-matrix
    bound and should be as large as activation memory allows.
    """
    return _active


# ----------------------------------------------------------------- helpers

_LAYER_ATTRS = ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")

# Below this, take the stock path unconditionally. Covers decode (1 token) and
# speculative verify batches, which must not pay for plan arithmetic on every
# layer of every token.
_MIN_SPLIT_TOKENS = 8


def layer_is_fusable(layer: Any) -> bool:
    """Standard pre-norm block: attn -> residual -> mlp -> residual.

    Every MoE architecture in the catalog (DeepSeek-V3.2, GLM-4.x, Qwen3-MoE,
    Qwen3-Next) uses exactly this shape. Anything else keeps stock behaviour.
    """
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
    """The full chunk's mask, restricted to sub-chunk rows and live keys.

    Row ``i`` of the full mask already encodes "query at position offset+i vs
    key j", so the sub-chunk's mask is a plain slice - no rebuild, and no risk
    of disagreeing with the mask the stock path would have produced. Keys are
    truncated to what the cache holds once this sub-chunk is appended.

    A string mask ("causal") is position-independent: mlx aligns queries to the
    end of the keys, which is what a sub-chunk at a non-zero offset wants.
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
    """Sub-chunk boundaries for `total` new tokens starting at KV `offset`.

    Each sub-chunk is sized against the score matrix it will materialize
    *given the keys present when it runs*, so later sub-chunks (longer KV)
    shrink. Returns [(start, end), ...] covering [0, total).
    """
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
        # Decode (1 token) and speculative verify batches take the stock path
        # without so much as a size computation: this runs once per layer per
        # token, and a batch this small cannot need splitting. `kw` may carry
        # per-architecture extras the fused path does not model.
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
            # Force this sub-chunk now: its score matrix is the largest buffer
            # in the pass, and holding the graph lazily across sub-chunks would
            # keep every one of them alive at once - the exact allocation this
            # split exists to avoid. The cache state is read fresh each time
            # because growing it rebinds `keys`/`values` to new arrays.
            state = _cache_state(cache)
            mx.eval(r) if state is None else mx.eval(r, state)
            parts.append(r)
        r = mx.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        # Drop the per-sub-chunk references; Metal's own pool keeps the freed
        # score buffers, which is what the MoE half then allocates out of. (An
        # explicit clear_cache here would throw that reuse away and pay for a
        # fresh, zero-filled allocation on every layer.)
        del parts

        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))

    __call__.__wrapped__ = orig
    return __call__


# ----------------------------------------------------------------- install


def install(model: Any) -> bool:
    """Patch this model's decoder layers to sub-chunk their own attention.

    Returns True when at least one layer class was fused. Idempotent, and a
    no-op for architectures that do not match the standard block.
    """
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
