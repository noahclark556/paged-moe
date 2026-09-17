# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Prefill chunk sizing.

Two different quantities want two different chunk sizes, and mlx-lm's single
`prefill_step_size` has to be both:

* **Attention** on a DSA-family model materializes a dense
  `pe_scores [heads, chunk, keys]`, so `chunk · (kv + chunk)` is bounded by
  Metal's max single buffer. Against a long KV that forces a tiny chunk.
* **The MoE half** cares only about how many times the prompt is walked: each
  walk streams the entire expert mass off the SSD (~368 GB on DeepSeek-V3.2),
  so it wants the *largest* chunk that fits in activation memory.

`prefill_fused` resolves the conflict by sub-chunking attention inside each
decoder layer, which leaves this module with two clean jobs:

  `attention_sub_chunk(model, kv_len)` - the score-matrix bound (per layer).
  `next_chunk(...)`                    - the model-level step, an activation
                                         bound, large and KV-independent once
                                         fused prefill is active.

Modes (EXPERT_STREAM_ADAPTIVE_PREFILL_MODE):
  auto  - score-matrix bound when the model looks like deepseek_v32 / has an
          indexer; fused-SDPA MoEs need no attention split at all
  dsa   - always apply the score-matrix bound
  fused - never split attention
  off   - disabled (same as ADAPTIVE_PREFILL=0)

Quality: sizing changes only how SDPA's reduction is blocked - the same class
of variation as any other prefill_step_size, logit-near-identical, and
bit-identical for a fixed schedule. Never feeds predicted experts into compute.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx

from . import config

# Model types known to build a dense pe_scores / score matrix over (chunk × keys)
# before any sparse mask. Keep this list inclusive - a false positive only makes
# attention sub-chunks smaller, and with fused prefill that no longer costs an
# expert-mass pass. `deepseek_v3` is deliberately absent: plain MLA has no
# indexer and runs on fused SDPA, so it sizes like the rest.
_DSA_MODEL_TYPES = frozenset(
    {
        "deepseek_v32",
        "glm4_moe_lite",  # shares deepseek DSA attention in mlx-lm
        "glm_moe_dsa",
    }
)


def enabled() -> bool:
    return bool(config.ADAPTIVE_PREFILL)


def mode_name() -> str:
    return (config.ADAPTIVE_PREFILL_MODE or "auto").strip().lower() or "auto"


_max_buffer: Optional[int] = None


def metal_max_buffer() -> int:
    # Queried once: this sits on the per-sub-chunk sizing path now.
    global _max_buffer
    if _max_buffer is None:
        n = 0
        try:
            n = int(mx.device_info().get("max_buffer_length") or 0)
        except Exception:
            n = 0
        _max_buffer = n if n > 0 else 30_150_672_384
    return _max_buffer


def cache_offset(prompt_cache: Any) -> int:
    """Best-effort current KV length from an mlx-lm prompt cache list."""
    if not prompt_cache:
        return 0
    best = 0
    for entry in prompt_cache:
        off = getattr(entry, "offset", None)
        if off is None:
            # CacheList (DeepSeek-V3.2: [kv, indexer]) subclasses _BaseCache,
            # not list, and exposes no `offset` - it proxies by index only.
            # Reading the attribute alone silently reported KV length 0 for
            # every DeepSeek prompt, which made every chunk size as if it were
            # the first and put chunk 2 far over Metal's buffer.
            inner = getattr(entry, "caches", None)
            if inner is None:
                try:
                    inner = (entry[0],)
                except (TypeError, IndexError, KeyError, AttributeError):
                    inner = None
            if inner:
                off = getattr(inner[0], "offset", None)
        if off is None:
            continue
        try:
            best = max(best, int(off))
        except (TypeError, ValueError):
            continue
    return best


def is_dsa_model(model: Any) -> bool:
    args = getattr(model, "args", None)
    mt = str(getattr(args, "model_type", "") or "").lower()
    if mt in _DSA_MODEL_TYPES or "dsa" in mt:
        return True
    # Heuristic: first layer attention exposes an indexer (DeepSeek sparse attn).
    layers = getattr(model, "layers", None)
    if layers:
        attn = getattr(layers[0], "self_attn", None) or getattr(
            layers[0], "attention", None
        )
        if attn is not None and hasattr(attn, "indexer"):
            return True
    return False


def resolve_mode(model: Any) -> str:
    m = mode_name()
    if m in ("off", "0", "false", "no"):
        return "off"
    if m == "dsa":
        return "dsa"
    if m == "fused":
        return "fused"
    # auto
    return "dsa" if is_dsa_model(model) else "fused"


def _n_heads(model: Any) -> int:
    args = getattr(model, "args", None)
    for attr in ("num_attention_heads", "n_heads", "num_heads"):
        v = getattr(args, attr, None)
        if v:
            return int(v)
    return 128


def dsa_max_chunk(
    kv_len: int,
    *,
    n_heads: int,
    dtype_bytes: int = 2,
    max_buf: Optional[int] = None,
    safety: Optional[float] = None,
    c_max: Optional[int] = None,
) -> int:
    """Largest ``C`` whose dense ``pe_scores [heads, C, kv+C]`` fits Metal.

    Solves ``heads · C · (kv + C) · dtype_bytes <= safety · max_buffer`` for C.

    There is deliberately no lower clamp: this is a hard allocation bound, and
    a floor that overrides it just relocates the OOM. The old floor existed
    because a small chunk used to mean a whole extra pass over the expert mass.
    Under fused prefill a small C costs *less* attention work, not more (see
    `config.ATTN_SUB_CHUNK`), so there is nothing left to trade against.
    """
    max_buf = int(max_buf if max_buf is not None else metal_max_buffer())
    safety = float(
        safety if safety is not None else config.ADAPTIVE_PREFILL_SAFETY
    )
    c_max = int(c_max if c_max is not None else config.ADAPTIVE_PREFILL_MAX)
    budget = max(1.0, safety * max_buf)

    a = float(max(1, n_heads) * max(1, dtype_bytes))
    kv = max(0, int(kv_len))
    disc = kv * kv + 4.0 * (budget / a)
    c = int((-kv + math.sqrt(disc)) / 2.0)
    return max(1, min(c_max, c))


def _attention_of(obj: Any) -> Any:
    return getattr(obj, "self_attn", None) or getattr(obj, "attention", None)


def _layer_shape(layer: Any) -> tuple[bool, int]:
    """(builds a dense score matrix, head count) read off a decoder layer.

    Preferred over asking the owning model: a layer knows its own head count
    and whether it has an indexer, so two models sharing a layer class (a
    draft model, say) cannot be sized with each other's config.
    """
    attn = _attention_of(layer)
    heads = 0
    for attr in ("num_heads", "n_heads", "num_attention_heads"):
        v = getattr(attn, attr, 0)
        if v:
            heads = int(v)
            break
    return hasattr(attn, "indexer"), heads or 128


def attention_sub_chunk(owner: Any, kv_len: int) -> int:
    """Query rows one attention call may process against `kv_len` keys.

    `owner` is a decoder layer (preferred) or the whole model. DSA-family
    attention materializes a dense score matrix, so this shrinks quadratically
    with KV. Fused-SDPA attention tiles internally and never builds that
    matrix, so it gets the whole chunk - which makes prefill_fused's split a
    no-op on those architectures rather than a cost.
    """
    cap = int(config.ADAPTIVE_PREFILL_MAX or config.PREFILL_CHUNK)
    mode = mode_name()
    if mode in ("off", "0", "false", "no", "fused") or owner is None:
        return cap

    forced = mode == "dsa"
    if _attention_of(owner) is not None:
        dsa, heads = _layer_shape(owner)
    elif getattr(owner, "layers", None) is not None or hasattr(owner, "args"):
        dsa, heads = is_dsa_model(owner), _n_heads(owner)
    else:
        return cap
    if not (dsa or forced):
        return cap
    # Two independent caps. The Metal bound says what *fits*; ATTN_SUB_CHUNK
    # says what is *fast*. Sizing to the Metal bound was leaving score work on
    # the table: a sub-chunk only ever needs its own causal triangle, and a
    # materialized mask cannot skip it, so overhead scales with C/T (see
    # config.ATTN_SUB_CHUNK). Take the smaller of the two.
    bound = dsa_max_chunk(kv_len, n_heads=heads, c_max=cap)
    target = int(config.ATTN_SUB_CHUNK or 0)
    return bound if target <= 0 else max(1, min(bound, target))


def next_chunk(
    kv_len: int,
    remaining: int,
    model: Any,
    *,
    prefill_cap: Optional[int] = None,
) -> int:
    """Tokens to prefill on this step (always ≥ 1 when remaining ≥ 1).

    With fused prefill installed the decoder layers sub-chunk their own
    attention, so this step is bounded by activation memory (the cap) rather
    than by the score matrix - which is the whole point: the expert mass is
    read once per step, so a step should be as large as it can be.
    """
    rem = max(0, int(remaining))
    if rem <= 0:
        return 0
    if not enabled():
        cap = int(prefill_cap or config.PREFILL_CHUNK)
        return max(1, min(cap, rem))

    cap = int(prefill_cap or config.ADAPTIVE_PREFILL_MAX or config.PREFILL_CHUNK)
    mode = resolve_mode(model)
    if mode == "off":
        return max(1, min(cap, rem))

    from . import prefill_fused

    if mode == "dsa" and not prefill_fused.active():
        c = dsa_max_chunk(kv_len, n_heads=_n_heads(model), c_max=cap)
    else:
        c = cap
    return max(1, min(c, rem))


def default_cli_prefill_step(model: Any) -> int:
    """Value for --prefill-step-size when adaptive owns the loop.

    mlx-lm still passes this into generate_step; our wrapper uses it as the
    per-step *cap*. Fused prefill makes the cap an activation bound for every
    architecture. Without it, a DSA model has to fall back to the score-matrix
    bound - and with adaptive off that bound must hold for the whole context
    window, since there is no per-step resizing to shrink it later.
    """
    cap = int(config.ADAPTIVE_PREFILL_MAX or config.PREFILL_CHUNK)
    from . import prefill_fused

    if prefill_fused.active() or not is_dsa_model(model):
        return cap
    kv = 0 if enabled() else int(config.RESERVE_CTX or 65_536)
    return dsa_max_chunk(kv, n_heads=_n_heads(model), c_max=cap)
