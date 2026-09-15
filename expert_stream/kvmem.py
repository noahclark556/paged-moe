# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""KV-cache memory management for streamed models.

On a streamed MoE the expert cache is the throughput dial: every GB it holds is
a GB of expert weights decode does not re-read from the SSD. Every other tenant
of unified memory is therefore taking speed away, and the KV cache is by far
the largest of them - on GLM-4.7 (92 full-attention layers, 8 KV heads, head
dim 128) one token of context costs 368 KB at fp16 and 196 KB at 8 bits, so a
24k-token agent conversation is 9.0 GB / 4.8 GB. That is a third of the whole
engine budget for one conversation.

mlx-lm's own KV handling is written for machines where that does not matter, and
it keeps up to four whole copies of a conversation's cache alive at once:

  1. the prompt-cache store's entry,
  2. `fetch_nearest_cache`'s `deepcopy` of it,
  3. the fp16 dequantize prefill needs (`unquantize`, ~2x the 8-bit copy),
  4. `maybe_quantize_kv_cache`'s output, built while (3) is still referenced.

and the store itself is effectively unbounded for us: `--prompt-cache-bytes` is
only enforced on the batched request path (`LRUPromptCache` is constructed with
`max_size` alone, and `trim_to` is called only from the batch branch), while
streamed models are forced onto the sequential path - they must be, since the
batch engine bypasses `stream_generate` and would prefill several prompts, each
streaming gigabytes of experts, at once. So the only bound in effect was
`--prompt-cache-size`: a *count* of caches, six of which is 29 GB on GLM.

Measured on GLM-4.7 at 13.4k tokens, turn two: 9.58 GB backbone + 19.02 GB
expert slab + 5.56 GB store + 2.73 GB working copy + 5.05 GB fp16 dequantize
~= 42 GB against a 35.05 GB budget, which Metal reports as an asynchronous
command-buffer failure ("Insufficient Memory") that kills the request.

This module removes copies 1-2 and the peaks in 3-4, and enforces the byte
bound the sequential path never got:

  `take_nearest_cache`   hand the stored cache over instead of copying it
  `unquantize`           dequantize layer by layer, freeing as it goes
  `requantize`           the same for the reverse conversion
  `bound_store`          enforce a byte budget on the prompt-cache LRU

Nothing here changes a single logit: it is the same cache in the same numeric
format, just never duplicated. What it buys is memory, which the loader then
hands to the expert cache as throughput (`kv_reserve_bytes`).
"""

from __future__ import annotations

import mlx.core as mx

from . import config

# Bytes the prompt-cache store may hold. Set by the loader once the model's
# memory profile is known (see `set_store_budget`); 0 means "not yet known",
# in which case we leave upstream's behavior alone.
_store_budget: int = 0

# Per-token KV cost of the loaded model, in bytes at the cache's decode-time
# precision. Filled in by the loader, read by the bench/probe scripts.
_kv_bytes_per_token: int = 0


# --------------------------------------------------------------- KV geometry


def kv_bytes_per_token(model_config: dict, *, bits: int | None = 8) -> int:
    """Bytes of KV cache one token of context costs.

    Derived from the checkpoint config rather than measured, because measuring
    means running a forward pass, and one prefill step on a streamed MoE reads
    tens of GB of experts before it can tell us anything.

    `bits=None` means an fp16 cache. Quantized caches also carry a scale and a
    bias per group, which at the 64-group default is 4 bytes per 64 elements -
    small, but it is the difference between fitting and not.

    Over-estimating is safe (the expert cache gets a little less than it could
    have) and under-estimating is not, so unknown architectures fall through to
    the dense full-attention formula, which is the largest of the shapes.
    """
    layers = int(model_config.get("num_hidden_layers", 0) or 0)
    if layers <= 0:
        return 0

    if bits is None:
        per_elem = 2.0
    else:
        per_elem = 1.0 * bits / 8 + 2 * 2 / float(config.KV_GROUP_SIZE)

    lora_rank = model_config.get("kv_lora_rank")
    if lora_rank:
        # MLA (DeepSeek-V3 family): the cache holds one compressed latent plus
        # the RoPE part per token, not K and V per head - an order of magnitude
        # less than the dense formula would claim.
        rope_dim = int(model_config.get("qk_rope_head_dim", 64) or 64)
        elems = layers * (int(lora_rank) + rope_dim)
        return int(elems * per_elem)

    kv_heads = int(
        model_config.get("num_key_value_heads")
        or model_config.get("num_attention_heads")
        or 0
    )
    head_dim = int(model_config.get("head_dim") or 0)
    if not head_dim:
        heads = int(model_config.get("num_attention_heads") or 0)
        hidden = int(model_config.get("hidden_size") or 0)
        head_dim = hidden // heads if heads else 0
    if not (kv_heads and head_dim):
        return 0

    # Hybrid-attention models (qwen3-next) replace most layers with a
    # fixed-size recurrent state, so this over-counts them - deliberately.
    return int(2 * layers * kv_heads * head_dim * per_elem)


def kv_reserve_bytes(model_config: dict, ctx_tokens: int) -> int:
    """Memory to keep out of the expert cache's reach for KV, at `ctx_tokens`.

    What has to coexist with the expert cache is the *decode-time* cache: one
    conversation, in whatever precision decode uses. The fp16 peak a resumed
    turn needs is bigger, but it happens during prefill, and prefill hands the
    whole slab back first (`ExpertCache.enter_prefill`), so it is not what
    bounds the steady state.

    Below `KV_FP16_CTX` the cache stays fp16 for the decode-throughput reason
    documented on that knob, so a short-context model reserves the fp16 cost of
    the threshold instead.
    """
    if ctx_tokens <= 0:
        return 0
    # A serving config can ask for more context than the checkpoint has
    # positions for (the host app ships Qwen3-235B at numCtx 65536 against the model's
    # 40960), and reserving for context the model cannot attend to would take
    # GBs off the expert cache for nothing.
    limit = int(model_config.get("max_position_embeddings") or 0)
    if limit > 0:
        ctx_tokens = min(ctx_tokens, limit)
    quantized = kv_bytes_per_token(model_config, bits=config.KV_BITS) * ctx_tokens
    fp16_span = min(ctx_tokens, config.KV_FP16_CTX)
    fp16 = kv_bytes_per_token(model_config, bits=None) * fp16_span
    # A snapshot (see server._snapshot_user_segment) is a second live cache for
    # part of the turn; STORE_SLACK covers it and the store's own steady state.
    return int(max(quantized, fp16) * config.KV_STORE_SLACK)


def set_store_budget(nbytes: int, kv_per_token: int = 0) -> None:
    """Publish the prompt-cache byte bound (loader -> `bound_store`)."""
    global _store_budget, _kv_bytes_per_token
    _store_budget = max(0, int(nbytes))
    if kv_per_token:
        _kv_bytes_per_token = int(kv_per_token)


def store_budget() -> int:
    return _store_budget


# ------------------------------------------------------- copy-free conversion


def unquantize(cache: list) -> int:
    """Turn reused quantized entries back into plain KVCaches, in place.

    Prefill must never run against a quantized cache. Quantized attention has
    no fused causal kernel, so it falls back to materializing the score matrix
    in full: q_heads x chunk x context x 4 B, which at this server's 32k
    prefill step is tens of GB. The request then either aborts the process
    ("[METAL] Command buffer execution failed: Insufficient Memory", which is
    an *async* command-buffer failure, so there is nothing to catch) or
    survives with numerically broken activations and decodes fluent nonsense.

    `quantized_kv_start` keeps *this* request's prefill off that path, but it
    cannot help when the cache arrives already quantized - which is every turn
    after the first, since decode quantizes the cache that then gets stored and
    reused. Undoing it here costs one dequantize and the float bytes a
    non-reusing request would have spent anyway; decode re-quantizes once the
    prompt is in.

    The conversion is done one layer at a time, evaluating each layer's floats
    and dropping the quantized arrays that produced them before starting the
    next. Building the whole fp16 cache as one lazy graph - which is what the
    obvious loop does - holds every quantized layer alive until the eval that
    consumes it, so the peak is both formats at once: 7.8 GB on GLM-4.7 at 13k
    tokens instead of 5.1 GB. We always own this cache (it is either our own
    handover from the store or upstream's deepcopy of it), so freeing the
    source as we go cannot be seen by anyone else.

    Returns the number of layers converted.
    """
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    converted = 0
    for i, c in enumerate(cache):
        if not isinstance(c, QuantizedKVCache):
            continue
        plain = KVCache()
        if c.keys is None:
            plain.offset = c.offset
        else:

            def deq(packed_scales_biases):
                return mx.dequantize(
                    *(x[..., : c.offset, :] for x in packed_scales_biases),
                    group_size=c.group_size,
                    bits=c.bits,
                )

            # The state setter takes the offset from the arrays' length.
            plain.state = (deq(c.keys), deq(c.values))
            mx.eval(plain.keys, plain.values)
            # This layer's floats exist now, so its quantized source is dead
            # weight. Dropping it here is what keeps the peak at one format
            # plus one layer rather than two whole caches.
            c.keys = None
            c.values = None
        cache[i] = plain
        converted += 1
    if converted:
        mx.clear_cache()
    return converted


def requantize(prompt_cache, quantized_kv_start, kv_group_size, kv_bits) -> None:
    """Drop-in for `mlx_lm.generate.maybe_quantize_kv_cache`, without the peak.

    Upstream assigns `c.to_quantized(...)` for every layer in one pass. Those
    are lazy, and each one holds a reference to the fp16 arrays it quantizes,
    so by the time anything evaluates them both formats of the whole cache are
    live: 13.8 GB on GLM-4.7 at 24k tokens where 4.8 GB is the steady state.
    That peak lands immediately after a long prefill, next to a slab the
    decode path has just rebuilt - the worst possible moment.

    Same conversion, same result, one layer at a time.
    """
    if kv_bits is None:
        return
    converted = 0
    for e, c in enumerate(prompt_cache):
        if not (hasattr(c, "to_quantized") and c.offset >= quantized_kv_start):
            continue
        q = c.to_quantized(group_size=kv_group_size, bits=kv_bits)
        if q.keys is not None:
            mx.eval(q.keys, q.values)
            # Same reasoning as unquantize(): the floats this layer came from
            # are unreachable the moment the quantized version is materialized,
            # but only if we say so - the lazy graph would hold them otherwise.
            c.keys = None
            c.values = None
        prompt_cache[e] = q
        converted += 1
    if converted:
        mx.clear_cache()


# ------------------------------------------------------------ the LRU store


def take_nearest_cache(store, model_key, tokens):
    """`LRUPromptCache.fetch_nearest_cache`, handing the cache over by
    reference instead of deep-copying it.

    Upstream copies because a stored cache may be reused by a later request
    that shares the same prefix. For an agent loop that is a copy nothing ever
    reads: the turn extends the very prefix it resumed from, and the longer
    cache it stores at the end supersedes the entry it came from -
    `insert_cache` even evicts stored prefixes of a trimmable cache itself.
    So the copy is a second 4.8 GB of KV (GLM at 24k) that exists to be
    thrown away, plus the seconds it takes to allocate and memcpy.

    Handing the entry over instead means the store drops it now and gets the
    longer version back when the turn ends. The cost is that a turn which dies
    mid-flight takes the prefix with it, so the next request re-prefills; that
    is a slow turn, where keeping the copy is a dead turn.

    Set EXPERT_STREAM_PROMPT_CACHE_MOVE=0 to restore upstream's copying.
    """
    import copy

    from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

    result = store._trie.search(model_key, tokens)

    def take(key):
        """Remove `key` from the store and return its cache, uncopied."""
        entry = store._trie.pop(model_key, key)
        store._n_bytes -= entry.nbytes
        store._n_bytes_by_type[entry.cache_type] -= entry.nbytes
        # The LRU holds (model, tokens) and compares by value, so the freshly
        # sliced key from the trie search matches the list insert_cache pushed.
        store._lru.remove(model_key, key)
        return entry.prompt_cache

    if result.exact is not None:
        return take(result.exact), []

    short_length = len(result.shorter) if result.shorter is not None else 0
    if result.longer is not None and result.common_prefix > short_length:
        key = result.longer
        entry_cache = store._trie.get(result.model, key).prompt_cache
        if can_trim_prompt_cache(entry_cache):
            prefix = min(len(tokens) - 1, result.common_prefix)
            num_to_trim = len(key) - prefix
            # A longer stored sequence has diverged from this request, so its
            # tail is what we are dropping anyway; trimming it in place is
            # only destructive to a branch that already lost the prefix race.
            cache = take(key)
            trim_prompt_cache(cache, num_to_trim)
            return cache, tokens[prefix:]

    if short_length > 0:
        return take(result.shorter), tokens[short_length:]

    return None, tokens


def bound_store(store) -> int:
    """Trim the prompt-cache LRU to the published byte budget.

    Call after every insert. `LRUPromptCache` can do this itself - it just
    never gets told the number on this code path (see the module docstring).

    Returns the bytes trimmed away, for logging.
    """
    budget = _store_budget
    if budget <= 0:
        return 0
    before = store.nbytes
    if before <= budget:
        return 0
    store.trim_to(n_bytes=budget)
    # Entries the store just dropped are the only reference to those arrays.
    mx.clear_cache()
    return before - store.nbytes
