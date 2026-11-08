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
  `ResumedKVCache`       resume an 8-bit cache without an fp16 copy at all
  `unquantize`           dequantize layer by layer, freeing as it goes
  `requantize`           the same for the reverse conversion
  `bound_store`          enforce a byte budget on the prompt-cache LRU

Nothing here changes a single logit: it is the same cache in the same numeric
format, just never duplicated. What it buys is memory, which the loader then
hands to the expert cache as throughput (`kv_reserve_bytes`).
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import _BaseCache

from . import config

# Bytes the prompt-cache store may hold. Set by the loader once the model's
# memory profile is known (see `set_store_budget`); 0 means "not yet known",
# in which case we leave upstream's behavior alone.
_store_budget: int = 0

# Per-token KV cost of the loaded model, in bytes at the cache's decode-time
# precision. Filled in by the loader, read by the bench/probe scripts.
_kv_bytes_per_token: int = 0

# Served context (clamped to the model's positions) and the fp16 per-token
# cost, published by the loader for `kv_room`.
_reserve_ctx: int = 0
_kv_fp16_per_token: int = 0

# Growth `kv_room` always allows for, even past the served context: about two
# agent turns.
_ROOM_MIN_TOKENS = 2048


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
        index_dim = int(model_config.get("index_head_dim") or 0)
        if index_dim and model_config.get("index_topk"):
            # DSA (deepseek_v32 family) keeps a CacheList per layer, which
            # requantize cannot convert, so it is model dtype at any length.
            # Layers that run the indexer also cache one key per token.
            from .dsa_shared import indexer_types

            types = indexer_types(model_config)
            elems += (types.count("full") if types else layers) * index_dim
            per_elem = 2.0
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
    conversation, in whatever precision decode uses. A resumed turn does not
    need an fp16 copy of it on top (`ResumedKVCache`), so that is all there is.

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


def set_kv_reserve(model_config: dict, ctx_tokens: int) -> None:
    global _reserve_ctx, _kv_fp16_per_token
    limit = int(model_config.get("max_position_embeddings") or 0)
    ctx = max(0, int(ctx_tokens))
    _reserve_ctx = min(ctx, limit) if limit > 0 else ctx
    _kv_fp16_per_token = kv_bytes_per_token(model_config, bits=None)


def kv_room(n_tokens: int) -> int:
    """KV bytes a conversation at `n_tokens` can still grow by.

    Up to the served context (numCtx), and never less than _ROOM_MIN_TOKENS
    of growth. Below KV_FP16_CTX the cache is fp16, so the peak on the way up
    may be the fp16 cache just under the threshold rather than the 8-bit one
    at numCtx. The expert cache keeps this much free when it sizes a slab, so
    the slab it builds now still fits a few turns later.
    """
    q = _kv_bytes_per_token
    f = _kv_fp16_per_token or 2 * q
    fp16_below = int(config.KV_FP16_CTX)
    n = max(0, int(n_tokens))

    def held(m: int) -> int:
        return (f if m < fp16_below else q) * m

    peak = held(_reserve_ctx)
    if n < fp16_below:
        peak = max(peak, held(min(_reserve_ctx, fp16_below - 1)))
    floor = (f if n < fp16_below else q) * _ROOM_MIN_TOKENS
    return max(floor, peak - held(n))


# ------------------------------------------------------- copy-free conversion


def _carry_sides(src, dst):
    """Move caches hung off a KV layer (qwen4_exp's `_AttnCache.indexer`) onto
    its converted replacement and return `dst`.

    The model looks them up with hasattr. A conversion that drops one does not
    fail: QSA attention silently goes dense for the rest of the conversation,
    and a hybrid checkpoint no longer matches the stored entry's layout.
    """
    for name, v in vars(src).items():
        if name.startswith("_"):
            continue  # ResumedKVCache._tail is its own storage, not a side cache
        if isinstance(v, _BaseCache) and v is not src and not hasattr(dst, name):
            setattr(dst, name, v)
    return dst


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
        cache[i] = _carry_sides(c, plain)
        converted += 1
    if converted:
        mx.clear_cache()
    return converted


class ResumedKVCache(_BaseCache):
    """A reused 8-bit cache that prefill extends as if it were fp16.

    `unquantize` builds the whole fp16 cache before prefill starts, which is a
    second copy of the conversation at twice the size. On a 48 GB machine that
    copy is what made every resumed agent turn give the expert slab back:
    Qwen3-235B at 15k tokens needed 1.55 GB more than the 1.38 GB of headroom
    next to its slab, so the slab and every resident expert went, and the
    700-token turn re-read 131 GB.

    This keeps the reused prefix in 8 bits and dequantizes it one layer at a
    time, inside that layer's attention, so the extra memory is one layer's
    floats rather than the cache. New tokens go into an ordinary fp16 tail.

    Bit-identical to the fp16 path. Attention gets dequantize(prefix) followed
    by the new keys at full precision, which is exactly what unquantize plus
    KVCache hand it. `to_quantized` quantizes the same rows; quantization is
    per row, so building them from two pieces does not change a bit.

    Not a KVCache subclass and no `bits` attribute: mlx-lm switches to
    quantized attention on `hasattr(cache, "bits")`.
    """

    def __init__(self, keys=None, values=None, offset: int = 0, group_size: int = 64, bits: int = 8):
        from mlx_lm.models.cache import KVCache

        self._qk = keys
        self._qv = values
        self._p = int(offset) if keys is not None else 0
        self._group = int(group_size)
        self._qbits = int(bits)
        self._tail = KVCache()

    @classmethod
    def wrap(cls, c) -> "ResumedKVCache | None":
        """The same KV as QuantizedKVCache `c`, or None if `c` holds no rows."""
        n = int(c.offset)
        if c.keys is None or n <= 0:
            return None
        keys = tuple(x[..., :n, :] for x in c.keys)
        values = tuple(x[..., :n, :] for x in c.values)
        return cls(keys, values, n, c.group_size, c.bits)

    @property
    def offset(self) -> int:
        return self._p + self._tail.offset

    def _deq(self, q):
        return mx.dequantize(
            *(x[..., : self._p, :] for x in q), group_size=self._group, bits=self._qbits
        )

    def _tail_rows(self):
        t = self._tail
        if t.keys is None or t.offset == 0:
            return None, None
        return t.keys[..., : t.offset, :], t.values[..., : t.offset, :]

    def _full(self, k, v):
        if not self._p:
            return k, v
        dk, dv = self._deq(self._qk), self._deq(self._qv)
        if k is None:
            return dk, dv
        return mx.concatenate([dk, k], axis=2), mx.concatenate([dv, v], axis=2)

    def update_and_fetch(self, keys, values):
        k, v = self._tail.update_and_fetch(keys, values)
        return self._full(k, v)

    def size(self) -> int:
        return self.offset

    @property
    def state(self):
        if self._p:
            qk = tuple(x[..., : self._p, :] for x in self._qk)
            qv = tuple(x[..., : self._p, :] for x in self._qv)
        else:
            qk = qv = None
        tk, tv = self._tail_rows()
        return qk, qv, tk, tv

    @state.setter
    def state(self, v):
        from mlx_lm.models.cache import KVCache

        qk, qv, tk, tv = v
        self._qk, self._qv = qk, qv
        self._p = 0 if qk is None else int(qk[0].shape[2])
        self._tail = KVCache()
        if tk is not None:
            self._tail.state = (tk, tv)

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, int(n))
        from_tail = self._tail.trim(n)
        self._p -= n - from_tail
        return n

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        from mlx_lm.models.cache import QuantizedKVCache

        q = QuantizedKVCache(group_size=group_size, bits=bits)
        q.offset = self.offset
        k, v = self._full(*self._tail_rows())
        if k is not None:
            q.keys = mx.quantize(k, group_size=group_size, bits=bits)
            q.values = mx.quantize(v, group_size=group_size, bits=bits)
        return q

    def release(self) -> None:
        """Drop this object's arrays (after `to_quantized` has been evaluated)."""
        from mlx_lm.models.cache import KVCache

        self._qk = self._qv = None
        self._p = 0
        self._tail = KVCache()

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self) -> bool:
        return self._p == 0 and self._tail.empty()

    @property
    def nbytes(self) -> int:
        n = self._tail.nbytes
        if self._p:
            for x in (*self._qk, *self._qv):
                n += int(x.nbytes) * self._p // max(1, int(x.shape[2]))
        return n


def resume_lazily(cache: list) -> int:
    """Replace reused QuantizedKVCache layers with ResumedKVCache, in place.

    The alternative to `unquantize` for a cache made only of plain attention
    layers. Allocates nothing. Returns the number of layers replaced.
    """
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    done = 0
    for i, c in enumerate(cache):
        if not isinstance(c, QuantizedKVCache):
            continue
        lazy = ResumedKVCache.wrap(c)
        if lazy is None:
            lazy = KVCache()
            lazy.offset = c.offset
        cache[i] = _carry_sides(c, lazy)
        done += 1
    return done


def lazy_resume_ok(cache: list) -> bool:
    """True when every layer is plain attention KV, the only layout
    `resume_lazily` is written for. Hybrid and MLA layouts keep `unquantize`."""
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    return bool(cache) and all(
        type(c) in (KVCache, QuantizedKVCache, ResumedKVCache) for c in cache
    )


def has_resumed(cache) -> bool:
    return bool(cache) and any(isinstance(c, ResumedKVCache) for c in cache)


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
            if isinstance(c, ResumedKVCache):
                c.release()
            else:
                c.keys = None
                c.values = None
        prompt_cache[e] = _carry_sides(c, q)
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

    Non-trimmable caches (hybrid / linear attention) are the exception: a later
    longer insert cannot be trimmed back to this prefix, so MOVE would leave a
    digest/rewrite follow-up with nothing shorter to hit. Those stay as copies.

    Set PAGED_MOE_PROMPT_CACHE_MOVE=0 to restore upstream's copying.
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

    def borrow(key):
        """Copy a stored cache and leave the store entry in place.

        Hybrid / linear-attention KV (qwen4_exp, qwen3-next) cannot trim a
        longer later insert back down to this prefix. If we MOVE the short
        snap out and only insert the extended end, a follow-up whose prompt
        shortens (tool digest, cancel rewrite) has nothing to hit and cold-
        prefills from kv=0 even though the store still shows GBs of sequences.
        """
        entry = store._trie.get(model_key, key)
        return copy.deepcopy(entry.prompt_cache)

    if result.exact is not None:
        entry_cache = store._trie.get(result.model, result.exact).prompt_cache
        if can_trim_prompt_cache(entry_cache):
            return take(result.exact), []
        return borrow(result.exact), []

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
        key = result.shorter
        entry_cache = store._trie.get(result.model, key).prompt_cache
        if can_trim_prompt_cache(entry_cache):
            return take(key), tokens[short_length:]
        return borrow(key), tokens[short_length:]

    return None, tokens


def _lru_tokens(store, model_key, tokens):
    """The tokens object the LRU holds for this entry, or None.

    The LRU removes by equality on (model, tokens), and a tuple never equals
    the list it was inserted as.
    """
    want = tuple(tokens)
    for lru in (getattr(store._lru, "_lrus", None) or {}).values():
        for m, t in lru:
            if m == model_key and tuple(t) == want:
                return t
    return None


def bound_store(store, keep=None) -> int:
    """Trim the prompt-cache LRU to the published byte budget.

    Call after every insert. `LRUPromptCache` can do this itself - it just
    never gets told the number on this code path (see the module docstring).
    Under recent memory pressure the budget is PROMPT_CACHE_PRESSURE_FRAC of
    that number.

    `keep` is the (model_key, tokens) just inserted, or a list of such keys.
    Those entries are never evicted, even when they alone are over budget: it
    is the conversation the next turn resumes, and dropping it only means
    rebuilding the same KV by re-prefilling everything past the last snapshot
    (8k tokens / ~40 s on a 235B once the chat outgrows the reserved context).
    Holding it costs no more than the turn that just ran, and the next turn
    takes it back out.

    Returns the bytes trimmed away, for logging.
    """
    budget = _store_budget
    if budget <= 0:
        return 0
    from . import hybrid_ckpt, mempressure

    pressed = mempressure.recent(config.SLAB_REGROW_CALM_S)
    if pressed:
        frac = min(1.0, max(0.0, float(config.PROMPT_CACHE_PRESSURE_FRAC)))
        budget = int(budget * frac)
    # Recurrent-state checkpoints hold arrays no store entry accounts for.
    budget = max(1, budget - hybrid_ckpt.nbytes())
    before = store.nbytes
    if before <= budget:
        return 0
    if keep is None:
        keep = []
    elif isinstance(keep, tuple):
        keep = [keep]
    hidden = []
    seen = set()
    for key in keep:
        try:
            model_key, tokens = key
            entry = store._trie.get(model_key, tokens)
        except (KeyError, TypeError, ValueError):
            continue
        tokens = _lru_tokens(store, model_key, tokens)
        if entry is None or tokens is None or (model_key, tuple(tokens)) in seen:
            continue
        seen.add((model_key, tuple(tokens)))
        hidden.append((model_key, tokens, entry))
    # mlx-lm pops by cache type before age, so a size floor alone could still
    # take a kept entry ahead of older snapshots. Hide them instead.
    held = 0
    for model_key, tokens, entry in hidden:
        store._lru.remove(model_key, tokens)
        store._n_bytes -= entry.nbytes
        store._n_bytes_by_type[entry.cache_type] -= entry.nbytes
        held += entry.nbytes
    try:
        store.trim_to(n_bytes=max(0, budget - held))
    finally:
        for model_key, tokens, entry in hidden:
            store._n_bytes += entry.nbytes
            store._n_bytes_by_type[entry.cache_type] += entry.nbytes
            store._lru.push(model_key, tokens, entry.cache_type)
    # Entries the store just dropped are the only reference to those arrays.
    mx.clear_cache()
    freed = before - store.nbytes
    if pressed and freed > 0:
        print(
            f"[paged-moe] memory pressure: prompt cache {before / 1e9:.2f} -> {store.nbytes / 1e9:.2f} GB",
            flush=True,
        )
    return freed
