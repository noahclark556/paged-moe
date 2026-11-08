# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""KV-cache memory management: byte math, copy-free conversion, store bounds.

These are the guards on the changes that stopped GLM-4.7 Metal-OOMing on turn
two. The properties worth pinning down are (a) the conversions are numerically
identical to mlx-lm's, since they run on every resumed turn, (b) the handover
fetch returns the same (cache, rest) split as upstream's copying fetch, and
(c) the store actually gets bounded.

No checkpoint needed - tiny synthetic caches exercise all of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import KVCache, LRUPromptCache, QuantizedKVCache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pagedmoe import config, kvmem

GLM_47 = {
    "num_hidden_layers": 92,
    "num_attention_heads": 96,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_size": 5120,
}
QWEN3_235B = {
    "num_hidden_layers": 94,
    "num_attention_heads": 64,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "hidden_size": 4096,
}
DEEPSEEK_V3 = {
    "num_hidden_layers": 61,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "hidden_size": 7168,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
}


def _filled(offset: int, layers: int = 4, kv_heads: int = 2, head_dim: int = 64):
    """A KV cache holding `offset` tokens of deterministic junk."""
    cache = []
    for i in range(layers):
        c = KVCache()
        k = mx.random.normal((1, kv_heads, offset, head_dim), key=mx.random.key(i))
        v = mx.random.normal(
            (1, kv_heads, offset, head_dim), key=mx.random.key(100 + i)
        )
        c.update_and_fetch(k, v)
        cache.append(c)
    mx.eval([c.state for c in cache])
    return cache


def _close(a: float, b: float, rel: float = 1e-6) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1.0)


# ----------------------------------------------------------------- byte math


def test_kv_bytes_matches_hand_computed_geometry():
    # 2 (K+V) * 92 layers * 8 heads * 128 dim = 188,416 elements per token.
    assert kvmem.kv_bytes_per_token(GLM_47, bits=None) == 188_416 * 2
    # 8-bit plus one fp16 scale and bias per 64-element group.
    assert kvmem.kv_bytes_per_token(GLM_47, bits=8) == int(188_416 * (1 + 4 / 64))


def test_glm_kv_is_roughly_twice_qwen_per_token():
    """The whole reason GLM needed its own memory accounting."""
    ratio = kvmem.kv_bytes_per_token(GLM_47, bits=8) / kvmem.kv_bytes_per_token(
        QWEN3_235B, bits=8
    )
    assert 1.9 < ratio < 2.0, ratio


def test_mla_is_not_charged_the_dense_rate():
    """DeepSeek-V3 caches one latent per layer, not K and V per head. Charging
    it the dense rate would reserve ~30x too much and starve the experts."""
    dense = kvmem.kv_bytes_per_token({**DEEPSEEK_V3, "kv_lora_rank": None}, bits=8)
    mla = kvmem.kv_bytes_per_token(DEEPSEEK_V3, bits=8)
    assert mla < dense / 10, (mla, dense)


def test_dsa_is_priced_as_it_is_held():
    """DSA's CacheList never quantizes, and the indexer caches a key per token
    on the layers that run it. GLM-5.2 at 49k was reserving 2.9 GB for 4.7."""
    v32 = {**DEEPSEEK_V3, "index_head_dim": 128, "index_topk": 2048}
    want = 61 * (576 + 128) * 2
    assert kvmem.kv_bytes_per_token(v32, bits=8) == want
    assert kvmem.kv_bytes_per_token(v32, bits=None) == want
    glm52 = {
        **v32,
        "num_hidden_layers": 78,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
    }
    assert kvmem.kv_bytes_per_token(glm52, bits=8) == (78 * 576 + 21 * 128) * 2
    # Plain MLA keeps the quantized rate.
    assert kvmem.kv_bytes_per_token(DEEPSEEK_V3, bits=8) == int(61 * 576 * (1 + 4 / 64))


def test_reserve_is_clamped_to_the_checkpoints_positions():
    """The host app ships Qwen3-235B at numCtx 65536 against a 40960-position checkpoint.
    Reserving for the difference would cost ~2 GB of expert residency to hold KV
    for tokens the model cannot attend to."""
    cfg = {**QWEN3_235B, "max_position_embeddings": 40960}
    assert kvmem.kv_reserve_bytes(cfg, 65536) == kvmem.kv_reserve_bytes(cfg, 40960)
    assert kvmem.kv_reserve_bytes(cfg, 16384) < kvmem.kv_reserve_bytes(cfg, 40960)


def test_unknown_architecture_reserves_nothing():
    """No layer count means no estimate; the loader falls back to its flat
    allowance rather than guessing a number that starves the cache."""
    assert kvmem.kv_bytes_per_token({}, bits=8) == 0
    assert kvmem.kv_reserve_bytes({}, 24576) == 0
    assert kvmem.kv_reserve_bytes(GLM_47, 0) == 0


def test_reserve_follows_the_fp16_policy_for_short_contexts():
    """Under KV_FP16_CTX the cache stays fp16, which costs ~2x - the reserve has
    to follow the policy, not the nominal precision."""
    saved = config.KV_FP16_CTX
    try:
        config.KV_FP16_CTX = 8192
        short = kvmem.kv_reserve_bytes(GLM_47, 4096)
        assert _close(
            short,
            kvmem.kv_bytes_per_token(GLM_47, bits=None) * 4096 * config.KV_STORE_SLACK,
        )
        # Past the threshold the 8-bit cost at the full context dominates.
        long = kvmem.kv_reserve_bytes(GLM_47, 24576)
        assert _close(
            long,
            kvmem.kv_bytes_per_token(GLM_47, bits=8) * 24576 * config.KV_STORE_SLACK,
        )
    finally:
        config.KV_FP16_CTX = saved


# ------------------------------------------------------ copy-free conversion


def test_unquantize_is_numerically_identical_to_upstream():
    """Freeing each layer's quantized source as we go must not change a value -
    this cache is what the next 24k tokens of context attend to."""
    plain = _filled(96)
    quantized = [c.to_quantized(group_size=64, bits=8) for c in plain]
    reference = [
        mx.dequantize(*q.keys, group_size=q.group_size, bits=q.bits)[..., : q.offset, :]
        for q in quantized
    ]
    mx.eval(reference)

    converted = list(quantized)
    assert kvmem.unquantize(converted) == len(plain)
    for got, want, q in zip(converted, reference, quantized):
        assert isinstance(got, KVCache)
        assert got.offset == 96
        assert mx.array_equal(got.keys[..., :96, :], want)
        # The source is released, which is the point of doing it layer by layer.
        assert q.keys is None


def test_unquantize_leaves_a_plain_cache_alone():
    plain = _filled(32)
    assert kvmem.unquantize(plain) == 0
    assert all(isinstance(c, KVCache) for c in plain)


def test_unquantize_handles_an_empty_cache():
    """A stored cache can be reused before anything was appended to it."""
    empty = [QuantizedKVCache(group_size=64, bits=8) for _ in range(3)]
    assert kvmem.unquantize(empty) == 3
    assert all(isinstance(c, KVCache) and c.offset == 0 for c in empty)


def test_requantize_matches_upstream():
    from mlx_lm.generate import maybe_quantize_kv_cache

    mine = _filled(96)
    theirs = _filled(96)
    kvmem.requantize(mine, 0, 64, 8)
    maybe_quantize_kv_cache(theirs, 0, 64, 8)
    for a, b in zip(mine, theirs):
        assert isinstance(a, QuantizedKVCache)
        assert a.offset == b.offset
        for x, y in zip(a.keys, b.keys):
            assert mx.array_equal(x, y)
        for x, y in zip(a.values, b.values):
            assert mx.array_equal(x, y)


def test_requantize_respects_the_start_threshold():
    """quantized_kv_start is what keeps prefill on the fused causal kernel; a
    replacement that ignored it would decode fluent nonsense."""
    cache = _filled(64)
    kvmem.requantize(cache, 1024, 64, 8)
    assert all(isinstance(c, KVCache) for c in cache)


def test_requantize_is_a_noop_without_kv_bits():
    cache = _filled(8)
    kvmem.requantize(cache, 0, 64, None)
    assert all(isinstance(c, KVCache) for c in cache)


def test_conversion_does_not_compound_across_turns():
    """A resumed turn does quantize -> store -> dequantize -> prefill ->
    quantize, so a 20-turn session round-trips its oldest tokens 20 times.

    Values already sitting on the quantization grid re-quantize to themselves
    except for fp16 rounding of the scale and bias, so the drift has to
    *converge* rather than accumulate - otherwise a long agent session slowly
    corrupts the context it is trying to remember.
    """
    cache = _filled(64)

    def round_trip():
        kvmem.requantize(cache, 0, 64, 8)
        kvmem.unquantize(cache)
        out = [c.keys[..., :64, :] for c in cache]
        mx.eval(out)
        return out

    spread = float(mx.max(mx.abs(cache[0].keys)).item())
    passes = [round_trip() for _ in range(4)]
    drifts = [
        max(float(mx.max(mx.abs(a - b)).item()) for a, b in zip(prev, nxt))
        for prev, nxt in zip(passes, passes[1:])
    ]
    # One 8-bit step over the observed range; anything larger is not rounding.
    step = 2 * spread / 255
    assert all(d <= step for d in drifts), (drifts, step)
    assert drifts[-1] <= drifts[0], drifts


# ------------------------------------------------------------- the LRU store


def test_take_nearest_cache_splits_like_upstream():
    """Same (cache, rest) contract, with the handover instead of the copy."""
    tokens = list(range(1, 65))
    cases = {
        "exact": (tokens, tokens),
        "stored prefix is shorter": (tokens[:32], tokens),
        "stored sequence diverges": (tokens, tokens[:48] + [999]),
    }
    for name, (stored, asked) in cases.items():
        mine = LRUPromptCache(max_size=4)
        theirs = LRUPromptCache(max_size=4)
        mine.insert_cache("m", list(stored), _filled(len(stored)))
        theirs.insert_cache("m", list(stored), _filled(len(stored)))

        got, rest = kvmem.take_nearest_cache(mine, "m", asked)
        want, want_rest = theirs.fetch_nearest_cache("m", asked)
        assert rest == want_rest, name
        assert (got is None) == (want is None), name
        if got is not None:
            assert [c.offset for c in got] == [c.offset for c in want], name


def test_take_nearest_cache_removes_the_entry_it_hands_over():
    """The store must stop accounting for bytes it no longer owns, or the byte
    bound trims live caches to satisfy a phantom."""
    tokens = list(range(1, 65))
    store = LRUPromptCache(max_size=4)
    store.insert_cache("m", list(tokens), _filled(64))
    assert store.nbytes > 0

    cache, rest = kvmem.take_nearest_cache(store, "m", tokens)
    assert cache is not None and rest == []
    assert len(store) == 0
    assert store.nbytes == 0
    # And the entry is really gone from the trie, not just from the counters.
    assert kvmem.take_nearest_cache(store, "m", tokens) == (None, tokens)


def test_take_nearest_keeps_non_trimmable_prefixes():
    """Hybrid caches cannot trim a longer insert back down. MOVE would drop the
    short snap and leave a shortened follow-up (digest / rewrite) cold."""

    class _NoTrim:
        def __init__(self, offset: int, nbytes: int = 1000):
            self.offset = offset
            self.nbytes = nbytes

        def is_trimmable(self):
            return False

    tokens = list(range(1, 65))
    store = LRUPromptCache(max_size=8)
    store.insert_cache("m", list(tokens[:32]), [_NoTrim(32) for _ in range(2)], cache_type="user")
    assert len(store) == 1

    # Longer ask that shares the short prefix: borrow, do not remove.
    cache, rest = kvmem.take_nearest_cache(store, "m", tokens)
    assert cache is not None
    assert rest == tokens[32:]
    assert len(store) == 1
    assert store.nbytes > 0

    # Shortened follow-up still hits the kept prefix.
    cache2, rest2 = kvmem.take_nearest_cache(store, "m", tokens[:40])
    assert cache2 is not None
    assert rest2 == tokens[32:40]
    assert len(store) == 1


def test_take_nearest_cache_hands_over_the_same_arrays():
    """The saving only exists if nothing was copied."""
    tokens = list(range(1, 33))
    store = LRUPromptCache(max_size=4)
    original = _filled(32)
    keys = [c.keys for c in original]
    store.insert_cache("m", list(tokens), original)

    cache, _ = kvmem.take_nearest_cache(store, "m", tokens)
    assert [c.keys for c in cache] == keys


def test_take_nearest_cache_misses_cleanly_on_an_empty_store():
    store = LRUPromptCache(max_size=4)
    assert kvmem.take_nearest_cache(store, "m", [1, 2, 3]) == (None, [1, 2, 3])


def test_bound_store_trims_to_the_published_budget():
    store = LRUPromptCache(max_size=10)
    for i in range(4):
        store.insert_cache("m", [i, i + 1, i + 2], _filled(64))
    full = store.nbytes
    assert len(store) == 4

    try:
        kvmem.set_store_budget(full // 2)
        assert kvmem.bound_store(store) > 0
        assert store.nbytes <= full // 2
        assert len(store) < 4
    finally:
        kvmem.set_store_budget(0)


def test_bound_store_keeps_the_live_conversation_over_budget():
    """The entry just inserted survives even when it alone exceeds the budget,
    and older entries of any cache type go first."""
    store = LRUPromptCache(max_size=10)
    store.insert_cache("m", [9, 9, 9], _filled(64), cache_type="user")
    store.insert_cache("m", [8, 8, 8], _filled(64), cache_type="system")
    live = [1, 2, 3, 4]
    store.insert_cache("m", list(live), _filled(128))
    live_bytes = store._trie.get("m", live).nbytes

    try:
        kvmem.set_store_budget(live_bytes // 2)
        assert kvmem.bound_store(store, keep=("m", live)) > 0
        assert len(store) == 1
        assert store.nbytes == live_bytes
        # A new chat misses it; the fetch-time trim (no keep) drops it.
        cache, rest = kvmem.take_nearest_cache(store, "m", [7, 7, 7, 7])
        assert cache is None
        assert kvmem.bound_store(store) == live_bytes
        assert len(store) == 0
    finally:
        kvmem.set_store_budget(0)


def test_bound_store_keeps_every_listed_entry():
    """A branch keeps the entry it came from as well as its own, and a tuple
    key protects the entry the LRU holds as a list."""
    store = LRUPromptCache(max_size=10)
    store.insert_cache("m", [9, 9, 9], _filled(64))
    main = [1, 2, 3, 4]
    branch = [1, 2, 5, 6]
    store.insert_cache("m", list(main), _filled(64))
    store.insert_cache("m", list(branch), _filled(64))
    one = store._trie.get("m", main).nbytes
    try:
        kvmem.set_store_budget(one)
        kvmem.bound_store(store, keep=[("m", tuple(main)), ("m", branch)])
        assert len(store) == 2
        assert store._trie.get("m", main) is not None
        assert store._trie.get("m", branch) is not None
        assert store.nbytes == 2 * one
        assert sum(len(v) for v in store._lru._lrus.values()) == 2, "LRU got a duplicate key"
    finally:
        kvmem.set_store_budget(0)


def test_branch_keep_holds_the_hybrid_entry_a_prompt_branched_from():
    from mlx_lm.models.cache import ArraysCache

    from pagedmoe import server

    def hybrid(n):
        c = ArraysCache(1)
        c[0] = mx.zeros((1, 4))
        return [c, *_filled(n, layers=1)]

    store = LRUPromptCache(max_size=10)
    main = list(range(1, 401))
    store.insert_cache("m", list(main), hybrid(400))
    saved = server._live_key, config.HYBRID_CKPT_MIN_TOKENS
    try:
        server._live_key = ("m", tuple(main))
        config.HYBRID_CKPT_MIN_TOKENS = 64
        branched = main[:300] + [7, 7, 7]
        assert server._branch_keep(store, "m", branched) == [("m", tuple(main))]
        assert server._branch_keep(store, "m", main[:50] + [7] * 300) == [], "new chat"
        assert server._branch_keep(store, "m", main + [7]) == [], "continues it"
        assert server._branch_keep(store, "other", branched) == []
    finally:
        server._live_key, config.HYBRID_CKPT_MIN_TOKENS = saved


def test_bound_store_is_inert_until_the_loader_publishes_a_budget():
    """A budget of 0 means "not known yet" - trimming then would throw away
    prefixes on the strength of no information at all."""
    store = LRUPromptCache(max_size=10)
    store.insert_cache("m", [1, 2, 3], _filled(64))
    kvmem.set_store_budget(0)
    assert kvmem.bound_store(store) == 0
    assert len(store) == 1


def test_bound_store_shrinks_under_recent_pressure():
    """Recent pressure keeps PROMPT_CACHE_PRESSURE_FRAC of the budget, and the
    live conversation still survives."""
    from unittest import mock

    from pagedmoe import config, mempressure

    store = LRUPromptCache(max_size=10)
    for i in range(4):
        store.insert_cache("m", [i, i + 1, i + 2], _filled(64))
    live = [1, 2, 3, 4]
    store.insert_cache("m", list(live), _filled(64))
    full = store.nbytes
    live_bytes = store._trie.get("m", live).nbytes
    try:
        kvmem.set_store_budget(full)
        with mock.patch.object(mempressure, "recent", lambda _w: False):
            assert kvmem.bound_store(store, keep=("m", live)) == 0
        with mock.patch.object(mempressure, "recent", lambda _w: True), \
                mock.patch.object(config, "PROMPT_CACHE_PRESSURE_FRAC", 0.4):
            assert kvmem.bound_store(store, keep=("m", live)) > 0
            assert store.nbytes <= max(int(full * 0.4), live_bytes)
            assert store._trie.get("m", live) is not None
        with mock.patch.object(mempressure, "recent", lambda _w: True), \
                mock.patch.object(config, "PROMPT_CACHE_PRESSURE_FRAC", 1.0):
            assert kvmem.bound_store(store, keep=("m", live)) == 0, "1 must be off"
    finally:
        kvmem.set_store_budget(0)


def test_pressure_trim_is_logged(capsys):
    from unittest import mock

    from pagedmoe import config, mempressure

    store = LRUPromptCache(max_size=10)
    for i in range(3):
        store.insert_cache("m", [i, i + 1, i + 2], _filled(64))
    try:
        kvmem.set_store_budget(store.nbytes)
        with mock.patch.object(mempressure, "recent", lambda _w: True), \
                mock.patch.object(config, "PROMPT_CACHE_PRESSURE_FRAC", 0.3):
            assert kvmem.bound_store(store) > 0
        assert "memory pressure: prompt cache" in capsys.readouterr().out
    finally:
        kvmem.set_store_budget(0)


def test_recent_pressure_is_off_without_the_poller():
    from unittest import mock

    from pagedmoe import mempressure

    now = mempressure.time.monotonic()
    with mock.patch.object(mempressure, "_last_bad", None):
        assert not mempressure.recent(60.0)
    with mock.patch.object(mempressure, "_last_bad", now - 10), \
            mock.patch.object(mempressure, "_seen_bad", now - 10), \
            mock.patch.object(mempressure, "level", lambda: mempressure.NORMAL):
        assert mempressure.recent(60.0)
        assert not mempressure.recent(5.0)
    with mock.patch.object(mempressure, "_last_bad", now - 999), \
            mock.patch.object(mempressure, "_seen_bad", None), \
            mock.patch.object(mempressure, "level", lambda: mempressure.WARN):
        assert mempressure.recent(5.0), "warn right now counts"


def test_poller_start_is_not_pressure():
    from unittest import mock

    from pagedmoe import mempressure

    with mock.patch.object(mempressure, "_last_bad", mempressure.time.monotonic()), \
            mock.patch.object(mempressure, "_seen_bad", None), \
            mock.patch.object(mempressure, "level", lambda: mempressure.NORMAL):
        assert not mempressure.recent(60.0)
        assert mempressure.calm_s() < 60.0, "regrow still waits for measured calm"


def _quantized(offset: int, dtype, layers: int = 3):
    out = []
    for c in _filled(offset, layers=layers):
        c.keys = c.keys.astype(dtype)
        c.values = c.values.astype(dtype)
        out.append(c.to_quantized(group_size=64, bits=8))
    mx.eval([c.state for c in out])
    return out


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a.shape == b.shape and a.dtype == b.dtype and bool(mx.array_equal(a, b).item())


def _attend(q, k, v, n: int):
    return mx.fast.scaled_dot_product_attention(
        q, k, v, scale=0.125, mask="causal" if n > 1 else None
    )


def test_resumed_cache_is_bit_identical_to_unquantize():
    """Prefill chunks, decode steps and the decode-time re-quantize all see
    exactly what the full fp16 dequantize gave them, in both KV dtypes."""
    import copy

    for dtype in (mx.float16, mx.bfloat16):
        src = _quantized(300, dtype)
        full, lazy = copy.deepcopy(src), copy.deepcopy(src)
        kvmem.unquantize(full)
        assert kvmem.lazy_resume_ok(lazy)
        assert kvmem.resume_lazily(lazy) == len(lazy)
        assert all(isinstance(c, kvmem.ResumedKVCache) for c in lazy)
        step = 0
        for n in (37, 100, 1, 1):
            for i, (a, b) in enumerate(zip(full, lazy)):
                key = mx.random.key(1000 + 10 * step + i)
                k = mx.random.normal((1, 2, n, 64), key=key).astype(dtype)
                v = mx.random.normal((1, 2, n, 64), key=mx.random.split(key)[0]).astype(dtype)
                q = mx.random.normal((1, 4, n, 64), key=mx.random.split(key)[1]).astype(dtype)
                ka, va = a.update_and_fetch(k, v)
                kb, vb = b.update_and_fetch(k, v)
                assert _same(ka, kb) and _same(va, vb), (dtype, n, i)
                assert _same(_attend(q, ka, va, n), _attend(q, kb, vb, n)), (dtype, n, i)
                assert a.offset == b.offset
            step += 1
        kvmem.requantize(full, 0, 64, 8)
        kvmem.requantize(lazy, 0, 64, 8)
        for a, b in zip(full, lazy):
            assert isinstance(b, QuantizedKVCache)
            assert a.offset == b.offset
            assert _same(a.state, b.state)
            k = mx.ones((1, 2, 1, 64), dtype=dtype)
            assert _same(a.update_and_fetch(k, k), b.update_and_fetch(k, k))


def test_resumed_cache_trims_like_a_plain_cache():
    """Trimming back into the reused prefix and extending again (a store
    entry cut to a shorter prompt, verify's unfeed) matches KVCache."""
    import copy

    src = _quantized(200, mx.float16, layers=1)
    full, lazy = copy.deepcopy(src), copy.deepcopy(src)
    kvmem.unquantize(full)
    kvmem.resume_lazily(lazy)
    a, b = full[0], lazy[0]
    k = mx.random.normal((1, 2, 50, 64), key=mx.random.key(7)).astype(mx.float16)
    a.update_and_fetch(k, k)
    b.update_and_fetch(k, k)
    for n in (20, 60):  # inside the tail, then through it into the prefix
        assert a.trim(n) == b.trim(n) == n
        assert a.offset == b.offset
    k2 = mx.random.normal((1, 2, 9, 64), key=mx.random.key(8)).astype(mx.float16)
    assert _same(a.update_and_fetch(k2, k2), b.update_and_fetch(k2, k2))
    held = a.offset
    assert a.trim(10_000) == b.trim(10_000) == held
    assert a.offset == b.offset == 0


def test_resumed_cache_round_trips_state_and_copies_independently():
    import copy

    lazy = _quantized(128, mx.float16, layers=1)
    kvmem.resume_lazily(lazy)
    c = lazy[0]
    k = mx.random.normal((1, 2, 5, 64), key=mx.random.key(3)).astype(mx.float16)
    c.update_and_fetch(k, k)
    before = c.offset
    c.state = c.state
    assert c.offset == before == 133
    snap = copy.deepcopy(c)
    snap.trim(40)
    one = mx.zeros((1, 2, 1, 64), dtype=mx.float16)
    c.update_and_fetch(one, one)
    assert snap.offset == 93 and c.offset == 134
    mx.eval(snap.state, c.state)


def test_resumed_cache_stays_off_the_quantized_attention_path():
    """mlx-lm picks quantized SDPA on hasattr(cache, "bits"); the floats this
    cache hands back must go to the fused kernel."""
    lazy = _quantized(64, mx.float16, layers=1)
    held = sum(x.nbytes for part in lazy[0].state for x in part)
    kvmem.resume_lazily(lazy)
    c = lazy[0]
    assert not hasattr(c, "bits")
    assert c.is_trimmable() and c.size() == 64
    assert abs(c.nbytes - held) <= held // 64


def test_lazy_resume_only_takes_plain_attention_layouts():
    from mlx_lm.models.cache import ArraysCache

    q = _quantized(64, mx.float16, layers=2)
    assert kvmem.lazy_resume_ok(q)
    assert not kvmem.lazy_resume_ok(q + [ArraysCache(2)])
    assert not kvmem.lazy_resume_ok([])
    empty = QuantizedKVCache(group_size=64, bits=8)
    mixed = [empty]
    assert kvmem.resume_lazily(mixed) == 1
    assert type(mixed[0]) is KVCache and mixed[0].offset == 0
    assert not kvmem.has_resumed(mixed)


def test_kv_room_follows_the_served_context():
    cfg = dict(QWEN3_235B, max_position_embeddings=40960)
    q = kvmem.kv_bytes_per_token(cfg, bits=8)
    f = kvmem.kv_bytes_per_token(cfg, bits=None)
    old = (kvmem._kv_bytes_per_token, kvmem._reserve_ctx, kvmem._kv_fp16_per_token)
    try:
        kvmem.set_store_budget(1 << 30, q)
        kvmem.set_kv_reserve(cfg, 20480)
        assert kvmem.kv_room(16000) == q * (20480 - 16000)
        assert kvmem.kv_room(30000) == q * kvmem._ROOM_MIN_TOKENS
        # Short context is fp16 until KV_FP16_CTX; the peak on the way to
        # numCtx is whichever of the two caches is bigger.
        peak = max(q * 20480, f * (config.KV_FP16_CTX - 1))
        assert kvmem.kv_room(1000) == peak - f * 1000
        kvmem.set_kv_reserve(cfg, 65536)
        assert kvmem._reserve_ctx == 40960
    finally:
        kvmem._kv_bytes_per_token, kvmem._reserve_ctx, kvmem._kv_fp16_per_token = old
        kvmem.set_store_budget(0)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"OK: kvmem ({len(tests)} tests)")
