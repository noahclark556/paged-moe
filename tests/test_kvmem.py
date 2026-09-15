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

from expert_stream import config, kvmem

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


def test_bound_store_is_inert_until_the_loader_publishes_a_budget():
    """A budget of 0 means "not known yet" - trimming then would throw away
    prefixes on the strength of no information at all."""
    store = LRUPromptCache(max_size=10)
    store.insert_cache("m", [1, 2, 3], _filled(64))
    kvmem.set_store_budget(0)
    assert kvmem.bound_store(store) == 0
    assert len(store) == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"OK: kvmem ({len(tests)} tests)")
