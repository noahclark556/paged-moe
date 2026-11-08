# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Recurrent-state checkpoints (hybrid_ckpt.py) on a tiny qwen3_next.

The claim is that a longer stored entry rewound to a checkpoint is the cache a
prefill of the checkpoint's tokens leaves behind, bit for bit, so a request
that branches back one message resumes instead of prefilling cold. Checked
against a fresh prefill, with and without an indexer-style side cache, plus
the ways it has to refuse: trimmable caches, offset mismatch, an entry from a
different lineage, and a registry over its bounds.

Run:
    python tests/test_hybrid_ckpt.py
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import (  # noqa: E402
    KVCache,
    LRUPromptCache,
    QuantizedKVCache,
    make_prompt_cache,
)

from tests.test_lookup_hybrid import ARGS, _Side, build  # noqa: E402

from pagedmoe import config as es_config  # noqa: E402
from pagedmoe import hybrid_ckpt, kvmem  # noqa: E402

KEY = ("tiny", None, None)


def _prefill(model, cache, toks):
    mx.eval(model(mx.array([toks]), cache=cache))


def _attach_side(cache, n):
    for c in cache:
        if c.is_trimmable():
            c.indexer = _Side()
            c.indexer.keys = mx.arange(n * 4, dtype=mx.float32).reshape(1, n, 4)


def _grow_side(cache, n_old, n_new):
    for c in cache:
        if c.is_trimmable():
            extra = mx.full((1, n_new - n_old, 4), -1.0)
            c.indexer.keys = mx.concatenate([c.indexer.keys, extra], axis=1)


def check_restore_exact(model, side: bool):
    hybrid_ckpt.clear()
    rng = np.random.default_rng(4)
    a = rng.integers(0, ARGS["vocab_size"], size=60).tolist()
    p = 36
    b = a[:p] + rng.integers(0, ARGS["vocab_size"], size=10).tolist()

    live = make_prompt_cache(model)
    _prefill(model, live, a[:p])
    if side:
        _attach_side(live, p)
    assert hybrid_ckpt.capture(KEY, a[:p], live), "capture refused a hybrid cache"
    _prefill(model, live, a[p:])
    if side:
        _grow_side(live, p, len(a))
    store = LRUPromptCache(max_size=4)
    store.insert_cache(KEY, list(a), live, cache_type="assistant")

    # Without the checkpoint the store has nothing for b: its only key runs
    # past the branch point and cannot trim.
    cache0, rest0 = store.fetch_nearest_cache(KEY, list(b))
    assert cache0 is None and rest0 == b, "store should miss b on its own"

    hit = hybrid_ckpt.resume(store, KEY, b, have=0)
    assert hit is not None, "checkpoint did not serve the branch"
    cache, rest, n = hit
    assert n == p and rest == b[p:], f"resumed at {n}, rest {len(rest)}"
    for c in cache:
        if c.is_trimmable():
            assert c.offset == p, f"attention offset {c.offset} != {p}"
            if side:
                keys = c.indexer.keys
                assert keys.shape[1] == p, f"side cache kept {keys.shape[1]}"
                want = mx.arange(p * 4, dtype=mx.float32).reshape(1, p, 4)
                assert bool(mx.all(keys == want)), "side cache content moved"

    # The stored entry is untouched.
    entry = store._trie.get(KEY, list(a))
    for c in entry.prompt_cache:
        if c.is_trimmable():
            assert c.offset == len(a), "restore trimmed the stored entry itself"

    ref = make_prompt_cache(model)
    _prefill(model, ref, b[:p])
    la = model(mx.array([rest]), cache=cache)
    lb = model(mx.array([b[p:]]), cache=ref)
    d = float(mx.abs(la - lb).max())
    assert d == 0.0, f"restored cache differs from a fresh prefill by {d}"
    tag = " with a side cache" if side else ""
    print(f"OK: checkpoint restore is bit-identical to a fresh prefill{tag}")


def check_refusals(model):
    hybrid_ckpt.clear()
    rng = np.random.default_rng(9)
    a = rng.integers(0, ARGS["vocab_size"], size=40).tolist()

    trim = [KVCache() for _ in range(2)]
    for c in trim:
        c.update_and_fetch(mx.zeros((1, 2, 8, 16)), mx.zeros((1, 2, 8, 16)))
    assert not hybrid_ckpt.capture(KEY, a[:8], trim), "captured a trimmable cache"

    live = make_prompt_cache(model)
    _prefill(model, live, a[:20])
    assert not hybrid_ckpt.capture(KEY, a[:19], live), "captured at the wrong offset"
    assert hybrid_ckpt.count() == 0

    assert hybrid_ckpt.capture(KEY, a[:20], live)
    _prefill(model, live, a[20:])
    store = LRUPromptCache(max_size=4)
    # A stored entry from another lineage that shares only 10 tokens.
    other = a[:10] + [(t + 1) % ARGS["vocab_size"] for t in a[10:]]
    store.insert_cache(KEY, other, live)
    q = a[:20] + [1, 2, 3]
    assert hybrid_ckpt.resume(store, KEY, q, have=0) is None, "served another lineage"
    # Never beat what the store already has, and always leave a token for decode.
    store.insert_cache(KEY, list(a), make_prompt_cache(model))
    assert hybrid_ckpt.resume(store, KEY, q, have=20) is None
    assert hybrid_ckpt.resume(store, KEY, a[:20], have=0) is None
    # Wrong layer layout (an entry that is not this model's cache).
    store2 = LRUPromptCache(max_size=4)
    store2.insert_cache(KEY, list(a), trim)
    assert hybrid_ckpt.resume(store2, KEY, q, have=0) is None
    es_config.HYBRID_CKPT = False
    try:
        assert not hybrid_ckpt.capture(KEY, a[:20], make_prompt_cache(model))
    finally:
        es_config.HYBRID_CKPT = True
    print("OK: refuses trimmable caches, bad offsets, other lineages, wrong layouts")


def check_bounds(model):
    hybrid_ckpt.clear()
    rng = np.random.default_rng(1)
    a = rng.integers(0, ARGS["vocab_size"], size=80).tolist()
    live = make_prompt_cache(model)
    old_max = es_config.HYBRID_CKPT_MAX
    es_config.HYBRID_CKPT_MAX = 3
    try:
        for i in range(1, 7):
            _prefill(model, live, a[(i - 1) * 10 : i * 10])
            assert hybrid_ckpt.capture(KEY, a[: i * 10], live)
        assert hybrid_ckpt.count() == 3, f"held {hybrid_ckpt.count()}"
        assert [c.length for c in hybrid_ckpt._REG] == [20, 50, 60], [c.length for c in hybrid_ckpt._REG]
        # Recapturing the same boundary replaces rather than duplicates.
        assert hybrid_ckpt.capture(KEY, a[:60], live)
        assert hybrid_ckpt.count() == 3
    finally:
        es_config.HYBRID_CKPT_MAX = old_max

    per = hybrid_ckpt._REG[-1].nbytes
    old_mb = es_config.HYBRID_CKPT_MB
    es_config.HYBRID_CKPT_MB = (per * 1.5) / (1 << 20)
    try:
        hybrid_ckpt.capture(KEY, a[:60], live)
        assert hybrid_ckpt.count() == 1, "byte cap not applied"
        es_config.HYBRID_CKPT_MB = 0.0
        hybrid_ckpt.capture(KEY, a[:60], live)
        assert hybrid_ckpt.count() == 1, "newest checkpoint must survive the cap"
    finally:
        es_config.HYBRID_CKPT_MB = old_mb
    print(f"OK: count and byte caps hold ({per} B per checkpoint here)")


def check_eviction_order():
    """An agent run leaves two checkpoints a round: the turn's resume point and
    the think pass the next turn drops. The dead branches go first, then the
    live conversation thins evenly, so a follow-up that re-renders from early
    in the run still finds a checkpoint near where it diverges."""
    hybrid_ckpt.clear()
    old_max = es_config.HYBRID_CKPT_MAX
    es_config.HYBRID_CKPT_MAX = 6
    main = list(range(1000))
    try:
        for p in range(100, 1000, 100):
            hybrid_ckpt._REG.append(hybrid_ckpt.Checkpoint(KEY, tuple(main[:p]), [], 1))
            hybrid_ckpt._bound()
            think = tuple(main[:p]) + (-1,) * 40
            hybrid_ckpt._REG.append(hybrid_ckpt.Checkpoint(KEY, think, [], 1))
            hybrid_ckpt._bound()
        lens = [c.length for c in hybrid_ckpt._REG]
        assert hybrid_ckpt._REG[-1].tokens[-1] == -1 and lens[-2] == 900, lens
        live = [c.length for c in hybrid_ckpt._REG[:-1]]
        assert all(c.tokens == tuple(main[: c.length]) for c in hybrid_ckpt._REG[:-1]), lens
        assert live[0] <= 300, f"nothing left early in the run: {lens}"
        assert max(b - a for a, b in zip([0] + live, live)) <= 300, f"uneven: {lens}"
    finally:
        es_config.HYBRID_CKPT_MAX = old_max
        hybrid_ckpt.clear()
    print(f"OK: dead branches evicted first, live checkpoints spread ({lens})")


def check_store_budget(model):
    hybrid_ckpt.clear()
    rng = np.random.default_rng(2)
    a = rng.integers(0, ARGS["vocab_size"], size=30).tolist()
    store = LRUPromptCache(max_size=10)
    caches = []
    for i in range(3):
        c = make_prompt_cache(model)
        _prefill(model, c, a[: 10 + i * 10])
        caches.append(c)
        store.insert_cache(KEY, a[: 10 + i * 10], c)
    total = store.nbytes
    old = kvmem._store_budget
    try:
        kvmem._store_budget = total
        assert kvmem.bound_store(store) == 0
        live = make_prompt_cache(model)
        _prefill(model, live, a[:12])
        hybrid_ckpt.capture(KEY, a[:12], live)
        assert hybrid_ckpt.nbytes() > 0
        freed = kvmem.bound_store(store)
        assert freed > 0, "checkpoint bytes were not charged to the store budget"
        assert store.nbytes + hybrid_ckpt.nbytes() <= total
    finally:
        kvmem._store_budget = old
        hybrid_ckpt.clear()
    print("OK: checkpoint bytes count against the prompt-store budget")


def check_quantized_entry():
    """Decode stores an 8-bit cache. Its side caches have to survive that, and
    a checkpoint has to restore onto it even when another extending entry has
    a layout it cannot use."""
    from mlx_lm.models import qwen3_next

    mx.random.seed(0)
    model = qwen3_next.Model(qwen3_next.ModelArgs(**{**ARGS, "head_dim": 32}))
    mx.eval(model.parameters())
    hybrid_ckpt.clear()
    rng = np.random.default_rng(5)
    a = rng.integers(0, ARGS["vocab_size"], size=60).tolist()
    p = 36
    b = a[:p] + rng.integers(0, ARGS["vocab_size"], size=10).tolist()

    live = make_prompt_cache(model)
    _prefill(model, live, a[:p])
    _attach_side(live, p)
    assert hybrid_ckpt.capture(KEY, a[:p], live), "capture refused a hybrid cache"
    _prefill(model, live, a[p:])
    _grow_side(live, p, len(a))
    kvmem.requantize(live, 0, 32, 8)
    for c in live:
        if c.is_trimmable():
            assert isinstance(c, QuantizedKVCache), "requantize left fp16 KV"
            assert isinstance(getattr(c, "indexer", None), _Side), "requantize dropped the side cache"

    # A longer entry from the same prefix without the side cache: restore must
    # refuse it and move on to the one that fits.
    bare = make_prompt_cache(model)
    _prefill(model, bare, a + [1, 2])
    store = LRUPromptCache(max_size=4)
    store.insert_cache(KEY, a + [1, 2], bare, cache_type="assistant")
    store.insert_cache(KEY, list(a), live, cache_type="assistant")

    hit = hybrid_ckpt.resume(store, KEY, b, have=0)
    assert hit is not None, "checkpoint did not restore onto the quantized entry"
    cache, rest, n = hit
    assert n == p and rest == b[p:], f"resumed at {n}, rest {len(rest)}"
    for c in cache:
        if c.is_trimmable():
            assert c.offset == p, f"attention offset {c.offset} != {p}"
            want = mx.arange(p * 4, dtype=mx.float32).reshape(1, p, 4)
            assert bool(mx.all(c.indexer.keys == want)), "side cache content moved"

    kvmem.unquantize(cache)
    for c in cache:
        if c.is_trimmable():
            assert isinstance(getattr(c, "indexer", None), _Side), "unquantize dropped the side cache"
    hybrid_ckpt.clear()
    print("OK: side caches survive KV quantization and the checkpoint restores onto it")


def main():
    old_min = es_config.HYBRID_CKPT_MIN_TOKENS
    es_config.HYBRID_CKPT_MIN_TOKENS = 1
    try:
        model = build()
        check_restore_exact(model, side=False)
        check_restore_exact(model, side=True)
        check_quantized_entry()
        check_refusals(model)
        check_bounds(model)
        check_eviction_order()
        check_store_budget(model)
    finally:
        es_config.HYBRID_CKPT_MIN_TOKENS = old_min
        hybrid_ckpt.clear()
    print("\nOK: hybrid checkpoint tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
