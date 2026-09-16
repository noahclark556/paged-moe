# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prefix snapshot forking: one prefill, then deepcopy+trim - no second pass."""

from __future__ import annotations

import unittest
from unittest import mock

import mlx.core as mx
from mlx_lm.models.cache import CacheList, KVCache, can_trim_prompt_cache, trim_prompt_cache


def _fill(cache: KVCache, n: int, dim: int = 4) -> None:
    keys = mx.zeros((1, 1, n, dim))
    values = mx.zeros((1, 1, n, dim))
    cache.update_and_fetch(keys, values)
    mx.eval(cache.keys, cache.values)


class TestMaterializeTrimmed(unittest.TestCase):
    def test_round_trip_drops_suffix_bytes(self):
        from expert_stream.server import _materialize_trimmed_prefix

        c = KVCache()
        _fill(c, 128)
        before = c.nbytes
        trim_prompt_cache([c], 48)  # leave offset=80
        self.assertEqual(c.offset, 80)
        # trim alone does not shrink the arrays
        self.assertEqual(c.nbytes, before)
        _materialize_trimmed_prefix([c])
        self.assertEqual(c.offset, 80)
        self.assertLess(c.nbytes, before)
        self.assertEqual(c.keys.shape[2], 80)

    def test_cachelist_deepseek_shape(self):
        from expert_stream.server import _materialize_trimmed_prefix

        layer = CacheList(KVCache(), KVCache())
        for child in layer.caches:
            _fill(child, 64)
        self.assertTrue(can_trim_prompt_cache([layer]))
        trim_prompt_cache([layer], 16)
        before = layer.nbytes
        _materialize_trimmed_prefix([layer])
        self.assertEqual(layer.caches[0].offset, 48)
        self.assertLess(layer.nbytes, before)


class TestForkPrefixSnapshots(unittest.TestCase):
    def test_forks_at_bound_without_touching_live_cache(self):
        from expert_stream.server import _fork_prefix_snapshots

        live = KVCache()
        _fill(live, 100)
        store = mock.Mock()
        store.insert_cache = mock.Mock()
        n = _fork_prefix_snapshots(
            store,
            "m",
            list(range(100)),
            [live],
            [60, 90],
            cached=0,
            processed=100,
            budget=1 << 62,
        )
        self.assertEqual(n, 2)
        self.assertEqual(live.offset, 100)  # live untouched
        self.assertEqual(store.insert_cache.call_count, 2)
        # Longest bound is inserted first so the shorter is not popped.
        args0 = store.insert_cache.call_args_list[0][0]
        self.assertEqual(len(args0[1]), 90)
        self.assertEqual(args0[2][0].offset, 90)
        args1 = store.insert_cache.call_args_list[1][0]
        self.assertEqual(len(args1[1]), 60)
        self.assertEqual(args1[2][0].offset, 60)

    def test_skips_when_over_budget(self):
        from expert_stream.server import _fork_prefix_snapshots

        live = KVCache()
        _fill(live, 64)
        store = mock.Mock()
        store.insert_cache = mock.Mock()
        n = _fork_prefix_snapshots(
            store,
            "m",
            list(range(64)),
            [live],
            [32],
            cached=0,
            processed=64,
            budget=1,  # anything * 2 > 1
        )
        self.assertEqual(n, 0)
        store.insert_cache.assert_not_called()


class TestTrimmableGate(unittest.TestCase):
    def test_deepseek_cache_is_trimmable(self):
        cache = [CacheList(KVCache(), KVCache()) for _ in range(3)]
        for layer in cache:
            for child in layer.caches:
                _fill(child, 8)
        self.assertTrue(can_trim_prompt_cache(cache))


if __name__ == "__main__":
    unittest.main()
