# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for prefill chunk sizing (no GPU required for the formula)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

MAX_BUF = 30_150_672_384


class TestDsaMaxChunk(unittest.TestCase):
    def test_grows_when_kv_is_short(self):
        from expert_stream.adaptive_prefill import dsa_max_chunk

        early = dsa_max_chunk(0, n_heads=128, max_buf=MAX_BUF, safety=0.75, c_max=8192)
        late = dsa_max_chunk(
            65_536, n_heads=128, max_buf=MAX_BUF, safety=0.75, c_max=8192
        )
        self.assertGreaterEqual(early, 8000)
        self.assertLessEqual(late, 2048)
        self.assertGreater(early, late)

    def test_respects_buffer_with_no_floor_override(self):
        """The bound is an allocation limit; nothing may raise the result past it.

        ADAPTIVE_PREFILL_MIN used to be applied after the bound, so at long KV
        the floor won and returned a chunk that could not be allocated.
        """
        from expert_stream.adaptive_prefill import dsa_max_chunk

        budget = 0.75 * MAX_BUF
        for kv in (0, 1024, 8192, 16384, 32768, 65536, 131_072):
            c = dsa_max_chunk(
                kv,
                n_heads=128,
                dtype_bytes=2,
                max_buf=MAX_BUF,
                safety=0.75,
                c_max=1_000_000,
            )
            self.assertLessEqual(128 * c * (kv + c) * 2, budget + 1)

    def test_sub_chunk_shrinks_with_kv(self):
        from expert_stream import adaptive_prefill as ap

        model = SimpleNamespace(
            args=SimpleNamespace(model_type="deepseek_v32", num_attention_heads=128)
        )
        with mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MODE", "auto"):
            wide = ap.attention_sub_chunk(model, 0)
            narrow = ap.attention_sub_chunk(model, 65_536)
        self.assertGreater(wide, narrow)

    def test_fused_models_need_no_attention_split(self):
        from expert_stream import adaptive_prefill as ap

        model = SimpleNamespace(
            args=SimpleNamespace(model_type="qwen3_moe", num_attention_heads=64),
            layers=[],
        )
        with mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MODE", "auto"), \
                mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MAX", 32768):
            self.assertEqual(ap.attention_sub_chunk(model, 65_536), 32768)


class TestLayerDerivedSizing(unittest.TestCase):
    """Sub-chunks are sized from the layer, not from an owning-model handle.

    A layer knows its own head count and whether it has an indexer, so two
    models sharing a decoder class cannot be sized with each other's config.
    """

    def _layer(self, *, indexer, heads):
        attn = SimpleNamespace(num_heads=heads)
        if indexer:
            attn.indexer = object()
        return SimpleNamespace(self_attn=attn)

    def test_indexer_layer_gets_the_score_bound(self):
        from expert_stream import adaptive_prefill as ap

        with mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MODE", "auto"), \
                mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MAX", 32768):
            n = ap.attention_sub_chunk(self._layer(indexer=True, heads=128), 65_536)
        self.assertLess(n, 32768)
        self.assertGreater(n, 0)

    def test_plain_attention_layer_is_not_split(self):
        from expert_stream import adaptive_prefill as ap

        with mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MODE", "auto"), \
                mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MAX", 32768):
            n = ap.attention_sub_chunk(self._layer(indexer=False, heads=64), 65_536)
        self.assertEqual(n, 32768)

    def test_head_count_comes_from_the_layer(self):
        from expert_stream import adaptive_prefill as ap

        # ATTN_SUB_CHUNK off, so only the score bound is in play and the head
        # count is the only thing that can move the answer.
        with mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MODE", "dsa"), \
                mock.patch.object(ap.config, "ATTN_SUB_CHUNK", 0), \
                mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MAX", 1 << 20):
            wide = ap.attention_sub_chunk(self._layer(indexer=True, heads=16), 8192)
            narrow = ap.attention_sub_chunk(self._layer(indexer=True, heads=128), 8192)
        self.assertGreater(wide, narrow)

    def test_perf_target_caps_the_metal_bound(self):
        """A cold KV fits far more than ATTN_SUB_CHUNK; we must not take it.

        Sizing to the Metal bound computes the sub-chunk's whole causal upper
        triangle for nothing, so the target is the operative cap whenever it is
        the smaller of the two.
        """
        from expert_stream import adaptive_prefill as ap

        layer = self._layer(indexer=True, heads=128)
        with mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MODE", "dsa"), \
                mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MAX", 1 << 20):
            with mock.patch.object(ap.config, "ATTN_SUB_CHUNK", 0):
                bound = ap.attention_sub_chunk(layer, 0)
            with mock.patch.object(ap.config, "ATTN_SUB_CHUNK", 2048):
                capped = ap.attention_sub_chunk(layer, 0)
                # ...and the bound still wins once KV makes it the tighter one.
                long_kv = ap.attention_sub_chunk(layer, 1 << 18)
        self.assertGreater(bound, 2048)
        self.assertEqual(capped, 2048)
        self.assertLess(long_kv, 2048)

    def test_mode_off_never_splits(self):
        from expert_stream import adaptive_prefill as ap

        with mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MODE", "off"), \
                mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MAX", 32768):
            n = ap.attention_sub_chunk(self._layer(indexer=True, heads=128), 65_536)
        self.assertEqual(n, 32768)

    def test_real_deepseek_layer_is_detected(self):
        """Guards against mlx-lm renaming the indexer or the head attribute."""
        from mlx_lm.models import deepseek_v32

        self.assertIn("indexer", deepseek_v32.DeepseekV32Attention.__init__.__code__.co_names)
        self.assertIn("num_heads", deepseek_v32.DeepseekV32Attention.__init__.__code__.co_names)


class TestCacheOffset(unittest.TestCase):
    def test_reads_cachelist(self):
        """CacheList is not a list and has no .offset - it proxies by index.

        Reading the attribute alone silently reported 0 for every DeepSeek
        prompt, which made every chunk size as if it were the first.
        """
        from expert_stream.adaptive_prefill import cache_offset

        class Inner:
            offset = 4096

        class CacheList:
            def __init__(self, *caches):
                self.caches = caches

            def __getitem__(self, i):
                return self.caches[i]

        self.assertEqual(cache_offset([CacheList(Inner(), Inner())]), 4096)

    def test_reads_plain_kv_cache(self):
        from expert_stream.adaptive_prefill import cache_offset

        self.assertEqual(
            cache_offset([SimpleNamespace(offset=128), SimpleNamespace(offset=128)]),
            128,
        )

    def test_empty_and_unknown(self):
        from expert_stream.adaptive_prefill import cache_offset

        self.assertEqual(cache_offset(None), 0)
        self.assertEqual(cache_offset([]), 0)
        self.assertEqual(cache_offset([object()]), 0)


class TestNextChunk(unittest.TestCase):
    def test_fused_prefill_keeps_the_step_large(self):
        """With attention sub-chunked per layer, the step is an activation
        bound - it must not shrink with KV, or the expert mass gets re-read."""
        from expert_stream import adaptive_prefill as ap
        from expert_stream import prefill_fused

        model = SimpleNamespace(
            args=SimpleNamespace(model_type="deepseek_v32", num_attention_heads=128)
        )
        with mock.patch.object(ap.config, "ADAPTIVE_PREFILL", True), \
                mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MODE", "auto"), \
                mock.patch.object(prefill_fused, "_active", True):
            self.assertEqual(ap.next_chunk(60_000, 16_000, model, prefill_cap=16384), 16_000)

    def test_without_fused_prefill_dsa_shrinks(self):
        from expert_stream import adaptive_prefill as ap
        from expert_stream import prefill_fused

        model = SimpleNamespace(
            args=SimpleNamespace(model_type="deepseek_v32", num_attention_heads=128)
        )
        with mock.patch.object(ap.config, "ADAPTIVE_PREFILL", True), \
                mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MODE", "auto"), \
                mock.patch.object(prefill_fused, "_active", False):
            n = ap.next_chunk(65_536, 16_000, model, prefill_cap=16384)
        self.assertLess(n, 2048)

    def test_next_chunk_fused_uses_cap(self):
        from expert_stream import adaptive_prefill as ap

        model = SimpleNamespace(
            args=SimpleNamespace(model_type="qwen3_moe", num_attention_heads=64),
            layers=[],
        )
        with mock.patch.object(ap.config, "ADAPTIVE_PREFILL", True), \
                mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MODE", "fused"), \
                mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MAX", 32768):
            self.assertEqual(ap.next_chunk(0, 20000, model, prefill_cap=32768), 20000)


class TestModeResolve(unittest.TestCase):
    def test_deepseek_v32_is_dsa(self):
        from expert_stream import adaptive_prefill as ap

        model = SimpleNamespace(args=SimpleNamespace(model_type="deepseek_v32"))
        self.assertTrue(ap.is_dsa_model(model))
        with mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MODE", "auto"):
            self.assertEqual(ap.resolve_mode(model), "dsa")

    def test_plain_deepseek_v3_is_fused(self):
        """No indexer, fused SDPA - sizing it as DSA only made it slower."""
        from expert_stream import adaptive_prefill as ap

        model = SimpleNamespace(
            args=SimpleNamespace(model_type="deepseek_v3"), layers=[]
        )
        self.assertFalse(ap.is_dsa_model(model))

    def test_indexer_heuristic(self):
        from expert_stream import adaptive_prefill as ap

        attn = SimpleNamespace(indexer=object())
        model = SimpleNamespace(
            args=SimpleNamespace(model_type="mystery_moe"),
            layers=[SimpleNamespace(self_attn=attn)],
        )
        self.assertTrue(ap.is_dsa_model(model))

    def test_qwen_is_fused(self):
        from expert_stream import adaptive_prefill as ap

        model = SimpleNamespace(args=SimpleNamespace(model_type="qwen3_moe"), layers=[])
        self.assertFalse(ap.is_dsa_model(model))
        with mock.patch.object(ap.config, "ADAPTIVE_PREFILL_MODE", "auto"):
            self.assertEqual(ap.resolve_mode(model), "fused")


if __name__ == "__main__":
    unittest.main()
