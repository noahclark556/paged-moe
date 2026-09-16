# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Layer-fused prefill must be equivalent to the stock decoder layer.

The whole claim of prefill_fused is that sub-chunking attention inside a layer
and running the MLP once over the concatenation is the *same computation* as
one full-width layer call - only with the expert mass read once instead of once
per attention chunk. These tests pin that: identical hidden states, identical
KV cache, for a real mlx-lm decoder layer.
"""

from __future__ import annotations

import unittest
from unittest import mock

import mlx.core as mx
import mlx.nn as nn


def _tiny_model():
    """A small real mlx-lm model with the standard pre-norm decoder block."""
    from mlx_lm.models import qwen3

    args = qwen3.ModelArgs(
        model_type="qwen3",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=128,
        head_dim=16,
        max_position_embeddings=512,
        rope_theta=10000.0,
        tie_word_embeddings=False,
    )
    model = qwen3.Model(args)
    mx.eval(model.parameters())
    return model


class TestFusedLayerEquivalence(unittest.TestCase):
    def setUp(self):
        from expert_stream import prefill_fused

        prefill_fused.uninstall()
        self.model = _tiny_model()
        self.tokens = mx.array([[i % 97 for i in range(48)]])

    def tearDown(self):
        from expert_stream import prefill_fused

        prefill_fused.uninstall()

    def _run(self, sub_chunk):
        """Forward the prompt with attention sub-chunks of `sub_chunk` rows."""
        from mlx_lm.models import cache as mlx_cache
        from expert_stream import prefill_fused

        prefill_fused.uninstall()
        c = mlx_cache.make_prompt_cache(self.model)
        if sub_chunk is None:
            out = self.model(self.tokens, cache=c)
        else:
            with mock.patch.object(
                prefill_fused.adaptive_prefill,
                "attention_sub_chunk",
                lambda model, kv: sub_chunk,
            ), mock.patch.object(prefill_fused.config, "FUSED_PREFILL", True):
                self.assertTrue(prefill_fused.install(self.model))
                out = self.model(self.tokens, cache=c)
        mx.eval(out)
        return out, c

    def test_matches_stock_layer(self):
        ref, ref_cache = self._run(None)
        for sub in (8, 16, 17):
            got, got_cache = self._run(sub)
            self.assertEqual(got.shape, ref.shape)
            err = float(mx.max(mx.abs(got - ref)))
            self.assertLess(err, 2e-3, f"sub_chunk={sub} diverged by {err}")
            for a, b in zip(ref_cache, got_cache):
                self.assertEqual(a.offset, b.offset)
                self.assertLess(
                    float(mx.max(mx.abs(a.keys[..., : a.offset, :]
                                       - b.keys[..., : b.offset, :]))),
                    1e-4,
                    f"sub_chunk={sub}: KV cache diverged",
                )

    def test_decode_after_fused_prefill_is_unaffected(self):
        """Prefill must leave the cache in the state decode expects."""
        ref, ref_cache = self._run(None)
        got, got_cache = self._run(8)
        nxt = mx.array([[7]])
        a = self.model(nxt, cache=ref_cache)
        b = self.model(nxt, cache=got_cache)
        mx.eval(a, b)
        self.assertLess(float(mx.max(mx.abs(a - b))), 2e-3)

    def test_single_token_is_untouched(self):
        """Decode goes through the stock path with no plan building at all."""
        from mlx_lm.models import cache as mlx_cache
        from expert_stream import prefill_fused

        with mock.patch.object(prefill_fused.config, "FUSED_PREFILL", True):
            prefill_fused.install(self.model)
        c = mlx_cache.make_prompt_cache(self.model)
        with mock.patch.object(
            prefill_fused, "attention_plan", side_effect=AssertionError("planned")
        ):
            out = self.model(mx.array([[5]]), cache=c)
        mx.eval(out)
        self.assertEqual(out.shape[1], 1)


class TestArrayMaskEquivalence(unittest.TestCase):
    """DSA models pass a materialized [L, S] mask, not the "causal" string.

    Sub-chunk k must see exactly rows [start:end] of that mask against the
    keys the cache holds once it is appended - a wrong slice here is a silent
    attention bug, not a crash.
    """

    def setUp(self):
        from expert_stream import prefill_fused

        prefill_fused.uninstall()
        self.model = _tiny_model()
        self.layer = self.model.model.layers[0]

    def tearDown(self):
        from expert_stream import prefill_fused

        prefill_fused.uninstall()

    def _forward(self, sub_chunk):
        from mlx_lm.models import cache as mlx_cache
        from mlx_lm.models.base import create_attention_mask
        from expert_stream import prefill_fused

        prefill_fused.uninstall()
        mx.random.seed(0)
        x = mx.random.normal((1, 40, 64))
        c = [mlx_cache.KVCache()]
        mask = create_attention_mask(x, c[0], return_array=True)
        if sub_chunk is not None:
            with mock.patch.object(
                prefill_fused.adaptive_prefill,
                "attention_sub_chunk",
                lambda model, kv: sub_chunk,
            ), mock.patch.object(prefill_fused.config, "FUSED_PREFILL", True):
                prefill_fused.install(self.model)
                out = self.layer(x, mask, c[0])
        else:
            out = self.layer(x, mask, c[0])
        mx.eval(out)
        return out, c[0]

    def test_array_mask_slices_correctly(self):
        ref, ref_cache = self._forward(None)
        for sub in (7, 13, 20):
            got, got_cache = self._forward(sub)
            err = float(mx.max(mx.abs(got - ref)))
            self.assertLess(err, 2e-3, f"sub_chunk={sub} diverged by {err}")
            self.assertEqual(ref_cache.offset, got_cache.offset)


class TestAttentionPlan(unittest.TestCase):
    def test_covers_every_token_exactly_once(self):
        from types import SimpleNamespace
        from expert_stream import prefill_fused

        model = SimpleNamespace(
            args=SimpleNamespace(model_type="deepseek_v32", num_attention_heads=128)
        )
        with mock.patch.object(
            prefill_fused.adaptive_prefill.config, "ADAPTIVE_PREFILL_MODE", "dsa"
        ), mock.patch.object(
            prefill_fused.adaptive_prefill.config, "ADAPTIVE_PREFILL_MAX", 4096
        ):
            plan = prefill_fused.attention_plan(model, 16_000, 0)
        self.assertEqual(plan[0][0], 0)
        self.assertEqual(plan[-1][1], 16_000)
        for (_, prev_end), (start, _) in zip(plan, plan[1:]):
            self.assertEqual(prev_end, start)

    def test_sub_chunks_shrink_as_keys_grow(self):
        from types import SimpleNamespace
        from expert_stream import prefill_fused

        model = SimpleNamespace(
            args=SimpleNamespace(model_type="deepseek_v32", num_attention_heads=128)
        )
        with mock.patch.object(
            prefill_fused.adaptive_prefill.config, "ADAPTIVE_PREFILL_MODE", "dsa"
        ), mock.patch.object(
            prefill_fused.adaptive_prefill.config, "ADAPTIVE_PREFILL_MAX", 1 << 20
        ):
            plan = prefill_fused.attention_plan(model, 200_000, 0)
        widths = [e - s for s, e in plan]
        self.assertGreater(widths[0], widths[-1])


class TestMaskSlicing(unittest.TestCase):
    def test_2d_mask_slice(self):
        from expert_stream.prefill_fused import _slice_mask

        m = mx.arange(6 * 10).reshape(6, 10)
        got = _slice_mask(m, 2, 4, 8)
        self.assertEqual(got.shape, (2, 8))
        self.assertTrue(mx.array_equal(got, m[2:4, :8]))

    def test_4d_mask_slice(self):
        from expert_stream.prefill_fused import _slice_mask

        m = mx.zeros((1, 1, 6, 10))
        self.assertEqual(_slice_mask(m, 0, 3, 5).shape, (1, 1, 3, 5))

    def test_string_and_none_pass_through(self):
        from expert_stream.prefill_fused import _slice_mask

        self.assertEqual(_slice_mask("causal", 0, 4, 4), "causal")
        self.assertIsNone(_slice_mask(None, 0, 4, 4))


if __name__ == "__main__":
    unittest.main()
