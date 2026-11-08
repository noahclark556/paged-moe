# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DeepSeek-V2 latent cache matches mlx-lm's expanded-head attention."""

from __future__ import annotations

import unittest

from mlx_lm.models.cache import KVCache
from mlx_lm.models.deepseek_v2 import DeepseekV2Attention, ModelArgs

# Captured before mla_v2.install replaces the class method.
_STOCK_CALL = DeepseekV2Attention.__call__


def _tiny_args() -> ModelArgs:
    return ModelArgs(
        hidden_size=64,
        intermediate_size=64,
        vocab_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        max_position_embeddings=128,
        rope_scaling={"factor": 1.0},
    )


class TestLatentCache(unittest.TestCase):
    def test_prefill_and_decode_match_stock(self):
        import mlx.core as mx

        from pagedmoe.mla_v2 import latent_forward

        attn = DeepseekV2Attention(_tiny_args())
        stock_cache = KVCache()
        latent_cache = KVCache()
        mx.random.seed(0)
        prefill = mx.random.normal((1, 5, 64))
        stock = _STOCK_CALL(attn, prefill, None, stock_cache)
        got = latent_forward(attn, prefill, None, latent_cache)
        mx.eval(stock, got)
        err = float(mx.max(mx.abs(stock - got)))
        self.assertLess(err, 2e-3, f"prefill diverged by {err}")
        self.assertEqual(latent_cache.keys.shape[1], 1)
        self.assertEqual(stock_cache.keys.shape[1], attn.num_heads)
        self.assertLess(latent_cache.keys.nbytes, stock_cache.keys.nbytes)

        step = mx.random.normal((1, 1, 64))
        stock2 = _STOCK_CALL(attn, step, None, stock_cache)
        got2 = latent_forward(attn, step, None, latent_cache)
        mx.eval(stock2, got2)
        err2 = float(mx.max(mx.abs(stock2 - got2)))
        self.assertLess(err2, 2e-3, f"decode diverged by {err2}")
        self.assertEqual(latent_cache.offset, stock_cache.offset)

    def _decode_steps_match(self, attn, steps=4):
        import mlx.core as mx

        from pagedmoe.mla_v2 import latent_forward

        stock_cache = KVCache()
        latent_cache = KVCache()
        mx.random.seed(1)
        prefill = mx.random.normal((1, 7, attn.hidden_size))
        _STOCK_CALL(attn, prefill, None, stock_cache)
        latent_forward(attn, prefill, None, latent_cache)
        for i in range(steps):
            step = mx.random.normal((1, 1, attn.hidden_size))
            stock = _STOCK_CALL(attn, step, None, stock_cache)
            got = latent_forward(attn, step, None, latent_cache)
            mx.eval(stock, got)
            err = float(mx.max(mx.abs(stock - got)))
            self.assertLess(err, 2e-3, f"decode step {i} diverged by {err}")

    def test_padded_causal_prefill_matches_stock(self):
        import mlx.core as mx

        from pagedmoe import mla_v2

        self.assertEqual(mla_v2._fused_pad_dim(16, 8), 64)
        self.assertEqual(mla_v2._fused_pad_dim(192, 128), 256)
        self.assertIsNone(mla_v2._fused_pad_dim(128, 128))
        attn = DeepseekV2Attention(_tiny_args())
        mx.random.seed(4)
        first = mx.random.normal((1, 6, 64))
        second = mx.random.normal((1, 9, 64))
        stock_cache, got_cache = KVCache(), KVCache()
        for x in (first, second):
            stock = _STOCK_CALL(attn, x, "causal", stock_cache)
            got = mla_v2.latent_forward(attn, x, "causal", got_cache)
            mx.eval(stock, got)
            err = float(mx.max(mx.abs(stock - got)))
            self.assertLess(err, 2e-3, f"padded prefill diverged by {err}")

    def test_absorbed_decode_matches_stock(self):
        self._decode_steps_match(DeepseekV2Attention(_tiny_args()))

    def test_absorbed_decode_matches_stock_quantized_kv_b(self):
        import mlx.nn as nn

        args = _tiny_args()
        args.kv_lora_rank = 64
        attn = DeepseekV2Attention(args)
        attn.kv_b_proj = nn.QuantizedLinear.from_linear(
            attn.kv_b_proj, group_size=64, bits=4
        )
        self._decode_steps_match(attn)


class TestGroupedGateInput(unittest.TestCase):
    def test_flat_rows_route_like_the_block(self):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_lm.models.deepseek_v2 import MoEGate

        from pagedmoe.streaming import _gate_input

        args = _tiny_args()
        args.n_routed_experts = 16
        args.num_experts_per_tok = 4
        args.n_group = 4
        args.topk_group = 2
        args.topk_method = "group_limited_greedy"
        gate = MoEGate(args)
        mx.random.seed(2)
        x = mx.random.normal((3, 64))
        with self.assertRaises(ValueError):
            mx.eval(gate(x))
        inds, _ = gate(_gate_input(gate, x))
        block_inds, _ = gate(x[None])
        self.assertTrue(mx.array_equal(inds, block_inds).item())
        self.assertEqual(inds.reshape(-1, 4).shape, (3, 4))

        plain = nn.Linear(64, 16)
        self.assertIs(_gate_input(plain, x), x)


if __name__ == "__main__":
    unittest.main()
