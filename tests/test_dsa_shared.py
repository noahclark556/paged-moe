# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared DSA indexer layers (GLM-5.2) against stock mlx-lm deepseek_v32."""

from __future__ import annotations

import unittest
from unittest import mock

import mlx.core as mx
from mlx.utils import tree_flatten

TYPES = ["full", "shared", "shared", "full", "shared", "shared"]


def _config(**over):
    cfg = dict(
        model_type="glm_moe_dsa",
        vocab_size=128,
        hidden_size=64,
        index_head_dim=16,
        index_n_heads=4,
        index_topk=4,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=len(TYPES),
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        attention_bias=False,
        indexer_types=list(TYPES),
    )
    cfg.update(over)
    return cfg


def _stock(cfg):
    from mlx_lm.utils import _get_classes

    model_cls, args_cls = _get_classes(cfg)
    mx.random.seed(0)
    model = model_cls(args_cls.from_dict(cfg))
    mx.eval(model.parameters())
    return model


def _ours(cfg, stock, classes=None):
    from pagedmoe import dsa_shared

    model_cls, args_cls = classes or dsa_shared.get_model_classes(cfg)
    model = model_cls(args_cls.from_dict(cfg))
    names = {k for k, _ in tree_flatten(model.parameters())}
    weights = [(k, v) for k, v in tree_flatten(stock.parameters()) if k in names]
    model.load_weights(weights, strict=True)
    mx.eval(model.parameters())
    return model


class _Record:
    def __init__(self, inner, box):
        self.inner, self.box = inner, box

    def __call__(self, x, qr, mask, cache=None):
        self.box["topk"] = self.inner(x, qr, mask, cache=cache)
        return self.box["topk"]


class _Echo:
    def __init__(self, box):
        self.box = box

    def __call__(self, x, qr, mask, cache=None):
        if cache is not None:
            empty = mx.zeros((x.shape[0], 1, x.shape[1], 0))
            cache.update_and_fetch(empty, empty)
        return self.box["topk"]


class _Recent:
    """Indexer stand-in: each row's k most recent positions (future ones wrap
    around and the causal mask drops them)."""

    def __init__(self, k):
        self.k = k

    def __call__(self, x, qr, mask, cache=None):
        B, L = x.shape[:2]
        empty = mx.zeros((B, 1, L, 0))
        keys, _ = cache.update_and_fetch(empty, empty)
        S = keys.shape[2]
        if S <= self.k:
            return None
        rows = S - L + mx.arange(L)
        idx = (rows[:, None] - mx.arange(self.k)[None]) % S
        return mx.broadcast_to(idx.astype(mx.uint32), (B, 1, L, self.k))


def _reference(stock):
    """Stock model whose shared layers reuse the last full layer's top-k."""
    box = {}
    for t, layer in zip(TYPES, stock.model.layers):
        attn = layer.self_attn
        attn.indexer = _Record(attn.indexer, box) if t == "full" else _Echo(box)
    return stock


def _steps(model, tokens, decode=(5, 9, 33)):
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    outs = [model(tokens, cache=cache)]
    for t in decode:
        outs.append(model(mx.array([[t]]), cache=cache))
    mx.eval(outs)
    return outs, cache


TOKENS = mx.array([[(i * 7) % 113 for i in range(16)]])


class TestIndexerTypes(unittest.TestCase):
    def test_explicit_list(self):
        from pagedmoe.dsa_shared import indexer_types

        self.assertEqual(indexer_types(_config()), TYPES)

    def test_glm52_freq_offset_matches_shipped_list(self):
        from pagedmoe.dsa_shared import indexer_types

        n = 78
        derived = indexer_types(
            {"num_hidden_layers": n, "index_topk_freq": 4, "index_skip_topk_offset": 3}
        )
        shipped = ["full"] * 3 + (["shared"] * 3 + ["full"]) * 18 + ["shared"] * 3
        self.assertEqual(derived, shipped)

    def test_pattern_string(self):
        from pagedmoe.dsa_shared import indexer_types

        cfg = {"num_hidden_layers": 4, "index_topk_pattern": "FSSF"}
        self.assertEqual(indexer_types(cfg), ["full", "shared", "shared", "full"])

    def test_no_sharing_is_none(self):
        from pagedmoe.dsa_shared import indexer_types

        self.assertIsNone(indexer_types({"num_hidden_layers": 4}))
        self.assertIsNone(indexer_types(_config(indexer_types=["full"] * len(TYPES))))
        self.assertIsNone(indexer_types({"num_hidden_layers": 4, "index_topk_freq": 1}))

    def test_rejects_shared_first_layer_and_short_lists(self):
        from pagedmoe.dsa_shared import indexer_types

        self.assertIsNone(indexer_types(_config(indexer_types=["shared"] + TYPES[1:])))
        self.assertIsNone(indexer_types(_config(indexer_types=TYPES[:3])))


class TestClassSelection(unittest.TestCase):
    def test_non_sharing_configs_follow_kill_switch(self):
        from mlx_lm.utils import _get_classes

        from pagedmoe import dsa_shared

        old = dsa_shared._PREFILL_ATTN
        self.addCleanup(dsa_shared.set_prefill_attention, old)
        for cfg in (
            _config(indexer_types=None),
            _config(model_type="deepseek_v32", indexer_types=None),
        ):
            dsa_shared.set_prefill_attention("stock")
            self.assertEqual(dsa_shared.get_model_classes(cfg), _get_classes(cfg))
            dsa_shared.set_prefill_attention("fused")
            model_cls, args_cls = dsa_shared.get_model_classes(cfg)
            self.assertNotEqual(model_cls, _get_classes(cfg)[0])
            model = model_cls(args_cls.from_dict(cfg))
            for layer in model.model.layers:
                self.assertIsNotNone(layer.self_attn.indexer)

    def test_other_families_untouched(self):
        from mlx_lm.utils import _get_classes

        from pagedmoe.dsa_shared import get_model_classes

        cfg = {"model_type": "qwen3_moe"}
        self.assertEqual(get_model_classes(cfg), _get_classes(cfg))

    def test_sharing_config_gets_patched_classes(self):
        from mlx_lm.utils import _get_classes

        from pagedmoe.dsa_shared import get_model_classes

        cfg = _config()
        self.assertNotEqual(get_model_classes(cfg), _get_classes(cfg))
        self.assertIs(get_model_classes(cfg)[0], get_model_classes(cfg)[0])

    def test_shared_layers_have_no_indexer_weights(self):
        stock = _stock(_config())
        model = _ours(_config(), stock)
        names = [k for k, _ in tree_flatten(model.parameters())]
        for i, t in enumerate(TYPES):
            has = any(k.startswith(f"model.layers.{i}.self_attn.indexer.") for k in names)
            self.assertEqual(has, t == "full", f"layer {i}")


def _stock_math(test):
    from pagedmoe import dsa_shared

    old = dsa_shared._PREFILL_ATTN
    dsa_shared.set_prefill_attention("stock")
    test.addCleanup(dsa_shared.set_prefill_attention, old)


class TestNumerics(unittest.TestCase):
    def setUp(self):
        _stock_math(self)

    def test_all_full_is_bit_identical_to_stock(self):
        from mlx_lm.utils import _get_classes

        from pagedmoe import dsa_shared

        cfg = _config(indexer_types=None)
        stock = _stock(cfg)
        classes = dsa_shared._shared_classes(*_get_classes(cfg))
        model = _ours(cfg, stock, classes)
        ref, _ = _steps(stock, TOKENS)
        got, _ = _steps(model, TOKENS)
        for a, b in zip(ref, got):
            self.assertTrue(mx.array_equal(a, b))

    def test_shared_layers_reuse_last_full_topk(self):
        cfg = _config()
        stock = _stock(cfg)
        model = _ours(cfg, stock)
        ref, _ = _steps(_reference(stock), TOKENS)
        got, _ = _steps(model, TOKENS)
        for a, b in zip(ref, got):
            self.assertTrue(mx.array_equal(a, b))

    def test_sharing_changes_output(self):
        cfg = _config()
        stock = _stock(cfg)
        model = _ours(cfg, stock)
        plain, _ = _steps(stock, TOKENS)
        got, _ = _steps(model, TOKENS)
        self.assertFalse(mx.array_equal(plain[0], got[0]))


class TestCacheLayout(unittest.TestCase):
    def test_shared_indexer_cache_tracks_offset_and_trims(self):
        from mlx_lm.models.cache import trim_prompt_cache

        cfg = _config()
        model = _ours(cfg, _stock(cfg))
        _, cache = _steps(model, TOKENS)
        n = TOKENS.shape[1] + 3
        for i, c in enumerate(cache):
            self.assertEqual(c[0].offset, n, f"layer {i}")
            self.assertEqual(c[1].offset, n, f"layer {i}")
            keys, _ = c[1].state
            if TYPES[i] == "shared":
                self.assertEqual(keys.shape[-1], 0)
                self.assertEqual(c[1].nbytes, 0)
        self.assertEqual(trim_prompt_cache(cache, 2), 2)
        for c in cache:
            self.assertEqual((c[0].offset, c[1].offset), (n - 2, n - 2))
        out = model(mx.array([[3]]), cache=cache)
        mx.eval(out)


class TestFastAttention(unittest.TestCase):
    """Sparse, fused and latent attention against mlx-lm's own order."""

    def _run(self, mode, rows, model, tokens, sparse_from=0):
        from pagedmoe import dsa_shared

        old = (dsa_shared._PREFILL_ATTN, dsa_shared._ABSORB_ROWS, dsa_shared._SPARSE_FROM)
        dsa_shared.set_prefill_attention(mode, sparse_from)
        dsa_shared._ABSORB_ROWS = rows
        # Toy shapes never pay for sparse under the real cost fit.
        wins = lambda r, k: 0 < sparse_from <= k  # noqa: E731
        try:
            from mlx_lm.models.cache import make_prompt_cache

            with mock.patch.object(dsa_shared, "_sparse_wins", wins):
                cache = make_prompt_cache(model)
                outs = [model(tokens, cache=cache)]
                outs.append(model(mx.array([[5, 9, 33]]), cache=cache))
                outs.append(model(mx.array([[7]]), cache=cache))
                mx.eval(outs)
            return outs
        finally:
            dsa_shared.set_prefill_attention(old[0], old[2])
            dsa_shared._ABSORB_ROWS = old[1]

    def _check(self, model):
        tokens = mx.array([[(i * 13) % 113 for i in range(24)]])
        ref = self._run("stock", 0, model, tokens)
        # The latent and sparse paths sum in a different order; against an
        # fp64 reference they are as close as the stock order (~1e-3 here).
        for rows, sparse_from, tol in (
            (0, 0, 1e-4), (2, 0, 1e-2), (256, 0, 1e-2), (0, 1, 1e-2), (256, 8, 1e-2),
        ):
            got = self._run("fused", rows, model, tokens, sparse_from)
            for i, (a, b) in enumerate(zip(ref, got)):
                err = float(mx.max(mx.abs(a - b)))
                self.assertLess(
                    err, tol, f"rows={rows} sparse={sparse_from} step {i} diverged by {err}"
                )

    def test_fast_paths_match_stock(self):
        cfg = _config()
        self._check(_ours(cfg, _stock(cfg)))

    def test_fast_paths_match_stock_without_sharing(self):
        # A rounding-level change can flip a near-tied top-k pick in the next
        # layer's indexer, so every layer gets the same fixed top-k here.
        from mlx_lm.utils import _get_classes

        from pagedmoe import dsa_shared

        cfg = _config(model_type="deepseek_v32", indexer_types=None)
        model = _ours(cfg, _stock(cfg), dsa_shared._shared_classes(*_get_classes(cfg)))
        for layer in model.model.layers:
            layer.self_attn.indexer = _Recent(cfg["index_topk"])
        self._check(model)

    def test_sparse_path_runs(self):
        cfg = _config()
        model = _ours(cfg, _stock(cfg))
        tokens = mx.array([[(i * 13) % 113 for i in range(24)]])
        fused = self._run("fused", 0, model, tokens, 0)
        sparse = self._run("fused", 0, model, tokens, 1)
        self.assertFalse(mx.array_equal(fused[0], sparse[0]))

    def test_sparse_cost_fit(self):
        from pagedmoe import dsa_shared

        old = dsa_shared._SPARSE_FROM
        self.addCleanup(dsa_shared.set_prefill_attention, dsa_shared._PREFILL_ATTN, old)
        dsa_shared.set_prefill_attention("fused", 5120)
        wins = dsa_shared._sparse_wins
        self.assertFalse(wins(12, 4096))
        self.assertTrue(wins(12, 8192))
        self.assertFalse(wins(570, 5200))
        self.assertTrue(wins(570, 14000))
        self.assertFalse(wins(2048, 8192))
        self.assertTrue(wins(2048, 14069))
        dsa_shared.set_prefill_attention("fused", 0)
        self.assertFalse(wins(570, 50000))

    def test_sparse_needs_topk(self):
        """Below index_topk keys there is no top-k, so nothing goes sparse."""
        cfg = _config(index_topk=64)
        model = _ours(cfg, _stock(cfg))
        tokens = mx.array([[(i * 13) % 113 for i in range(24)]])
        ref = self._run("fused", 0, model, tokens, 0)
        got = self._run("fused", 0, model, tokens, 1)
        for a, b in zip(ref, got):
            self.assertTrue(mx.array_equal(a, b))

    def test_kill_switch_is_stock(self):
        from pagedmoe import dsa_shared

        cfg = _config()
        stock = _stock(cfg)
        model = _ours(cfg, stock)
        tokens = mx.array([[(i * 13) % 113 for i in range(24)]])
        ref = self._run("stock", 0, _reference(stock), tokens)
        got = self._run("stock", dsa_shared._ABSORB_ROWS, model, tokens)
        for a, b in zip(ref, got):
            self.assertTrue(mx.array_equal(a, b))


class TestPrefixDisk(unittest.TestCase):
    """A DSA CacheList saves, loads whole and loads trimmed."""

    def setUp(self):
        import tempfile
        from pathlib import Path

        from pagedmoe import config, prefix_disk

        self.tmp = Path(tempfile.mkdtemp(prefix="dsa-prefix-disk-"))
        old = (config.PREFIX_DISK, config.PREFIX_DISK_DIR)
        config.PREFIX_DISK, config.PREFIX_DISK_DIR = True, str(self.tmp / "cache")
        prefix_disk.reset()

        def restore():
            config.PREFIX_DISK, config.PREFIX_DISK_DIR = old
            prefix_disk.reset()

        self.addCleanup(restore)
        d = self.tmp / "m"
        d.mkdir()
        (d / "config.json").write_text('{"model_type": "glm_moe_dsa"}')
        (d / "model-00001.safetensors").write_bytes(b"\0" * 64)
        self.key = (str(d), None, None)

    def test_round_trip_and_partial(self):
        import copy

        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

        from pagedmoe import prefix_disk

        cfg = _config(max_position_embeddings=1024)
        model = _ours(cfg, _stock(cfg))
        a = [(i * 37 + 11) % 113 for i in range(560)]
        p, k = 500, 400
        b = a[:k] + [(t + 1) % 113 for t in a[k:]]
        live = make_prompt_cache(model)
        mx.eval(model(mx.array([a[:p]]), cache=live))
        self.assertTrue(prefix_disk.trims_exactly(live))
        self.assertTrue(prefix_disk.save(self.key, a[:p], live, model, 1 << 40))

        prefix_disk.reset()
        toks, cache = prefix_disk.load(self.key, a, model, 0, 1 << 40)
        self.assertEqual(toks, a[:p])
        for i, c in enumerate(cache):
            self.assertEqual((c[0].offset, c[1].offset), (p, p), f"layer {i}")
            if TYPES[i] == "shared":
                self.assertEqual(c[1].state[0].shape[-1], 0)
        ref = copy.deepcopy(live)
        la = model(mx.array([a[p:]]), cache=cache)
        lb = model(mx.array([a[p:]]), cache=ref)
        self.assertTrue(mx.array_equal(la, lb), "whole hit differs from the live cache")

        got = prefix_disk.load(self.key, b, model, 0, 1 << 40)
        self.assertIsNotNone(got, "partial match on a DSA layout missed")
        toks, cache = got
        self.assertEqual(toks, b[:k])
        self.assertTrue(all(c[0].offset == c[1].offset == k for c in cache))
        ram = copy.deepcopy(live)
        trim_prompt_cache(ram, p - k)
        la = model(mx.array([b[k:]]), cache=cache)
        lb = model(mx.array([b[k:]]), cache=ram)
        self.assertTrue(mx.array_equal(la, lb), "trimmed hit differs from an in-memory trim")


class TestFusedPrefill(unittest.TestCase):
    def tearDown(self):
        from pagedmoe import prefill_fused

        prefill_fused.uninstall()

    def test_sub_chunked_attention_matches_whole_chunk(self):
        from mlx_lm.models.cache import make_prompt_cache

        from pagedmoe import prefill_fused

        cfg = _config()
        model = _ours(cfg, _stock(cfg))
        tokens = mx.array([[(i * 11) % 113 for i in range(40)]])
        prefill_fused.uninstall()
        ref_cache = make_prompt_cache(model)
        ref = model(tokens, cache=ref_cache)
        mx.eval(ref)
        for sub in (9, 16, 17):
            prefill_fused.uninstall()
            cache = make_prompt_cache(model)
            with mock.patch.object(
                prefill_fused.adaptive_prefill,
                "attention_sub_chunk",
                lambda owner, kv, s=sub: s,
            ), mock.patch.object(prefill_fused.config, "FUSED_PREFILL", True):
                self.assertTrue(prefill_fused.install(model))
                got = model(tokens, cache=cache)
                mx.eval(got)
            err = float(mx.max(mx.abs(got - ref)))
            self.assertLess(err, 2e-3, f"sub_chunk={sub} diverged by {err}")
            for a, b in zip(ref_cache, cache):
                self.assertEqual(a[1].offset, b[1].offset)


if __name__ == "__main__":
    unittest.main()
