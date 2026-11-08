# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
MiniMax routing and checkpoint-local model classes.

MiniMax blocks keep a plain nn.Linear router but select by
sigmoid(logits) + e_score_correction_bias and mix with the normalized
sigmoid, so the softmax assumptions of the Linear path do not hold.

The M2 checks use mlx-lm's minimax.py. The M3 check needs minimax_m3.py,
which mlx-lm does not ship; it runs when a MiniMax-M3 checkpoint is on disk
(or MINIMAX_M3_PY points at the file) and skips otherwise.

    python tests/test_minimax.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.cache import make_prompt_cache

from pagedmoe import config as es_config
from pagedmoe import get_stats, load
from pagedmoe import local_arch
from pagedmoe import streaming as st

_COMMON = {
    "hidden_size": 128,
    "intermediate_size": 64,
    "shared_intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "rotary_dim": 16,
    "num_hidden_layers": 4,
    "num_local_experts": 16,
    "num_experts_per_tok": 4,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "vocab_size": 512,
    "max_position_embeddings": 2048,
    "tie_word_embeddings": False,
    "quantization": {"group_size": 32, "bits": 4},
}

M2_CONFIG = {**_COMMON, "model_type": "minimax"}

M3_CONFIG = {
    **_COMMON,
    "model_type": "minimax_m3",
    "dense_intermediate_size": 256,
    "routed_scaling_factor": 2.0,
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
}


def _m3_source() -> Path | None:
    env = os.environ.get("MINIMAX_M3_PY")
    cands = [Path(env)] if env else []
    cands.append(Path("~/mlx-models/MiniMax-M3-MLX-4bit/minimax_m3.py").expanduser())
    for c in cands:
        if c.is_file():
            return c
    return None


def _build(dest: Path, module, cfg: dict) -> None:
    mx.random.seed(11)
    model = module.Model(module.ModelArgs.from_dict(cfg))
    # Zero correction bias would make sigmoid selection match logit order and
    # hide exactly what these tests are about.
    for layer in model.model.layers:
        moe = getattr(layer, "block_sparse_moe", None)
        if moe is not None:
            moe.e_score_correction_bias = mx.random.normal(
                (cfg["num_local_experts"],)
            ) * 0.5
    nn.quantize(model, group_size=32, bits=4)
    mx.save_safetensors(
        str(dest / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    with open(dest / "config.json", "w") as f:
        json.dump(cfg, f)
    from tokenizers import Tokenizer, models

    vocab = {f"t{i}": i for i in range(cfg["vocab_size"])}
    Tokenizer(models.WordLevel(vocab, unk_token="t0")).save(
        str(dest / "tokenizer.json")
    )
    (dest / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
    )


def _greedy(model, prompt, n_new):
    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt]), cache=cache)
    tok = int(mx.argmax(logits[:, -1, :]))
    out = [tok]
    for _ in range(n_new - 1):
        logits = model(mx.array([[tok]]), cache=cache)
        tok = int(mx.argmax(logits[:, -1, :]))
        out.append(tok)
    return out


def _check_streamed_matches_resident(path: Path, vocab: int) -> None:
    rng = np.random.default_rng(5)
    prompt = rng.integers(0, vocab, size=40).tolist()
    n_new = 16

    model_r, _ = load(str(path), mode="resident", verbose=False)
    ref = _greedy(model_r, prompt, n_new)
    del model_r

    model_s, _ = load(str(path), mode="streamed", cache_gb=1.0, verbose=False)
    preds = [g.pred for g in model_s._pagedmoe_ring.layers]
    assert preds and all(p is not None and p.score_bias is not None for p in preds)
    got = _greedy(model_s, prompt, n_new)
    assert got == ref, f"streamed != resident\n  ref={ref}\n  got={got}"
    # A stitched slab reads every expert at load, so decode itself never misses.
    stitched = model_s._pagedmoe_ring.layers[0].cache.stitched
    assert stitched or get_stats(model_s)["cache"]["misses"] > 0
    del model_s

    model_t, _ = load(str(path), mode="streamed", cache_gb=0.0002, verbose=False)
    got = _greedy(model_t, prompt, n_new)
    assert got == ref, f"starved streamed != resident\n  ref={ref}\n  got={got}"
    assert get_stats(model_t)["cache"]["evictions"] > 0
    del model_t

    es_config.PRUNE = 1e-9
    try:
        model_p, _ = load(str(path), mode="streamed", cache_gb=1.0, verbose=False)
        got = _greedy(model_p, prompt, n_new)
        assert got == ref, f"prune eps changed output\n  ref={ref}\n  got={got}"
        del model_p

        es_config.PRUNE = 0.9
        es_config.KEEP_FREE = False
        model_q, _ = load(str(path), mode="streamed", cache_gb=1.0, verbose=False)
        assert len(_greedy(model_q, prompt, n_new)) == n_new
        assert get_stats(model_q)["cache"]["pruned_frac"] > 0
        del model_q
    finally:
        es_config.PRUNE = 0.0
        es_config.KEEP_FREE = True


def test_local_arch_registers_checkpoint_module():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "config.json").write_text(json.dumps({"model_type": "es_toy_arch"}))
        (tmp / "es_toy_arch.py").write_text(
            "from .qwen3_moe import Model, ModelArgs  # noqa: F401\n"
        )
        assert local_arch.register(tmp) == str(tmp / "es_toy_arch.py")
        from mlx_lm.models import es_toy_arch, qwen3_moe

        assert es_toy_arch.Model is qwen3_moe.Model
        assert local_arch.register(tmp) is None
        sys.modules.pop("mlx_lm.models.es_toy_arch", None)

        # A type mlx-lm already has is never shadowed by a checkpoint file.
        (tmp / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))
        (tmp / "qwen3_moe.py").write_text("raise RuntimeError('must not run')\n")
        assert local_arch.register(tmp) is None


def test_sigmoid_router_weights_match_block():
    mx.random.seed(3)
    E, K, D = 16, 4, 32
    gate = nn.Linear(D, E, bias=False)
    bias = mx.random.normal((E,)) * 0.5
    x = mx.random.normal((3, D))

    logits = gate(x.astype(mx.float32))
    scores = mx.sigmoid(logits)
    inds = mx.argpartition(-(scores + bias), kth=K - 1, axis=-1)[..., :K]
    want = mx.take_along_axis(scores, inds, axis=-1)
    want = want / want.sum(axis=-1, keepdims=True)

    pred = st._RoutePredictor(gate, None, bias)
    sel = st._router_select(pred, logits)
    picked = mx.argpartition(sel, kth=-K, axis=-1)[..., -K:]
    assert np.array_equal(
        np.sort(np.asarray(picked), axis=-1), np.sort(np.asarray(inds), axis=-1)
    )

    w = st._route_weights_from_gate(gate, x, inds, score_bias=bias)
    assert np.allclose(w, np.asarray(want), atol=1e-6)
    w = st._route_weights_from_gate(gate, x, inds, tapped=logits, score_bias=bias)
    assert np.allclose(w, np.asarray(want), atol=1e-6)

    st._tap_router(gate)
    xf = x.astype(mx.float32)
    gate(xf)
    rank, ratio = st._route_rank_mx(pred, x, inds)
    w = np.asarray(want)
    assert np.allclose(np.asarray(ratio), w / w.max(axis=-1, keepdims=True), atol=1e-6)
    assert np.array_equal(np.argsort(np.asarray(rank)), np.argsort(w))


def test_minimax_m2_streamed_matches_resident():
    from mlx_lm.models import minimax

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _build(tmp, minimax, M2_CONFIG)
        _check_streamed_matches_resident(tmp, M2_CONFIG["vocab_size"])


def test_minimax_m3_streamed_matches_resident():
    src = _m3_source()
    if src is None:
        print("skip: no minimax_m3.py on disk")
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(src, tmp / "minimax_m3.py")
        (tmp / "config.json").write_text(json.dumps(M3_CONFIG))
        local_arch.register(tmp)
        from mlx_lm.models import minimax_m3

        _build(tmp, minimax_m3, M3_CONFIG)
        _check_streamed_matches_resident(tmp, M3_CONFIG["vocab_size"])

        from pagedmoe import prefill_fused

        model, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
        for layer in model.model.layers:
            assert prefill_fused.layer_is_fusable(layer)
            assert prefill_fused._ffn(layer) is not None
        del model


if __name__ == "__main__":
    test_local_arch_registers_checkpoint_module()
    test_sigmoid_router_weights_match_block()
    test_minimax_m2_streamed_matches_resident()
    test_minimax_m3_streamed_matches_resident()
    print("OK: MiniMax routing")
