# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for checkpoint key aliasing and PLE skip accounting."""

from __future__ import annotations

from pagedmoe.checkpoint_keys import lookup_tensor, tensor_key_candidates
from pagedmoe.safetensors_index import TensorLoc


def _loc(name: str, nbytes: int = 100) -> TensorLoc:
    return TensorLoc(file="/tmp/x.safetensors", dtype="F16", shape=(10,), offset=0, nbytes=nbytes)


def test_tensor_key_candidates_adds_language_model_prefix():
    keys = tensor_key_candidates("model.layers.0.mlp.switch_mlp")
    assert keys[0] == "model.layers.0.mlp.switch_mlp"
    assert "language_model.model.layers.0.mlp.switch_mlp" in keys


def test_lookup_tensor_resolves_language_model_prefix():
    idx = {
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": _loc("w"),
    }
    found = lookup_tensor(idx, "model.layers.0.mlp.switch_mlp.gate_proj.weight")
    assert found is not None
    assert found.nbytes == 100


def test_skipped_resident_excludes_ple_and_vision():
    from pagedmoe.ple_offload import skipped_resident_tensor_names

    idx = {
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": _loc(
            "ple", 1_000_000
        ),
        "vision_tower.foo.weight": _loc("v", 50_000),
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": _loc("e", 200_000),
        "language_model.model.embed_tokens.weight": _loc("emb", 10_000),
    }
    skip = skipped_resident_tensor_names(idx)
    assert "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight" in skip
    assert "vision_tower.foo.weight" in skip
    assert "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight" not in skip
    assert "language_model.model.embed_tokens.weight" not in skip
