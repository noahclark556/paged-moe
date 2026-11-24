# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fold a LoRA / DoRA adapter into float weights.

Uses mlx-lm's own fuse() on a float stand-in of the layer, so the merged
weight is exactly what `mlx_lm.fuse` would give, one module at a time.
"""

from __future__ import annotations


def merge(kind: str, weight, tensors: dict, adapter: dict):
    """weight with the adapter for one module folded in, same dtype."""
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchLinear
    from mlx_lm.tuner.dora import DoRALinear
    from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear

    a, b = tensors["lora_a"], tensors["lora_b"]
    scale = adapter["scale"]
    dora = adapter["kind"] == "dora"
    if kind == "linear":
        out_dims, in_dims = weight.shape
        base = nn.Linear(in_dims, out_dims, bias=False)
        base.weight = weight
        r = a.shape[-1]
        layer = (DoRALinear if dora else LoRALinear).from_base(base, r=r, scale=scale)
        if dora:
            if "m" not in tensors:
                raise ValueError("DoRA adapter without magnitudes (m)")
            layer.m = tensors["m"]
    elif kind == "switch":
        n_exp, out_dims, in_dims = weight.shape
        base = SwitchLinear(in_dims, out_dims, n_exp, bias=False)
        base.weight = weight
        layer = LoRASwitchLinear.from_base(base, r=b.shape[-1], scale=scale)
    elif kind == "embedding":
        n, dims = weight.shape
        base = nn.Embedding(n, dims)
        base.weight = weight
        layer = LoRAEmbedding.from_base(base, r=a.shape[-1], scale=scale)
    else:
        raise ValueError(f"can't merge into a {kind} layer")
    layer.lora_a = a
    layer.lora_b = b
    fused = layer.fuse()
    return fused.weight.astype(weight.dtype)
