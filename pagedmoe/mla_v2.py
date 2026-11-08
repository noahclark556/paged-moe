# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keep DeepSeek-V2's KV cache in the compressed MLA latent.

mlx-lm's DeepseekV2Attention expands every token to num_attention_heads and
stores that in the cache. On V2.5 that is about 5.6 MB per token across 60
layers, so an 8k prompt is ~45 GB of KV on top of the backbone and the score
matrix, and Metal aborts. The checkpoint was trained to cache one latent
(kv_lora_rank + RoPE) per token. kv_b_proj is linear, so expanding at
attention time matches the stock math and the resident cache stays small.
"""

from __future__ import annotations

import os

import mlx.core as mx

_INSTALLED = False
# Latent-space attention for single-token decode. 0 falls back to expanding
# the cache every step (the prefill path).
_ABSORB = os.environ.get("PAGED_MOE_MLA_ABSORB", "1") != "0"
# Zero-pad prefill attention to a fused-kernel head size. 0 keeps the
# unpadded (score-matrix) path.
_PAD_PREFILL = os.environ.get("PAGED_MOE_MLA_PAD", "1") != "0"
_FUSED_DIMS = (64, 80, 128, 256)


def install() -> None:
    """Replace DeepseekV2Attention.forward once. Safe to call again."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from mlx_lm.models.deepseek_v2 import DeepseekV2Attention
    except ImportError:
        return
    if getattr(DeepseekV2Attention.__call__, "_mla_v2", False):
        _INSTALLED = True
        return

    def _call(self, x, mask=None, cache=None):
        return latent_forward(self, x, mask, cache)

    _call._mla_v2 = True  # type: ignore[attr-defined]
    DeepseekV2Attention.__call__ = _call
    _INSTALLED = True


def latent_forward(self, x, mask=None, cache=None):
    """Same projections as mlx-lm, but the cache stores the latent."""
    B, L, _D = x.shape

    if self.q_lora_rank is None:
        q = self.q_proj(x)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

    q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
    q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)

    compressed_kv = self.kv_a_proj_with_mqa(x)
    compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
    k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
    kv_latent = self.kv_a_layernorm(compressed_kv)
    kv_latent = mx.expand_dims(kv_latent, axis=1)

    offset = cache.offset if cache is not None else 0
    q_pe = self.rope(q_pe, offset)
    k_pe = self.rope(k_pe, offset)

    if cache is not None:
        kv_latent, k_pe = cache.update_and_fetch(kv_latent, k_pe)

    from mlx_lm.models.base import scaled_dot_product_attention

    if L == 1 and _ABSORB:
        return _absorbed_decode(self, q_nope, q_pe, kv_latent, k_pe, mask, B)

    # kv_b_proj is linear and per token, so expanding the whole cache matches
    # expanding each chunk before it was stored.
    S = kv_latent.shape[2]
    flat = kv_latent.reshape(B, S, self.kv_lora_rank)
    expanded = self.kv_b_proj(flat)
    expanded = expanded.reshape(B, S, self.num_heads, -1).transpose(0, 2, 1, 3)
    k_nope, values = mx.split(expanded, [self.qk_nope_head_dim], axis=-1)
    k_pe = mx.repeat(k_pe, self.num_heads, axis=1)
    k_parts = [k_nope, k_pe]
    q_parts = [q_nope, q_pe]
    v_dim = values.shape[-1]
    pad = _fused_pad_dim(self.q_head_dim, v_dim) if (mask is None or isinstance(mask, str)) else None
    if pad is not None:
        # Zero columns add nothing to q.k, so the scores are unchanged, and a
        # head size Metal has a fused kernel for skips the score buffer.
        extra = pad - self.q_head_dim
        k_parts.append(mx.zeros((*k_nope.shape[:-1], extra), dtype=k_nope.dtype))
        q_parts.append(mx.zeros((*q_nope.shape[:-1], extra), dtype=q_nope.dtype))
        values = mx.concatenate(
            [values, mx.zeros((*values.shape[:-1], pad - v_dim), dtype=values.dtype)], axis=-1
        )
    keys = mx.concatenate(k_parts, axis=-1)
    queries = mx.concatenate(q_parts, axis=-1)

    # cache=None keeps this on the fused kernel. A quantized cache would
    # materialize [heads, queries, keys] in one buffer.
    output = scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache=None,
        scale=self.scale,
        mask=mask,
    )
    if pad is not None:
        output = output[..., :v_dim]
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output)


def _fused_pad_dim(qk_dim: int, v_dim: int):
    """Head size to pad q/k/v to for Metal's fused prefill SDPA, or None.

    V2.5 attends with 192-wide q/k and 128-wide v. mlx 0.32 has no fused
    kernel for that pair, so it builds the full score matrix instead.
    """
    if not _PAD_PREFILL or (qk_dim == v_dim and qk_dim in _FUSED_DIMS):
        return None
    for d in _FUSED_DIMS:
        if d >= max(qk_dim, v_dim):
            return d
    return None


def _kv_b_matmul(self, x, transpose: bool):
    """x @ kv_b_proj per head, weight viewed as [heads, nope + v, kv_lora]."""
    proj = self.kv_b_proj
    H = self.num_heads
    rows = self.qk_nope_head_dim + self.v_head_dim
    w = proj.weight.reshape(H, rows, -1)
    if "scales" not in proj:
        return x @ (w.swapaxes(-1, -2) if transpose else w)
    scales = proj.scales.reshape(H, rows, -1)
    biases = proj.get("biases")
    if biases is not None:
        biases = biases.reshape(H, rows, -1)
    kwargs = {"group_size": proj.group_size, "bits": proj.bits}
    mode = getattr(proj, "mode", None)
    if mode is not None:
        kwargs["mode"] = mode
    return mx.quantized_matmul(
        x, w, scales=scales, biases=biases, transpose=transpose, **kwargs
    )


def _absorbed_decode(self, q_nope, q_pe, kv_latent, k_pe, mask, B):
    """One query token, attended in the latent space (mlx-lm's V3 decode).

    Folding kv_b_proj into the query and the output means each step reads the
    cached latent once, instead of expanding every cached token to all heads.
    At 10k context on V2.5 that is ~100x less attention work per token.
    The packed kv_b_proj rows are used as is, so there is no requant and no
    extra resident memory.
    """
    from mlx_lm.models.base import scaled_dot_product_attention

    nope = self.qk_nope_head_dim
    pad = mx.zeros((*q_nope.shape[:-1], self.v_head_dim), dtype=q_nope.dtype)
    q_lat = _kv_b_matmul(self, mx.concatenate([q_nope, pad], axis=-1), False)

    pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
    if isinstance(mask, mx.array):
        pe_scores = mx.where(
            mask, pe_scores, mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype)
        )

    out_lat = scaled_dot_product_attention(
        q_lat.astype(kv_latent.dtype),
        kv_latent,
        kv_latent,
        cache=None,
        scale=self.scale,
        mask=pe_scores.astype(kv_latent.dtype),
    )
    output = _kv_b_matmul(self, out_lat, True)[..., nope:]
    output = output.transpose(0, 2, 1, 3).reshape(B, 1, -1)
    return self.o_proj(output)
