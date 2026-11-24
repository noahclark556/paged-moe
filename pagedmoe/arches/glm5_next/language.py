# Adapted from mlx-vlm (mlx_vlm/models/glm5_next/language.py),
# Copyright (c) 2025 Prince Canuma, MIT License. See NOTICE.
# Changes for PagedMoE Copyright (C) 2026 Noah Clark,
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import BaseModelArgs, create_attention_mask, create_ssm_mask, scaled_dot_product_attention
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache
from mlx_lm.models.deepseek_v32 import group_expert_select
from mlx_lm.models import gated_delta as _gd
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.switch_layers import SwitchGLU

from .hyper_connection import HyperConnection, hc_expand


def gated_delta_update(
    q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True, lower_bound=None
):
    """mlx-lm's gated_delta plus the safe lower-bound gate GLM-5.3-Flash uses."""
    beta = mx.sigmoid(b)
    if lower_bound is None:
        g = _gd.compute_g(A_log, a, dt_bias)
    else:
        g = mx.exp(
            lower_bound
            * mx.sigmoid(
                mx.exp(A_log.astype(mx.float32))
                * (a.astype(mx.float32) + dt_bias.astype(mx.float32))
            )
        )
    if state is None:
        B, _, Hk, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    if (
        not use_kernel
        or mx.default_device() != mx.gpu
        or not mx.metal.is_available()
        or k.shape[-1] < 32
        or k.shape[-1] % 32 != 0
    ):
        return _gd.gated_delta_ops(q, k, v, g, beta, state, mask)
    return _gd.gated_delta_kernel(q, k, v, g, beta, state, mask)


# Kimi delta attention: the decay g is per key channel [B, T, H, Dk], not per
# head. train/linear_attn.py looks for this flag to swap in its chunked form.
gated_delta_update.kda = True


@dataclass
class TextConfig(BaseModelArgs):
    model_type: str = "glm5_next"
    vocab_size: int = 154880
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_shared_experts: Optional[int] = 1
    n_routed_experts: Optional[int] = 288
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    qk_nope_head_dim: int = 256
    num_experts_per_tok: int = 8
    first_k_dense_replace: int = 3
    max_position_embeddings: int = 202752
    rms_norm_eps: float = 1e-5
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    layer_types: List[str] = field(default_factory=list)
    mlp_layer_types: List[str] = field(default_factory=list)
    linear_attn_config: Dict[str, Any] = field(default_factory=dict)
    linear_num_heads: int = 64
    linear_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: Optional[float] = -5.0
    qk_head_dim: int = 256
    head_dim: int = 0
    n_group: int = 1
    topk_group: int = 1
    moe_layer_freq: int = 1
    norm_topk_prob: bool = True
    scoring_func: str = "sigmoid"
    topk_method: str = "noaux_tc"
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = False
    mla_use_nope: bool = True
    indexer_rope_interleave: bool = True
    indexer_types: Optional[List[str]] = None
    index_kpool: int = 4
    index_kpool_compress: bool = True
    index_kpool_always_select_tail: bool = True
    swiglu_limit: float = 10.0
    hc_mult: int = 4
    hc_eps: float = 1e-06
    hc_sinkhorn_iters: int = 20
    mhc: bool = True
    moe_router_dtype: str = "float32"
    num_nextn_predict_layers: int = 0
    router_aux_loss_coef: float = 0.001
    output_router_logits: bool = False
    rope_theta: float = 10000.0
    rope_parameters: Optional[Dict] = None
    rope_scaling: Optional[Dict] = None
    pad_token_id: int = 154820
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        if self.rope_parameters is not None:
            self.rope_scaling = self.rope_parameters
            self.rope_theta = self.rope_parameters.get("rope_theta", self.rope_theta)
        cfg = self.linear_attn_config or {}
        self.linear_num_heads = cfg.get("num_heads", self.linear_num_heads)
        self.linear_head_dim = cfg.get("head_dim", self.linear_head_dim)
        self.linear_conv_kernel_dim = cfg.get("short_conv_kernel_size", self.linear_conv_kernel_dim)
        self.linear_lower_bound = cfg.get("gate_lower_bound", self.linear_lower_bound)
        if cfg.get("safe_gate", True) and self.linear_lower_bound is None:
            self.linear_lower_bound = -5.0


class Glm5NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array) -> mx.array:
        dt = hidden_states.dtype
        x = hidden_states.astype(mx.float32)
        var = (x * x).mean(-1, keepdims=True)
        x = x * mx.rsqrt(var + self.eps)
        x = self.weight.astype(mx.float32) * x
        x = x * mx.sigmoid(gate.astype(mx.float32))
        return x.astype(dt)


class Glm5NextForgetGate(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.qkv_dim = self.head_dim * self.num_heads
        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.dt_bias = mx.zeros(self.qkv_dim)
        self.A_log = mx.zeros(self.num_heads)
        self.safe_gate_lower_bound = config.linear_lower_bound

    def __call__(self, hidden_states: mx.array) -> mx.array:
        B, S, _ = hidden_states.shape
        fg = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (fg.astype(mx.float32) + self.dt_bias.astype(mx.float32)).reshape(
            B, S, self.num_heads, self.head_dim
        )
        decay = mx.exp(self.A_log.astype(mx.float32)).reshape(1, 1, self.num_heads, 1)
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * mx.sigmoid(decay * g)
        g_softplus = mx.where(g > 20.0, g, mx.log(1.0 + mx.exp(g)))
        return -decay * g_softplus


def _l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def recurrent_kimi_delta(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
):
    # Reference O(S) recurrence for Kimi Delta Attention, kept as the readable
    # spec and the equivalence oracle for tests. The forward path runs this on
    # the shared fused gated_delta kernel (see Glm5NextLinearAttention).
    dt = query.dtype
    query = _l2norm(query.astype(mx.float32))
    key = _l2norm(key.astype(mx.float32))
    value = value.astype(mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    B, S, H, Dk = key.shape
    Dv = value.shape[-1]
    query = query * (Dk**-0.5)
    if state is None:
        state = mx.zeros((B, H, Dk, Dv), dtype=mx.float32)
    else:
        state = state.astype(mx.float32)
    outs = []
    for i in range(S):
        q_i = query[:, i]
        k_i = key[:, i]
        v_i = value[:, i]
        g_i = mx.exp(g[:, i])[..., None]
        b_i = beta[:, i][..., None]
        state = state * g_i
        kv_mem = (state * k_i[..., None]).sum(axis=-2)
        delta = (v_i - kv_mem) * b_i
        state = state + k_i[..., None] * delta[..., None, :]
        out_i = (state * q_i[..., None]).sum(axis=-2)
        outs.append(out_i)
    out = mx.stack(outs, axis=1).astype(dt)
    return out, state


class Glm5NextLinearAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.qkv_dim = self.num_heads * self.head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.q_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)

        self.conv_dim = self.qkv_dim * 3
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.forget_gate = Glm5NextForgetGate(config)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = Glm5NextRMSNormGated(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(self.qkv_dim, self.hidden_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # Separate projections on purpose. A fused concat of q/k/v/gates is a
        # second copy of those weights (~6.5 GB bf16 across the 34 KDA layers).
        B, S, _ = inputs.shape
        mixed = mx.concatenate(
            [self.q_proj(inputs), self.k_proj(inputs), self.v_proj(inputs)], axis=-1
        )
        fa_o = self.forget_gate.f_a_proj(inputs)
        ga_o = self.g_a_proj(inputs)
        b_o = self.b_proj(inputs)
        if mask is not None and mask.dtype == mx.bool_:
            mixed = mx.where(mask[..., None], mixed, 0)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )
        conv_input = mx.concatenate([conv_state, mixed], axis=1)
        if cache is not None:
            cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1) :, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = mx.split(conv_out, [self.qkv_dim, 2 * self.qkv_dim], axis=-1)
        q = q.reshape(B, S, self.num_heads, self.head_dim)
        k = k.reshape(B, S, self.num_heads, self.head_dim)
        v = v.reshape(B, S, self.num_heads, self.head_dim)

        fg = self.forget_gate
        a = fg.f_b_proj(fa_o).reshape(B, S, self.num_heads, self.head_dim)
        in_dtype = q.dtype
        q = (_l2norm(q.astype(mx.float32)) * (self.head_dim**-0.5)).astype(in_dtype)
        k = _l2norm(k.astype(mx.float32)).astype(in_dtype)

        state = cache[1] if cache is not None else None
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b_o,
            fg.A_log.reshape(self.num_heads, 1),
            fg.dt_bias.reshape(self.num_heads, self.head_dim),
            state=state,
            # the Metal kernel has no vjp
            use_kernel=not self.training,
            lower_bound=fg.safe_gate_lower_bound,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(S)

        gate = self.g_b_proj(ga_o).reshape(B, S, self.num_heads, self.head_dim)
        out = self.o_norm(out, gate).reshape(B, S, -1)
        return self.o_proj(out)


class Glm5NextIndexer(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.index_kpool = args.index_kpool
        self.index_kpool_always_select_tail = args.index_kpool_always_select_tail
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self.index_kpool_compress_ape = mx.zeros((self.index_kpool, self.head_dim))
        self.index_kpool_compress_gate = mx.zeros((self.head_dim, self.dim))

    def _pooled_states(self, keys, gate_scores, valid):
        B, S, hd = keys.shape
        kp = self.index_kpool
        P = (S + kp - 1) // kp
        any_valid = mx.any(valid, axis=-1)
        first_key = mx.where(
            any_valid, mx.argmax(valid.astype(mx.int32), axis=-1), mx.array(S)
        )
        pool_offsets = mx.arange(P * kp).reshape(1, P, kp)
        pool_indices = first_key[:, None, None] + pool_offsets
        safe = mx.clip(pool_indices, 0, S - 1)
        flat = safe.reshape(B, P * kp)
        idxC = mx.broadcast_to(flat[..., None], (B, P * kp, hd))
        grouped_keys = mx.take_along_axis(keys, idxC, axis=1).reshape(B, P, kp, hd)
        grouped_gate = mx.take_along_axis(gate_scores, idxC, axis=1).reshape(
            B, P, kp, hd
        )
        grouped_valid = (
            mx.take_along_axis(valid.astype(mx.int32), flat, axis=1).reshape(B, P, kp)
            > 0
        )
        grouped_valid = grouped_valid & (pool_indices < S)
        pool_valid = mx.all(grouped_valid, axis=-1)
        pool_indices = mx.where(grouped_valid, pool_indices, -1)
        logits = grouped_gate + self.index_kpool_compress_ape[None, None]
        logits = mx.where(grouped_valid[..., None], logits, -1e30)
        probs = mx.softmax(logits, axis=2)
        probs = mx.where(mx.isnan(probs), 0.0, probs)
        pool_keys = mx.sum(probs * grouped_keys, axis=2)
        return pool_keys, pool_indices, pool_valid

    def _visible_tail(self, visible, valid):
        B, S, Kv = visible.shape
        kp = self.index_kpool
        mtw = kp - 1
        any_valid = mx.any(valid, axis=-1)
        first_key = mx.where(
            any_valid, mx.argmax(valid.astype(mx.int32), axis=-1), mx.array(Kv)
        )
        visible_count = mx.sum(visible.astype(mx.int32), axis=-1)
        tail_count = visible_count - (visible_count // kp) * kp
        tail_offsets = mx.arange(mtw)
        tail_start = first_key[:, None] + visible_count - tail_count
        tail_indices = tail_start[..., None] + tail_offsets
        tail_valid = (tail_offsets[None, None, :] < tail_count[..., None]) & (
            tail_indices < Kv
        )
        kv_idx = mx.clip(tail_indices, 0, Kv - 1)
        tail_vis = mx.take_along_axis(visible, kv_idx, axis=-1)
        tail_indices = mx.where(tail_valid & tail_vis, tail_indices, -1)
        return tail_indices

    def __call__(self, x, qr, mask, cache=None):
        B, S, _ = x.shape
        q = self.wq_b(qr).reshape(B, S, self.n_heads, self.head_dim)
        k = self.k_norm(self.wk(x)).reshape(B, S, self.head_dim)
        gate_scores = x @ self.index_kpool_compress_gate.swapaxes(-1, -2)

        if mask is not None and mask.dtype == mx.bool_ and mask.shape == (B, S):
            valid_cur = mask
        else:
            valid_cur = mx.ones((B, S), dtype=mx.bool_)

        # Pack per-token state and append to the indexer cache so pooling/selection
        # run over the full cached sequence -- unifies prefill and incremental decode.
        packed = mx.concatenate(
            [k, gate_scores, valid_cur.astype(k.dtype)[..., None]], axis=-1
        )
        if cache is not None:
            keys, _ = cache.update_and_fetch(packed[:, None], mx.zeros((B, 1, S, 0)))
            packed_full = keys[:, 0]
        else:
            packed_full = packed
        T = packed_full.shape[1]
        # Short-context bypass: when the whole cache fits within index_topk the indexer
        # would select every token, so skip the O(T) pooling/scoring/topk and let the
        # DSA fall through to dense MLA. The cache is already updated above so state
        # stays consistent; the full pool is rebuilt once when T first exceeds index_topk.
        if getattr(self, "bypass_short", True) and T <= self.index_topk:
            return None
        k_full, gate_full, valid_ch = mx.split(
            packed_full, [self.head_dim, 2 * self.head_dim], axis=-1
        )
        valid = valid_ch[..., 0] > 0

        offset = T - S
        kv_len = T
        kv_pos = mx.arange(T)

        # Incremental pooling at decode: complete pools are stable across steps, so
        # recompute only the suffix (last partial pool + any new pool) and reuse the
        # cached complete pools -- turns the per-step pool cost from O(T) to O(kpool).
        # Exact; falls back to full pooling on prefill, when padding is present, or when
        # the cached pool's batch axis no longer matches the current batch. That last
        # guard matters under continuous batching: BatchGenerator grows/shrinks the
        # batch (extend/filter) on the batch axis but does not carry this per-cache
        # _pool along, so a stale _pool must be discarded and rebuilt for one step.
        # A trimmed cache (rejected drafts, rewinds) leaves _pool covering
        # positions that are gone, so reuse it only when it ends right here.
        if (
            S == 1
            and cache is not None
            and getattr(cache, "_pool", None) is not None
            and getattr(cache, "_no_pad", False)
            and cache._pool[0].shape[0] == B
            and cache._pool[3] == T - S
        ):
            ck, ci, cv, t_prev = cache._pool
            n_stable = t_prev // self.index_kpool
            s0 = n_stable * self.index_kpool
            pk_s, pi_s, pv_s = self._pooled_states(
                k_full[:, s0:], gate_full[:, s0:], valid[:, s0:]
            )
            pi_s = mx.where(pi_s >= 0, pi_s + s0, -1)
            pool_keys = mx.concatenate([ck[:, :n_stable], pk_s], axis=1)
            pool_indices = mx.concatenate([ci[:, :n_stable], pi_s], axis=1)
            pool_valid = mx.concatenate([cv[:, :n_stable], pv_s], axis=1)
        else:
            pool_keys, pool_indices, pool_valid = self._pooled_states(
                k_full, gate_full, valid
            )
            if cache is not None:
                cache._no_pad = bool(mx.all(valid))
        if cache is not None:
            cache._pool = (pool_keys, pool_indices, pool_valid, T)
        P = pool_keys.shape[1]
        select_k = min(self.index_topk // self.index_kpool, P)
        pool_end = mx.clip(pool_indices[..., -1], 0, kv_len - 1)
        pool_keys_t = pool_keys[:, None].swapaxes(-1, -2)
        tail_on = self.index_kpool_always_select_tail and self.index_kpool > 1
        output_width = self.index_topk + (self.index_kpool - 1 if tail_on else 0)

        # Chunk over the query dimension. A one-shot prefill otherwise materializes
        # [B, S, n_heads, P] scores (O(S*P)) and OOMs at long context; chunking bounds
        # peak to O(chunk*P). Decode (S=1) is a single chunk -> identical to before.
        chunk = 512 if S > 512 else S
        out = []
        for c0 in range(0, S, chunk):
            c1 = min(c0 + chunk, S)
            cs = c1 - c0
            q_pos = offset + mx.arange(c0, c1)
            visible = (kv_pos[None, None, :] <= q_pos[None, :, None]) & valid[:, None, :]
            scores = q[:, c0:c1] @ pool_keys_t
            scores = mx.maximum(scores * self.softmax_scale, 0.0)
            weights = self.weights_proj(x[:, c0:c1]) * (self.n_heads**-0.5)
            index_scores = mx.sum(weights[..., None] * scores, axis=2)
            pool_visible = mx.take_along_axis(
                visible, mx.broadcast_to(pool_end[:, None, :], (B, cs, P)), axis=-1
            )
            valid_candidates = pool_visible & pool_valid[:, None]
            index_scores = mx.where(valid_candidates, index_scores, -1e30)
            order = mx.argsort(-index_scores, axis=-1)
            selected = order[..., :select_k]
            selected_valid = mx.take_along_axis(valid_candidates, selected, axis=-1)
            pi = mx.broadcast_to(pool_indices[:, None], (B, cs, P, self.index_kpool))
            sel_exp = mx.broadcast_to(
                selected[..., None], (B, cs, select_k, self.index_kpool)
            )
            selected_indices = mx.take_along_axis(pi, sel_exp, axis=2)
            topk = selected_indices.reshape(B, cs, select_k * self.index_kpool)
            sv = mx.broadcast_to(
                selected_valid[..., None], (B, cs, select_k, self.index_kpool)
            ).reshape(B, cs, select_k * self.index_kpool)
            topk = mx.where(sv, topk, -1)
            if tail_on:
                topk = mx.concatenate([topk, self._visible_tail(visible, valid)], axis=-1)
            if topk.shape[-1] < output_width:
                pad = mx.full(
                    (B, cs, output_width - topk.shape[-1]), -1, dtype=topk.dtype
                )
                topk = mx.concatenate([topk, pad], axis=-1)
            topk = topk[..., :output_width]
            topk = mx.where(valid_cur[:, c0:c1][..., None], topk, -1)
            out.append(topk)
        topk = out[0] if len(out) == 1 else mx.concatenate(out, axis=1)
        return topk[:, None].astype(mx.int32)


class Glm5NextSparseAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.use_nope = config.mla_use_nope or config.qk_rope_head_dim == 0
        # GLM-5-Next is NoPE by design (qk_rope_head_dim=0, mla_use_nope=True); the
        # config carries no rope parameters. Fail loudly rather than run wrong math
        # if a future config ever requests a RoPE MLA.
        if not self.use_nope:
            raise NotImplementedError(
                "glm5_next implements NoPE MLA only; qk_rope_head_dim>0 with "
                "mla_use_nope=False is not supported."
            )
        self.q_head_dim = config.qk_nope_head_dim
        self.scale = self.q_head_dim**-0.5

        self.q_a_proj = nn.Linear(
            self.hidden_size, self.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank, bias=config.attention_bias
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.indexer = Glm5NextIndexer(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)

        compressed_kv = self.kv_a_proj_with_mqa(x)
        kv_latent = self.kv_a_layernorm(compressed_kv)
        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            # NoPE: keys and values are both the latent, so cache it once.
            kv_latent, _ = cache[0].update_and_fetch(
                kv_latent, mx.zeros((B, 1, L, 0), dtype=kv_latent.dtype)
            )
        else:
            cache = [None] * 2

        topk_indices = self.indexer(x, qr, mask, cache=cache[1])
        attn_mask = mask
        if topk_indices is not None:
            Kv = kv_latent.shape[2]
            valid_sel = topk_indices >= 0
            if L == 1:
                clamped = mx.clip(topk_indices[:, :, 0, :], 0, Kv - 1)
                idx = clamped[..., None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                sel_mask = valid_sel[:, :, 0, :][:, :, None, :]
                if mask is not None and mask.dtype == mx.bool_:
                    # Single-stream decode passes mask=None here; under continuous
                    # batching the batched cache supplies a left-pad mask that can be
                    # 4-D ([B, 1, 1, Kv]) while `clamped` is 3-D. At S=1 the mask is
                    # purely per-key (no causal), so reduce it to [B, Kv] and gather the
                    # selected key positions -- rank-agnostic and batch-safe.
                    mkeys = mask.reshape(B, -1, Kv)[:, 0, :]
                    gathered = mx.take_along_axis(
                        mx.broadcast_to(mkeys[:, None, :], (B, clamped.shape[1], Kv)),
                        clamped,
                        axis=-1,
                    )
                    sel_mask = sel_mask & gathered[:, :, None, :]
                attn_mask = sel_mask
            else:
                shape = list(topk_indices.shape)
                shape[-1] = Kv + 1
                safe_idx = mx.where(valid_sel, topk_indices, Kv)
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, safe_idx, mx.array(True), axis=-1
                )
                sparse_mask = sparse_mask[..., :Kv]
                if mask is not None and mask.dtype == mx.bool_:
                    sparse_mask = sparse_mask & mask
                attn_mask = sparse_mask

        if (
            cache is not None
            and cache[0] is not None
            and cache[1] is not None
            and cache[1].keys is not None
        ):
            cache[0].keys = mx.depends(cache[0].keys, (cache[1].keys, cache[1].values))

        if L == 1:
            q = self.embed_q(q)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=attn_mask
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class ClampedSwiGLU(nn.Module):
    """silu(min(gate, L)) * clip(up, -L, L), GLM-5.3's swiglu_limit."""

    def __init__(self, limit: float):
        super().__init__()
        self.limit = float(limit)

    def __call__(self, x: mx.array, gate: mx.array) -> mx.array:
        lim = self.limit
        return nn.silu(mx.minimum(gate, lim)) * mx.clip(x, -lim, lim)


class Glm5NextMLP(nn.Module):
    def __init__(self, config: TextConfig, intermediate_size: Optional[int] = None):
        super().__init__()
        hidden = config.hidden_size
        inter = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)
        self.act = ClampedSwiGLU(config.swiglu_limit)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act(self.up_proj(x), self.gate_proj(x)))


class Glm5NextGate(nn.Module):
    # Router logits in fp32 (moe_router_dtype). The checkpoint stores the
    # weight in bf16, and a bf16 matmul can flip near-tied expert picks.
    def __init__(self, config: TextConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.weight = mx.zeros((self.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed_experts,))

    def __call__(self, x: mx.array):
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        return group_expert_select(
            logits,
            self.e_score_correction_bias.astype(mx.float32),
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class Glm5NextMoE(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=ClampedSwiGLU(config.swiglu_limit),
        )
        self.gate = Glm5NextGate(config)
        self.shared_experts = None
        if config.n_shared_experts:
            self.shared_experts = Glm5NextMLP(
                config, config.moe_intermediate_size * config.n_shared_experts
            )

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class Glm5NextDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        layer_type = config.layer_types[layer_idx]
        self.is_linear = layer_type == "linear_attention"
        if self.is_linear:
            self.self_attn = Glm5NextLinearAttention(config)
        else:
            self.self_attn = Glm5NextSparseAttention(config)

        is_sparse = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and config.mlp_layer_types[layer_idx] == "sparse"
        )
        self.mlp = Glm5NextMoE(config) if is_sparse else Glm5NextMLP(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)
        # Never mx.compile the FFN: StreamedSwitchGLU pages experts from disk
        # inside the call, and compile forbids eval during the transform.

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        return self._ffn_block(self._attn_block(x, mask, cache))

    # Both halves are per token outside self_attn, so prefill_fused can run
    # _attn_block on row slices and _ffn_block once over the whole chunk.
    def _attn_block(self, x: mx.array, mask=None, cache=None) -> mx.array:
        xc, post, comb = self.attn_hc(x)
        r = self.self_attn(self.input_layernorm(xc), mask, cache)
        return hc_expand(r, x, post, comb)

    def _ffn_block(self, x: mx.array) -> mx.array:
        xc, post, comb = self.ffn_hc(x)
        m = self.mlp(self.post_attention_layernorm(xc))
        return hc_expand(m, x, post, comb)


class _MTPCache(CacheList):
    """Attention + indexer cache for the nextn block. The drafter reads
    offset to cap how long the head's context grows."""

    @property
    def offset(self) -> int:
        return self.caches[0].offset


class Glm5NextMTPLayer(nn.Module):
    """The nextn block (layer num_hidden_layers): sparse MLA and MoE on one
    plain residual stream. The checkpoint ships no mHC weights for it."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.self_attn = Glm5NextSparseAttention(config)
        self.mlp = Glm5NextMoE(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        mask = create_attention_mask(
            x, cache[0] if cache is not None else None, return_array=True
        )
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class Glm5NextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hc_mult = config.hc_mult
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Glm5NextDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ssm_idx = next((i for i, l in enumerate(self.layers) if l.is_linear), 0)
        self.fa_idx = next((i for i, l in enumerate(self.layers) if not l.is_linear), 0)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        fa_cache = cache[self.fa_idx]
        fa_mask = create_attention_mask(
            h, fa_cache[0] if fa_cache else None, return_array=True
        )
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])

        h = mx.broadcast_to(
            h[:, :, None, :], (h.shape[0], h.shape[1], self.hc_mult, h.shape[2])
        )
        h = mx.contiguous(h)

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask, cache=c)

        h = h.mean(axis=2)
        return self.norm(h)


def _dequant_rows(w, scales, biases, in_dims):
    group_size = in_dims // int(scales.shape[-1])
    bits = int(w.shape[-1]) * 32 // in_dims
    return mx.dequantize(w, scales, biases, group_size=group_size, bits=bits)


_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.")


class LanguageModel(nn.Module):
    """Text model with the stock mlx-lm layout: model.layers.N + lm_head.

    The checkpoint's per-layer quantization map is keyed on these paths
    (model.layers.N.mlp.switch_mlp.down_proj etc.), so do not nest this
    under another attribute.
    """

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model = Glm5NextModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache=cache, inputs_embeds=input_embeddings)
        if self.config.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        n_layers = int(self.config.num_hidden_layers)
        text = {}
        for k, v in weights.items():
            if k.startswith(("model.visual.", "visual.", "vision_model.", "vision_tower.")):
                continue
            if k.startswith("mtp.") or ".mtp." in k:
                continue
            if k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model.") :]
            elif k.startswith("language_model."):
                k = k[len("language_model.") :]
            m = _LAYER_KEY.match(k)
            # The MTP block is stored as one extra decoder layer past the last.
            if m and int(m.group(1)) >= n_layers:
                continue
            text[k] = v
        weights = text

        # Stack per-expert HF tensors into SwitchGLU stacks (mlx-lm MoE layout).
        n_exp = int(self.config.n_routed_experts or 0)
        for l in range(n_layers):
            prefix = f"model.layers.{l}"
            for m in ("gate_proj", "down_proj", "up_proj"):
                for k in ("weight", "scales", "biases"):
                    key0 = f"{prefix}.mlp.experts.0.{m}.{k}"
                    if key0 in weights:
                        stacked = [
                            weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                            for e in range(n_exp)
                        ]
                        weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(stacked)
            # HF packs MLA's kv_b_proj; mlx-lm MultiLinear wants embed_q + unembed_out.
            attn = f"{prefix}.self_attn"
            if f"{attn}.kv_b_proj.weight" in weights and hasattr(
                self.model.layers[l].self_attn, "embed_q"
            ):
                cfg = self.config
                v = weights.pop(f"{attn}.kv_b_proj.weight")
                scales = weights.pop(f"{attn}.kv_b_proj.scales", None)
                biases = weights.pop(f"{attn}.kv_b_proj.biases", None)
                if scales is not None:
                    v = _dequant_rows(v, scales, biases, cfg.kv_lora_rank)
                nope = cfg.qk_nope_head_dim
                v = v.reshape(cfg.num_attention_heads, nope + cfg.v_head_dim, -1)
                weights[f"{attn}.embed_q.weight"] = mx.contiguous(
                    v[:, :nope, :].swapaxes(-1, -2)
                )
                weights[f"{attn}.unembed_out.weight"] = mx.contiguous(v[:, nope:, :])

        remapped = {}
        conv_parts = {}
        fg_parts = ("A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight")
        for k, v in weights.items():
            nk = k.replace(".hc_attn_", ".attn_hc.").replace(".hc_ffn_", ".ffn_hc.")

            fused = False
            for part in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"):
                suffix = ".self_attn." + part
                if nk.endswith(suffix):
                    prefix = nk[: -len(part)]
                    conv_parts.setdefault(prefix, {})[part[0]] = v
                    fused = True
                    break
            if fused:
                continue

            for p in fg_parts:
                suffix = ".self_attn." + p
                if nk.endswith(suffix):
                    nk = nk[: -len(p)] + "forget_gate." + p
                    break

            remapped[nk] = v

        for prefix, parts in conv_parts.items():
            if all(c in parts for c in ("q", "k", "v")):
                remapped[prefix + "conv1d.weight"] = mx.concatenate(
                    [parts["q"], parts["k"], parts["v"]], axis=0
                )
            else:
                for c, w in parts.items():
                    remapped[prefix + c + "_conv1d.weight"] = w

        weights = remapped
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            elif "_hc." in k and v.dtype != mx.float32:
                # mHC mixing runs in fp32, and the Metal kernel reads base as float.
                weights[k] = v.astype(mx.float32)
        return weights

    @property
    def layers(self):
        return self.model.layers

    # Hooks for pagedmoe.mtp: the nextn block is not a clone of any backbone
    # layer (no mHC), so the model builds it and its cache itself.

    # Share of the expert cache the 4.6 GB head may take. On a 48 GB Mac it
    # takes 36% (12.8 -> 8.7 GB) and decode still goes 0.54 -> 0.93 tok/s:
    # verify passes read in big batches, and with 105 GB of experts a few GB
    # less cache barely moves the hit rate.
    mtp_max_frac = 0.4

    @property
    def mtp_proxy(self) -> int:
        """A sparse-attention MoE layer whose sanitize rules fit the nextn block."""
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            if not layer.is_linear and hasattr(layer.self_attn, "embed_q"):
                return i
        raise ValueError("no sparse-attention layer to model the MTP block on")

    def mtp_layer(self, experts: Optional[int] = None) -> Glm5NextMTPLayer:
        cfg = self.config
        if experts and experts != cfg.n_routed_experts:
            cfg = copy.copy(cfg)
            cfg.n_routed_experts = experts
        return Glm5NextMTPLayer(cfg)

    def mtp_cache(self):
        return _MTPCache(KVCache(), KVCache())

    @property
    def cast_predicate(self):
        def predicate(k):
            return "e_score_correction_bias" not in k

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if (
                path.endswith("mlp.gate")
                or "e_score_correction_bias" in path
                or ".indexer" in path
            ):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=2))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches
