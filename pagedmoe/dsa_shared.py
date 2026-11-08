# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DSA attention for the deepseek_v32 family (DeepSeek-V3.2, GLM-5, GLM-5.2).

GLM-5.2 lists `indexer_types` in config.json. A "full" layer runs the DSA
indexer, a "shared" layer has no indexer weights and reuses the top-k the
last full layer picked for the same query rows. mlx-lm 0.31 builds an
indexer on every deepseek_v32-family layer, so the load dies with the
shared layers' indexer weights missing. Checkpoints that do not share get
the same classes with every layer "full", for the attention paths below.

The classes here are mlx-lm's own (same module tree, same CacheList(kv,
indexer) per layer), with attention swapped for one that reads top-k from
the source layer on shared layers. A shared layer still advances its
indexer cache with zero-width rows, so trim, snapshot and prefix-disk see
the same layout as DeepSeek-V3.2.

Top-k is keyed by (KV offset, query rows). fused prefill splits attention
per layer, so the shared layer looks up the slice the full layer computed
for the same rows, not whatever it computed last.

Only used while the installed mlx-lm lacks `indexer_types`. A release that
adds it takes over.

Attention math is mlx-lm's, reordered for speed. PAGED_MOE_DSA_FUSED_ATTN=0
restores the stock order (and the stock classes on checkpoints that do not
share). Decode (one row) is always the stock path.

  sparse   rows that average past DSA_SPARSE_FROM keys gather their own
           top-k latent rows and attend to those only. The cost per row is
           flat in context, where the masked dense paths grow with it.
  fused    prefill chunks run one SDPA over [nope | rope] heads with a
           boolean mask, instead of a dense pe_scores bias.
  latent   small batches (verify, short turns) attend in the latent space
           like decode, instead of expanding every cached key to K/V.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx

_MISSING = object()
_built: dict = {}
_PREFILL_ATTN = "fused" if os.environ.get("PAGED_MOE_DSA_FUSED_ATTN", "1") != "0" else "stock"
# Up to this many query rows (verify batches, short turns) attend in the
# latent space the way decode does. Expanding every cached key to per-head
# K/V costs the same at 2 rows as at 2000, so it only pays on real chunks.
# The latent path builds a dense [heads, rows, keys] score matrix, so it is
# also capped by cells (1 GB of fp32), which keeps long contexts fused.
_ABSORB_ROWS = int(os.environ.get("PAGED_MOE_DSA_ABSORB_ROWS", "256") or 0)
_ABSORB_CELLS = 1 << 28
# Fewest cached keys at which the sparse path may run (0 turns it off). Below
# this the latent path wins on small batches; prefill chunks also have to
# pass _sparse_wins.
_SPARSE_FROM = int(os.environ.get("PAGED_MOE_DSA_SPARSE_FROM", "5120") or 0)
# Gathered latent rows per block (bf16 at 2048 x 576 is ~2.4 MB a row).
_SPARSE_BLOCK_BYTES = 1 << 28


def _sparse_wins(rows: int, keys: int) -> bool:
    """Sparse against the masked path it would replace, at `keys` cached.

    Fit on GLM-5.2 shapes (M5 Pro, microseconds per layer): sparse is ~56 per
    row whatever the context; fused prefill is ~1 per cached key (expanding
    K/V) plus 0.0063 per live (row, key) pair. A 570-row turn crosses near 7k
    keys, a 2048-row chunk near 8k mean keys; at 50k sparse is ~5x.
    """
    if _SPARSE_FROM <= 0 or keys < _SPARSE_FROM:
        return False
    if rows <= _ABSORB_ROWS:
        return True
    mean = keys - (rows - 1) / 2
    return 56.0 * rows < keys + 0.0063 * rows * mean


def set_prefill_attention(mode: str, sparse_from: Optional[int] = None) -> None:
    """"fused" (sparse / fused / latent paths) or "stock" (mlx-lm's math as
    written). Bench / tests."""
    global _PREFILL_ATTN, _SPARSE_FROM
    _PREFILL_ATTN = mode
    if sparse_from is not None:
        _SPARSE_FROM = int(sparse_from)


def indexer_types(config: dict) -> Optional[list]:
    """Per-layer "full" / "shared" list, or None when no layer shares.

    Same precedence as transformers: an explicit `indexer_types`, then
    `index_topk_pattern` ("FSSS..." or a list), then `index_topk_freq` with
    `index_skip_topk_offset`.
    """
    n = int(config.get("num_hidden_layers") or 0)
    types = config.get("indexer_types")
    if types is None:
        pattern = config.get("index_topk_pattern")
        if isinstance(pattern, str):
            m = {"F": "full", "S": "shared"}
            if any(c not in m for c in pattern):
                return None
            types = [m[c] for c in pattern]
        elif isinstance(pattern, (list, tuple)):
            types = list(pattern)
        elif config.get("index_topk_freq") is not None:
            freq = max(int(config.get("index_topk_freq") or 1), 1)
            offset = config.get("index_skip_topk_offset")
            offset = 2 if offset is None else int(offset)
            types = [
                "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared"
                for i in range(n)
            ]
    if not isinstance(types, (list, tuple)) or n <= 0 or len(types) < n:
        return None
    types = [str(t) for t in types[:n]]
    if any(t not in ("full", "shared") for t in types) or types[0] != "full":
        return None
    if "shared" not in types:
        return None
    return types


def _is_dsa(model_cls) -> bool:
    try:
        from mlx_lm.models.deepseek_v32 import Model as DSV32Model
    except ImportError:
        return False
    return isinstance(model_cls, type) and issubclass(model_cls, DSV32Model)


def _upstream_shares(args_cls) -> bool:
    return "indexer_types" in getattr(args_cls, "__dataclass_fields__", {})


def get_model_classes(config: dict):
    """Drop-in for mlx_lm.utils._get_classes."""
    from mlx_lm.utils import _get_classes

    model_cls, args_cls = _get_classes(config)
    if not _is_dsa(model_cls) or _upstream_shares(args_cls):
        return model_cls, args_cls
    types = indexer_types(config)
    if types is None and _PREFILL_ATTN != "fused":
        return model_cls, args_cls
    key = (model_cls, args_cls)
    if key not in _built:
        _built[key] = _shared_classes(model_cls, args_cls)
        mt = config.get("model_type")
        if types is not None:
            print(
                f"[paged-moe] {mt}: shared DSA indexer ({types.count('full')}/"
                f"{config.get('num_hidden_layers')} layers run it)",
                flush=True,
            )
        if _PREFILL_ATTN == "fused":
            sparse = f"sparse top-k from {_SPARSE_FROM} keys" if _SPARSE_FROM > 0 else "sparse off"
            print(f"[paged-moe] {mt}: DSA attention fused, {sparse}", flush=True)
    return _built[key]


class TopkStore:
    """Top-k of the most recent full layer, per (offset, rows)."""

    def __init__(self):
        self.owner = None
        self.entries: dict = {}

    def put(self, layer: int, key, topk) -> None:
        if self.owner != layer:
            self.entries.clear()
            self.owner = layer
        self.entries[key] = topk

    def get(self, layer: int, source: int, key):
        got = self.entries.get(key, _MISSING) if self.owner == source else _MISSING
        if got is _MISSING:
            raise RuntimeError(
                f"shared DSA layer {layer}: no top-k from layer {source} "
                f"at offset {key[0]} for {key[1]} rows"
            )
        return got

    def clear(self) -> None:
        self.owner = None
        self.entries.clear()


def _shared_classes(model_cls, args_cls):
    from mlx_lm.models.base import scaled_dot_product_attention
    from mlx_lm.models.deepseek_v32 import DeepseekV32Attention

    @dataclass
    class ModelArgs(args_cls):
        indexer_types: Optional[List[str]] = None
        index_topk_pattern: Optional[Any] = None
        index_topk_freq: Optional[int] = None
        index_skip_topk_offset: Optional[int] = None

        def __post_init__(self):
            parent = getattr(super(), "__post_init__", None)
            if parent is not None:
                parent()
            self.indexer_types = indexer_types(
                {
                    "num_hidden_layers": self.num_hidden_layers,
                    "indexer_types": self.indexer_types,
                    "index_topk_pattern": self.index_topk_pattern,
                    "index_topk_freq": self.index_topk_freq,
                    "index_skip_topk_offset": self.index_skip_topk_offset,
                }
            )

    class SharedIndexerAttention(DeepseekV32Attention):
        """deepseek_v32 attention with the top-k source made swappable.

        Same math as mlx-lm's __call__; only where topk_indices comes from
        differs on shared layers.
        """

        def __call__(self, x, mask=None, cache=None):
            B, L, D = x.shape

            qr = self.q_a_layernorm(self.q_a_proj(x))
            q = self.q_b_proj(qr)

            q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
            q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
            compressed_kv = self.kv_a_proj_with_mqa(x)
            compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
            k_pe = k_pe.reshape(B, L, 1, self.qk_rope_head_dim).transpose(0, 2, 1, 3)
            kv_latent = self.kv_a_layernorm(compressed_kv)

            offset = cache[0].offset if cache is not None else 0
            q_pe = self.rope(q_pe, offset)
            k_pe = self.rope(k_pe, offset)

            kv_latent = mx.expand_dims(kv_latent, axis=1)

            if cache is not None:
                kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
            else:
                cache = [None] * 2

            key = (int(offset), int(L))
            if self.indexer is not None:
                topk_indices = self.indexer(x, qr, mask, cache=cache[1])
                self._topk_store.put(self._layer_idx, key, topk_indices)
            else:
                topk_indices = self._topk_store.get(
                    self._layer_idx, self._topk_source, key
                )
                if cache[1] is not None:
                    empty = mx.zeros((B, 1, L, 0), dtype=kv_latent.dtype)
                    cache[1].update_and_fetch(empty, empty)

            fast = _PREFILL_ATTN == "fused"
            sparse = (
                fast
                and topk_indices is not None
                and L > 1
                and _sparse_wins(L, kv_latent.shape[2])
            )
            if topk_indices is not None and not sparse:
                if L == 1:
                    idx = topk_indices[:, :, 0, :, None]
                    kv_latent = mx.take_along_axis(
                        kv_latent,
                        mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                        axis=2,
                    )
                    k_pe = mx.take_along_axis(
                        k_pe,
                        mx.broadcast_to(idx, idx.shape[:-1] + (k_pe.shape[-1],)),
                        axis=2,
                    )
                    if mask is not None:
                        mask = mx.take_along_axis(mask, topk_indices, axis=-1)
                else:
                    shape = list(topk_indices.shape)
                    shape[-1] = kv_latent.shape[2]
                    sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                    sparse_mask = mx.put_along_axis(
                        sparse_mask, topk_indices, mx.array(True), axis=-1
                    )
                    if mask is not None:
                        sparse_mask = sparse_mask & mask
                    mask = sparse_mask
            if cache is not None and cache[0] is not None:
                cache[0].keys = mx.depends(
                    cache[0].keys, (cache[1].keys, cache[1].values)
                )

            if sparse:
                return self._sparse_attend(
                    q_nope, q_pe, kv_latent, k_pe, topk_indices, mask, B, L
                )
            absorb = L == 1 or (
                fast
                and L <= _ABSORB_ROWS
                and self.num_heads * L * kv_latent.shape[2] <= _ABSORB_CELLS
            )
            if not absorb and fast and self._fused_ok:
                return self._fused_prefill(q_nope, q_pe, kv_latent, k_pe, mask, B, L)

            pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
            if mask is not None:
                pe_scores = mx.where(
                    mask,
                    pe_scores,
                    mx.array(mx.finfo(pe_scores.dtype).min, pe_scores.dtype),
                )

            if absorb:
                q_nope = self.embed_q(q_nope)
                k = v = kv_latent
            else:
                k = self.embed_q(kv_latent, transpose=False)
                v = self.unembed_out(kv_latent)

            output = scaled_dot_product_attention(
                q_nope, k, v, cache=cache, scale=self.scale, mask=pe_scores
            )
            if absorb:
                output = self.unembed_out(output)

            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output)

        @property
        def _fused_ok(self) -> bool:
            return self.qk_nope_head_dim + self.qk_rope_head_dim == self.v_head_dim

        def _fused_prefill(self, q_nope, q_pe, kv_latent, k_pe, mask, B, L):
            """Prefill attention as one SDPA over [nope | rope] heads.

            q.k = q_nope.k_nope + q_pe.k_pe, so concatenating gives the same
            scores the stock path builds as a dense pe_scores bias. With
            qk and v the same width the mask can stay boolean and the bias
            matrix is never materialized.
            """
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)
            k_pe = mx.broadcast_to(k_pe, (*k.shape[:-1], k_pe.shape[-1]))
            q = mx.concatenate([q_nope, q_pe], axis=-1)
            k = mx.concatenate([k, k_pe], axis=-1)
            if mask is not None and not isinstance(mask, str) and mask.dtype != mx.bool_:
                mask = mask > mx.finfo(mask.dtype).min / 2
            output = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self.scale, mask=mask
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output)

        def _sparse_attend(self, q_nope, q_pe, kv_latent, k_pe, topk, mask, B, L):
            """Each row attends to its own top-k keys, in the latent space.

            Same scores as the masked paths (q.k = embed_q(q_nope).latent +
            q_pe.k_pe), but over the k gathered latent rows instead of every
            cached key. Latent rows are shared by all heads, so the gather is
            576 values per key, not heads x (qk + v).
            """
            S = kv_latent.shape[2]
            K = topk.shape[-1]
            q = mx.concatenate([self.embed_q(q_nope), q_pe], axis=-1) * self.scale
            q = q.transpose(0, 2, 1, 3)
            kv = mx.concatenate([kv_latent, k_pe], axis=-1).reshape(B * S, -1)
            idx = topk.reshape(B, L, K)
            if B > 1:
                idx = idx + (mx.arange(B, dtype=idx.dtype) * S)[:, None, None]
            valid = None
            if isinstance(mask, mx.array):
                if mask.dtype != mx.bool_:
                    mask = mask > mx.finfo(mask.dtype).min / 2
                mask = mx.broadcast_to(mask, (B, 1, L, S))
                valid = mx.take_along_axis(mask, topk, axis=-1).reshape(B, L, 1, K)
            elif mask is not None:
                rows = S - L + mx.arange(L)
                valid = (topk.reshape(B, L, K) <= rows[None, :, None])[:, :, None, :]
            row_bytes = kv.shape[-1] * kv.itemsize
            rb = max(1, min(L, _SPARSE_BLOCK_BYTES // (K * row_bytes)))
            # mlx gathers one element per thread; moving 8-byte words is ~4x
            # faster than bf16 elements, and the bits are the same.
            wide = next((t for t in (mx.uint64, mx.uint32) if row_bytes % t.size == 0), None)
            src = kv.view(wide) if wide is not None else kv
            outs = []
            for a in range(0, L, rb):
                b = min(L, a + rb)
                ks = src[idx[:, a:b]]
                if wide is not None:
                    ks = ks.view(kv.dtype)
                s = q[:, a:b] @ ks.swapaxes(-1, -2)
                if valid is not None:
                    s = mx.where(valid[:, a:b], s, mx.array(mx.finfo(s.dtype).min, s.dtype))
                p = mx.softmax(s, axis=-1, precise=True)
                outs.append(p @ ks[..., : self.kv_lora_rank])
            out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=1)
            out = self.unembed_out(out.transpose(0, 2, 1, 3))
            out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(out)

    class Model(model_cls):
        def __init__(self, config):
            super().__init__(config)
            types = config.indexer_types
            store = TopkStore()
            source = None
            for i, layer in enumerate(self.model.layers):
                attn = layer.self_attn
                if type(attn) is not DeepseekV32Attention:
                    raise TypeError(
                        f"layer {i}: expected DeepseekV32Attention, got {type(attn).__name__}"
                    )
                attn.__class__ = SharedIndexerAttention
                attn._topk_store = store
                attn._layer_idx = i
                if types is not None and types[i] == "shared":
                    attn.indexer = None
                    attn._topk_source = source
                else:
                    source = i
                    attn._topk_source = i
            self._topk_store = store

        def __call__(self, *args, **kwargs):
            # Lookups are exact, so this only frees the last prefill's
            # indices between forwards.
            self._topk_store.clear()
            try:
                return super().__call__(*args, **kwargs)
            finally:
                self._topk_store.clear()

    ModelArgs.__name__ = ModelArgs.__qualname__ = args_cls.__name__
    Model.__name__ = Model.__qualname__ = model_cls.__name__
    return Model, ModelArgs
