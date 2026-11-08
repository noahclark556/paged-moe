# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where one DSA layer's prefill time goes, at a checkpoint's real shapes.

Synthetic weights (random packed ints, so nothing is read from disk). Times,
per layer, for a prompt of T tokens in attention sub-chunks of C rows:

  attn     the layer's attention over all sub-chunks (full indexer layer)
  attn-sh  same for a shared-indexer layer (top-k from the full layer)
  moe-loop streaming._run style: one quantized_matmul per expert
  moe-gath one sorted gather_qmm per projection (mlx-lm SwitchGLU style)

Defaults are GLM-5.2's. Variants: stock (mlx-lm order), fused (sparse off),
sparse (fused below PAGED_MOE_DSA_SPARSE_FROM, sparse past it). --tail N times
only the last N rows, the way a tool turn lands on a long KV.

  python bench/dsa_prefill_probe.py --tokens 13224 --sub 2048 --variants stock fused sparse
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def glm52(n_layers=4):
    types = ["full", "shared", "shared", "shared"][:n_layers]
    return dict(
        model_type="glm_moe_dsa", vocab_size=1024, hidden_size=6144,
        index_head_dim=128, index_n_heads=32, index_topk=2048,
        intermediate_size=12288, moe_intermediate_size=2048,
        num_hidden_layers=n_layers, num_attention_heads=64, num_key_value_heads=64,
        n_shared_experts=1, n_routed_experts=8, routed_scaling_factor=2.5,
        kv_lora_rank=512, q_lora_rank=2048, qk_rope_head_dim=64, v_head_dim=256,
        qk_nope_head_dim=192, topk_method="noaux_tc", scoring_func="sigmoid",
        norm_topk_prob=True, n_group=1, topk_group=1, num_experts_per_tok=8,
        moe_layer_freq=1, first_k_dense_replace=n_layers, max_position_embeddings=1 << 20,
        rms_norm_eps=1e-5, rope_parameters={"rope_theta": 8e6, "rope_type": "default"},
        attention_bias=False, indexer_types=types,
    )


def timed(fn, reps=1):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def attention_case(T, C, variant, tail=0):
    import mlx.nn as nn
    from mlx_lm.models.cache import make_prompt_cache

    from pagedmoe import dsa_shared

    sparse_from = int(os.environ.get("PAGED_MOE_DSA_SPARSE_FROM", "5120") or 0)
    cfg = glm52()
    model_cls, args_cls = dsa_shared.get_model_classes(cfg)
    model = model_cls(args_cls.from_dict(cfg))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4, class_predicate=lambda p, m: hasattr(m, "to_quantized") and "embed_tokens" not in p and "lm_head" not in p)
    mx.eval(model.parameters())
    if variant == "sparse":
        dsa_shared.set_prefill_attention("fused", sparse_from)
    elif variant is not None:
        dsa_shared.set_prefill_attention(variant, 0)
    layers = model.model.layers
    x = (mx.random.normal((1, T, 6144)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)
    out = {}
    for name, idx in (("attn", 0), ("attn-sh", 1)):
        def run():
            cache = make_prompt_cache(model)
            model._topk_store.clear()
            starts = list(range(0, T - tail, C)) + ([T - tail] if tail else [])
            for n, start in enumerate(starts):
                end = starts[n + 1] if n + 1 < len(starts) else T
                timed_rows = not tail or start >= T - tail
                xs = x[:, start:end]
                mask = _mask(end - start, start)
                if idx == 1:
                    # Shared layer needs the full layer's top-k for these rows.
                    r0 = layers[0].self_attn(xs, mask, cache[0])
                    mx.eval(r0, cache[0].state)
                    t0 = time.perf_counter()
                    r = layers[1].self_attn(xs, mask, cache[1])
                    mx.eval(r, cache[1].state)
                    run.t += (time.perf_counter() - t0) * timed_rows
                else:
                    t0 = time.perf_counter()
                    r = layers[0].self_attn(xs, mask, cache[0])
                    mx.eval(r, cache[0].state)
                    run.t += (time.perf_counter() - t0) * timed_rows
        run.t = 0.0
        run()
        run.t = 0.0
        run()
        out[name] = run.t
    return out


def _mask(n, offset):
    from mlx_lm.models.base import create_causal_mask

    return create_causal_mask(n, offset)


def moe_case(T, D=6144, I=2048, E=256, K=8, gs=64, bits=4):
    def qw(out_f, in_f):
        w = mx.random.randint(0, 1 << 31, (E, out_f, in_f * bits // 32)).astype(mx.uint32)
        s = (mx.random.uniform(shape=(E, out_f, in_f // gs)) * 0.01).astype(mx.bfloat16)
        b = (mx.random.uniform(shape=(E, out_f, in_f // gs)) * -0.005).astype(mx.bfloat16)
        return w, s, b

    W = {"up": qw(I, D), "gate": qw(I, D), "down": qw(D, I)}
    mx.eval(W)
    x = (mx.random.normal((T, D)) * 0.1).astype(mx.bfloat16)
    inds = np.stack([np.random.choice(E, K, replace=False) for _ in range(T)]).astype(np.uint32)
    flat = inds.reshape(-1)
    order = np.argsort(flat, kind="stable")
    sorted_e = flat[order]
    rows = order // K
    mx.eval(x)

    def loop():
        xs = x[mx.array(rows)]
        ys = []
        for e in range(E):
            lo, hi = np.searchsorted(sorted_e, e, "left"), np.searchsorted(sorted_e, e, "right")
            if lo == hi:
                continue
            xe = xs[lo:hi]
            u = mx.quantized_matmul(xe, *[a[e] for a in W["up"]], transpose=True, group_size=gs, bits=bits)
            g = mx.quantized_matmul(xe, *[a[e] for a in W["gate"]], transpose=True, group_size=gs, bits=bits)
            h = (g * mx.sigmoid(g)) * u
            ys.append(mx.quantized_matmul(h, *[a[e] for a in W["down"]], transpose=True, group_size=gs, bits=bits))
            if len(ys) % 18 == 0:
                mx.eval(ys[-1])
        y = mx.concatenate(ys, axis=0)
        mx.eval(y)

    rhs = mx.array(sorted_e)
    lhs = mx.array(rows.astype(np.uint32))

    def gath():
        xe = x[:, None, :]
        u = mx.gather_qmm(xe, *W["up"], lhs_indices=lhs, rhs_indices=rhs, transpose=True, group_size=gs, bits=bits, sorted_indices=True)
        g = mx.gather_qmm(xe, *W["gate"], lhs_indices=lhs, rhs_indices=rhs, transpose=True, group_size=gs, bits=bits, sorted_indices=True)
        h = (g * mx.sigmoid(g)) * u
        y = mx.gather_qmm(h, *W["down"], rhs_indices=rhs, transpose=True, group_size=gs, bits=bits, sorted_indices=True)
        mx.eval(y)

    return {"moe-loop": timed(loop), "moe-gath": timed(gath)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[13224])
    ap.add_argument("--sub", type=int, default=2048)
    ap.add_argument("--skip-moe", action="store_true")
    ap.add_argument("--variants", nargs="+", default=["stock"])
    ap.add_argument("--tail", type=int, default=0)
    args = ap.parse_args()
    mx.random.seed(0)
    np.random.seed(0)
    for T in args.tokens:
        for v in args.variants:
            r = attention_case(T, args.sub, v, args.tail)
            tail = f" tail={args.tail}" if args.tail else ""
            print(f"T={T} C={args.sub}{tail} [{v}] " + "  ".join(f"{k} {s:.3f}s" for k, s in r.items()), flush=True)
        if not args.skip_moe:
            r = moe_case(T)
            print(f"T={T} " + "  ".join(f"{k} {s:.3f}s" for k, s in r.items()), flush=True)
        mx.clear_cache()


if __name__ == "__main__":
    main()
