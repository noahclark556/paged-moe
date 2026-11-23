# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Paged training must compute the same loss and gradients as plain mlx-lm
autodiff on the resident model.

A tiny random Qwen3-MoE checkpoint (4-bit) is loaded resident and streamed
(starved cache, so experts are evicted and re-read inside the backward pass).
LoRA weights are randomized identically on both, then:

  reference  nn.value_and_grad over model(tokens) on the resident model
  fused      FusedExecutor, resident
  layered    LayeredExecutor, resident, 3 micro-batches
  streamed   LayeredExecutor, streamed, 3 micro-batches

    python tests/test_train_parity.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_map

from test_stream import CONFIG, build_tiny_model

from pagedmoe.train import env as train_env

train_env.apply()
train_env.index_ops_stop_gradient()

from pagedmoe import load  # noqa: E402
from pagedmoe.train import experts as train_experts  # noqa: E402
from pagedmoe.train import lora as train_lora  # noqa: E402
from pagedmoe.train.executor import (  # noqa: E402
    Batch,
    FusedExecutor,
    LayeredExecutor,
)

V = CONFIG["vocab_size"]


def batches(n=3, B=2, T=40, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        ids = rng.integers(0, V, size=(B, T + 1))
        mask = (rng.random((B, T)) < 0.8).astype(np.float32)
        out.append(
            Batch(
                tokens=mx.array(ids[:, :-1].astype(np.int32)),
                targets=mx.array(ids[:, 1:].astype(np.int32)),
                mask=mx.array(mask),
            )
        )
    return out


def total_tokens(bs):
    return float(sum(b.n_tokens for b in bs))


def make_loss(norm):
    def loss_fn(lp, b):
        nll = -(lp * b.mask).sum() / norm
        return nll, {"nll": nll}

    return loss_fn


def prep(model, streamed):
    train_lora.apply(model, train_lora.LoRAConfig(rank=4, alpha=8), streamed=streamed)
    mx.random.seed(11)
    p = model.trainable_parameters()
    model.update(tree_map(lambda a: mx.random.normal(a.shape) * 0.05, p))
    mx.eval(model.parameters())
    model.train()


def flat_np(tree):
    return {k: np.asarray(v.astype(mx.float32)) for k, v in tree_flatten(tree)}


def compare(name, ref, got, tol):
    worst = 0.0
    for k, r in ref.items():
        g = got[k]
        scale = max(1e-6, float(np.abs(r).max()))
        worst = max(worst, float(np.abs(r - g).max()) / scale)
    print(f"{name}: worst relative grad diff {worst:.2e}")
    assert worst < tol, f"{name}: gradients differ ({worst:.2e} >= {tol})"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build_tiny_model(tmp)
        bs = batches()
        norm = total_tokens(bs)
        loss_fn = make_loss(norm)

        model_r, _ = load(str(tmp), mode="resident", verbose=False)
        prep(model_r, streamed=False)

        def ref_loss(model):
            tot = 0.0
            for b in bs:
                logits = model(b.tokens).astype(mx.float32)
                lp = mx.take_along_axis(logits, b.targets[..., None], -1).squeeze(-1)
                lp = lp - mx.logsumexp(logits, -1)
                tot = tot + loss_fn(lp, b)[0]
            return tot

        ref_val, ref_g = nn.value_and_grad(model_r, ref_loss)(model_r)
        mx.eval(ref_val, ref_g)
        ref_val = float(ref_val.item())
        ref = flat_np(ref_g)
        print(f"reference loss {ref_val:.6f}, {len(ref)} grad tensors")

        fused = FusedExecutor(model_r)
        val, g, _, _ = fused.step(bs, loss_fn)
        print(f"fused loss {val:.6f}")
        assert abs(val - ref_val) < 1e-4 * max(1, abs(ref_val))
        compare("fused", ref, flat_np(g), 2e-2)

        fused_ck = FusedExecutor(model_r, checkpoint=True)
        val, g, _, _ = fused_ck.step(bs, loss_fn)
        assert abs(val - ref_val) < 1e-4 * max(1, abs(ref_val))
        compare("fused+checkpoint", ref, flat_np(g), 2e-2)

        lay = LayeredExecutor(model_r, head_chunk=16)
        val, g, _, _ = lay.step(bs, loss_fn)
        print(f"layered loss {val:.6f}")
        assert abs(val - ref_val) < 1e-4 * max(1, abs(ref_val))
        compare("layered", ref, flat_np(g), 2e-2)
        del model_r

        # Starved cache: a few experts' worth, so the backward pass re-reads.
        model_s, _ = load(str(tmp), mode="streamed", cache_gb=0.0003, verbose=False)
        switches = train_experts.install(model_s)
        assert switches, "expected streamed expert layers"
        prep(model_s, streamed=True)
        cache = model_s._pagedmoe_cache
        before = cache.bytes_read
        lay_s = LayeredExecutor(model_s, head_chunk=16)
        val, g, _, _ = lay_s.step(bs, loss_fn)
        read = cache.bytes_read - before
        print(f"streamed loss {val:.6f}, read {read / 1e6:.2f} MB, evictions {cache.evictions}")
        assert abs(val - ref_val) < 1e-4 * max(1, abs(ref_val))
        compare("streamed", ref, flat_np(g), 2e-2)
        assert read > 0 and cache.evictions > 0

        # Second step reuses the previous step's routing to read ahead.
        lay_s.step(bs, loss_fn)
        snap = train_experts.COUNTERS.snapshot()
        print("expert counters:", snap)
        assert snap["ahead_hit"] is not None and snap["ahead_hit"] > 0.5
    print("OK")


def test_train_parity():
    main()


if __name__ == "__main__":
    main()
