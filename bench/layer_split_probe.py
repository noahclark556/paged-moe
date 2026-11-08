#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-sub-block decode timing for hyper-connection models (qwen4_exp).

Syncs after every sub-block of every decoder layer on 1-token steps, so the
absolute step time is inflated by the extra syncs, but the split shows where
GPU time goes: ple, attn mixer, attention, merge, mlp mixer, moe.

    python bench/layer_split_probe.py --model ~/mlx-models/qwen38-coder-flash \
        --prompt-tokens 1024 --max-tokens 48
"""

import atexit
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import decode_profile  # noqa: E402
import pagedmoe  # noqa: E402

_load = pagedmoe.load
BUCKET = defaultdict(float)
STEPS = [0]


def _patch(model):
    layers = model.model.layers
    dl_cls = type(layers[0])
    dl_orig = dl_cls.__call__
    eval_ = decode_profile._orig_eval

    def tick(name, t0, *arrs):
        eval_(*arrs)
        t1 = time.perf_counter()
        BUCKET[name] += t1 - t0
        return t1

    def dcall(self, h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx):
        if h.shape[-2] != 1:
            return dl_orig(self, h, rope, mask, conv_mask, cache, idx_cache, ids, prev_ctx)
        if self is layers[0]:
            STEPS[0] += 1
        t = tick("pre", time.perf_counter(), h)
        if self.ple is not None:
            h = h + self.ple(h, ids, prev_ctx, cache)
            t = tick("ple", t, h)
        x, hyper, inject = self.attn_hyper_connection(h)
        t = tick("mix_attn", t, x, hyper, inject)
        if self.layer_type == "linear_attention":
            x = self.linear_attn(x, conv_mask, cache)
            t = tick("deltanet", t, x)
        else:
            x = self.self_attn(x, rope, mask, cache, idx_cache)
            t = tick("attention", t, x)
        h = hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)
        x, hyper, inject = self.mlp_hyper_connection(h)
        t = tick("mix_mlp", t, x, hyper, inject)
        x = self.mlp(x)
        t = tick("moe", t, x)
        out = hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)
        tick("merge", t, out)
        return out

    dl_cls.__call__ = dcall


def _report():
    n = max(1, STEPS[0])
    total = sum(BUCKET.values())
    print(f"\n[glue] per-token split over {n} steps ({total / n * 1000:.1f} ms synced)")
    for k, v in sorted(BUCKET.items(), key=lambda kv: -kv[1]):
        print(f"[glue]   {k:10s} {v / n * 1000:7.2f} ms")


atexit.register(_report)


def load(*a, **k):
    out = _load(*a, **k)
    _patch(out[0])
    return out


decode_profile.load = load

if __name__ == "__main__":
    decode_profile.main()
