# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Time one decode layer's GPU work in isolation (no disk, no router sync).

Loads a model, decodes a few tokens so the slab holds real experts, then for
a handful of layers times, each as its own mx.eval:

  attn    the decoder layer's attention on a 1-token input (no KV cache)
  router  gate matvec + top-k
  experts _run_slab over K resident experts
  layer   all of the above as one graph, ending in the router sync of the
          next layer (what decode actually pays per layer)

Usage: python bench/layer_gpu_probe.py --model ~/mlx-models/<ckpt>
"""

import argparse
import time

import mlx.core as mx
import numpy as np
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from pagedmoe import load
from pagedmoe.streaming import StreamedSwitchGLU


GAP_S = 0.0
PING_S = 0.0
PING_N = 1024
_ping = None


def _idle():
    if not PING_S:
        time.sleep(GAP_S)
        return
    global _ping
    if _ping is None:
        _ping = mx.zeros((PING_N,), mx.float32)
        mx.eval(_ping)
    end = time.perf_counter() + GAP_S
    while True:
        mx.eval(_ping + 1)
        now = time.perf_counter()
        if now >= end:
            return
        time.sleep(min(PING_S, end - now))


def bench(fn, reps=30):
    for _ in range(3):
        mx.eval(fn())
    spent = 0.0
    for _ in range(reps):
        if GAP_S:
            _idle()
        t0 = time.perf_counter()
        mx.eval(fn())
        spent += time.perf_counter() - t0
    return spent / reps * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", default="0,10,40,80")
    ap.add_argument("--gap-ms", type=float, default=0.0,
                    help="idle the GPU this long before each timed eval (the "
                    "shape of a disk wait between layers)")
    ap.add_argument("--ping-us", type=float, default=0.0,
                    help="during the gap, run a tiny GPU op this often")
    ap.add_argument("--ping-n", type=int, default=1024)
    args = ap.parse_args()
    global GAP_S, PING_S, PING_N
    GAP_S = args.gap_ms / 1000.0
    PING_S = args.ping_us / 1e6
    PING_N = args.ping_n

    model, tok = load(args.model)
    for _ in stream_generate(model, tok, "Write a haiku about disks.",
                             max_tokens=8, sampler=make_sampler(temp=0.0)):
        pass
    cache = model._pagedmoe_cache
    inner = getattr(model, "language_model", model)
    layers = inner.model.layers
    D = layers[0].input_layernorm.weight.shape[0]
    x = mx.random.normal((1, 1, D)).astype(mx.bfloat16)
    mx.eval(x)

    tot = {"attn": 0.0, "router": 0.0, "experts": 0.0, "layer": 0.0}
    picks = [int(s) for s in args.layers.split(",") if int(s) < len(layers)]
    for li in picks:
        layer = layers[li]
        mlp = getattr(layer, "mlp", None)
        sw = getattr(mlp, "switch_mlp", None)
        if not isinstance(sw, StreamedSwitchGLU):
            print(f"layer {li}: not a streamed MoE layer, skipped")
            continue
        K = mlp.top_k if hasattr(mlp, "top_k") else 8
        with cache._lock:
            res = [k[1] for k in cache._lru if k[0] == sw.layer_key]
        if len(res) < K:
            print(f"layer {li}: only {len(res)} resident, skipped")
            continue
        ids = np.array(sorted(res[:K]), dtype=np.uint32)[None, :]
        slots = {int(e): cache._lru[(sw.layer_key, int(e))]["__slot__"] for e in ids[0]}
        xf = x.reshape(1, D)

        attn = getattr(layer, "self_attn", None)
        try:
            t_attn = bench(lambda: attn(layer.input_layernorm(x), None, None))
        except Exception as e:  # noqa: BLE001 - linear-attention layers etc.
            print(f"layer {li}: attention not probed ({type(e).__name__})")
            continue
        t_router = bench(lambda: mx.argpartition(-mlp.gate(xf), kth=K - 1, axis=-1)[..., :K])
        t_exp = bench(lambda: sw._run_slab(xf, ids, slots))

        def full():
            h = x + layer.self_attn(layer.input_layernorm(x), None, None)
            hn = layer.post_attention_layernorm(h).reshape(1, D)
            g = mx.argpartition(-mlp.gate(hn), kth=K - 1, axis=-1)[..., :K]
            y = sw._run_slab(hn, ids, slots).sum(axis=1)
            return h.reshape(1, D) + y, g

        t_full = bench(full)
        for k, v in zip(tot, (t_attn, t_router, t_exp, t_full)):
            tot[k] += v
        print(f"layer {li}: attn {t_attn:.3f} ms  router {t_router:.3f} ms  "
              f"experts {t_exp:.3f} ms  layer-graph {t_full:.3f} ms")
    n = max(1, len(picks))
    nl = len(layers)
    print("projected per token over %d layers: " % nl + "  ".join(
        f"{k} {v / n * nl:.1f} ms" for k, v in tot.items()))


if __name__ == "__main__":
    main()
