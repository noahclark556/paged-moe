#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cost of the small chunk that follows a big prefill.

The server splits a turn's prefill at the stable-prefix snapshot, so every
turn ends with a few-token chunk right after a big one. This times that chunk
(and the step after it) on its own.

    python bench/tail_chunk_probe.py --model ~/mlx-models/qwen38-coder-flash \
        --big 2700 --tail 5
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import make_prompt_cache

from pagedmoe import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--big", type=int, default=2700)
    ap.add_argument("--tail", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()

    model, tok = load(args.model, verbose=False)
    cache_obj = model._pagedmoe_cache
    spent = {"leave_prefill": 0.0}
    orig_leave = cache_obj.leave_prefill

    def timed_leave(*a, **k):
        t = time.perf_counter()
        try:
            return orig_leave(*a, **k)
        finally:
            spent["leave_prefill"] += time.perf_counter() - t

    cache_obj.leave_prefill = timed_leave

    from pagedmoe import cache as cache_mod
    from pagedmoe import slab as slab_mod

    def timer(name, fn):
        def run(*a, **k):
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                spent[name] = spent.get(name, 0.0) + time.perf_counter() - t

        return run

    orig_init = slab_mod.SlabStore.__init__
    slab_mod.SlabStore.__init__ = timer("slab_init", orig_init)
    cache_mod.mx.clear_cache = timer("clear_cache", mx.clear_cache)
    cache_obj._evict_key = timer("evict", cache_obj._evict_key)
    text = Path(__file__).resolve().parent.parent.joinpath("pagedmoe", "cache.py").read_text()
    ids = tok.encode(text)
    rng = np.random.default_rng(0)
    for r in range(args.repeat):
        cache = make_prompt_cache(model)
        start = int(rng.integers(0, max(1, len(ids) - args.big - args.tail - 2)))
        big = ids[start : start + args.big]
        tail = ids[start + args.big : start + args.big + args.tail]
        t0 = time.perf_counter()
        mx.eval(model(mx.array([big]), cache=cache))
        t1 = time.perf_counter()
        r0 = cache_obj.bytes_read
        w0 = cache_obj.disk_wait_s
        for k in spent:
            spent[k] = 0.0
        out = model(mx.array([tail]), cache=cache)
        mx.eval(out)
        t2 = time.perf_counter()
        r1 = cache_obj.bytes_read
        parts = " ".join(f"{k} {v:.2f}s" for k, v in spent.items())
        print(f"[tail]   disk wait {cache_obj.disk_wait_s - w0:.2f}s, {parts}", flush=True)
        nxt = int(mx.argmax(out[0, -1]))
        mx.eval(model(mx.array([[nxt]]), cache=cache))
        t3 = time.perf_counter()
        print(
            f"[tail] run {r + 1}: big {len(big)} tok {t1 - t0:.1f}s | "
            f"tail {len(tail)} tok {t2 - t1:.2f}s ({(r1 - r0) / 1e9:.2f} GB read) | "
            f"next 1 tok {t3 - t2:.2f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
