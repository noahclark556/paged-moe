#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What a pressure-shrunk slab costs, and what growing it back costs.

Four agent-shaped turns (small prefill on seeded KV, then greedy decode):

  full     slab at its load-time size
  shrunk   slab cut to --frac, regrow held off (a session after a spike)
  regrow   same, regrow allowed: this turn's prefill pays the rebuild
  after    steady state after the regrow

The real kernel decides room (available_bytes, budget); only the calm window
is skipped, since the process just started.

    python bench/regrow_probe.py --model ~/mlx-models/qwen3-235-4bit
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from pagedmoe import config, get_stats, load, mempressure  # noqa: E402
from pagedmoe.streaming import StreamedSwitchGLU  # noqa: E402


def cache_of(model):
    for m in model.modules():
        if isinstance(m, StreamedSwitchGLU):
            return m.cache
    raise SystemExit("not a streamed model")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frac", type=float, default=0.7)
    ap.add_argument("--base-tokens", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--decode", type=int, default=64)
    args = ap.parse_args()

    config.SLAB_REGROW_CALM_S = 0.0
    mempressure.calm_s = lambda: 0.0 if mempressure.level() >= mempressure.WARN else 1e9
    model, tok = load(args.model, verbose=False)
    c = cache_of(model)
    text = "\n".join(p.read_text(errors="replace") for p in sorted(Path(__file__).parents[1].rglob("*.py"))[:80])
    ids = tok.encode(text)
    while len(ids) < args.base_tokens + 5 * args.chunk:
        ids = ids + ids
    kv = make_prompt_cache(model)
    for i in range(0, args.base_tokens, 2048):
        model(mx.array(ids[i : min(i + 2048, args.base_tokens)])[None], cache=kv)
        mx.eval([s.state for s in kv])
    at = args.base_tokens

    def turn(label):
        nonlocal at
        chunk = ids[at : at + args.chunk]
        at += args.chunk
        r0 = get_stats(model)["cache"]["read_gb"]
        t0 = time.perf_counter()
        model(mx.array(chunk[:-1])[None], cache=kv)
        mx.eval([s.state for s in kv])
        logits = model(mx.array(chunk[-1:])[None], cache=kv)
        mx.eval(logits)
        pre = time.perf_counter() - t0
        r1 = get_stats(model)["cache"]["read_gb"]
        t1 = time.perf_counter()
        for _ in range(args.decode):
            logits = model(mx.argmax(logits[:, -1, :], axis=-1)[None], cache=kv)
            mx.eval(logits)
        dt = time.perf_counter() - t1
        r2 = get_stats(model)["cache"]["read_gb"]
        print(
            f"{label:>7}: slab {c.slab.slots if c.slab else 0:5} slots  "
            f"prefill {pre:5.1f}s {r1 - r0:5.1f} GB  "
            f"decode {args.decode / dt:5.2f} tok/s {(r2 - r1) * 1e3 / args.decode:5.0f} MB/tok  "
            f"| avail {(mempressure.available_bytes() or 0) / 1e9:4.1f} GB "
            f"{mempressure.name(mempressure.level())}",
            flush=True,
        )

    turn("warmup")
    turn("full")
    small = int(c._slab_full_slots * args.frac)
    c._release_slab()
    with c._lock:
        specs, _ = c._slab_pending
        c._build_slab_locked(specs, small)
    c._regrow_after = float("inf")
    turn("shrunk")
    turn("shrunk")
    c._regrow_after = 0.0
    turn("regrow")
    turn("after")
    turn("after")
    return 0


if __name__ == "__main__":
    sys.exit(main())
