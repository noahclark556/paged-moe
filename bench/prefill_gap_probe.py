#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How long does the drive sit idle between prefill layers?

A prefill layer can't name its experts until its router runs, and its router
needs the previous layer's output. So between the last read of layer L and
the first read of layer L+1 the drive has nothing to do: the last group of L
computes, then L+1's attention and router run. This times that gap per layer
on a real incremental chunk, next to the time spent reading.

  gap      last read of L landed -> L+1 asked for its experts (drive idle)
  inside   L asked for its experts -> its last read landed

    python bench/prefill_gap_probe.py --model ~/mlx-models/qwen3-235-4bit
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from pagedmoe import load  # noqa: E402
from pagedmoe.cache import ExpertCache  # noqa: E402

EVENTS: list[tuple[str, float, float, int]] = []


def install():
    orig = ExpertCache.fetch_groups

    def timed(self, layer_key, ids, *a, **k):
        t0 = time.perf_counter()
        n = len(ids)
        gen = orig(self, layer_key, ids, *a, **k)
        last = t0
        for item in gen:
            last = time.perf_counter()
            yield item
        EVENTS.append((layer_key, t0, last, n))

    ExpertCache.fetch_groups = timed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-tokens", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--decode", type=int, default=8)
    args = ap.parse_args()

    install()
    model, tok = load(args.model, verbose=False)
    text = "\n".join(p.read_text(errors="replace") for p in sorted(Path(__file__).parents[1].rglob("*.py"))[:80])
    ids = tok.encode(text)
    kv = make_prompt_cache(model)
    for i in range(0, args.base_tokens, 2048):
        logits = model(mx.array(ids[i : min(i + 2048, args.base_tokens)])[None], cache=kv)
        mx.eval(logits)
    for _ in range(args.decode):
        logits = model(mx.argmax(logits[:, -1, :], axis=-1)[None], cache=kv)
        mx.eval(logits)

    at = args.base_tokens
    for turn in (1, 2):
        EVENTS.clear()
        t0 = time.perf_counter()
        logits = model(mx.array(ids[at : at + args.chunk])[None], cache=kv)
        mx.eval(logits)
        wall = time.perf_counter() - t0
        at += args.chunk
        gaps = [EVENTS[i + 1][1] - EVENTS[i][2] for i in range(len(EVENTS) - 1)]
        inside = [e[2] - e[1] for e in EVENTS]
        head = EVENTS[0][1] - t0 if EVENTS else 0.0
        tail = t0 + wall - EVENTS[-1][2] if EVENTS else 0.0
        frac = sum(e[3] for e in EVENTS) / max(1, len(EVENTS))
        print(
            f"turn {turn}: {args.chunk} tok in {wall:.2f}s over {len(EVENTS)} MoE layers, "
            f"{frac:.0f} experts/layer routed\n"
            f"  inside {sum(inside):.2f}s  gaps {sum(gaps):.2f}s "
            f"(median {sorted(gaps)[len(gaps) // 2] * 1e3:.1f} ms)  head {head:.2f}s  tail {tail:.2f}s",
            flush=True,
        )
        for _ in range(args.decode):
            logits = model(mx.argmax(logits[:, -1, :], axis=-1)[None], cache=kv)
            mx.eval(logits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
