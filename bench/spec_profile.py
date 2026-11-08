# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Speculative-decoding profiler for streamed MoE models.

Why this exists: speculative decoding's payoff on a disk-streamed MoE is
entirely decided by two numbers that a poem prompt flatters and real agent
traffic does not:

  * **acceptance** - expected accepted tokens per round is sum(p^i for
    i in 1..k) + 1. At p=0.73 and k=4 that is 2.9 tokens/round; at p=0.5 it
    is 1.9. The round costs one verify pass over k+1 positions either way.
  * **bytes read per emitted token** - the verify pass needs each layer's
    expert *union* over k+1 positions, and rejected positions' experts are
    read for nothing.

So the bench reports both, per configuration, on a reasoning-style prompt.
Run it before trusting any speculative config:

    python bench/spec_profile.py --model ~/mlx-models/qwen3-235-4bit \
        --draft-model ~/mlx-models/qwen3-1.7b-4bit --tokens 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
from mlx_lm.generate import generate_step, speculative_generate_step
from mlx_lm.models.cache import make_prompt_cache

from pagedmoe import get_stats, load, relieve_pressure

# A planning/reasoning task: the workload where the 235B "slows down and
# thinks", which is exactly where a small draft model is weakest. Poems and
# boilerplate overstate acceptance badly.
PROMPT = """You are a senior systems engineer. Think carefully step by step.

We stream mixture-of-experts weights from an SSD so a 132 GB model runs on a
48 GB machine. Decode is disk-bound: each token demands ~8 GB of expert
weights, of which ~2 GB miss a 24 GB LRU cache. Design the highest-impact
optimization you would try next, explain the mechanism, estimate the speedup,
and name the failure mode that would make it a regression."""


def run_baseline(model, tokenizer, prompt, n_tokens, kv):
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    n = 0
    for _tok, _lp in generate_step(
        mx.array(prompt, mx.uint32), model, prompt_cache=cache, max_tokens=n_tokens, **kv
    ):
        n += 1
    return {"tokens": n, "seconds": time.perf_counter() - t0, "accept": None, "rounds": {}}


def run_spec(model, tokenizer, draft, prompt, n_tokens, num_draft, kv):
    cache = make_prompt_cache(model) + make_prompt_cache(draft)
    t0 = time.perf_counter()
    n = accepted = 0
    rounds: Counter = Counter()
    run = 0
    for _tok, _lp, from_draft in speculative_generate_step(
        mx.array(prompt, mx.uint32),
        model,
        draft,
        num_draft_tokens=num_draft,
        max_tokens=n_tokens,
        prompt_cache=cache,
        **kv,
    ):
        n += 1
        if from_draft:
            accepted += 1
            run += 1
        else:
            # sgs emits the accepted drafts, then one target-sampled token:
            # that token closes the round.
            rounds[run] += 1
            run = 0
    return {
        "tokens": n,
        "seconds": time.perf_counter() - t0,
        "accept": accepted / n if n else 0.0,
        "rounds": dict(sorted(rounds.items())),
    }


def report(label, res, stats_before, stats_after):
    tps = res["tokens"] / res["seconds"] if res["seconds"] else 0.0
    read = stats_after["read_gb"] - stats_before["read_gb"]
    wasted = stats_after["spec_wasted_gb"] - stats_before["spec_wasted_gb"]
    per_tok = read / res["tokens"] if res["tokens"] else 0.0
    line = (
        f"{label:<22} {tps:5.2f} tok/s  "
        f"read {read:7.1f} GB ({per_tok:4.2f} GB/tok)  "
        f"wasted {wasted:6.1f} GB  "
        f"hit {stats_after['hit_rate']:.3f}"
    )
    if res["accept"] is not None:
        line += f"  accept {res['accept']:.0%}  rounds {res['rounds']}"
    print(line, flush=True)
    return tps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft-model", default=None)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--cache-gb", type=float, default=None)
    ap.add_argument(
        "--draft-tokens",
        default="2,4",
        help="comma-separated num_draft_tokens values to sweep",
    )
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()

    model, tokenizer = load(args.model, cache_gb=args.cache_gb, verbose=True)
    streamed = getattr(model, "_pagedmoe_cache", None) is not None
    kv = {"kv_bits": 8, "kv_group_size": 64} if streamed else {}

    messages = [{"role": "user", "content": PROMPT}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    print(f"prompt: {len(prompt)} tokens, decoding {args.tokens}\n", flush=True)

    def snapshot():
        return get_stats(model)["cache"] if streamed else {
            "read_gb": 0.0, "spec_wasted_gb": 0.0, "hit_rate": 0.0
        }

    # Between configurations, return staging + Metal pages exactly like the
    # server does between requests. Without this, configs measured later in
    # the sweep run against a degraded allocator and lose for reasons that
    # have nothing to do with speculation - which is how an early version of
    # this bench "proved" that speculation was a 2x regression.
    results = {}
    if not args.no_baseline:
        before = snapshot()
        res = run_baseline(model, tokenizer, prompt, args.tokens, kv)
        results["no draft"] = report("no draft", res, before, snapshot())
        relieve_pressure(model)

    if args.draft_model:
        draft, _ = load(args.draft_model, verbose=True)
        for k in [int(x) for x in args.draft_tokens.split(",") if x.strip()]:
            before = snapshot()
            res = run_spec(model, tokenizer, draft, prompt, args.tokens, k, kv)
            results[f"draft k={k}"] = report(f"draft k={k}", res, before, snapshot())
            relieve_pressure(model)

    print("\n" + json.dumps({k: round(v, 2) for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
