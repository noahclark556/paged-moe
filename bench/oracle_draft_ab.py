#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Oracle draft probe: what if accept rate were 100%?

Answers the question draft / lookup / EAGLE all depend on, without needing a
trained drafter: if every proposed token were correct, does verifying k of them
in one MoE pass beat plain single-token decode on *this* machine?

  plain     greedy one token at a time (baseline)
  oracle-k  same tokens, verified in chunks of k (forced accept)

Reports tok/s and GB/token (post-prefill counter deltas). If oracle-k does not
beat plain, no real drafter can - rejects only make it worse. If it does, the
break-even accept rate is roughly plain/oracle of the draft portion.

Usage:
  PAGED_MOE_PRUNE=0.7 PAGED_MOE_PREDICT=0 PAGED_MOE_SIDECAR=0 \\
  python bench/oracle_draft_ab.py --model ~/mlx-models/qwen3-235-4bit \\
      --prompt-tokens 2048 --max-tokens 96 --k 2,4,6
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pagedmoe import get_stats, load


def snap(model):
    c = get_stats(model)["cache"]
    return float(c["read_gb"]), float(c.get("spec_wasted_gb", 0.0))


def prefill(model, ids, cache, chunk=2048):
    body, last = ids[:-1], ids[-1:]
    for i in range(0, len(body), chunk):
        model(mx.array(body[i : i + chunk], mx.uint32)[None], cache=cache)
        mx.eval([c.state for c in cache])
    return mx.array(last, mx.uint32)


def greedy(model, prompt_ids, n):
    cache = make_prompt_cache(model)
    y = prefill(model, prompt_ids, cache)
    mx.eval(y)
    out = []
    for tok, _ in generate_step(y, model, max_tokens=n, prompt_cache=cache, prefill_step_size=2048):
        out.append(int(tok))
    return out


def run_plain(model, prompt_ids, tokens):
    cache = make_prompt_cache(model)
    y = prefill(model, prompt_ids, cache)
    mx.eval(y)
    r0, w0 = snap(model)
    t0 = time.perf_counter()
    got = []
    for tok, _ in generate_step(
        y, model, max_tokens=len(tokens), prompt_cache=cache, prefill_step_size=2048
    ):
        got.append(int(tok))
    dt = time.perf_counter() - t0
    r1, w1 = snap(model)
    return got, dt, r1 - r0, w1 - w0


def run_oracle(model, prompt_ids, tokens, k):
    """Teacher-force the known-good continuation in chunks of k.

    One MoE forward covers k positions (input [y]+chunk[:-1]), then y advances
    to chunk[-1]. Same expert-union cost as a fully-accepted verify of k tokens,
    without a drafter and without trim. If this does not beat plain, no real
    draft can - rejects only add work.
    """
    cache = make_prompt_cache(model)
    y = int(prefill(model, prompt_ids, cache).item())
    mx.eval(mx.array([y], mx.uint32))
    r0, w0 = snap(model)
    t0 = time.perf_counter()
    i = 0
    passes = 0
    while i < len(tokens):
        take = min(k, len(tokens) - i)
        chunk = tokens[i : i + take]
        if take == 1:
            inp = mx.array([[y]], mx.uint32)
        else:
            inp = mx.array([[y] + chunk[:-1]], mx.uint32)
        logits = model(inp, cache=cache)[0]
        mx.eval(logits)
        y = chunk[-1]
        i += take
        passes += 1
    dt = time.perf_counter() - t0
    r1, w1 = snap(model)
    return dt, r1 - r0, w1 - w0, passes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--k", default="2,4,6")
    args = ap.parse_args()
    depths = [int(x) for x in args.k.split(",") if x.strip()]

    model, tok = load(args.model, verbose=False)
    # Longish, varied-ish prompt: repeat a technical paragraph, not a poem.
    text = (
        "Storage system design notes. Capacity vs bandwidth vs latency. "
        "SSD streaming of MoE experts on a 48 GB machine. "
    ) * 200
    ids = tok.encode(text)[: args.prompt_tokens]
    if len(ids) < args.prompt_tokens:
        ids = (ids * (args.prompt_tokens // max(len(ids), 1) + 1))[: args.prompt_tokens]
    print(f"prompt {len(ids)} tokens, generating {args.max_tokens}\n")

    print("collecting greedy reference...", flush=True)
    ref = greedy(model, ids, args.max_tokens)
    print(f"got {len(ref)} tokens\n")

    plain, t_p, read_p, waste_p = run_plain(model, ids, ref)
    assert plain == ref, "plain diverged from reference collection"
    print(
        f"plain     {len(plain)/t_p:6.2f} tok/s  "
        f"({t_p:.2f}s)  {read_p/len(plain):.3f} GB/tok  "
        f"wasted {waste_p:.2f} GB"
    )

    for k in depths:
        t_o, read_o, waste_o, passes = run_oracle(model, ids, ref, k)
        speed = len(ref) / t_o
        print(
            f"oracle k={k} {speed:6.2f} tok/s  "
            f"({t_o:.2f}s)  {read_o/len(ref):.3f} GB/tok  "
            f"wasted {waste_o:.2f} GB  "
            f"passes={passes}  "
            f"{speed/(len(plain)/t_p):.2f}x vs plain"
        )
        # Break-even accept: if oracle is Sx faster at 100% accept, a real
        # drafter that keeps fraction a of draft slots (and always keeps the
        # bonus token) needs roughly a such that effective rate still beats 1.
        # Rough: tokens/pass_oracle = len/passes; tokens/pass_real ~ 1 + a*(k)
        # wait more carefully: oracle emits k per pass; real emits 1+a*k on
        # average for speculative with k drafts. Cost scales with positions
        # (k+1) similar to oracle's k when bonus is included... keep it simple:
        be = t_o / t_p  # fraction of oracle speed you need ~ accept quality
        print(
            f"          break-even vs plain: need >= {be:.0%} of oracle's "
            f"effective rate (perfect accept is the ceiling)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
