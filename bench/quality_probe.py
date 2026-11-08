# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Measure what a decode-time approximation (PRUNE / TOP_K) costs in quality.

Why this exists: PRUNE and TOP_K only apply to *decode* (prefill computes the
full mixture so prompt states stay exact). A normal teacher-forced perplexity
run pushes the whole sequence through in one call - that is the prefill path,
so it would report zero difference no matter how aggressive the settings are.
This probe therefore walks the text one token at a time through the decode
path, which is what a served token actually pays.

Two numbers per config, both on identical inputs so configs are comparable:

  nll        mean -log P(actual next token). The information-theoretic cost;
             lower is better. Comparable across configs because the context at
             every step is the *real* text (teacher forcing), not the model's
             own output, so no divergence-amplification.
  agree      fraction of positions where argmax matches the reference config
             (default: no prune, no cap). This is the interpretable one --
             "how often would this config have picked a different token".

Usage (one config per process; settings come from the environment):

    PAGED_MOE_PRUNE=0.5 PAGED_MOE_ROUTE_TOP_K=4 \
      python bench/quality_probe.py --model ~/mlx-models/qwen3-235-4bit \
        --tokens 192 --save ref.json

    # then compare another config against that reference
    ... python bench/quality_probe.py ... --ref ref.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import make_prompt_cache

from pagedmoe import config, get_stats, load

# Mixed content on purpose: prose, code, and structured output. A setting that
# only hurts one of those is worth knowing about, and an average over a single
# domain hides it.
TEXT = '''The expert cache keeps recently used experts in unified memory, and
the router decides which experts a token needs. When the cache misses, the
weights are read from the SSD, which is the slowest step in the whole pipeline.

def merge_intervals(intervals):
    """Merge overlapping intervals and return them sorted by start."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda pair: pair[0])
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(pair) for pair in merged]

To call the tool, emit exactly one JSON object and nothing else:

{"name": "search_files", "arguments": {"pattern": "*.safetensors", "limit": 20}}

The tradeoff is straightforward: dropping a weak expert saves a disk read, but
the mixture it produces is no longer the one the model was trained to compute.
Whether that matters depends on how much weight the dropped expert carried.
Because the router normalizes over the experts it selected, a token whose
routing is confident loses almost nothing, while an ambiguous token can shift
enough to change the next word entirely.'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=192,
                    help="decode steps to score")
    ap.add_argument("--prefill", type=int, default=48,
                    help="tokens of context established via prefill first")
    ap.add_argument("--warmup", type=int, default=0,
                    help="decode steps run (teacher-forced) but not scored "
                    "before the scored ones. Residency-dependent settings "
                    "(FLOW, KEEP_FREE) look worse on a cold cache than they "
                    "run in a long turn; this measures the steady state")
    ap.add_argument("--text-file", default=None,
                    help="score this file instead of the built-in text. The "
                    "built-in one repeats to fill long runs, and repeated text "
                    "is easy to predict from context")
    ap.add_argument("--cache-gb", type=float, default=None)
    ap.add_argument("--ref", help="reference json to compare argmax against")
    ap.add_argument("--save", help="write this run's results (use as --ref)")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    label = args.label or f"prune={config.PRUNE} top_k={config.ROUTE_TOP_K or 'off'}"
    model, tokenizer = load(
        args.model, mode="streamed", cache_gb=args.cache_gb, verbose=False
    )

    text = Path(args.text_file).read_text(errors="replace") if args.text_file else TEXT
    ids = tokenizer.encode(text)
    need = args.prefill + args.warmup + args.tokens + 1
    if len(ids) < need:
        # Repeat rather than truncate the request: a short text should still
        # produce the requested number of decode steps.
        reps = -(-need // len(ids))
        ids = (ids * reps)[:need]
    ids = ids[:need]

    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = model(mx.array([ids[: args.prefill]]), cache=cache)
    mx.eval(logits)
    for i in range(args.prefill, args.prefill + args.warmup):
        logits = model(mx.array([[ids[i]]]), cache=cache)
        mx.eval(logits)
    start = args.prefill + args.warmup
    t_scored = time.perf_counter()

    nlls: list[float] = []
    picks: list[int] = []
    for i in range(start, start + args.tokens):
        row = logits[0, -1, :].astype(mx.float32)
        logp = row - mx.logsumexp(row)
        target = ids[i]
        nlls.append(-float(logp[target]))
        picks.append(int(mx.argmax(row)))
        # Teacher forcing: feed the REAL token, so every config sees the same
        # context and the comparison isn't contaminated by drift.
        logits = model(mx.array([[target]]), cache=cache)
    mx.eval(logits)
    scored_s = time.perf_counter() - t_scored
    dt = time.perf_counter() - t0

    nll = float(np.mean(nlls))
    out = {
        "label": label,
        "prune": config.PRUNE,
        "route_top_k": config.ROUTE_TOP_K,
        "route_renorm": config.ROUTE_RENORM,
        "tokens": args.tokens,
        "nll": round(nll, 5),
        "ppl": round(float(np.exp(nll)), 4),
        "picks": picks,
        "elapsed_s": round(dt, 2),
        # Scored steps only, each with a host sync for the NLL: comparable
        # between configs, lower than a served decode loop.
        "scored_tok_s": round(args.tokens / max(scored_s, 1e-9), 2),
    }
    cstats = (get_stats(model) or {}).get("cache", {})
    for k in ("pruned_frac", "crit_per_layer", "mass_sync", "mass_flow"):
        if k in cstats:
            out[k] = cstats[k]

    if args.ref:
        ref = json.loads(Path(args.ref).read_text())
        rp = ref["picks"][: len(picks)]
        agree = sum(int(a == b) for a, b in zip(rp, picks)) / len(rp)
        out["ref_label"] = ref["label"]
        out["agree"] = round(agree, 4)
        out["nll_delta"] = round(nll - ref["nll"], 5)

    if args.save:
        Path(args.save).write_text(json.dumps(out))

    slim = {k: v for k, v in out.items() if k != "picks"}
    print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
