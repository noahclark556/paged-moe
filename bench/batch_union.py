#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Do k tokens cost k times the expert bytes, or less?

Decode on a big checkpoint is disk-bandwidth-bound: measured tok/s times
measured GB/token comes out at the drive's rate no matter how hard PRUNE is
turned up. Under a bandwidth ceiling the only thing that raises tok/s is
raising *tokens per byte read*, and the way to do that is to put several tokens
through one pass so they share the expert reads they have in common.

This reads a decode demand trace (`decode_profile.py --trace`) and reports, for
each window size k, the size of the per-layer expert union over k consecutive
tokens divided by k. That ratio is the bytes/token a perfect k-token verify
pass would pay, so 1/ratio is the ceiling on the speedup available from
batching k tokens - before any draft accept rate is applied.

Usage:
  python bench/batch_union.py /tmp/glm_trace.json --expert-mb 13.3
"""

import argparse
import json
from collections import defaultdict


def load_tokens(path):
    """Regroup a flat [(layer, ids), ...] trace into per-token layer maps.

    The trace is emitted in call order, so a layer repeating marks the start of
    the next token. That is more robust than assuming a layer count, since
    prefill and decode do not touch the same number of layers.
    """
    with open(path) as fh:
        raw = json.load(fh)

    tokens, cur, seen = [], defaultdict(set), set()
    for layer, ids in raw:
        if layer in seen:
            tokens.append(cur)
            cur, seen = defaultdict(set), set()
        seen.add(layer)
        cur[layer].update(int(e) for e in ids)
    if cur:
        tokens.append(cur)
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument(
        "--expert-mb",
        type=float,
        default=13.3,
        help="bytes per expert, for a GB/token column (GLM-4.7 4-bit: 13.3)",
    )
    ap.add_argument("--max-k", type=int, default=16)
    ap.add_argument(
        "--bandwidth",
        type=float,
        default=5.8,
        help="measured effective read bandwidth in GB/s",
    )
    args = ap.parse_args()

    tokens = load_tokens(args.trace)
    # The first token after prefill routes differently (and its reads are
    # already paid for by the prompt), so leave it out of the averages.
    tokens = tokens[1:]
    if len(tokens) < 4:
        raise SystemExit(f"trace has only {len(tokens)} decode tokens")

    per_tok = sum(len(v) for t in tokens for v in t.values()) / len(tokens)
    print(f"{len(tokens)} decode tokens, {per_tok:.0f} expert reads/token")
    print(f"assuming {args.expert_mb} MB/expert and {args.bandwidth} GB/s\n")
    print(f"{'k':>3}  {'experts/token':>13}  {'vs k=1':>7}  {'GB/token':>8}  {'tok/s ceiling':>13}")

    base = None
    for k in range(1, min(args.max_k, len(tokens)) + 1):
        # Average over every window of k, so the number is not an artifact of
        # where the windows happen to land.
        totals = []
        for start in range(len(tokens) - k + 1):
            union = defaultdict(set)
            for t in tokens[start : start + k]:
                for layer, ids in t.items():
                    union[layer].update(ids)
            totals.append(sum(len(v) for v in union.values()))
        per = sum(totals) / len(totals) / k
        if base is None:
            base = per
        gb = per * args.expert_mb / 1024
        print(
            f"{k:>3}  {per:>13.1f}  {per / base:>6.2f}x  {gb:>8.3f}  "
            f"{args.bandwidth / gb:>12.1f}"
        )


if __name__ == "__main__":
    main()
