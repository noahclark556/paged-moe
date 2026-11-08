# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure route prediction against the previous-token heuristic.

Runs one streamed generation and prints a JSON line with routing-prediction
accuracy and the throughput it bought. Configuration comes from the
PAGED_MOE_* environment (so each variant runs in a clean process - cache
state and Metal residency from a previous variant would poison the next).

  python bench/predict_probe.py --model <path> --cache-gb 1.0 --max-tokens 96

Metrics:
  pred_coverage   fraction of the experts a layer actually used that prediction
                  had already asked for
  heur_coverage   same, for "what this layer used for the previous token"  - 
                  the baseline, measured on the identical traffic
  pred_precision  fraction of predicted experts that were actually used
                  (1 - wasted speculative reads)
  disk_wait_s     time decode spent blocked on reads; the number prediction is
                  supposed to move
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from pagedmoe import get_stats, load

PROMPT = (
    "Write a Python function that reads a JSONL file, groups records by a key, "
    "and returns the group sizes sorted descending. Explain your choices."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-gb", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--label", default="")
    ap.add_argument(
        "--read-delay-us",
        type=float,
        default=0.0,
        help="add this much latency to every disk read, to stand in for a "
        "slower (external) SSD than the one this machine has. A small model "
        "whose files the drive caches is not disk-bound, so without this there "
        "is no read latency to hide and prefetching cannot be measured.",
    )
    args = ap.parse_args()

    if args.read_delay_us:
        from pagedmoe import cache as _cache_mod

        _real_read = _cache_mod.read_tensor_numpy
        _delay = args.read_delay_us / 1e6

        def _slow_read(pool, loc):
            time.sleep(_delay)
            return _real_read(pool, loc)

        _cache_mod.read_tensor_numpy = _slow_read

    model, tokenizer = load(args.model, mode="streamed", cache_gb=args.cache_gb)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True
    )

    tokens = []
    t0 = None
    for resp in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=args.max_tokens,
        sampler=make_sampler(temp=0.0),
    ):
        if t0 is None:  # start the clock after prefill
            t0 = time.perf_counter()
        tokens.append(resp.token)
    decode_s = time.perf_counter() - t0

    cache = get_stats(model)["cache"]
    out = {
        "label": args.label,
        "decode_tps": round((len(tokens) - 1) / decode_s, 2) if decode_s else 0.0,
        "tokens": len(tokens),
        "checksum": sum(tokens),
        **{
            k: cache[k]
            for k in (
                "hit_rate",
                "pred_coverage",
                "heur_coverage",
                "pred_precision",
                "pred_issued",
                "disk_wait_s",
                "read_gb",
                "spec_wasted_gb",
                "spec_skipped",
                "spec_s",
                "evictions",
                "staging_drops",
                "peak_resident_gb",
            )
            if k in cache
        },
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
