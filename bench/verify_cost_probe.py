# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What a k-position verify pass costs next to a 1-token step, warm.

Draft-and-verify only pays when a pass over k+1 positions costs much less than
k+1 single steps. On a disk-bound model the expert union sets that price
(n-gram self-draft). On a backbone-bound one (Flash) the backbone
weights are read once per pass either way, so the question is just timing.

  python bench/verify_cost_probe.py --model ~/mlx-models/qwen38-coder-flash
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from pagedmoe import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefill", type=int, default=2048)
    ap.add_argument("--warm", type=int, default=96)
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--sizes", default="1,3,5,9")
    args = ap.parse_args()

    model, tok = load(args.model, mode="streamed", verbose=False)
    root = Path(__file__).resolve().parents[1] / "pagedmoe"
    ids = tok.encode((root / "cache.py").read_text(errors="replace"))
    ids = (ids * 4)[: args.prefill + args.warm + args.reps * 20]
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([ids[: args.prefill]]), cache=cache))
    pos = args.prefill
    for _ in range(args.warm):
        mx.eval(model(mx.array([[ids[pos]]]), cache=cache))
        pos += 1

    base = None
    for n in [int(s) for s in args.sizes.split(",")]:
        times = []
        for _ in range(args.reps):
            inp = mx.array([ids[pos : pos + n]])
            t = time.perf_counter()
            mx.eval(model(inp, cache=cache))
            times.append(time.perf_counter() - t)
            pos += n
        times.sort()
        med = times[len(times) // 2]
        base = base or med
        print(
            f"N={n:2d}  median {med * 1e3:6.1f} ms  x{med / base:4.2f} of N=1  "
            f"per position {med / n * 1e3:5.1f} ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
