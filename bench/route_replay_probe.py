# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How alike do a text and its copy route? (route_log replay ceiling)

Prefills a source text twice in one prompt and compares, per layer, the
recorded experts at each position of the first copy with the same token's
position in the second. Overlap of the top-m sets (m = kept per row at the
app's PRUNE) is what a replayed plan could hope to name right; it says
nothing about residency, so it is an upper bound on what the plan buys.

  python bench/route_replay_probe.py --model ~/mlx-models/qwen38-coder-flash
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from pagedmoe import load, streaming  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=600)
    ap.add_argument("--m", type=int, default=0, help="experts per row to compare (0 = all)")
    args = ap.parse_args()

    model, tok = load(args.model, mode="streamed", verbose=False)
    log = model._pagedmoe_routes
    src = (Path(__file__).resolve().parents[1] / "pagedmoe" / "slab.py").read_text()
    ids = tok.encode(src, add_special_tokens=False)[: args.tokens]
    prompt = ids + ids
    cache = make_prompt_cache(model)
    with streaming.exact_step(), streaming.cache_only():
        model(mx.array([prompt]), cache=cache)
        mx.eval([c.state for c in cache])
    n = len(ids)
    a = np.arange(8, n)  # skip the first few: their context differs most
    b = a + n
    ok = (log.pos[a % log.cap] == a) & (log.pos[b % log.cap] == b)
    print(f"positions recorded: {ok.mean():.3f} of {len(a)}; tokens match: "
          f"{(log.tok[a % log.cap] == log.tok[b % log.cap]).mean():.3f}")
    m = args.m or log.k
    over = []
    for li in range(len(log.layer_keys)):
        ra = log.ids[a % log.cap, li, :m]
        rb = log.ids[b % log.cap, li, :m]
        same = [len(set(x) & set(y)) / m for x, y in zip(ra.tolist(), rb.tolist())]
        over.append(float(np.mean(same)))
    over = np.array(over)
    q = len(over) // 4 or 1
    print(f"top-{m} overlap, copy vs source: mean {over.mean():.3f}  "
          f"first quarter {over[:q].mean():.3f}  middle {over[q:-q].mean():.3f}  "
          f"last quarter {over[-q:].mean():.3f}")
    print("per layer:", " ".join(f"{v:.2f}" for v in over))


if __name__ == "__main__":
    main()
