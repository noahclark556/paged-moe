#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Same logits with and without PREFILL_AHEAD, on a real checkpoint.

tests/test_prefill_slab.py proves it on the tiny model. This runs the same
seed + incremental turns a real session does and prints a hash of every
prefill's last-row logits and of the greedy tokens after it. Run it once per
setting and compare the lines.

    PAGED_MOE_PREFILL_AHEAD=0 python bench/ahead_identity.py --model ...
    PAGED_MOE_PREFILL_AHEAD=1 python bench/ahead_identity.py --model ...
"""

import argparse
import hashlib
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from pagedmoe import config, get_stats, load  # noqa: E402


def digest(a: mx.array) -> str:
    return hashlib.sha256(np.asarray(a.astype(mx.float32)).tobytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-tokens", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--decode", type=int, default=8)
    args = ap.parse_args()

    model, tok = load(args.model, verbose=False)
    root = Path(__file__).resolve().parent.parent / "pagedmoe"
    text = "\n".join(p.read_text(errors="replace") for p in sorted(root.rglob("*.py")))
    ids = tok.encode(text)[: args.base_tokens + args.turns * args.chunk]
    cache = make_prompt_cache(model)
    print(
        f"PREFILL_AHEAD={int(config.PREFILL_AHEAD)} "
        f"LOOSE_MB={config.PREFILL_AHEAD_LOOSE_MB}",
        flush=True,
    )
    spans = [(0, args.base_tokens)] + [
        (args.base_tokens + i * args.chunk, args.base_tokens + (i + 1) * args.chunk)
        for i in range(args.turns)
    ]
    for n, (a, b) in enumerate(spans):
        logits = model(mx.array(ids[a:b])[None], cache=cache)
        mx.eval(logits)
        pre = digest(logits[0, -1])
        out = []
        for _ in range(args.decode):
            t = mx.argmax(logits[:, -1, :], axis=-1)
            out.append(int(t.item()))
            logits = model(t[None], cache=cache)
            mx.eval(logits)
        print(f"span {n}: prefill {pre}  decode {out}", flush=True)
    c = get_stats(model)["cache"]
    print(
        f"ahead reads {c.get('prefill_ahead_reads', 0)} used {c.get('prefill_ahead_used', 0)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
