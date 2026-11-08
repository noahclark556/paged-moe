#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Do slab slots go missing during decode?

Every slot should be in exactly one place: free, resident, staged, or under a
read in flight. One that is none of those is gone until the next slab rebuild,
which in an agent session is rare, so a leak is cache that quietly shrinks.
Counts unowned slots after a prefill and after each block of decode.
--turns ends each block the way the server ends a response.

    python bench/slot_leak_probe.py --model ~/mlx-models/qwen38-coder-flash
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from pagedmoe import load  # noqa: E402
from pagedmoe.streaming import StreamedSwitchGLU  # noqa: E402


def cache_of(model):
    for m in model.modules():
        if isinstance(m, StreamedSwitchGLU):
            return m.cache
    raise SystemExit("not a streamed model")


def unowned(c) -> tuple[int, int]:
    time.sleep(0.2)
    with c._lock:
        owned = set(c._free_slots)
        owned |= {e["__slot__"] for e in c._lru.values() if "__slot__" in e}
        owned |= {r["__slot__"] for r in c._raw_ready.values() if "__slot__" in r}
        owned |= {c._slot_of[k] for k in c._inflight if k in c._slot_of}
        # Dropped while pinned: reclaimed when the next fetch clears the pins.
        owned |= set(getattr(c, "_unfreed", ()))
        return c.slab.slots - len(owned), c.staging_drops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", type=int, default=1024)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--per-block", type=int, default=48)
    ap.add_argument("--cache-gb", type=float, default=0.0, help="small cache = small staging, as on a low-RAM machine")
    ap.add_argument("--turns", action="store_true", help="end each block like a server turn (relieve_pressure)")
    args = ap.parse_args()

    kw = {"cache_gb": args.cache_gb} if args.cache_gb else {}
    model, tok = load(args.model, verbose=False, **kw)
    c = cache_of(model)
    text = "\n".join(p.read_text(errors="replace") for p in sorted(Path(__file__).parents[1].rglob("*.py"))[:40])
    ids = tok.encode(text)[: args.prompt]
    kv = make_prompt_cache(model)
    logits = model(mx.array(ids)[None], cache=kv)
    mx.eval(logits)
    logits = model(mx.argmax(logits[:, -1, :], axis=-1)[None], cache=kv)
    mx.eval(logits)
    lost, drops = unowned(c)
    print(f"after prefill: {c.slab.slots} slots, {lost} unowned, staging drops {drops}", flush=True)
    for b in range(1, args.blocks + 1):
        for _ in range(args.per_block):
            logits = model(mx.argmax(logits[:, -1, :], axis=-1)[None], cache=kv)
            mx.eval(logits)
        if args.turns:
            c.relieve_pressure()
        lost, drops = unowned(c)
        print(
            f"after {b * args.per_block:4} decode tokens: {lost} unowned "
            f"({lost * c._slab_per_expert / 1e9:.2f} GB), staging drops {drops}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
