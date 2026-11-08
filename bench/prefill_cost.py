#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where does an incremental prefill's wall time actually go?

The case this measures is the one that sets TTFT on an agent turn, and it is not
the cold prompt. A turn arrives with most of its prompt already in the prefix
cache and a few thousand new tokens on the end. That remainder routes to
essentially every expert, so it costs a full pass over the expert mass no matter
how short it is: on GLM-4.7, ~174 GB for under 2k tokens.

Splits the chunk into terms that can each be acted on separately:

  blocked  a layer waited on a read that had not landed yet
  copy     _materialize: host-side copy of every byte read, on this thread
  rest     attention, MoE matmuls, router syncs, Python

Runs the same chunk size twice, cold cache and warm, because a term that only
appears on the cold pass is a cache-warming cost and not a per-turn one.

    python bench/prefill_cost.py --model ~/mlx-models/glm-47-4bit \
        --context-dir ./pagedmoe --base-tokens 11700 --chunk 1944
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import make_prompt_cache

from pagedmoe import get_stats, load


def counters(model) -> tuple[float, float, float, int]:
    c = get_stats(model)["cache"]
    return (
        c["read_gb"],
        c["disk_wait_s"],
        c.get("materialize_s", 0.0),
        c.get("prefill_layers", 0),
    )


def context_text(root: Path, ) -> str:
    files = sorted(
        (p for p in root.rglob("*.py") if p.is_file()),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    return "\n".join(
        f"# ==== {p.name} ====\n{p.read_text(errors='replace')}" for p in files[:40]
    )


def decode(model, cache, logits, n: int, label: str = "") -> mx.array:
    """Greedy-decode `n` tokens on whatever residency the last prefill left.
    Prints tok/s and GB read when `label` is set."""
    r0 = get_stats(model)["cache"]["read_gb"]
    t0 = time.perf_counter()
    for _ in range(n):
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        logits = model(tok[None], cache=cache)
        mx.eval(logits)
    if label and n:
        dt = time.perf_counter() - t0
        read = get_stats(model)["cache"]["read_gb"] - r0
        print(f"{label:>14}  decode {n} tok at {n / dt:5.2f} tok/s, {read * 1e3 / n:6.0f} MB/tok")
    return logits


def run_chunk(model, cache, ids, label: str, moe_layers: int) -> mx.array:
    """Prefill `ids` as one chunk and report the split."""
    r0, w0, m0, l0 = counters(model)
    t0 = time.perf_counter()
    logits = model(mx.array(ids)[None], cache=cache)
    mx.eval(logits, [c.state for c in cache])
    elapsed = time.perf_counter() - t0
    r1, w1, m1, l1 = counters(model)
    c = get_stats(model)["cache"]
    print(
        f"{'':>14}  slab slots {c.get('slab_slots', 0)} free {c.get('slab_free', 0)}  "
        f"prefill slot reads {c.get('prefill_slot_reads', 0)} "
        f"starved {c.get('prefill_slot_starved', 0)}  "
        f"ahead {c.get('prefill_ahead_reads', 0)} used {c.get('prefill_ahead_used', 0)} "
        f"wasted {c.get('prefill_ahead_wasted_gb', 0.0):.2f} GB "
        f"skipped {c.get('prefill_ahead_skipped', 0)} (cumulative)"
    )
    read, blocked, copy = r1 - r0, w1 - w0, m1 - m0
    rest = elapsed - blocked - copy
    passes = (l1 - l0) / moe_layers if moe_layers else 0.0
    print(
        f"{label:>14}: {len(ids):5} tok in {elapsed:6.1f}s  "
        f"{read:6.1f} GB ({passes:.1f} passes, {read / max(elapsed, 1e-9):4.1f} GB/s wall)"
    )
    print(
        f"{'':>14}  blocked {blocked:5.1f}s  "
        f"copy {copy:5.1f}s ({read / max(copy, 1e-9):5.1f} GB/s)  "
        f"rest {rest:5.1f}s   "
        f"| copy+blocked = {(blocked + copy) / max(elapsed, 1e-9) * 100:.0f}% of wall"
    )
    return logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context-dir", required=True)
    ap.add_argument(
        "--base-tokens",
        type=int,
        default=11700,
        help="prompt already in the KV cache, i.e. what the prefix cache served",
    )
    ap.add_argument(
        "--chunk", type=int, default=1944, help="new tokens this turn appended"
    )
    ap.add_argument(
        "--turns",
        type=int,
        default=2,
        help="how many successive chunks to time. More than 2 matters when a "
        "learning component is in the loop (the sidecar prefill head has no "
        "history on turn 1, so a 2-turn run understates it).",
    )
    ap.add_argument(
        "--decode",
        type=int,
        default=8,
        help="greedy decode tokens after the seed and after each turn, timed. "
        "A real turn follows a decode, which rebuilds the slab a big prefill "
        "released; with 0 every turn after the seed runs slab-less, which no "
        "agent session does.",
    )
    args = ap.parse_args()

    model, tok = load(args.model, verbose=False)
    moe_layers = int(getattr(model, "_pagedmoe_info", {}).get("moe_layers") or 0)
    ids = tok.encode(context_text(Path(args.context_dir).expanduser()))
    need = args.base_tokens + (args.turns + 1) * args.chunk
    if len(ids) < need:
        ids = (ids * (need // max(len(ids), 1) + 1))[:need]
    print(f"moe_layers={moe_layers}  context={len(ids)} tokens\n")

    cache = make_prompt_cache(model)
    # The served prefix. Not the measurement, just how we get the KV to the
    # length a real turn starts from.
    t0 = time.perf_counter()
    step = 2048
    base = ids[: args.base_tokens]
    for i in range(0, len(base), step):
        logits = model(mx.array(base[i : i + step])[None], cache=cache)
        mx.eval(logits, [c.state for c in cache])
    print(f"    (seeded {len(base)} tokens of KV in {time.perf_counter() - t0:.0f}s)\n")
    decode(model, cache, logits, args.decode)

    at = args.base_tokens
    for turn in range(1, args.turns + 1):
        label = f"turn {turn}" + (" (cold)" if turn == 1 else "")
        logits = run_chunk(model, cache, ids[at : at + args.chunk], label, moe_layers)
        decode(model, cache, logits, args.decode, label)
        at += args.chunk
    return 0


if __name__ == "__main__":
    sys.exit(main())
