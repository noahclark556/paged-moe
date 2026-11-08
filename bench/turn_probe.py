# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure the per-turn cost of an agent session, not a single generation.

The complaint this exists to diagnose is "a noticeable delay between each edit".
A single long generation does not show it: agent traffic is a long resident
prefix plus, every turn, a few hundred fresh tokens (the assistant's last
message and a tool result) followed by a short burst of decode.

That shape is the expensive one. A short prefill chunk still routes its tokens
across essentially every expert of every layer, so it reads the whole
non-resident expert mass -- the same bytes a 30k-token chunk would read. Cost is
therefore roughly constant in chunk length and set by (expert mass - cache) /
bandwidth, which is why it shows up as a fixed delay before every edit.

This replays that pattern against a real prompt cache and reports prefill and
decode separately per turn.

  python bench/turn_probe.py --model ~/mlx-models/qwen3-next --cache-gb 18
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm import stream_generate  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from pagedmoe import get_stats, load  # noqa: E402

SYSTEM = (
    "You are a coding agent. You read files, make small edits, and report what "
    "you changed. Keep responses short and concrete."
)


def filler(repo: Path, want_chars: int) -> list[str]:
    """Real source text, to stand in for tool results (file reads, diffs)."""
    out, total = [], 0
    for p in sorted(repo.rglob("*.py")):
        try:
            text = p.read_text()
        except Exception:
            continue
        if len(text) < 400:
            continue
        out.append(f"--- {p.name} ---\n{text}")
        total += len(text)
        if total > want_chars:
            break
    if not out:
        raise SystemExit(f"no source files under {repo} to use as tool results")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-gb", type=float, default=None)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--new-tokens", type=int, default=900,
                    help="fresh prompt tokens per turn (measured sessions: 609-1077)")
    ap.add_argument("--gen-tokens", type=int, default=48)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    model, tokenizer = load(args.model, mode="streamed", cache_gb=args.cache_gb)
    sampler = make_sampler(temp=0.0)
    kv = make_prompt_cache(model)
    stats = get_stats(model)["cache"]
    print(json.dumps({"label": args.label, "budget_gb": stats.get("budget_gb")}),
          flush=True)

    chunks = filler(Path(args.repo), want_chars=args.new_tokens * 4 * (args.turns + 1))
    # Chars per token differs by content; calibrate once on real text.
    per_tok = len(chunks[0]) / max(1, len(tokenizer.encode(chunks[0])))
    take = int(args.new_tokens * per_tok)

    turns = []
    prev_read = 0.0
    for t in range(args.turns + 1):
        body = chunks[t % len(chunks)][:take]
        if t == 0:
            ids = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": f"Here is the code:\n{body}\nSummarize it."}],
                add_generation_prompt=True,
            )
        else:
            # Only the fresh span: the prompt cache already holds the prefix,
            # which is exactly what the proxy does between agent turns.
            ids = tokenizer.encode(
                f"\nTool result:\n{body}\nMake one small edit and say what you changed.\n"
            )

        t0 = time.perf_counter()
        first = None
        n = 0
        for resp in stream_generate(
            model, tokenizer, ids,
            max_tokens=args.gen_tokens, sampler=sampler, prompt_cache=kv,
        ):
            if first is None:
                first = time.perf_counter()
            n += 1
        done = time.perf_counter()

        cache = get_stats(model)["cache"]
        read_gb = cache["read_gb"]
        row = {
            "turn": t,
            "prompt_tokens": len(ids),
            "prefill_s": round(first - t0, 2),
            "decode_tps": round((n - 1) / (done - first), 1) if n > 1 else 0.0,
            "read_gb_this_turn": round(read_gb - prev_read, 1),
            "hit_rate": cache["hit_rate"],
        }
        prev_read = read_gb
        # Turn 0 is the cold long prefill, not the per-edit cost being measured.
        if t:
            turns.append(row)
        print(json.dumps(row), flush=True)

    if turns:
        steady = {
            "label": args.label,
            "budget_gb": stats.get("budget_gb"),
            "turns": len(turns),
            "median_prefill_s": round(sorted(t["prefill_s"] for t in turns)[len(turns) // 2], 2),
            "mean_prefill_s": round(sum(t["prefill_s"] for t in turns) / len(turns), 2),
            "mean_decode_tps": round(sum(t["decode_tps"] for t in turns) / len(turns), 1),
            "mean_read_gb_per_turn": round(
                sum(t["read_gb_this_turn"] for t in turns) / len(turns), 1),
            "final_hit_rate": turns[-1]["hit_rate"],
        }
        print("SUMMARY " + json.dumps(steady), flush=True)


if __name__ == "__main__":
    main()
