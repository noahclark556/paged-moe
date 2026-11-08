# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Replay a multi-turn agent session through the real serving path and watch
unified memory.

This exists because the failure it measures is invisible to every other bench
here: a single generation, or a session that reuses one KV cache object, never
touches mlx-lm's prompt-cache store - and the store is where GLM-4.7 died.
Turn two fetches the stored cache (a whole extra copy), dequantizes it to fp16
for prefill (another, at ~2x), re-quantizes it afterwards (both formats live at
once), and stores the result next to the entry it came from. Four copies of a
4.8 GB cache, against a committed expert slab, on a 35 GB budget.

So: drive `stream_generate` exactly as the HTTP server does - same patches
(`server.install_patches`), same store, same quantization thresholds, same
snapshotting - and report what Metal is holding at each stage of each turn.

  python bench/session_memory.py --model ~/mlx-models/glm-47-4bit --turns 4
  python bench/session_memory.py --model ~/mlx-models/glm-47-4bit --no-fixes

`--no-fixes` restores upstream's copy-and-batch-convert behavior, which is how
the before/after numbers in docs/improvements.md were produced. Expect it to
abort the process on a fat-KV model: an over-budget Metal command buffer fails
asynchronously, so there is nothing to catch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402
import mlx_lm.server as _mlx_server  # noqa: E402
from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from pagedmoe import config, get_stats, kvmem, load, server  # noqa: E402

SYSTEM = (
    "You are a coding agent. You read files, make small edits, and report what "
    "you changed. Keep responses short and concrete."
)


def source_chunks(repo: Path, want_chars: int) -> list[str]:
    """Real source text, standing in for the tool results an agent turn appends."""
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


def mem() -> dict:
    return {
        "active_gb": round(mx.get_active_memory() / 1e9, 2),
        "peak_gb": round(mx.get_peak_memory() / 1e9, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache-gb", type=float, default=None)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument(
        "--first-tokens",
        type=int,
        default=12000,
        help="opening prompt length; the crash reproduced at ~13.4k",
    )
    ap.add_argument("--new-tokens", type=int, default=700,
                    help="fresh prompt tokens per later turn (sessions: 609-1077)")
    ap.add_argument("--gen-tokens", type=int, default=32)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--no-fixes", action="store_true",
                    help="restore mlx-lm's copying fetch and batched KV conversion")
    ap.add_argument(
        "--drop-reply",
        action="store_true",
        help="next turn omits the assistant's reply, as a client that does not "
        "replay reasoning verbatim does. This is what makes the stored cache a "
        "*sibling* of the next prompt rather than its prefix, so insert_cache's "
        "prefix eviction cannot collect it and the store accumulates whole KV "
        "caches - the shape that Metal-OOM'd GLM-4.7.",
    )
    ap.add_argument(
        "--ballast-gb",
        type=float,
        default=0.0,
        help="Metal memory held from after load to the end, like the server's "
        "pinned front-matter prefix (2.5 GB on qwen3-235b at 20k numCtx)",
    )
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    if args.no_fixes:
        config.PROMPT_CACHE_MOVE = False

    model, tokenizer = load(args.model, mode="streamed", cache_gb=args.cache_gb)
    server.install_patches(model)
    if args.no_fixes:
        import mlx_lm.generate as _mlx_generate
        from mlx_lm.generate import maybe_quantize_kv_cache

        _mlx_generate.maybe_quantize_kv_cache = maybe_quantize_kv_cache
        kvmem.set_store_budget(0)

    ballast = None
    if args.ballast_gb > 0:
        ballast = mx.zeros((max(1, int(args.ballast_gb * 1e3)), 1 << 20), dtype=mx.uint8)
        mx.eval(ballast)

    info = dict(model._pagedmoe_info)
    print(
        "CONFIG "
        + json.dumps(
            {
                "label": args.label,
                "fixes": not args.no_fixes,
                "cache_gb": info.get("cache_gb"),
                "slab_gb": info.get("slab_gb"),
                "backbone_gb": info.get("backbone_gb"),
                "kv_reserve_gb": info.get("kv_reserve_gb"),
                "prompt_cache_gb": info.get("prompt_cache_gb"),
                "ram_budget_gb": info.get("ram_budget_gb"),
            }
        ),
        flush=True,
    )

    sampler = make_sampler(temp=0.0)
    # The store the HTTP server would have built, with the flags it injects.
    store = LRUPromptCache(max_size=6)

    chunks = source_chunks(Path(args.repo), want_chars=args.first_tokens * 5)
    per_tok = len(chunks[0]) / max(1, len(tokenizer.encode(chunks[0])))

    # Turn 0: one long prompt, as a fresh agent session sends.
    body = "".join(chunks)[: int(args.first_tokens * per_tok)]
    tokens = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Here is the code:\n{body}\nSummarize it."},
        ],
        add_generation_prompt=True,
    )

    rows = []
    prev_read = 0.0
    for turn in range(args.turns):
        mx.reset_peak_memory()
        before = mem()

        t0 = time.perf_counter()
        # Exactly what mlx-lm's ResponseGenerator does per request.
        cache, rest = store.fetch_nearest_cache("m", tokens)
        reused = len(tokens) - len(rest)
        fetched = mem()
        t_fetch = time.perf_counter() - t0

        if cache is None:
            cache = make_prompt_cache(model)

        first = None
        n = 0
        text = ""
        t1 = time.perf_counter()
        for resp in _mlx_server.stream_generate(
            model,
            tokenizer,
            prompt=rest,
            max_tokens=args.gen_tokens,
            sampler=sampler,
            prompt_cache=cache,
            # The server injects this as --prefill-step-size. Without it mlx-lm
            # defaults to 2048, and since a prefill chunk reads ~the whole
            # expert mass regardless of its length, a 12k prompt would make six
            # passes over 189 GB instead of one - a benchmark measuring a
            # configuration nothing runs.
            prefill_step_size=config.PREFILL_CHUNK,
        ):
            if first is None:
                first = time.perf_counter()
            n += 1
            text += resp.text
        done = time.perf_counter()
        generated = tokenizer.encode(text) if text else []

        peak_gen = mem()
        store.insert_cache("m", list(tokens) + list(generated), cache)
        stored = mem()

        stats = get_stats(model)["cache"]
        row = {
            "turn": turn,
            "prompt_tokens": len(tokens),
            "reused": reused,
            "prefilled": len(rest),
            "fetch_s": round(t_fetch, 2),
            "prefill_s": round(first - t1, 1) if first else 0.0,
            "decode_tps": round((n - 1) / (done - first), 2) if n > 1 else 0.0,
            "read_gb": round(stats["read_gb"] - prev_read, 1),
            "slab_slots": stats.get("slab_slots", 0),
            "gen_sha": hashlib.sha1(text.encode()).hexdigest()[:12],
            "store_gb": round(store.nbytes / 1e9, 2),
            "store_seqs": len(store),
            "mem_before": before,
            "mem_after_fetch": fetched,
            "mem_after_gen": peak_gen,
            "mem_after_store": stored,
        }
        prev_read = stats["read_gb"]
        rows.append(row)
        print(json.dumps(row), flush=True)

        # Next turn: the assistant's reply plus a fresh tool result, which is
        # the shape that makes the stored prefix reusable.
        body = chunks[(turn + 1) % len(chunks)][: int(args.new_tokens * per_tok)]
        tokens = (
            list(tokens)
            + ([] if args.drop_reply else list(generated))
            + tokenizer.encode(
                f"\nTool result:\n{body}\nMake one small edit and say what you changed.\n"
            )
        )

    steady = [r for r in rows[1:]]
    print(
        "SUMMARY "
        + json.dumps(
            {
                "label": args.label,
                "fixes": not args.no_fixes,
                "budget_gb": info.get("ram_budget_gb"),
                "worst_peak_gb": max(r["mem_after_gen"]["peak_gb"] for r in rows),
                "worst_store_gb": max(r["store_gb"] for r in rows),
                "mean_decode_tps": (
                    round(sum(r["decode_tps"] for r in steady) / len(steady), 2)
                    if steady
                    else 0.0
                ),
                "mean_prefill_s": (
                    round(sum(r["prefill_s"] for r in steady) / len(steady), 1)
                    if steady
                    else 0.0
                ),
                "mean_fetch_s": (
                    round(sum(r["fetch_s"] for r in steady) / len(steady), 2)
                    if steady
                    else 0.0
                ),
                "reuse_ok": all(r["reused"] > 0 for r in steady),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
