#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does n-gram self-draft decoding actually raise tok/s? A/B on one prompt.

Reports tok/s with and without lookup decode, plus the accept rate and
tokens-per-pass that explain the difference, and asserts the two paths emit the
same tokens (greedy, so identity is the contract - see tests/test_lookup.py).

The prompt matters more here than in most benches. Lookup drafts by finding the
recent context earlier in the buffer, so it pays off on the repetitive,
structured text this thing actually serves and does nothing on prose that never
repeats. Random token ids would measure the worst case and call it the result, so
the default prompt is source code and the model is asked to continue it.

Usage:
  PAGED_MOE_PRUNE=0.8 python bench/lookup_ab.py \
      --model ~/mlx-models/glm-47-4bit --max-tokens 96
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.generate import generate_step as mlx_generate_step
from mlx_lm.models.cache import make_prompt_cache

from pagedmoe import config, get_stats, load, lookup_decode


def read_counters(model) -> dict:
    try:
        return dict(get_stats(model)["cache"])
    except Exception:
        return {}


def cache_line(model, tokens: int, before: dict) -> str:
    """Bytes and prediction quality per decode token.

    `read_gb` and `spec_wasted_gb` are process-wide cumulative counters, so they
    have to be differenced against a snapshot taken before the run. Dividing the
    raw counter by one run's token count charges that run for every prefill the
    process ever did, which reads as read amplification that is not there.

    Prediction waste is the number that matters once the drive is saturated: a
    wasted prefetch is bandwidth taken from a read something is waiting on.
    """
    c = read_counters(model)
    if not c:
        return ""
    read = c.get("read_gb", 0.0) - before.get("read_gb", 0.0)
    wasted = c.get("spec_wasted_gb", 0.0) - before.get("spec_wasted_gb", 0.0)
    return (
        f"    read={read / max(tokens, 1):.3f} GB/tok "
        f"wasted={wasted / max(tokens, 1):.3f} GB/tok "
        f"pred_p={c['pred_precision']:.2f} pred_cov={c['pred_coverage']:.2f} "
        f"hit={c['hit_rate']:.2f}"
    )

# Repetitive but not degenerate: a run of similar function definitions, which is
# the shape of most of what a coding agent emits.
PROMPT = '''"""Vector helpers."""


def add_x(a, b):
    """Add along x."""
    return a.x + b.x


def add_y(a, b):
    """Add along y."""
    return a.y + b.y


def add_z(a, b):
    """Add along z."""
    return a.z + b.z


def sub_x(a, b):
    """Subtract along x."""
    return a.x - b.x


def sub_y(a, b):
    """Subtract along y."""
    return a.y - b.y


def sub_z(a, b):
    """Subtract along z."""
    return a.z - b.z


def mul_x(a, b):
    """Multiply along x."""
    return a.x * b.x


def mul_y(a, b):
'''


def prefill(model, cache, ids, chunk=2048):
    """Consume everything but the last token; both paths decode from there."""
    body = ids[:-1]
    for i in range(0, len(body), chunk):
        model(mx.array(body[i : i + chunk], mx.uint32)[None], cache=cache)
        mx.eval([c.state for c in cache])
    return mx.array(ids[-1:], mx.uint32)


def run_plain(model, ids, n):
    cache = make_prompt_cache(model)
    tail = prefill(model, cache, ids)
    mx.eval(tail)
    # After prefill: the reported bytes should be decode's, not the prompt's.
    before = read_counters(model)
    t0 = time.perf_counter()
    out = [
        int(tok)
        for tok, _ in mlx_generate_step(
            tail, model, max_tokens=n, prompt_cache=cache, prefill_step_size=2048
        )
    ]
    return out, time.perf_counter() - t0, before


def run_lookup(model, ids, n):
    cache = make_prompt_cache(model)
    tail = prefill(model, cache, ids)
    mx.eval(tail)
    lookup_decode.reset_stats()
    before = read_counters(model)
    t0 = time.perf_counter()
    # Seed the drafter with the prompt: the context it searches is what makes
    # lookup work, so starting from an empty history would understate it badly.
    out = [
        int(tok)
        for tok, _ in lookup_decode.generate_step(
            tail, model, max_tokens=n, prompt_cache=cache, history=ids[:-1]
        )
    ]
    return out, time.perf_counter() - t0, lookup_decode.stats(), before


def _context_from_files(root: Path) -> str:
    """Concatenate distinct source files, largest first, as agent-turn context."""
    files = sorted(
        (p for p in root.rglob("*.py") if p.is_file()),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    parts = []
    for p in files[:24]:
        parts.append(f"# ==== {p.name} ====\n{p.read_text(errors='replace')}")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--repeat-prompt", type=int, default=1)
    ap.add_argument(
        "--context-dir",
        help="build the prompt from real source files under this directory "
        "instead of repeating the built-in one. A repeated block makes n-gram "
        "matching trivially easy; agent turns are long and varied, and that is "
        "the regime where decode speed actually matters.",
    )
    ap.add_argument(
        "--context-tokens",
        type=int,
        default=12000,
        help="target prompt length when --context-dir is used",
    )
    ap.add_argument(
        "--k",
        type=int,
        action="append",
        help="draft depths to try (repeatable; default 2, 4, 6)",
    )
    args = ap.parse_args()
    depths = args.k or [2, 4, 6]

    model, tok = load(args.model, verbose=False)
    if args.context_dir:
        text = _context_from_files(Path(args.context_dir).expanduser())
        ids = tok.encode(text)[: args.context_tokens]
    else:
        ids = tok.encode(PROMPT * args.repeat_prompt)
    print(f"prompt: {len(ids)} tokens, generating {args.max_tokens}\n")

    plain, t_plain, before_plain = run_plain(model, ids, args.max_tokens)
    print(f"plain : {len(plain) / t_plain:6.2f} tok/s  ({t_plain:.2f}s)")
    print(cache_line(model, len(plain), before_plain))

    print(
        f"ngram window: {config.LOOKUP_NGRAM_MIN}..{config.LOOKUP_NGRAM_MAX}"
    )
    for k in depths:
        config.LOOKUP_TOKENS = k
        got, t_l, st, before_l = run_lookup(model, ids, args.max_tokens)
        tag = "OK" if got == plain else "DIVERGED"
        print(
            f"k={k}   : {len(got) / t_l:6.2f} tok/s  ({t_l:.2f}s)  "
            f"{len(plain) / t_plain and (len(got) / t_l) / (len(plain) / t_plain):.2f}x  "
            f"accept={st.get('accept_rate')} "
            f"tok/pass={st.get('tokens_per_pass')} "
            f"no_draft={st.get('no_draft_frac')}  [{tag}]"
        )
        print(cache_line(model, len(got), before_l))
        # Under PRUNE a k-token pass prunes on different mass than k single
        # steps, so divergence there is expected and only reported.
        if got != plain and config.PRUNE <= 0:
            print(f"  plain ={plain[:20]}")
            print(f"  lookup={got[:20]}")
            return 1
    # The first plain run decodes on a cold expert cache and every run after it
    # starts warm, so plain again is the fair baseline for the ratios above.
    again, t_again, before_again = run_plain(model, ids, args.max_tokens)
    print(
        f"plain again: {len(again) / t_again:6.2f} tok/s  ({t_again:.2f}s)  "
        "(warm cache; compare lookup against this)"
    )
    print(cache_line(model, len(again), before_again))
    return 0


if __name__ == "__main__":
    sys.exit(main())
