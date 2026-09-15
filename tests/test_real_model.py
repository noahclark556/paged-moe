# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
End-to-end test on a real MoE model (OLMoE-1B-7B, 4-bit, 64 experts/layer,
top-8 routing, ~4 GB on disk).

Downloads on first run (into EXPERT_STREAM_MODELS_DIR), then:
  1. resident generation (reference)
  2. streamed generation with a cache too small for the full expert mass,
     so real evictions + disk traffic happen
and asserts the generated token ids are identical. Greedy sampling, so any
divergence at all is a bug.

    python tests/test_real_model.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from expert_stream import get_stats, load

MODEL = "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit"
PROMPT = "Briefly explain what a mixture-of-experts model is."
MAX_TOKENS = 60


def generate(model, tokenizer):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=True,
    )
    sampler = make_sampler(temp=0.0)  # greedy -> deterministic
    tokens, text, last = [], "", None
    t0 = time.perf_counter()
    for chunk in stream_generate(
        model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=sampler
    ):
        tokens.append(chunk.token)
        text += chunk.text
        last = chunk
    dt = time.perf_counter() - t0
    return tokens, text, last, dt


def main():
    print("== resident (reference) ==")
    model, tokenizer = load(MODEL, mode="resident", verbose=True)
    ref_tokens, ref_text, last, dt = generate(model, tokenizer)
    print(f"text: {ref_text!r}")
    print(f"[{last.generation_tps:.1f} gen tok/s, total {dt:.1f}s]")
    del model

    print("\n== streamed (1.5 GB cache, expert mass is ~3.5 GB) ==")
    model, tokenizer = load(MODEL, mode="streamed", cache_gb=1.5, verbose=True)
    got_tokens, got_text, last, dt = generate(model, tokenizer)
    print(f"text: {got_text!r}")
    print(f"[{last.generation_tps:.1f} gen tok/s, total {dt:.1f}s]")
    stats = get_stats(model)
    print("stats:", json.dumps(stats, indent=2))

    assert got_tokens == ref_tokens, (
        f"streamed output diverged\n  ref: {ref_text!r}\n  got: {got_text!r}"
    )
    assert stats["cache"]["evictions"] > 0, "cache should have been under pressure"
    print("\nOK: streamed output identical to resident output under real eviction")


if __name__ == "__main__":
    main()
