#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
PagedMoE CLI: generate text or chat with a (possibly bigger-than-RAM)
MoE model.

    python run.py --model mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit \
                  --prompt "Explain expert streaming in two sentences."

    python run.py --model <path-or-repo> --chat          # interactive REPL
    python run.py --model <path-or-repo> --prompt ... --stats

Useful flags: --mode resident|streamed, --cache-gb N, --max-tokens N.
For an HTTP API instead, use: python -m expert_stream.server --model ...
"""

import argparse
import json
import sys

from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from expert_stream import config, get_stats, load, relieve_pressure


def build_prompt(tokenizer, messages):
    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
    return tokenizer.encode(messages[-1]["content"])


def generate(
    model,
    tokenizer,
    messages,
    max_tokens,
    temp,
    prefill_step=None,
    draft_model=None,
    draft_tokens=3,
):
    prompt = build_prompt(tokenizer, messages)
    sampler = make_sampler(temp=temp)
    kwargs = {}
    if getattr(model, "_expert_stream_cache", None) is not None:
        # Streamed models are memory-starved by definition: keep the KV cache
        # at 8-bit (≈half the fp16 footprint, no meaningful quality loss).
        kwargs = {"kv_bits": 8, "kv_group_size": 64}
        # Each prefill chunk activates ~every expert, so it reads ~the whole
        # uncached expert mass. Fewer, bigger chunks = fewer passes over disk
        # (the cache shrinks itself during prefill to pay for the activations).
        kwargs["prefill_step_size"] = prefill_step or config.PREFILL_CHUNK
    elif prefill_step:
        kwargs["prefill_step_size"] = prefill_step
    if draft_model is not None:
        kwargs["draft_model"] = draft_model
        kwargs["num_draft_tokens"] = draft_tokens
    text = ""
    last = None
    accepted = 0
    total = 0
    for chunk in stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler, **kwargs
    ):
        print(chunk.text, end="", flush=True)
        text += chunk.text
        total += 1
        accepted += bool(getattr(chunk, "from_draft", False))
        last = chunk
    print()
    if draft_model is not None and total:
        print(
            f"[draft acceptance: {accepted}/{total} = {accepted/total:.0%}]",
            file=sys.stderr,
        )
    return text, last


def main():
    ap = argparse.ArgumentParser(description="PagedMoE text generation")
    ap.add_argument("--model", required=True, help="local path or HF repo id")
    ap.add_argument("--prompt", help="one-shot prompt")
    ap.add_argument("--chat", action="store_true", help="interactive chat REPL")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--mode", choices=["auto", "resident", "streamed"], default=None)
    ap.add_argument("--cache-gb", type=float, default=None)
    ap.add_argument("--stats", action="store_true", help="print cache/memory stats")
    ap.add_argument(
        "--prefill-step",
        type=int,
        default=None,
        help="prompt tokens per prefill pass (bigger = fewer passes over the expert mass)",
    )
    ap.add_argument(
        "--draft-model",
        default=None,
        help="small same-tokenizer model for speculative decoding "
        "(e.g. ~/mlx-models/qwen3-0.6b-8bit). Greedy output is identical; "
        "decode gets faster whenever the draft guesses right.",
    )
    ap.add_argument(
        "--draft-tokens",
        type=int,
        default=3,
        help="draft tokens per speculation round (with --draft-model)",
    )
    args = ap.parse_args()

    if not args.prompt and not args.chat:
        ap.error("need --prompt or --chat")

    model, tokenizer = load(
        args.model, mode=args.mode, cache_gb=args.cache_gb, verbose=True
    )

    draft_model = None
    if args.draft_model:
        draft_model, _draft_tok = load(args.draft_model, verbose=True)

    if args.prompt:
        messages = [{"role": "user", "content": args.prompt}]
        _, last = generate(
            model,
            tokenizer,
            messages,
            args.max_tokens,
            args.temp,
            args.prefill_step,
            draft_model=draft_model,
            draft_tokens=args.draft_tokens,
        )
        if last is not None:
            print(
                f"\n[{last.prompt_tokens} prompt tok @ "
                f"{last.prompt_tps:.1f} tok/s | {last.generation_tokens} gen tok @ "
                f"{last.generation_tps:.1f} tok/s | peak mem {last.peak_memory:.1f} GB]",
                file=sys.stderr,
            )
        if args.stats:
            print(json.dumps(get_stats(model), indent=2), file=sys.stderr)
        relieve_pressure(model)
        return

    # chat REPL
    messages = []
    print("chat mode - empty line or Ctrl-D to exit")
    while True:
        try:
            user = input("\nyou> ").strip()
        except EOFError:
            break
        if not user:
            break
        messages.append({"role": "user", "content": user})
        print("model> ", end="")
        text, last = generate(
            model,
            tokenizer,
            messages,
            args.max_tokens,
            args.temp,
            args.prefill_step,
            draft_model=draft_model,
            draft_tokens=args.draft_tokens,
        )
        messages.append({"role": "assistant", "content": text})
        if last is not None:
            print(
                f"[{last.generation_tps:.1f} tok/s | cache hit "
                f"{get_stats(model).get('cache', {}).get('hit_rate', 0):.0%}]",
                file=sys.stderr,
            )
        relieve_pressure(model)
        if args.stats:
            print(json.dumps(get_stats(model).get("cache", {})), file=sys.stderr)


if __name__ == "__main__":
    main()
