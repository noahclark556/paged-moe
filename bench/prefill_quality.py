# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does a long prefill corrupt the context? Greedy A/B on one knob at a time.

A prompt over PREFILL_SHRINK_TOKENS makes the cache `enter_prefill`, which with
slabs on hands the whole slab back to Metal and rebuilds it when decode resumes.
If that transition loses or mis-addresses anything, the damage lands in the KV
cache for the *whole* prompt and decode emits fluent nonsense - which is what a
13k-token request produced on the server.

Lower PREFILL_SHRINK_TOKENS to make a ~1.5k-token prompt take the same path in
seconds, then compare greedy output against a run of the same prompt that stays
under the threshold. Identical tokens clear the transition; divergence indicts
it. Run one config per process (config is read at load time):

    python bench/prefill_quality.py --model ~/mlx-models/qwen3-235-4bit \
        --slab on --shrink 512 --tokens 40
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import mlx.core as mx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pagedmoe import config as es_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--slab", choices=["on", "off", "auto"], default="auto")
    ap.add_argument("--shrink", type=int, default=None, help="PREFILL_SHRINK_TOKENS")
    ap.add_argument("--prune", type=float, default=0.0)
    ap.add_argument("--cache-gb", type=float, default=28.0)
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        help="quantize the KV cache after the prompt, as the server does (8)",
    )
    ap.add_argument("--question", default="Name the file's main class and what it stores.")
    ap.add_argument(
        "--dequantize-reuse",
        action="store_true",
        help="apply the server's fix: convert the reused quantized cache back "
        "to a plain KVCache before prefilling into it",
    )
    ap.add_argument(
        "--prefill-step",
        type=int,
        default=None,
        help="prompt tokens per prefill pass (the server uses PREFILL_CHUNK, "
        "32768, so a long prompt is one pass; mlx-lm defaults to 2048)",
    )
    ap.add_argument(
        "--warmup",
        action="store_true",
        help="decode a short prompt first, so the long prefill follows a decode "
        "(the server always does this from turn 2 on; a fresh process never does)",
    )
    ap.add_argument(
        "--reuse-quantized",
        action="store_true",
        help="prefill into a reused *quantized* cache, exactly as the server "
        "does from turn 2 on: a previous turn's final cache is quantized by "
        "decode, gets stored, and is then trimmed to the common prefix and "
        "prefilled into",
    )
    ap.add_argument(
        "--prompt-file",
        default=str(
            pathlib.Path(__file__).resolve().parents[1] / "pagedmoe" / "slab.py"
        ),
    )
    ap.add_argument("--prompt-chars", type=int, default=6000)
    args = ap.parse_args()

    es_config.SLAB = args.slab
    es_config.PRUNE = args.prune
    if args.shrink is not None:
        es_config.PREFILL_SHRINK_TOKENS = args.shrink

    from pagedmoe.loader import load  # after config is set

    model, tok = load(
        str(pathlib.Path(args.model).expanduser()),
        mode="streamed",
        cache_gb=args.cache_gb,
        verbose=False,
    )

    from mlx_lm.generate import stream_generate as _sg

    if args.warmup:
        warm = tok.apply_chat_template(
            [{"role": "user", "content": "What is an LRU cache? One sentence."}],
            add_generation_prompt=True,
        )
        n = 0
        for _ in _sg(model, tok, prompt=mx.array(warm, mx.uint32), max_tokens=24):
            n += 1
        print(f"warmup: decoded {n} tokens")

    body = pathlib.Path(args.prompt_file).read_text()[: args.prompt_chars]
    prompt = tok.apply_chat_template(
        [
            {
                "role": "user",
                "content": (
                    "Read this file:\n\n```python\n" + body + "\n```\n\n" + args.question
                ),
            }
        ],
        add_generation_prompt=True,
    )
    ids = mx.array(prompt, mx.uint32)
    n_prompt = ids.size
    crosses = n_prompt > es_config.PREFILL_SHRINK_TOKENS

    # Decode through mlx-lm's own loop so the KV-quantization schedule matches
    # what the HTTP server sets up (kv_bits=8 starting after the prompt).
    from mlx_lm.generate import stream_generate

    gen_kwargs = {}
    if args.kv_bits:
        gen_kwargs["kv_bits"] = args.kv_bits
        gen_kwargs["kv_group_size"] = 64
        gen_kwargs["quantized_kv_start"] = int(n_prompt) + 1

    if args.prefill_step:
        gen_kwargs["prefill_step_size"] = args.prefill_step

    reused_kind = None
    if args.reuse_quantized:
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

        warm_ids = mx.array(
            tok.apply_chat_template(
                [{"role": "user", "content": "What is an LRU cache? One sentence."}],
                add_generation_prompt=True,
            ),
            mx.uint32,
        )
        turn1 = make_prompt_cache(model)
        for _ in _sg(
            model,
            tok,
            prompt=warm_ids,
            max_tokens=24,
            prompt_cache=turn1,
            kv_bits=8,
            kv_group_size=64,
            quantized_kv_start=int(warm_ids.size) + 1,
        ):
            pass
        reused_kind = type(turn1[0]).__name__
        # Keep only the tokens this prompt shares, as fetch_nearest_cache does.
        keep = 15
        trim_prompt_cache(turn1, turn1[0].offset - keep)
        if args.dequantize_reuse:
            from pagedmoe.server import _dequantize_kv_cache

            _dequantize_kv_cache(turn1)
            reused_kind += f" -> {type(turn1[0]).__name__}"
        gen_kwargs["prompt_cache"] = turn1
        ids = ids[keep:]

    t0 = time.time()
    out: list[int] = []
    prefill_s = None
    for resp in stream_generate(
        model, tok, prompt=ids, max_tokens=args.tokens, **gen_kwargs
    ):
        if prefill_s is None:
            prefill_s = time.time() - t0
            t1 = time.time()
        out.append(int(resp.token))
    decode_s = time.time() - t1

    print(
        json.dumps(
            {
                "slab": args.slab,
                "shrink_tokens": es_config.PREFILL_SHRINK_TOKENS,
                "prune": args.prune,
                "prompt_tokens": int(n_prompt),
                "reused_cache": reused_kind,
                "crosses_shrink": bool(crosses),
                "prefill_s": round(prefill_s, 1),
                "decode_tok_s": round((args.tokens - 1) / max(decode_s, 1e-9), 2),
                "token_ids": out[:12],
            }
        )
    )
    print("TEXT:", tok.decode(out).replace("\n", " ")[:300])


if __name__ == "__main__":
    main()
