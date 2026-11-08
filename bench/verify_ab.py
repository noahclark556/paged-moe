#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exact-verify decode vs plain decode, on an edit and on prose.

Two prompts, each decoded greedily with mlx-lm's generate_step and with
verify_decode, alternating so both see the same warm cache:

  edit   a real source file and a request to write it back with a small
         change. This is what an edit tool call looks like: long runs copied
         from the prompt. The n-gram drafter should engage and win.
  prose  the same file and a request to explain it. Little to copy, so the
         governor should keep the drafter off and tok/s should match plain.

VERIFY_SPANS is forced to "all" so the whole reply counts as a strict span,
the way an edit tool argument does in production.

Usage (Flash with its tuned env):
  PAGED_MOE_PRUNE=0.7 PAGED_MOE_FLOW=1 ... python bench/verify_ab.py \
      --model ~/mlx-models/qwen38-coder-flash --file pagedmoe/fidelity.py
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.generate import generate_step as mlx_generate_step
from mlx_lm.models.cache import make_prompt_cache

from pagedmoe import config, load, verify_decode
from pagedmoe.streaming import exact_step


def encode(tok, text: str) -> list[int]:
    msgs = [{"role": "user", "content": text}]
    for kw in ({"enable_thinking": False}, {}):
        try:
            return list(tok.apply_chat_template(msgs, add_generation_prompt=True, **kw))
        except TypeError:
            continue
    return tok.encode(text)


def prefill(model, ids, chunk=4096):
    cache = make_prompt_cache(model)
    body = ids[:-1]
    with exact_step():
        for i in range(0, len(body), chunk):
            model(mx.array(body[i : i + chunk], mx.uint32)[None], cache=cache)
            mx.eval([c.state for c in cache])
    return cache, mx.array(ids[-1:], mx.uint32)


def run(model, ids, n, verify: bool):
    cache, tail = prefill(model, ids)
    t0 = time.perf_counter()
    if verify:
        gen = verify_decode.generate_step(
            tail, model, max_tokens=n, prompt_cache=cache, history=ids[:-1]
        )
    else:
        gen = mlx_generate_step(tail, model, max_tokens=n, prompt_cache=cache)
    out = [int(t) for t, _ in gen]
    return out, len(out) / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--file", default=str(Path(__file__).resolve().parent.parent / "pagedmoe" / "fidelity.py"))
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    config.VERIFY_SPANS = "all"
    model, tok = load(args.model, verbose=False)
    src = Path(args.file).read_text()
    prompts = {
        "edit": (
            f"Here is a Python file:\n\n```python\n{src}\n```\n\n"
            "Write the whole file back unchanged except rename the function "
            "`strict` to `is_strict` everywhere. Output only the file in one "
            "code block."
        ),
        "prose": (
            f"Here is a Python file:\n\n```python\n{src}\n```\n\n"
            "In a few paragraphs, explain what problem this module solves and "
            "how its pieces fit together. Do not quote code."
        ),
    }
    for name, text in prompts.items():
        ids = encode(tok, text)
        print(f"== {name}: prompt {len(ids)} tok, {args.max_tokens} new ==", flush=True)
        run(model, ids, 32, verify=False)  # warm the expert cache for this prompt
        plain, ver = [], []
        for r in range(args.rounds):
            _, tps = run(model, ids, args.max_tokens, verify=False)
            plain.append(tps)
            out, tvs = run(model, ids, args.max_tokens, verify=True)
            ver.append(tvs)
            st = verify_decode.stats()
            print(
                f"  round {r}: plain {tps:6.2f} tok/s   verify {tvs:6.2f} tok/s   "
                f"passes={st.get('passes', 0)} exact_tok={st.get('emitted', 0)} "
                f"plain_tok={verify_decode.STATS['plain']} "
                f"accept={st.get('accept_rate', 0)} tok/pass={st.get('tokens_per_pass', 0)}",
                flush=True,
            )
        p, v = sum(plain) / len(plain), sum(ver) / len(ver)
        print(f"  mean: plain {p:.2f}  verify {v:.2f}  ({v / p:.2f}x)", flush=True)
        print(f"  sample: {tok.decode(out[:80])!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
