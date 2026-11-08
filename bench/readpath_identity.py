# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Greedy token ids from the streamed path, for bit-identity A/B.

The pooled slot-read path (ExpertCache._ReadPool / native C pool) changes
*when* bytes land, never which bytes, so decode has to emit exactly the same
token sequence as the ThreadPoolExecutor path it replaces. Run this once per
variant and diff:

  PAGED_MOE_READ_POOL_THREADS=-1 python bench/readpath_identity.py -m ... > old
  PAGED_MOE_NATIVE_READ=0        python bench/readpath_identity.py -m ... > py
  PAGED_MOE_NATIVE_READ=1        python bench/readpath_identity.py -m ... > native
  diff old py && diff py native

Greedy sampling only - anything temperature-driven would differ for reasons
that have nothing to do with the read path.
"""

from __future__ import annotations

import argparse
import hashlib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    args = ap.parse_args()

    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from pagedmoe.loader import load

    model, tokenizer = load(args.model)

    para = (
        "The design of storage systems involves fundamental tradeoffs between "
        "capacity, bandwidth, and latency. Solid state drives offer high "
        "sequential throughput but random access patterns reveal their "
        "internal parallelism limits. Memory hierarchies exploit locality. "
    )
    ids = tokenizer.encode(para * 400)[: args.prompt_tokens]
    msgs = [
        {"role": "user", "content": tokenizer.decode(ids) + "\n\nSummarize the above."}
    ]
    prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)

    out = [
        r.token
        for r in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=make_sampler(temp=0.0),
        )
    ]
    print("tokens:", out)
    print("sha256:", hashlib.sha256(repr(out).encode()).hexdigest())
    print("text:", repr(tokenizer.decode(out)))


if __name__ == "__main__":
    main()
