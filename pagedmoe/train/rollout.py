# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sampling completions for GRPO with the policy being trained.

Generation runs on the engine's own decode path: the streamed expert layers
go back in for the duration (the training stand-ins are swapped out), the
model drops to eval mode so mlx-lm uses its fast kernels, and the current
LoRA weights are live. All prompts x generations decode as one batch, so a
decode step reads each routed expert once for every sequence that wants it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx

from . import env as train_env
from . import experts as train_experts
from . import fastgrad, linear_attn


@dataclass
class Rollouts:
    tokens: list[list[int]]  # completion ids, one per (prompt, generation)
    finished: list[bool]  # ended on a stop token
    seconds: float
    gen_tokens: int

    @property
    def tok_s(self) -> float:
        return self.gen_tokens / self.seconds if self.seconds > 0 else 0.0


def generate(model, tokenizer, prompt_ids: list[list[int]], *, max_tokens: int,
             temperature: float = 1.0, top_p: float = 1.0, streamed: bool = False,
             batch_cap: int = 64) -> Rollouts:
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    t0 = time.perf_counter()
    if streamed:
        train_experts.uninstall(model)
    model.eval()
    chunked = linear_attn.uninstall()
    fast = fastgrad.uninstall()
    ctx = train_env.compiled()
    ctx.__enter__()
    gen = None
    try:
        sampler = make_sampler(temp=float(temperature), top_p=float(top_p))
        eos = sorted(getattr(tokenizer, "eos_token_ids", None) or [tokenizer.eos_token_id])
        n = len(prompt_ids)
        gen = BatchGenerator(
            model,
            max_tokens=max_tokens,
            stop_tokens=[[t] for t in eos],
            sampler=sampler,
            completion_batch_size=min(batch_cap, max(1, n)),
            prefill_batch_size=min(8, max(1, n)),
        )
        uids = gen.insert(prompt_ids, [max_tokens] * n)
        out = {u: [] for u in uids}
        done = {u: False for u in uids}
        while responses := gen.next_generated():
            for r in responses:
                out[r.uid].append(int(r.token))
                if r.finish_reason == "stop":
                    done[r.uid] = True
    finally:
        if gen is not None:
            gen.close()
        ctx.__exit__(None, None, None)
        if chunked:
            linear_attn.install()
        if fast:
            fastgrad.install()
        model.train()
        if streamed:
            train_experts.reinstall(model)
        mx.clear_cache()
    toks = [out[u] for u in uids]
    return Rollouts(
        tokens=toks,
        finished=[done[u] for u in uids],
        seconds=time.perf_counter() - t0,
        gen_tokens=sum(len(t) for t in toks),
    )
