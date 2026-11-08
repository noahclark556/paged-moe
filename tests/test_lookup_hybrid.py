# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Lookup decode on a hybrid (GatedDeltaNet + attention) cache.

The recurrent layers keep an ArraysCache that cannot be trimmed, so a rejected
draft used to mean no drafting at all on these models. Now the recurrent state
is snapshotted before the verify pass and restored on a partial accept, the
attention KV is trimmed back, and the kept tokens are run again.

Two checks. A replayed cache must be bit-identical to one that was only ever
fed the kept tokens. And every token lookup emits must be the model's greedy
pick given what came before, within a small margin: a multi-token pass through
the recurrent layers does not round exactly like one-token steps (logits move
~0.02 in fp32 here), so an exact match against plain decoding would test the
kernels rather than the rewind.

Run:
    python tests/test_lookup_hybrid.py
"""

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models import qwen3_next  # noqa: E402
from mlx_lm.models.cache import (  # noqa: E402
    _BaseCache,
    can_trim_prompt_cache,
    make_prompt_cache,
)

from pagedmoe import config as es_config  # noqa: E402
from pagedmoe import lookup_decode  # noqa: E402

ARGS = dict(
    model_type="qwen3_next",
    hidden_size=64,
    num_hidden_layers=4,
    intermediate_size=64,
    num_attention_heads=4,
    linear_num_value_heads=4,
    linear_num_key_heads=2,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    num_experts=4,
    num_experts_per_tok=2,
    decoder_sparse_step=1,
    shared_expert_intermediate_size=32,
    mlp_only_layers=[],
    moe_intermediate_size=32,
    rms_norm_eps=1e-6,
    vocab_size=96,
    num_key_value_heads=2,
    rope_theta=10000.0,
    partial_rotary_factor=0.25,
    max_position_embeddings=1024,
    head_dim=16,
    full_attention_interval=4,
)


def build():
    mx.random.seed(5)
    model = qwen3_next.Model(qwen3_next.ModelArgs(**ARGS))
    model.set_dtype(mx.float32)
    # Sharper logits than a fresh init gives, so greedy picks are not ties.
    model.update(
        nn.utils.tree_map(lambda p: p * 3.0 if p.ndim == 2 else p, model.parameters())
    )
    mx.eval(model.parameters())
    return model


def plain(model, prompt, n):
    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt]), cache=cache)
    out = []
    for _ in range(n):
        y = int(mx.argmax(logits[0, -1]))
        out.append(y)
        logits = model(mx.array([[y]]), cache=cache)
    return out


def lookup(model, prompt, n):
    cache = make_prompt_cache(model)
    assert not can_trim_prompt_cache(cache), "tiny hybrid should not be trimmable"
    assert lookup_decode.can_rewind(cache), "hybrid cache should rewind by snapshot"
    mx.eval(model(mx.array([prompt[:-1]]), cache=cache))
    gen = lookup_decode.generate_step(
        mx.array(prompt[-1:], dtype=mx.uint32),
        model,
        max_tokens=n,
        prompt_cache=cache,
        history=prompt[:-1],
    )
    return [tok for tok, _ in gen]


def check_replay_exact(model, prompt):
    fed = [5, 9, 13, 17, 21]
    keep = fed[:2]
    a = make_prompt_cache(model)
    mx.eval(model(mx.array([prompt]), cache=a))
    snap = lookup_decode._snapshot(a)
    mx.eval(model(mx.array([fed]), cache=a))
    lookup_decode._replay(model, a, snap, len(fed), keep)
    b = make_prompt_cache(model)
    mx.eval(model(mx.array([prompt]), cache=b))
    mx.eval(model(mx.array([keep]), cache=b))
    la = model(mx.array([[30]]), cache=a)
    lb = model(mx.array([[30]]), cache=b)
    d = float(mx.abs(la - lb).max())
    assert d == 0.0, f"replayed cache differs from a clean one by {d}"
    print("OK: replay leaves the cache bit-identical to feeding only the kept tokens")


class _Side(_BaseCache):
    def __init__(self):
        self.keys = None

    @property
    def state(self):
        return self.keys

    @state.setter
    def state(self, v):
        self.keys = v


def check_side_cache_restored(model, prompt):
    # Shape of qwen4_exp: a side cache on the attention entry that trim() skips.
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([prompt]), cache=cache))
    attn = [c for c in cache if c.is_trimmable()]
    for c in attn:
        c.indexer = _Side()
        c.indexer.keys = mx.zeros((1, len(prompt), 4))
    snap = lookup_decode._snapshot(cache)
    mx.eval(model(mx.array([[5, 9, 13]]), cache=cache))
    for c in attn:
        c.indexer.keys = mx.zeros((1, len(prompt) + 3, 4))
    lookup_decode._replay(model, cache, snap, 3, [5])
    for c in attn:
        got = c.indexer.keys.shape[1]
        assert got == len(prompt), f"side cache kept {got} positions, want {len(prompt)}"
    print("OK: replay restores side caches the KV trim does not know about")


def check_greedy(model, prompt, got, margin=0.1):
    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt]), cache=cache)[0, -1]
    worst = 0.0
    for i, t in enumerate(got):
        best = int(mx.argmax(logits))
        gap = float(logits[best] - logits[t])
        worst = max(worst, gap)
        assert gap <= margin, (
            f"token {i} is {t} but greedy says {best} (logit gap {gap:.3f}): "
            "the rewind left a wrong state behind"
        )
        logits = model(mx.array([[t]]), cache=cache)[0, -1]
    print(f"OK: every emitted token is the greedy pick (worst gap {worst:.4f})")


def main():
    model = build()
    rng = np.random.default_rng(2)
    # A repeated phrase gives the drafter something to copy.
    phrase = rng.integers(0, ARGS["vocab_size"], size=12).tolist()
    prompt = phrase * 5
    n = 40

    check_replay_exact(model, prompt)
    check_side_cache_restored(model, prompt)
    ref = plain(model, prompt, n)
    es_config.LOOKUP_TOKENS = 4
    es_config.LOOKUP_ADAPTIVE = False

    # A random model never repeats the phrase, so draft from the answer
    # instead, spoiled at a rotating position: full accepts, partial accepts
    # and outright rejections all come up.
    def draft(tokens, k):
        g = len(tokens) - len(prompt)
        out = list(ref[g : g + k])
        bad = g % 4
        if out and bad < len(out):
            out[bad] = (out[bad] + 1) % ARGS["vocab_size"]
        return out

    real = lookup_decode.draft_ngram
    lookup_decode.draft_ngram = draft
    try:
        got = lookup(model, prompt, n)
    finally:
        lookup_decode.draft_ngram = real
    st = lookup_decode.stats()
    print(f"lookup stats: {st}")
    assert len(got) == n, f"asked for {n} tokens, got {len(got)}"
    check_greedy(model, prompt, got)
    assert st.get("drafted", 0) > 0, "drafter never offered a draft"
    assert st.get("untrimmable", 0) == 0, "hybrid cache was still treated as untrimmable"
    assert st.get("replays", 0) > 0, "no partial accept exercised the replay path"
    assert st.get("accepted", 0) > 0, "no draft token was ever accepted"
    print("OK: hybrid lookup stays greedy, drafts and replays ran")
    return 0


if __name__ == "__main__":
    sys.exit(main())
