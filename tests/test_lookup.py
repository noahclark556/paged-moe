# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""N-gram self-draft decoding must not change what the model says.

Unlike PRUNE and FLOW, lookup decode is not a fidelity knob: a drafted token is
kept only when the model itself picks it, so greedy output has to come out
*identical* to sequential greedy decoding. That is the whole reason it is worth
having, so it is the thing to test.

Covered here:
  - identical greedy tokens, streamed and resident, at several draft depths
  - identical when the drafter is guaranteed to fire (highly repetitive input)
  - the KV cache is trimmed correctly after a rejection (a missed trim shows up
    as divergence a few tokens later, which the identity checks above catch)
  - draft_ngram's own contract, including the empty cases
  - acceptance is actually happening, so the identity checks are not passing
    because the drafter never fires

Run:

    python tests/test_lookup.py
"""

import importlib
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_stream import CONFIG, build_tiny_model, greedy_generate

from expert_stream import config as es_config
from expert_stream import load, lookup_decode


def lookup_generate(model, prompt_ids, n_new):
    cache = make_prompt_cache(model)
    return [
        tok
        for tok, _lp in lookup_decode.generate_step(
            mx.array(prompt_ids, mx.uint32),
            model,
            max_tokens=n_new,
            prompt_cache=cache,
        )
    ]


def test_draft_ngram():
    """The drafter's contract, before any model is involved."""
    es_config.LOOKUP_NGRAM_MIN = 3
    es_config.LOOKUP_NGRAM_MAX = 8

    # "1 2 3" appeared before, followed by 4 5.
    assert lookup_decode.draft_ngram([1, 2, 3, 4, 5, 9, 1, 2, 3], 2) == [4, 5]
    # Capped by k.
    assert lookup_decode.draft_ngram([1, 2, 3, 4, 5, 9, 1, 2, 3], 1) == [4]
    # No repeat of any suffix >= n_min: nothing to copy.
    assert lookup_decode.draft_ngram([1, 2, 3, 4, 5, 6], 3) == []
    # Shorter than n_min.
    assert lookup_decode.draft_ngram([1, 2], 3) == []
    # k <= 0 never drafts.
    assert lookup_decode.draft_ngram([1, 2, 3, 1, 2, 3], 0) == []
    # Prefers the longest match: "2 3 4" -> 7, not the shorter "3 4" -> ...
    seq = [3, 4, 9, 2, 3, 4, 7, 7, 7, 2, 3, 4]
    assert lookup_decode.draft_ngram(seq, 1) == [7]
    # A run of one token must still draft k, not the single trailing token the
    # nearest (shifted-by-one) match would give.
    assert lookup_decode.draft_ngram([5] * 20, 4) == [5, 5, 5, 5]
    # The continuation may run into the suffix region: those are still tokens
    # that really followed the match, so copying them is correct.
    assert lookup_decode.draft_ngram([1, 2, 3, 4, 1, 2, 3], 4) == [4, 1, 2, 3]
    # Falls back to the best partial when no match can supply a full k.
    assert lookup_decode.draft_ngram([7, 7, 7, 7], 4) == [7]
    print("OK: draft_ngram contract")


def check_identical(model, prompt, n_new, label):
    ref = greedy_generate(model, prompt, n_new)
    for k in (1, 2, 4, 7):
        es_config.LOOKUP_TOKENS = k
        lookup_decode.reset_stats()
        got = lookup_generate(model, prompt, n_new)
        st = lookup_decode.stats()
        assert got == ref, (
            f"{label}: k={k} changed greedy output\n  ref={ref}\n  got={got}"
        )
        assert len(got) == n_new, f"{label}: k={k} emitted {len(got)} != {n_new}"
        print(
            f"OK: {label} k={k} identical, "
            f"accept_rate={st.get('accept_rate')} "
            f"tokens_per_pass={st.get('tokens_per_pass')}"
        )
    return ref


def main():
    test_draft_ngram()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build_tiny_model(tmp)

        rng = np.random.default_rng(3)
        prompt = rng.integers(0, CONFIG["vocab_size"], size=50).tolist()
        n_new = 24

        es_config.LOOKUP_NGRAM_MIN = 3
        es_config.LOOKUP_NGRAM_MAX = 8
        try:
            print("== resident ==")
            model_r, _ = load(str(tmp), mode="resident", verbose=False)
            check_identical(model_r, prompt, n_new, "resident")
            del model_r

            print("== streamed (experts paged through the slab) ==")
            model_s, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
            check_identical(model_s, prompt, n_new, "streamed")

            # A prompt built from a repeated block gives the drafter plenty to
            # match on, so this is the case that exercises verify and the
            # rejection trim rather than the no-draft fallback.
            print("== repetitive prompt: drafting and batching happen ==")
            block = rng.integers(0, CONFIG["vocab_size"], size=12).tolist()
            rep = (block * 6)[:60]
            es_config.LOOKUP_TOKENS = 4
            lookup_decode.reset_stats()
            ref_rep = greedy_generate(model_s, rep, n_new)
            got_rep = lookup_generate(model_s, rep, n_new)
            st = lookup_decode.stats()
            assert got_rep == ref_rep, (
                f"repetitive: changed greedy output\n  ref={ref_rep}\n  got={got_rep}"
            )
            assert st["no_draft_frac"] < 1.0, "drafter never fired at all"
            assert st["accepted"] > 0, "no draft token was ever accepted"
            assert st["tokens_per_pass"] > 1.0, (
                f"no batching happened: {st['tokens_per_pass']} tokens/pass"
            )
            print(
                f"OK: identical, accept_rate={st['accept_rate']}, "
                f"tokens_per_pass={st['tokens_per_pass']}, steps={st['steps']}"
            )

            # The server reaches lookup through its patched generate_step, which
            # prefills first and so hands the drafter a one-token `prompt` plus
            # the rest as `history`. That wiring is easy to get wrong in a way
            # the direct calls above would never notice (an idle drafter still
            # produces correct tokens), so check both output and that it ran.
            print("== via the server's patched generate_step ==")
            from expert_stream import server

            server._install_adaptive_prefill_generate()
            mlx_generate = importlib.import_module("mlx_lm.generate")
            es_config.LOOKUP = True
            es_config.LOOKUP_TOKENS = 4
            lookup_decode.reset_stats()
            got_srv = [
                int(t)
                for t, _lp in mlx_generate.generate_step(
                    mx.array(prompt, mx.uint32),
                    model_s,
                    max_tokens=n_new,
                    prompt_cache=make_prompt_cache(model_s),
                )
            ]
            st_srv = lookup_decode.stats()
            ref_s = greedy_generate(model_s, prompt, n_new)
            assert got_srv == ref_s, (
                f"server path changed output\n  ref={ref_s}\n  got={got_srv}"
            )
            assert st_srv.get("steps", 0) > 0, "lookup never ran via the patch"
            assert st_srv["accepted"] > 0, (
                "drafter idled through the server path - history is not reaching it"
            )
            print(
                f"OK: identical, accept_rate={st_srv['accept_rate']}, "
                f"tokens_per_pass={st_srv['tokens_per_pass']}"
            )
            # A rotating KV cache stops being trimmable once it wraps, and
            # trim_prompt_cache reports that by returning 0 rather than raising.
            # Absorbing it would condition every later token on tokens that were
            # never emitted, so the drafter has to stand down instead. Sized to
            # wrap partway through, so both branches get exercised in one run.
            print("== rotating KV cache: drafting stands down when it wraps ==")
            es_config.LOOKUP_TOKENS = 4
            lookup_decode.reset_stats()
            rot = make_prompt_cache(model_s, max_kv_size=len(prompt) + 8)
            got_rot = [
                int(t)
                for t, _lp in lookup_decode.generate_step(
                    mx.array(prompt, mx.uint32),
                    model_s,
                    max_tokens=n_new,
                    prompt_cache=rot,
                )
            ]
            st_rot = lookup_decode.stats()
            assert len(got_rot) == n_new, (
                f"rotating cache emitted {len(got_rot)} != {n_new}"
            )
            assert st_rot["untrimmable"] > 0, (
                "cache never went untrimmable, so the guard was not exercised"
            )
            print(
                f"OK: {n_new} tokens, untrimmable steps={st_rot['untrimmable']}"
                f"/{st_rot['steps']}, no crash and no silent bad trim"
            )
            # Warm agent turn: the prompt cache serves nearly the whole prompt,
            # so generate_step's `prompt` is a short remainder and the context
            # the drafter needs lives only in the request slot. Shipping this
            # unwired made lookup inert on real traffic (no_draft=1.00) while
            # every test above still passed.
            print("== warm prompt cache: drafter sees the served prefix ==")
            served, remainder = prompt[:-3], prompt[-3:]
            server._req_ctx.draft_history = list(prompt)
            hist = server._drafting_history(mx.array(remainder, mx.uint32), 1)
            assert hist == list(prompt[:-1]), (
                f"history lost the cached prefix: {len(hist)} of {len(prompt) - 1}"
            )
            assert server._drafting_history(mx.array(remainder, mx.uint32), 1) == list(
                remainder[:-1]
            ), "request slot was reused across generations"
            # A slot left by some other request must not be spliced onto this
            # prompt: the KV cache never saw those tokens.
            server._req_ctx.draft_history = [t + 1 for t in prompt]
            assert server._drafting_history(mx.array(remainder, mx.uint32), 1) == list(
                remainder[:-1]
            ), "history accepted a sequence that is not this prompt's prefix"
            print(f"OK: {len(served)}-token prefix recovered, stale slots rejected")
            del model_s
        finally:
            es_config.LOOKUP = False
            es_config.LOOKUP_TOKENS = 4
            server._req_ctx.draft_history = None

    print("\nOK: lookup decode tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
