# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""N-gram self-draft decoding: more tokens per byte read.

Decode on a big streamed checkpoint is disk-bandwidth-bound. Measured tok/s
times GB/token comes out at the drive's ~5.8 GB/s no matter where PRUNE sits,
and at PRUNE=0.8 the engine is already at ~91% of that bound. Reading bytes
faster is not available and reading fewer bytes costs quality, so the only lever
left is getting more tokens out of the same bytes.

Consecutive tokens route to heavily overlapping experts, so k tokens verified in
one forward pass share their expert reads. On a GLM-4.7 decode trace, per-layer
expert union per token:

    k=2  0.74x    k=4  0.54x    k=8  0.39x

The draft has to be free, though, or it spends the bytes it saves: a second
model would take RAM the expert slab needs, and a draft that streams experts
pays the very cost this is meant to avoid. So the draft here is not a model at
all. It looks for the last few tokens somewhere earlier in the context and
copies whatever followed them - zero weights, zero expert bytes, and on the
repetitive text this thing actually serves (code, diffs, structured output) it
hits often.

Output is unchanged. For greedy sampling this is bit-identical to sequential
decoding; with a sampler it draws from the same distributions in the same order.
See `_verify` for why.
"""

from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

from . import config

# Accounting for the [lookup] line and the stats blob. Acceptance is the number
# that decides whether this is paying for itself: every accepted draft token is
# a token whose expert reads were shared with the ones around it.
STATS = {
    "steps": 0,  # verify passes
    "drafted": 0,  # draft tokens offered
    "accepted": 0,  # draft tokens the model agreed with
    "emitted": 0,  # tokens handed to the caller
    "no_draft": 0,  # steps with no n-gram hit (plain single-token decode)
    "untrimmable": 0,  # steps that had to skip drafting: cache cannot rewind
}
_EMITTED_SINCE_LOG = 0


def reset_stats() -> None:
    global _EMITTED_SINCE_LOG
    for k in STATS:
        STATS[k] = 0
    _EMITTED_SINCE_LOG = 0


def stats() -> dict:
    """Acceptance and the byte saving it implies, or empty when unused."""
    if not STATS["steps"]:
        return {}
    drafted = STATS["drafted"]
    return {
        "steps": STATS["steps"],
        "drafted": drafted,
        "accepted": STATS["accepted"],
        "accept_rate": round(STATS["accepted"] / drafted, 4) if drafted else 0.0,
        "no_draft_frac": round(STATS["no_draft"] / STATS["steps"], 4),
        "untrimmable": STATS["untrimmable"],
        # Tokens per verify pass. Each pass reads roughly one token's worth of
        # expert union, so this is the throughput multiplier over single-token
        # decode, before the union growth in batch_union.py is applied.
        "tokens_per_pass": round(STATS["emitted"] / STATS["steps"], 3),
    }


def _log(msg: str) -> None:
    print(f"[lookup] {msg}", flush=True)


def tick_log(*, final: bool = False) -> None:
    """Print accept / tok-per-pass so a live `tail -f` can see the win.

    Cumulative over the current generation (reset at generate_step start), same
    cadence as the sidecar tick so the two lines land next to each other.
    """
    st = stats()
    if not st:
        return
    tag = "done" if final else "tick"
    _log(
        f"{tag} accept={st['accept_rate']:.2f} "
        f"tok/pass={st['tokens_per_pass']:.2f} "
        f"no_draft={st['no_draft_frac']:.2f} "
        f"drafted={st['drafted']} accepted={st['accepted']} "
        f"steps={st['steps']} k={int(config.LOOKUP_TOKENS)}"
        + (
            f" untrimmable={st['untrimmable']}"
            if st.get("untrimmable")
            else ""
        )
    )


def _maybe_tick(emitted: int) -> None:
    """Fire a tick every SIDECAR_LOG_EVERY decode tokens (default 32)."""
    global _EMITTED_SINCE_LOG
    _EMITTED_SINCE_LOG += emitted
    every = max(1, int(getattr(config, "SIDECAR_LOG_EVERY", 32) or 32))
    if _EMITTED_SINCE_LOG >= every:
        _EMITTED_SINCE_LOG = 0
        tick_log()


def draft_ngram(tokens: list[int], k: int) -> list[int]:
    """Up to `k` tokens continuing the longest recent suffix match.

    Searches longest suffix first: a longer match is a more specific context and
    a better bet, and finding it costs one scan of the tail either way. Returns
    [] when nothing matches, which the caller turns into an ordinary
    single-token step rather than a wasted verify.
    """
    n_max = min(config.LOOKUP_NGRAM_MAX, len(tokens))
    n_min = config.LOOKUP_NGRAM_MIN
    if k <= 0 or n_max < n_min:
        return []
    for n in range(n_max, n_min - 1, -1):
        suffix = tokens[-n:]
        best: list[int] = []
        # Scan back from the most recent occurrence: recent context resembles
        # what comes next more than the top of the prompt does. Stop short of
        # the suffix itself so a match has something after it to copy.
        #
        # Take the most recent match that can supply a *full* k tokens rather
        # than simply the most recent match. Inside a run of repeated tokens the
        # nearest match is the suffix shifted by one, whose continuation is the
        # single token at the end - so preferring it would silently pin every
        # draft to length 1 and give up the batching this exists for.
        for start in range(len(tokens) - n - 1, -1, -1):
            if tokens[start : start + n] == suffix:
                out = tokens[start + n : start + n + k]
                if len(out) >= k:
                    return out
                if len(out) > len(best):
                    best = out
        if best:
            return best
    return []


def _verify(model, cache, y, draft, sampler, logits_processors, tokens):
    """One forward over [y] + draft; return the tokens that survive.

    Position i's logits predict token i+1, so logits[0] judges draft[0] and
    logits[-1] is the bonus token for a fully accepted draft - k+1 positions
    yield up to k+1 tokens.

    Accepting on equality is what keeps the output distribution exact. The
    target's distribution at a position depends only on the prefix, and the
    prefix is the same whether we arrived sequentially or by verification, so
    sampling from the target and keeping the result when it happens to equal the
    draft is just sampling from the target. The draft never contributes
    probability mass; it only saves the pass that would have revealed the same
    token. (This is why no rejection-sampling correction is needed, and why a
    drafter with no distribution of its own is fine here.)
    """
    n_draft = len(draft)
    inp = mx.array([[y] + draft], dtype=mx.uint32)
    logits = model(inp, cache=cache)[0]  # (1 + n_draft, vocab)

    if logits_processors:
        # Processors expect the running sequence; apply per position so a
        # repetition penalty sees the same history it would have seen.
        rows = []
        seq = list(tokens)
        for i in range(logits.shape[0]):
            row = logits[i : i + 1]
            for proc in logits_processors:
                row = proc(mx.array(seq, dtype=mx.uint32), row)
            rows.append(row)
            if i < n_draft:
                seq.append(draft[i])
        logits = mx.concatenate(rows, axis=0)

    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    picks = sampler(logprobs) if sampler is not None else mx.argmax(logprobs, axis=-1)
    mx.eval(picks, logprobs)
    picked = [int(t) for t in picks]

    out = []
    for i, tok in enumerate(picked):
        out.append((tok, logprobs[i]))
        if i >= n_draft or tok != draft[i]:
            break  # rejected here (or this was the bonus token)

    accepted = len(out) - 1  # the last entry is always the model's own token
    STATS["steps"] += 1
    STATS["drafted"] += n_draft
    STATS["accepted"] += accepted
    STATS["emitted"] += len(out)

    # Positions the draft consumed but the model rejected are still in the KV
    # cache. They must go, or the next step conditions on tokens that were never
    # emitted. n_draft + 1 fed, accepted + 1 kept.
    extra = n_draft - accepted
    if extra > 0:
        # The caller only drafts when the cache can rewind, so a short trim here
        # is a bug rather than a condition to absorb - and absorbing it would
        # corrupt the output silently, conditioning every later token on tokens
        # that were never emitted. trim_prompt_cache returns 0 rather than
        # raising when it cannot trim, so check the count.
        trimmed = trim_prompt_cache(cache, extra)
        if trimmed != extra:
            raise RuntimeError(
                f"lookup decode could not rewind the KV cache: asked {extra}, "
                f"trimmed {trimmed}"
            )
    return out


def generate_step(
    prompt,
    model,
    *,
    max_tokens: int = 256,
    sampler=None,
    logits_processors=None,
    prompt_cache,
    history=(),
    quantize_cache_fn=None,
    generation_stream=None,
):
    """Decode loop with n-gram drafting; yields (token, logprobs) like mlx-lm.

    `prompt` is the un-prefilled remainder (the caller's prefill loop leaves one
    token, same contract as mlx_lm.generate_step). `history` is the part of the
    prompt already in the KV cache, and passing it is not optional in practice:
    the prompt is the only context to match against for the first tokens, and
    without it the drafter sits idle exactly when it would help most.
    """
    if not isinstance(prompt, mx.array):
        prompt = mx.array(prompt, dtype=mx.uint32)

    k = int(config.LOOKUP_TOKENS)
    tokens = [int(t) for t in history] + [int(t) for t in prompt]
    if generation_stream is None:
        generation_stream = mx.default_stream(mx.default_device())
    reset_stats()

    with mx.stream(generation_stream):
        # Last prompt token: an ordinary step, and it seeds the history the
        # drafter reads from.
        logits = model(prompt[None], cache=prompt_cache)[0, -1:]
        if logits_processors:
            for proc in logits_processors:
                logits = proc(mx.array(tokens, dtype=mx.uint32), logits)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = int(sampler(logprobs)[0]) if sampler is not None else int(mx.argmax(logprobs[0]))
        if quantize_cache_fn is not None:
            quantize_cache_fn(prompt_cache)
        yield y, logprobs[0]

        n = 1
        try:
            while n < max_tokens:
                tokens.append(y)
                # Verifying a draft means feeding tokens that may be rejected, which
                # only works if they can be taken back out. A rotating KV cache
                # stops being trimmable once it wraps, so this is checked every step
                # rather than once: mid-generation is exactly when it flips.
                if can_trim_prompt_cache(prompt_cache):
                    # Leave room for the bonus token, and stay inside the streamed
                    # decode path's batch ceiling (streaming._DECODE_MAX_TOKENS).
                    draft = draft_ngram(tokens, min(k, max_tokens - n - 1))
                else:
                    draft = []
                    STATS["untrimmable"] += 1
                if not draft:
                    STATS["no_draft"] += 1
                got = _verify(
                    model, prompt_cache, y, draft, sampler, logits_processors, tokens
                )
                if quantize_cache_fn is not None:
                    quantize_cache_fn(prompt_cache)

                batch = 0
                for i, (tok, lp) in enumerate(got):
                    if n >= max_tokens:
                        break
                    yield tok, lp
                    n += 1
                    batch += 1
                    if i < len(got) - 1:
                        tokens.append(tok)
                if batch:
                    _maybe_tick(batch)
                if n >= max_tokens:
                    return
                y = got[-1][0]
        finally:
            # One summary line even on short replies that never hit the tick
            # cadence, and always once at the end of a long generation.
            tick_log(final=True)
