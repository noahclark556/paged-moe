# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""N-gram self-draft decoding: more tokens per byte read.

Decode on a big streamed checkpoint is disk-bandwidth-bound. A PRUNE sweep
pins it: tok/s times GB/token comes out at the drive's ~5.8 GB/s no
matter where PRUNE sits, and at PRUNE=0.8 the engine is already at ~91% of that
bound. Reading bytes faster is not available and reading fewer bytes costs
quality, so the only lever left is getting more tokens out of the same bytes.

Consecutive tokens route to heavily overlapping experts, so k tokens verified in
one forward pass share their expert reads. Measured on a GLM-4.7 decode trace
(`bench/batch_union.py`), per-layer expert union per token:

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

import time

import mlx.core as mx
from mlx_lm.models.cache import (
    ArraysCache,
    _BaseCache,
    can_trim_prompt_cache,
    trim_prompt_cache,
)

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
    "replays": 0,  # partial accepts on a recurrent cache (restore + re-run)
    "gated": 0,  # matches the adaptive gate skipped (drafting measured slower)
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
        "replays": STATS["replays"],
        "gated": STATS["gated"],
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
        + (f" replays={st['replays']}" if st.get("replays") else "")
        + (f" gated={st['gated']}" if st.get("gated") else "")
    )


def _maybe_tick(emitted: int) -> None:
    """Fire a tick every SIDECAR_LOG_EVERY decode tokens (default 32)."""
    global _EMITTED_SINCE_LOG
    _EMITTED_SINCE_LOG += emitted
    every = max(1, int(getattr(config, "SIDECAR_LOG_EVERY", 32) or 32))
    if _EMITTED_SINCE_LOG >= every:
        _EMITTED_SINCE_LOG = 0
        tick_log()


# Suffix length of the last match draft_ngram found, for the gate below.
_LAST_MATCH = 0


class _Gate:
    """Draft only where drafting has been measured to beat plain decode.

    Whether a pass pays depends on the model and the text. On a hybrid model a
    partial accept costs a replay on top of the pass, and on Flash varied 12k
    context ran at 0.86x of plain decode without this. The gate keeps an
    EMA of seconds per emitted token for plain steps and, per matched suffix
    length (short, medium, full window), for draft passes. A bucket drafts
    while its EMA is at or under plain; a losing bucket still drafts once every
    EXPLORE chances so it can come back when the text turns repetitive.
    """

    ALPHA = 0.3
    EXPLORE = 16

    @staticmethod
    def bucket(n: int) -> int:
        if n >= config.LOOKUP_NGRAM_MAX:
            return 2
        return 1 if n >= 5 else 0

    def __init__(self, k: int = 0):
        self.k = k
        self.plain: float | None = None
        self.draft: dict[int, float] = {}
        self.skips: dict[int, int] = {}

    def want(self, n: int) -> bool:
        d = self.draft.get(n)
        if d is None or self.plain is None or d <= self.plain:
            return True
        self.skips[n] = self.skips.get(n, 0) + 1
        if self.skips[n] >= self.EXPLORE:
            self.skips[n] = 0
            return True
        return False

    def note(self, n: int | None, sec_per_tok: float) -> None:
        if n is None:
            old = self.plain
        else:
            old = self.draft.get(n)
        new = sec_per_tok if old is None else old + self.ALPHA * (sec_per_tok - old)
        if n is None:
            self.plain = new
        else:
            self.draft[n] = new


_GATE = _Gate()


def draft_ngram(tokens: list[int], k: int) -> list[int]:
    """Up to `k` tokens continuing the longest recent suffix match.

    Searches longest suffix first: a longer match is a more specific context and
    a better bet, and finding it costs one scan of the tail either way. Returns
    [] when nothing matches, which the caller turns into an ordinary
    single-token step rather than a wasted verify.
    """
    global _LAST_MATCH
    _LAST_MATCH = 0
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
                    _LAST_MATCH = n
                    return out
                if len(out) > len(best):
                    best = out
        if best:
            _LAST_MATCH = n
            return best
    return []


def _side_caches(c) -> list:
    """Caches hung off a trimmable entry that its own trim() does not know about.

    qwen4_exp keeps the sparse indexer's raw keys on `_AttnCache.indexer`. A
    KVCache trim leaves them in place, and once the context passes the indexer
    budget the stale keys break the mask shape.
    """
    return [v for v in vars(c).values() if isinstance(v, _BaseCache) and v is not c]


def _snapshot(cache) -> list | None:
    """Recurrent state to come back to, or None if some entry cannot rewind.

    Hybrid models (GatedDeltaNet next to attention) keep an ArraysCache per
    recurrent layer, and a recurrent state cannot be trimmed. It does not need
    to be: mlx arrays are immutable and ArraysCache replaces its entries rather
    than writing into them, so holding the old references is the snapshot.
    Attention entries still trim; only their side caches are held.
    """
    snap = []
    for c in cache:
        if c.is_trimmable():
            snap.append(("trim", [(s, s.state) for s in _side_caches(c)]))
        elif isinstance(c, ArraysCache):
            snap.append(("arrays", list(c.cache)))
        else:
            return None
    return snap


def _needs_snapshot(cache) -> bool:
    return not can_trim_prompt_cache(cache) or any(_side_caches(c) for c in cache)


def can_rewind(cache) -> bool:
    return can_trim_prompt_cache(cache) or _snapshot(cache) is not None


def _replay(model, cache, snap, fed: int, keep: list[int]) -> None:
    """Put a hybrid cache where it would be after only `keep` was fed.

    Restore the recurrent state, take all `fed` positions back off attention,
    and run the kept tokens again. The extra pass is the price of a partial
    accept on these models; a full accept never pays it.
    """
    for c, (kind, s) in zip(cache, snap):
        if kind == "arrays":
            c.cache = list(s)
            continue
        got = c.trim(fed)
        if got != fed:
            raise RuntimeError(
                f"lookup decode could not rewind the KV cache: asked {fed}, "
                f"trimmed {got}"
            )
        for side, state in s:
            side.state = state
    mx.eval(model(mx.array([keep], dtype=mx.uint32), cache=cache))
    STATS["replays"] += 1


def unfeed(model, cache, snap, kept: list[int], n_keep: int) -> None:
    """Leave only kept[:n_keep] of the tokens a verify pass kept in the cache.

    For a caller that stops partway through a pass's accepted tokens: the
    server keys the stored cache on the tokens it was handed, so anything fed
    past the last one has to come back out. `snap` is the pre-pass snapshot
    (None on a trimmable cache).
    """
    extra = len(kept) - n_keep
    if extra <= 0:
        return
    if can_trim_prompt_cache(cache) and not any(_side_caches(c) for c in cache):
        got = trim_prompt_cache(cache, extra)
        if got != extra:
            raise RuntimeError(f"could not take back {extra} tokens, trimmed {got}")
        return
    if snap is None:
        raise RuntimeError("cannot take back tokens from a cache with no snapshot")
    for c, (kind, s) in zip(cache, snap):
        if kind == "arrays":
            c.cache = list(s)
            continue
        got = c.trim(len(kept))
        if got != len(kept):
            raise RuntimeError(f"could not take back {len(kept)} tokens, trimmed {got}")
        for side, state in s:
            side.state = state
    if n_keep:
        mx.eval(model(mx.array([kept[:n_keep]], dtype=mx.uint32), cache=cache))


_LAST_SNAP = None


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
    global _LAST_SNAP
    n_draft = len(draft)
    snap = None
    if n_draft and _needs_snapshot(cache):
        snap = _snapshot(cache)
    _LAST_SNAP = snap
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
    if extra > 0 and snap is not None:
        _replay(model, cache, snap, n_draft + 1, [y] + draft[:accepted])
    elif extra > 0:
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

    global _GATE
    k = int(config.LOOKUP_TOKENS)
    if _GATE.k != k:
        # Pass cost depends on k, so timings taken at another depth do not carry.
        _GATE = _Gate(k)
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
        n = 1
        failed = False
        inflight = None
        shown = 0
        got = None
        try:
            yield y, logprobs[0]
            while n < max_tokens:
                tokens.append(y)
                # Verifying a draft means feeding tokens that may be rejected, which
                # only works if they can be taken back out. A rotating KV cache
                # stops being trimmable once it wraps, so this is checked every step
                # rather than once: mid-generation is exactly when it flips.
                if can_rewind(prompt_cache):
                    # Leave room for the bonus token, and stay inside the streamed
                    # decode path's batch ceiling (streaming._DECODE_MAX_TOKENS).
                    draft = draft_ngram(tokens, min(k, max_tokens - n - 1))
                else:
                    draft = []
                    STATS["untrimmable"] += 1
                bucket = _Gate.bucket(_LAST_MATCH) if draft else None
                if draft and config.LOOKUP_ADAPTIVE and not _GATE.want(bucket):
                    draft = []
                    bucket = None
                    STATS["gated"] += 1
                if not draft:
                    STATS["no_draft"] += 1
                t0 = time.perf_counter()
                got = _verify(
                    model, prompt_cache, y, draft, sampler, logits_processors, tokens
                )
                _GATE.note(bucket, (time.perf_counter() - t0) / len(got))
                if quantize_cache_fn is not None:
                    quantize_cache_fn(prompt_cache)

                # The cache now holds y and the accepted drafts; got[-1] is not
                # fed until the next pass.
                inflight = (_LAST_SNAP, [y] + [t for t, _ in got[:-1]])
                shown = 1
                batch = 0
                for i, (tok, lp) in enumerate(got):
                    if n >= max_tokens:
                        break
                    shown += 1
                    yield tok, lp
                    n += 1
                    batch += 1
                    if i < len(got) - 1:
                        tokens.append(tok)
                inflight = None
                if batch:
                    _maybe_tick(batch)
                y = got[-1][0]
                if n >= max_tokens:
                    return
        except GeneratorExit:
            raise
        except BaseException:
            failed = True
            raise
        finally:
            if not failed:
                # mlx-lm's loop has fed every token it yields, and the server
                # keys the stored cache on exactly those. Match that: take back
                # accepted tokens the caller never saw, or feed the last one.
                if inflight is not None and shown <= len(inflight[1]):
                    unfeed(model, prompt_cache, inflight[0], inflight[1], shown)
                else:
                    last = got[-1][0] if inflight is not None else y
                    model(mx.array([[last]], dtype=mx.uint32), cache=prompt_cache)
                if quantize_cache_fn is not None:
                    quantize_cache_fn(prompt_cache)
                mx.eval([c.state for c in prompt_cache])
            # One summary line even on short replies that never hit the tick
            # cadence, and always once at the end of a long generation.
            tick_log(final=True)
