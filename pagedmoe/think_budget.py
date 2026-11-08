# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Close a reasoning block at a char budget inside the same decode.

The host caps reasoning by characters. It used to close the stream at the cap
and send a second request with the reasoning as a think prefill. On a hybrid
cache (qwen3_next, qwen4_exp) the stored entry holds every id the cut pass
decoded, the resume only reuses the ones up to its last whole sentence, and a
hybrid entry cannot be trimmed back to that point. So each resume prefilled
the whole reasoning again: one more pass over the experts, about 6 s on Flash,
plus a new request and a cold first token.

This is a logits processor instead. Once the reasoning passes the budget, it
waits for the next sentence end (at most SLACK_TOKENS), then forces the close
text (a commit line and the think end tag) one token per step, and the model
goes on into its answer or tool call in the same stream.

It only counts reasoning that is really open. A template that opens <think> in
the prompt (`open_in_prompt`) counts from the first generated token. Anything
else counts only after the model emits its own think start, and a reply that
never opens one is left alone: Coder-Next has no think mode, wrote its plan
as plain text straight into a tool call, and got the close forced into the
middle of the code it was writing.

It only reads the token sequence it is handed, so a verify pass that rewinds
drafts sees the same answer for the same prefix. Everything stays on the GPU
(lookup tables over the vocab), so a plain decode step never syncs for it.
Before the budget is reached the logits pass through unchanged.
"""

from __future__ import annotations

import mlx.core as mx

SLACK_TOKENS = 48
_ENDS = (".", "!", "?", ":", "\n")
_TABLES: dict[int, tuple] = {}


def _piece_tables(tokenizer) -> tuple:
    """Decoded length and sentence-end flag per token id, built once per tokenizer
    (about 0.5 s for a 248k vocab)."""
    key = id(tokenizer)
    got = _TABLES.get(key)
    if got is not None:
        return got
    n = max(tokenizer.get_vocab().values()) + 1
    ids = [[i] for i in range(n)]
    try:
        pieces = tokenizer.batch_decode(ids)
    except Exception:
        pieces = [tokenizer.decode(i) for i in ids]
    lens = [len(p) for p in pieces]
    ends = [p.rstrip(" ").endswith(_ENDS) for p in pieces]
    got = (mx.array(lens, dtype=mx.int32), mx.array(ends, dtype=mx.bool_))
    _TABLES[key] = got
    return got


class ThinkBudget:
    def __init__(self, tokenizer, chars: int, close: str, open_in_prompt: bool = True):
        self.chars = int(chars)
        self.open_in_prompt = bool(open_in_prompt)
        self.end_id = int(tokenizer.think_end_id)
        start = getattr(tokenizer, "think_start_id", None)
        self.start_id = int(start) if start is not None else -1
        ids = tokenizer.encode(close, add_special_tokens=False)
        if not ids or ids[-1] != self.end_id:
            raise ValueError("close text must end with the think end tag")
        self.close_ids = list(ids)
        self.close = mx.array(ids, dtype=mx.int32)
        self.base: int | None = None
        self.lens, self.ends = _piece_tables(tokenizer)

    def __call__(self, seq: mx.array, logits: mx.array) -> mx.array:
        if self.base is None:
            # The first call carries the last prompt token(s); generation starts after.
            self.base = int(seq.shape[0])
            return logits
        n = int(seq.shape[0]) - self.base
        if n <= 0:
            return logits
        vocab = int(logits.shape[-1])
        lens, ends = self.lens, self.ends
        gen = seq[self.base :].astype(mx.int32)
        ids = mx.minimum(gen, lens.shape[0] - 1)
        at = mx.arange(n, dtype=mx.int32)
        piece = lens[ids]
        if self.open_in_prompt:
            opened = mx.array(True)
        else:
            starts = gen == self.start_id
            opened = mx.any(starts)
            piece = mx.where(at > mx.argmax(starts).astype(mx.int32), piece, 0)
        over = mx.cumsum(piece) >= self.chars
        first = mx.argmax(over).astype(mx.int32)
        stop = over & (ends[ids] | (at >= first + SLACK_TOKENS))
        cut = mx.argmax(stop).astype(mx.int32)
        step = n - 1 - cut
        k = int(self.close.shape[0])
        force = (
            opened
            & mx.any(stop)
            & mx.logical_not(mx.any(gen == self.end_id))
            & (step < k)
        )
        tok = self.close[mx.minimum(step, k - 1)]
        only = mx.where(mx.arange(vocab, dtype=mx.int32) == tok, logits, -mx.inf)
        return mx.where(force, only, logits)

    def closed_in(self, tokens: list) -> bool:
        """True when `tokens` (the decoded ids) contain the forced close."""
        k = len(self.close_ids)
        for i in range(len(tokens) - k + 1):
            if tokens[i : i + k] == self.close_ids:
                return True
        return False
