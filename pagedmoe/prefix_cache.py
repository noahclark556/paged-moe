# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stable prefix pin + explicit cache breakpoints.

Causal attention means KV at position i depends on tokens[0:i]. A middle
span cannot be cached on its own and spliced after a different prefix - that
is why ``<cache>island</cache>`` tags are not a thing this engine will do.

What *is* valid:

* Prefix breakpoints. Snapshot KV at token N. Any later request whose first
  N tokens match reuses that KV and prefills only the tail.
* A pinned copy of a short stable head (system / tools / front matter) that
  the LRU is not allowed to evict or MOVE away. Follow-up turns that rewrite
  later messages still hit that head.

Hosts mark breakpoints with ``cache_control`` on a message, a top-level
``cache_breaks: [token_index, ...]`` list, or the ``<|cache_break|>``
sentinel at the end of a message. All three are prefix marks, not islands.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

CACHE_BREAK_MARKERS = ("<|cache_break|>", "<!--cache_break-->")


def _cache_len(cache: list) -> int:
    for c in cache or []:
        off = getattr(c, "offset", None)
        if off is None:
            try:
                off = getattr(c[0], "offset", None)
            except (TypeError, IndexError, AttributeError):
                off = None
        if off is not None:
            try:
                return max(0, int(off))
            except (TypeError, ValueError):
                return 0
    return 0

# Keep the pin short: system + tool schemas + pinned front matter. A 24k
# conversation pin would fight the byte budget and duplicate the LRU.
_MAX_PIN_TOKENS = 8192
_MIN_PIN_TOKENS = 64
# But "short" is relative to the prompt. An agent host's system prompt plus a
# couple dozen tool schemas runs well past 8k on its own, and a flat 8192 cap
# refused to pin the one span that never changes: every turn that rewrote its
# front matter then re-prefilled from token 0. Allow up to half the prompt,
# which keeps the pin a minority of the budget at any context length.
_PIN_CEILING_TOKENS = 32768
PIN_CEILING_TOKENS = _PIN_CEILING_TOKENS


def pin_cap(prompt_tokens: int) -> int:
    """Longest head worth pinning for a prompt of this size."""
    if prompt_tokens <= 0:
        return _MAX_PIN_TOKENS
    return max(_MAX_PIN_TOKENS, min(_PIN_CEILING_TOKENS, prompt_tokens // 2))


def common_prefix_len(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def explicit_bounds(body: Optional[dict]) -> list[int]:
    """Token offsets from the request body (``cache_breaks`` / ``cacheBreaks``)."""
    if not body:
        return []
    raw = body.get("cache_breaks") or body.get("cacheBreaks") or []
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[int] = []
    for x in raw:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if i > 0:
            out.append(i)
    return out


def message_has_cache_control(msg: Any) -> bool:
    if not isinstance(msg, dict):
        return False
    cc = msg.get("cache_control") if "cache_control" in msg else msg.get("cacheControl")
    return bool(cc)


def content_has_cache_break(msg: Any) -> bool:
    if not isinstance(msg, dict):
        return False
    text = msg.get("content")
    if not isinstance(text, str):
        return False
    return any(m in text for m in CACHE_BREAK_MARKERS)


def strip_cache_markers(text: str) -> str:
    for m in CACHE_BREAK_MARKERS:
        text = text.replace(m, "")
    return text


def pick_pin_bound(
    bounds: list[int], *, max_pin: int = _MAX_PIN_TOKENS, prefer: int = 0
) -> int:
    """Snapshot to pin: long enough to be worth it, short enough not to hog the
    prompt-cache budget.

    ``prefer`` is the end of the first message (system + tool schemas). That span
    is the most stable thing in any chat prompt, so pin it when it fits. The pin
    only ever extends forward, so a pin placed at some volatile boundary can
    never be repaired - falling back to the earliest usable bound keeps that
    mistake cheap.
    """
    usable = [b for b in bounds if _MIN_PIN_TOKENS <= b <= max_pin]
    if not usable:
        return 0
    if _MIN_PIN_TOKENS <= prefer <= max_pin:
        return prefer
    return min(usable)


class StablePrefixPin:
    """One pinned prefix KV per model_key. Always copied out, never moved."""

    def __init__(self) -> None:
        self._pins: dict[str, tuple[list, list]] = {}

    def get(self, model_key: str) -> Optional[tuple[list, list]]:
        return self._pins.get(model_key)

    def clear(self, model_key: Optional[str] = None) -> None:
        if model_key is None:
            self._pins.clear()
        else:
            self._pins.pop(model_key, None)

    def update(self, model_key: str, tokens: list, cache: list, bound: int) -> bool:
        """Pin ``tokens[:bound]`` if it extends (or creates) the stored head."""
        if bound < _MIN_PIN_TOKENS or bound > len(tokens):
            return False
        snap_tok = list(tokens[:bound])
        existing = self._pins.get(model_key)
        if existing is not None:
            old_tok, _ = existing
            # Extends the stored head: take the longer one. Diverges from it:
            # replace, because a pin the live traffic no longer starts with can
            # never be repaired (the pin only grows forward) and on a
            # non-trimmable cache it cannot even serve a partial match.
            if snap_tok[: len(old_tok)] == old_tok and bound <= len(old_tok):
                return False
        try:
            snap = copy.deepcopy(cache)
        except Exception:
            return False
        off = _cache_len(snap)
        extra = off - bound
        if extra > 0:
            try:
                from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

                if not can_trim_prompt_cache(snap):
                    return False
                trim_prompt_cache(snap, extra)
            except Exception:
                return False
        elif off < bound:
            return False
        # The pin's token list and its KV must describe the same span. A cache
        # that would not trim (hybrid / linear-attention layers) has to be
        # refused here, or lookup hands back KV for more tokens than it claims.
        if _cache_len(snap) != bound:
            return False
        self._pins[model_key] = (snap_tok, snap)
        return True

    def put(self, model_key: str, tokens: list, cache: list) -> bool:
        """Pin ``cache`` as the KV for exactly ``tokens``, without a copy."""
        if len(tokens) < _MIN_PIN_TOKENS or _cache_len(cache) != len(tokens):
            return False
        self._pins[model_key] = (list(tokens), cache)
        return True

    def lookup(self, model_key: str, tokens: list) -> Optional[tuple[list, list, int]]:
        """If ``tokens`` starts with the pin, return (copied cache, rest, n)."""
        pin = self._pins.get(model_key)
        if pin is None:
            return None
        pin_tok, pin_cache = pin
        n = common_prefix_len(pin_tok, tokens)
        if n < min(len(pin_tok), _MIN_PIN_TOKENS):
            return None
        try:
            cache = copy.deepcopy(pin_cache)
        except Exception:
            return None
        extra = len(pin_tok) - n
        if extra > 0:
            # Partial match: the KV past `n` belongs to tokens this prompt does
            # not have. Without a trim the returned cache is longer than the
            # prefix we report, and the tail would attend to the wrong keys at
            # the wrong positions - wrong output, not just a slow one.
            try:
                from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

                if not can_trim_prompt_cache(cache):
                    return None
                trim_prompt_cache(cache, extra)
            except Exception:
                return None
            if _cache_len(cache) != n:
                return None
        return cache, list(tokens[n:]), n
