# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recurrent-state checkpoints: resume a hybrid prompt cache mid-lineage.

A hybrid model (GatedDeltaNet next to attention: qwen3_next, qwen4_exp) keeps
an ArraysCache per recurrent layer, and a recurrent state cannot be trimmed.
So the prompt store can only serve such a model an entry whose key is an exact
prefix of the new prompt. When the chat moves one message back from the last
stored entry (a retry, a recovery request, an edited history), every stored key
runs past the branch point and the turn prefills cold from token 0. On a
streamed checkpoint that is one or more full passes over the expert mass.

The attention half of such an entry trims fine. What is missing is the
recurrent state at the branch point, and that is small: Flash's 36 DeltaNet
layers hold about 110 MB where the attention KV at 30k tokens is several GB.
mlx arrays are immutable and ArraysCache replaces its entries rather than
writing into them, so holding references to the entries at some boundary
costs nothing at capture time. That is a checkpoint.

To resume at a checkpoint of length p: copy any stored entry whose key extends
the same p tokens, trim its attention layers (and their side caches, see
`lookup_decode._side_caches`) back to p, and put the checkpoint's recurrent
state in. The result is the cache a prefill of those p tokens would have left,
from the same KV the store already serves on every other resumed turn.

Anything this module does not recognize makes it step aside: capture returns
False, lookup finds nothing, and the turn takes exactly the path it took
before.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, can_trim_prompt_cache

from . import config, live_config
from .lookup_decode import _side_caches

STATS = {"captured": 0, "hits": 0, "restored_tokens": 0, "rejected": 0}


@dataclass
class Checkpoint:
    model_key: object
    tokens: tuple
    layers: list  # per cache layer: ("trim", n_side) or ("arrays", [state refs])
    nbytes: int

    @property
    def length(self) -> int:
        return len(self.tokens)


_REG: list[Checkpoint] = []


def enabled() -> bool:
    return bool(config.HYBRID_CKPT)


def clear() -> None:
    _REG.clear()


def nbytes() -> int:
    return sum(c.nbytes for c in _REG)


def count() -> int:
    return len(_REG)


def _byte_cap() -> int:
    cap = max(0, int(float(config.HYBRID_CKPT_MB) * (1 << 20)))
    from . import kvmem

    store = int(getattr(kvmem, "_store_budget", 0) or 0)
    if store > 0:
        cap = min(cap, store // 4)
    return cap


def _layout(cache, length: int) -> tuple[list, int] | None:
    """Checkpoint layout for `cache` at `length`, or None if it cannot resume."""
    if can_trim_prompt_cache(cache):
        return None  # the store already serves any prefix of a trimmable cache
    layers = []
    size = 0
    recurrent = False
    for c in cache:
        if isinstance(c, ArraysCache):
            if getattr(c, "left_padding", None) is not None:
                return None
            if getattr(c, "lengths", None) is not None:
                return None
            states = list(c.cache)
            size += sum(int(a.nbytes) for a in states if a is not None)
            layers.append(("arrays", states))
            recurrent = True
        elif c.is_trimmable():
            if int(getattr(c, "offset", -1)) != length:
                return None
            sides = _side_caches(c)
            for s in sides:
                st = s.state
                if not isinstance(st, mx.array) or st.ndim < 2 or st.shape[1] != length:
                    return None
            layers.append(("trim", len(sides)))
        else:
            return None
    if not recurrent:
        return None
    return layers, size


def capture(model_key, tokens, cache) -> bool:
    """Record the recurrent state of `cache` as the state after `tokens`.

    Call only where `cache` holds exactly `tokens` (offsets are checked). Holds
    references, not copies.
    """
    if not enabled() or len(tokens) < int(config.HYBRID_CKPT_MIN_TOKENS):
        return False
    if live_config.kv_stale():
        return False
    got = _layout(cache, len(tokens))
    if got is None:
        return False
    layers, size = got
    key = tuple(int(t) for t in tokens)
    for i, old in enumerate(_REG):
        if old.model_key == model_key and old.tokens == key:
            del _REG[i]
            break
    _REG.append(Checkpoint(model_key, key, layers, size))
    STATS["captured"] += 1
    _bound()
    return True


def from_slots(model_key, tokens, cache, slots: dict) -> Checkpoint | None:
    """A checkpoint at `tokens` from recurrent state read mid-pass
    (`inpass_snap.run`), while `cache` itself has already moved past it.

    Registered like `capture` when checkpoints are on; returned either way,
    since the caller restores its store snapshot from it.
    """
    p = len(tokens)
    layers = []
    size = 0
    for i, c in enumerate(cache):
        if isinstance(c, ArraysCache):
            states = slots.get(i)
            if states is None or len(states) != len(c.cache):
                return None
            size += sum(int(a.nbytes) for a in states if a is not None)
            layers.append(("arrays", list(states)))
        elif c.is_trimmable() and int(getattr(c, "offset", -1)) >= p:
            layers.append(("trim", len(_side_caches(c))))
        else:
            return None
    ckpt = Checkpoint(model_key, tuple(int(t) for t in tokens), layers, size)
    if enabled() and p >= int(config.HYBRID_CKPT_MIN_TOKENS) and not live_config.kv_stale():
        for i, old in enumerate(_REG):
            if old.model_key == model_key and old.tokens == ckpt.tokens:
                del _REG[i]
                break
        _REG.append(ckpt)
        STATS["captured"] += 1
        _bound()
    return ckpt


_KEEP_NEWEST = 2


def _on_lineage(c: Checkpoint, heads: list[Checkpoint]) -> bool:
    return any(
        h.model_key == c.model_key and h.length >= c.length and h.tokens[: c.length] == c.tokens
        for h in heads
    )


def _victim() -> int:
    """Index of the checkpoint to drop. The newest two stay: the next turn
    resumes from one of them. Then branches the conversation left (a cut
    think pass, a retried turn) go first, oldest first, since nothing will
    ask for them again. Among checkpoints on the live conversation, drop the
    one that leaves the smallest gap, so a follow-up that re-renders history
    far back (a template that drops old reasoning) still finds one close."""
    heads = _REG[-_KEEP_NEWEST:]
    old = range(len(_REG) - len(heads))
    for i in old:
        if not _on_lineage(_REG[i], heads):
            return i
    def gap(i: int) -> int:
        c = _REG[i]
        lens = sorted({x.length for x in _REG if x.model_key == c.model_key})
        j = lens.index(c.length)
        lo = lens[j - 1] if j > 0 else 0
        hi = lens[j + 1] if j + 1 < len(lens) else c.length
        return hi - lo

    return min(old, key=lambda i: (gap(i), i))


def _bound() -> None:
    cap_n = max(1, int(config.HYBRID_CKPT_MAX))
    cap_b = _byte_cap()
    while len(_REG) > 1 and (len(_REG) > cap_n or nbytes() > cap_b):
        if len(_REG) <= _KEEP_NEWEST:
            _REG.pop(0)
        else:
            _REG.pop(_victim())


def _entries_for(store, model_key, prefix: list[int]):
    """Every stored entry whose key extends `prefix`, as (key, prompt_cache).

    The trie names at most one longer key, and that one can be an entry the
    checkpoint cannot restore onto while another one would.
    """
    trie = getattr(store, "_trie", None)
    if trie is None:
        return
    keys = []
    try:
        res = trie.search(model_key, prefix)
        keys += [k for k in (res.exact, res.longer) if k is not None]
    except Exception:
        pass
    lru = getattr(store, "_lru", None)
    for cache_type in getattr(lru, "_ordering", None) or []:
        for stored_model, key in list((getattr(lru, "_lrus", None) or {}).get(cache_type) or []):
            if stored_model == model_key:
                keys.append(key)
    p = len(prefix)
    want = tuple(prefix)
    seen = set()
    for key in keys:
        key = tuple(key)
        if key in seen or len(key) < p or key[:p] != want:
            continue
        seen.add(key)
        try:
            entry = trie.get(model_key, list(key))
        except Exception:
            continue
        cache = getattr(entry, "prompt_cache", None)
        if cache is not None:
            yield list(key), cache


def restore(entry_cache, ckpt: Checkpoint):
    """A copy of `entry_cache` rewound to the checkpoint, or None if it does not fit."""
    if len(entry_cache) != len(ckpt.layers):
        return None
    p = ckpt.length
    for c, (kind, data) in zip(entry_cache, ckpt.layers):
        if kind == "arrays":
            if not isinstance(c, ArraysCache) or len(c.cache) != len(data):
                return None
        elif not c.is_trimmable() or int(getattr(c, "offset", -1)) < p:
            return None
    out = copy.deepcopy(entry_cache)
    for c, (kind, data) in zip(out, ckpt.layers):
        if kind == "arrays":
            c.cache = list(data)
            continue
        n = int(c.offset) - p
        if n and c.trim(n) != n:
            return None
        sides = _side_caches(c)
        if len(sides) != data:
            return None
        for s in sides:
            st = s.state
            if not isinstance(st, mx.array) or st.ndim < 2 or st.shape[1] < p:
                return None
            s.state = st[:, :p]
        if int(c.offset) != p:
            return None
    return out


def resume(store, model_key, tokens, have: int):
    """Beat the store's own `have` cached tokens with a checkpoint, if one fits.

    Returns (cache, rest, cached) or None. Leaves one token for decode, like
    the store does, and never touches the stored entry itself.
    """
    if not enabled() or not _REG:
        return None
    tokens = list(tokens)
    limit = len(tokens) - 1
    fits = [
        c
        for c in _REG
        if c.model_key == model_key
        and have < c.length <= limit
        and tuple(tokens[: c.length]) == c.tokens
    ]
    for ckpt in sorted(fits, key=lambda c: c.length, reverse=True):
        p = ckpt.length
        tried = 0
        for _key, entry_cache in _entries_for(store, model_key, tokens[:p]):
            tried += 1
            cache = restore(entry_cache, ckpt)
            if cache is not None:
                STATS["hits"] += 1
                STATS["restored_tokens"] += p
                return cache, tokens[p:], p
        STATS["rejected"] += 1
        print(
            f"[prefix]   hybrid checkpoint @ {p} unusable: "
            + (
                f"{tried} stored entr{'y extends' if tried == 1 else 'ies extend'} it, none fit"
                if tried
                else "no stored entry extends it"
            ),
            flush=True,
        )
    return None
