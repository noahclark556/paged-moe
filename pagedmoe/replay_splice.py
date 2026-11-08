# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Replay an assistant turn with the ids decode produced.

After a turn, the store holds prompt + every generated id. The next request
sends that turn back as text, the template renders it, and the tokenizer
encodes it again. That rarely gives the same ids: BPE has many spellings of
one string and sampling takes the odd ones, and hosts trim whitespace off
arguments. A trimmable cache loses everything past the first differing id. A
hybrid one (qwen3_next, qwen4_exp) cannot trim at all, so the stored entry is
never a prefix and every agent turn prefilled its own output again (79 of 80
turns in one Flash log, a few seconds and a full pass over the experts each).

So when the template's text for the latest assistant turn matches what a
stored entry under the previous prompt decodes to (whitespace aside), that
span is spelled with the stored ids. Spellings are remembered by the turn's
rendered text, and every later prompt spells each turn it replays the same
way, so the prompt stays an append-only extension of what the cache holds.

Only the spelling changes, and only to ids the model itself produced for that
text. A turn whose text differs beyond whitespace keeps the template's
encoding. A tokenizer whose chat template does not encode the same as a plain
encode of its rendered text turns this off.
"""

from __future__ import annotations

from collections import OrderedDict

from . import config, vision_chat

STATS = {"spliced": 0, "learned": 0, "missed": 0}

# (model_key, turn text) -> (ids, chars of the turn text they spell)
_SPELL: "OrderedDict[tuple, tuple[tuple, int]]" = OrderedDict()
# id(tokenizer) -> whether a plain encode of the rendered text is the prompt
_PLAIN_OK: dict[int, bool] = {}
# (model_key, turn text) with no stored decode under it, e.g. a host-written ack
_TRIED: "OrderedDict[tuple, None]" = OrderedDict()
_SCAN = 4
# Assistant turns checked per request, newest first. A host can append its own
# note and ack after the turn decode wrote, so the newest is not always it.
_LOOK_BACK = 4


def reset() -> None:
    _SPELL.clear()
    _PLAIN_OK.clear()
    _TRIED.clear()
    for k in STATS:
        STATS[k] = 0


def align(turn: str, gen: str) -> int | None:
    """Chars of `turn` that `gen` spells, skipping whitespace on either side.

    None when they differ in anything but whitespace.
    """
    i = j = 0
    n, m = len(turn), len(gen)
    while j < m:
        if i < n and turn[i] == gen[j]:
            i += 1
            j += 1
        elif i < n and turn[i].isspace():
            i += 1
        elif gen[j].isspace():
            j += 1
        else:
            return None
    return i


def _render(tokenizer, messages, add_generation_prompt, kw) -> str:
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=False, **kw
    )
    return vision_chat.expand_rendered(text)


def _encode(tokenizer, text: str) -> list:
    return list(tokenizer.encode(text, add_special_tokens=False)) if text else []


def _decode(tokenizer, ids) -> str:
    inner = getattr(tokenizer, "_tokenizer", tokenizer)
    try:
        return inner.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return inner.decode(list(ids))


def _eos(tokenizer) -> set:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids:
        return set(int(t) for t in ids)
    one = getattr(tokenizer, "eos_token_id", None)
    return {int(one)} if one is not None else set()


def spell(tokenizer, model_key, text: str) -> list:
    """Ids for `text`, with every remembered turn spelled the way decode wrote it."""
    spans = []
    for (key, turn), (ids, end) in _SPELL.items():
        if key != model_key:
            continue
        at = text.find(turn)
        while at >= 0:
            spans.append((at, at + end, ids, (key, turn)))
            at = text.find(turn, at + len(turn))
    spans.sort(key=lambda s: s[0])
    out, pos = [], 0
    for start, end, ids, k in spans:
        if start < pos:
            continue
        out += _encode(tokenizer, text[pos:start])
        out += ids
        pos = end
        _SPELL.move_to_end(k)
    out += _encode(tokenizer, text[pos:])
    return out


def _continuations(node: dict, limit: int = _SCAN, budget: int = 1 << 16) -> list:
    """Id paths below a trie node that end at a stored entry, deepest per branch."""
    out, stack = [], [(node, [])]
    while stack and len(out) < limit and budget > 0:
        cur, path = stack.pop()
        budget -= 1
        kids = [(t, n) for t, n in cur.items() if t != "__value__"]
        if not kids:
            if path and "__value__" in cur:
                out.append(path)
            continue
        for t, n in kids:
            stack.append((n, path + [t]))
    return out


def _node(store, model_key, ids):
    try:
        node = store._trie._trie[model_key]
        for t in ids:
            node = node[t]
        return node
    except Exception:
        return None


def _first_diff(turn: str, gen: str) -> int:
    i = j = 0
    while j < len(gen) and i < len(turn):
        if turn[i] == gen[j]:
            i += 1
            j += 1
        elif turn[i].isspace():
            i += 1
        elif gen[j].isspace():
            j += 1
        else:
            break
    return i


def learn(tokenizer, store, model_key, messages, kw) -> int:
    """Remember how decode spelled recent assistant turns. Ids learned, else 0.

    Walks back from the newest assistant turn and stops at one it already
    knows, so each turn is learned once, on the request right after it.
    Learning runs oldest first: a turn's stored entry sits under a prompt
    that already spells the turns before it as decoded.
    """
    picks = [i for i in range(len(messages) - 1, 0, -1) if messages[i].get("role") == "assistant"]
    todo = []
    for k in picks[:_LOOK_BACK]:
        head = _render(tokenizer, messages[:k], True, kw)
        upto = _render(tokenizer, messages[: k + 1], False, kw)
        if not upto.startswith(head):
            continue  # the template renders history turns differently from a live one
        turn = upto[len(head):]
        if (model_key, turn) in _SPELL:
            break
        if (model_key, turn) in _TRIED or not turn.strip():
            continue
        todo.append((head, turn))
    return sum(_learn_turn(tokenizer, store, model_key, head, turn) for head, turn in reversed(todo))


def _learn_turn(tokenizer, store, model_key, head, turn) -> int:
    _TRIED[(model_key, turn)] = None
    while len(_TRIED) > max(1, int(config.REPLAY_SPLICE_MAX)):
        _TRIED.popitem(last=False)
    node = _node(store, model_key, spell(tokenizer, model_key, head))
    if node is None:
        return 0
    eos = _eos(tokenizer)
    eos_text = [_decode(tokenizer, [t]) for t in eos]
    best, best_end, near = None, -1, None
    for path in _continuations(node):
        cut = next((i + 1 for i, t in enumerate(path) if t in eos), len(path))
        ids = path[:cut]
        gen = _decode(tokenizer, ids)
        end = align(turn, gen)
        if end is None:
            if near is None:
                near = gen
            continue
        rest = turn[end:].lstrip()
        if rest and not any(e and rest.startswith(e) for e in eos_text):
            continue  # stored entry stops partway through the turn
        if end > best_end:
            best, best_end = ids, end
    if best is None:
        if near is not None:
            STATS["missed"] += 1
            at = _first_diff(turn, near)
            print(
                f"[splice] turn text differs from what decode wrote at char {at}: "
                f"sent {turn[max(0, at - 40):at + 40]!r} / wrote {near[max(0, at - 40):at + 40]!r}",
                flush=True,
            )
        return 0
    _SPELL[(model_key, turn)] = (tuple(best), best_end)
    while len(_SPELL) > max(1, int(config.REPLAY_SPLICE_MAX)):
        _SPELL.popitem(last=False)
    STATS["learned"] += 1
    return len(best)


def apply(tokenizer, store, model_key, messages, kw, prompt) -> list | None:
    """The prompt with replayed turns spelled as decoded, or None to keep `prompt`."""
    if not config.REPLAY_SPLICE or store is None:
        return None
    if not any(m.get("role") == "assistant" for m in messages):
        return None
    kw = {k: v for k, v in kw.items() if k != "tokenize"}
    full = _render(tokenizer, messages, True, kw)
    ok = _PLAIN_OK.get(id(tokenizer))
    if ok is None:
        ok = _encode(tokenizer, full) == list(prompt)
        _PLAIN_OK[id(tokenizer)] = ok
        if not ok:
            print("[splice] off: this chat template does not encode like its rendered text", flush=True)
    if not ok:
        return None
    learned = learn(tokenizer, store, model_key, messages, kw)
    ids = spell(tokenizer, model_key, full)
    if ids == list(prompt):
        return None
    STATS["spliced"] += 1
    print(
        f"[splice] replayed turns spelled with decoded ids "
        f"({len(ids)} tok vs {len(prompt)} plain"
        + (f", learned {learned}" if learned else "")
        + ")",
        flush=True,
    )
    return ids


def resegment(old: list, segments: list, types: list, new: list) -> tuple[list, list]:
    """mlx-lm's segments for `new`: same system head and think tail, new middle."""
    if len(segments) <= 1:
        return [new], list(types) or ["assistant"]
    head = len(segments[0]) if types[0] == "system" else 0
    tail = len(segments[-1]) if types[-1] == "assistant" else 0
    if new[:head] != old[:head] or (tail and new[len(new) - tail:] != old[len(old) - tail:]):
        return [new], ["assistant"]
    segs, kinds = [], []
    if head:
        segs.append(new[:head])
        kinds.append("system")
    mid = new[head: len(new) - tail]
    if mid:
        segs.append(mid)
        kinds.append("user")
    if tail:
        segs.append(new[len(new) - tail:])
        kinds.append("assistant")
    return segs, kinds
