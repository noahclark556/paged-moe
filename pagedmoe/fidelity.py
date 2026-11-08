# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which decode tokens get the tighter PRUNE / WAIT_ABOVE (see config.FIDELITY).

The server feeds every emitted token id through `observe`. mlx-lm runs the
forward that consumes a token before it yields that token, so a span marker
takes effect one token late: the newline after `<tool_call>` still decodes
loose, everything after it is strict. That lag is fine for spans hundreds of
tokens long.

Guard (config.FIDELITY_GUARD): an approximate decode that falls into a
degenerate loop (the same one or two tokens over and over, e.g. a run of
";\\n\\n" lines in the reasoning) is the prune approximation losing the
thread. When the last tokens collapse like that, the next
FIDELITY_GUARD_TOKENS decode exact (every routed expert, no flow, no prune
head), and after FIDELITY_GUARD_STICKY trips the rest of the turn does. A
false trip only costs speed. At an exact config the guard does nothing.
"""

from __future__ import annotations

from collections import deque

from . import config

_markers: dict[str, tuple[int, ...]] = {}
_recent: list[int] = []
_keep = 1
_in_tool = False
_in_reply = False
_turn_left = 0
_strict_tokens = 0
_tool_tokens = 0
_window: deque[int] = deque()
_run_tok: int | None = None
_run_len = 0
_rescue_left = 0
_rescue_sticky = False
_guard_trips = 0
_rescue_tokens = 0


def _ids(seq) -> tuple[int, ...]:
    return tuple(int(t) for t in seq) if seq else ()


def configure(tokenizer) -> None:
    """Take the tool-call and think markers from the mlx-lm tokenizer wrapper."""
    global _markers, _keep
    found = {
        "tool_start": _ids(getattr(tokenizer, "tool_call_start_tokens", None)),
        "tool_end": _ids(getattr(tokenizer, "tool_call_end_tokens", None)),
        "think_start": _ids(getattr(tokenizer, "_think_start_tokens", None)),
        "think_end": _ids(getattr(tokenizer, "_think_end_tokens", None)),
    }
    _markers = {k: v for k, v in found.items() if v}
    _keep = max([len(v) for v in _markers.values()] + [1])


def _reset_guard() -> None:
    global _run_tok, _run_len, _rescue_left, _rescue_sticky
    _window.clear()
    _run_tok = None
    _run_len = 0
    _rescue_left = 0
    _rescue_sticky = False


def begin_turn() -> None:
    global _in_tool, _in_reply, _turn_left, _strict_tokens, _tool_tokens
    global _guard_trips, _rescue_tokens
    _recent.clear()
    _in_tool = _in_reply = False
    _turn_left = max(0, int(float(config.FIDELITY_TURN_START)))
    _strict_tokens = 0
    _tool_tokens = 0
    _guard_trips = 0
    _rescue_tokens = 0
    _reset_guard()


def end_turn() -> int:
    """Close the turn and return how many tokens decoded strict."""
    global _in_tool, _in_reply, _turn_left
    _in_tool = _in_reply = False
    _turn_left = 0
    _recent.clear()
    _reset_guard()
    return _strict_tokens


def tool_tokens() -> int:
    """Tokens of the last turn decoded inside a tool call."""
    return _tool_tokens


def guard_stats() -> tuple[int, int]:
    """(trips, tokens decoded exact by the guard) for the last turn."""
    return _guard_trips, _rescue_tokens


def _approximate() -> bool:
    """Same test as streaming.decode_is_exact, inverted (no import cycle)."""
    return (
        float(config.PRUNE) > 0.0
        or float(config.WAIT_ABOVE) > 0.0
        or 0 < int(config.ROUTE_TOP_K)
        or 0.0 < float(config.ROUTE_TOP_P) < 1.0
        or bool(config.SIDECAR and config.SIDECAR_HEAD_PRUNE)
    )


def rescuing() -> bool:
    """True while the guard holds decode exact."""
    return _rescue_sticky or _rescue_left > 0


def force_exact(tokens: int, sticky: bool = False) -> None:
    """Hold decode exact for the next `tokens` emitted tokens (rescue.py)."""
    global _rescue_left, _rescue_sticky
    _rescue_left = max(_rescue_left, max(1, int(tokens)))
    _rescue_sticky = _rescue_sticky or bool(sticky)


def _guard(token: int) -> None:
    global _run_tok, _run_len, _rescue_left, _rescue_sticky, _guard_trips, _rescue_tokens
    if rescuing():
        _rescue_tokens += 1
        if _rescue_left > 0:
            _rescue_left -= 1
    if not config.FIDELITY_GUARD or _rescue_sticky:
        return
    if token == _run_tok:
        _run_len += 1
    else:
        _run_tok, _run_len = token, 1
    width = max(8, int(config.FIDELITY_GUARD_WINDOW))
    _window.append(token)
    while len(_window) > width:
        _window.popleft()
    if rescuing() or not _approximate():
        return
    run = _run_len >= max(4, int(config.FIDELITY_GUARD_RUN))
    narrow = len(_window) >= width and len(set(_window)) <= int(config.FIDELITY_GUARD_DISTINCT)
    if not (run or narrow):
        return
    distinct = len(set(_window))
    _guard_trips += 1
    sticky = _guard_trips >= max(1, int(config.FIDELITY_GUARD_STICKY))
    if sticky:
        _rescue_sticky = True
    else:
        _rescue_left = max(1, int(config.FIDELITY_GUARD_TOKENS))
    why = f"token {token} x{_run_len}" if run else f"{distinct} distinct in {width}"
    _window.clear()
    _run_tok, _run_len = None, 0
    how = "rest of the turn" if sticky else f"next {_rescue_left} tokens"
    print(f"[fidelity] guard trip {_guard_trips}: degenerate decode ({why}), exact for the {how}", flush=True)


def _hit(name: str) -> bool:
    seq = _markers.get(name)
    if not seq:
        return False
    n = len(seq)
    return len(_recent) >= n and tuple(_recent[-n:]) == seq


def observe(token: int) -> None:
    global _in_tool, _in_reply, _turn_left, _strict_tokens, _tool_tokens
    _guard(int(token))
    if strict():
        _strict_tokens += 1
    if _in_tool:
        _tool_tokens += 1
    if _turn_left > 0:
        _turn_left -= 1
    if not _markers:
        return
    _recent.append(int(token))
    if len(_recent) > _keep:
        del _recent[:-_keep]
    if _hit("tool_start"):
        _in_tool = True
    elif _hit("tool_end"):
        _in_tool = False
    elif _hit("think_end"):
        # What follows the reasoning is what the user reads.
        _in_reply = True
    elif _hit("think_start"):
        _in_reply = False


def in_tool() -> bool:
    """True while emitted tokens are inside a tool call (one token after the marker)."""
    return _in_tool


def strict() -> bool:
    if not config.FIDELITY:
        return False
    return _in_tool or _turn_left > 0 or (_in_reply and bool(config.FIDELITY_REPLY))


def adjust(prune: float, wait_above: float) -> tuple[float, float]:
    """Tighten a layer's prune / wait for a strict token; pass others through.

    The live prune head only ever raises both above the configured values, so a
    strict token starts from the configured values, not the head's.
    """
    if not strict():
        return prune, wait_above
    scale = config.FIDELITY_TOOL_SCALE if _in_tool else config.FIDELITY_SCALE
    scale = min(1.0, max(0.0, float(scale)))
    return (
        min(prune, float(config.PRUNE)) * scale,
        min(wait_above, float(config.WAIT_ABOVE)) * scale,
    )
