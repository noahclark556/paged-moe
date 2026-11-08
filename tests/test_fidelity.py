# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fidelity regions: which decode tokens get the tighter PRUNE / WAIT_ABOVE.

Covered here:
  - tool-call spans are strict, prose around them is not
  - the visible reply after a closing think marker is strict, reasoning is not
  - the turn-start window, and that it resets per turn
  - adjust never loosens anything and ignores the prune head's extra bump
  - PRUNE 0 / WAIT_ABOVE 0 stays exact
  - no markers on the tokenizer means only the turn-start window applies

Run:

    python tests/test_fidelity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from pagedmoe import config, fidelity  # noqa: E402

_TOUCHED = ("FIDELITY", "FIDELITY_TURN_START", "FIDELITY_REPLY", "PRUNE", "WAIT_ABOVE")


@pytest.fixture(autouse=True)
def _restore_config():
    saved = {k: getattr(config, k) for k in _TOUCHED}
    yield
    for k, v in saved.items():
        setattr(config, k, v)
    fidelity.end_turn()


START, END, THINK, UNTHINK = 900, 901, 902, 903


class _Tok:
    tool_call_start_tokens = (START,)
    tool_call_end_tokens = (END,)
    _think_start_tokens = (THINK,)
    _think_end_tokens = (UNTHINK,)


def _run(ids):
    flags = []
    fidelity.begin_turn()
    for t in ids:
        flags.append(fidelity.strict())
        fidelity.observe(t)
    return flags, fidelity.end_turn()


def test_spans():
    config.FIDELITY, config.FIDELITY_TURN_START = True, 0
    fidelity.configure(_Tok())
    flags, n = _run([1, 2, START, 3, 4, END, 5, 6])
    assert flags == [False, False, False, True, True, True, False, False], flags
    assert n == 3


def test_reply():
    config.FIDELITY, config.FIDELITY_TURN_START, config.FIDELITY_REPLY = True, 0, True
    fidelity.configure(_Tok())
    # reasoning, close, visible reply, tool call, reply again
    flags, _ = _run([1, 2, UNTHINK, 3, 4, START, 5, END, 6])
    assert flags == [False, False, False, True, True, True, True, True, True], flags
    flags, _ = _run([1, UNTHINK, 3, THINK, 4, 5])
    assert flags == [False, False, True, True, False, False], flags
    config.FIDELITY_REPLY = False
    flags, _ = _run([1, UNTHINK, 3, 4])
    assert flags == [False, False, False, False], flags
    config.FIDELITY_REPLY = True


def test_turn_start():
    config.FIDELITY, config.FIDELITY_TURN_START = True, 2
    fidelity.configure(_Tok())
    flags, _ = _run([1, 2, 3, 4])
    assert flags == [True, True, False, False], flags
    flags, _ = _run([1, 2, 3])
    assert flags == [True, True, False], flags
    assert not fidelity.strict()


def test_adjust():
    config.FIDELITY, config.FIDELITY_TURN_START = True, 1
    config.PRUNE, config.WAIT_ABOVE, config.FIDELITY_SCALE = 0.7, 0.25, 0.6
    fidelity.configure(_Tok())
    fidelity.begin_turn()
    p, w = fidelity.adjust(0.9, 0.4)
    assert abs(p - 0.42) < 1e-9 and abs(w - 0.15) < 1e-9, (p, w)
    p, w = fidelity.adjust(0.3, 0.1)
    assert p <= 0.3 and w <= 0.1
    fidelity.observe(1)
    assert fidelity.adjust(0.9, 0.4) == (0.9, 0.4)
    config.PRUNE, config.WAIT_ABOVE = 0.0, 0.0
    fidelity.begin_turn()
    assert fidelity.adjust(0.0, 0.0) == (0.0, 0.0)
    config.FIDELITY = False
    fidelity.begin_turn()
    assert fidelity.adjust(0.9, 0.4) == (0.9, 0.4)


def test_tool_scale():
    config.FIDELITY, config.FIDELITY_TURN_START = True, 1
    config.PRUNE, config.WAIT_ABOVE = 0.8, 0.4
    config.FIDELITY_SCALE, config.FIDELITY_TOOL_SCALE = 0.6, 0.0
    fidelity.configure(_Tok())
    fidelity.begin_turn()
    p, w = fidelity.adjust(0.8, 0.4)
    assert abs(p - 0.48) < 1e-9 and abs(w - 0.24) < 1e-9, "turn start keeps FIDELITY_SCALE"
    for t in (1, START):
        fidelity.observe(t)
    assert fidelity.adjust(0.9, 0.4) == (0.0, 0.0), "inside a call runs exact"
    for t in (3, 4, END, 5):
        fidelity.observe(t)
    assert fidelity.adjust(0.9, 0.4) == (0.9, 0.4)
    fidelity.end_turn()
    assert fidelity.tool_tokens() == 3, fidelity.tool_tokens()
    config.FIDELITY_TOOL_SCALE = config.FIDELITY_SCALE


def test_no_markers():
    config.FIDELITY, config.FIDELITY_TURN_START = True, 1
    fidelity.configure(object())
    flags, _ = _run([START, 3, END, 4])
    assert flags == [True, False, False, False], flags


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
