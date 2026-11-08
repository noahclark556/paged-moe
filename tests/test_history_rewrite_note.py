# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The history-rewritten note: real rewrites only, not a new chat after a warmup."""

from pagedmoe import server


class _Tok:
    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def _note(prev, tokens, stable_at=0, capsys=None):
    server._recent_prompts.clear()
    server._recent_prompts.append(list(prev))
    server._note_history_rewrite(_Tok(), list(tokens), stable_at)
    return capsys.readouterr().out


def test_new_chat_after_the_stable_front_is_not_a_rewrite(capsys):
    front = list(range(4000))
    warm = front + [9000 + i for i in range(17)]
    ask = front + [7000 + i for i in range(60)]
    assert "history rewritten" in _note(warm, ask, 0, capsys)
    assert _note(warm, ask, len(front), capsys) == ""


def test_a_change_inside_the_front_is_still_a_rewrite(capsys):
    front = list(range(4000))
    warm = front + [9000 + i for i in range(17)]
    moved = front[:3500] + [5] * 500 + [7000 + i for i in range(60)]
    assert "history rewritten at token 3500" in _note(warm, moved, len(front), capsys)


def test_a_change_deep_in_a_chat_is_still_a_rewrite(capsys):
    front = list(range(4000))
    chat = front + [8000 + i for i in range(3000)]
    edited = front + [8000 + i for i in range(1000)] + [6] * 2000
    assert "history rewritten at token 5000" in _note(chat, edited, len(front), capsys)
