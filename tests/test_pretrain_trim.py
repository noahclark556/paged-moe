# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prompt-trim for sidecar pretrain: drop tool spam, keep task signal."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from expert_stream.pretrain_sidecar import (  # noqa: E402
    CorpusItem,
    build_prompt,
    trim_messages,
    trim_text,
)


class _FakeTok:
    """Whitespace-ish tokenizer: 1 token ≈ 1 word."""

    def encode(self, text: str):
        return text.split() if text.strip() else []

    def decode(self, ids):
        return " ".join(ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        body = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return body + ("\nassistant:" if add_generation_prompt else "")


def test_trim_drops_tools_keeps_first_user():
    tok = _FakeTok()
    msgs = [
        {"role": "system", "content": "You are a coding agent " + "x " * 100},
        {"role": "user", "content": "Rename Best-in-Class on the landing page please"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "<!doctype html>" + (" dump " * 500)},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "2"}]},
        {"role": "tool", "content": "<html>" + (" more " * 500)},
        {"role": "assistant", "content": "I updated the brand string."},
    ]
    out = trim_messages(msgs, tok, max_tokens=80)
    roles = [m["role"] for m in out]
    assert "tool" not in roles
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert "Best-in-Class" in out[1]["content"]
    # No giant HTML
    blob = " ".join(m["content"] for m in out)
    assert "doctype" not in blob.lower()
    assert sum(len(tok.encode(m["content"])) for m in out) <= 80


def test_trim_text_keeps_head():
    tok = _FakeTok()
    text = " ".join(f"w{i}" for i in range(200))
    clipped = trim_text(text, tok, 20)
    assert clipped.startswith("w0 ")
    assert len(tok.encode(clipped)) <= 20


def test_build_prompt_chat():
    tok = _FakeTok()
    item = CorpusItem(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "do the thing"},
            {"role": "tool", "content": "noise"},
        ]
    )
    prompt = build_prompt(tok, item, 64)
    assert "do the thing" in prompt
    assert "noise" not in prompt


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"OK: pretrain trim ({len(tests)} tests)")
