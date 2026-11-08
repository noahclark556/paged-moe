# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Replayed assistant turns are spelled with the ids decode produced.

The store holds prompt + generated ids after each turn. The host sends the
turn back as text with its arguments trimmed, and a fresh encode of the
rendered text spells it differently, so a hybrid cache never found the stored
entry as a prefix. With the splice, the next prompt must start with exactly
prompt + generated ids, keep doing so turn after turn (append-only), and fall
back to the plain encode for a turn whose text really changed.

Needs a chat template with tools (qwen38-coder-flash, or
PAGED_MOE_TEST_TOKENIZER).

Run:
    python tests/test_replay_splice.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import LRUPromptCache
from mlx_lm.tokenizer_utils import load as load_tokenizer

from pagedmoe import config, replay_splice

KEY = ("flash", None, None)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "replace_string_in_file",
            "description": "Replace one occurrence of oldString with newString.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "oldString": {"type": "string"},
                    "newString": {"type": "string"},
                },
                "required": ["filePath", "oldString", "newString"],
            },
        },
    }
]
KW = {"tools": TOOLS, "preserve_thinking": True, "reasoning_effort": "low"}


def odd_spelling(tok, text):
    """Ids for `text` the way sampling might spell it: long words in two pieces."""
    out = []
    for i, word in enumerate(text.split(" ")):
        piece = word if i == 0 else " " + word
        cut = len(piece) // 2 if len(word) > 6 else len(piece)
        for part in (piece[:cut], piece[cut:]):
            if part:
                out += tok.encode(part, add_special_tokens=False)
    return out


def ids(x):
    return list(x["input_ids"]) if isinstance(x, dict) or hasattr(x, "keys") else list(x)


def render(tok, messages):
    return ids(tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **KW))


def call(path, old, new):
    return {"function": {"name": "replace_string_in_file",
                         "arguments": {"filePath": path, "oldString": old, "newString": new}}}


def emitted(reasoning, path, old, new):
    """What the model writes after the generation prompt, before EOS."""
    return (
        f"{reasoning}\n</think>\n\n<tool_call>\n<function=replace_string_in_file>\n"
        f"<parameter=filePath>\n{path}\n</parameter>\n"
        f"<parameter=oldString>\n{old}\n</parameter>\n"
        f"<parameter=newString>\n{new}\n</parameter>\n"
        f"</function>\n</tool_call>"
    )


def main():
    path = os.environ.get(
        "PAGED_MOE_TEST_TOKENIZER", str(Path.home() / "mlx-models/qwen38-coder-flash")
    )
    if not Path(path, "tokenizer_config.json").exists():
        print(f"skip: no tokenizer at {path}")
        return
    tok = load_tokenizer(Path(path))
    eos = tok.encode("<|im_end|>", add_special_tokens=False)
    replay_splice.reset()

    base = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Make the counter start at two and count by three."},
    ]
    r1 = "The counter initializer sits in lib/counter.dart; changing the starting value is enough."
    old1, new1 = "  int value = 0;", "  int value = 2;"
    p1 = render(tok, base)
    # Sampled spelling, and the host trims leading indentation off arguments.
    g1 = odd_spelling(tok, emitted(r1, "lib/counter.dart", old1, new1)) + eos
    store = LRUPromptCache(max_size=8)
    store._trie.add(KEY, p1 + g1, "turn 1")

    turn1 = {"role": "assistant", "content": "", "reasoning_content": r1,
             "tool_calls": [call("lib/counter.dart", old1.strip(), new1.strip())]}
    m2 = base + [turn1, {"role": "tool", "content": "Changed line 4 of 30."}]
    plain2 = render(tok, m2)
    assert plain2[: len(p1) + len(g1)] != p1 + g1, "pick a turn whose plain encode differs"
    n2 = replay_splice.apply(tok, store, KEY, m2, KW, plain2)
    assert n2 is not None, "turn 1 should splice"
    assert n2[: len(p1) + len(g1)] == p1 + g1, "prompt 2 must extend the stored turn"
    shared = next((i for i, (a, b) in enumerate(zip(plain2, p1 + g1)) if a != b), len(p1))
    print(f"turn 1: reuse {len(p1) + len(g1)} tok, a plain encode matched {shared}")

    # Turn 2 extends prompt 2; prompt 3 must spell both turns the same way again.
    r2 = "Now the increment: the step lives in the increment method."
    old2, new2 = "    value += 1;", "    value += 3;"
    g2 = odd_spelling(tok, emitted(r2, "lib/counter.dart", old2, new2)) + eos
    store._trie.add(KEY, n2 + g2, "turn 2")
    turn2 = {"role": "assistant", "content": "", "reasoning_content": r2,
             "tool_calls": [call("lib/counter.dart", old2.strip(), new2.strip())]}
    m3 = m2 + [turn2, {"role": "tool", "content": "Changed line 9 of 30."}]
    n3 = replay_splice.apply(tok, store, KEY, m3, KW, render(tok, m3))
    assert n3 is not None and n3[: len(n2) + len(g2)] == n2 + g2, "prompt 3 must extend turn 2"
    tail = tok.decode(n3[len(n2) + len(g2):])
    assert tail.startswith("\n<|im_start|>user\n<tool_response>\nChanged line 9"), tail[:80]
    print(f"turn 2: reuse {len(n2) + len(g2)} of {len(n3)} tok")

    # A host note and ack after the tool result, with neither decoded turn
    # learned yet: both are found past the ack and learned oldest first.
    replay_splice.reset()
    m4 = m3 + [{"role": "user", "content": "Note: files already searched."},
               {"role": "assistant", "content": "Noted."}]
    n4 = replay_splice.apply(tok, store, KEY, m4, KW, render(tok, m4))
    assert n4 is not None and n4[: len(n2) + len(g2)] == n2 + g2, "both turns should splice past the ack"
    print(f"after ack: reuse {len(n2) + len(g2)} of {len(n4)} tok")

    # The host changed turn 2 for real: turn 1 still splices, turn 2 encodes plain.
    edited = dict(turn2, tool_calls=[call("lib/other.dart", old2.strip(), new2.strip())])
    m3b = m2 + [edited, {"role": "tool", "content": "Changed line 9 of 30."}]
    plain3b = render(tok, m3b)
    n3b = replay_splice.apply(tok, store, KEY, m3b, KW, plain3b)
    assert n3b is not None and n3b[: len(p1) + len(g1)] == p1 + g1
    assert n3b[: len(n2) + len(g2)] != n2 + g2
    sent, spelled = tok.decode(plain3b), tok.decode(n3b)
    assert replay_splice.align(sent, spelled) == len(sent), "only whitespace may differ"
    assert "lib/other.dart" in spelled

    # Nothing stored under the previous prompt: plain encode.
    replay_splice.reset()
    assert replay_splice.apply(tok, LRUPromptCache(), KEY, m2, KW, plain2) is None

    # Off switch.
    replay_splice.reset()
    config.REPLAY_SPLICE = False
    try:
        assert replay_splice.apply(tok, store, KEY, m2, KW, plain2) is None
    finally:
        config.REPLAY_SPLICE = True

    assert replay_splice.align("a  b\n", "a b") == 4
    assert replay_splice.align("ab", "a b") == 2
    assert replay_splice.align("abc", "abd") is None
    print("all ok")


if __name__ == "__main__":
    main()
