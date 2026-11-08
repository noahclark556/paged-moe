# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Replay splice keeps every image token of a vision prompt.

vision_chat expands each picture to one pad per vision feature. The splice
re-renders the messages to find replayed turns, and the template gives one
pad per picture. Before expand_rendered, a picture followed by a tool round
went out with 1 image token instead of 1632, and the model answered without
seeing the picture.

Needs a Qwen3.5 / 3.6 vision tokenizer (Qwen36-35B-4bit, or
PAGED_MOE_TEST_VISION_TOKENIZER). Loads no weights.

Run:
    python tests/test_vision_splice.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import LRUPromptCache
from mlx_lm.tokenizer_utils import load as load_tokenizer

from pagedmoe import replay_splice, vision_chat

KEY = ("vision", None, None)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Open a document.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]
KW = {"tools": TOOLS, "enable_thinking": False}
GRID = [1, 8, 12]  # 96 patches, 24 tokens after the 2x2 merge
DIGEST = "0123456789ab"
PNG = "data:image/png;base64,AAAA"


def tokenize(tok, messages):
    """What vision_chat._vision_tokenize hands mlx-lm for these messages."""
    rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **KW)
    return list(tok.encode(vision_chat._expand_pads(rendered, [GRID], 2, [DIGEST]), add_special_tokens=False))


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


def main():
    path = os.environ.get(
        "PAGED_MOE_TEST_VISION_TOKENIZER", str(Path.home() / "mlx-models/Qwen36-35B-4bit")
    )
    if not Path(path, "tokenizer_config.json").exists():
        print(f"skip: no tokenizer at {path}")
        return
    tok = load_tokenizer(Path(path))
    image_id = tok.convert_tokens_to_ids("<|image_pad|>")
    eos = tok.encode("<|im_end|>", add_special_tokens=False)
    replay_splice.reset()
    vision_chat._tls.state = {
        "image_id": int(image_id), "feats": None, "cursor": 0, "done": False,
        "grids": [GRID], "merge": 2, "digests": [DIGEST],
    }
    try:
        base = [
            {"role": "system", "content": "You help a law office."},
            {"role": "user", "content": [
                {"type": "text", "text": "Make these whiteboard notes into next steps with owners."},
                {"type": "image_url", "image_url": {"url": PNG}},
            ]},
        ]
        p1 = tokenize(tok, base)
        assert p1.count(image_id) == 24, p1.count(image_id)
        emitted = (
            "<tool_call>\n<function=read_document>\n<parameter=path>\n"
            "Clients/Patterson/engagement letter.docx\n</parameter>\n</function>\n</tool_call>"
        )
        g1 = odd_spelling(tok, emitted) + eos
        store = LRUPromptCache(max_size=8)
        store._trie.add(KEY, p1 + g1, "turn 1")

        call = {"function": {"name": "read_document",
                             "arguments": {"path": "Clients/Patterson/engagement letter.docx"}}}
        m2 = base + [
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "content": "[p1] Engagement letter for Patterson."},
        ]
        plain = tokenize(tok, m2)
        assert plain[: len(p1) + len(g1)] != p1 + g1, "pick a turn whose plain encode differs"
        new = replay_splice.apply(tok, store, KEY, m2, KW, plain)
        assert new is not None, "the tool round should splice"
        assert new[: len(p1) + len(g1)] == p1 + g1, "prompt must extend the stored turn"
        assert new.count(image_id) == plain.count(image_id) == 24, new.count(image_id)
        assert vision_chat.image_tokens(new) == vision_chat.image_tokens(plain)
        assert f"[image {DIGEST[:8]}]" in tok.decode(new), "content tag must survive the splice"
        print(f"spliced {len(new)} tok vs {len(plain)} plain, all 24 image tokens kept")

        try:
            vision_chat.expand_rendered("<|image_pad|>" * 2)
        except RuntimeError as e:
            assert "image slots" in str(e), e
        else:
            raise AssertionError("two slots for one image must not expand")
    finally:
        vision_chat._tls.state = None

    # No pictures in this request: text is left alone.
    assert vision_chat.expand_rendered("a <|image_pad|> b") == "a <|image_pad|> b"
    assert vision_chat.image_tokens([1, 2, 3]) is None
    print("all ok")


if __name__ == "__main__":
    main()
