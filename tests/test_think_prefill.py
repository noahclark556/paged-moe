# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
A think-budget resume reuses the cut pass's own tokens.

The cut pass stores prompt + the reasoning ids decode produced. The client
sends that reasoning back as text, and encoding it again gives different ids
wherever the sampled spelling isn't the tokenizer's first choice, so the
store used to match a few tokens in and prefill the rest again. The resume
prompt must spell the reused part with the stored ids, decode to exactly the
same text as before, and only encode what the cut pass never wrote.

Needs a Qwen3 tokenizer (qwen3-235-4bit, or PAGED_MOE_TEST_TOKENIZER).

Run:
    python tests/test_think_prefill.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import LRUPromptCache
from mlx_lm.tokenizer_utils import load as load_tokenizer

from pagedmoe import server

REASONING = (
    "Okay, so the user wants't to adjust the tools's menu so the remix option "
    "stays visible when collapsed. The _ToolDef list in myapp_toolbox.dart "
    "builds modelTools; remix should move into that list."
)
COMMIT = "I will begin making these changes now."


class Gen:
    def __init__(self, store):
        self.prompt_cache = store
        self.model_provider = type("P", (), {"model_key": "m"})()


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
        "PAGED_MOE_TEST_TOKENIZER", str(Path.home() / "mlx-models/qwen3-235-4bit")
    )
    if not Path(path, "tokenizer_config.json").exists():
        print(f"skip: no tokenizer at {path}")
        return
    tok = load_tokenizer(Path(path))

    prompt = tok.encode("<|im_start|>user\nfix it<|im_end|>\n<|im_start|>assistant\n<think>\n",
                        add_special_tokens=False)
    cut = odd_spelling(tok, REASONING + " Then I")
    canonical = tok.encode(REASONING, add_special_tokens=False)
    assert tok.decode(cut).startswith(REASONING)
    first_diff = next(i for i, (a, b) in enumerate(zip(cut, canonical)) if a != b)
    assert first_diff < len(canonical) // 2, "pick a text whose spellings differ early"

    text = REASONING + "\n\n" + COMMIT + "\n</think>\n\n"
    store = LRUPromptCache(max_size=4)
    store._trie.add("m", prompt, "prompt snapshot")
    store._trie.add("m", prompt + cut, "cut pass")
    store._trie.add("m", prompt[:-3], "older turn")

    own, rest = server._reuse_cut_tokens(Gen(store), tok, prompt, text)
    full = prompt + own + tok.encode(rest, add_special_tokens=False)
    assert own == cut[: len(own)], "reused ids must be the stored ones"
    assert tok.decode(own) == text[: len(text) - len(rest)]
    assert tok.decode(full[len(prompt):]) == text, "resume text changed"
    stored = len(prompt) + len(own)
    print(
        f"reused {len(own)}/{len(cut)} cut tokens; a fresh encode matched {first_diff}; "
        f"prefill {len(full) - stored} instead of {len(full) - len(prompt) - first_diff}"
    )
    assert len(own) >= len(cut) - 3

    # Leading whitespace the client stripped stays as the model's own tokens.
    lead = tok.encode("\n\n", add_special_tokens=False) + cut
    store2 = LRUPromptCache(max_size=4)
    store2._trie.add("m", prompt + lead, "cut pass")
    own2, rest2 = server._reuse_cut_tokens(Gen(store2), tok, prompt, text)
    assert own2[:2] == lead[:2] and len(own2) >= len(lead) - 3
    assert tok.decode(own2).lstrip() + rest2 == text

    # Nothing stored past the prompt, or a different turn: plain encode.
    for other in ([], tok.encode("Something else entirely.", add_special_tokens=False)):
        s = LRUPromptCache(max_size=4)
        s._trie.add("m", prompt + other, "x")
        assert server._reuse_cut_tokens(Gen(s), tok, prompt, text) == ([], text)
    assert server._reuse_cut_tokens(Gen(LRUPromptCache()), tok, prompt, text) == ([], text)
    print("all ok")


if __name__ == "__main__":
    main()
