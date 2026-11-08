# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
The think budget closes reasoning inside the same decode.

Under the budget the logits must come back unchanged (bit for bit). Past it,
the processor waits for a sentence end, then forces the close text one token
per step and stops once the think end tag is out. It must give the same
answer for the same prefix, since a verify pass replays rows after a rewind.

Needs a Qwen3 tokenizer (qwen3-235-4bit, or PAGED_MOE_TEST_TOKENIZER).

Run:
    python tests/test_think_budget.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
from mlx_lm.tokenizer_utils import load as load_tokenizer

from pagedmoe import server
from pagedmoe.think_budget import SLACK_TOKENS

REASONING = (
    "The user wants the footer links to wrap on small screens. The Row in "
    "footer.dart has no wrap, so it overflows below 400 px. I should swap it "
    "for a Wrap with spacing and keep the same order. Then the test that "
    "pumps a 360 px surface should pass without the overflow error. "
) * 4
ANSWER = " I will edit footer.dart now."


def run(tok, budget, script, vocab):
    """Greedy decode that follows `script` unless the processor forces a token."""
    budget(mx.array([7], dtype=mx.uint32), mx.zeros((1, vocab)))
    out, j = [], 0
    while len(out) < len(script) + 64:
        want = script[j] if j < len(script) else 0
        logits = mx.zeros((1, vocab)).at[0, want].add(5.0)
        got = budget(mx.array([7] + out, dtype=mx.uint32), logits)
        pick = int(mx.argmax(got, axis=-1).item())
        assert got[0, pick].item() == logits[0, pick].item(), "forced logit changed"
        out.append(pick)
        if pick == want:
            j += 1
        if pick == tok.think_end_id:
            break
    return out


def main():
    path = os.environ.get(
        "PAGED_MOE_TEST_TOKENIZER", str(Path.home() / "mlx-models/qwen3-235-4bit")
    )
    if not Path(path, "tokenizer_config.json").exists():
        print(f"skip: no tokenizer at {path}")
        return 0
    tok = load_tokenizer(Path(path))
    vocab = len(tok.get_vocab())
    script = tok.encode(REASONING + ANSWER, add_special_tokens=False)
    chars = 300

    # Under the budget nothing changes.
    b = server._think_budget_for(tok, {"chars": 10**6, "close": "Acting now."})
    b(mx.array([7], dtype=mx.uint32), mx.zeros((1, vocab)))
    logits = mx.random.normal((1, vocab))
    same = b(mx.array([7] + script[:40], dtype=mx.uint32), logits)
    assert mx.array_equal(same, logits).item(), "logits changed under the budget"

    b = server._think_budget_for(tok, {"chars": chars, "close": "Acting now."})
    assert b is not None and b.close_ids[-1] == tok.think_end_id
    out = run(tok, b, script, vocab)
    assert b.closed_in(out), "close text never forced"
    k = len(b.close_ids)
    start = next(i for i in range(len(out)) if out[i : i + k] == b.close_ids)
    before = tok.decode(out[:start])
    assert len(before) >= chars, f"closed early at {len(before)}ch"
    assert out[:start] == script[:start], "reasoning before the close was altered"
    first = next(i for i in range(len(out)) if len(tok.decode(out[: i + 1])) >= chars)
    assert start - first <= SLACK_TOKENS + 1, "waited past the slack"
    assert before.rstrip(" ").endswith((".", "!", "?", ":", "\n")) or start - first > SLACK_TOKENS, (
        f"did not stop on a sentence end: {before[-40:]!r}"
    )

    # Same prefix, same answer: replay rows the way a verify pass does.
    b2 = server._think_budget_for(tok, {"chars": chars, "close": "Acting now."})
    b2(mx.array([7], dtype=mx.uint32), mx.zeros((1, vocab)))
    for n in range(start - 2, start + k):
        row = mx.zeros((1, vocab)).at[0, 11].add(5.0)
        a = b2(mx.array([7] + out[:n], dtype=mx.uint32), row)
        c = b2(mx.array([7] + out[:n], dtype=mx.uint32), row)
        assert mx.array_equal(a, c).item(), f"row {n} not deterministic"
        pick = int(mx.argmax(a, axis=-1).item())
        assert pick == out[n] or n < start, f"row {n} picked {pick}, decode had {out[n]}"

    # After the end tag, logits pass through again.
    row = mx.random.normal((1, vocab))
    after = b2(mx.array([7] + out[: start + k] + [11, 12], dtype=mx.uint32), row)
    assert mx.array_equal(after, row).item(), "processor still active after close"

    # Template that never opens <think> and a reply that never does either
    # (Coder-Next writing its plan as text straight into a tool call): no force.
    b3 = server._think_budget_for(tok, {"chars": chars, "close": "Acting now."})
    b3.open_in_prompt = False
    plain = run(tok, b3, script, vocab)
    assert not b3.closed_in(plain), "forced a close into a reply that never opened reasoning"
    assert plain[: len(script)] == script, "plain reply was altered"

    # Same template, but the model opens <think> itself: counts from the tag.
    lead = tok.encode("Plan first.", add_special_tokens=False)
    b4 = server._think_budget_for(tok, {"chars": chars, "close": "Acting now."})
    b4.open_in_prompt = False
    own = run(tok, b4, lead + [tok.think_start_id] + script, vocab)
    assert b4.closed_in(own), "own <think> block never closed"
    at = next(i for i in range(len(own)) if own[i : i + k] == b4.close_ids)
    opened_at = own.index(tok.think_start_id)
    assert len(tok.decode(own[opened_at + 1 : at])) >= chars, "counted text before <think>"

    for bad in ({}, {"chars": 0, "close": "x"}, {"chars": 10, "close": ""}, "nope"):
        assert server._think_budget_for(tok, bad) is None

    print(f"ok: closed at {len(before)}ch, {start - first} tokens past the budget, {k} forced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
