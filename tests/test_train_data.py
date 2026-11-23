# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat rows through Formatter with real Jinja templates.

    python tests/test_train_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jinja2 import Environment

from pagedmoe.train.data import Formatter

# Qwen3's original template strips content with no null check
STRICT = (
    "{%- for m in messages %}<|{{ m.role }}|>"
    "{%- if m.role == 'assistant' %}{{ m.content.lstrip('\\n') }}"
    "{%- for c in m.tool_calls or [] %}<call>{{ c.function.name }}</call>{%- endfor %}"
    "{%- else %}{{ m.content }}{%- endif %}|{%- endfor %}"
    "{%- if add_generation_prompt %}<|assistant|>{%- endif %}"
)
# newer templates handle null themselves
LENIENT = (
    "{%- for m in messages %}<|{{ m.role }}|>"
    "{%- if m.content is not none %}{{ m.content }}{%- else %}(none){%- endif %}"
    "{%- for c in m.tool_calls or [] %}<call>{{ c.function.name }}</call>{%- endfor %}|"
    "{%- endfor %}{%- if add_generation_prompt %}<|assistant|>{%- endif %}"
)


class Tok:
    eos_token_id = 1
    pad_token_id = 0

    def __init__(self, template):
        self.chat_template = template
        self._t = Environment().from_string(template)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        return self._t.render(messages=messages, add_generation_prompt=add_generation_prompt, **kw)

    def encode(self, text, add_special_tokens=False):
        return [2 + b for b in text.encode()]


ROW = [
    {"role": "user", "content": "weather?"},
    {"role": "assistant", "content": None,
     "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
    {"role": "tool", "content": "sunny"},
    {"role": "assistant", "content": "It is sunny."},
]


def main():
    strict = Formatter(Tok(STRICT), max_len=512).render(ROW)
    assert "<|assistant|><call>get_weather</call>|" in strict, strict

    lenient = Formatter(Tok(LENIENT), max_len=512).render(ROW)
    assert "(none)<call>get_weather</call>" in lenient, "a template that handles null must render as is"

    ex = Formatter(Tok(STRICT), max_len=512).example({"messages": ROW})
    assert sum(ex.loss_mask) > 0

    broken = Formatter(Tok("{{ messages[0].nope.lstrip() }}"), max_len=512)
    try:
        broken.render([{"role": "user", "content": "hi"}])
    except ValueError as e:
        assert "chat template failed" in str(e), e
    else:
        raise AssertionError("a broken template should raise")
    check_split()
    print("OK")


def check_split():
    import random

    from pagedmoe.train.data import Schedule

    rng = random.Random(1)
    s = Schedule([[0, 1]], 1, 4096, max_rows=6)
    for _ in range(200):
        lens = sorted((32 * rng.randint(1, 160) for _ in range(rng.randint(1, 20))), reverse=True)
        cuts = s.split(lens)
        assert cuts[0][0] == 0 and cuts[-1][1] == len(lens)
        assert all(a < b and b == c for (a, b), (c, _) in zip(cuts, cuts[1:] + [(len(lens), 0)]))
        for a, b in cuts:
            assert b - a == 1 or ((b - a) * lens[a] <= 4096 and b - a <= 6)
        # never worse than filling greedily
        g, cur = [], []
        for n in lens:
            if cur and ((len(cur) + 1) * cur[0] > 4096 or len(cur) + 1 > 6):
                g.append(cur)
                cur = []
            cur.append(n)
        g.append(cur)
        cost = lambda groups: sum(len(m) * m[0] + Schedule.MICRO_COST for m in groups)
        assert cost([lens[a:b] for a, b in cuts]) <= cost(g)


def test_train_data():
    main()


if __name__ == "__main__":
    main()
