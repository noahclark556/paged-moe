# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Template kwargs (chat_compat.install_template_kwargs).

The templates here are cut down from the real ones, keeping only the parts
that read thinking and effort kwargs:
  - MiniMax-M3 switches on thinking_mode and never reads enable_thinking
  - DeepSeek-V3.1 switches on a bare `thinking`
  - Flash keeps prior thinking when preserve_thinking is unset and raises on
    an effort outside (xhigh, medium, low)
  - Qwen3.6, GLM-5.2 and gpt-oss need nothing and must render as before

Rendering goes through mlx-lm's real TokenizerWrapper.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jinja2.exceptions import TemplateError  # noqa: E402
from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402
from mlx_lm.tokenizer_utils import TokenizerWrapper  # noqa: E402

from pagedmoe import chat_compat, config  # noqa: E402

MINIMAX_M3 = """
{%- if thinking_mode is defined -%}
{%- if thinking_mode == "enabled" -%}mode: enabled
{%- elif thinking_mode == "disabled" -%}mode: disabled
{%- else -%}mode: adaptive{%- endif -%}
{%- else -%}mode: adaptive{%- endif %}
{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endfor %}
{%- if add_generation_prompt %}<assistant>{% endif %}"""

DEEPSEEK_V31 = """{% if not thinking is defined %}{% set thinking = false %}{% endif %}
{%- for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}
{%- if add_generation_prompt %}<assistant>{% if thinking %}<think>{% else %}</think>{% endif %}{% endif %}"""

FLASH = """
{%- if enable_thinking is undefined or enable_thinking is true %}
{%- set r = reasoning_effort|default('xhigh') %}
{%- if r not in ('xhigh', 'medium', 'low') %}
{{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort) }}
{%- endif %}effort: {{ r }}
{%- endif %}
{%- set ns = namespace(last=-1) %}
{%- for m in messages %}{% if m.role == 'user' %}{% set ns.last = loop.index0 %}{% endif %}{% endfor %}
{%- for m in messages %}<{{ m.role }}>
{%- if m.role == 'assistant' and m.reasoning_content is defined %}
{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last %}<think>{{ m.reasoning_content }}</think>{% endif %}
{%- endif %}{{ m.content }}{% endfor %}
{%- if add_generation_prompt %}<assistant>{% endif %}"""

QWEN36 = """{%- set ns = namespace(last=-1) %}
{%- for m in messages %}{% if m.role == 'user' %}{% set ns.last = loop.index0 %}{% endif %}{% endfor %}
{%- for m in messages %}<{{ m.role }}>
{%- if m.role == 'assistant' and m.reasoning_content is defined %}
{%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last) %}<think>{{ m.reasoning_content }}</think>{% endif %}
{%- endif %}{{ m.content }}{% endfor %}
{%- if add_generation_prompt %}<assistant>{% if enable_thinking is defined and enable_thinking is false %}<think></think>{% endif %}{% endif %}"""

GLM52 = """{%- set e = 'high' if reasoning_effort is defined and reasoning_effort == 'high' else 'max' -%}
{%- if enable_thinking is not defined or enable_thinking %}Reasoning Effort: {{ e }}{% endif %}
{%- for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}"""

GPT_OSS = """{%- if reasoning_effort is not defined %}{%- set reasoning_effort = "medium" %}{%- endif %}
Reasoning: {{ reasoning_effort }}
{%- for m in messages %}<{{ m.role }}>{% if "thinking" in m %}{{ m.thinking }}{% endif %}{{ m.content }}{% endfor %}"""

HISTORY = [
    {"role": "user", "content": "First"},
    {"role": "assistant", "content": "Answer", "reasoning_content": "old thoughts"},
    {"role": "user", "content": "Second"},
]
HI = [{"role": "user", "content": "Hi"}]


def _raise(msg):
    raise TemplateError(msg)


class FakeHF:
    """Just enough of a Hugging Face tokenizer for TokenizerWrapper."""

    eos_token_id = 0

    def __init__(self, template):
        self.chat_template = template
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = _raise
        self._tmpl = env.from_string(template)

    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, return_dict=False, tools=None, **kw):
        out = self._tmpl.render(messages=messages, add_generation_prompt=add_generation_prompt, tools=tools, **kw)
        return self.encode(out) if tokenize else out


def _pair(template):
    """(stock wrapper, wrapper with the rules installed, rules)."""
    plain = TokenizerWrapper(FakeHF(template))
    fixed = TokenizerWrapper(FakeHF(template))
    rules = chat_compat.install_template_kwargs(fixed)
    return plain, fixed, rules


def _render(tok, msgs=HI, **kw):
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)


def test_minimax_m3_enable_thinking_reaches_thinking_mode():
    plain, fixed, rules = _pair(MINIMAX_M3)
    assert rules is not None and rules.thinking_mode == ("enabled", "disabled")
    assert "mode: disabled" in _render(fixed, enable_thinking=False)
    assert "mode: enabled" in _render(fixed, enable_thinking=True)
    # Stock: the switch never reached the template.
    assert "mode: adaptive" in _render(plain, enable_thinking=False)
    # Nothing sent keeps the template default; thinking_mode sent wins.
    assert _render(fixed) == _render(plain)
    for mode in ("enabled", "disabled", "adaptive"):
        for on in (True, False):
            assert _render(fixed, thinking_mode=mode, enable_thinking=on) == _render(
                plain, thinking_mode=mode, enable_thinking=on
            )


def test_deepseek_v31_enable_thinking_reaches_thinking():
    plain, fixed, rules = _pair(DEEPSEEK_V31)
    assert rules is not None and rules.thinking_flag
    assert _render(fixed, enable_thinking=True).endswith("<think>")
    assert _render(fixed, enable_thinking=False).endswith("</think>")
    assert _render(plain, enable_thinking=True).endswith("</think>")
    assert _render(fixed) == _render(plain)
    assert _render(fixed, enable_thinking=True, thinking=False) == _render(plain, thinking=False)


def test_flash_preserve_off_unless_sent():
    plain, fixed, rules = _pair(FLASH)
    assert rules is not None and rules.preserve_off
    assert "old thoughts" in _render(plain, HISTORY)
    assert "old thoughts" not in _render(fixed, HISTORY)
    assert _render(fixed, HISTORY, preserve_thinking=True) == _render(plain, HISTORY, preserve_thinking=True)
    assert _render(fixed, HISTORY, preserve_thinking=False) == _render(plain, HISTORY, preserve_thinking=False)


def test_flash_effort_moves_to_nearest_accepted():
    plain, fixed, rules = _pair(FLASH)
    assert rules.efforts == ("low", "medium", "xhigh")
    for sent, want in (("high", "xhigh"), ("max", "xhigh"), ("minimal", "low"), ("none", "low"), ("HIGH", "xhigh")):
        assert f"effort: {want}" in _render(fixed, enable_thinking=True, reasoning_effort=sent), sent
    try:
        _render(plain, enable_thinking=True, reasoning_effort="high")
        raise AssertionError("stock template should raise on high")
    except TemplateError:
        pass
    # Accepted values pass through; unknown words fall back to the default.
    for ok in ("low", "medium", "xhigh"):
        kw = {"enable_thinking": True, "reasoning_effort": ok}
        assert _render(fixed, **kw) == _render(plain, preserve_thinking=False, **kw)
    assert "effort: xhigh" in _render(fixed, enable_thinking=True, reasoning_effort="banana")


def test_templates_that_need_nothing_are_left_alone():
    for template in (QWEN36, GLM52, GPT_OSS):
        plain, fixed, rules = _pair(template)
        assert rules is None
        assert "apply_chat_template" not in vars(fixed)
        for kw in ({}, {"enable_thinking": False}, {"reasoning_effort": "low"}, {"reasoning_effort": "high"}):
            assert _render(fixed, HISTORY, **kw) == _render(plain, HISTORY, **kw)


def test_switch_off_leaves_tokenizer_untouched():
    prev = config.TEMPLATE_KWARGS
    config.TEMPLATE_KWARGS = False
    try:
        plain, fixed, rules = _pair(FLASH)
        assert rules is None
        assert _render(fixed, HISTORY) == _render(plain, HISTORY)
        assert chat_compat.lift_request_kwargs({"reasoning_effort": "low"}) is None
    finally:
        config.TEMPLATE_KWARGS = prev


def test_install_is_idempotent_and_tokenizes():
    _, fixed, rules = _pair(FLASH)
    assert chat_compat.install_template_kwargs(fixed) is rules
    ids = fixed.apply_chat_template(
        HI, add_generation_prompt=True, enable_thinking=True, reasoning_effort="high"
    )
    assert "effort: xhigh" in "".join(chr(i) for i in ids)


def test_deepseek_v32_python_template():
    chat_compat.install_deepseek_v32_chat_template()
    import mlx_lm.chat_templates.deepseek_v32 as dsv32

    def render(**kw):
        return dsv32.apply_chat_template(HISTORY, add_generation_prompt=True, **kw)

    assert render(enable_thinking=True).endswith("<think>")
    assert render(enable_thinking=False).endswith("</think>")
    assert render(thinking_mode="chat", enable_thinking=True).endswith("</think>")
    # Kwargs meant for Jinja templates must not reach encode_messages.
    assert render(enable_thinking=True, reasoning_effort="high", preserve_thinking=True)
    # A bare assistant turn after the last user gets a stub instead of failing.
    ack = HISTORY + [{"role": "assistant", "content": "Noted."}]
    assert dsv32.apply_chat_template(ack, enable_thinking=True)


def test_nearest_effort():
    acc = ("low", "medium", "xhigh")
    assert chat_compat.nearest_effort("medium", acc) == "medium"
    assert chat_compat.nearest_effort("high", acc) == "xhigh"
    assert chat_compat.nearest_effort("max", acc) == "xhigh"
    assert chat_compat.nearest_effort("none", acc) == "low"
    assert chat_compat.nearest_effort("auto", acc) is None
    assert chat_compat.nearest_effort("high", ("custom",)) is None


def test_lift_request_kwargs():
    lift = chat_compat.lift_request_kwargs
    assert lift({"messages": []}) is None
    assert lift({"reasoning_effort": "High"}) == {"reasoning_effort": "high"}
    assert lift({"reasoning": {"effort": "low"}}) == {"reasoning_effort": "low"}
    assert lift(
        {"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": True}}
    ) == {"enable_thinking": True, "reasoning_effort": "low"}
    # Already set by the client in chat_template_kwargs: left alone.
    assert lift({"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "xhigh"}}) is None
    assert lift({"reasoning_effort": ""}) is None
    assert lift({"reasoning_effort": None}) is None
    body = {"reasoning_effort": "low", "chat_template_kwargs": {"x": 1}}
    lift(body)
    assert body["chat_template_kwargs"] == {"x": 1}


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
