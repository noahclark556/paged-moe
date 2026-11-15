# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat-template shims so mlx-lm host kwargs match model-specific templates.

mlx-lm's TokenizerWrapper always injects ``enable_thinking``. DeepSeek-V3.2's
Python template wants ``thinking_mode`` ("thinking"|"chat") instead, so without
this map every apply_chat_template dies with:

    encode_messages() got an unexpected keyword argument 'enable_thinking'

Installed once at import from loader (and re-exported by the server) so CLI,
server, and benches that call ``load()`` all get the same path.
"""

from __future__ import annotations

import re

from . import config

_installed = False


def install_deepseek_v32_chat_template() -> None:
    global _installed
    if _installed:
        return
    try:
        import mlx_lm.chat_templates.deepseek_v32 as mod
    except ImportError:
        return

    _orig = mod.apply_chat_template

    def apply_chat_template(
        messages,
        continue_final_message=False,
        add_generation_prompt=False,
        **kwargs,
    ):
        if "thinking_mode" not in kwargs and "enable_thinking" in kwargs:
            kwargs["thinking_mode"] = (
                "thinking" if kwargs["enable_thinking"] else "chat"
            )
        kwargs.pop("enable_thinking", None)
        # Clients pass these for Jinja templates; this Python template does not.
        if "preserve_thinking" in kwargs:
            # drop_thinking=True is the template default (strip prior reasoning).
            kwargs.setdefault(
                "drop_thinking", not bool(kwargs.pop("preserve_thinking"))
            )
        else:
            kwargs.pop("preserve_thinking", None)
        kwargs.pop("reasoning_effort", None)

        # Thinking mode asserts that every assistant turn after the last user
        # has reasoning_content or tool_calls. Host apps often park a bare ACK
        # there (task_state ACK). Give those a stub so the request does not 404.
        thinking_mode = kwargs.get("thinking_mode", "thinking")
        if thinking_mode == "thinking" and isinstance(messages, list):
            last_user = -1
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], dict) and messages[i].get("role") == "user":
                    last_user = i
                    break
            fixed = []
            for i, msg in enumerate(messages):
                if (
                    i > last_user
                    and isinstance(msg, dict)
                    and msg.get("role") == "assistant"
                    and not msg.get("tool_calls")
                    and not msg.get("reasoning_content")
                ):
                    msg = {**msg, "reasoning_content": "\n"}
                fixed.append(msg)
            messages = fixed

        return _orig(
            messages,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    mod.apply_chat_template = apply_chat_template
    _installed = True


# Think tags mlx-lm's _infer_thinking does not know. It takes the first pair
# found in the vocab, and MiniMax-M3's vocab has a plain <think></think> its
# template never uses, so the server's reasoning split never fires and
# </mm:think> lands in content.
_TEMPLATE_THINK_PAIRS = (("<mm:think>", "</mm:think>"),)


def fix_think_tokens(tokenizer) -> str | None:
    """Point the wrapper's think tags at the pair the chat template uses.

    Only switches to a pair whose tags are single tokens in the vocab and both
    appear in the template. Returns the new start tag, or None if unchanged.
    """
    tmpl = getattr(tokenizer, "_chat_template", None)
    if not isinstance(tmpl, str):
        try:
            tmpl = tokenizer.chat_template
        except Exception:
            tmpl = None
    if not isinstance(tmpl, str):
        return None
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        return None
    for start, end in _TEMPLATE_THINK_PAIRS:
        if start not in tmpl or end not in tmpl:
            continue
        if start not in vocab or end not in vocab:
            continue
        if getattr(tokenizer, "_think_start", None) == start:
            return None
        tokenizer._think_start = start
        tokenizer._think_end = end
        tokenizer._think_start_tokens = (vocab[start],)
        tokenizer._think_end_tokens = (vocab[end],)
        print(f"[paged-moe] think tags from chat template: {start} {end}", flush=True)
        return start
    return None

# --- template kwargs ---------------------------------------------------------
#
# Clients send one vocabulary (enable_thinking, preserve_thinking,
# reasoning_effort) and templates read another. Each rule below is switched on
# only after rendering the template at load shows it matters, so a template
# that already handles a kwarg renders exactly as before.

_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_EFFORT_RANK = {e: i for i, e in enumerate(_EFFORTS)}

_PROBE = [{"role": "user", "content": "Hi"}]
_HISTORY = [
    {"role": "user", "content": "First question"},
    {
        "role": "assistant",
        "content": "First answer",
        "reasoning_content": "first thoughts",
    },
    {"role": "user", "content": "Second question"},
]

_ON_MODES = ("enabled", "thinking", "on")
_OFF_MODES = ("disabled", "chat", "off")
# A bare `thinking` switch (DeepSeek-V3.1), not message.thinking (gpt-oss).
_BARE_THINKING = re.compile(r"(?<![\w.'\"])thinking\s+is\s+(?:not\s+)?defined")


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(v)


def nearest_effort(value, accepted) -> str | None:
    """The accepted effort closest to `value`, ties going up. None = drop it."""
    s = str(value).strip().lower()
    if s in accepted:
        return s
    if s not in _EFFORT_RANK:
        return None
    want = _EFFORT_RANK[s]
    ranked = [a for a in accepted if a in _EFFORT_RANK]
    if not ranked:
        return None
    return min(ranked, key=lambda a: (abs(_EFFORT_RANK[a] - want), -_EFFORT_RANK[a]))


class TemplateKwargs:
    """What this template needs done to the kwargs a client sends."""

    def __init__(self):
        self.thinking_mode: tuple[str, str] | None = None
        self.thinking_flag = False
        self.preserve_off = False
        self.efforts: tuple[str, ...] | None = None

    def active(self) -> bool:
        return bool(self.thinking_mode or self.thinking_flag or self.preserve_off or self.efforts)

    def describe(self) -> str:
        parts = []
        if self.thinking_mode:
            parts.append(f"enable_thinking -> thinking_mode {self.thinking_mode[0]}/{self.thinking_mode[1]}")
        if self.thinking_flag:
            parts.append("enable_thinking -> thinking")
        if self.preserve_off:
            parts.append("preserve_thinking off unless sent")
        if self.efforts:
            parts.append(f"reasoning_effort one of {', '.join(self.efforts)}")
        return "; ".join(parts)

    def apply(self, kwargs: dict) -> dict:
        kw = dict(kwargs)
        sent = "enable_thinking" in kw
        if self.thinking_mode and sent and "thinking_mode" not in kw:
            on, off = self.thinking_mode
            kw["thinking_mode"] = on if _truthy(kw["enable_thinking"]) else off
        if self.thinking_flag and sent and "thinking" not in kw:
            kw["thinking"] = _truthy(kw["enable_thinking"])
        if self.preserve_off and "preserve_thinking" not in kw:
            kw["preserve_thinking"] = False
        if self.efforts and kw.get("reasoning_effort") is not None:
            fixed = nearest_effort(kw["reasoning_effort"], self.efforts)
            if fixed is None:
                kw.pop("reasoning_effort")
            else:
                kw["reasoning_effort"] = fixed
        return kw


def template_text(tokenizer) -> str | None:
    """The Jinja chat template, or None for a Python template or none at all."""
    if callable(getattr(tokenizer, "_chat_template", None)):
        return None
    try:
        t = tokenizer.chat_template
    except Exception:
        return None
    if isinstance(t, dict):
        t = t.get("default") or next((v for v in t.values() if isinstance(v, str)), None)
    return t if isinstance(t, str) and t.strip() else None


def detect_template_kwargs(text: str, render) -> TemplateKwargs:
    """Work out the rules for one template by rendering it.

    `render(messages, **kwargs)` returns the prompt text and raises when the
    template does. A template that cannot render the probe gets no rules.
    """
    rules = TemplateKwargs()

    def out(msgs, **kw):
        try:
            return render(msgs, **kw)
        except Exception:
            return None

    if out(_PROBE) is None:
        return rules

    reads_enable = re.search(r"\benable_thinking\b", text) is not None
    if not reads_enable and "thinking_mode" in text:
        modes = set(re.findall(r"thinking_mode\s*==\s*['\"](\w+)['\"]", text))
        on = next((m for m in _ON_MODES if m in modes), None)
        off = next((m for m in _OFF_MODES if m in modes), None)
        if on and off and out(_PROBE, thinking_mode=on) != out(_PROBE, thinking_mode=off):
            rules.thinking_mode = (on, off)
    if not reads_enable and _BARE_THINKING.search(text):
        if out(_PROBE, thinking=True) != out(_PROBE, thinking=False):
            rules.thinking_flag = True

    if "preserve_thinking" in text:
        unset = out(_HISTORY)
        if unset is not None and unset != out(_HISTORY, preserve_thinking=False):
            rules.preserve_off = True

    if "reasoning_effort" in text and out(_PROBE, enable_thinking=True, reasoning_effort="zz-not-an-effort") is None:
        ok = tuple(
            e for e in _EFFORTS if out(_PROBE, enable_thinking=True, reasoning_effort=e) is not None
        )
        if ok:
            rules.efforts = ok
    return rules


def install_template_kwargs(tokenizer) -> TemplateKwargs | None:
    """Wrap tokenizer.apply_chat_template with this template's rules.

    Every render goes through it (the server, the stable prefix probe, replay
    splice), so they all see the same kwargs. Returns the rules, or None when
    the template needs nothing or PAGED_MOE_TEMPLATE_KWARGS=0.
    """
    if not config.TEMPLATE_KWARGS:
        return None
    if getattr(tokenizer, "_pm_template_kwargs", None) is not None:
        return tokenizer._pm_template_kwargs
    text = template_text(tokenizer)
    if not text:
        return None
    orig = tokenizer.apply_chat_template

    def render(msgs, **kw):
        return orig(msgs, tokenize=False, add_generation_prompt=True, **kw)

    rules = detect_template_kwargs(text, render)
    if not rules.active():
        return None

    def apply_chat_template(*args, **kwargs):
        return orig(*args, **rules.apply(kwargs))

    # TokenizerWrapper forwards public setattr to the HF tokenizer underneath.
    object.__setattr__(tokenizer, "apply_chat_template", apply_chat_template)
    object.__setattr__(tokenizer, "_pm_template_kwargs", rules)
    print(f"[paged-moe] chat template: {rules.describe()}", flush=True)
    return rules


def lift_request_kwargs(body) -> dict | None:
    """chat_template_kwargs with a top-level reasoning effort folded in.

    OpenAI clients send `reasoning_effort` (or `reasoning: {effort}`) at the
    top of the request, where mlx-lm ignores it. Returns None when there is
    nothing to fold in or the client already set it in chat_template_kwargs.
    """
    if not config.TEMPLATE_KWARGS or not isinstance(body, dict):
        return None
    effort = body.get("reasoning_effort")
    if effort is None and isinstance(body.get("reasoning"), dict):
        effort = body["reasoning"].get("effort")
    if not isinstance(effort, str) or not effort.strip():
        return None
    kw = body.get("chat_template_kwargs")
    kw = dict(kw) if isinstance(kw, dict) else {}
    if "reasoning_effort" in kw:
        return None
    kw["reasoning_effort"] = effort.strip().lower()
    return kw
