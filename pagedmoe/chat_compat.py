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
