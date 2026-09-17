# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat-template shims so mlx-lm host kwargs match model-specific templates.

mlx-lm's TokenizerWrapper always injects ``enable_thinking``. DeepSeek-V3.2's
Python template wants ``thinking_mode`` ("thinking"|"chat") instead, so without
this map every apply_chat_template dies with:

    encode_messages() got an unexpected keyword argument 'enable_thinking'

Installed once at import from loader (and re-exported by the server) so CLI,
server, and anything else that calls ``load()`` share the same path.
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
        # Host apps pass these for Jinja templates; this Python template does not.
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
        # there (task-state acks). Give those a stub so the request does not 404.
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
