# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shims for custom tokenizer code written against older transformers.

Checkpoints that ship their own tokenizer (trust_remote_code) import names
from wherever transformers kept them at the time. transformers 5 moved some
of those, and the checkpoint code then fails at import before anything loads.
"""


def install_transformers_compat() -> None:
    """Put moved names back where old tokenizer code looks for them. Idempotent."""
    try:
        from transformers.models.gpt2 import tokenization_gpt2
    except Exception:
        return
    # Kimi K2 tokenization_kimi.py: from ...gpt2.tokenization_gpt2 import bytes_to_unicode
    if not hasattr(tokenization_gpt2, "bytes_to_unicode"):
        try:
            from transformers.convert_slow_tokenizer import bytes_to_unicode
        except Exception:
            return
        tokenization_gpt2.bytes_to_unicode = bytes_to_unicode
