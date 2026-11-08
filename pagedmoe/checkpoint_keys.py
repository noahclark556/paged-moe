# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Map module-tree paths to on-disk safetensors keys.

Multimodal checkpoints (qwen4_exp / Flash) keep text weights under
`language_model.model....` while sanitize() strips that prefix into the module
tree (`model....`). Expert streaming must resolve both spellings or it treats
the entire MoE mass as resident backbone.
"""

from __future__ import annotations

from .safetensors_index import TensorLoc


def tensor_key_candidates(path: str) -> list[str]:
    """Possible checkpoint keys for one module / tensor path."""
    keys = [path]
    if not path.startswith("language_model."):
        keys.append(f"language_model.{path}")
    return keys


def lookup_tensor(
    tensor_index: dict[str, TensorLoc], path: str
) -> TensorLoc | None:
    for key in tensor_key_candidates(path):
        loc = tensor_index.get(key)
        if loc is not None:
            return loc
    return None


def names_with_prefix(
    tensor_index: dict[str, TensorLoc], prefixes: tuple[str, ...]
) -> set[str]:
    """All tensor names whose key (or language_model-stripped form) starts
    with any of the given prefixes."""
    out: set[str] = set()
    for name in tensor_index:
        bare = (
            name[len("language_model.") :]
            if name.startswith("language_model.")
            else name
        )
        if name.startswith(prefixes) or bare.startswith(prefixes):
            out.add(name)
    return out
