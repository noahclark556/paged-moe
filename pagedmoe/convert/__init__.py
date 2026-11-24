# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Convert Hugging Face checkpoints to MLX with per-component recipes.

    from pagedmoe.convert import make_plan, convert

    p = make_plan("zai-org/GLM-4.7", "balanced", out_dir="~/mlx-models/glm-47-mine")
    print(p.summary()["out_bytes"], p.summary()["fit"])
    convert("zai-org/GLM-4.7", "~/mlx-models/glm-47-mine", "balanced")

Everything heavy (mlx, huggingface_hub) imports on first use.
"""

from __future__ import annotations

from typing import Any

from .recipe import PRESETS, Recipe, RecipeError

_LAZY = {
    "make_plan": ("plan", "make"),
    "compare": ("plan", "compare"),
    "Plan": ("plan", "Plan"),
    "PlanError": ("plan", "PlanError"),
    "convert": ("run", "convert"),
    "ConvertError": ("run", "ConvertError"),
    "check_fidelity": ("check", "check"),
    "upload": ("hub", "upload"),
    "open_source": ("source", "open_source"),
    "SourceError": ("source", "SourceError"),
}

__all__ = ["PRESETS", "Recipe", "RecipeError", *_LAZY]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(f".{mod}", __name__), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
