# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Site .pth entrypoint - keep separate from ``mlx_hook`` CLI to avoid runpy warnings."""

from __future__ import annotations


def autoload() -> None:
    from .mlx_hook import install

    install()
