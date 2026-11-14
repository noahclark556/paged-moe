# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Per-model settings for `paged-moe-server` and `run.py`.

The mlx_lm.load hook already applies `~/paged-moe-config.yaml`. The server and
run.py call the loader directly, so without this they ran on bare engine
defaults. This puts the same settings in the environment before any engine
module is imported (most knobs are read once at import).

Order, lowest to highest: `defaults:` in the YAML, then the matched model's
`env:` (or a paged_moe.yaml marker), then anything already in the environment.

`PAGED_MOE_HOOK=0` skips it, so host apps that set their own env per model get
exactly what they set.
"""

from __future__ import annotations

import os
import sys


def _flag_value(argv: list[str], flag: str) -> str | None:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _pop_flag(argv: list[str], flag: str) -> str | None:
    """Remove `flag VALUE` / `flag=VALUE` from argv and return the value."""
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            value = argv[i + 1]
            del argv[i : i + 2]
            return value
        if arg.startswith(flag + "="):
            del argv[i]
            return arg.split("=", 1)[1]
    return None


def apply(argv: list[str] | None = None, *, quiet: bool = False) -> dict[str, str]:
    """Fill os.environ for the model named by --model. Returns what was set.

    Strips `--config PATH` from argv (sys.argv by default) so the wrapped CLI
    never sees it. Must run before pagedmoe.config is imported.
    """
    argv = sys.argv if argv is None else argv
    cfg_path = _pop_flag(argv, "--config")
    if cfg_path:
        os.environ["PAGED_MOE_CONFIG"] = os.path.expanduser(cfg_path)

    from .mlx_hook import _hook_disabled, _merged_env

    model = _flag_value(argv, "--model")
    if not model or _hook_disabled():
        return {}
    try:
        layered = _merged_env(model)
    except Exception as e:
        print(f"[paged-moe] could not read the config: {e!r}", file=sys.stderr, flush=True)
        return {}

    applied = {k: v for k, v in layered.items() if k not in os.environ}
    os.environ.update(applied)
    if applied and not quiet:
        keys = " ".join(f"{k.removeprefix('PAGED_MOE_')}={v}" for k, v in sorted(applied.items()))
        print(f"[paged-moe] config settings: {keys}", file=sys.stderr, flush=True)
    return applied


def server_main() -> None:
    """Entry point for `paged-moe-server`."""
    apply()
    from .server import main

    main()
