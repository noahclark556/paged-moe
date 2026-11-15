# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bench any MLX checkpoint on this Mac.

Replays a fixed agent task (8 requests: 7 turns building a small express app,
then a new chat) against `pagedmoe.server`, the way a coding agent drives it,
and reads the engine's own per-request trace. Speed runs also get a loss probe
against the exact model, so every speed number has its fidelity next to it.

  paged-moe bench ~/mlx-models/<model>                # speed, then exact
  paged-moe bench ~/mlx-models/<model> --exact --quick

From Python:

  from pagedmoe.benchmark import run, report
  doc = run("~/mlx-models/<model>", mode="speed", quick=True)
  print(report([doc]))

Nothing in here imports mlx; the server and the probe run as child processes
so Metal is torn down between loads.
"""

from .runner import DEFAULT_SPEED, report, run

__all__ = ["run", "report", "DEFAULT_SPEED", "main"]


def main(argv: list[str] | None = None) -> int:
    from .cli import main as _main

    return _main(argv)
