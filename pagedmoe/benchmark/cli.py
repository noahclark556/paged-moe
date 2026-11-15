# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""`paged-moe bench <model> [--speed S | --exact | --both] [--quick]`."""

from __future__ import annotations

import argparse
import json
import sys

from .runner import DEFAULT_CTX, DEFAULT_SPEED, report, run

USAGE = """paged-moe bench <model-path> [--speed S | --exact | --both] [--quick]

Replays a fixed agent task (8 requests at ~13k tokens of context) on this Mac and
prints decode tok/s, time to first token, prefill, memory and, for speed runs,
the loss against the exact model. Default runs speed, then exact.

  paged-moe bench ~/mlx-models/qwen3-235-4bit --quick
  paged-moe bench ~/mlx-models/qwen3-235-4bit --exact
  paged-moe bench ~/mlx-models/qwen3-235-4bit --speed 0.5

Results go to ~/.paged_moe/bench/ (or --out DIR), one JSON per mode.
"""


def _speed(raw: str):
    s = raw.strip().lower()
    if s == "auto":
        return "auto"
    v = float(s)
    if not 0.0 <= v < 1.0:
        raise argparse.ArgumentTypeError("--speed takes 0 (fastest) up to but not including 1, or auto. Use --exact for 1")
    return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="paged-moe bench", usage=USAGE, add_help=True)
    ap.add_argument("model", help="local MLX checkpoint folder (or a downloaded HF repo id)")
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--exact", action="store_true", help="exact mode only: every routed expert, bit-identical")
    which.add_argument("--both", action="store_true", help="speed, then exact (the default)")
    which.add_argument("--speed-only", action="store_true", help="speed mode only")
    ap.add_argument("--speed", type=_speed, default=None,
                    help=f"speed setting, 0 fastest .. 1 exact, or auto (default {DEFAULT_SPEED}). Implies speed mode only unless --both")
    ap.add_argument("--quick", action="store_true", help="half the decode budgets and a short loss probe. A smoke test, not a number to publish")
    ap.add_argument("--out", default=None, help="results folder (default ~/.paged_moe/bench)")
    ap.add_argument("--port", type=int, default=18799)
    ap.add_argument("--ctx", type=int, default=DEFAULT_CTX, help=f"context to reserve KV memory for (default {DEFAULT_CTX})")
    ap.add_argument("--no-probe", action="store_true", help="skip the loss probe")
    ap.add_argument("--no-think", action="store_true", help="turn thinking off on models whose template supports it")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VAL", help="extra engine env (recorded in the result)")
    ap.add_argument("--stall-s", type=float, default=300.0, help="stop the run after this long with no engine progress")
    ap.add_argument("--allow-busy", action="store_true", help="skip the other-server and GPU memory checks")
    ap.add_argument("--keep-state", action="store_true", help="keep the run's prefix cache and sidecar files")
    ap.add_argument("--json", action="store_true", help="print the result documents as JSON at the end")
    args = ap.parse_args(argv)

    if args.exact:
        modes = ["exact"]
    elif args.speed_only or (args.speed is not None and not args.both):
        modes = ["speed"]
    else:
        modes = ["speed", "exact"]
    extra = {}
    for kv in args.env:
        if "=" not in kv:
            ap.error(f"--env takes KEY=VAL, got {kv!r}")
        k, v = kv.split("=", 1)
        extra[k] = v

    docs = []
    for mode in modes:
        try:
            doc = run(
                args.model,
                mode,
                speed=DEFAULT_SPEED if args.speed is None else args.speed,
                quick=args.quick,
                out_dir=args.out,
                port=args.port,
                probe=not args.no_probe,
                ctx=args.ctx,
                think=not args.no_think,
                extra_env=extra,
                allow_busy=args.allow_busy,
                stall_s=args.stall_s,
                keep_state=args.keep_state,
            )
        except KeyboardInterrupt:
            print("\n[bench] stopped", file=sys.stderr)
            return 130
        docs.append(doc)
        if doc["status"] in ("failed", "unsupported"):
            break

    print(report(docs))
    if args.json:
        print(json.dumps(docs, indent=2))
    bad = [d for d in docs if d["status"] != "ok"]
    return 1 if bad else 0
