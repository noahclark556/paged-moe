#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-arm decode probe for sidecar A/B (see bench/sidecar_ab.sh).

Wall-clock decode tok/s after a short warmup. Config comes from the
PAGED_MOE_* environment so each arm runs in a clean process.

  PAGED_MOE_SIDECAR=0 python bench/sidecar_probe.py --model ~/mlx-models/qwen3-235-4bit \\
      --cache-gb 28 --prune 0.5 --max-tokens 256 --label no-sidecar

  PAGED_MOE_SIDECAR=1 python bench/sidecar_probe.py --model ~/mlx-models/qwen3-235-4bit \\
      --cache-gb 28 --prune 0.5 --max-tokens 256 --label trained-sidecar
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPT = (
    "You are helping with a multi-file Python refactor. Outline the steps to "
    "split a large module into packages, keep imports working, and verify with "
    "tests. Be concrete; list files you would touch."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="checkpoint path")
    ap.add_argument("--cache-gb", type=float, default=None)
    ap.add_argument("--prune", type=float, default=None)
    ap.add_argument("--wait-above", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup-tokens", type=int, default=16)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--label", default="")
    ap.add_argument(
        "--sidecar-dir",
        default="",
        help="override PAGED_MOE_SIDECAR_DIR (warm weights live here)",
    )
    args = ap.parse_args()

    if args.cache_gb is not None:
        os.environ["PAGED_MOE_CACHE_GB"] = str(args.cache_gb)
    if args.prune is not None:
        os.environ["PAGED_MOE_PRUNE"] = str(args.prune)
    if args.wait_above is not None:
        os.environ["PAGED_MOE_WAIT_ABOVE"] = str(args.wait_above)
    if args.sidecar_dir.strip():
        os.environ["PAGED_MOE_SIDECAR_DIR"] = str(
            Path(args.sidecar_dir).expanduser()
        )

    # Re-bind knobs after env edits (module may already be imported via path).
    from pagedmoe import config as es_config

    es_config.sync_from_environ()

    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    from pagedmoe import get_stats, load, relieve_pressure

    model, tokenizer = load(
        str(Path(args.model).expanduser()),
        mode="streamed",
        cache_gb=args.cache_gb,
        verbose=False,
    )
    info = dict(getattr(model, "_pagedmoe_info", {}))

    # Build a mid-length prompt (not the memory ladder's 2k filler).
    filler = (ROOT / "pagedmoe" / "cache.py").read_text()
    body = (filler * 8)[: max(200, args.prompt_tokens * 4)]
    messages = [
        {"role": "system", "content": "You are a terse coding assistant."},
        {"role": "user", "content": PROMPT + "\n\nContext excerpt:\n" + body},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if len(prompt) > args.prompt_tokens + 128:
        prompt = prompt[: args.prompt_tokens]

    sampler = make_sampler(temp=0.0)

    # Warm expert cache; do not bill cold misses to the timed window.
    for _ in stream_generate(
        model,
        tokenizer,
        prompt=prompt[: min(128, len(prompt))],
        max_tokens=args.warmup_tokens,
        sampler=sampler,
    ):
        pass
    relieve_pressure(model)
    mx.reset_peak_memory()

    kv = make_prompt_cache(model)
    t0 = time.perf_counter()
    first = None
    n = 0
    for _ in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        prompt_cache=kv,
    ):
        now = time.perf_counter()
        if first is None:
            first = now
        n += 1
    t1 = time.perf_counter()
    decode_s = max(t1 - (first or t0), 1e-9)
    prefill_s = (first - t0) if first else (t1 - t0)

    stats = get_stats(model)
    cache = stats.get("cache", {}) or {}
    ring = stats.get("ring", {}) or {}
    sidecar = None
    try:
        sc = getattr(getattr(model, "_pagedmoe_ring", None), "sidecar", None)
        if sc is not None:
            sidecar = {
                "slot": getattr(sc, "slot_id", None),
                "mode": getattr(sc, "_mode", None),
                "decode_tok_s_ema": getattr(sc, "decode_tok_s", None),
                "wrap_precision": (
                    float(sc.wrap.slot.mean_precision())
                    if getattr(sc, "wrap", None) is not None
                    else None
                ),
            }
    except Exception:
        sidecar = None

    out = {
        "label": args.label or ("sidecar" if es_config.SIDECAR else "no-sidecar"),
        "model": str(Path(args.model).expanduser()),
        "sidecar": bool(es_config.SIDECAR),
        "sidecar_dir": es_config.SIDECAR_DIR,
        "prune": float(es_config.PRUNE),
        "cache_gb": args.cache_gb,
        "prompt_tokens": len(prompt),
        "warmup_tokens": args.warmup_tokens,
        "decode_tokens": n,
        "prefill_s": round(prefill_s, 3),
        "decode_s": round(decode_s, 3),
        "decode_tok_s": round(n / decode_s, 2) if n else 0.0,
        "prefill_tok_s": round(len(prompt) / max(prefill_s, 1e-9), 2),
        "hit_rate": cache.get("hit_rate"),
        "disk_wait_s": cache.get("disk_wait_s"),
        "pred_coverage": cache.get("pred_coverage"),
        "info": {
            "moe_layers": info.get("moe_layers"),
            "expert_gb": info.get("expert_gb"),
            "slab_gb": info.get("slab_gb"),
            "predict": info.get("predict"),
        },
        "sidecar_live": sidecar,
        "ring": {k: ring.get(k) for k in ("predict_on", "predict_mode") if k in ring},
    }
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
