# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How far a run's decode settings are from the exact model, on fixed text.

Teacher-forced, one token at a time through the decode path. Prefill never
prunes, so a one-shot pass would report zero loss whatever the settings.
Every arm sees the same text in lockstep on one model load:

  exact   streaming.exact_step(), the router's full top-k (the reference)
  run     the env's PRUNE / WAIT_ABOVE, what a plain decode token gets
  tool    the same scaled by the tool-call fidelity scale, what a tool call
          or turn-start token gets

Per arm against exact: top-1 agreement, NLL of the real next token, NLL
delta and mean KL(exact || arm). Run it with the server's env.

  python -m pagedmoe.benchmark.probe --model <dir> --out f.json [--tokens 160]
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default=str(DATA / "probe.txt"))
    ap.add_argument("--tokens", type=int, default=160)
    ap.add_argument("--prefill", type=int, default=512)
    ap.add_argument(
        "--context",
        default=None,
        help="text prefilled first (the bench's system prompt) so context length and "
        "expert residency look like an agent turn. Pruning only drops experts that "
        "are not resident, so a fresh cache on a short text shows no loss at all.",
    )
    args = ap.parse_args()

    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import make_prompt_cache

    from pagedmoe import config, load, streaming

    t0 = time.perf_counter()
    model, tok = load(args.model, verbose=False)
    load_s = time.perf_counter() - t0
    streamed = getattr(model, "_pagedmoe_info", {}).get("mode") == "streamed"
    ids = tok.encode(Path(args.text).read_text())
    prefill = min(args.prefill, max(64, len(ids) - args.tokens - 1))
    need = prefill + args.tokens + 1
    if len(ids) < need:
        raise SystemExit(f"probe text is {len(ids)} tokens, needs {need}")
    ids = ids[:need]

    prune, wait = float(config.PRUNE), float(config.WAIT_ABOVE)
    scale = float(config.FIDELITY_TOOL_SCALE if config.FIDELITY_TOOL_SCALE is not None else config.FIDELITY_SCALE)
    arms = {"exact": None, "run": (prune, wait)}
    if prune > 0 or wait > 0:
        arms["tool"] = (prune * scale, wait * scale)

    ctx = tok.encode(Path(args.context).read_text()) if args.context else []
    base = make_prompt_cache(model)
    with streaming.exact_step(), streaming.cache_only():
        seq = ctx + ids[: prefill - 1]
        for i in range(0, len(seq), 8192):
            model(mx.array([seq[i : i + 8192]]), cache=base)
            mx.eval([c.state for c in base])
    caches = {}
    for name in arms:
        c = copy.deepcopy(base)
        mx.eval([e.state for e in c])
        caches[name] = c

    ec = getattr(model, "_pagedmoe_cache", None)
    stats = {n: {"nll": [], "agree": 0, "kl": [], "dropped": 0, "routed": 0} for n in arms}

    def step(name, token):
        cfg = arms[name]
        y = mx.array([[token]])
        before = (ec.pruned_slots, ec.demand_slots) if ec is not None else (0, 0)
        if cfg is None:
            with streaming.exact_step():
                logits = model(y, cache=caches[name])
        else:
            config.PRUNE, config.WAIT_ABOVE = cfg
            logits = model(y, cache=caches[name])
        row = logits[0, -1].astype(mx.float32)
        out = row - mx.logsumexp(row)
        mx.eval(out)
        if ec is not None:
            stats[name]["dropped"] += ec.pruned_slots - before[0]
            stats[name]["routed"] += ec.demand_slots - before[1]
        return out

    t1 = time.perf_counter()
    for i in range(args.tokens):
        feed = ids[prefill - 1 + i]
        target = ids[prefill + i]
        rows = {n: step(n, feed) for n in arms}
        ref_np = np.asarray(rows["exact"])
        ref_top = int(ref_np.argmax())
        p_ref = np.exp(ref_np)
        for n, row in rows.items():
            r = np.asarray(row)
            stats[n]["nll"].append(-float(r[target]))
            stats[n]["agree"] += int(int(r.argmax()) == ref_top)
            stats[n]["kl"].append(float(np.sum(p_ref * (ref_np - r))))
    config.PRUNE, config.WAIT_ABOVE = prune, wait

    ref_nll = float(np.mean(stats["exact"]["nll"]))
    out = {
        "streamed": streamed,
        "tokens": args.tokens,
        "prefill": prefill,
        "context_tokens": len(ctx),
        "load_s": round(load_s, 1),
        "scored_s": round(time.perf_counter() - t1, 1),
        "settings": {"prune": prune, "wait_above": wait, "tool_scale": scale},
        "exact_nll": round(ref_nll, 4),
        "exact_ppl": round(float(np.exp(ref_nll)), 3),
        "arms": {},
    }
    for n, s in stats.items():
        if n == "exact":
            continue
        nll = float(np.mean(s["nll"]))
        out["arms"][n] = {
            "settings": {"prune": arms[n][0], "wait_above": arms[n][1]},
            "top1_agree": round(s["agree"] / args.tokens, 4),
            "nll": round(nll, 4),
            "nll_delta": round(nll - ref_nll, 4),
            "ppl_ratio": round(float(np.exp(nll - ref_nll)), 4),
            "kl_mean": round(float(np.mean(s["kl"])), 5),
            # Routed slots the settings dropped after keeping every resident
            # expert. 0 means this cache state computed the exact mixture.
            "dropped_frac": round(s["dropped"] / s["routed"], 4) if s["routed"] else 0.0,
        }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
