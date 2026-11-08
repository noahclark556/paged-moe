# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does decode end a turn mid-thought? Replay one logged agent request.

Takes the exact chat request a client posted, saved as JSON. This rebuilds that
prompt (tools and chat_template_kwargs included), prefills it once, then samples
several continuations with the request's sampler and counts how many hit EOS
before </think>. A turn that ends inside its own reasoning comes back to the
client as an empty reply with no tool call.

  python bench/early_stop_probe.py --model ~/mlx-models/qwen38-coder-flash \
      --request request.json
"""

import argparse
import contextlib
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402
from mlx_lm import stream_generate  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.sample_utils import make_logits_processors, make_sampler  # noqa: E402

from pagedmoe import config, get_stats, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--request", required=True, help="saved chat request JSON")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument(
        "--exact", action="store_true",
        help="sample under streaming.exact_step (no PRUNE, WAIT_ABOVE, flow or prune head)",
    )
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--presence", type=float, default=1.0)
    ap.add_argument("--label", default=None)
    ap.add_argument(
        "--score", default=None,
        help="JSON list of replies to teacher-force, exact and pruned, instead of sampling",
    )
    ap.add_argument(
        "--variants", default=None,
        help="name:KEY=VAL,KEY=VAL;name2:... config overrides sampled off one prefill",
    )
    args = ap.parse_args()

    req = json.loads(Path(args.request).read_text())
    # The server hands the template tool-call arguments as objects, not strings.
    for m in req["messages"]:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except ValueError:
                    pass
    model, tok = load(args.model, mode="streamed", verbose=False)
    # Decode through the server's generate_step, as production does.
    from pagedmoe import server

    server._install_adaptive_prefill_generate()
    prompt = tok.apply_chat_template(
        req["messages"],
        tools=req.get("tools") or None,
        add_generation_prompt=True,
        tokenize=False,
        **(req.get("chat_template_kwargs") or {}),
    )
    ids = tok.encode(prompt)
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])

    # Prefill everything but the last token once; each sample feeds that token
    # into its own copy of the cache.
    base = make_prompt_cache(model)
    t0 = time.perf_counter()
    for _ in stream_generate(model, tok, ids[:-1], max_tokens=1, prompt_cache=base):
        pass
    prefill_s = time.perf_counter() - t0

    if args.score:
        for reply in json.loads(Path(args.score).read_text()):
            score(model, tok, base, ids, eos, reply, args.temp)
        return

    variants = [(args.label or f"prune={config.PRUNE} flow={int(config.FLOW)}", {})]
    if args.variants:
        variants = []
        for part in args.variants.split(";"):
            name, _, kvs = part.partition(":")
            over = {}
            for kv in filter(None, kvs.split(",")):
                k, _, v = kv.partition("=")
                over[k] = type(getattr(config, k))(float(v))
            variants.append((name, over))
    for name, over in variants:
        for k, v in over.items():
            setattr(config, k, v)
        print(f"== variant {name} {over}", flush=True)
        if args.exact:
            from pagedmoe.streaming import exact_step

            with exact_step():
                sample(model, tok, base, ids, eos, args, name, prefill_s)
        else:
            sample(model, tok, base, ids, eos, args, name, prefill_s)


def score(model, tok, base, ids, eos, reply, temp):
    """Where does a reply that ended without a tool call fork?

    Feeds the reply one token at a time, once with the full mixture and once
    pruned, and prints at each position the sampler's odds (at `temp`, before
    top-p / top-k) of an end token and of a tool call opening.
    """
    from pagedmoe.streaming import exact_step

    call = tok.convert_tokens_to_ids("<tool_call>")
    toks = [ids[-1]] + tok.encode(reply, add_special_tokens=False)
    rows = {}
    # Pruned first: the exact pass installs every expert these tokens route to,
    # and flow computes whatever is resident, so pruned after exact is exact.
    for label in ("pruned", "exact"):
        cache = copy.deepcopy(base)
        out = []
        ctx = exact_step() if label == "exact" else contextlib.nullcontext()
        with ctx:
            for i, t in enumerate(toks):
                logits = model(mx.array([[t]]), cache=cache)[0, -1].astype(mx.float32)
                p = mx.softmax(logits / temp)
                mx.eval(p)
                nxt = toks[i + 1] if i + 1 < len(toks) else None
                out.append((
                    float(sum(p[e] for e in eos).item()), float(p[call].item()),
                    tok.decode([int(mx.argmax(p).item())]),
                    float(p[nxt].item()) if nxt is not None else float("nan"),
                ))
        rows[label] = out
        del cache
        mx.clear_cache()
    print(f"== score {reply[:70]!r}")
    print("  after               | exact p_end p_call p_next top | pruned p_end p_call p_next top")
    for i, t in enumerate(toks):
        e, q = rows["exact"][i], rows["pruned"][i]
        print(
            f"  {tok.decode([t])!r:20.20}| {e[0]:.3f} {e[1]:.3f} {e[3]:.3f} {e[2]!r:10.10}"
            f"| {q[0]:.3f} {q[1]:.3f} {q[3]:.3f} {q[2]!r:10.10}",
            flush=True,
        )


def sample(model, tok, base, ids, eos, args, label, prefill_s):
    runs = []
    for s in range(args.seed0, args.seed0 + args.samples):
        mx.random.seed(s)
        cache = copy.deepcopy(base)
        text, n, last = "", 0, None
        t1 = time.perf_counter()
        for r in stream_generate(
            model, tok, ids[-1:], max_tokens=args.max_tokens, prompt_cache=cache,
            sampler=make_sampler(temp=args.temp, top_p=args.top_p, top_k=args.top_k),
            logits_processors=make_logits_processors(presence_penalty=args.presence),
        ):
            text += r.text
            n += 1
            last = int(r.token)
        dt = time.perf_counter() - t1
        in_think = "</think>" not in text
        stop = "eos" if last in eos else "length"
        runs.append({
            "seed": s, "tokens": n, "stop": stop, "in_think": in_think,
            "tool": "<tool_call>" in text,
            "tok_s": round(n / max(dt, 1e-9), 1), "tail": text[-90:],
        })
        print(json.dumps(runs[-1]), flush=True)
        del cache
        mx.clear_cache()

    early = sum(1 for r in runs if r["stop"] == "eos" and r["in_think"])
    # An agent turn that stops without a tool call comes back to the client as a
    # dead end, whether or not the reasoning was closed first.
    no_tool = sum(1 for r in runs if r["stop"] == "eos" and not r["tool"])
    res = {
        "label": label,
        "prompt_tokens": len(ids),
        "prefill_s": round(prefill_s, 1),
        "early_eos_in_think": f"{early}/{len(runs)}",
        "eos_without_tool": f"{no_tool}/{len(runs)}",
        "mean_tokens": round(sum(r["tokens"] for r in runs) / len(runs), 1),
    }
    cstats = (get_stats(model) or {}).get("cache", {})
    for k in ("pruned_frac", "crit_per_layer", "mass_sync", "mass_flow"):
        if k in cstats:
            res[k] = cstats[k]
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
