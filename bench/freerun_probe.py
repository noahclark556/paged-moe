# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Free-running decode after a long code context, the way an agent turn runs.

quality_probe teacher-forces every step, so one bad token cannot derail the
next. Agent decode feeds the model its own output, and that is where flow on
Qwen3-235B turned into "00't read more files, need 2 edit". This replays that
shape: a long code prompt, a chat instruction, greedy decode, and the text.

  python bench/freerun_probe.py --model ~/mlx-models/qwen3-235-4bit --save /tmp/ref.json
  PAGED_MOE_PRUNE=0.7 PAGED_MOE_FLOW=1 \
    python bench/freerun_probe.py --model ~/mlx-models/qwen3-235-4bit --ref /tmp/ref.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402
from mlx_lm import stream_generate  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from pagedmoe import config, get_stats, load  # noqa: E402

ASK = (
    "Above is part of an inference engine. In a few sentences, explain what "
    "ExpertCache.flow_admit does and why it installs reads directly instead of "
    "staging them. Then give one concrete edit you would make to it, as a "
    "replace_string_in_file call with oldString and newString."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context-tokens", type=int, default=14000)
    ap.add_argument("--gen-tokens", type=int, default=200)
    ap.add_argument("--save")
    ap.add_argument("--ref")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    model, tok = load(args.model, mode="streamed", verbose=False)
    root = Path(__file__).resolve().parents[1] / "pagedmoe"
    code = "\n\n".join(
        f"# file: {p.name}\n{p.read_text(errors='replace')}"
        for p in (root / "cache.py", root / "streaming.py", root / "slab.py")
    )
    ids = tok.encode(code)[: args.context_tokens]
    code = tok.decode(ids)
    msgs = [{"role": "user", "content": f"```python\n{code}\n```\n\n{ASK}"}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    cache = make_prompt_cache(model)
    out_ids: list[int] = []
    text = ""
    t_first = None
    t0 = time.perf_counter()
    for r in stream_generate(
        model, tok, prompt, max_tokens=args.gen_tokens,
        sampler=make_sampler(temp=0.0), prompt_cache=cache,
    ):
        if t_first is None:
            t_first = time.perf_counter()
        out_ids.append(int(r.token))
        text += r.text
    t1 = time.perf_counter()

    res = {
        "label": args.label or f"prune={config.PRUNE} flow={int(config.FLOW)}",
        "prompt_tokens": len(tok.encode(prompt)),
        "prefill_s": round((t_first or t1) - t0, 1),
        "decode_tok_s": round((len(out_ids) - 1) / max(t1 - (t_first or t1), 1e-9), 2),
        "ids": out_ids,
        "text": text,
    }
    cstats = (get_stats(model) or {}).get("cache", {})
    for k in ("pruned_frac", "crit_per_layer", "mass_sync", "mass_flow"):
        if k in cstats:
            res[k] = cstats[k]
    if args.ref:
        ref = json.loads(Path(args.ref).read_text())
        same = 0
        for a, b in zip(ref["ids"], out_ids):
            if a != b:
                break
            same += 1
        res["same_prefix"] = same
    if args.save:
        Path(args.save).write_text(json.dumps(res))
    slim = {k: v for k, v in res.items() if k not in ("ids", "text")}
    print(json.dumps(slim, indent=2))
    print("----- text -----")
    print(text)
    mx.clear_cache()


if __name__ == "__main__":
    main()
