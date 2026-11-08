# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Copy-heavy decode: plain steps vs verify passes, exact vs plain fidelity.

The model is shown a source file and asked to copy one function back, the
shape of an edit call's oldString / newString. One load, arms interleaved on
fresh caches built from the same prefill:

  plain   verify governor never engages (VERIFY off for the loop)
  exact   VERIFY_FIDELITY=exact
  match   VERIFY_FIDELITY=plain (default)
  wide    match with VERIFY_TRUNCATE=0 (match-length widths only)

  PAGED_MOE_PRUNE=0.9 PAGED_MOE_WAIT_ABOVE=0.45 \
  python bench/copy_decode_ab.py --model ~/mlx-models/qwen38-coder-flash
"""

import argparse
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from pagedmoe import config as es_config  # noqa: E402
from pagedmoe import load, streaming  # noqa: E402
from pagedmoe import verify_decode as vd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=192)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--arms", default="plain,exact,match",
                    help="comma list; exact is slow on big disk-bound models")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]

    model, tok = load(args.model, mode="streamed", verbose=False)
    src = (Path(__file__).resolve().parents[1] / "pagedmoe" / "slab.py").read_text()
    msg = (
        "Here is a file:\n```python\n" + src + "\n```\n"
        "Copy the method `expert` of SlabStore back verbatim, then the method after it, "
        "inside one python code block. Nothing else."
    )
    text = tok.apply_chat_template(
        [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True,
        enable_thinking=False, thinking_mode="disabled",
    )
    prompt = tok.encode(text, add_special_tokens=False)
    base = make_prompt_cache(model)
    with streaming.exact_step(), streaming.cache_only():
        model(mx.array([prompt[:-1]]), cache=base)
        mx.eval([c.state for c in base])
    print(f"prompt {len(prompt)} tok", flush=True)
    es_config.VERIFY_SPANS = "all"

    def run(arm):
        cache = copy.deepcopy(base)
        mx.eval([c.state for c in cache])
        es_config.VERIFY = arm != "plain"
        es_config.VERIFY_FIDELITY = "exact" if arm == "exact" else "plain"
        es_config.VERIFY_TRUNCATE = arm != "wide"
        vd._GOV = vd._Governor()
        vd._ROWS = vd._RowCost()
        if arm == "plain":
            vd._Governor.ready = lambda self, span: False
        gen = vd.generate_step(
            mx.array(prompt[-1:], dtype=mx.uint32), model, max_tokens=args.tokens,
            prompt_cache=cache, history=prompt[:-1],
        )
        t = time.perf_counter()
        out = [int(t_) for t_, _ in gen]
        dt = time.perf_counter() - t
        if arm == "plain":
            vd._Governor.ready = ready
        st = dict(vd.STATS)
        del cache
        mx.clear_cache()
        return len(out) / dt, st, out

    ready = vd._Governor.ready
    res = {a: [] for a in arms}
    texts = {}
    for r in range(args.rounds):
        for arm in (arms if r % 2 == 0 else arms[::-1]):
            tps, st, out = run(arm)
            res[arm].append(tps)
            texts[arm] = out
            print(f"  {arm:5} {tps:6.2f} tok/s  passes={st.get('passes', 0)} plain={st.get('plain', 0)} "
                  f"accepted={st.get('accepted', 0)}/{st.get('drafted', 0)}", flush=True)
    for arm, xs in res.items():
        print(f"{arm:5} mean {sum(xs) / len(xs):6.2f} tok/s", flush=True)
    print("sample (match):", tok.decode(texts["match"][:80]).replace("\n", "\\n")[:300])


if __name__ == "__main__":
    main()
