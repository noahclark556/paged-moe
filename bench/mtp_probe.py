# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Teacher-forced MTP head accuracy on a real checkpoint.

One main-model forward over prompt + a greedy reply records the hidden state
at every position. The head then predicts token t+2 from (hidden t, token t+1)
over the reply, once as a batch with some prompt context and once one
position at a time through Drafter.draft, the way decode drives it.

    python bench/mtp_probe.py --model DIR --request req.json --reply resp.json

req.json is the chat request, resp.json the server's greedy reply to it.
A head that converted wrong scores near 0; Flash scores ~0.8.
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from pagedmoe import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--request", required=True, help="chat completion request JSON")
    ap.add_argument("--reply", required=True, help="greedy chat completion response JSON")
    args = ap.parse_args()

    model, tok = load(args.model)
    drafter = getattr(model, "_pagedmoe_mtp", None)
    if drafter is None:
        print("no MTP head attached")
        return 1
    req = json.load(open(args.request))
    msg = json.load(open(args.reply))["choices"][0]["message"]
    reply = msg.get("content") or msg.get("reasoning") or ""
    prompt = tok.apply_chat_template(req["messages"], add_generation_prompt=True, tokenize=True)
    n_p = len(prompt)
    ids = mx.array(list(prompt) + tok.encode(reply, add_special_tokens=False), dtype=mx.uint32)
    T = int(ids.shape[0])
    print(f"prompt {n_p} tok, reply {T - n_p} tok")

    with drafter.capture():
        logits = model(ids[None], cache=make_prompt_cache(model))
    h = drafter.tap.value
    mx.eval(logits, h)
    lo = n_p - 1
    agree = mx.mean((mx.argmax(logits[0], -1)[lo:T - 1] == ids[n_p:T]).astype(mx.float32)).item()
    print(f"main model agrees with the reply on {agree:.3f} (below ~0.9: reply was not greedy)")

    for ctx in (0, 64):
        drafter.reset()
        a = max(0, lo - ctx)
        out, _ = drafter._fwd(h[:, a:T - 2], ids[None, a + 1:T - 1])
        pred = mx.argmax(out[0], axis=-1)[lo - a:]
        hit = mx.mean((pred == ids[lo + 2:T]).astype(mx.float32)).item()
        print(f"batch, {ctx:2d} prompt positions of context: t+2 hit {hit:.3f}")

    drafter.reset()
    hits = n = 0
    for t in range(lo, T - 3):
        drafter.tap.value = h[:, t:t + 1]
        drafter.fed(ids[t:t + 1])
        d = drafter.draft(int(ids[t + 1].item()), 1)
        drafter.pre = None
        if d:
            n += 1
            hits += int(d[0] == int(ids[t + 2].item()))
    print(f"Drafter.draft in decode order: t+2 hit {hits}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
