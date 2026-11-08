# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What are decode's blocking misses?

For every decode fetch, each requested expert is one of:
  hit        resident or staged
  inflight   a prefetch read is still landing (claimed, partial wait)
  pred_cold  route prediction named it for this token, but no read was issued
  last_cold  last-token replay had it (last_ids), no read issued
  cold       nobody predicted it

Usage:
  PAGED_MOE_PRUNE=0 PAGED_MOE_WAIT_ABOVE=0 PAGED_MOE_PREDICT=1 \
  python bench/miss_anatomy.py --model ~/mlx-models/qwen3-235-4bit \
      --prompt-file /tmp/esb/prompt.txt
"""

from __future__ import annotations

import argparse
import collections

from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from pagedmoe import config, load
from pagedmoe.cache import ExpertCache
from pagedmoe.streaming import StreamedSwitchGLU

CTX = {"pred": set(), "last": set(), "on": False}
C = collections.Counter()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=48)
    args = ap.parse_args()

    orig_call = StreamedSwitchGLU.__call__

    def call(self, x, indices):
        CTX["pred"] = set(self.predicted)
        CTX["last"] = set(self.last_ids)
        return orig_call(self, x, indices)

    StreamedSwitchGLU.__call__ = call
    orig_begin = ExpertCache._begin

    def begin(self, layer_key, expert_ids, use_slab=False, prefill=False):
        if CTX["on"] and use_slab:
            with self._lock:
                for e in expert_ids:
                    k = (layer_key, int(e))
                    if k in self._lru or k in self._raw_ready:
                        C["hit"] += 1
                    elif k in self._inflight:
                        C["inflight"] += 1
                    elif int(e) in CTX["pred"]:
                        C["pred_cold"] += 1
                    elif int(e) in CTX["last"]:
                        C["last_cold"] += 1
                    else:
                        C["cold"] += 1
            C["layers"] += 1
        return orig_begin(self, layer_key, expert_ids, use_slab=use_slab, prefill=prefill)

    ExpertCache._begin = begin

    # Tag every speculative slot read with who asked for it, and count its fate.
    SRC = {"cur": "other"}
    FATE = collections.Counter()
    orig_pm, orig_pf = ExpertCache.prefetch_many, ExpertCache.prefetch

    def pm(self, batch):
        SRC["cur"] = "predict"
        try:
            return orig_pm(self, batch)
        finally:
            SRC["cur"] = "other"

    def pf(self, lk, ids):
        prev = SRC["cur"]
        if prev == "other":
            SRC["cur"] = "replay"
        try:
            return orig_pf(self, lk, ids)
        finally:
            SRC["cur"] = prev

    ExpertCache.prefetch_many, ExpertCache.prefetch = pm, pf
    orig_sub = ExpertCache._submit_slab

    def sub(self, layer_key, expert_id, evict_ok=True, urgent=False):
        fut = orig_sub(self, layer_key, expert_id, evict_ok=evict_ok, urgent=urgent)
        if fut is not None and CTX["on"]:
            src = "demand" if urgent else SRC["cur"]
            fut._src = src
            FATE[(src, "issued")] += 1
            self._srcmap = getattr(self, "_srcmap", {})
            self._srcmap[(layer_key, expert_id)] = src
        return fut

    ExpertCache._submit_slab = sub
    orig_drop = ExpertCache._drop_raw

    def drop(self, key, claim=False):
        raw = orig_drop(self, key, claim=claim)
        if raw is not None and CTX["on"]:
            src = getattr(self, "_srcmap", {}).get(key, "pre")
            FATE[(src, "used" if claim else "wasted")] += 1
        return raw

    ExpertCache._drop_raw = drop

    model, tok = load(args.model)
    cache = model._pagedmoe_cache
    with open(args.prompt_file, encoding="utf-8") as fh:
        ids = tok.encode(fh.read())[: args.prompt_tokens]
    msgs = [{"role": "user", "content": tok.decode(ids) + "\n\nSummarize the above."}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    gen = stream_generate(
        model, tok, prompt, max_tokens=args.max_tokens,
        sampler=make_sampler(temp=0.0), kv_bits=8, kv_group_size=64,
        prefill_step_size=config.PREFILL_CHUNK,
    )
    next(gen)
    CTX["on"] = True
    s0 = cache.spec_skipped
    n = sum(1 for _ in gen)
    per = {k: v / n for k, v in C.items()}
    print("\nper token:", {k: round(v, 1) for k, v in sorted(per.items())})
    print("fate per token:", {f"{s}/{f}": round(v / n, 1) for (s, f), v in sorted(FATE.items())})
    print("spec_skipped/token", (cache.spec_skipped - s0) / n,
          "free slots", len(cache._free_slots), "raw_bytes GB", cache._raw_bytes / 1e9,
          "staging GB", cache.staging_bytes / 1e9, "cap", cache._spec_inflight_cap)


if __name__ == "__main__":
    main()
