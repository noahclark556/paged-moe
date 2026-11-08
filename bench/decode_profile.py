#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where does a decode token's wall time actually go?

Times the *current* production code path by wrapping methods (no logic
duplication - the old version of this probe forked an outdated __call__ and
silently bypassed the slab/prune paths).

Per decode token, wall time decomposes as:

  moe_total   sum of StreamedSwitchGLU.__call__ wall time (all MoE layers)
    fetch     ExpertCache.fetch wall (disk wait not hidden + install)
    spec      route prediction + prefetch queueing (ring.speculate)
    compute   _run_slab/_run graph construction + kernel launch (lazy)
    sync+py   the remainder: router eval sync (GPU drain) + python glue
  attn+rest   step time minus moe_total: attention, sampler, mlx-lm loop

Also reports decode-only cache deltas: miss GB/token, hit rate, staged
prediction efficiency, disk wait.

Usage:
  PAGED_MOE_CACHE_GB=28 PAGED_MOE_PRUNE=0.5 \
  python bench/decode_profile.py --model ~/mlx-models/qwen3-235-4bit \
      --max-tokens 48 --prompt-tokens 2048
"""

import argparse
import json
import threading
import time

import mlx.core as mx
import numpy as np
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from pagedmoe import config, load
from pagedmoe.cache import ExpertCache
from pagedmoe.streaming import PrefetchRing, StreamedSwitchGLU

T = {"moe": 0.0, "fetch": 0.0, "spec": 0.0, "compute": 0.0, "eval": 0.0,
     "flush": 0.0, "note": 0.0, "asarray": 0.0}
COUNT = {"moe_calls": 0, "fetch_calls": 0, "eval_calls": 0}

# Time every main-thread mx.eval: this is where the lazy graph actually
# executes (GPU drain), as opposed to python glue between evals. Note the
# eval inside cache._finish overlaps the "fetch" bucket.
_MAIN = threading.get_ident()
_orig_eval = mx.eval


def _timed_eval(*a, **k):
    if threading.get_ident() != _MAIN:
        return _orig_eval(*a, **k)
    t0 = time.perf_counter()
    try:
        return _orig_eval(*a, **k)
    finally:
        T["eval"] += time.perf_counter() - t0
        COUNT["eval_calls"] += 1


mx.eval = _timed_eval


class _TimedNumpy:
    """numpy stand-in for streaming.py: times main-thread asarray, which is
    where the router sync lands when nothing else evaluates first."""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asarray(*a, **k):
        if threading.get_ident() != _MAIN:
            return np.asarray(*a, **k)
        t0 = time.perf_counter()
        try:
            return np.asarray(*a, **k)
        finally:
            T["asarray"] += time.perf_counter() - t0


def _wrap(cls, name, bucket, count_key=None):
    orig = getattr(cls, name)

    def timed(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return orig(self, *a, **k)
        finally:
            T[bucket] += time.perf_counter() - t0
            if count_key:
                COUNT[count_key] += 1

    setattr(cls, name, timed)
    return orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--json", action="store_true", help="print JSON summary line")
    ap.add_argument(
        "--prompt-file",
        help="use this file's text (trimmed to --prompt-tokens) instead of the "
        "repeated paragraph; repeated text routes far more warmly than agent traffic",
    )
    ap.add_argument("--ask", default="Summarize the above.")
    ap.add_argument(
        "--then-prompt-file",
        help="run a second generation on this file (same process, warm slab) "
        "and report that one: the shape of a new agent turn",
    )
    ap.add_argument("--then-ask", default=None)
    ap.add_argument("--draft", help="draft model path for speculative decoding")
    ap.add_argument("--draft-tokens", type=int, default=3)
    ap.add_argument("--kv-bits", type=int, default=8,
                    help="KV cache quantization bits (0 = fp16 KV)")
    ap.add_argument("--trace", help="write decode demand trace (layer, ids) to file")
    ap.add_argument(
        "--repeat",
        action="store_true",
        help="run the generation twice and report the second pass (warm "
        "cache: the same greedy tokens route to the same experts, so pass 2 "
        "approximates the zero-disk floor of the decode loop)",
    )
    args = ap.parse_args()

    import pagedmoe.streaming as _streaming
    _streaming.np = _TimedNumpy()
    _wrap(StreamedSwitchGLU, "__call__", "moe", "moe_calls")
    _wrap(StreamedSwitchGLU, "_run_slab", "compute")
    _wrap(StreamedSwitchGLU, "_run", "compute")
    _wrap(ExpertCache, "fetch", "fetch", "fetch_calls")
    _wrap(PrefetchRing, "speculate", "spec")
    _wrap(PrefetchRing, "flush", "flush")
    _wrap(PrefetchRing, "note_demand", "note")

    model, tokenizer = load(args.model, verbose=True)
    cache = model._pagedmoe_cache
    if args.trace:
        cache.demand_trace = []
    draft_model = None
    if args.draft:
        draft_model, _ = load(args.draft, verbose=True)

    # Deterministic prompt of the requested length (same trick as the
    # memory benchmark: repeat a paragraph, then trim to token count).
    para = (
        "The design of storage systems involves fundamental tradeoffs between "
        "capacity, bandwidth, and latency. Solid state drives offer high "
        "sequential throughput but random access patterns reveal their "
        "internal parallelism limits. Memory hierarchies exploit locality. "
    )
    def render(path, ask):
        source = para * 400
        if path:
            with open(path, encoding="utf-8") as fh:
                source = fh.read()
        ids = tokenizer.encode(source)[: args.prompt_tokens]
        msgs = [{"role": "user", "content": tokenizer.decode(ids) + "\n\n" + ask}]
        return tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True
        )

    toks = render(args.prompt_file, args.ask)
    print(f"[probe] prompt tokens: {len(toks)}")
    prompts = [toks]
    if args.then_prompt_file:
        prompts.append(render(args.then_prompt_file, args.then_ask or args.ask))
    elif args.repeat:
        prompts.append(toks)

    sampler = make_sampler(temp=0.0)
    passes = len(prompts)
    steps = []
    snap = {}
    text = []
    for p in range(passes):
        steps = []
        text = []
        t_prev = None
        t_start = time.perf_counter()
        for chunk in stream_generate(
            model,
            tokenizer,
            prompts[p],
            max_tokens=args.max_tokens,
            sampler=sampler,
            kv_bits=args.kv_bits or None,
            kv_group_size=64,
            prefill_step_size=config.PREFILL_CHUNK,
            draft_model=draft_model,
            num_draft_tokens=args.draft_tokens,
        ):
            now = time.perf_counter()
            text.append(chunk.text)
            if t_prev is not None:
                steps.append(now - t_prev)
            else:
                print(f"[pass {p + 1} prefill+tok1: {now - t_start:.2f}s]")
                for k in T:
                    T[k] = 0.0
                for k in COUNT:
                    COUNT[k] = 0
                snap = {
                    "disk_wait_s": cache.disk_wait_s,
                    "bytes_read": cache.bytes_read,
                    "hits": cache.hits,
                    "misses": cache.misses,
                    "spec_wasted": cache.spec_wasted_bytes,
                    "spec_skipped": cache.spec_skipped,
                    "staging_drops": cache.staging_drops,
                    "pred_issued": cache.pred_issued,
                    "pred_used": cache.pred_used,
                    "route_total": cache.route_total,
                    "spec_s": cache.spec_s,
                }
            t_prev = now

    decode_s = sum(steps)
    n = len(steps)
    if not n:
        print("no decode steps measured")
        return

    d = lambda key, attr: getattr(cache, attr) - snap[key]  # noqa: E731
    dw = d("disk_wait_s", "disk_wait_s")
    rd = d("bytes_read", "bytes_read") / 1e9
    hits, misses = d("hits", "hits"), d("misses", "misses")
    spec_s = d("spec_s", "spec_s")
    per = 1000.0 / n
    moe, fetch, spec, comp = T["moe"], T["fetch"], T["spec"], T["compute"]
    flush, note = T.get("flush", 0.0), T.get("note", 0.0)
    sync_py = moe - fetch - spec - comp - flush - note

    print(f"\ndecode: {n} tokens in {decode_s:.2f}s = {n / decode_s:.2f} tok/s")
    head = min(32, n // 2)
    if head:
        print(
            f"first {head} tok: {head / sum(steps[:head]):.2f} tok/s, "
            f"rest: {(n - head) / sum(steps[head:]):.2f} tok/s"
        )
    print(f"read during decode: {rd:.2f} GB ({rd / n * 1000:.0f} MB/token)")
    print(f"decode hit rate: {hits / max(1, hits + misses):.3f} "
          f"({hits} hits / {misses} misses)")
    print(f"  moe layers total   : {moe * per:7.1f} ms/token "
          f"({COUNT['moe_calls'] // n} layer calls/token)")
    print(f"    fetch (wall)     : {fetch * per:7.1f} ms/token")
    print(f"      of which disk  : {dw * per:7.1f} ms/token")
    print(f"    speculate        : {spec * per:7.1f} ms/token "
          f"(cache.spec_s {spec_s * per:.1f})")
    print(f"    compute launch   : {comp * per:7.1f} ms/token")
    print(f"    flush (pred io)  : {flush * per:7.1f} ms/token")
    print(f"    note_demand      : {note * per:7.1f} ms/token")
    print(f"    sync + python    : {sync_py * per:7.1f} ms/token")
    print(f"  attn + sampler     : {(decode_s - moe) * per:7.1f} ms/token")
    print(f"  [mx.eval main-thread total: {T['eval'] * per:7.1f} ms/token over "
          f"{COUNT['eval_calls'] // n} calls/token - overlaps buckets above]")
    print(f"spec during decode: wasted {d('spec_wasted', 'spec_wasted_bytes') / 1e9:.2f} GB, "
          f"skipped {d('spec_skipped', 'spec_skipped')}, "
          f"staging_drops {d('staging_drops', 'staging_drops')}")
    pi, pu = d("pred_issued", "pred_issued"), d("pred_used", "pred_used")
    rt = d("route_total", "route_total")
    print(f"prediction: precision {pu / max(1, pi):.3f}  coverage {pu / max(1, rt):.3f} "
          f"(issued {pi}, used {pu}, demand {rt})")
    st = cache.stats()
    print(f"cache: {json.dumps(st)}")
    if args.trace:
        with open(args.trace, "w") as fh:
            json.dump(cache.demand_trace, fh)
        print(f"trace: {len(cache.demand_trace)} layer-calls -> {args.trace}")
    print(f"text: {''.join(text)[:200]!r}")

    if args.json:
        print("JSON:" + json.dumps({
            "tok_s": n / decode_s,
            "ms_token": decode_s / n * 1000,
            "moe_ms": moe * per,
            "fetch_ms": fetch * per,
            "disk_ms": dw * per,
            "spec_ms": spec * per,
            "compute_ms": comp * per,
            "sync_py_ms": sync_py * per,
            "flush_ms": flush * per,
            "note_ms": note * per,
            "eval_ms": T["eval"] * per,
            "asarray_ms": T["asarray"] * per,
            "attn_ms": (decode_s - moe) * per,
            "read_gb_token": rd / n,
            "hit_rate": hits / max(1, hits + misses),
            "pred_precision": pu / max(1, pi),
            "pred_coverage": pu / max(1, rt),
        }))


if __name__ == "__main__":
    main()
