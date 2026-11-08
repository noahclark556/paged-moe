# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Walk one real streamed request and stop inside the engine.

Quit Global Agent (or any other MLX server) before running this. One process
owns the GPU. A second load will run the machine out of memory.

From pagedmoe-core, with the mlx env active:

    python walkthrough/tour.py
    python walkthrough/tour.py --model ~/mlx-models/qwen38-coder-flash --tokens 4

Each stop is a pdb breakpoint on the first line of a major engine function.
You are in that function, with its real arguments.

    c          continue to the next breakpoint
    n          next line in this function
    s          step into the call
    p <name>   print a local (self, indices, expert_ids, ...)
    ll         show the source around this line
    q          abort the tour

q ends the process. It does not resume. Use c to move on.
The breakpoint fires once per stage, at the top of the function.
Later calls of that function run normally.
"""

from __future__ import annotations

import argparse
import pdb
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm import stream_generate  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from pagedmoe import config, load  # noqa: E402
from pagedmoe import cache as cache_mod  # noqa: E402
from pagedmoe import loader as loader_mod  # noqa: E402
from pagedmoe import streaming as streaming_mod  # noqa: E402
from pagedmoe.cache import ExpertCache  # noqa: E402
from pagedmoe.loader import load as load_model  # noqa: E402
from pagedmoe.streaming import (  # noqa: E402
    PrefetchRing,
    StreamedSwitchGLU,
    _DECODE_MAX_TOKENS,
    patch_model,
)

PROMPT = (
    "Explain, in two short sentences, what an expert in a mixture-of-experts "
    "model is, and why a decoder can keep most of those experts on disk."
)


class _Gate:
    def __init__(self) -> None:
        self.hit: set[str] = set()
        self.phase = "prefill"
        self.announce = False


GATE = _Gate()


def _banner(title: str, lines: list[str]) -> None:
    GATE.announce = True
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}", flush=True)
    for line in lines:
        print(line, flush=True)
    print(flush=True)


def _sample(ids, n: int = 8) -> str:
    try:
        return str([int(e) for e in list(ids[:n])])
    except Exception:
        return "?"


def _tour_load(path_or_hf_repo, mode):
    if "load" in GATE.hit:
        return False
    GATE.hit.add("load")
    _banner(
        "Breakpoint: loader.load",
        [
            f"path: {path_or_hf_repo}",
            f"mode argument: {mode!r} (None means the streamed default)",
            "",
            "Top of load. Nothing has been read yet. From here the loader",
            "reads safetensors headers, builds the model with lazy weights,",
            "and swaps every SwitchGLU for StreamedSwitchGLU.",
            "",
            "Locals: path_or_hf_repo, mode, cache_gb, read_threads.",
            "c continues the load. The next stop is patch_model,",
            "after the headers are read and the skeleton exists.",
        ],
    )
    return True


def _tour_patch(model, cache, prefetch_depth):
    if "patch" in GATE.hit:
        return False
    GATE.hit.add("patch")
    _banner(
        "Breakpoint: streaming.patch_model",
        [
            f"prefetch_depth: {prefetch_depth}",
            f"cache budget: {getattr(cache, 'budget_bytes', 0) / 1e9:.1f} GB",
            "",
            "Top of patch_model. The backbone modules exist. Expert weights",
            "are still on disk. This function replaces each mlx-lm SwitchGLU",
            "with StreamedSwitchGLU and builds the prefetch ring.",
            "",
            "Locals: model, cache, tensor_index, prefetch_depth.",
            "c finishes the patch and the rest of load, then prefill starts.",
        ],
    )
    return True


def _tour_forward(self, indices):
    if "patch" not in GATE.hit:
        return False
    try:
        n = int(indices.reshape(-1, indices.shape[-1]).shape[0])
        k = int(indices.shape[-1])
    except Exception:
        return False
    if "prefill_forward" not in GATE.hit and n > _DECODE_MAX_TOKENS:
        GATE.hit.add("prefill_forward")
        GATE.phase = "prefill"
        _banner(
            "Breakpoint: StreamedSwitchGLU._forward (prefill)",
            [
                f"layer: {self.layer_key}",
                f"batch: {n} positions, top-{k}",
                "",
                "Top of the MoE forward. Attention and the router for this",
                "layer have already run. indices holds the expert ids the",
                "router picked. The expert weights are not in RAM yet.",
                "",
                f"A batch over {_DECODE_MAX_TOKENS} positions is prefill.",
                "Prefill computes the full mixture. Flow and prune stay off",
                "on this path, because the KV cache is being written here.",
                "",
                "Locals: self, x, indices.",
                "p self.layer_key",
                "p indices.shape",
                "c runs this layer until fetch_groups.",
            ],
        )
        return True
    if (
        "decode_forward" not in GATE.hit
        and "prefill_forward" in GATE.hit
        and n <= _DECODE_MAX_TOKENS
    ):
        GATE.hit.add("decode_forward")
        GATE.phase = "decode"
        flow = (
            "Flow is on. A later stop is flow_token_start, on layer 0,"
            " before this function decides whether to fetch misses."
            if config.FLOW and config.PRUNE > 0
            else "Flow is off, so this token takes the per-layer path:"
            " sync the router result, then _begin reads misses into slab slots."
        )
        _banner(
            "Breakpoint: StreamedSwitchGLU._forward (decode)",
            [
                f"layer: {self.layer_key}",
                f"batch: {n} position(s), top-{k}",
                "",
                "Top of the MoE forward for a new token. The KV cache holds",
                "the prompt. The backbone is resident. This call may read",
                "a few experts from disk.",
                "",
                flow,
                "",
                "Locals: self, x, indices.",
                "c continues into this token's fetch.",
            ],
        )
        return True
    return False


def _tour_groups(layer_key, expert_ids, group_bytes):
    if "patch" not in GATE.hit or "groups" in GATE.hit:
        return False
    GATE.hit.add("groups")
    _banner(
        "Breakpoint: ExpertCache.fetch_groups",
        [
            f"layer: {layer_key}",
            f"routed experts: {len(expert_ids)}",
            f"first ids: {_sample(expert_ids)}",
            f"group budget: {group_bytes / 1e6:.0f} MB",
            "",
            "Top of the prefill read splitter. One layer can want more",
            "experts than fit in memory at once, so they are read and",
            "computed in groups. Group i runs on the GPU while group i+1",
            "is read from disk.",
            "",
            "Locals: self, layer_key, expert_ids, group_bytes, install.",
            "c enters the first group's disk read (_begin).",
        ],
    )
    return True


def _tour_begin(self, layer_key, expert_ids, prefill, use_slab):
    if "patch" not in GATE.hit:
        return False
    if prefill and "prefill_read" not in GATE.hit:
        GATE.hit.add("prefill_read")
        _banner(
            "Breakpoint: ExpertCache._begin (prefill)",
            [
                f"layer: {layer_key}",
                f"experts in this group: {len(expert_ids)}",
                f"first ids: {_sample(expert_ids)}",
                f"prefill={prefill}  use_slab={use_slab}",
                "",
                "Top of the read. Each id is about to be classified as a",
                "cache hit or a disk read. Misses are coalesced into long",
                "sequential preads (F_NOCACHE). Prefill does not use the",
                "decode slab. The bytes land in the cache.",
                "",
                "Locals: self, layer_key, expert_ids, prefill, use_slab.",
                "c performs the read and finishes this layer.",
                "Later prefill layers do not stop.",
            ],
        )
        return True
    if use_slab and GATE.phase == "decode" and "decode_read" not in GATE.hit:
        GATE.hit.add("decode_read")
        _banner(
            "Breakpoint: ExpertCache._begin (decode)",
            [
                f"layer: {layer_key}",
                f"unique experts: {len(expert_ids)}",
                f"ids: {_sample(expert_ids, 12)}",
                f"prefill={prefill}  use_slab={use_slab}",
                f"hits {self.hits}, misses {self.misses}, "
                f"resident {self.resident_bytes / 1e9:.2f} GB",
                "",
                "Top of the decode read. A hit is already in a slab slot.",
                "A miss is read straight into its own slot, which stays",
                "reserved so the next token can reuse it.",
                "",
                "Locals: self, layer_key, expert_ids, use_slab.",
                "p expert_ids",
                "c finishes this token. Further layers do not stop.",
            ],
        )
        return True
    return False


def _tour_flow(self):
    if "flow" in GATE.hit:
        return False
    GATE.hit.add("flow")
    _banner(
        "Breakpoint: PrefetchRing.flow_token_start",
        [
            "Top of the flow gate, called from layer 0 of a decode token.",
            "The body below updates the token count and the hit-rate",
            "averages. flow_ready() is just self._flow_live.",
            "",
            "The gate needs a couple of tokens of history. Until it is",
            "ready, every layer still fetches misses. Once it is ready,",
            "later layers compute only experts already in slab slots and",
            "write zeros for the rest.",
            "",
            "Locals: self.",
            "n a few times, then p self._flow_tokens",
            "p self._flow_live",
            "c returns to _forward.",
        ],
    )
    return True


def _install_conditions() -> None:
    loader_mod._tour_load = _tour_load
    streaming_mod._tour_patch = _tour_patch
    streaming_mod._tour_forward = _tour_forward
    streaming_mod._tour_flow = _tour_flow
    cache_mod._tour_groups = _tour_groups
    cache_mod._tour_begin = _tour_begin


class TourPdb(pdb.Pdb):
    def __init__(self, stdin=None, stdout=None) -> None:
        super().__init__(readrc=False, stdin=stdin, stdout=stdout)
        self.prompt = "(tour) "
        self._ready = False
        self.aborted = False

    def user_line(self, frame):
        # The first line is the tour runner itself. Skip it so the first
        # place you land is the engine breakpoint.
        if not self._ready:
            self._ready = True
            self.set_continue()
            return
        if GATE.announce:
            GATE.announce = False
            print(
                "c = next breakpoint    n = next line    s = step in\n"
                "p <name> = inspect    ll = source    q = abort",
                flush=True,
            )
        super().user_line(frame)

    def do_quit(self, arg):
        self.aborted = True
        return super().do_quit(arg)


def _arm(dbg: TourPdb, fn, cond: str) -> None:
    code = fn.__code__
    err = dbg.set_break(
        code.co_filename,
        code.co_firstlineno,
        cond=cond,
        funcname=code.co_name,
    )
    if err:
        raise SystemExit(err)
    rel = Path(code.co_filename)
    try:
        rel = rel.resolve().relative_to(Path(__file__).resolve().parent.parent)
    except ValueError:
        pass
    print(f"  {rel}:{code.co_firstlineno}  {fn.__qualname__}", flush=True)


def _arm_all(dbg: TourPdb) -> None:
    _install_conditions()
    print("Breakpoints (first line of each function):", flush=True)
    _arm(dbg, load_model, "_tour_load(path_or_hf_repo, mode)")
    _arm(dbg, patch_model, "_tour_patch(model, cache, prefetch_depth)")
    _arm(dbg, StreamedSwitchGLU._forward, "_tour_forward(self, indices)")
    _arm(dbg, ExpertCache.fetch_groups, "_tour_groups(layer_key, expert_ids, group_bytes)")
    _arm(
        dbg,
        ExpertCache._begin,
        "_tour_begin(self, layer_key, expert_ids, prefill, use_slab)",
    )
    _arm(dbg, PrefetchRing.flow_token_start, "_tour_flow(self)")


def run(args) -> None:
    t0 = time.perf_counter()
    model, tok = load(args.model, mode="streamed", verbose=True)
    prompt = args.prompt
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    while len(tok.encode(text)) <= _DECODE_MAX_TOKENS:
        prompt += " Say it in plain language."
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    n_prompt = len(tok.encode(text))
    print(
        f"\nLoaded in {time.perf_counter() - t0:.1f}s. "
        f"Prompt is {n_prompt} tokens. Generating {args.tokens}.\n"
        f"Flow {'on' if config.FLOW else 'off'}, prune {config.PRUNE}.",
        flush=True,
    )
    prompt_cache = make_prompt_cache(model)
    produced = ""
    t_first = None
    t1 = time.perf_counter()
    for piece in stream_generate(
        model,
        tok,
        text,
        max_tokens=args.tokens,
        sampler=make_sampler(temp=0.0),
        prompt_cache=prompt_cache,
    ):
        if t_first is None:
            t_first = time.perf_counter()
        produced += piece.text
        print(piece.text, end="", flush=True)
    dt = time.perf_counter() - (t_first or t1)
    print(
        f"\n\nDone. {args.tokens} tokens in {dt:.1f}s after the first.\n{produced}\n",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default=str(Path.home() / "mlx-models" / "qwen38-coder-flash"),
    )
    ap.add_argument("--tokens", type=int, default=4)
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    print(
        "One GPU owner. Quit Global Agent or any other pagedmoe server first.\n"
        f"Model: {args.model}\n"
        "You will land in pdb at the top of each major function.\n"
        "c moves to the next breakpoint. q aborts the process.\n",
        flush=True,
    )
    dbg = TourPdb()
    _arm_all(dbg)
    print(flush=True)
    dbg.runcall(run, args)
    if dbg.aborted:
        print(
            "\nTour aborted. q leaves the debugger and stops the process. "
            "c continues to the next breakpoint.\n",
            flush=True,
        )


if __name__ == "__main__":
    main()
