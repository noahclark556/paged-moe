# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""paged-moe convert: plan, run, check and upload conversions."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from . import recipe as R


def _gb(n) -> str:
    return f"{(n or 0) / 1e9:.2f} GB"


def parse_spec(raw: str):
    """4 | keep | model | mxfp4 | mxfp8 | nvfp4 | 4/32 (bits/group size)."""
    s = str(raw).strip().lower()
    if s in ("keep", "float", "none"):
        return "keep"
    if s == "model":
        return "model"
    if s in R.MODES and R.MODES[s]:
        return {"mode": s}
    if "/" in s:
        b, g = s.split("/", 1)
        return {"bits": int(b), "group_size": int(g), "mode": "affine"}
    try:
        return int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{raw!r} isn't bits, keep, model, mxfp4, mxfp8, nvfp4 or bits/group") from None


def _rule(raw: str) -> dict:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("rules look like REGEX=SETTING, e.g. 'layers\\.0\\.=8'")
    pat, _, val = raw.rpartition("=")
    spec = parse_spec(val)
    r = {"match": pat}
    if isinstance(spec, dict):
        r.update(spec)
    else:
        r["bits"] = spec
    return r


def add_recipe_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("recipe")
    g.add_argument("--preset", default=None, help=f"start from a preset: {', '.join(R.PRESETS)} (default balanced)")
    g.add_argument("--recipe", default=None, help="recipe JSON file, or a pagedmoe_conversion.json to redo one")
    for c in R.COMPONENTS:
        g.add_argument(f"--{c.replace('_', '-')}", dest=c, type=parse_spec, default=None, metavar="S",
                       help=f"{R.LABELS[c].lower()}: bits, keep, model, mxfp4/mxfp8/nvfp4 or bits/group")
    g.add_argument("--mode", choices=list(R.MODES), default=None, help="quantization mode for plain bit counts")
    g.add_argument("--group-size", type=int, choices=R.GROUP_SIZES, default=None)
    g.add_argument("--dtype", choices=R.DTYPES, default=None, help="float dtype for scales and kept tensors")
    g.add_argument("--edge-layers", type=int, default=None, help="first and last N layers get --edge-bits")
    g.add_argument("--edge-bits", type=int, choices=R.AFFINE_BITS, default=None)
    g.add_argument("--edge-experts", action="store_true", default=None, help="edge bump applies to experts too")
    g.add_argument("--rule", action="append", type=_rule, default=None, metavar="REGEX=S",
                   help="per-module override, first match wins (repeatable)")
    g.add_argument("--target", choices=R.TARGETS, default=None, help="for --preset fit")


def recipe_from_args(a) -> R.Recipe:
    d: dict = {}
    if a.recipe:
        data = json.loads(Path(a.recipe).expanduser().read_text())
        d = dict(data.get("recipe") or data) if isinstance(data, dict) else {}
    if a.preset:
        d["preset"] = a.preset
        for c in R.COMPONENTS:
            d.pop(c, None)
    for k in R.COMPONENTS + ("mode", "group_size", "dtype", "edge_layers", "edge_bits", "edge_experts", "target"):
        v = getattr(a, k, None)
        if v is not None:
            d[k] = v
    if a.rule:
        d["rules"] = list(d.get("rules") or []) + a.rule
    return R.Recipe.from_dict(d)


# ------------------------------------------------------------ printing


def print_plan(s: dict, compare_rows=None) -> None:
    src = s["source"]
    print(f"source   {src['ref']}" + (f" @ {str(src['revision'])[:12]}" if src.get("revision") else ""))
    print(f"model    {s['model_type']} ({s['arch']}), {s['layers']} layers"
          + (f", {s['experts']} experts, {s['experts_per_token']} per token" if s.get("experts") else "")
          + f", {s['params'] / 1e9:.1f}B params")
    if s.get("adapter"):
        ad = s["adapter"]
        print(f"adapter  {ad['kind']} rank {ad['rank']}, {ad['modules']} modules, merged")
    fc = s.get("fit_choice")
    if fc:
        print(f"fit      picked {fc['picked']}: {fc['why']}")
    print()
    print(f"  {'part':34s} {'setting':22s} {'float':>10s} {'out':>10s}")
    for r in s["components"]:
        print(f"  {r['label'][:34]:34s} {(r.get('setting') or '-')[:22]:22s} {_gb(r['float_bytes']):>10s} "
              f"{_gb(r['out_bytes']):>10s}")
        for k in r.get("kept") or []:
            print(f"    {k['modules']} kept float: {k['why']}")
    print()
    print(f"output   {_gb(s['out_bytes'])} ({s['bits_per_weight']} bits/weight), "
          f"experts {_gb(s['expert_bytes'])}, backbone {_gb(s['backbone_bytes'])}")
    fit = s.get("fit") or {}
    line = f"this Mac {fit.get('status', '?')}"
    if fit.get("reason"):
        line += f": {fit['reason']}"
    print(line)
    if fit.get("cold_tok_s"):
        print(f"         about {fit['cold_tok_s']} tok/s with a cold cache, "
              f"{int(100 * (fit.get('cached_share') or 0))}% of the experts fit in the cache")
    d = s.get("disk") or {}
    if d.get("free") is not None:
        print(f"disk     needs {_gb(d['need'])}" + (f" (incl. {_gb(d['staging'])} source download)" if d.get("staging") else "")
              + f", {_gb(d['free'])} free")
    if s.get("windowed"):
        print("stream   downloads one layer at a time, the source never fully lands on disk")
    print(f"memory   about {_gb(s['peak_bytes'])} at peak")
    for w in s.get("warnings") or []:
        print(f"note     {w}")
    if compare_rows:
        print()
        print(f"  {'preset':10s} {'size':>10s} {'bits/w':>7s} {'fit':10s} {'cold tok/s':>10s}")
        for r in compare_rows:
            print(f"  {r['preset']:10s} {_gb(r['out_bytes']):>10s} {r['bits_per_weight']:7.2f} {str(r['fit']):10s} "
                  f"{(r['cold_tok_s'] or '-')!s:>10s}")


class _Printer:
    def __init__(self, as_json: bool):
        self.json = as_json
        self.last = 0.0

    def __call__(self, ev: dict) -> None:
        if self.json:
            print(json.dumps(ev, default=str), flush=True)
            return
        k = ev["kind"]
        if k == "phase":
            extra = ev.get("why") or (f"{ev['done']}/{ev['units']} units already done" if "done" in ev else "")
            print(f"[{ev['phase']}] {extra}".rstrip(), flush=True)
        elif k == "download" and ev.get("state") == "done":
            print(f"  got {ev['file']} ({_gb(ev['bytes'])}, {ev.get('gb_s')} GB/s)", flush=True)
        elif k == "unit_done":
            snr = ", ".join(f"{c} {v} dB" for c, v in (ev.get("snr_db") or {}).items() if v is not None)
            print(f"  {ev['done']}/{ev['units']} {ev['key']}  {_gb(ev['out_bytes'])} in {ev['seconds']} s"
                  + (f"  ({snr})" if snr else ""), flush=True)
        elif k == "progress":
            now = time.time()
            if now - self.last > 10:
                self.last = now
                eta = ev.get("eta_s")
                print(f"  {_gb(ev['done_bytes'])} / {_gb(ev['total_bytes'])}  {ev['gb_s']} GB/s"
                      + (f"  eta {int(eta // 60)}m{int(eta % 60):02d}s" if eta else "")
                      + f"  mem {ev['active_gb']} GB", flush=True)
        elif k == "stopped":
            print(f"stopped after {ev['done']}/{ev['units']} units. Run the same command again to resume.")
        elif k == "error":
            print(f"error: {ev['message']}", file=sys.stderr)


# ------------------------------------------------------------ commands


def cmd_presets(a) -> int:
    for name, p in R.PRESETS.items():
        print(f"{name:10s} {p['about']}")
        if name != "fit":
            r = R.Recipe.from_preset(name)
            print(" " * 11 + ", ".join(f"{c} {r.describe(c)}" for c in ("experts", "attention", "embed", "lm_head")))
    return 0


def cmd_plan(a) -> int:
    from .plan import compare, make

    plan = make(a.source, recipe_from_args(a), out_dir=a.out, adapter=a.adapter,
                stream=False if a.no_stream else None, revision=a.revision, ctx=a.ctx, shard_gb=a.shard_gb)
    rows = compare(plan) if a.compare else None
    if a.json:
        print(json.dumps({"summary": plan.summary(), "compare": rows, "quantization": plan.quant_config()},
                         indent=2, default=str))
    else:
        print_plan(plan.summary(), rows)
    return 0


def cmd_run(a) -> int:
    from .run import convert

    stop = {"n": 0}

    def on_sig(*_):
        stop["n"] += 1
        if stop["n"] > 1:
            raise KeyboardInterrupt
        print("\nstopping after this tensor (Ctrl-C again to quit now)", file=sys.stderr)

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    printer = _Printer(a.json)
    recipe = recipe_from_args(a)

    def on_event(ev):
        if ev["kind"] == "plan" and not a.json:
            print_plan(ev["summary"])
            print()
        else:
            printer(ev)

    rep = convert(a.source, a.out, recipe, adapter=a.adapter, stream=False if a.no_stream else None,
                  revision=a.revision, shard_gb=a.shard_gb, ctx=a.ctx, resume=not a.no_resume,
                  overwrite=a.overwrite, measure=not a.no_measure, keep_source=a.keep_source,
                  ignore_disk=a.ignore_disk, on_event=on_event, should_stop=lambda: stop["n"] > 0)
    if rep["status"] != "done":
        return 3
    if not a.json:
        print(f"\ndone: {rep['out']}  {_gb(rep['out_bytes'])} in {rep['wall_seconds']} s")
        we = (rep.get("weight_error") or {}).get("components") or {}
        if we:
            print("weight SNR  " + ", ".join(f"{c} {v['snr_db']} dB" for c, v in we.items()))
    if a.upload:
        from .hub import upload

        url = upload(a.out, a.upload, private=not a.public)
        print(f"uploaded to {url}")
    return 0


def cmd_check(a) -> int:
    from .check import check

    printer = _Printer(a.json)
    res = check(a.model, a.reference, prompts=a.prompts, max_tokens=a.tokens, on_event=printer)
    if a.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"top-1 agreement {res['top1'] * 100:.1f}%   KL {res['kl']:.4f} nats/token   "
              f"perplexity {res['ppl_ref']:.3f} -> {res['ppl']:.3f}  ({res['tokens']} tokens)")
    return 0


def cmd_upload(a) -> int:
    from .hub import upload

    print(upload(a.dir, a.repo, private=not a.public))
    return 0


def main(argv=None, prog: str = "paged-moe convert") -> int:
    ap = argparse.ArgumentParser(prog=prog,
                                 description="Convert Hugging Face checkpoints to MLX, your way.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("presets", help="list the recipe presets")

    def common(p, out_required: bool):
        p.add_argument("source", help="local folder or Hugging Face repo id (org/name)")
        if out_required:
            p.add_argument("out", help="output folder")
        else:
            p.add_argument("--out", default=None, help="where it would go (for the disk check)")
        p.add_argument("--revision", default=None, help="repo branch, tag or commit")
        p.add_argument("--adapter", default=None, help="LoRA/DoRA adapter or training run folder to merge in")
        p.add_argument("--no-stream", action="store_true", help="download the whole repo first")
        p.add_argument("--shard-gb", type=float, default=5.0)
        p.add_argument("--ctx", type=int, default=32768, help="context length for the fit estimate")
        p.add_argument("--json", action="store_true", help="machine-readable output")
        add_recipe_args(p)

    p = sub.add_parser("plan", help="sizes, fit and settings, without converting")
    common(p, False)
    p.add_argument("--compare", action="store_true", help="also compare every preset")

    p = sub.add_parser("run", help="convert")
    common(p, True)
    p.add_argument("--no-measure", action="store_true", help="skip the per-tensor error measurement")
    p.add_argument("--keep-source", action="store_true", help="keep downloaded shards")
    p.add_argument("--overwrite", action="store_true", help="replace what's in the output folder")
    p.add_argument("--no-resume", action="store_true", help="start over even if a stopped run matches")
    p.add_argument("--ignore-disk", action="store_true", help="convert even if the disk check fails")
    p.add_argument("--upload", default=None, metavar="REPO", help="upload to this Hugging Face repo when done")
    p.add_argument("--public", action="store_true", help="make the uploaded repo public")

    p = sub.add_parser("check", help="compare a converted model's logits against a reference")
    p.add_argument("model", help="converted model folder")
    p.add_argument("reference", help="reference model folder (the float or a higher-bit build)")
    p.add_argument("--prompts", default=None, help="text or jsonl file, default is a built-in mix")
    p.add_argument("--tokens", type=int, default=2048, help="tokens to score")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("upload", help="upload a converted folder to the Hugging Face Hub")
    p.add_argument("dir")
    p.add_argument("repo")
    p.add_argument("--public", action="store_true")

    a = ap.parse_args(argv)
    from .plan import PlanError
    from .run import ConvertError
    from .source import SourceError

    try:
        return {"presets": cmd_presets, "plan": cmd_plan, "run": cmd_run, "check": cmd_check,
                "upload": cmd_upload}[a.cmd](a)
    except (PlanError, SourceError, ConvertError, R.RecipeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
