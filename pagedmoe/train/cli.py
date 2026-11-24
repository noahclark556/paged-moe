# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""python -m pagedmoe.train

  python -m pagedmoe.train run.yaml                    train from a config file
  python -m pagedmoe.train sft --model M --data D      train with flags
  python -m pagedmoe.train plan run.yaml               plan + probe, no training
  python -m pagedmoe.train resume RUN_DIR              continue a run
  python -m pagedmoe.train chat RUN_DIR                talk to a run's adapter
  python -m pagedmoe.train eval RUN_DIR --data D       loss and answers per checkpoint
  python -m pagedmoe.train export RUN_DIR              adapter as PEFT, for GPUs
  python -m pagedmoe.train bench run MODEL ...         speed and memory benchmark

Any config field works as a flag (--batch-size 16, --lora.rank 32,
--template-kwargs '{"enable_thinking": false}'). Values parse as JSON when
they can, so numbers, booleans, lists and dicts all work.

GRPO rewards: --rewards exact_answer,think_format or file.py:fn / module:fn.
Custom rows and inputs: --formatter file.py:fn, --encoder file.py:make
(returns an nn.Module; gets hidden=<model hidden size> if it takes that
argument). See docs/paged-training.md.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path


def _value(raw: str):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _ref(spec: str):
    """'file.py:name' or 'pkg.module:name' -> object."""
    mod, _, name = spec.rpartition(":")
    if not mod:
        raise ValueError(f"expected module:name, got {spec!r}")
    if mod.endswith(".py") or "/" in mod:
        path = Path(mod).expanduser().resolve()
        sys.path.insert(0, str(path.parent))
        s = importlib.util.spec_from_file_location(path.stem, path)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, name)


def _rewards(spec) -> list:
    from . import rewards as R

    items = spec if isinstance(spec, list) else [s for s in str(spec).split(",") if s]
    out = []
    for item in items:
        item = item.strip()
        if item.startswith("regex:"):
            out.append(R.regex_format(item[len("regex:"):]))
        elif ":" in item:
            out.append(_ref(item))
        elif hasattr(R, item):
            obj = getattr(R, item)
            out.append(obj() if item in ("exact_answer", "think_format") else obj)
        else:
            raise ValueError(f"unknown reward {item!r}")
    return out


def _parse_overrides(rest: list[str]) -> dict:
    out: dict = {}
    i = 0
    while i < len(rest):
        tok = rest[i]
        if not tok.startswith("--"):
            raise SystemExit(f"unexpected argument {tok!r}")
        key = tok[2:]
        if "=" in key:
            key, raw = key.split("=", 1)
            i += 1
        elif i + 1 < len(rest) and not rest[i + 1].startswith("--"):
            raw = rest[i + 1]
            i += 2
        else:
            raw = "true"
            i += 1
        key = key.replace("-", "_")
        if key.startswith("lora."):
            out.setdefault("lora", {})[key[5:]] = _value(raw)
        else:
            out[key] = _value(raw)
    return out


def build(target: str, overrides: dict, **trainer_kw):
    """Config file or method name plus flag overrides -> a constructed trainer.

    trainer_kw goes to the trainer as is (plan_only=True, callbacks=[...]).
    """
    from . import config as C
    from .trainer import TRAINERS

    if target.endswith((".yaml", ".yml", ".json")):
        p = Path(target).expanduser()
        if p.suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(p.read_text()) or {}
        else:
            data = json.loads(p.read_text())
        # relative data paths in a config file mean next to the file
        for k in ("data", "eval_data"):
            v = data.get(k)
            if isinstance(v, str) and not Path(v).expanduser().is_absolute() and not Path(v).exists():
                near = p.parent / v
                if near.exists():
                    data[k] = str(near.resolve())
    else:
        data = {"method": target}
    lora = {**(data.get("lora") or {}), **overrides.pop("lora", {})}
    data.update(overrides)
    if lora:
        data["lora"] = lora
    extras = {k: data.pop(k) for k in ("rewards", "formatter", "encoder") if k in data}
    cfg = C.from_dict(data)
    kw: dict = dict(trainer_kw)
    if "formatter" in extras:
        kw["formatter"] = _ref(extras["formatter"])
    if "encoder" in extras:
        import inspect

        from .plan import model_facts

        make = _ref(extras["encoder"])
        takes_hidden = "hidden" in inspect.signature(make).parameters
        kw["encoder"] = make(hidden=model_facts(cfg.model).hidden) if takes_hidden else make()
    cls = TRAINERS[cfg.method]
    if cfg.method == "grpo":
        if "rewards" in extras:
            kw["reward_funcs"] = _rewards(extras["rewards"])
        elif trainer_kw.get("plan_only"):
            kw["reward_funcs"] = [lambda completion, row: 0.0]
        else:
            raise SystemExit("grpo needs --rewards (e.g. exact_answer,think_format)")
    trainer = cls(cfg, **kw)
    trainer.cli_extras = extras
    return trainer


def resume_config(run_dir, overrides: dict | None = None) -> dict:
    """The config a run was started with, set to continue from its last save."""
    run = Path(run_dir).expanduser()
    rec = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else None
    if rec is None:
        ck = run / "checkpoints"
        if not (ck / "latest").exists():
            raise SystemExit(f"{run} has no run.json and no checkpoints")
        latest = (ck / "latest").read_text().strip()
        st = json.loads((ck / latest / "trainer_state.json").read_text())
        raise SystemExit(
            f"{run} has no run.json yet (stopped at step {st.get('step')}); "
            "rerun the original command with --resume true"
        )
    data = dict(rec["config"])
    if isinstance(data.get("data"), str) and data["data"].startswith("<"):
        raise SystemExit("this run was given in-memory data; resume it from Python with resume=True")
    data.update(rec.get("cli") or {})
    ov = dict(overrides or {})
    lora = {**(data.get("lora") or {}), **ov.pop("lora", {})}
    data.update(ov)
    data["lora"] = lora
    data["resume"] = True
    data["output_dir"] = str(run)
    return data


def resume(run_dir, overrides: dict | None = None, **trainer_kw):
    """A trainer that continues `run_dir` from its last checkpoint."""
    data = resume_config(run_dir, overrides)
    run = Path(run_dir).expanduser()
    tmp = run / ".resume.json"
    tmp.write_text(json.dumps(data, default=str))
    try:
        return build(str(tmp), {}, **trainer_kw)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "plan":
        if len(argv) < 2:
            raise SystemExit("plan needs a config file or method")
        ov = _parse_overrides(argv[2:])
        ov.setdefault("probe", "always")
        build(argv[1], ov)
        return 0
    if cmd == "resume":
        if len(argv) < 2:
            raise SystemExit("resume needs a run directory")
        resume(argv[1], _parse_overrides(argv[2:])).train()
        return 0
    if cmd == "bench":
        from . import bench

        return bench.main(argv[1:])
    if cmd in ("eval", "chat"):
        if len(argv) < 2:
            raise SystemExit(f"{cmd} needs a run directory")
        ov = _parse_overrides(argv[2:])
        return _chat(argv[1], ov) if cmd == "chat" else _eval(argv[1], ov)
    if cmd == "export":
        if len(argv) < 2:
            raise SystemExit("export needs a run or checkpoint directory")
        return _export(argv[1], _parse_overrides(argv[2:]))
    trainer = build(cmd, _parse_overrides(argv[1:]))
    trainer.train()
    return 0


def _eval(run_dir: str, ov: dict) -> int:
    """eval RUN_DIR [--data D] [--checkpoints base,final] [--generate] [--loss false]
    [--limit N] [--max-tokens N] [--temperature T] [--rewards R] [--out results.csv]"""
    from . import data as D
    from .evaluate import Evaluator, checkpoints

    run = Path(run_dir).expanduser()
    rec = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    cfg = rec.get("config") or {}
    src = ov.get("data") or cfg.get("eval_data")
    if not src or str(src).startswith("<"):
        src = cfg.get("data")
        if not src or str(src).startswith("<"):
            raise SystemExit("eval needs --data (the run has no eval data on disk)")
        print("[eval] no eval data given; using the training rows (measures fit, not generalization)")
    rows = D.load_rows(src, "valid") or D.load_rows(src, "train")
    have = [c["id"] for c in checkpoints(run)]
    cks = [c for c in str(ov.get("checkpoints") or "base," + (have[-1] if have else "final")).split(",") if c]
    ev = Evaluator(run, checkpoint=next((c for c in cks if c != "base"), None))
    fns = _rewards(ov["rewards"]) if ov.get("rewards") else None
    t0 = [0.0]

    def prog(p):
        import time

        if time.time() - t0[0] > 2:
            t0[0] = time.time()
            print(f"[eval] {p['checkpoint']}: {p['stage']} {p['done']}/{p['total']}", flush=True)

    rep = ev.run(rows, checkpoints=cks, loss=ov.get("loss", True) not in (False, "false"),
                 generate=bool(ov.get("generate")), limit=ov.get("limit"),
                 max_tokens=int(ov.get("max_tokens", 512)), temperature=float(ov.get("temperature", 0.0)),
                 rewards=fns, on_progress=prog)
    for s in rep["summary"]:
        print("[eval] " + ", ".join(f"{k} {v}" for k, v in s.items()))
    saved = ev.save_report(rep)
    print(f"[eval] report in {saved}")
    if ov.get("out"):
        print(f"[eval] wrote {ev.export(rep, ov['out'])}")
    return 0


def _export(path: str, ov: dict) -> int:
    """export RUN_DIR [--checkpoint ID] [--format peft] [--out DIR] [--base REPO_OR_DIR]

    Writes <adapter>/peft/ unless --out is given. --base is only needed when
    the original HF repo can't be read off the model card."""
    from .export import ExportError, to_peft

    fmt = str(ov.get("format", "peft")).lower()
    if fmt != "peft":
        raise SystemExit(f"unknown export format {fmt!r} (peft is the one there is)")
    try:
        to_peft(path, ov.get("out"), checkpoint=ov.get("checkpoint"), base=ov.get("base"))
    except ExportError as e:
        raise SystemExit(f"[export] {e}")
    return 0


def _chat(run_dir: str, ov: dict) -> int:
    """chat RUN_DIR [--checkpoint ID] [--temperature T] [--max-tokens N]

    /use ID switches checkpoint, /base turns the adapter off, /reset clears
    the conversation, /quit leaves."""
    from .evaluate import Evaluator

    ev = Evaluator(run_dir, checkpoint=ov.get("checkpoint"))
    print(f"[chat] {ev.model_path} with {ev.current}. Checkpoints: {', '.join(c['id'] for c in ev.cks)}, base")
    msgs: list[dict] = []
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return 0
        if line == "/reset":
            msgs = []
            continue
        if line == "/base" or line.startswith("/use "):
            try:
                print(f"[chat] now {ev.use('base' if line == '/base' else line[5:].strip())}")
            except (KeyError, ValueError) as e:
                print(f"[chat] {e}")
            continue
        msgs.append({"role": "user", "content": line})
        reply = ""
        try:
            for piece in ev.chat(msgs, max_tokens=int(ov.get("max_tokens", 1024)),
                                 temperature=float(ov.get("temperature", 0.7))):
                if piece.get("done"):
                    print(f"\n[chat] {piece['tokens']} tokens, {piece['tok_s']} tok/s")
                else:
                    reply += piece["text"]
                    print(piece["text"], end="", flush=True)
        except KeyboardInterrupt:
            ev.cancel.set()
            print("\n[chat] stopped")
        msgs.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    raise SystemExit(main())
