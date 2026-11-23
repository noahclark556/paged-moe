# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Try a run's adapter: chat with it, and score it on rows.

The base model loads once with the run's LoRA layers on it. Switching
checkpoints only swaps the adapter weights, and "base" scales every adapter
to zero (exact, so it is the untouched model). Loading goes through
pagedmoe.load like serving does, so a model too big for RAM streams its
experts here too.

    from pagedmoe.train.evaluate import Evaluator

    ev = Evaluator("~/.paged_moe/train/runs/sft-qwen")
    ev.use("step-000200")                       # or "final" / "latest" / "base"
    for piece in ev.chat([{"role": "user", "content": "hi"}]):
        print(piece["text"], end="")
    rep = ev.run(rows, checkpoints=["base", "final"], loss=True, generate=True)
    ev.export(rep, "eval.csv")

CLI: python -m pagedmoe.train eval RUN_DIR --data valid.jsonl ...
     python -m pagedmoe.train chat RUN_DIR [--checkpoint step-000200]
"""

from __future__ import annotations

import csv
import json
import math
import re
import threading
import time
from pathlib import Path

import mlx.core as mx

from . import data as D
from . import lora as train_lora

LOSS_CHUNK = 512  # tokens per forward when scoring a row (bounds the logits)


class Cancelled(Exception):
    pass


def _json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def checkpoints(run_dir) -> list[dict]:
    """Adapters a run has on disk, oldest first, plus the run folder's own.

    The run folder holds a copy of the last save: "final" once the run
    finished, "latest" while it is going or was stopped.
    """
    run = Path(run_dir).expanduser()
    out = []
    for p in sorted((run / "checkpoints").glob("step-*")):
        if not (p / "adapters.safetensors").exists():
            continue
        st = _json(p / "trainer_state.json") or {}
        try:
            step = int(st.get("step", int(p.name.split("-")[1])))
        except ValueError:
            step = None
        out.append({"id": p.name, "path": str(p), "step": step, "total_steps": st.get("total_steps"),
                    "time": st.get("time") or p.stat().st_mtime})
    if (run / "adapters.safetensors").exists():
        rec = _json(run / "run.json") or {}
        summ = rec.get("summary") or {}
        step = summ.get("steps") if summ else (out[-1]["step"] if out else None)
        out.append({"id": "final" if summ else "latest", "path": str(run), "step": step,
                    "total_steps": summ.get("total_steps") if summ else None,
                    "time": (run / "adapters.safetensors").stat().st_mtime})
    return out


def _expert_keys(acfg: dict) -> bool:
    keys = (acfg.get("lora_parameters") or {}).get("keys") or []
    return any("switch" in k or "experts" in k for k in keys)


def split_row(row: dict):
    """(prompt, reference, tools) for generation, or None for rows with no prompt."""
    if "messages" in row or "conversations" in row:
        msgs = row.get("messages") or D._sharegpt(row["conversations"])
        k = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=None)
        if k is None:
            return msgs, row.get("answer"), row.get("tools")
        ref = msgs[k].get("content")
        return msgs[:k], row.get("answer", ref), row.get("tools")
    if "prompt" in row:
        ref = row.get("completion", row.get("chosen", row.get("answer")))
        if isinstance(ref, list):
            ref = next((m.get("content") for m in reversed(ref) if isinstance(m, dict)), None)
        return row["prompt"], ref, None
    return None


def _strip_think(s: str) -> str:
    return str(s or "").split("</think>")[-1].strip()


def _norm(s) -> str:
    return re.sub(r"\s+", " ", _strip_think(s)).strip().lower().rstrip(".")


def _f1(got: str, want: str) -> float:
    a = re.findall(r"\w+", _strip_think(got).lower())
    b = re.findall(r"\w+", str(want).lower())
    if not a or not b:
        return float(a == b)
    common = {}
    for w in b:
        common[w] = common.get(w, 0) + 1
    hit = 0
    for w in a:
        if common.get(w, 0) > 0:
            hit += 1
            common[w] -= 1
    if not hit:
        return 0.0
    p, r = hit / len(a), hit / len(b)
    return 2 * p * r / (p + r)


def score(output: str, row: dict, ref) -> dict:
    """Simple per-row checks against the reference. Empty when there is none."""
    from . import rewards as R

    out: dict = {}
    if row.get("answer") is not None:
        out["exact"] = R.exact_answer()(output, row)
    elif ref:
        out["exact"] = float(_norm(output) == _norm(ref))
    if ref:
        out["contains"] = float(_norm(ref) in _norm(output))
        out["f1"] = round(_f1(output, ref), 4)
    return out


class Evaluator:
    """One loaded model, any of a run's checkpoints on it."""

    def __init__(self, run_dir, *, checkpoint: str | None = None, mode: str | None = None,
                 cache_gb: float | None = None, verbose: bool = True):
        self.run_dir = Path(run_dir).expanduser()
        self.cks = checkpoints(self.run_dir)
        if not self.cks:
            raise FileNotFoundError(f"{self.run_dir} has no saved adapter yet (the first save writes one)")
        rec = _json(self.run_dir / "run.json") or {}
        cfg = rec.get("config") or {}
        first = self._find(checkpoint) if checkpoint and checkpoint != "base" else self.cks[-1]
        self.acfg = _json(Path(first["path"]) / "adapter_config.json") or {}
        self.model_path = cfg.get("model") or self.acfg.get("base_model")
        if not self.model_path:
            raise ValueError(f"{self.run_dir}: cannot tell the base model (no run.json, no base_model)")
        self.method = cfg.get("method") or self.acfg.get("method")
        self.template_kwargs = dict(cfg.get("template_kwargs") or {})
        self.system = cfg.get("system_prompt")
        self.chat_template = cfg.get("chat_template")
        self.completion_only = cfg.get("completion_only", True)
        plan = rec.get("plan") or {}
        self.max_len = int(cfg.get("max_seq_len") or plan.get("max_len") or 8192)
        self.dora = self.acfg.get("fine_tune_type") == "dora"
        if _expert_keys(self.acfg):
            if mode == "streamed":
                raise ValueError("this adapter trains the routed experts, which needs the model resident")
            mode = "resident"
        self.cancel = threading.Event()

        from .. import load

        t0 = time.perf_counter()
        kw = {"mode": mode, "verbose": verbose, "adapter_path": first["path"]}
        if cache_gb:
            kw["cache_gb"] = cache_gb
        self.model, self.tok = load(self.model_path, **kw)
        self.model.eval()
        self.info = dict(getattr(self.model, "_pagedmoe_info", {}) or {})
        self.load_s = time.perf_counter() - t0
        self.current = first["id"]
        self.loaded = first["path"]
        self.base = False
        if checkpoint == "base":
            self.use("base")
        self.fmt = D.Formatter(self.tok, max_len=self.max_len, chat=self.chat_template,
                               completion_only=self.completion_only,
                               template_kwargs=self.template_kwargs, system=self.system)

    # ------------------------------------------------------------ checkpoints

    def _find(self, cid: str) -> dict:
        for c in self.cks:
            if c["id"] == cid or c["path"] == str(Path(cid).expanduser()):
                return c
        if cid in ("final", "latest") and self.cks:
            return self.cks[-1]
        raise KeyError(f"no checkpoint {cid!r} in {self.run_dir} (have: {', '.join(c['id'] for c in self.cks)})")

    def refresh(self) -> list[dict]:
        """Rescan the run folder (a run still going keeps saving)."""
        self.cks = checkpoints(self.run_dir)
        return self.cks

    def use(self, cid: str) -> str:
        """Put a checkpoint's adapter on the model. "base" turns LoRA off."""
        if cid == "base":
            if self.dora:
                raise ValueError("a DoRA adapter cannot be turned off exactly; compare against a plain load instead")
            self.base = True
            self.current = "base"
            return self.current
        c = self._find(cid)
        acfg = _json(Path(c["path"]) / "adapter_config.json") or {}
        a = self.acfg.get("lora_parameters") or {}
        b = acfg.get("lora_parameters") or {}
        if (a.get("rank"), sorted(a.get("keys") or []), self.acfg.get("num_layers")) != (
                b.get("rank"), sorted(b.get("keys") or []), acfg.get("num_layers")):
            raise ValueError(f"{c['id']} has a different LoRA shape from the loaded adapter; load it on its own")
        if c["path"] != self.loaded:
            train_lora.load_weights(self.model, Path(c["path"]))
            self.loaded = c["path"]
        self.base = False
        self.current = c["id"]
        return self.current

    def _scope(self):
        import contextlib

        return train_lora.disabled(self.model) if self.base else contextlib.nullcontext()

    # ------------------------------------------------------------------ chat

    def prompt_ids(self, prompt, tools=None) -> list[int]:
        if isinstance(prompt, list) and self.fmt.chat:
            return self.fmt.encode(self.fmt.render(self.fmt._msgs(prompt), gen=True, tools=tools))
        return self.fmt.prompt_ids(prompt)

    def chat(self, messages, *, max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 1.0,
             top_k: int = 0, min_p: float = 0.0, seed: int | None = None, tools=None):
        """Stream a reply: {"text": piece} dicts, then one {"done": True, ...stats}."""
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler

        self.cancel.clear()
        ids = self.prompt_ids(messages, tools=tools)
        if seed is not None:
            mx.random.seed(int(seed))
        sampler = make_sampler(temp=float(temperature), top_p=float(top_p), min_p=float(min_p),
                               top_k=int(top_k))
        t0 = time.perf_counter()
        first = None
        last = None
        text = []
        with self._scope():
            for r in stream_generate(self.model, self.tok, ids, max_tokens=int(max_tokens), sampler=sampler):
                if first is None:
                    first = time.perf_counter() - t0
                last = r
                if r.text:
                    text.append(r.text)
                    yield {"text": r.text}
                if self.cancel.is_set():
                    break
        dt = time.perf_counter() - t0
        n = int(getattr(last, "generation_tokens", 0) or 0)
        yield {
            "done": True,
            "text_all": "".join(text),
            "checkpoint": self.current,
            "prompt_tokens": len(ids),
            "tokens": n,
            "prompt_tok_s": round(float(getattr(last, "prompt_tps", 0) or 0), 2),
            "tok_s": round(float(getattr(last, "generation_tps", 0) or 0), 2),
            "first_token_s": round(first or 0.0, 3),
            "seconds": round(dt, 3),
            "peak_gb": round(float(getattr(last, "peak_memory", 0) or 0), 2),
            "finish": "stopped" if self.cancel.is_set() else (getattr(last, "finish_reason", None) or "length"),
        }

    def generate(self, prompt, **kw) -> dict:
        out = {}
        for piece in self.chat(prompt, **kw):
            if piece.get("done"):
                out = piece
        return out

    # ------------------------------------------------------------------ loss

    def row_loss(self, ex: D.Example) -> tuple[float, int]:
        """Summed NLL over the row's trained tokens, and how many there were.

        Forward in LOSS_CHUNK pieces with a KV cache, so the logits for a long
        row never sit in memory at once. Same tokens and mask as training.
        """
        from mlx_lm.models.cache import make_prompt_cache

        ids, mask = ex.ids, ex.loss_mask
        cache = make_prompt_cache(self.model)
        total, count = 0.0, 0
        with self._scope():
            for a in range(0, len(ids) - 1, LOSS_CHUNK):
                if self.cancel.is_set():
                    raise Cancelled()
                b = min(len(ids) - 1, a + LOSS_CHUNK)
                logits = self.model(mx.array(ids[a:b])[None], cache=cache)[0].astype(mx.float32)
                m = mx.array(mask[a + 1 : b + 1], dtype=mx.float32)
                tgt = mx.array(ids[a + 1 : b + 1])
                ll = mx.take_along_axis(logits, tgt[:, None], axis=-1)[:, 0] - mx.logsumexp(logits, axis=-1)
                s = -(ll * m).sum()
                mx.eval(s, [c.state for c in cache])
                total += float(s.item())
                count += int(sum(mask[a + 1 : b + 1]))
        return total, count

    # ------------------------------------------------------------- batch eval

    def run(self, rows: list[dict], *, checkpoints: list[str] | None = None, loss: bool = True,
            generate: bool = False, limit: int | None = None, max_tokens: int = 512,
            temperature: float = 0.0, top_p: float = 1.0, seed: int = 0, rewards=None,
            on_progress=None, on_row=None) -> dict:
        """Score rows under each checkpoint. Returns a report (see export).

        loss: mean NLL and perplexity on the rows' trained tokens.
        generate: answer each row's prompt and compare with its reference
          (exact, contains, token F1; `answer` fields use exact_answer), plus
          any reward functions.
        """
        self.cancel.clear()
        rows = list(rows)[: limit or None]
        cids = list(checkpoints or [self.current])
        fns = rewards or []
        report = {
            "kind": "pagedmoe-train-eval",
            "run": str(self.run_dir),
            "model": self.model_path,
            "method": self.method,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "settings": {"rows": len(rows), "loss": loss, "generate": generate, "max_tokens": max_tokens,
                         "temperature": temperature, "top_p": top_p, "seed": seed,
                         "rewards": [getattr(f, "__name__", "reward") for f in fns]},
            "load": {"mode": self.info.get("mode"), "seconds": round(self.load_s, 1)},
            "summary": [],
            "rows": [],
        }
        units = []
        if loss:
            for i, r in enumerate(rows):
                try:
                    units.append((i, self.fmt.example(r)))
                except Exception:
                    continue
        total = len(cids) * ((len(units) if loss else 0) + (len(rows) if generate else 0))
        done = 0
        try:
            for cid in cids:
                self.use(cid)
                ck = next((c for c in self.cks if c["id"] == self.current), {})
                t0 = time.perf_counter()
                recs = {i: {"checkpoint": self.current, "i": i} for i in range(len(rows))}
                nll = ntok = 0
                for i, ex in units:
                    if ex.embeds is not None:
                        continue
                    s, n = self.row_loss(ex)
                    nll, ntok = nll + s, ntok + n
                    recs[i]["loss"] = round(s / max(1, n), 5)
                    recs[i]["tokens"] = n
                    done += 1
                    if on_progress:
                        on_progress({"checkpoint": self.current, "stage": "loss", "done": done, "total": total})
                gen = {"n": 0, "tokens": 0, "seconds": 0.0, "scores": {}}
                if generate:
                    for i, r in enumerate(rows):
                        sp = split_row(r)
                        if sp is None:
                            continue
                        prompt, ref, tools = sp
                        g = self.generate(prompt, max_tokens=max_tokens, temperature=temperature,
                                          top_p=top_p, seed=seed + i, tools=tools)
                        if self.cancel.is_set():
                            raise Cancelled()
                        out = g.get("text_all", "")
                        sc = score(out, r, ref)
                        for f in fns:
                            try:
                                sc[getattr(f, "__name__", "reward")] = float(f(out, r))
                            except Exception:
                                sc[getattr(f, "__name__", "reward")] = None
                        rec = recs[i]
                        rec.update(prompt=_show(prompt), reference=_show(ref), output=out,
                                   gen_tokens=g.get("tokens"), tok_s=g.get("tok_s"), **sc)
                        gen["n"] += 1
                        gen["tokens"] += g.get("tokens") or 0
                        gen["seconds"] += g.get("seconds") or 0
                        for k, v in sc.items():
                            if v is not None:
                                gen["scores"].setdefault(k, []).append(v)
                        done += 1
                        if on_progress:
                            on_progress({"checkpoint": self.current, "stage": "generate", "done": done, "total": total})
                        if on_row:
                            on_row(rec)
                summ = {"checkpoint": self.current, "step": ck.get("step") if self.current != "base" else 0,
                        "seconds": round(time.perf_counter() - t0, 1)}
                if loss and ntok:
                    mean = nll / ntok
                    summ.update(loss=round(mean, 5), ppl=round(math.exp(min(mean, 50)), 3), loss_tokens=ntok)
                if gen["n"]:
                    summ["generated"] = gen["n"]
                    summ["gen_tok_s"] = round(gen["tokens"] / max(gen["seconds"], 1e-9), 2)
                    for k, vs in gen["scores"].items():
                        summ[k] = round(sum(vs) / len(vs), 4)
                report["summary"].append(summ)
                report["rows"].extend(r for r in recs.values() if len(r) > 2)
        except Cancelled:
            report["cancelled"] = True
        return report

    # ---------------------------------------------------------------- export

    @staticmethod
    def export(report: dict, path, what: str = "auto") -> Path:
        """.json: everything. .jsonl: one row per line. .csv: rows, or the
        summary table with what="summary" (or when there are no rows)."""
        return export(report, path, what)

    def save_report(self, report: dict) -> Path:
        """Keep a copy next to the run: RUN/evals/eval-<time>.json."""
        out = self.run_dir / "evals" / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
        return export(report, out)

    def close(self) -> None:
        self.model = None
        mx.clear_cache()


def _show(v) -> str:
    if v is None:
        return ""
    if isinstance(v, list):
        return "\n".join(f"{m.get('role', '?')}: {m.get('content') or ''}" for m in v if isinstance(m, dict))
    return str(v)


def export(report: dict, path, what: str = "auto") -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    suf = path.suffix.lower()
    if suf == ".jsonl":
        with open(path, "w") as f:
            for r in report.get("rows") or report.get("summary") or []:
                f.write(json.dumps(r, default=str) + "\n")
    elif suf == ".csv":
        table = report.get("summary") if what == "summary" or not report.get("rows") else report["rows"]
        cols: list[str] = []
        for r in table or []:
            cols += [k for k in r if k not in cols]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in table or []:
                w.writerow({k: r.get(k, "") for k in cols})
    else:
        path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    return path
