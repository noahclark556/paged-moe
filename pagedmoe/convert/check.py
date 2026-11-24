# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How close a converted model is to a reference, on real text.

Both models read the same tokens (teacher forced, no sampling). The reference
goes first and its top tokens per position are saved; then the converted
model runs and is scored against them: top-1 agreement, KL divergence over
the reference's top 100 tokens, and perplexity for both. Each model loads in
its own process through pagedmoe.load, so models bigger than RAM stream and
only one of them ever holds the GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

TOP_K = 100
SEQ = 512

SAMPLES = [
    "The river had been rising for three days before anyone in the valley took it seriously. "
    "By the time the school closed, the water was already over the low bridge, and the farmers "
    "on the east side were moving their animals to the ridge. Nobody remembered a spring like it.",
    "def merge_intervals(intervals):\n    intervals.sort(key=lambda x: x[0])\n    out = []\n"
    "    for start, end in intervals:\n        if out and start <= out[-1][1]:\n"
    "            out[-1][1] = max(out[-1][1], end)\n        else:\n            out.append([start, end])\n"
    "    return out\n\n\nprint(merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]))\n",
    "To solve 3x + 7 = 22, subtract 7 from both sides to get 3x = 15, then divide by 3, so x = 5. "
    "Check it: 3 times 5 is 15, plus 7 is 22. For a quadratic like x^2 - 5x + 6 = 0, look for two "
    "numbers that multiply to 6 and add to -5. Those are -2 and -3, so x = 2 or x = 3.",
    "User: My build fails with 'undefined reference to pthread_create'. What's wrong?\n"
    "Assistant: The linker can't find the pthread library. Add -pthread to both the compile and "
    "link flags, for example gcc main.c -pthread -o main. In CMake, use find_package(Threads) and "
    "link Threads::Threads to your target.",
    "SELECT c.name, COUNT(o.id) AS orders, SUM(o.total) AS revenue\nFROM customers c\n"
    "JOIN orders o ON o.customer_id = c.id\nWHERE o.created_at >= '2025-01-01'\n"
    "GROUP BY c.name\nHAVING COUNT(o.id) > 5\nORDER BY revenue DESC\nLIMIT 20;",
    "Photosynthesis turns light into chemical energy. In the light reactions, chlorophyll absorbs "
    "photons and splits water, releasing oxygen and producing ATP and NADPH. The Calvin cycle then "
    "uses that ATP and NADPH to fix carbon dioxide into sugars inside the stroma of the chloroplast.",
    "La ciudad estaba tranquila esa manana. Los cafes abrian despacio y el olor a pan recien hecho "
    "llenaba las calles estrechas del centro. Der Zug nach Berlin hatte wieder Verspaetung, aber "
    "niemand am Bahnsteig schien sich darueber zu wundern.",
    "Steps to replace a bike chain: 1. Shift to the smallest cog. 2. Use a chain tool to push out a "
    "pin, or open the quick link. 3. Size the new chain against the old one. 4. Thread it through "
    "the derailleurs. 5. Join it and check that every link moves freely.",
    "{\n  \"name\": \"sensor-7\",\n  \"location\": {\"lat\": 47.61, \"lon\": -122.33},\n"
    "  \"readings\": [21.4, 21.9, 22.3, 22.1],\n  \"status\": \"ok\",\n  \"battery\": 0.82\n}",
    "Rust's borrow checker enforces that you either have one mutable reference or any number of "
    "immutable references to a value at a time. fn longest<'a>(x: &'a str, y: &'a str) -> &'a str "
    "{ if x.len() > y.len() { x } else { y } } ties both inputs and the output to one lifetime.",
]


def _texts(prompts: str | None) -> list:
    if not prompts:
        return list(SAMPLES)
    p = Path(prompts).expanduser()
    raw = p.read_text()
    if p.suffix == ".jsonl":
        out = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                if "text" in row:
                    out.append(str(row["text"]))
                elif "messages" in row:
                    out.append("\n".join(f"{m.get('role')}: {m.get('content')}" for m in row["messages"]))
                elif "prompt" in row:
                    out.append(str(row["prompt"]) + str(row.get("completion", "")))
            else:
                out.append(str(row))
        return out
    return [t for t in raw.split("\n\n") if t.strip()] or [raw]


# ------------------------------------------------------------ child


def _say(**ev) -> None:
    print(json.dumps(ev), flush=True)


def _child(model_path: str, work: str, role: str, prompts: str | None, max_tokens: int) -> None:
    import numpy as np

    import mlx.core as mx

    from .. import load

    model, tok = load(model_path, verbose=False)
    work_p = Path(work)
    if role == "ref":
        ids: list = []
        for t in _texts(prompts):
            ids.extend(tok.encode(t))
            if len(ids) >= max_tokens + 1:
                break
        ids = ids[: max_tokens + 1]
        seqs = [ids[i: i + SEQ + 1] for i in range(0, max(1, len(ids) - 1), SEQ)]
        seqs = [s for s in seqs if len(s) > 1]
    else:
        seqs = json.loads((work_p / "seqs.json").read_text())
        ref = np.load(work_p / "ref.npz")
    rows_i, rows_lp, tgt, top1 = [], [], [], []
    kl_sum = agree = n = 0
    for k, s in enumerate(seqs):
        x = mx.array(s[:-1])[None]
        logits = model(x)[0].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = mx.array(s[1:])
        t_lp = mx.take_along_axis(lp, y[:, None], axis=-1)[:, 0]
        am = mx.argmax(lp, axis=-1)
        if role == "ref":
            idx = mx.argpartition(-lp, TOP_K - 1, axis=-1)[:, :TOP_K]
            top = mx.take_along_axis(lp, idx, axis=-1)
            mx.eval(idx, top, t_lp, am)
            rows_i.append(np.array(idx, dtype=np.int32))
            rows_lp.append(np.array(top, dtype=np.float32))
        else:
            a, b = ref["off"][k], ref["off"][k + 1]
            r_idx = mx.array(ref["idx"][a:b])
            r_lp = mx.array(ref["lp"][a:b])
            q = mx.take_along_axis(lp, r_idx, axis=-1)
            kl = mx.sum(mx.exp(r_lp) * (r_lp - q))
            ag = mx.sum(am == mx.array(ref["top1"][a:b]))
            mx.eval(kl, ag, t_lp, am)
            kl_sum += kl.item()
            agree += ag.item()
        tgt.append(np.array(t_lp, dtype=np.float32))
        top1.append(np.array(am, dtype=np.int32))
        n += len(s) - 1
        _say(kind="check", role=role, seq=k + 1, seqs=len(seqs), tokens=n)
    tgt_all = np.concatenate(tgt) if tgt else np.zeros(0, np.float32)
    ppl = float(np.exp(-tgt_all.mean())) if len(tgt_all) else float("nan")
    if role == "ref":
        off = np.cumsum([0] + [len(r) for r in rows_i])
        np.savez(work_p / "ref.npz", idx=np.concatenate(rows_i), lp=np.concatenate(rows_lp),
                 top1=np.concatenate(top1), off=off)
        (work_p / "seqs.json").write_text(json.dumps(seqs))
        _say(kind="result", ppl=ppl, tokens=n)
    else:
        _say(kind="result", ppl=ppl, tokens=n, kl=kl_sum / max(1, n), top1=agree / max(1, n))


# ------------------------------------------------------------ parent


def _run_child(args: list, on_event) -> dict:
    cmd = [sys.executable, "-m", "pagedmoe.convert.check", *args]
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = None
    with tempfile.TemporaryFile(mode="w+") as errf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, text=True, env=env)
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("kind") == "result":
                    result = ev
                elif on_event:
                    on_event(ev)
            code = proc.wait()
        except BaseException:
            proc.kill()  # never leave a model on the GPU behind
            proc.wait()
            raise
        errf.seek(0)
        err = errf.read()
    if code != 0 or result is None:
        tail = "\n".join(err.strip().splitlines()[-5:])
        raise RuntimeError(f"scoring {args[2]} failed: {tail}")
    return result


def check(model: str, reference: str, *, prompts: str | None = None, max_tokens: int = 2048,
          on_event=None) -> dict:
    """Score `model` against `reference`. Returns top1, kl, ppl, ppl_ref, tokens."""
    with tempfile.TemporaryDirectory(prefix="pagedmoe-check-") as work:
        common = ["--work", work, "--tokens", str(int(max_tokens))]
        if prompts:
            common += ["--prompts", str(prompts)]
        if on_event:
            on_event({"kind": "phase", "phase": "reference"})
        ref = _run_child(["--role", "ref", str(reference), *common], on_event)
        if on_event:
            on_event({"kind": "phase", "phase": "converted"})
        got = _run_child(["--role", "test", str(model), *common], on_event)
    res = {"model": str(model), "reference": str(reference), "tokens": got["tokens"],
           "top1": got["top1"], "kl": got["kl"], "ppl": got["ppl"], "ppl_ref": ref["ppl"],
           "kl_over": f"reference top {TOP_K}", "prompts": prompts or "built-in"}
    rep_p = Path(model).expanduser() / "pagedmoe_conversion.json"
    if rep_p.exists():
        try:
            rep = json.loads(rep_p.read_text())
            rep["check"] = res
            rep_p.write_text(json.dumps(rep, indent=2) + "\n")
        except (OSError, ValueError):
            pass
    return res


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("ref", "test"), required=True)
    ap.add_argument("model")
    ap.add_argument("--work", required=True)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--prompts", default=None)
    a = ap.parse_args()
    _child(a.model, a.work, a.role, a.prompts, a.tokens)


if __name__ == "__main__":
    _main()
