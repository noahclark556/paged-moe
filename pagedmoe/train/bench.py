# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Training benchmark: speed, memory and checkpoint cost, model by model.

Each case is one model and one setup (method, sequence length, LoRA rank and
targets). It trains a few real steps on a fixed synthetic corpus with the
same planner a normal run uses, then records:

  speed        tokens/s, seconds per step (mean, median, spread), time per
               1M tokens, time per epoch of 1k rows, the probe's estimate
  memory       peak Metal per step, peak process footprint, budget, the
               compressor and swap before and at peak, worst pressure level
  setup        data, model load and probe time, the first (cold) step
  streaming    GB read per step, passes over the experts, cache hit rate
  checkpoint   save time, adapter and optimizer size, adapter load time
  the plan     resident or streamed, executor, rows per step, micro-batch

Cases run one at a time, each in its own process, so one model never sees
another's memory and only one MLX job holds the GPU. Results are JSON files
in <train dir>/bench/results; `table` turns them into Markdown, CSV or JSON.

    python -m pagedmoe.train bench status
    python -m pagedmoe.train bench run qwen3-235b flash --seq 2048,8192
    python -m pagedmoe.train bench run ~/mlx-models/MyModel --steps 6
    python -m pagedmoe.train bench table --format md > TRAIN_BENCHMARK.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import plan as P

VERSION = 1
EVENT = "@@bench "  # child -> parent event lines on stdout

# Models the published table is built from. A checkpoint is recognized by its
# architecture (`arch`: model_type, layers, hidden size, routed experts, bits)
# whatever its folder is called; `dirs` are known folder names as a fallback.
# `hf` is where to get the checkpoint if it is missing. Any other checkpoint
# in the model folders can be benched too, under its folder name.
LADDER: list[dict] = [
    {"id": "pagedmoe-27b", "name": "PagedMoE-27B-A3B", "dirs": ["PagedMoE-27B-A3B-4bit"], "hf": None, "approx_gb": 15,
     "arch": ("qwen3_5_moe", 40, 2048, 192, 4)},
    {"id": "pagedmoe-35b", "name": "PagedMoE-35B-A3B", "dirs": ["PagedMoE-35B-A3B-4bit"], "hf": None, "approx_gb": 19,
     "arch": ("qwen3_5_moe", 40, 2048, 256, 4)},
    {"id": "coder-next", "name": "Qwen3-Coder-Next", "dirs": ["coder-next-6bit-think", "Qwen3-Coder-Next-6bit"],
     "hf": "mlx-community/Qwen3-Coder-Next-6bit", "approx_gb": 60, "arch": ("qwen3_next", 48, 2048, 512, 6)},
    {"id": "qwen3-235b", "name": "Qwen3-235B-A22B", "dirs": ["qwen3-235-4bit", "Qwen3-235B-A22B-4bit"],
     "hf": "mlx-community/Qwen3-235B-A22B-4bit", "approx_gb": 123, "arch": ("qwen3_moe", 94, 4096, 128, 4)},
    {"id": "flash", "name": "Qwen3.8-Flash-Next", "dirs": ["qwen38-coder-flash", "Qwen3.8-Flash-Next-MLX-6bit"],
     "hf": None, "approx_gb": 138, "arch": ("qwen4_exp", 48, 2560, 512, 6)},
    {"id": "glm-47", "name": "GLM-4.7", "dirs": ["glm-47-4bit", "GLM-4.7-4bit"],
     "hf": "mlx-community/GLM-4.7-4bit", "approx_gb": 199, "arch": ("glm4_moe", 92, 5120, 160, 4)},
]

SETUP = {"method": "sft", "seq": 2048, "rank": 16, "target": "all-linear", "batch": None}
METHODS = ("sft", "cpt", "orpo", "dpo", "simpo", "cpo", "kto", "grpo")


def bench_dir() -> Path:
    return P.train_dir() / "bench"


def results_dir() -> Path:
    return bench_dir() / "results"


# ------------------------------------------------------------------ models


def model_dirs() -> list[Path]:
    """PAGED_MOE_BENCH_DIR (one folder or several joined with ':'), then
    ~/mlx-models and ~/models."""
    raws = [r for r in (os.environ.get("PAGED_MOE_BENCH_DIR") or "").split(os.pathsep) if r]
    out = []
    for raw in raws + ["~/mlx-models", "~/models"]:
        p = Path(raw).expanduser()
        if p.is_dir() and p not in out:
            out.append(p)
    return out


def scan_models() -> list[Path]:
    """Every checkpoint in the model folders, plus the HF cache when
    PAGED_MOE_BENCH_HF=1."""
    out = scan_dirs(model_dirs())
    if os.environ.get("PAGED_MOE_BENCH_HF") == "1":
        hub = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
        for repo in sorted(hub.glob("models--*")) if hub.is_dir() else []:
            snaps = sorted((repo / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
            hit = next((s for s in snaps if _ready(s)), None)
            if hit and hit not in out:
                out.append(hit)
    return out


def scan_dirs(roots: list[Path]) -> list[Path]:
    """Checkpoints (config.json plus safetensors): each folder itself and up
    to two levels down."""
    out: list[Path] = []
    for root in roots:
        level = [root]
        for _ in range(3):
            nxt = []
            for d in level:
                if _ready(d):
                    if d not in out:
                        out.append(d)
                    continue
                try:
                    nxt += [c for c in sorted(d.iterdir()) if c.is_dir() and not c.name.startswith(".")]
                except OSError:
                    pass
            level = nxt
    return out


def display_name(path: Path) -> str:
    """Folder name, or the repo name for a Hugging Face cache snapshot."""
    if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        return path.parent.parent.name.split("--")[-1]
    return path.name


def arch(path: Path) -> tuple | None:
    try:
        c = json.loads((path / "config.json").read_text())
    except Exception:
        return None
    t = c.get("text_config") or {}

    def g(k):
        return c.get(k, t.get(k))

    q = c.get("quantization") or c.get("quantization_config") or {}
    experts = g("num_experts") or g("n_routed_experts") or g("num_local_experts")
    return (g("model_type"), g("num_hidden_layers"), g("hidden_size"), experts, q.get("bits"))


def ladder_entry(path: Path) -> dict | None:
    """The ladder model this checkpoint is, by architecture, else by folder name."""
    a = arch(path)
    for e in LADDER:
        if a and tuple(e.get("arch") or ()) == a:
            return e
    low = path.name.lower()
    return next((e for e in LADDER if low in (d.lower() for d in e["dirs"])), None)


def _hf_cached(repo: str) -> Path | None:
    root = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    snaps = root / f"models--{repo.replace('/', '--')}" / "snapshots"
    for s in sorted(snaps.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True) if snaps.is_dir() else []:
        if (s / "config.json").exists() and any(s.glob("*.safetensors")):
            return s
    return None


def _ready(p: Path) -> bool:
    return p.is_dir() and (p / "config.json").exists() and any(p.glob("*.safetensors"))


def _other(p: Path) -> dict:
    n = display_name(p)
    return {"id": n, "name": n, "dirs": [p.name], "hf": None}


def find_model(spec: str, found: list[Path] | None = None) -> tuple[dict, Path | None]:
    """A ladder id, a folder name in the model folders, or a path. A path or
    folder that is a ladder model under another name still counts as it."""
    entry = next((e for e in LADDER if e["id"] == spec), None)
    if entry is None:
        p = Path(spec).expanduser()
        if _ready(p):
            return ladder_entry(p) or _other(p), p
        entry = {"id": spec, "name": spec, "dirs": [spec], "hf": spec if spec.count("/") == 1 else None}
    found = scan_models() if found is None else found
    if entry in LADDER:
        hit = next((p for p in found if ladder_entry(p) is entry), None)
        if hit:
            return entry, hit
    else:
        hit = next((p for p in found if spec.lower() in (p.name.lower(), display_name(p).lower())), None)
        if hit:
            return ladder_entry(hit) or _other(hit), hit
    if entry.get("hf"):
        hit = _hf_cached(entry["hf"])
        if hit:
            return entry, hit
    return entry, None


def gib(gb) -> float | None:
    """Decimal GB (what the trainer logs) to GiB, the unit a Mac's RAM is sold
    in: a "48 GB" Mac has 48 GiB, about 51.5 decimal GB."""
    return None if gb is None else round(float(gb) * 1e9 / 2**30, 1)


def _gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.glob("*.safetensors")) / 1e9


def status(folder: str | None = None) -> list[dict]:
    """One row per checkpoint found (id = path), flagged `in_folder` when it is
    in `folder`, then the ladder models not found here. Ladder checkpoints keep
    the ladder id as `case_id` whatever the folder is called."""
    _ram, budget, _ws = P.machine_budget()
    budget_now, _ = P.live_budget(budget)
    mine = scan_dirs([Path(folder).expanduser()]) if folder and Path(folder).expanduser().is_dir() else []
    found = mine + [p for p in scan_models() if p not in mine]
    out, seen = [], set()
    for p in found:
        e = ladder_entry(p)
        rec = {"id": str(p), "case_id": e["id"] if e else display_name(p), "name": display_name(p),
               "hf": e.get("hf") if e else None, "ladder": e is not None, "in_folder": p in mine,
               "path": str(p), "found": True, "size_gb": round(_gb(p), 1), "mode": None}
        try:
            f = P.model_facts(str(p))
            rec["mode"] = "resident" if f.total_bytes + P.room_floor(budget_now) <= budget_now else "streamed"
        except Exception:
            pass
        if e:
            seen.add(e["id"])
        out.append(rec)
    for e in LADDER:
        if e["id"] not in seen:
            out.append({"id": e["id"], "case_id": e["id"], "name": e["name"], "hf": e.get("hf"),
                        "approx_gb": e.get("approx_gb"), "ladder": True, "in_folder": False,
                        "path": None, "found": False})
    return out


def done_cases(results: list[dict] | None = None) -> set[str]:
    return {r.get("case_key") for r in (results if results is not None else load_results())
            if r.get("status") == "ok" and r.get("machine", {}).get("key") == machine_key()}


# ----------------------------------------------------------------- machine


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True,
                              timeout=3).stdout.strip() or None
    except Exception:
        return None


def machine_key() -> str:
    chip = _sysctl("machdep.cpu.brand_string") or platform.machine()
    ram = int(_sysctl("hw.memsize") or 0)
    return f"{chip} {round(ram / 2**30)}GB"


def machine() -> dict:
    """What the numbers were measured on."""
    ram = int(_sysctl("hw.memsize") or 0)
    out = {
        "key": machine_key(),
        "chip": _sysctl("machdep.cpu.brand_string") or platform.machine(),
        "hw_model": _sysctl("hw.model"),
        "ram_gb": round(ram / 2**30),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
    }
    try:
        from ..machine import get_provider

        hw = get_provider().hardware
        out.update(gpu_cores=hw.gpu_cores, perf_cores=hw.perf_cores, efficiency_cores=hw.efficiency_cores,
                   disk_gb_s=hw.disk_gb_s, mem_bw_gb_s=hw.mem_bw_gb_s)
    except Exception:
        pass
    try:
        import mlx.core as mx
        import mlx_lm

        from .. import __version__

        out.update(mlx=mx.__version__, mlx_lm=mlx_lm.__version__, pagedmoe=__version__)
    except Exception:
        pass
    try:
        _, budget, ws = P.machine_budget()
        out.update(budget_gb=round(budget / 1e9, 1), working_set_gb=round(ws / 1e9, 1) if ws else None)
    except Exception:
        pass
    return out


def gpu_users() -> list[str]:
    """Other processes that look like they hold the GPU (serving, training)."""
    try:
        text = subprocess.run(["/bin/ps", "-Ao", "pid=,ppid=,command="], capture_output=True, text=True,
                              timeout=4).stdout
    except Exception:
        return []
    pat = re.compile(r"pagedmoe\.server|pagedmoe-server|mlx_lm\.(server|generate|lora|chat)|"
                     r"pagedmoe\.train(?!\.bench)|trainer_bridge\.py (train|serve)")
    me = {os.getpid(), os.getppid()}
    hits = []
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)\s+(.*)", line)
        if m and int(m.group(1)) not in me and int(m.group(2)) not in me and pat.search(m.group(3)):
            hits.append(f"{m.group(1)}  {m.group(3)[:160]}")
    return hits


# ------------------------------------------------------------------ corpus

_WORDS = (
    "the of and to in is that for it as was with be by on not he this are or his from at which but have "
    "an had they you were their one all we can her has there been if more when will would who so no "
    "time people year way day man thing woman life child world school state family student group country "
    "problem hand part place case week company system program question work government number night point "
    "home water room mother area money story fact month lot right study book eye job word business issue "
    "side kind head house service friend father power hour game line end member law car city community name "
    "president team minute idea kid body information back parent face others level office door health person "
    "art war history party result change morning reason research girl guy moment air teacher force education "
    "function return value list map index buffer layer cache memory stream token batch model weight expert "
    "gradient update step loss shape array tensor kernel device thread queue read write file path config"
).split()


def _text(rng: random.Random, words: int) -> str:
    out, n = [], 0
    while n < words:
        k = rng.randint(8, 18)
        s = " ".join(rng.choice(_WORDS) for _ in range(k))
        out.append(s[0].upper() + s[1:] + ".")
        n += k
    return " ".join(out)


def corpus(method: str, seq: int, rows: int = 256, seed: int = 0) -> list[dict]:
    """Fixed synthetic rows that fill `seq` tokens after truncation.

    Same seed, same rows, on every machine. Rows run a little past `seq` on
    purpose so every one is cut to exactly the sequence length.
    """
    rng = random.Random(1000 + seed)
    words = int(seq * 1.1)
    q = 48
    out = []
    for i in range(rows):
        if method == "cpt":
            out.append({"text": _text(rng, words)})
        elif method in ("orpo", "dpo", "simpo", "cpo"):
            out.append({"prompt": _text(rng, q), "chosen": _text(rng, words), "rejected": _text(rng, words)})
        elif method == "kto":
            out.append({"prompt": _text(rng, q), "completion": _text(rng, words), "label": i % 2 == 0})
        elif method == "grpo":
            a, b = rng.randint(10, 999), rng.randint(10, 999)
            out.append({"prompt": f"What is {a} + {b}? Put the answer in \\boxed{{}}.", "answer": str(a + b)})
        else:
            out.append({"messages": [{"role": "user", "content": _text(rng, q)},
                                     {"role": "assistant", "content": _text(rng, words)}]})
    return out


# -------------------------------------------------------------- one case


class _Sampler(threading.Thread):
    """Process footprint, Metal, compressor, swap and pressure, twice a second."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop = threading.Event()
        from .. import mempressure as M

        self.M = M
        self.start_comp = M.compressor_gb()
        self.start_swap = _swap_gb()
        self.peak = {"footprint": 0.0, "metal": 0.0, "compressor": self.start_comp, "swap": self.start_swap,
                     "pressure": M.level()}
        # the same peaks per phase (setup, load, probe, train, checkpoint), so
        # a spike can be placed
        self.phase = "setup"
        self.by_phase: dict[str, dict] = {}

    def run(self):
        import mlx.core as mx

        while not self.stop.wait(0.5):
            try:
                now = {"footprint": self.M.footprint_gb(), "metal": mx.get_active_memory() / 1e9,
                       "compressor": self.M.compressor_gb(), "pressure": self.M.level()}
                if int(time.time() * 2) % 10 == 0:
                    now["swap"] = _swap_gb()
                ph = self.by_phase.setdefault(self.phase, {})
                for k, v in now.items():
                    self.peak[k] = max(self.peak[k], v)
                    ph[k] = max(ph.get(k, v), v)
            except Exception:
                pass

    def phases(self) -> dict:
        out = {}
        for name, pk in self.by_phase.items():
            out[name] = {"metal_gb": round(pk.get("metal", 0.0), 2), "process_gb": round(pk.get("footprint", 0.0), 2),
                         "compressor_gb": round(pk.get("compressor", 0.0), 2),
                         "swap_gb": round(pk["swap"], 2) if "swap" in pk else None,
                         "pressure": self.M.name(pk.get("pressure", 0))}
        return out


def _swap_gb() -> float:
    m = re.search(r"used = ([\d.]+)([MG])", _sysctl("vm.swapusage") or "")
    if not m:
        return 0.0
    return float(m.group(1)) * (1.0 if m.group(2) == "G" else 1 / 1024)


def _emit(**ev) -> None:
    print(EVENT + json.dumps(ev, default=str), flush=True)


def case_key(model_id: str, setup: dict) -> str:
    s = {**SETUP, **setup}
    b = f"-b{s['batch']}" if s.get("batch") else ""
    return f"{model_id}|{s['method']}|{s['seq']}|r{s['rank']}|{s['target']}{b}"


def _quant(path: Path) -> tuple[int | None, str]:
    try:
        cfg = json.loads((path / "config.json").read_text())
    except Exception:
        return None, "?"
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    bits = q.get("bits")
    return bits, (f"{bits}-bit" if bits else str(cfg.get("torch_dtype") or "bf16"))


def run_case(case: dict) -> dict:
    """Train one case in this process and return its result."""
    import mlx.core as mx

    from . import config as C
    from . import lora as train_lora
    from .stats import Callback
    from .trainer import TRAINERS

    path = Path(case["path"])
    setup = {**SETUP, **(case.get("setup") or {})}
    steps, warm = int(case.get("steps", 4)), int(case.get("warmup", 1))
    out_dir = bench_dir() / "runs" / re.sub(r"[^\w.-]+", "_", case["key"])
    bits, quant = _quant(path)
    res: dict = {
        "kind": "pagedmoe-train-bench", "version": VERSION, "case_key": case["key"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": machine(),
        "model": {"id": case["model_id"], "name": case.get("name") or path.name, "path": str(path),
                  "quant": quant, "bits": bits, "size_gb": round(_gb(path), 1)},
        "setup": dict(setup, steps=steps, warmup=warm, probe=case.get("probe", "always"),
                      adapter=("QLoRA" if bits else "LoRA") + f" r{setup['rank']} {setup['target']}"),
        "status": "error",
    }
    sampler = _Sampler()
    sampler.start()
    logs: list[dict] = []
    phases: dict[str, float] = {}
    t0 = time.perf_counter()

    class CB(Callback):
        def on_progress(self, trainer, info):
            ph = info.get("phase")
            if ph and ph not in phases:
                phases[ph] = time.perf_counter()
            if ph:
                sampler.phase = "train" if ph == "step" else ph
            _emit(t="progress", **info)

        def on_step_end(self, trainer, step, lg):
            sampler.phase = "train"
            rec = {k: v for k, v in lg.items() if k not in ("experts", "timing")}
            rec["step"], rec["time"] = step, time.perf_counter() - t0
            logs.append(rec)
            _emit(t="step", step=step, total=trainer.total_steps, loss=lg.get("loss"), tok_s=lg.get("tok_s"),
                  step_s=lg.get("step_s"), peak_gb=lg.get("peak_gb"))

    data = {
        "method": setup["method"], "model": str(path), "max_seq_len": int(setup["seq"]),
        "max_steps": steps + warm, "probe": case.get("probe", "always"), "seed": 0,
        "save_steps": 0, "save_minutes": 0, "keep_checkpoints": 1, "output_dir": str(out_dir),
        "lora": {"rank": int(setup["rank"]), "alpha": 2.0 * int(setup["rank"]), "target": setup["target"]},
        "log_steps": 1,
    }
    if setup.get("batch"):
        data["batch_size"] = int(setup["batch"])
    if setup["method"] == "grpo":
        data["max_completion_tokens"] = min(512, int(setup["seq"]) // 2)
    try:
        cfg = C.from_dict(data)
        rows = corpus(setup["method"], int(setup["seq"]), int(case.get("rows", 256)))
        kw: dict = {"train_data": rows, "callbacks": [CB()]}
        if setup["method"] == "grpo":
            from . import rewards as R

            kw["reward_funcs"] = [R.exact_answer()]
        tr = TRAINERS[setup["method"]](cfg, **kw)
        t_ready = time.perf_counter()
        info = tr.plan_info()
        est = tr.estimate()
        save = {}
        real_save = tr.save_checkpoint

        def timed_save(final: bool = False):
            a = time.perf_counter()
            ck = real_save(final=final)
            save["s"] = time.perf_counter() - a
            save["path"] = ck
            return ck

        tr.save_checkpoint = timed_save
        tr.train()
        t_end = time.perf_counter()
        sampler.phase = "checkpoint"

        ck = save.get("path")
        ck_info = {}
        if ck:
            ad, opt = Path(ck) / "adapters.safetensors", Path(ck) / "optimizer.safetensors"
            a = time.perf_counter()
            train_lora.load_weights(tr.model, Path(ck))
            ck_info = {"save_s": round(save["s"], 2), "load_s": round(time.perf_counter() - a, 2),
                       "adapter_mb": round(ad.stat().st_size / 1e6, 1) if ad.exists() else None,
                       "optimizer_mb": round(opt.stat().st_size / 1e6, 1) if opt.exists() else None}

        timed = logs[warm:] or logs[-1:]
        # A memory refit or the guard changes the setup mid-case; time only
        # the steps that ran on the final setup when there are enough of them.
        changed = [i for i, r in enumerate(logs) if r.get("memory_guard") or r.get("memory_refit")]
        settled = None
        if changed and len(logs) - (changed[-1] + 1) >= 2:
            settled = changed[-1] + 1
            timed = logs[max(warm, settled):]
        ss = [r["step_s"] for r in timed]
        toks = sum(r["tokens"] for r in timed)
        tsec = sum(ss)
        tok_s = toks / max(tsec, 1e-9)
        p = tr.plan
        per_step = p.per_step
        step_mean = statistics.mean(ss)
        res["plan"] = {"mode": p.mode, "executor": p.executor, "rows_per_step": per_step,
                       "micro_tokens": p.micro_tokens, "cache_gb": p.cache_gb, "budget_gb": p.budget_gb,
                       "step_tokens": round(toks / len(timed)), "reasons": p.reasons}
        lr = info.get("lora") or {}
        res["lora"] = {"trainable": lr.get("trainable"), "total": lr.get("total"),
                       "pct": round(100 * lr["trainable"] / lr["total"], 4) if lr.get("total") else None}
        load_at = phases.get("load", t0)
        probe_at = phases.get("probe")
        res["time"] = {"data_s": round(load_at - t0, 1), "load_s": round(tr.load_s, 1),
                       "probe_s": round(t_ready - probe_at, 1) if probe_at else 0.0,
                       "first_step_s": round(logs[0]["step_s"], 2) if logs else None,
                       "total_s": round(t_end - t0, 1)}
        res["speed"] = {
            "tok_s": round(tok_s, 1),
            "trained_tok_s": round(sum(r["loss_tokens"] for r in timed) / max(tsec, 1e-9), 1),
            "step_s": round(step_mean, 2), "step_s_median": round(statistics.median(ss), 2),
            "step_s_min": round(min(ss), 2), "step_s_max": round(max(ss), 2),
            "step_s_std": round(statistics.pstdev(ss), 2) if len(ss) > 1 else 0.0,
            "rows_per_s": round(per_step / max(step_mean, 1e-9), 3),
            "h_per_1m_tokens": round(1e6 / max(tok_s, 1e-9) / 3600, 2),
            "epoch_1k_rows_s": round(math.ceil(1000 / per_step) * step_mean, 1),
            "timed_steps": len(timed),
            "timed_from_step": timed[0].get("step") if timed else None,
            "settled_after_change": settled is not None and settled > warm,
            "mixed_setup": bool(changed) and settled is None and changed[-1] >= warm,
        }
        if est:
            res["speed"]["est_tok_s"] = round(est["tok_s"], 1)
            res["speed"]["est_error_pct"] = round(100 * (tok_s / max(est["tok_s"], 1e-9) - 1), 1)
        pk = sampler.peak
        res["memory"] = {
            "peak_step_gb": round(max(r["peak_gb"] for r in logs), 2),
            "peak_metal_gb": round(max(pk["metal"], max(r["peak_gb"] for r in logs)), 2),
            "peak_process_gb": round(pk["footprint"], 2),
            "budget_gb": p.budget_gb,
            "peak_metal_gib": gib(max(pk["metal"], max(r["peak_gb"] for r in logs))),
            "peak_process_gib": gib(pk["footprint"]),
            "budget_gib": gib(p.budget_gb),
            "compressor_start_gb": round(sampler.start_comp, 2), "compressor_peak_gb": round(pk["compressor"], 2),
            "swap_start_gb": round(sampler.start_swap, 2), "swap_peak_gb": round(pk["swap"], 2),
            "pressure_max": sampler.M.name(pk["pressure"]),
            "guard": [r["memory_guard"] for r in logs if r.get("memory_guard")],
            "refit": [r["memory_refit"] for r in logs if r.get("memory_refit")],
            "by_phase": sampler.phases(),
        }
        if p.mode == "streamed":
            rd = [r.get("read_gb", 0) for r in timed]
            res["io"] = {"read_gb_per_step": round(statistics.mean(rd), 2),
                         "passes": round(statistics.mean(r.get("passes", 0) for r in timed), 2),
                         "cache_hit": round(statistics.mean(r.get("cache_hit", 0) for r in timed), 3),
                         "disk_wait_pct": round(100 * sum(r.get("disk_wait_s", 0) for r in timed) / max(tsec, 1e-9), 1),
                         "read_gb_s": round(sum(rd) / max(tsec, 1e-9), 2)}
        res["checkpoint"] = ck_info
        res["loss"] = {"first": round(logs[0]["loss"], 4), "last": round(logs[-1]["loss"], 4)} if logs else {}
        res["steps"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in logs]
        res["status"] = "stopped" if tr._stop else "ok"
    except BaseException as e:  # noqa: BLE001 - a failed case is a result too
        res["error"] = f"{type(e).__name__}: {e}"[:600]
        res["steps"] = logs
        if isinstance(e, KeyboardInterrupt):
            res["status"] = "stopped"
    finally:
        sampler.stop.set()
        if not case.get("keep"):
            import shutil

            shutil.rmtree(out_dir, ignore_errors=True)
        mx.clear_cache()
    return res


# --------------------------------------------------------------- the run


def _cool_down(max_s: float = 90.0, on_line=print) -> None:
    """Let the last case's memory settle before the next one starts."""
    try:
        from .. import mempressure as M
    except Exception:
        time.sleep(5)
        return
    t = time.time()
    time.sleep(5)
    while M.level() >= M.WARN and time.time() - t < max_s:
        on_line("[bench] waiting for memory pressure to settle ...")
        time.sleep(10)


def run(models: list[str], setups: list[dict] | None = None, *, steps: int = 4, warmup: int = 1,
        probe: str = "always", rows: int = 256, keep: bool = False, skip_done: bool = False,
        on_event=None, on_line=print, should_stop=None) -> list[dict]:
    """Run every model x setup, one process each. Returns the results."""
    setups = setups or [dict(SETUP)]
    cases = []
    for spec in models:
        entry, path = find_model(spec)
        for s in setups:
            cases.append({"model_id": entry["id"], "name": entry["name"], "path": str(path) if path else None,
                          "hf": entry.get("hf"), "setup": {**SETUP, **s},
                          "key": case_key(entry["id"], s), "steps": steps, "warmup": warmup,
                          "probe": probe, "rows": rows, "keep": keep})
    done = done_cases() if skip_done else set()
    emit = on_event or (lambda ev: None)
    results = []
    child: dict = {}

    def stop_child(sig=signal.SIGTERM):
        # SIGTERM: the trainer stops after the step it is on. SIGINT: now.
        p = child.get("p")
        if p and p.poll() is None and child.get("sent") != sig:
            child["sent"] = sig
            p.send_signal(sig)

    for i, case in enumerate(cases):
        if should_stop and should_stop():
            break
        emit({"t": "case", "i": i, "n": len(cases), "key": case["key"], "name": case["name"],
              "setup": case["setup"]})
        if not case["path"]:
            hint = f" (hf download {case['hf']} --local-dir ~/mlx-models/{case['hf'].split('/')[-1]})" if case.get("hf") else ""
            on_line(f"[bench] {case['name']}: not found in {', '.join(map(str, model_dirs()))}{hint}")
            emit({"t": "result", "i": i, "result": {"case_key": case["key"], "status": "missing",
                                                     "model": {"name": case["name"]}, "setup": case["setup"]}})
            continue
        if case["key"] in done:
            on_line(f"[bench] {case['key']}: already measured on this Mac, skipping")
            continue
        busy = gpu_users()
        if busy:
            on_line("[bench] warning: another MLX job looks like it is running; numbers will be off:\n  "
                    + "\n  ".join(busy))
        on_line(f"[bench] case {i + 1}/{len(cases)}: {case['key']}  ({case['path']})")
        slug = re.sub(r"[^\w.-]+", "_", case["key"])
        out = results_dir() / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "pagedmoe.train.bench", "one", json.dumps(dict(case, out=str(out)))]
        env = dict(os.environ, PYTHONUNBUFFERED="1", PAGED_MOE_TRAIN_HINT="0")
        env.pop("PTUI_EVENTS_FD", None)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        child.clear()
        child["p"] = p
        watch = threading.Thread(target=_watch_stop, args=(p, should_stop, stop_child), daemon=True)
        watch.start()
        try:
            for line in p.stdout:
                line = line.rstrip("\n")
                if line.startswith(EVENT):
                    try:
                        emit(dict(json.loads(line[len(EVENT):]), i=i))
                    except ValueError:
                        pass
                else:
                    on_line(line)
            p.wait()
        except KeyboardInterrupt:
            stop_child(signal.SIGINT)
            p.wait()
            raise
        stopped = bool(should_stop and should_stop())
        res = json.loads(out.read_text()) if out.exists() else {
            "case_key": case["key"], "status": "stopped" if stopped else "error", "model": {"name": case["name"]},
            "setup": case["setup"], "machine": machine(),
            "error": "stopped during setup" if stopped else f"bench process exited with {p.returncode} before writing a result"}
        if not out.exists():
            out.write_text(json.dumps(res, indent=2) + "\n")
        results.append(res)
        emit({"t": "result", "i": i, "result": res, "file": str(out)})
        on_line("[bench] " + summary_line(res))
        if i + 1 < len(cases):
            _cool_down(on_line=on_line)
    emit({"t": "done", "n": len(results)})
    return results


def _watch_stop(p, should_stop, stop_child) -> None:
    while should_stop and p.poll() is None:
        if should_stop():
            stop_child()
            return
        time.sleep(0.5)


def summary_line(r: dict) -> str:
    name = f"{r.get('model', {}).get('name', '?')} {r.get('setup', {}).get('method', '')} seq {r.get('setup', {}).get('seq', '')}"
    if r.get("status") != "ok":
        return f"{name}: {r.get('status')} {r.get('error', '')}".strip()
    s, m, pl = r["speed"], r["memory"], r["plan"]
    line = (f"{name}: {s['tok_s']:,.0f} tok/s, {s['step_s']:.1f} s/step ({pl['rows_per_step']} rows), "
            f"peak {gib(m['peak_metal_gb'])} GiB Metal / {gib(m['peak_process_gb'])} GiB process, {pl['mode']}")
    if s.get("settled_after_change"):
        line += f" (memory setup changed mid-case; timed from step {s['timed_from_step']})"
    elif s.get("mixed_setup"):
        line += " (memory setup changed during the timed steps; rerun with more --steps)"
    worst = max((m.get("by_phase") or {}).items(), key=lambda kv: kv[1].get("metal_gb", 0), default=None)
    if worst and worst[1].get("metal_gb", 0) > m.get("budget_gb", 0):
        line += f"; over budget during {worst[0]} ({gib(worst[1]['metal_gb'])} GiB Metal)"
    return line


# ---------------------------------------------------------------- tables


def load_results(where: Path | None = None) -> list[dict]:
    out = []
    for f in sorted((where or results_dir()).glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        if r.get("kind") == "pagedmoe-train-bench" or r.get("case_key"):
            r["_file"] = str(f)
            out.append(r)
    return out


def latest(results: list[dict]) -> list[dict]:
    """Newest result per machine and case."""
    best: dict = {}
    for r in sorted(results, key=lambda r: r.get("created", "")):
        best[(r.get("machine", {}).get("key"), r.get("case_key"))] = r
    return list(best.values())


COLUMNS = [
    ("Model", lambda r: r["model"]["name"]),
    ("Quant", lambda r: r["model"].get("quant")),
    ("Size GB", lambda r: r["model"].get("size_gb")),
    ("Mode", lambda r: r["plan"]["mode"]),
    ("Method", lambda r: r["setup"]["method"].upper()),
    ("Adapter", lambda r: r["setup"].get("adapter")),
    ("Seq", lambda r: r["setup"]["seq"]),
    ("Rows/step", lambda r: r["plan"]["rows_per_step"]),
    ("tok/s", lambda r: round(r["speed"]["tok_s"])),
    ("s/step", lambda r: r["speed"]["step_s"]),
    ("h per 1M tok", lambda r: r["speed"]["h_per_1m_tokens"]),
    ("Epoch (1k rows)", lambda r: P.fmt_time(r["speed"]["epoch_1k_rows_s"])),
    ("Peak Metal GiB", lambda r: gib(r["memory"]["peak_metal_gb"])),
    ("Peak process GiB", lambda r: gib(r["memory"]["peak_process_gb"])),
    ("GB read/step", lambda r: (r.get("io") or {}).get("read_gb_per_step", "-")),
    ("Load s", lambda r: r["time"]["load_s"]),
    ("Probe s", lambda r: r["time"]["probe_s"]),
    ("Ckpt MB", lambda r: (r.get("checkpoint") or {}).get("adapter_mb")),
    ("Save s", lambda r: (r.get("checkpoint") or {}).get("save_s")),
]


def rows_for(results: list[dict]) -> list[dict]:
    out = []
    for r in latest(results):
        if r.get("status") != "ok":
            continue
        row = {"Machine": r["machine"].get("key")}
        for name, fn in COLUMNS:
            try:
                row[name] = fn(r)
            except Exception:
                row[name] = None
        out.append(row)
    out.sort(key=lambda x: (x["Machine"] or "", x.get("Size GB") or 0, x["Seq"] or 0))
    return out


def _machine_line(m: dict) -> str:
    bits = [m.get("chip"), f"{m.get('ram_gb')} GB"]
    if m.get("gpu_cores"):
        bits.append(f"{m['gpu_cores']}-core GPU")
    if m.get("disk_gb_s"):
        bits.append(f"SSD ~{m['disk_gb_s']:.1f} GB/s")
    if m.get("macos"):
        bits.append(f"macOS {m['macos']}")
    if m.get("mlx"):
        bits.append(f"mlx {m['mlx']}, mlx-lm {m.get('mlx_lm')}, paged-moe {m.get('pagedmoe')}")
    return ", ".join(str(b) for b in bits if b)


HEADLINE = ["Model", "Size GB", "Mode", "Method", "Seq", "tok/s", "s/step", "Epoch (1k rows)", "Peak Metal GiB"]


def table(results: list[dict], fmt: str = "md", columns: list[str] | None = None) -> str:
    rows = rows_for(results)
    if fmt == "json":
        return json.dumps(rows, indent=2, default=str) + "\n"
    cols = ["Machine"] + [c for c, _ in COLUMNS]
    if fmt == "csv":
        import csv
        import io

        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
        return buf.getvalue()
    machines = {}
    for r in latest(results):
        if r.get("status") == "ok":
            machines[r["machine"].get("key")] = r["machine"]
    out = []
    for key, m in machines.items():
        out.append(f"### {key}\n\n{_machine_line(m)}\n")
        mine = [r for r in rows if r["Machine"] == key]
        head = columns or [c for c, _ in COLUMNS]
        out.append("| " + " | ".join(head) + " |")
        out.append("|" + "|".join("---" for _ in head) + "|")
        for r in mine:
            out.append("| " + " | ".join("-" if r[c] is None else str(r[c]) for c in head) + " |")
        out.append("")
    return "\n".join(out) + "\n"


def write_doc(results: list[dict], path: Path) -> Path:
    """A Markdown page: the table per machine, then each case in detail."""
    lines = [
        "# Training benchmark",
        "",
        "LoRA training on Apple Silicon with paged-moe. Each row is a few real optimizer steps on a fixed",
        "synthetic corpus, planned the same way a normal run is. tok/s counts every position processed",
        "(prompt and answer). Peak Metal is the most MLX held at once; peak process is the whole footprint.",
        "Memory is in GiB, the unit a Mac's RAM is sold in (a 48 GB Mac has 48 GiB). Model size and disk reads",
        "are decimal GB, like Finder shows. Each case's details have the memory in decimal GB too.",
        "Streamed models keep the backbone in memory and read experts from the SSD each step.",
        "",
        "Run it yourself: `python -m pagedmoe.train bench run <model>` (or the Bench tab in Paged Trainer).",
        "",
        "## Results",
        "",
        table(results, "md", HEADLINE),
        "## All columns",
        "",
        table(results, "md"),
        "## Details",
        "",
    ]
    for r in sorted(latest(results), key=lambda r: (r.get("machine", {}).get("key", ""), r.get("case_key", ""))):
        if r.get("status") != "ok":
            continue
        s, m, t, pl = r["speed"], r["memory"], r["time"], r["plan"]
        io = r.get("io") or {}
        ck = r.get("checkpoint") or {}
        lines += [
            f"<details><summary>{r['model']['name']}, {r['setup']['method']} seq {r['setup']['seq']}, "
            f"{r['machine'].get('key')}</summary>",
            "",
            f"- plan: {pl['mode']}, {pl['executor']} executor, {pl['rows_per_step']} rows per step, "
            f"micro-batch {pl['micro_tokens']} tokens" + (f", expert cache {pl['cache_gb']} GB" if pl.get("cache_gb") else ""),
            f"- speed: {s['tok_s']:,} tok/s ({s['trained_tok_s']:,} trained), {s['step_s']} s/step "
            f"(median {s['step_s_median']}, {s['step_s_min']} to {s['step_s_max']}, {s['timed_steps']} timed)"
            + (f", probe estimate {s['est_tok_s']:,} tok/s ({s['est_error_pct']:+}%)" if s.get("est_tok_s") else ""),
            f"- setup: data {t['data_s']} s, load {t['load_s']} s, probe {t['probe_s']} s, first step {t['first_step_s']} s",
            f"- memory: peak {gib(m['peak_metal_gb'])} GiB Metal ({m['peak_metal_gb']} GB decimal), "
            f"{gib(m['peak_process_gb'])} GiB process ({m['peak_process_gb']} GB decimal), "
            f"training budget {gib(m['budget_gb'])} GiB ({m['budget_gb']} GB decimal), "
            f"compressor {gib(m['compressor_start_gb'])} to {gib(m['compressor_peak_gb'])} GiB, "
            f"swap {gib(m['swap_start_gb'])} to {gib(m['swap_peak_gb'])} GiB, pressure {m['pressure_max']}",
        ]
        if m.get("by_phase"):
            lines.append("- memory by phase: " + "; ".join(
                f"{k} {gib(v['metal_gb'])} GiB Metal / {gib(v['process_gb'])} GiB process, {v['pressure']}"
                for k, v in m["by_phase"].items()))
        changes = (m.get("refit") or []) + (m.get("guard") or [])
        if changes:
            lines.append("- memory changes: " + "; ".join(changes)
                         + (f" (timed from step {s['timed_from_step']})" if s.get("settled_after_change") else ""))
        if io:
            lines.append(f"- streaming: {io['read_gb_per_step']} GB read per step ({io['passes']} passes), "
                         f"{io['read_gb_s']} GB/s, cache hit {io['cache_hit']:.0%}, disk wait {io['disk_wait_pct']}%")
        if ck:
            lines.append(f"- checkpoint: adapter {ck.get('adapter_mb')} MB, optimizer {ck.get('optimizer_mb')} MB, "
                         f"save {ck.get('save_s')} s, load {ck.get('load_s')} s")
        lr = r.get("lora") or {}
        if lr.get("trainable"):
            lines.append(f"- LoRA: {lr['trainable'] / 1e6:.1f}M trainable ({lr['pct']}% of {lr['total'] / 1e9:.1f}B)")
        lines += ["", "</details>", ""]
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


# ------------------------------------------------------------------- cli


def _die_with_parent() -> None:
    """A case left running after its parent died would hold the GPU with
    nobody watching, so it ends itself when the parent goes away."""
    parent = os.getppid()

    def watch():
        while True:
            time.sleep(2)
            if os.getppid() != parent:
                os.kill(os.getpid(), signal.SIGKILL)

    threading.Thread(target=watch, daemon=True).start()


def _setups(a) -> list[dict]:
    out = []
    for method in a.method.split(","):
        if method not in METHODS:
            raise SystemExit(f"unknown method {method!r} (one of {', '.join(METHODS)})")
        for seq in [int(x) for x in str(a.seq).split(",")]:
            for rank in [int(x) for x in str(a.rank).split(",")]:
                for target in a.target.split(","):
                    out.append({"method": method, "seq": seq, "rank": rank, "target": target,
                                "batch": a.batch})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pagedmoe.train bench",
                                 description="Training benchmark: speed, memory and checkpoint cost per model.")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("list", help="the ladder")
    sub.add_parser("status", help="which ladder models are here and what is measured")
    r = sub.add_parser("run", help="run cases (ladder ids, folder names or paths)")
    r.add_argument("models", nargs="*", help="default: every ladder model found here")
    r.add_argument("--method", default="sft", help="comma list, e.g. sft,dpo")
    r.add_argument("--seq", default="2048", help="comma list of sequence lengths, e.g. 2048,8192")
    r.add_argument("--rank", default="16", help="comma list of LoRA ranks")
    r.add_argument("--target", default="all-linear", help="attention | all-linear | experts (comma list)")
    r.add_argument("--batch", type=int, default=None, help="rows per step (default: planned for this Mac)")
    r.add_argument("--steps", type=int, default=4, help="timed steps")
    r.add_argument("--warmup", type=int, default=1, help="untimed steps first (cold caches, kernel compile)")
    r.add_argument("--probe", default="always", choices=["always", "auto"], help="auto reuses a cached probe")
    r.add_argument("--rows", type=int, default=256)
    r.add_argument("--keep", action="store_true", help="keep the bench run folders")
    r.add_argument("--skip-done", action="store_true", help="skip cases already measured on this Mac")
    r.add_argument("--doc", default=None, help="also rewrite this Markdown page from all results")
    t = sub.add_parser("table", help="print the results table")
    t.add_argument("--format", default="md", choices=["md", "csv", "json"])
    t.add_argument("--doc", default=None, help="write the full Markdown page here instead")
    one = sub.add_parser("one", help=argparse.SUPPRESS)
    one.add_argument("case")
    a = ap.parse_args(argv)

    if a.cmd == "one":
        case = json.loads(a.case)
        _die_with_parent()
        res = run_case(case)
        Path(case["out"]).write_text(json.dumps(res, indent=2, default=str) + "\n")
        return 0 if res["status"] == "ok" else 1
    if a.cmd == "list":
        for e in LADDER:
            print(f"{e['id']:<14} {e['name']:<22} ~{e['approx_gb']} GB  {e.get('hf') or ''}")
        return 0
    if a.cmd == "status":
        done = done_cases()
        print(_machine_line(machine()))
        for s in status():
            where = s["path"] or f"missing{' (' + s['hf'] + ')' if s.get('hf') else ''}"
            meas = sorted(k for k in done if k.startswith(s["case_id"] + "|"))
            label = s["case_id"] if s["ladder"] else f"{s['name']} (other)"
            print(f"  {label:<14} {s.get('mode') or '':<9} {where}")
            for k in meas:
                print(f"      measured: {k}")
        return 0
    if a.cmd == "table":
        res = load_results()
        if a.doc:
            print(write_doc(res, Path(a.doc)))
        else:
            sys.stdout.write(table(res, a.format))
        return 0
    if a.cmd == "run":
        models = a.models or list({s["case_id"]: s["path"] for s in status() if s["found"] and s["ladder"]}.values())
        if not models:
            raise SystemExit("no ladder models found; pass a model path")
        results = run(models, _setups(a), steps=a.steps, warmup=a.warmup, probe=a.probe, rows=a.rows,
                      keep=a.keep, skip_done=a.skip_done)
        print()
        sys.stdout.write(table(results, "md"))
        print(f"results in {results_dir()}")
        if a.doc:
            print(write_doc(load_results(), Path(a.doc)))
        return 0 if all(r.get("status") == "ok" for r in results) else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
