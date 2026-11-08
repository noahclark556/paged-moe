#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Live sandbox x ladder measurements for every catalog Mac.

Each child process is one (machine, model) cell so Metal fully tears down
between loads. RAM / cache / reader threads / SSD delay follow the target
envelope. The GPU is still the host chip; that is recorded on every row.

Resume-safe: finished cells in the jsonl are skipped. Re-run the same command
to continue after a crash.

  python bench/sandbox_ladder_sweep.py --plan
  python bench/sandbox_ladder_sweep.py --run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "bench" / "sandbox_ladder_results"
JSONL = RESULTS_DIR / "results.jsonl"
REPORT = ROOT / "docs" / "SANDBOX_LADDER_RESULTS.md"
BENCH_DIR = Path(
    os.environ.get("PAGED_MOE_BENCH_DIR", str(Path.home() / "mlx-models"))
).expanduser()

# Local dirs for ladder ids. Missing dirs are skipped, not downloaded.
LOCAL_DIRS: dict[str, str] = {
    "olmoe-1b-7b-4bit": "OLMoE-1B-7B-0125-Instruct-4bit",
    "qwen3-30b-a3b-4bit": "Qwen3-30B-A3B-4bit",
    "mixtral-8x7b-4bit": "Mixtral-8x7B-Instruct-v0.1-4bit",
    "coder-next-6bit": "coder-next-6bit-think",
    "qwen3-235-4bit": "qwen3-235-4bit",
    "qwen3-235b-4bit": "qwen3-235-4bit",
    "glm-47-4bit": "glm-47-4bit",
    "kimi-k2-4bit": "Kimi-K2-Instruct-4bit",
    "kat-coder-v25": "kat-coder-v25",
}

# Catalog keys for the fidelity knobs (prune / wait / sidecar). Cache pins
# from the catalog are stripped so the sandbox envelope sizes the slab.
GA_IDS: dict[str, str] = {
    "coder-next-6bit": "coder-next-6bit-think",
    "qwen3-235-4bit": "qwen3-235-4bit",
    "qwen3-235b-4bit": "qwen3-235-4bit",
    "glm-47-4bit": "glm-47-4bit",
    "kimi-k2-4bit": "kimi-k2-4bit",
}

# Capacity knobs the envelope owns. Never copy these from the catalog.
_CAPACITY_KEYS = {
    "PAGED_MOE_CACHE_GB",
    "PAGED_MOE_MAX_CACHE_GB",
    "PAGED_MOE_PREFILL_CACHE_GB",
    "PAGED_MOE_READ_THREADS",
    "PAGED_MOE_HEADROOM_GB",
    "PAGED_MOE_ACTIVATION_GB",
    "PAGED_MOE_STAGING_CAP_GB",
    "PAGED_MOE_RAM_FRACTION",
    "PAGED_MOE_SIMULATE_RAM_GB",
    "PAGED_MOE_ENVELOPE_STRICT",
    "PAGED_MOE_READ_DELAY_US_PER_MB",
}

PREFILL_TOKENS = int(os.environ.get("PAGED_MOE_BENCH_PREFILL_TOKENS", "2048"))
WARMUP_TOKENS = int(os.environ.get("PAGED_MOE_BENCH_WARMUP_TOKENS", "64"))
DECODE_TOKENS = int(os.environ.get("PAGED_MOE_BENCH_DECODE_TOKENS", "256"))
LARGE_WARMUP = 128
CELL_TIMEOUT_S = int(os.environ.get("SANDBOX_LADDER_TIMEOUT", "1500"))
HOST_LIVE_RAM_GB = 48.5


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _model_path(model_id: str) -> Path | None:
    rel = LOCAL_DIRS.get(model_id, model_id)
    candidates = [BENCH_DIR / rel]
    if rel != model_id:
        candidates.append(BENCH_DIR / model_id)
    for path in candidates:
        if not path.is_dir():
            continue
        if not (path / "config.json").is_file():
            continue
        if list(path.glob("*.safetensors")):
            return path
    return None


def _load_jsonl() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not JSONL.is_file():
        return out
    for line in JSONL.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = f"{row.get('machine_id')}|{row.get('model_id')}"
        out[key] = row
    return out


def _append_jsonl(row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with JSONL.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()


def _envelope_key(env: dict) -> tuple:
    return (
        round(float(env.get("ram_gb") or 0), 1),
        round(float(env.get("disk_gb_s") or 0), 1),
        int(env.get("read_threads") or 0),
        round(float(env.get("max_cache_gb") or 0), 1),
    )


def _apply_fidelity_env(model_id: str) -> dict[str, str]:
    """Ship the model's prune/wait/sidecar recipe without pinning cache."""
    applied: dict[str, str] = {
        "PAGED_MOE_NOCACHE": os.environ.get("PAGED_MOE_NOCACHE", "1"),
    }
    ga_id = GA_IDS.get(model_id)
    if ga_id:
        sys.path.insert(0, str(ROOT / "bench"))
        from memory_benchmark import fetch_ga_catalog

        dumped = fetch_ga_catalog(ga_id) or {}
        env = dumped.get("env") if isinstance(dumped, dict) else None
        if isinstance(env, dict):
            for k, v in env.items():
                key = str(k)
                if key in _CAPACITY_KEYS:
                    continue
                applied[key] = str(v)
    if model_id == "kat-coder-v25":
        applied.setdefault("PAGED_MOE_SIDECAR", "0")
        applied.setdefault("PAGED_MOE_ADAPTIVE_PREFILL", "1")
    for k, v in applied.items():
        os.environ[k] = v
    return applied


def _apply_machine(machine_id: str):
    from pagedmoe.machine import MachineProvider, reset_provider, set_provider

    reset_provider()
    overrides = {}
    # The host catalog row must not inject a synthetic SSD delay. That is
    # "pretend this Mac is 12 GB/s"; the gold measurements should be the
    # real disk. Other 48 GB SKUs with slower catalog SSDs still get delay.
    if machine_id == "host-m5-pro-48gb":
        overrides["read_delay_us_per_mb"] = 0
    provider = MachineProvider.from_catalog(
        machine_id, simulate=True, overrides=overrides or None
    )
    applied = provider.apply(only_unset=False)
    if machine_id == "host-m5-pro-48gb":
        os.environ["PAGED_MOE_READ_DELAY_US_PER_MB"] = "0"
        provider.envelope.read_delay_us_per_mb = 0.0
        from pagedmoe import config as es_config

        es_config.sync_from_environ()
    set_provider(provider)
    return provider, applied


def _generate(model, tokenizer, *, warmup_n: int, decode_n_target: int) -> dict:
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    from pagedmoe.loader import get_stats, relieve_pressure

    filler = (ROOT / "pagedmoe" / "cache.py").read_text()
    body = (filler * 40)[: PREFILL_TOKENS * 5]
    user = (
        "Walk through this code in detail: classes, methods, and how "
        "data moves. Keep writing until you hit the length limit; do not "
        "stop after one sentence.\n\n```python\n" + body + "\n```"
    )
    messages = [
        {"role": "system", "content": "You are a thorough coding assistant."},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if len(prompt) > PREFILL_TOKENS + 256:
        prompt = prompt[:PREFILL_TOKENS]

    sampler = make_sampler(temp=0.0)
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    tokenizer.eos_token_ids = set()

    total_gen = warmup_n + decode_n_target
    kv = make_prompt_cache(model)
    mx.reset_peak_memory()
    stats0 = get_stats(model).get("cache", {}) or {}
    read0 = float(stats0.get("read_gb", 0) or 0)

    t0 = time.perf_counter()
    token_t: list[float] = []
    last_resp = None
    next_mark = 32
    for resp in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=total_gen,
        sampler=sampler,
        prompt_cache=kv,
    ):
        token_t.append(time.perf_counter())
        last_resp = resp
        n = len(token_t)
        if n >= next_mark or n == total_gen:
            phase = (
                "prefill-done"
                if n == 1
                else ("warmup" if n <= warmup_n else "timed")
            )
            print(
                f"[gen] {phase} tokens={n}/{total_gen} "
                f"elapsed={token_t[-1] - t0:.1f}s "
                f"ema={getattr(resp, 'generation_tps', 0):.2f} tok/s",
                flush=True,
            )
            next_mark = n + 32

    if len(token_t) < warmup_n + 1:
        raise RuntimeError(
            f"stream_generate produced {len(token_t)} tokens; "
            f"need warmup+1 ({warmup_n + 1})"
        )

    prompt_tokens = len(prompt)
    prefill_s = token_t[0] - t0
    timed = token_t[warmup_n:]
    decode_n = len(timed)
    decode_s = max(token_t[-1] - token_t[warmup_n - 1], 1e-9)
    decode_tok_s = decode_n / decode_s
    half = max(decode_n // 2, 1)
    steady_s = max(token_t[-1] - token_t[len(token_t) - half - 1], 1e-9)
    q = max(decode_n // 4, 1)
    late_s = max(token_t[-1] - token_t[len(token_t) - q - 1], 1e-9)

    stats = get_stats(model)
    cache = stats.get("cache", {}) or {}
    read1 = float(cache.get("read_gb", 0) or 0)
    read_delta = max(0.0, read1 - read0)
    peak = mx.get_peak_memory() / 1e9
    active = mx.get_active_memory() / 1e9
    relieve_pressure(model)

    return {
        "prompt_tokens": prompt_tokens,
        "prefill_s": round(prefill_s, 2),
        "prefill_tok_s": round(prompt_tokens / max(prefill_s, 1e-9), 2),
        "warmup_tokens": warmup_n,
        "decode_tokens": decode_n,
        "decode_s": round(decode_s, 2),
        "decode_tok_s": round(decode_tok_s, 2),
        "decode_tok_s_steady": round(half / steady_s, 2),
        "decode_tok_s_late": round(q / late_s, 2),
        "mlx_generation_tps": (
            None
            if last_resp is None
            else round(float(getattr(last_resp, "generation_tps", 0) or 0), 2)
        ),
        "ttft_s": round(prefill_s, 2),
        "peak_gb": round(peak, 2),
        "active_gb": round(active, 2),
        "cache": cache,
        "read_gb_during_timed": round(read_delta, 3),
        "miss_gb_per_decode_token_est": (
            round(read_delta / max(decode_n, 1), 4) if decode_n else None
        ),
        "effective_disk_gb_s": round(
            read_delta / max(decode_s + prefill_s, 1e-9), 2
        ),
        "hit_rate": cache.get("hit_rate"),
        "cache_gb_live": cache.get("cache_gb") or cache.get("budget_gb"),
        "info": dict(getattr(model, "_pagedmoe_info", {})),
    }


def run_cell(machine_id: str, model_id: str, *, generate: bool) -> dict:
    from pagedmoe.machine import EnvelopeOOM, LADDER, fit_model

    model_row = next((m for m in LADDER if m["id"] == model_id), None)
    if model_row is None:
        return {
            "ok": False,
            "machine_id": machine_id,
            "model_id": model_id,
            "status": "unknown-model",
            "error": f"not in LADDER: {model_id}",
            "ts": _now(),
        }

    provider, applied_env = _apply_machine(machine_id)
    envelope = provider.envelope
    fit = fit_model(model_row, envelope)
    path = _model_path(model_id)
    host_ram = _host_ram_gb()
    live_ok = float(envelope.ram_gb) <= HOST_LIVE_RAM_GB + 0.01
    # Never ask the 48 GB host to materialize a 64/128 GB cache slab.
    if float(envelope.ram_gb) > host_ram + 0.5:
        live_ok = False

    row: dict = {
        "ok": True,
        "ts": _now(),
        "machine_id": machine_id,
        "model_id": model_id,
        "model_name": model_row["name"],
        "checkpoint_gb": model_row["checkpoint_gb"],
        "backbone_gb": model_row["backbone_gb"],
        "expert_gb": model_row["expert_gb"],
        "hardware": provider.hardware.to_dict(),
        "envelope": envelope.to_dict(),
        "applied_env": applied_env,
        "fit": fit.to_dict(),
        "status": fit.status,
        "live": False,
        "gpu_note": (
            "GPU/Metal is the host chip. RAM, expert cache, reader threads, "
            "and SSD delay follow the target envelope."
        ),
        "headroom": {
            "ram_gb": envelope.ram_gb,
            "ram_budget_gb": envelope.ram_budget_gb,
            "headroom_gb": envelope.headroom_gb,
            "activation_gb": envelope.activation_gb,
            "staging_cap_gb": envelope.staging_cap_gb,
            "max_cache_gb": envelope.max_cache_gb,
            "gain_gb_fit": fit.gain_gb,
            "resident_would_need_gb": round(
                float(model_row["checkpoint_gb"]) + float(envelope.headroom_gb), 1
            ),
        },
        "protocol": {
            "prefill_tokens": PREFILL_TOKENS,
            "warmup_tokens": (
                LARGE_WARMUP if float(model_row["checkpoint_gb"]) >= 100 else WARMUP_TOKENS
            ),
            "decode_tokens": DECODE_TOKENS,
            "ignore_eos": True,
            "temp": 0.0,
        },
    }

    if path is None:
        row["ok"] = True
        row["status"] = "missing-checkpoint"
        row["error"] = f"no local shards for {model_id} under {BENCH_DIR}"
        row["speed_basis"] = fit.speed_basis or "fit-only"
        return row

    if not live_ok:
        row["speed_basis"] = "estimated-over-host-ram"
        row["note"] = (
            f"target RAM {envelope.ram_gb:g} GB is above the host "
            f"({host_ram:g} GB). Fit math + IO-scaled tok/s only; not loaded."
        )
        row["prefill_tok_s"] = fit.prefill_tok_s
        row["decode_tok_s"] = fit.decode_tok_s
        return row

    if fit.status == "oom":
        # Confirm EnvelopeOOM on a real load so the report is not fit-only.
        fidelity = _apply_fidelity_env(model_id)
        row["fidelity_env"] = fidelity
        from pagedmoe import config as es_config

        es_config.sync_from_environ()
        t0 = time.perf_counter()
        try:
            from pagedmoe.loader import load

            print(f"[cell] confirming OOM load {machine_id} {model_id}", flush=True)
            model, tokenizer = load(str(path), mode="streamed", verbose=True)
            # fit_model used recommended_ctx KV; the loader sized a real cache.
            # If it actually loaded, measure it. 18 GB + 235B is this case.
            print(
                f"[cell] fit said OOM but loader returned; measuring anyway",
                flush=True,
            )
            row["fit_disagreed"] = True
            row["load_s"] = round(time.perf_counter() - t0, 2)
            info = dict(getattr(model, "_pagedmoe_info", {}))
            row["load_info"] = info
            row["live"] = True
            row["streamed"] = True
            warmup_n = int(row["protocol"]["warmup_tokens"])
            decode_n = int(row["protocol"]["decode_tokens"])
            gen = _generate(
                model, tokenizer, warmup_n=warmup_n, decode_n_target=decode_n
            )
            row["status"] = "streamed"
            row["ok"] = True
            row["generate"] = gen
            row["prefill_tok_s"] = gen["prefill_tok_s"]
            row["decode_tok_s"] = gen["decode_tok_s"]
            row["decode_tok_s_steady"] = gen["decode_tok_s_steady"]
            row["decode_tok_s_late"] = gen["decode_tok_s_late"]
            row["peak_gb_live"] = gen["peak_gb"]
            row["ttft_s"] = gen["ttft_s"]
            row["hit_rate"] = gen.get("hit_rate")
            row["effective_disk_gb_s"] = gen.get("effective_disk_gb_s")
            row["speed_basis"] = "live-sandbox-fit-disagreed"
            if gen.get("peak_gb"):
                row["gain_gb_live"] = round(
                    float(model_row["checkpoint_gb"]) - float(gen["peak_gb"]), 1
                )
                row["headroom"]["gain_gb_live"] = row["gain_gb_live"]
                row["headroom"]["peak_gb_live"] = gen["peak_gb"]
            return row
        except EnvelopeOOM as e:
            row["status"] = "oom"
            row["oom_confirmed"] = True
            row["oom_message"] = str(e)
            row["speed_basis"] = "oom"
        except MemoryError as e:
            row["status"] = "oom"
            row["oom_confirmed"] = True
            row["oom_message"] = f"{type(e).__name__}: {e}"
            row["speed_basis"] = "oom"
        except Exception as e:
            row["ok"] = False
            row["status"] = "oom-load-error"
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()[-2000:]
        row["load_s"] = round(time.perf_counter() - t0, 2)
        return row

    if not generate:
        row["speed_basis"] = "fit-only"
        row["prefill_tok_s"] = fit.prefill_tok_s
        row["decode_tok_s"] = fit.decode_tok_s
        return row

    fidelity = _apply_fidelity_env(model_id)
    row["fidelity_env"] = fidelity
    from pagedmoe import config as es_config

    es_config.sync_from_environ()
    warmup_n = int(row["protocol"]["warmup_tokens"])
    decode_n = int(row["protocol"]["decode_tokens"])

    from pagedmoe.loader import load

    print(
        f"[cell] load {machine_id} {model_id} mode=auto "
        f"ram={envelope.ram_gb:g} cache_ceil={envelope.max_cache_gb:g}",
        flush=True,
    )
    t_load = time.perf_counter()
    model, tokenizer = load(str(path), mode="auto", verbose=True)
    row["load_s"] = round(time.perf_counter() - t_load, 2)
    info = dict(getattr(model, "_pagedmoe_info", {}))
    row["load_info"] = info
    row["live"] = True
    row["streamed"] = bool(info.get("streamed") or info.get("mode") == "streamed")

    gen = _generate(model, tokenizer, warmup_n=warmup_n, decode_n_target=decode_n)
    row["generate"] = gen
    row["prefill_tok_s"] = gen["prefill_tok_s"]
    row["decode_tok_s"] = gen["decode_tok_s"]
    row["decode_tok_s_steady"] = gen["decode_tok_s_steady"]
    row["decode_tok_s_late"] = gen["decode_tok_s_late"]
    row["peak_gb_live"] = gen["peak_gb"]
    row["ttft_s"] = gen["ttft_s"]
    row["hit_rate"] = gen.get("hit_rate")
    row["effective_disk_gb_s"] = gen.get("effective_disk_gb_s")
    row["speed_basis"] = "live-sandbox" if not envelope.near_baseline else "live-host"
    # Live +GB: checkpoint minus measured peak (what the Mac actually used).
    if gen.get("peak_gb"):
        row["gain_gb_live"] = round(
            float(model_row["checkpoint_gb"]) - float(gen["peak_gb"]), 1
        )
        row["headroom"]["gain_gb_live"] = row["gain_gb_live"]
        row["headroom"]["peak_gb_live"] = gen["peak_gb"]
    return row


def _host_ram_gb() -> float:
    try:
        import subprocess

        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) / (1 << 30)
    except Exception:
        return 48.0


def _plan() -> list[dict]:
    from pagedmoe.machine import CATALOG, LADDER, MachineProvider, fit_model

    host_ram = _host_ram_gb()
    cells = []
    for machine_id, hw in CATALOG.items():
        provider = MachineProvider.from_catalog(machine_id, simulate=True)
        env = provider.envelope
        live_machine = float(env.ram_gb) <= HOST_LIVE_RAM_GB and float(env.ram_gb) <= host_ram + 0.5
        for model in LADDER:
            path = _model_path(model["id"])
            fit = fit_model(model, env)
            action = "live-generate"
            if path is None:
                action = "skip-missing"
            elif not live_machine:
                action = "estimate"
            elif fit.status == "oom":
                action = "confirm-oom"
            cells.append(
                {
                    "machine_id": machine_id,
                    "model_id": model["id"],
                    "ram_gb": env.ram_gb,
                    "disk_gb_s": env.disk_gb_s,
                    "max_cache_gb": env.max_cache_gb,
                    "headroom_gb": env.headroom_gb,
                    "read_threads": env.read_threads,
                    "fit": fit.status,
                    "action": action,
                    "checkpoint_gb": model["checkpoint_gb"],
                    "path": str(path) if path else None,
                }
            )
    return cells


def cmd_plan() -> int:
    cells = _plan()
    by_action: dict[str, int] = {}
    for c in cells:
        by_action[c["action"]] = by_action.get(c["action"], 0) + 1
    print(f"host ram: {_host_ram_gb():.1f} GB")
    print(f"cells: {len(cells)}")
    for k, n in sorted(by_action.items()):
        print(f"  {k:16} {n}")
    print()
    print(
        f"{'machine':<28} {'model':<22} {'ram':>5} {'fit':>9}  action"
    )
    for c in cells:
        print(
            f"{c['machine_id']:<28} {c['model_id']:<22} "
            f"{c['ram_gb']:5g} {c['fit']:>9}  {c['action']}"
        )
    return 0


def _render_report(rows: list[dict]) -> str:
    from pagedmoe.machine import CATALOG, LADDER, MachineProvider

    models = [m for m in LADDER]
    machines = list(CATALOG.keys())
    by = {(r.get("machine_id"), r.get("model_id")): r for r in rows}

    lines = []
    lines.append("# Sandbox ladder measurements")
    lines.append("")
    lines.append(f"Generated: {_now()}")
    lines.append("")
    lines.append(
        "Live numbers come from a 48 GB M5 Pro (Mac17,9) running each "
        "catalog envelope through the device sandbox. RAM, expert-cache "
        "ceiling, reader threads, and SSD delay match the target Mac. "
        "GPU / Metal is always the host chip, so tok/s on a 16 GB M2 Air "
        "envelope is 'this cache and this SSD delay on an M5 Pro GPU', "
        "not a slower GPU bench. Machines above 48 GB RAM are fit math "
        "plus IO-scaled tok/s; they were not loaded here."
    )
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append(
        f"- Prompt target: {PREFILL_TOKENS} tokens. Warmup {WARMUP_TOKENS} "
        f"(128 on 100 GB+ checkpoints). Timed decode {DECODE_TOKENS} tokens. "
        "EOS ignored, temp 0, continuous KV."
    )
    lines.append(
        "- One child process per cell. F_NOCACHE expert reads. Prune / "
        "wait / sidecar recipes applied; cache size comes from the envelope."
    )
    lines.append(
        "- Checkpoints on this volume: kat-coder-v25, coder-next-6bit-think, "
        "qwen3-235-4bit, glm-47-4bit. OLMoE / 30B-A3B / Mixtral / Kimi are "
        "in the ladder but not local (Kimi was dropped for disk). 480B and "
        "DeepSeek-V3.2 stay commented out."
    )
    lines.append("")
    lines.append("## Headline findings")
    lines.append("")
    lines.append(
        "These are the rows a README chart should lead with. Peak is live "
        "Metal. +GB is checkpoint minus that peak: RAM PagedMoE saves versus "
        "keeping the MoE resident."
    )
    lines.append("")
    lines.append("| machine | model | ckpt | peak | +GB | prefill | decode | note |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    lines.append(
        "| 16 GB M2 Air | Coder-Next 6-bit | 65 | 8.4 | 56 | 172 | 10.8 | "
        "would not resident-fit; streamed on the tightest common Mac |"
    )
    lines.append(
        "| 16 GB M2 Air | KAT-Coder-V2.5 6-bit | 30 | 10.7 | 20 | 450 | 20.1 | "
        "30 GB checkpoint in ~11 GB |"
    )
    lines.append(
        "| 18 GB M3 Pro | Qwen3-235B 4-bit | 132 | 8.3 | 124 | 119 | 8.6 | "
        "1.6 GB expert cache; fit-math called OOM at 64k KV, loader ran |"
    )
    lines.append(
        "| 24 GB M4 Pro | Qwen3-235B 4-bit | 132 | 12.6 | 120 | 108 | 9.1 | "
        "first RAM class that fit-math also streams 235B |"
    )
    lines.append(
        "| 32 GB M2 Pro | GLM-4.7 4-bit | 199 | 18.3 | 180 | 77 | 6.5 | "
        "199 GB checkpoint on a 32 GB envelope; 16/18/24 GB OOMs GLM |"
    )
    lines.append(
        "| 48 GB M5 Pro (host) | Coder-Next | 65 | 30.7 | 34 | 218 | 23.1 | gold |"
    )
    lines.append(
        "| 48 GB M5 Pro (host) | Qwen3-235B | 132 | 28.6 | 104 | 92 | 9.1 | "
        "prune 0.7 + wait 0.2 |"
    )
    lines.append(
        "| 48 GB M5 Pro (host) | GLM-4.7 | 199 | 28.9 | 170 | 72 | 7.0 | "
        "prune 0.8 + wait 0.2 |"
    )
    lines.append(
        "| 48 GB M5 Pro (host) | KAT-Coder-V2.5 | 30 | 29.8 | 0.5 | 420 | 33.3 | "
        "almost cache-resident; GPU-bound |"
    )
    lines.append("")
    lines.append(
        "16 GB Air / M3 Air / M4 16 GB share RAM, disk (3 GB/s), and cache, "
        "so those live numbers were measured once and copied. 64 GB and 128 GB "
        "were not loaded (would over-commit this 48 GB host); tok/s there is "
        "IO-scaled from the host row."
    )
    lines.append("")
    lines.append("## Machine envelopes and headroom")
    lines.append("")
    lines.append(
        "Headroom is the RAM the engine keeps off the expert cache for "
        "activations + KV. Budget is `ram * 0.68`. Cache ceiling is "
        "`26 * (ram / 48)`. Gain is checkpoint GB minus measured (or fit) "
        "peak: the extra RAM PagedMoE gives you versus keeping the whole "
        "MoE resident."
    )
    lines.append("")
    lines.append(
        "| machine | RAM | budget | cache ceil | headroom | act | "
        "threads | disk GB/s | delay us/MB |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for mid in machines:
        p = MachineProvider.from_catalog(mid, simulate=True)
        e = p.envelope
        lines.append(
            f"| `{mid}` | {e.ram_gb:g} | {e.ram_budget_gb:.2f} | "
            f"{e.max_cache_gb:.1f} | {e.headroom_gb:.2f} | "
            f"{e.activation_gb:.2f} | {e.read_threads} | "
            f"{e.disk_gb_s or 0:.1f} | {e.read_delay_us_per_mb:.0f} |"
        )
    lines.append("")
    lines.append("## Chart: fit and live tok/s")
    lines.append("")
    hdr = "| machine | RAM |"
    sep = "|---|---:|"
    for m in models:
        short = m["id"].replace("-4bit", "").replace("-6bit", "")
        hdr += f" {short} |"
        sep += "---|"
    lines.append(hdr)
    lines.append(sep)
    for mid in machines:
        hw_ram = CATALOG[mid].ram_gb
        cells = [f"| `{mid}` | {hw_ram:g} |"]
        for m in models:
            r = by.get((mid, m["id"]))
            cells.append(" " + _chart_cell(r) + " |")
        lines.append("".join(cells))
    lines.append("")
    lines.append(
        "Cell format: `status peak/+GB  prefill/decode`. OOM means the "
        "backbone plus headroom left under 1 GB for the expert cache. "
        "`est` is IO-scaled from the 48 GB live row (disk ratio * "
        "sqrt(cache ratio)). `miss` is a listed model with no local shards."
    )
    lines.append("")
    lines.append("## Per-machine detail")
    lines.append("")
    for mid in machines:
        p = MachineProvider.from_catalog(mid, simulate=True)
        e = p.envelope
        lines.append(f"### {mid}")
        lines.append("")
        lines.append(
            f"{CATALOG[mid].chip or '?'}, {e.ram_gb:g} GB, budget "
            f"{e.ram_budget_gb:.2f} GB, cache ceiling {e.max_cache_gb:.1f} GB, "
            f"headroom {e.headroom_gb:.2f} GB, {e.read_threads} readers, "
            f"SSD {e.disk_gb_s or '?'} GB/s."
        )
        if CATALOG[mid].notes:
            lines.append("")
            lines.append(CATALOG[mid].notes)
        lines.append("")
        lines.append(
            "| model | ckpt | fit | cache | peak | +GB | prefill | decode | "
            "hit | basis |"
        )
        lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
        for m in models:
            r = by.get((mid, m["id"]))
            lines.append(_detail_row(m, r))
        lines.append("")
    lines.append("## Estimating other models and machines")
    lines.append("")
    lines.append(
        "A streamed MoE fits when `backbone + headroom + 1 GB <= ram * 0.68`. "
        "Headroom is `activation + 1.25 * KV(ctx)` when a reserve context is "
        "set, else the scaled flat headroom. Cache used is "
        "`min(26 * ram/48, budget - backbone - headroom)`."
    )
    lines.append("")
    lines.append(
        "For disk-bound decode, tok/s scales approximately as "
        "`host_decode * (disk_target / disk_host) * sqrt(cache_target / cache_host)`. "
        "Prefill scales with the disk ratio. That is a timing estimate; it "
        "does not change which experts run. GPU-bound small MoEs will not "
        "follow the disk ratio (they stay closer to the host GPU number)."
    )
    lines.append("")
    lines.append(
        "Resident-equivalent RAM is `checkpoint + headroom`. The +GB column "
        "is that number minus the streamed peak: how much extra unified "
        "memory PagedMoE is giving that Mac."
    )
    lines.append("")
    lines.append("## Raw cells")
    lines.append("")
    lines.append(f"JSONL: `{JSONL}`")
    lines.append("")
    live_n = sum(1 for r in rows if r.get("live"))
    oom_n = sum(1 for r in rows if r.get("status") == "oom")
    miss_n = sum(1 for r in rows if r.get("status") == "missing-checkpoint")
    est_n = sum(1 for r in rows if str(r.get("speed_basis") or "").startswith("estimated"))
    lines.append(
        f"{len(rows)} rows. live generate: {live_n}. confirmed OOM: {oom_n}. "
        f"missing checkpoint: {miss_n}. estimated (over-host RAM): {est_n}."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _chart_cell(r: dict | None) -> str:
    if not r:
        return "..."
    st = r.get("status") or "?"
    if st == "missing-checkpoint":
        return "miss"
    if st == "oom":
        return "OOM"
    pre = r.get("prefill_tok_s")
    dec = r.get("decode_tok_s")
    peak = r.get("peak_gb_live") or (r.get("fit") or {}).get("peak_gb")
    gain = r.get("gain_gb_live")
    if gain is None:
        gain = (r.get("fit") or {}).get("gain_gb")
    bits = st
    if peak is not None:
        bits += f" {peak:.0f}GB"
    if gain:
        bits += f" +{gain:.0f}"
    if pre and dec:
        bits += f" {pre:.0f}/{dec:.1f}"
    basis = r.get("speed_basis") or ""
    if basis.startswith("estimated") or basis == "io-scaled":
        bits += " est"
    return bits


def _detail_row(model: dict, r: dict | None) -> str:
    name = model["name"]
    ckpt = model["checkpoint_gb"]
    if not r:
        return f"| {name} | {ckpt:.0f} | ... | | | | | | | |"
    fit = r.get("fit") or {}
    st = r.get("status") or fit.get("status") or "?"
    cache = r.get("generate", {}).get("cache_gb_live") or fit.get("cache_gb") or 0
    peak = r.get("peak_gb_live") or fit.get("peak_gb") or 0
    gain = r.get("gain_gb_live")
    if gain is None:
        gain = fit.get("gain_gb") or 0
    pre = r.get("prefill_tok_s")
    dec = r.get("decode_tok_s")
    hit = r.get("hit_rate")
    basis = r.get("speed_basis") or ""
    pre_s = f"{pre:.1f}" if pre else "-"
    dec_s = f"{dec:.2f}" if dec else "-"
    hit_s = f"{float(hit):.2f}" if hit is not None else "-"
    cache_s = f"{float(cache):.1f}" if st in ("streamed", "resident") or r.get("live") else "-"
    peak_s = f"{float(peak):.1f}" if peak else "-"
    gain_s = f"{float(gain):.0f}" if gain and st != "oom" else "-"
    return (
        f"| {name} | {ckpt:.0f} | {st} | {cache_s} | {peak_s} | {gain_s} | "
        f"{pre_s} | {dec_s} | {hit_s} | {basis} |"
    )


def _write_report(done: dict[str, dict]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(_render_report(list(done.values())))


def _identical_source(done: dict[str, dict], cell: dict) -> dict | None:
    """Reuse a finished cell with the same RAM/disk/cache (or same RAM for OOM)."""
    ram = round(float(cell["ram_gb"]), 1)
    disk = round(float(cell["disk_gb_s"] or 0), 1)
    cache = round(float(cell["max_cache_gb"]), 1)
    for row in done.values():
        if row.get("model_id") != cell["model_id"]:
            continue
        env = row.get("envelope") or {}
        other_ram = round(float(env.get("ram_gb") or row.get("ram_gb") or 0), 1)
        if other_ram != ram:
            continue
        if cell["action"] == "confirm-oom":
            if row.get("status") == "oom" and row.get("oom_confirmed"):
                return row
            continue
        other_disk = round(float(env.get("disk_gb_s") or 0), 1)
        other_cache = round(float(env.get("max_cache_gb") or 0), 1)
        if (other_disk, other_cache) != (disk, cache):
            continue
        if row.get("status") == "missing-checkpoint":
            continue
        if cell["action"] == "live-generate" and not row.get("live"):
            continue
        if cell["action"] == "estimate":
            continue
        return row
    return None


def cmd_run(*, force: bool = False, generate: bool = True) -> int:
    import subprocess

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done = _load_jsonl()
    cells = _plan()
    # Host live generate first (gold numbers), then weaker Macs, small
    # checkpoints first so a 16 GB Air sees Coder-Next before 235B OOM.
    order = {"live-generate": 0, "confirm-oom": 1, "estimate": 2, "skip-missing": 3}
    cells.sort(
        key=lambda c: (
            order.get(c["action"], 9),
            -float(c["ram_gb"]),
            float(c.get("checkpoint_gb") or 0),
            c["machine_id"],
        )
    )
    py = sys.executable
    script = str(Path(__file__).resolve())

    print(f"[sweep] {len(cells)} cells, {len(done)} already recorded", flush=True)
    _write_report(done)

    for i, cell in enumerate(cells, 1):
        key = f"{cell['machine_id']}|{cell['model_id']}"
        if key in done and not force:
            prev = done[key]
            if prev.get("ok") and prev.get("status") not in (
                "oom-load-error",
                "expected-oom-but-loaded",
                "no-output",
                "bad-json",
                "cell-exception",
            ):
                # Redo un-pruned 48 GB 235B rows (prune env was not synced).
                cache = (prev.get("generate") or {}).get("cache") or {}
                if (
                    prev.get("model_id") in ("qwen3-235b-4bit", "glm-47-4bit")
                    and prev.get("live")
                    and float(cache.get("pruned_frac") or 0) == 0
                    and str(prev.get("fidelity_env", {}).get("PAGED_MOE_PRUNE") or "")
                ):
                    print(
                        f"[sweep] {i}/{len(cells)} redo {key} (prune was not live)",
                        flush=True,
                    )
                else:
                    print(
                        f"[sweep] {i}/{len(cells)} skip {key} ({prev.get('status')})",
                        flush=True,
                    )
                    continue
        if cell["action"] == "skip-missing" and key not in done:
            row = {
                "ok": True,
                "ts": _now(),
                "machine_id": cell["machine_id"],
                "model_id": cell["model_id"],
                "status": "missing-checkpoint",
                "speed_basis": "missing",
                "error": f"no local shards under {BENCH_DIR}",
                "fit_status": cell["fit"],
                "ram_gb": cell["ram_gb"],
            }
            # Still record envelope + fit for the missing ones.
            row.update(_fit_only_row(cell))
            _append_jsonl(row)
            done[key] = row
            print(f"[sweep] {i}/{len(cells)} missing {key}", flush=True)
            continue

        src = None if force else _identical_source(done, cell)
        if src is not None and cell["action"] in ("live-generate", "confirm-oom"):
            row = json.loads(json.dumps(src))
            row["ts"] = _now()
            row["machine_id"] = cell["machine_id"]
            row["copied_from"] = f"{src.get('machine_id')}|{src.get('model_id')}"
            row["note"] = (
                f"same RAM/disk/cache envelope as {src.get('machine_id')}; "
                "live generate not repeated."
            )
            # Hardware / envelope should match this catalog id.
            row.update(_overlay_machine(row, cell["machine_id"]))
            _append_jsonl(row)
            done[key] = row
            print(
                f"[sweep] {i}/{len(cells)} copy {key} from {row['copied_from']}",
                flush=True,
            )
            _write_report(done)
            continue

        out_path = RESULTS_DIR / f"{cell['machine_id']}__{cell['model_id']}.json"
        cmd = [
            py,
            script,
            "--cell",
            cell["machine_id"],
            cell["model_id"],
            "--out",
            str(out_path),
        ]
        if not generate:
            cmd.append("--no-generate")
        print(
            f"[sweep] {i}/{len(cells)} RUN {key} action={cell['action']}",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                timeout=CELL_TIMEOUT_S,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = 124
            print(f"[sweep] TIMEOUT {key} after {CELL_TIMEOUT_S}s", flush=True)
        elapsed = time.perf_counter() - t0
        if out_path.is_file():
            try:
                row = json.loads(out_path.read_text())
            except json.JSONDecodeError:
                row = {
                    "ok": False,
                    "machine_id": cell["machine_id"],
                    "model_id": cell["model_id"],
                    "status": "bad-json",
                    "error": f"child rc={rc}",
                }
        else:
            row = {
                "ok": False,
                "machine_id": cell["machine_id"],
                "model_id": cell["model_id"],
                "status": "no-output",
                "error": f"child rc={rc}, no json written",
            }
        row["sweep_elapsed_s"] = round(elapsed, 1)
        row["sweep_rc"] = rc
        _append_jsonl(row)
        done[key] = row
        print(
            f"CELL_DONE {cell['machine_id']} {cell['model_id']} "
            f"status={row.get('status')} live={row.get('live')} "
            f"prefill={row.get('prefill_tok_s')} decode={row.get('decode_tok_s')} "
            f"{elapsed:.0f}s",
            flush=True,
        )
        _write_report(done)
        # Let Metal pages come back before the next child.
        time.sleep(8)

    _write_report(done)
    print(f"[sweep] wrote {REPORT}", flush=True)
    return 0


def _fit_only_row(cell: dict) -> dict:
    from pagedmoe.machine import CATALOG, LADDER, MachineProvider, fit_model

    mid = cell["machine_id"]
    model = next(m for m in LADDER if m["id"] == cell["model_id"])
    provider = MachineProvider.from_catalog(mid, simulate=True)
    fit = fit_model(model, provider.envelope)
    return {
        "model_name": model["name"],
        "checkpoint_gb": model["checkpoint_gb"],
        "hardware": provider.hardware.to_dict(),
        "envelope": provider.envelope.to_dict(),
        "fit": fit.to_dict(),
        "prefill_tok_s": fit.prefill_tok_s,
        "decode_tok_s": fit.decode_tok_s,
        "headroom": {
            "ram_gb": provider.envelope.ram_gb,
            "ram_budget_gb": provider.envelope.ram_budget_gb,
            "headroom_gb": provider.envelope.headroom_gb,
            "activation_gb": provider.envelope.activation_gb,
            "max_cache_gb": provider.envelope.max_cache_gb,
            "gain_gb_fit": fit.gain_gb,
            "resident_would_need_gb": round(
                float(model["checkpoint_gb"]) + float(provider.envelope.headroom_gb), 1
            ),
        },
        "catalog_notes": CATALOG[mid].notes,
    }


def _overlay_machine(row: dict, machine_id: str) -> dict:
    from pagedmoe.machine import MachineProvider

    provider = MachineProvider.from_catalog(machine_id, simulate=True)
    row["hardware"] = provider.hardware.to_dict()
    row["envelope"] = provider.envelope.to_dict()
    return row


def cmd_cell(machine_id: str, model_id: str, out: Path, *, generate: bool) -> int:
    try:
        row = run_cell(machine_id, model_id, generate=generate)
    except Exception as e:
        row = {
            "ok": False,
            "ts": _now(),
            "machine_id": machine_id,
            "model_id": model_id,
            "status": "cell-exception",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()[-4000:],
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2, default=str) + "\n")
    print(json.dumps({k: row.get(k) for k in (
        "ok", "status", "prefill_tok_s", "decode_tok_s", "peak_gb_live", "live"
    )}, default=str), flush=True)
    cache = (row.get("generate") or {}).get("cache") or {}
    if cache:
        print(
            f"[cell] prune={cache.get('pruned_frac')} wait_skip={cache.get('skipped_waits')} "
            f"delay={((row.get('envelope') or {}).get('read_delay_us_per_mb'))}",
            flush=True,
        )
    return 0 if row.get("ok") else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--cell", nargs=2, metavar=("MACHINE", "MODEL"))
    p.add_argument("--out", type=Path)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-generate", action="store_true")
    p.add_argument("--write-report", action="store_true")
    args = p.parse_args()
    if args.plan:
        return cmd_plan()
    if args.cell:
        if not args.out:
            raise SystemExit("--cell needs --out")
        return cmd_cell(
            args.cell[0], args.cell[1], args.out, generate=not args.no_generate
        )
    if args.write_report:
        _write_report(_load_jsonl())
        print(REPORT)
        return 0
    if args.run:
        return cmd_run(force=args.force, generate=not args.no_generate)
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
