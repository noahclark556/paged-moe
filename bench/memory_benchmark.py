#!/usr/bin/env python3
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One-tier memory / throughput ladder for paged-moe.

Driven by bench/memory_benchmark.sh. See docs/MEMORY_BENCHMARK.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DOC = ROOT / "docs" / "MEMORY_BENCHMARK.md"
RESULTS = ROOT / "bench" / "memory_results"
BENCH_DIR = Path(
    os.environ.get("PAGED_MOE_BENCH_DIR", str(Path.home() / "mlx-models"))
).expanduser()

# Fixed workload so tiers are comparable across machines and days.
#
# Decode timing used to be wrong in two ways:
#   1. "Warmup" ran on a different short prompt, then the timed run used a
#      fresh KV - so warmup never warmed the path being measured.
#   2. Only 64 tokens after a 2k prefill billed the post-prefill cache rebuild
#      as "decode tok/s" (GLM/480 looked like ~1 tok/s; agent runs are faster).
#
# Protocol now: one continuous generation on the real prompt. Prefill ends at
# the first token (TTFT). The next WARMUP_TOKENS rebuild the expert working
# set and are discarded. The following DECODE_TOKENS are the wall-clock rate.
#
# --steady (recommended for GLM / 480B / DeepSeek / Kimi): longer same-KV warmup,
# longer timed window, long-form prompt, and EOS ignored so the model cannot
# stop after a short answer before the cache is warm.
PREFILL_TARGET_TOKENS = int(os.environ.get("PAGED_MOE_BENCH_PREFILL_TOKENS", "2048"))
DECODE_TOKENS = int(os.environ.get("PAGED_MOE_BENCH_DECODE_TOKENS", "256"))
WARMUP_TOKENS = int(os.environ.get("PAGED_MOE_BENCH_WARMUP_TOKENS", "32"))

# --steady defaults (env still wins if set explicitly before launch).
STEADY_WARMUP_TOKENS = 128
STEADY_DECODE_TOKENS = 512


# Prefer local model paths when present; otherwise download HF -> BENCH_DIR/<dir>.
#
# Each streamed tier sets ``ga_id`` to its key in the Global Agent model
# catalog. When that catalog sits next to this repo the bench pulls the tier's
# env from it, so prune / wait / sidecar match what the app runs. Without it
# the tier env is empty and the shell's PAGED_MOE_* exports apply.
# Optional ``env`` on a tier is a bench-only overlay.
LADDER: list[dict] = [
    {
        "id": 1,
        "name": "Qwen3-Coder-Next (thinking, 6-bit)",
        "dir": "coder-next-6bit-think",
        "ga_id": "coder-next-6bit-think",
        "hf": "mlx-community/Qwen3-Coder-Next-6bit",
        "hf_note": (
            "Local thinking distill preferred. If missing, downloads stock "
            "Qwen3-Coder-Next-6bit (non-thinking) as a stand-in."
        ),
        "approx_gb": 60,
        "bits": 6,
        "arch": "qwen3_next",
        "active": "~3B",
        "cache_gb": None,
        "prune": None,
        "env": {},
    },
    {
        "id": 2,
        "name": "Qwen3-235B-A22B (4-bit)",
        "dir": "qwen3-235-4bit",
        "ga_id": "qwen3-235-4bit",
        "hf": "mlx-community/Qwen3-235B-A22B-4bit",
        "hf_note": "6-bit is ~190 GB and external-only on 48 GB machines - slower than 4-bit internal.",
        "approx_gb": 123,
        "bits": 4,
        "arch": "qwen3_moe",
        "active": "~22B",
        "cache_gb": None,
        "prune": None,
        "env": {},
    },
    {
        "id": 3,
        "name": "GLM-4.7 (4-bit)",
        "dir": "glm-47-4bit",
        "ga_id": "glm-47-4bit",
        "hf": "mlx-community/GLM-4.7-4bit",
        "hf_note": "6-bit (~287 GB) is better quality if disk + SSD bandwidth allow; default is 4-bit.",
        "approx_gb": 199,
        "bits": 4,
        "arch": "glm4_moe",
        "active": "MoE (glm4)",
        "cache_gb": None,
        "prune": None,
        "env": {},
    },
    {
        "id": 4,
        "name": "Qwen3-Coder-480B-A35B (4-bit)",
        "dir": "qwen3-coder-480-4bit",
        "ga_id": "qwen3-coder-480-4bit",
        "hf": "mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit",
        "hf_note": "8-bit is ~540 GB - not useful on typical Thunderbolt SSDs.",
        "approx_gb": 270,
        "bits": 4,
        "arch": "qwen3_moe",
        "active": "~35B",
        "cache_gb": None,
        "prune": None,
        "env": {},
    },
    {
        "id": 5,
        "name": "DeepSeek-V3.2 (4-bit)",
        "dir": "deepseek-v32-4bit",
        "ga_id": "deepseek-v32-4bit",
        "hf": "mlx-community/DeepSeek-V3.2-4bit",
        "hf_note": "Largest SwitchGLU tier before Kimi; expect disk-bound decode.",
        "approx_gb": 378,
        "bits": 4,
        "arch": "deepseek_v32",
        "active": "MoE (DeepSeek V3.2)",
        "cache_gb": None,
        "prune": None,
        "env": {},
    },
    {
        "id": 6,
        "name": "Kimi-K2-Instruct (4-bit)",
        "dir": "Kimi-K2-Instruct-4bit",
        "ga_id": "kimi-k2-4bit",
        "hf": "mlx-community/Kimi-K2-Instruct-4bit",
        "hf_note": (
            "DeepSeek-V3 MoE remapped from kimi_k2. ~180 shards / ~430-580 GB. "
            "Expect ~1-2 tok/s-class decode on 48 GB; keep on the fastest SSD."
        ),
        "approx_gb": 550,
        "bits": 4,
        "arch": "kimi_k2->deepseek_v3",
        "active": "~MoE top-8 / 384 routed",
        "expect_shards": 180,
        "cache_gb": None,
        "prune": None,
        "env": {},
    },
]


_GA_ENV_CACHE: dict[str, dict] = {}


def ga_root() -> Path:
    return ROOT.parent


def _parse_ga_env_from_ts(ga_id: str) -> dict | None:
    """Fallback: read expertStreamEnv / path / numCtx out of config.ts.

    Used when ``tsx`` cannot run (sandbox, missing npx). Keeps the ladder
    usable without a second hand-maintained env map.
    """
    cfg_path = ga_root() / "src" / "proxy" / "config.ts"
    if not cfg_path.is_file():
        return None
    text = cfg_path.read_text()
    # Find `"kimi-k2-4bit": { ... },` at model-entry depth (not nested).
    m = re.search(
        rf'"{re.escape(ga_id)}"\s*:\s*\{{',
        text,
    )
    if not m:
        return None
    start = m.end() - 1
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                block = text[start : i + 1]
                break
        i += 1
    else:
        return None

    env: dict[str, str] = {}
    env_m = re.search(r"expertStreamEnv\s*:\s*\{", block)
    if env_m:
        j = env_m.end() - 1
        d = 0
        k = j
        while k < len(block):
            if block[k] == "{":
                d += 1
            elif block[k] == "}":
                d -= 1
                if d == 0:
                    env_block = block[j : k + 1]
                    break
            k += 1
        else:
            env_block = ""
        for km in re.finditer(
            r'(PAGED_MOE_[A-Z0-9_]+)\s*:\s*"([^"]*)"',
            env_block,
        ):
            env[km.group(1)] = km.group(2)

    path = None
    pm = re.search(
        r"expertStreamPath\s*:\s*`\$\{process\.env\.HOME\}(/[^`]*)`",
        block,
    )
    if pm:
        path = str(Path.home() / pm.group(1).lstrip("/"))
    else:
        pm = re.search(r'expertStreamPath\s*:\s*"([^"]+)"', block)
        if pm:
            path = pm.group(1).replace("${process.env.HOME}", str(Path.home()))

    num_ctx = None
    nm = re.search(r"numCtx\s*:\s*(\d+)", block)
    if nm:
        num_ctx = int(nm.group(1))
        env.setdefault("PAGED_MOE_RESERVE_CTX", str(num_ctx))

    # Defaults the app always adds on top of the catalog entry.
    home = str(Path.home())
    env.setdefault(
        "PAGED_MOE_SIDECAR_DIR",
        f"{home}/.global_agent/expert_sidecar",
    )
    env.setdefault("PAGED_MOE_SIDECAR_DEBUG", "1")

    return {
        "id": ga_id,
        "path": path,
        "numCtx": num_ctx,
        "env": env,
        "source": "config.ts",
    }


def fetch_ga_catalog(ga_id: str) -> dict | None:
    """Pull expertStreamEnv / path / numCtx from the catalog via tsx."""
    if ga_id in _GA_ENV_CACHE:
        return _GA_ENV_CACHE[ga_id]
    script = ga_root() / "scripts" / "dump-paged-moe-env.ts"
    data = None
    if script.is_file():
        tsx = shutil.which("tsx")
        if tsx:
            cmd = [tsx, str(script), ga_id]
        else:
            npx = shutil.which("npx")
            cmd = [npx, "--yes", "tsx", str(script), ga_id] if npx else None
        if cmd:
            try:
                out = subprocess.check_output(
                    cmd,
                    cwd=str(ga_root()),
                    text=True,
                    stderr=subprocess.PIPE,
                )
                data = json.loads(out)
                data["source"] = "tsx"
            except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
                err = getattr(e, "stderr", None) or str(e)
                print(
                    f"[warn] catalog tsx dump failed for {ga_id}; "
                    f"falling back to config.ts parse ({str(err)[:120]})",
                    flush=True,
                )
    if data is None:
        data = _parse_ga_env_from_ts(ga_id)
        if data is None:
            print(f"[bench] no model catalog for {ga_id}, using shell env", flush=True)
            return None
    _GA_ENV_CACHE[ga_id] = data
    return data


def resolve_tier_env(m: dict) -> dict[str, str]:
    """Catalog env first, then optional ladder overrides."""
    env: dict[str, str] = {}
    ga_id = m.get("ga_id")
    if ga_id:
        dumped = fetch_ga_catalog(ga_id)
        if dumped and isinstance(dumped.get("env"), dict):
            env.update({str(k): str(v) for k, v in dumped["env"].items()})
            src = dumped.get("source", "?")
            print(
                f"[bench] catalog {ga_id} ({src}): "
                f"{len(dumped['env'])} env keys"
                + (f", path={dumped.get('path')}" if dumped.get("path") else ""),
                flush=True,
            )
        else:
            print(
                f"[bench] no catalog env for {ga_id}; "
                "tier runs on the shell's PAGED_MOE_* (exact if none set)",
                flush=True,
            )
    env.update({str(k): str(v) for k, v in (m.get("env") or {}).items()})
    return env


def tier_prune(m: dict, env: dict[str, str] | None = None) -> float:
    if m.get("prune") is not None:
        return float(m["prune"])
    e = env if env is not None else resolve_tier_env(m)
    raw = e.get("PAGED_MOE_PRUNE")
    if raw is not None and str(raw).strip() != "":
        return float(raw)
    return 0.0


def tier(n: int) -> dict:
    for m in LADDER:
        if m["id"] == n:
            return m
    raise SystemExit(f"unknown tier {n}; valid: 1..{len(LADDER)}")


def pause(msg: str) -> None:
    print(f"\n>>> {msg}")
    try:
        input("    Press Enter to continue (Ctrl-C to abort)... ")
    except EOFError:
        pass


def human_gb(n: float | int | None) -> str:
    if n is None:
        return "n/a"
    return f"{float(n):.2f} GB"


def probe_machine() -> dict:
    ram = None
    chip = None
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        ram = int(out)
    except Exception:
        pass
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        pass
    if not chip:
        try:
            sp = subprocess.check_output(
                ["system_profiler", "SPHardwareDataType"], text=True
            )
            m = re.search(r"Chip:\s*(.+)", sp)
            chip = m.group(1).strip() if m else None
        except Exception:
            pass
    disk = shutil.disk_usage(str(BENCH_DIR if BENCH_DIR.exists() else Path.home()))
    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "chip": chip or platform.processor() or "unknown",
        "ram_bytes": ram,
        "ram_gb": round(ram / (1 << 30), 2) if ram else None,
        "python": sys.version.split()[0],
        "bench_dir": str(BENCH_DIR),
        "bench_dir_free_gb": round(disk.free / (1 << 30), 2),
        "bench_dir_total_gb": round(disk.total / (1 << 30), 2),
    }
    try:
        from pagedmoe.machine import get_provider

        p = get_provider()
        out["envelope"] = {
            "catalog_id": p.catalog_id,
            "max_cache_gb": p.envelope.max_cache_gb,
            "ram_budget_gb": p.envelope.ram_budget_gb,
            "read_threads": p.envelope.read_threads,
            "disk_gb_s": p.hardware.disk_gb_s,
            "gpu_cores": p.hardware.gpu_cores,
            "near_baseline": p.envelope.near_baseline,
        }
        if p.hardware.disk_gb_s:
            out["disk_gb_s"] = p.hardware.disk_gb_s
    except Exception:
        pass
    return out


def model_path(m: dict) -> Path:
    primary = (BENCH_DIR / m["dir"]).expanduser()
    if primary.is_dir():
        return primary
    ga_id = m.get("ga_id")
    if ga_id:
        dumped = fetch_ga_catalog(ga_id)
        path = (dumped or {}).get("path")
        if path:
            p = Path(str(path)).expanduser()
            if p.is_dir():
                return p
    return primary


def model_ready(m: dict) -> bool:
    p = model_path(m)
    if not (p.is_dir() and any(p.glob("*.safetensors"))):
        return False
    expect = m.get("expect_shards")
    if expect:
        n = len(list(p.glob("model-*.safetensors")))
        if n < int(expect):
            return False
    return True


def ensure_model(m: dict, yes: bool) -> Path:
    dest = model_path(m)
    if model_ready(m):
        print(f"[ok] local model present: {dest}")
        return dest

    # Partial download (e.g. Kimi mid-hf): refuse to bench until shard count matches.
    expect = m.get("expect_shards")
    if dest.is_dir() and expect:
        n = len(list(dest.glob("model-*.safetensors")))
        if 0 < n < int(expect):
            raise SystemExit(
                f"[incomplete] {dest} has {n}/{expect} shards - "
                "finish the hf download before benching this tier"
            )

    # Tier 1 special-case: thinking distill may live under the usual name
    # even when BENCH_DIR differs - also accept ~/mlx-models default.
    alt = Path.home() / "mlx-models" / m["dir"]
    if alt.is_dir() and any(alt.glob("*.safetensors")):
        if not expect or len(list(alt.glob("model-*.safetensors"))) >= int(expect or 0):
            print(f"[ok] using existing model at {alt}")
            return alt

    need = m["approx_gb"]
    free = shutil.disk_usage(str(BENCH_DIR if BENCH_DIR.exists() else Path.home())).free / (
        1 << 30
    )
    print(f"[download] {m['hf']}")
    print(f"           -> {dest}")
    print(f"           ~{need} GB needed, {free:.1f} GB free on target volume")
    if m.get("hf_note"):
        print(f"           note: {m['hf_note']}")
    if free < need + 5:
        print(
            f"[warn] free space ({free:.1f} GB) is below ~{need + 5} GB. "
            "Delete other checkpoints, then continue."
        )
    if not yes:
        pause("Ready to download? Free disk first if the warning above applies.")

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "hf",
        "download",
        m["hf"],
        "--local-dir",
        str(dest),
    ]
    print("+", " ".join(cmd), flush=True)
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        # huggingface_hub CLI entrypoint variants
        cmd = [
            sys.executable,
            "-m",
            "huggingface_hub.commands.huggingface_cli",
            "download",
            m["hf"],
            "--local-dir",
            str(dest),
        ]
        print("+", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)
    if not model_ready(m):
        raise SystemExit(f"download finished but model not ready under {dest}")
    return dest


def apply_env(m: dict) -> None:
    """Push this tier's knobs into os.environ *and* config attrs.

    ``pagedmoe.config`` reads env once at import. Setting os.environ alone
    after that is a no-op for PRUNE / SIDECAR / WAIT_ABOVE / etc., which is why
    earlier ladder runs silently stayed at prune=0 even when env was listed.

    Env comes from the model catalog when one is found for the tier's
    ``ga_id``. Ladder ``env`` is only an optional overlay.

    Every tier also pins LOOKUP=0 and SIDECAR_HEAD_PREFILL=0: both lose on the
    workloads this ladder measures, and both have been left on by a stale
    shell export before.
    """
    from pagedmoe import config as es_config

    env = resolve_tier_env(m)
    if m.get("prune") is not None:
        env.setdefault("PAGED_MOE_PRUNE", str(m["prune"]))
    if m.get("cache_gb") is not None:
        env.setdefault("PAGED_MOE_CACHE_GB", str(m["cache_gb"]))
    # Hard pins: do not let a prior export or a stale tier block override these.
    env["PAGED_MOE_LOOKUP"] = "0"
    env["PAGED_MOE_SIDECAR_HEAD_PREFILL"] = "0"

    applied: list[str] = []
    for key, raw in env.items():
        val = str(raw)
        os.environ[key] = val
        if not key.startswith("PAGED_MOE_"):
            continue
        attr = key[len("PAGED_MOE_") :]
        if not hasattr(es_config, attr):
            continue
        cur = getattr(es_config, attr)
        if isinstance(cur, bool):
            setattr(es_config, attr, bool(int(val)))
        elif isinstance(cur, int):
            setattr(es_config, attr, int(float(val)))
        elif isinstance(cur, float):
            setattr(es_config, attr, float(val))
        elif cur is None and attr == "CACHE_GB":
            setattr(es_config, attr, float(val))
        else:
            setattr(es_config, attr, val)
        applied.append(f"{attr}={getattr(es_config, attr)!r}")

    # Fail loud if a pin did not stick - a silent on is worse than a crash.
    if getattr(es_config, "LOOKUP", False):
        raise SystemExit(
            "[bench] LOOKUP is on after apply_env; refuse to publish non-optimal numbers"
        )
    if getattr(es_config, "SIDECAR_HEAD_PREFILL", False):
        raise SystemExit(
            "[bench] SIDECAR_HEAD_PREFILL is on after apply_env; "
            "refuse to publish non-optimal numbers"
        )

    prune = tier_prune(m, env)
    m["_resolved_prune"] = prune
    print(
        f"[bench] knobs: prune={es_config.PRUNE} "
        f"wait_above={es_config.WAIT_ABOVE} sidecar={int(es_config.SIDECAR)} "
        f"cache_gb={es_config.CACHE_GB} "
        f"lookup={int(es_config.LOOKUP)} "
        f"prefill_head={int(es_config.SIDECAR_HEAD_PREFILL)}",
        flush=True,
    )
    if applied:
        print(f"[bench] patched config: {', '.join(applied)}", flush=True)


def header_sizes(path: Path) -> dict:
    from pagedmoe.safetensors_index import read_headers

    # Lightweight: headers only. Backbone/expert split comes from load() info.
    index = read_headers(path)  # dict[str, TensorLoc]
    total = sum(t.nbytes for t in index.values())
    return {"checkpoint_gb": round(total / 1e9, 2), "n_tensors": len(index)}


def try_resident(path: Path) -> dict:
    from pagedmoe.loader import load

    t0 = time.perf_counter()
    try:
        model, _tok = load(str(path), mode="resident", verbose=False)
        del model
        return {
            "resident_ok": True,
            "resident_error": None,
            "resident_attempt_s": round(time.perf_counter() - t0, 2),
        }
    except Exception as e:
        return {
            "resident_ok": False,
            "resident_error": f"{type(e).__name__}: {e}",
            "resident_attempt_s": round(time.perf_counter() - t0, 2),
        }


def run_streamed(path: Path, m: dict, *, steady: bool = False) -> dict:
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    from pagedmoe.loader import get_stats, load, relieve_pressure

    warmup_n = WARMUP_TOKENS
    decode_n_target = DECODE_TOKENS
    if steady:
        # Env overrides still apply when the user set them; otherwise use the
        # longer steady defaults so disk-bound MoEs can finish warming.
        if "PAGED_MOE_BENCH_WARMUP_TOKENS" not in os.environ:
            warmup_n = STEADY_WARMUP_TOKENS
        if "PAGED_MOE_BENCH_DECODE_TOKENS" not in os.environ:
            decode_n_target = STEADY_DECODE_TOKENS

    apply_env(m)
    t_load = time.perf_counter()
    model, tokenizer = load(
        str(path),
        mode="streamed",
        cache_gb=m.get("cache_gb"),
        verbose=True,
    )
    load_s = time.perf_counter() - t_load
    info = dict(getattr(model, "_pagedmoe_info", {}))

    # Long-form ask: a "one short sentence" prompt made GLM hit EOS at ~80
    # tokens and never reach a warm decode window.
    filler = (ROOT / "pagedmoe" / "cache.py").read_text()
    body = (filler * 40)[: PREFILL_TARGET_TOKENS * 5]
    if steady:
        user = (
            "You are writing a long technical walkthrough of the code below. "
            "Keep going for many paragraphs: cover classes, methods, data "
            "flow, edge cases, and performance notes. Do not stop early; do "
            "not summarize in one sentence.\n\n```python\n"
            + body
            + "\n```"
        )
    else:
        user = (
            "Walk through this code in detail - classes, methods, and how "
            "data moves. Keep writing until you hit the length limit; do not "
            "stop after one sentence.\n\n```python\n" + body + "\n```"
        )
    messages = [
        {"role": "system", "content": "You are a thorough coding assistant."},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if len(prompt) > PREFILL_TARGET_TOKENS + 256:
        prompt = prompt[:PREFILL_TARGET_TOKENS]

    sampler = make_sampler(temp=0.0)

    # Force length-limited runs: mlx_lm stops when it samples an EOS id.
    # Steady (and default) benches need the full warmup+decode window.
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    tokenizer.eos_token_ids = set()

    from mlx_lm.models.cache import make_prompt_cache

    # One continuous generation on the real prompt / KV:
    #   token[0]                         -> ends prefill (TTFT)
    #   tokens[0 .. warmup)              -> discarded (working-set rebuild)
    #   tokens[warmup .. warmup+decode)  -> wall-clock decode tok/s
    total_gen = warmup_n + decode_n_target
    if warmup_n < 1 or decode_n_target < 1:
        raise SystemExit("need warmup >= 1 and decode >= 1")

    print(
        f"[streamed] protocol={'steady' if steady else 'default'} "
        f"warmup={warmup_n} decode={decode_n_target} "
        f"(ignore EOS, same-KV continuous gen)",
        flush=True,
    )

    kv = make_prompt_cache(model)
    mx.reset_peak_memory()
    stats0 = get_stats(model).get("cache", {})
    read0 = float(stats0.get("read_gb", 0) or 0)

    t0 = time.perf_counter()
    token_t: list[float] = []
    text: list[str] = []
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
        text.append(getattr(resp, "text", "") or "")
        last_resp = resp
        n = len(token_t)
        if n >= next_mark or n == total_gen:
            phase = (
                "prefill-done"
                if n == 1
                else ("warmup" if n <= warmup_n else "timed")
            )
            elapsed = token_t[-1] - t0
            print(
                f"[streamed] {phase} tokens={n}/{total_gen} "
                f"elapsed={elapsed:.1f}s "
                f"ema={getattr(resp, 'generation_tps', 0):.2f} tok/s",
                flush=True,
            )
            next_mark = n + 32

    if len(token_t) < warmup_n + 1:
        raise RuntimeError(
            f"stream_generate produced {len(token_t)} tokens; "
            f"need at least warmup+1 ({warmup_n + 1}). "
            "If this keeps happening, pass --steady and check EOS handling."
        )

    prompt_tokens = len(prompt)
    prefill_s = token_t[0] - t0

    timed = token_t[warmup_n:]
    decode_n = len(timed)
    decode_s = max(token_t[-1] - token_t[warmup_n - 1], 1e-9)
    decode_tok_s = decode_n / decode_s

    half = max(decode_n // 2, 1)
    steady_start = len(token_t) - half - 1
    steady_s = max(token_t[-1] - token_t[steady_start], 1e-9)
    steady_tok_s = half / steady_s

    # Last-quartile window: closest to "picked up the pace" for big MoEs.
    q = max(decode_n // 4, 1)
    q_start = len(token_t) - q - 1
    q_s = max(token_t[-1] - token_t[q_start], 1e-9)
    late_tok_s = q / q_s

    mlx_gen_tps = getattr(last_resp, "generation_tps", None)

    stats = get_stats(model)
    cache = stats.get("cache", {})
    read1 = float(cache.get("read_gb", 0) or 0)
    read_delta = max(0.0, read1 - read0)
    miss_gb_tok = round(read_delta / max(decode_n, 1), 4) if decode_n else None

    peak = mx.get_peak_memory() / 1e9
    active = mx.get_active_memory() / 1e9
    relieve_pressure(model)

    return {
        "load_s": round(load_s, 2),
        "info": info,
        "prompt_tokens": prompt_tokens,
        "prefill_s": round(prefill_s, 2),
        "prefill_tok_s": round(prompt_tokens / max(prefill_s, 1e-9), 2),
        "warmup_tokens": warmup_n,
        "decode_tokens": decode_n,
        "decode_s": round(decode_s, 2),
        "decode_tok_s": round(decode_tok_s, 2),
        "decode_tok_s_steady": round(steady_tok_s, 2),
        "decode_tok_s_late": round(late_tok_s, 2),
        "mlx_generation_tps": (
            None if mlx_gen_tps is None else round(float(mlx_gen_tps), 2)
        ),
        "ttft_s": round(prefill_s, 2),
        "peak_gb": round(peak, 2),
        "active_gb": round(active, 2),
        "cache": cache,
        "read_gb_during_timed": round(read_delta, 3),
        "miss_gb_per_decode_token_est": miss_gb_tok,
        "effective_disk_gb_s": round(
            read_delta / max(decode_s + prefill_s, 1e-9), 2
        ),
        "sample_text": "".join(text)[:240].replace("\n", " "),
        "protocol": {
            "mode": "steady" if steady else "default",
            "prefill_target_tokens": PREFILL_TARGET_TOKENS,
            "warmup_tokens": warmup_n,
            "decode_tokens": decode_n_target,
            "ignore_eos": True,
            "note": (
                "decode_tok_s = post-warmup wall-clock on same KV; "
                "decode_tok_s_steady = last half; "
                "decode_tok_s_late = last quartile (pace after warm)"
            ),
        },
    }


def render_section(machine: dict, m: dict, path: Path, sizes: dict, resident: dict, streamed: dict | None, err: str | None) -> str:
    ram_gb = machine.get("ram_gb")
    budget = (streamed or {}).get("info", {}).get("ram_budget_gb")
    ckpt = sizes.get("checkpoint_gb") or m["approx_gb"]
    headroom = None
    if budget and ckpt:
        headroom = round(float(ckpt) / float(budget), 2)

    lines = [
        f"## Tier {m['id']}: {m['name']}",
        "",
        f"_Recorded {machine.get('timestamp_utc')} on `{machine.get('hostname')}` "
        f"({machine.get('chip')}, {human_gb(ram_gb)} unified)._",
        "",
        "### Identity",
        "",
        f"| Field | Value |",
        f"| --- | --- |",
        f"| Local path | `{path}` |",
        f"| HF artifact | `{m['hf']}` |",
        f"| Quant | {m['bits']}-bit |",
        f"| Arch | `{m['arch']}` |",
        f"| Active (nominal) | {m['active']} |",
        f"| Checkpoint (headers) | {human_gb(ckpt)} |",
        f"| Approx. catalog size | ~{m['approx_gb']} GB |",
        "",
        "### Without paged-moe (resident)",
        "",
    ]
    if resident.get("resident_ok"):
        lines += [
            f"**Loaded fully resident** in {resident['resident_attempt_s']}s. "
            "This machine has enough RAM that streaming is optional for this checkpoint.",
            "",
        ]
    else:
        lines += [
            f"**Cannot run resident** on this machine.",
            "",
            f"```",
            f"{resident.get('resident_error')}",
            f"```",
            "",
            f"Headroom ratio (checkpoint / RAM budget): "
            f"**{headroom if headroom is not None else 'n/a'}x** "
            f"({human_gb(ckpt)} vs budget {human_gb(budget)}).",
            "",
        ]

    lines += ["### With paged-moe (streamed)", ""]
    if err:
        lines += [f"**Streamed run failed:** `{err}`", ""]
        return "\n".join(lines) + "\n"

    assert streamed is not None
    info = streamed.get("info") or {}
    cache = streamed.get("cache") or {}
    lines += [
        f"| Metric | Value |",
        f"| --- | ---: |",
        f"| Load | {streamed['load_s']} s |",
        f"| Backbone / experts | {info.get('backbone_gb', 'n/a')} / {info.get('expert_gb', 'n/a')} GB |",
        f"| Cache budget | {info.get('cache_gb', m.get('cache_gb'))} GB |",
        f"| Prune | {m.get('_resolved_prune', m.get('prune', 0))} |",
        f"| Catalog id | `{m.get('ga_id') or 'n/a'}` |",
        f"| Lookup / prefill head | off / off (shipped optimal) |",
        f"| Slab | {info.get('slab_gb', 'off')} GB "
        f"({info.get('slab_slots', 0)} slots) |",
        f"| Prefill | {streamed['prompt_tokens']} tok @ "
        f"{streamed['prefill_tok_s']} tok/s ({streamed['prefill_s']} s) |",
        f"| Warmup (discarded) | {streamed.get('warmup_tokens', '?')} tok |",
        f"| Decode | {streamed['decode_tokens']} tok @ "
        f"**{streamed['decode_tok_s']} tok/s** ({streamed['decode_s']} s) |",
        f"| Decode steady (last half) | "
        f"{streamed.get('decode_tok_s_steady', 'n/a')} tok/s |",
        f"| Decode late (last quartile) | "
        f"{streamed.get('decode_tok_s_late', 'n/a')} tok/s |",
        f"| mlx generation_tps | "
        f"{streamed.get('mlx_generation_tps', 'n/a')} |",
        f"| Protocol | {(streamed.get('protocol') or {}).get('mode', 'default')} "
        f"(warmup={streamed.get('warmup_tokens')}, "
        f"ignore_eos="
        f"{(streamed.get('protocol') or {}).get('ignore_eos', False)}) |",
        f"| Peak Metal | {streamed['peak_gb']} GB |",
        f"| Active after | {streamed['active_gb']} GB |",
        f"| Hit rate | {cache.get('hit_rate', 'n/a')} |",
        f"| Disk read (timed) | {streamed['read_gb_during_timed']} GB |",
        f"| Eff. disk bandwidth | {streamed['effective_disk_gb_s']} GB/s |",
        f"| Est. miss GB/decode-tok | {streamed.get('miss_gb_per_decode_token_est')} |",
        "",
        "### Upside vs no streaming",
        "",
        (
            f"Resident is impossible ({human_gb(ckpt)} checkpoint on "
            f"{human_gb(ram_gb)} unified / ~{human_gb(budget)} usable). "
            if not resident.get("resident_ok")
            else "Resident also works; streaming is optional. "
        )
        + (
            f"Streamed decode held **{streamed['decode_tok_s']} tok/s** at "
            f"**{streamed['peak_gb']} GB** peak - "
            f"{'well under' if streamed['peak_gb'] < (ram_gb or 99) else 'near'} "
            f"physical RAM."
        ),
        "",
        f"Sample output: `{streamed.get('sample_text', '')}`",
        "",
        "<details><summary>Raw JSON</summary>",
        "",
        "```json",
        json.dumps(
            {"machine": machine, "model": m, "sizes": sizes, "resident": resident, "streamed": streamed},
            indent=2,
            default=str,
        )[:12000],
        "```",
        "",
        "</details>",
        "",
    ]
    return "\n".join(lines) + "\n"


def upsert_doc(section: str, tier_id: int) -> None:
    text = DOC.read_text() if DOC.exists() else ""
    start = f"## Tier {tier_id}:"
    if start in text:
        # Replace from this tier header through the next tier header or EOF.
        pattern = re.compile(
            rf"## Tier {tier_id}:.*?(?=\n## Tier \d+:|\Z)",
            re.S,
        )
        # Callable repl: section JSON can contain \u escapes; a string repl
        # would be parsed as a template and raise re.error.
        repl = section.rstrip() + "\n\n"
        text = pattern.sub(lambda _: repl, text)
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + section
    DOC.write_text(text)


def update_machine_table(machine: dict) -> None:
    if not DOC.exists():
        return
    text = DOC.read_text()
    # Light touch: ensure the reference block mentions last probe.
    marker = "<!-- AUTO_MACHINE -->"
    block = (
        f"{marker}\n"
        f"Last probe: {machine.get('timestamp_utc')} - "
        f"{machine.get('chip')}, {human_gb(machine.get('ram_gb'))}, "
        f"free on bench volume {machine.get('bench_dir_free_gb')} GB "
        f"(`{machine.get('bench_dir')}`).\n"
    )
    if marker in text:
        repl = block + "\n"
        text = re.sub(
            rf"{re.escape(marker)}.*?(?=\n## |\n---|\Z)",
            lambda _: repl,
            text,
            count=1,
            flags=re.S,
        )
    else:
        text = text.replace(
            "## Reference machine (author)",
            "## Reference machine (author)\n\n" + block,
            1,
        )
    DOC.write_text(text)


def cmd_list() -> None:
    print(f"{'#':<3} {'dir':<28} {'catalog':<22} {'approx':>7} {'bits':>4}  name")
    for m in LADDER:
        ready = "*" if model_ready(m) else " "
        print(
            f"{m['id']:<3} {m['dir']:<28} {str(m.get('ga_id') or '-'):<22} "
            f"~{m['approx_gb']:>4}G {m['bits']:>4}  "
            f"[{ready}] {m['name']}"
        )


def cmd_status() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for m in LADDER:
        js = RESULTS / f"tier_{m['id']}.json"
        local = "local" if model_ready(m) else "missing"
        if js.exists():
            data = json.loads(js.read_text())
            d = data.get("streamed", {})
            print(
                f"tier {m['id']}: DONE  decode={d.get('decode_tok_s')} tok/s  "
                f"peak={d.get('peak_gb')} GB  model={local}  ({js.name})"
            )
        else:
            print(f"tier {m['id']}: pending  model={local}  {m['name']}")


def cmd_run(
    n: int,
    yes: bool,
    skip_resident: bool,
    force: bool,
    steady: bool,
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS / f"tier_{n}.json"
    if out_json.exists() and not force:
        print(f"[skip] {out_json} exists (pass --force to re-run)")
        print("       Use: ./bench/memory_benchmark.sh status")
        return

    m = tier(n)
    machine = probe_machine()
    update_machine_table(machine)
    print(json.dumps({"machine": machine, "tier": m["id"], "name": m["name"]}, indent=2))

    if not yes:
        pause(
            f"Tier {n}: about to ensure model (~{m['approx_gb']} GB). "
            "Delete other large checkpoints now if you need room."
        )
    path = ensure_model(m, yes=yes)

    sizes = header_sizes(path)
    print(f"[headers] checkpoint ~ {sizes['checkpoint_gb']} GB")

    resident = {"resident_ok": None, "resident_error": "skipped", "resident_attempt_s": 0}
    if not skip_resident:
        if not yes:
            pause("Next: attempt resident load (expected to fail for this ladder).")
        print("[resident] attempting full load...", flush=True)
        resident = try_resident(path)
        print(json.dumps(resident, indent=2))

    streamed = None
    err = None
    if not yes:
        pause("Next: streamed load + timed prefill/decode.")
    print("[streamed] loading + measuring...", flush=True)
    try:
        streamed = run_streamed(path, m, steady=steady)
        print(
            f"[streamed] decode {streamed['decode_tok_s']} tok/s  "
            f"(steady {streamed.get('decode_tok_s_steady')}  "
            f"late {streamed.get('decode_tok_s_late')}  "
            f"mlx {streamed.get('mlx_generation_tps')})  "
            f"prefill {streamed['prefill_tok_s']} tok/s  "
            f"peak {streamed['peak_gb']} GB",
            flush=True,
        )
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[streamed] FAILED: {err}", flush=True)

    payload = {
        "machine": machine,
        "model": {k: v for k, v in m.items()},
        "path": str(path),
        "sizes": sizes,
        "resident": resident,
        "streamed": streamed,
        "error": err,
        # Self-document that this run used the shipped-optimal pins. A result
        # JSON without these (or with either True) is not comparable.
        "optimal_pins": {
            "PAGED_MOE_LOOKUP": "0",
            "PAGED_MOE_SIDECAR_HEAD_PREFILL": "0",
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    section = render_section(machine, m, path, sizes, resident, streamed, err)
    upsert_doc(section, m["id"])
    print(f"[doc] updated {DOC}")
    print(f"[json] {out_json}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["list", "status", "run"])
    ap.add_argument("tier", nargs="?", type=int)
    ap.add_argument("-y", "--yes", action="store_true", help="no interactive pauses")
    ap.add_argument("--skip-resident", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even if results exist")
    ap.add_argument(
        "--steady",
        action="store_true",
        help=(
            "longer same-KV warmup (128) + timed decode (512), long-form prompt, "
            "ignore EOS - use for GLM / 480B / DeepSeek so tok/s reflects a "
            "warm expert cache, not cold ramp / early stop"
        ),
    )
    args = ap.parse_args()

    if args.command == "list":
        cmd_list()
    elif args.command == "status":
        cmd_status()
    else:
        if not args.tier:
            raise SystemExit("usage: memory_benchmark.py run <1-5> [--steady]")
        cmd_run(
            args.tier,
            yes=args.yes,
            skip_resident=args.skip_resident,
            force=args.force,
            steady=args.steady,
        )


if __name__ == "__main__":
    main()
