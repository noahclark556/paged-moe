# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-machine model ladder: fit math + IO-scaled tok/s where we have a baseline.

Backbone / checkpoint GB for the big MoEs are measured (MEMORY_BENCHMARK.md).
Smaller 16 GB-class MoEs use published 4-bit sizes and an estimated backbone.
Prefill/decode for non-host machines scale from the 48 GB M5 Pro numbers by
disk bandwidth and sqrt(cache). GPU FLOPs still come from the host in a
sandbox, so those tok/s figures are IO estimates, not a slower-chip bench.
"""

from __future__ import annotations

from .scale import BASELINE, BASELINE_DISK_GB_S
from .types import Envelope, FitResult

# ctx used when a model does not name one. Matches common defaults for big MoEs.
_DEFAULT_CTX = 8192

LADDER: list[dict] = [
    {
        "id": "olmoe-1b-7b-4bit",
        "name": "OLMoE-1B-7B (4-bit)",
        "hf": "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit",
        "checkpoint_gb": 4.2,
        "backbone_gb": 0.8,
        "expert_gb": 3.4,
        "backbone_source": "estimated",
        "kv_kb_token": 40,
        "recommended_ctx": 8192,
        "disk_bound": False,
    },
    {
        "id": "qwen3-30b-a3b-4bit",
        "name": "Qwen3-30B-A3B (4-bit)",
        "hf": "mlx-community/Qwen3-30B-A3B-4bit",
        "checkpoint_gb": 18.0,
        "backbone_gb": 2.0,
        "expert_gb": 16.0,
        "backbone_source": "estimated",
        "kv_kb_token": 50,
        "recommended_ctx": 16384,
        "disk_bound": False,
    },
    {
        "id": "mixtral-8x7b-4bit",
        "name": "Mixtral-8x7B (4-bit)",
        "hf": "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
        "checkpoint_gb": 24.0,
        "backbone_gb": 5.0,
        "expert_gb": 19.0,
        "backbone_source": "estimated",
        "kv_kb_token": 80,
        "recommended_ctx": 8192,
        "disk_bound": False,
    },
    {
        "id": "coder-next-6bit",
        "name": "Qwen3-Coder-Next (6-bit)",
        "hf": "mlx-community/Qwen3-Coder-Next-6bit",
        "checkpoint_gb": 64.75,
        "backbone_gb": 1.94,
        "expert_gb": 62.81,
        "backbone_source": "measured",
        "kv_kb_token": 50,
        "recommended_ctx": 32768,
        "disk_bound": True,
        # 48 GB M5 Pro ladder (cold 2k prefill, steady decode).
        "measured_prefill_tok_s": 218.0,
        "measured_decode_tok_s": 23.1,
        "measured_peak_gb": 30.7,
        "measured_cache_gb": 23.4,
    },
    {
        "id": "qwen3-235b-4bit",
        "name": "Qwen3-235B-A22B (4-bit)",
        "hf": "mlx-community/Qwen3-235B-A22B-4bit",
        "checkpoint_gb": 132.24,
        "backbone_gb": 4.5,
        "expert_gb": 127.74,
        "backbone_source": "measured",
        "kv_kb_token": 102,
        "recommended_ctx": 65536,
        "disk_bound": True,
        "measured_prefill_tok_s": 92.5,
        "measured_decode_tok_s": 9.1,
        "measured_peak_gb": 28.6,
        "measured_cache_gb": 19.87,
    },
    {
        "id": "glm-47-4bit",
        "name": "GLM-4.7 (4-bit)",
        "hf": "mlx-community/GLM-4.7-4bit",
        "checkpoint_gb": 198.56,
        "backbone_gb": 9.58,
        "expert_gb": 188.98,
        "backbone_source": "measured",
        "kv_kb_token": 196,
        "recommended_ctx": 24576,
        "disk_bound": True,
        "measured_prefill_tok_s": 72.5,
        "measured_decode_tok_s": 7.0,
        "measured_peak_gb": 28.9,
        "measured_cache_gb": 14.5,
    },
    # {
    #     "id": "qwen3-coder-480-4bit",
    #     "name": "Qwen3-Coder-480B (4-bit)",
    #     "hf": "mlx-community/Qwen3-Coder-480B-A35B-Instruct-4bit",
    #     "checkpoint_gb": 270.09,
    #     "backbone_gb": 6.79,
    #     "expert_gb": 263.3,
    #     "backbone_source": "measured",
    #     "kv_kb_token": 100,
    #     "recommended_ctx": 65536,
    #     "disk_bound": True,
    #     "measured_prefill_tok_s": 80.0,
    #     "measured_decode_tok_s": 5.0,
    #     "measured_peak_gb": 34.0,
    #     "measured_cache_gb": 22.53,
    # },
    # {
    #     "id": "deepseek-v32-4bit",
    #     "name": "DeepSeek-V3.2 (4-bit)",
    #     "hf": "mlx-community/DeepSeek-V3.2-4bit",
    #     "checkpoint_gb": 378.09,
    #     "backbone_gb": 10.26,
    #     "expert_gb": 367.82,
    #     "backbone_source": "measured",
    #     "kv_kb_token": 80,
    #     "recommended_ctx": 65536,
    #     "disk_bound": True,
    #     "measured_prefill_tok_s": 45.0,
    #     "measured_decode_tok_s": 6.0,
    #     "measured_peak_gb": 33.0,
    #     "measured_cache_gb": 16.5,
    # },
    {
        "id": "kimi-k2-4bit",
        "name": "Kimi-K2-Instruct (4-bit)",
        "hf": "mlx-community/Kimi-K2-Instruct-4bit",
        "checkpoint_gb": 577.59,
        "backbone_gb": 6.83,
        "expert_gb": 570.76,
        "backbone_source": "measured",
        "kv_kb_token": 50,
        "recommended_ctx": 65536,
        "disk_bound": True,
        "measured_prefill_tok_s": 27.0,
        "measured_decode_tok_s": 8.75,
        "measured_peak_gb": 31.59,
        "measured_cache_gb": 19.74,
    },
    {
        "id": "kat-coder-v25",
        "name": "KAT-Coder-V2.5 (6-bit)",
        "hf": "leonsarmiento/KAT-Coder-V2.5-Dev-6bit-XL-mlx",
        "checkpoint_gb": 30.24,
        "backbone_gb": 4.07,
        "expert_gb": 26.17,
        "backbone_source": "estimated",
        "kv_kb_token": 50,
        "recommended_ctx": 32768,  # typical long-ctx default for this size
        "disk_bound": False,
        "measured_prefill_tok_s": 420.0,
        "measured_decode_tok_s": 33.3,
        "measured_peak_gb": 29.8,
        "measured_cache_gb": 22.08,
    },
]


def _bytes_gb(n: float) -> float:
    return float(n) / 1e9


def fit_model(
    model: dict,
    envelope: Envelope,
    *,
    ctx: int | None = None,
    storage_free_gb: float | None = None,
) -> FitResult:
    """Pure fit check matching loader.py budget math (GiB RAM, decimal GB sizes)."""
    ram_bytes = int(envelope.ram_gb * (1 << 30))
    budget_bytes = int(envelope.ram_fraction * ram_bytes)
    budget_gb = _bytes_gb(budget_bytes)
    backbone = float(model["backbone_gb"])
    total = float(model["checkpoint_gb"])
    expert = float(model.get("expert_gb") or max(0.0, total - backbone))
    ctx_n = int(
        ctx if ctx is not None else model.get("recommended_ctx") or _DEFAULT_CTX
    )
    kv_kb = float(model.get("kv_kb_token") or 0)
    kv_gb = (kv_kb * 1024 * ctx_n) / 1e9 if kv_kb else 0.0
    activation_gb = float(envelope.activation_gb)
    if kv_gb > 0:
        headroom_gb = activation_gb + kv_gb * 1.25  # KV_STORE_SLACK default
    else:
        headroom_gb = float(envelope.headroom_gb)
    backbone_bytes = int(backbone * 1e9)
    headroom_bytes = int(headroom_gb * (1 << 30))
    min_cache = 1 << 30

    name = str(model["name"])
    mid = str(model["id"])
    src = str(model.get("backbone_source") or "estimated")
    hf = model.get("hf")

    if storage_free_gb is not None and total > float(storage_free_gb) + 2:
        return FitResult(
            model_id=mid,
            name=name,
            status="disk",
            checkpoint_gb=total,
            backbone_gb=backbone,
            expert_gb=expert,
            ram_budget_gb=round(budget_gb, 2),
            cache_gb=0.0,
            peak_gb=0.0,
            gain_gb=0.0,
            recommended_ctx=ctx_n,
            reason=(
                f"checkpoint {total:.0f} GB does not fit on this volume "
                f"({storage_free_gb:.0f} GB free)"
            ),
            backbone_source=src,
            hf=hf,
        )

    # Resident if the whole checkpoint plus headroom sits in the budget.
    if int(total * 1e9) + headroom_bytes <= budget_bytes:
        peak = round(total + activation_gb, 1)
        return FitResult(
            model_id=mid,
            name=name,
            status="resident",
            checkpoint_gb=total,
            backbone_gb=backbone,
            expert_gb=expert,
            ram_budget_gb=round(budget_gb, 2),
            cache_gb=0.0,
            peak_gb=peak,
            gain_gb=0.0,
            recommended_ctx=ctx_n,
            reason="fits fully resident under the RAM budget",
            backbone_source=src,
            hf=hf,
            speed_basis="not-streamed",
        )

    cache_bytes = budget_bytes - backbone_bytes - headroom_bytes
    max_cache_bytes = int(envelope.max_cache_gb * 1e9)
    if cache_bytes > max_cache_bytes:
        cache_bytes = max_cache_bytes
    cache_gb = _bytes_gb(cache_bytes)

    if cache_bytes < min_cache:
        return FitResult(
            model_id=mid,
            name=name,
            status="oom",
            checkpoint_gb=total,
            backbone_gb=backbone,
            expert_gb=expert,
            ram_budget_gb=round(budget_gb, 2),
            cache_gb=round(max(0.0, cache_gb), 2),
            peak_gb=round(backbone + headroom_gb, 1),
            gain_gb=0.0,
            recommended_ctx=ctx_n,
            reason=(
                f"backbone {backbone:.1f} GB + headroom {headroom_gb:.1f} GB "
                f"(ctx {ctx_n}) leaves {cache_gb:.1f} GB cache on a "
                f"{budget_gb:.1f} GB budget"
            ),
            backbone_source=src,
            hf=hf,
        )

    peak = round(backbone + cache_gb + activation_gb + kv_gb, 1)
    gain = round(total - peak, 1)
    prefill = decode = basis = None
    if model.get("measured_decode_tok_s") and model.get("disk_bound"):
        disk = float(envelope.disk_gb_s or BASELINE_DISK_GB_S)
        disk_r = disk / BASELINE_DISK_GB_S
        meas_cache = float(model.get("measured_cache_gb") or BASELINE.max_cache_gb)
        cache_r = (cache_gb / meas_cache) ** 0.5 if meas_cache > 0 else 1.0
        # Cap cache_r so a huge cache cannot invent tok/s above the disk.
        cache_r = min(1.35, max(0.35, cache_r))
        prefill = round(float(model["measured_prefill_tok_s"]) * disk_r, 1)
        decode = round(float(model["measured_decode_tok_s"]) * disk_r * cache_r, 1)
        if envelope.near_baseline:
            prefill = float(model["measured_prefill_tok_s"])
            decode = float(model["measured_decode_tok_s"])
            peak = float(model.get("measured_peak_gb") or peak)
            basis = "measured"
        else:
            basis = "io-scaled"

    return FitResult(
        model_id=mid,
        name=name,
        status="streamed",
        checkpoint_gb=total,
        backbone_gb=backbone,
        expert_gb=expert,
        ram_budget_gb=round(budget_gb, 2),
        cache_gb=round(cache_gb, 2),
        peak_gb=peak,
        gain_gb=gain,
        recommended_ctx=ctx_n,
        reason=f"streamed, expert cache {cache_gb:.1f} GB, peak ~{peak:.0f} GB",
        backbone_source=src,
        hf=hf,
        prefill_tok_s=prefill,
        decode_tok_s=decode,
        speed_basis=basis,
    )


def render_ladder(
    envelope: Envelope,
    *,
    title: str = "",
    storage_free_gb: float | None = None,
) -> str:
    rows = [fit_model(m, envelope, storage_free_gb=storage_free_gb) for m in LADDER]
    lines = []
    if title:
        lines.append(title)
        lines.append("")
    lines.append(
        f"envelope: {envelope.ram_gb:g} GB RAM, budget {envelope.ram_budget_gb:g} GB, "
        f"cache ceiling {envelope.max_cache_gb:g} GB, "
        f"{envelope.read_threads} readers, "
        f"disk {envelope.disk_gb_s or '?'} GB/s"
    )
    lines.append("")
    hdr = (
        f"{'model':<28} {'ckpt':>6} {'fit':>9} {'cache':>6} {'peak':>6} "
        f"{'+GB':>6} {'prefill':>8} {'decode':>7}  notes"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        pre = f"{r.prefill_tok_s:.0f}" if r.prefill_tok_s else "-"
        dec = f"{r.decode_tok_s:.1f}" if r.decode_tok_s else "-"
        gain = f"{r.gain_gb:.0f}" if r.status == "streamed" else "-"
        cache = f"{r.cache_gb:.1f}" if r.status == "streamed" else "-"
        peak = f"{r.peak_gb:.0f}" if r.status != "disk" else "-"
        note = r.speed_basis or r.status
        if r.status == "oom":
            note = "OOM"
        lines.append(
            f"{r.name:<28} {r.checkpoint_gb:6.0f} {r.status:>9} {cache:>6} "
            f"{peak:>6} {gain:>6} {pre:>8} {dec:>7}  {note}"
        )
    lines.append("")
    lines.append(
        "fit is budget math (backbone + headroom + min 1 GB cache). "
        "tok/s is measured on the 48 GB M5 Pro, or IO-scaled from that "
        "for other envelopes. GPU is not sandboxed."
    )
    return "\n".join(lines)
