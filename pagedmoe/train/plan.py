# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decide how a run uses this Mac.

Same rule as the engine: detect, measure, and let an explicit setting win.

Before load (headers only, nothing read):
  mode        resident when weights + training room fit the RAM budget,
              otherwise streamed
  executor    streamed -> layered; resident -> fused, or checkpoint when a
              fused micro-batch would not fit

After load, a probe runs real steps on synthetic batches of the planned
shape (no optimizer update) and measures peak memory, step time and expert
bytes read. That sets the micro-batch size and, on a streamed model, the
step size: a layered step reads the expert mass about twice whatever its
size, so the step grows until compute covers the reads. Probe results are
kept per model and machine in the train dir (PAGED_MOE_TRAIN_DIR, default
~/.paged_moe/train) so later runs start straight away.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

GiB = 1 << 30
PRIOR_MARGIN = 2.5
FIT = 0.92  # share of the budget a planned step may use; the rest is slack


def train_dir() -> Path:
    raw = os.environ.get("PAGED_MOE_TRAIN_DIR")
    return Path(raw).expanduser() if raw else Path.home() / ".paged_moe" / "train"


@dataclass
class ModelFacts:
    path: str
    name: str
    model_type: str
    layers: int
    hidden: int
    vocab: int
    experts: int
    active: int
    total_bytes: int
    expert_bytes: int
    params: int = 0  # logical parameter count (quantized weights unpacked)
    # never resident in a streamed load: PLE n-gram tables (read a few rows
    # per token from disk) and vision / MTP weights (dropped for text)
    skip_bytes: int = 0
    # hidden states carried between layers: hyper-connection models carry
    # several (hc_count), so the layer boundary is that many times hidden
    streams: int = 1

    @property
    def backbone_bytes(self) -> int:
        """What a streamed load keeps resident besides the expert cache."""
        return self.total_bytes - self.expert_bytes - self.skip_bytes


def min_cache_bytes(f: ModelFacts) -> int:
    """Three layers of experts: the layer running, the one being read, one spare."""
    return int(3 * f.expert_bytes / max(1, f.layers))


def max_cache_bytes(f: ModelFacts) -> int:
    """Every expert, plus a little for per-entry overhead. More buys nothing."""
    return int(f.expert_bytes * 1.01)


@dataclass
class Plan:
    mode: str = "resident"
    executor: str = "fused"
    cache_gb: float | None = None
    micro_tokens: int = 8192
    max_rows: int = 64
    per_step: int = 8  # units per optimizer step
    head_chunk: int = 512
    budget_gb: float = 0.0
    disk_gb_s: float | None = None
    reasons: dict = field(default_factory=dict)
    probe: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def model_facts(path: str) -> ModelFacts:
    from ..loader import resolve_model_path
    from ..ple_offload import skipped_resident_tensor_names
    from ..safetensors_index import read_headers

    p = resolve_model_path(path)
    if not (p / "config.json").is_file():
        inside = sorted(c.parent.name for c in p.glob("*/config.json"))
        if inside:
            raise FileNotFoundError(
                f"{p} is a folder of models, not a model. Point at one checkpoint "
                f"inside it, e.g. {p / inside[0]} ({len(inside)} found: {', '.join(inside[:5])}"
                f"{', ...' if len(inside) > 5 else ''})"
            )
        raise FileNotFoundError(
            f"{p} isn't a model folder (no config.json). Point at the checkpoint "
            "folder itself, the one with config.json and the .safetensors files."
        )
    cfg = json.loads((p / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    index = read_headers(p)
    total = sum(loc.nbytes for loc in index.values())
    # same split as the engine loader, so the plan and the load agree
    skip = skipped_resident_tensor_names(index)
    skip_bytes = sum(index[n].nbytes for n in skip)
    expert = sum(
        loc.nbytes for name, loc in index.items()
        if name not in skip and (".switch_mlp." in name or ".experts." in name)
    )

    def first(*keys, default=0):
        for k in keys:
            if tc.get(k) is not None:
                return tc[k]
        return default

    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    params = 0
    for name, loc in index.items():
        mod, _, leaf = name.rpartition(".")
        w = index.get(mod + ".weight")
        if leaf in ("scales", "biases") and w is not None and w.dtype == "U32":
            continue  # quantization metadata, not parameters
        n = math.prod(loc.shape) if loc.shape else 1
        if loc.dtype == "U32":
            over = q.get(mod)
            bits = over.get("bits") if isinstance(over, dict) else None
            n = n * 32 // int(bits or q.get("bits", 4))
        params += n

    return ModelFacts(
        path=str(p),
        name=p.name,
        model_type=str(cfg.get("model_type", "?")),
        layers=int(first("num_hidden_layers", "n_layers")),
        hidden=int(first("hidden_size", "d_model")),
        vocab=int(first("vocab_size")),
        experts=int(first("num_experts", "n_routed_experts", "num_local_experts")),
        active=int(first("num_experts_per_tok", "moe_top_k", "top_k")),
        total_bytes=int(total),
        expert_bytes=int(expert),
        params=int(params),
        skip_bytes=int(skip_bytes),
        streams=max(1, int(first("hc_count", "hc_mult", default=1))),
    )


def machine_budget() -> tuple[int, int, int | None]:
    """(ram, training budget, Metal working set) in bytes."""
    from .. import config, mempressure

    try:
        from ..machine import apply_machine_defaults

        apply_machine_defaults()
    except Exception:
        pass
    ram = config.total_ram_bytes()
    budget = int(config.RAM_FRACTION * ram)
    ws = mempressure.gpu_working_set_bytes()
    if ws:
        budget = min(budget, int(ws) - int(min(4 * GiB, max(1 * GiB, 0.05 * ws))))
    return ram, budget, ws


def room_floor(budget: int) -> int:
    """Training room a resident model must leave: 4 GB on a 48 GB Mac, a
    share of the budget elsewhere so small Macs are not overcommitted and big
    ones keep room for the larger batches they plan."""
    return int(max(2 * GiB, 0.12 * budget))


def live_budget(budget: int) -> tuple[int, str | None]:
    """The budget, lowered when the Mac is already short on memory.

    The budget is a share of physical RAM and does not know what else is
    running. Under kernel memory pressure at the start of a run it is capped
    at what the kernel can hand out now, so the first step does not push
    other apps into the compressor. At normal pressure macOS keeps little
    free on purpose and the free count says nothing, so it is left alone.
    """
    if os.environ.get("PAGED_MOE_SIMULATE_RAM_GB"):
        return budget, None
    try:
        from .. import mempressure

        if mempressure.level() < mempressure.WARN:
            return budget, None
        avail = mempressure.available_bytes()
        if not avail:
            return budget, None
        cap = int(avail + mempressure.footprint_gb() * 1e9 - 2 * GiB)
    except Exception:
        return budget, None
    if cap >= budget:
        return budget, None
    new = max(budget // 2, cap)
    return new, (f"memory pressure at start, {avail / 1e9:.1f} GB free: "
                 f"{budget / 1e9:.1f} -> {new / 1e9:.1f} GB (close other apps for more)")


BASELINE_BUDGET = 0.68 * 48 * GiB  # the 48 GB Mac the caps below were measured on


def size_scale(budget_gb: float) -> float:
    """How many 48 GB Macs of training budget this is (1 at or below it)."""
    return max(1.0, budget_gb * 1e9 / BASELINE_BUDGET)


def pow2_cap(base: int, budget_gb: float, top: int) -> int:
    return int(min(top, 1 << int(math.log2(base * size_scale(budget_gb)))))


def disk_gb_s() -> float | None:
    """This Mac's SSD read speed when the machine layer knows it."""
    try:
        from ..machine import get_provider

        hw = get_provider().hardware
        if hw.disk_gb_s:
            return float(hw.disk_gb_s)
    except Exception:
        pass
    try:
        from ..machine.scale import BASELINE_DISK_GB_S

        return float(BASELINE_DISK_GB_S)
    except Exception:
        return None


def per_token_bytes(f: ModelFacts) -> dict:
    """Rough activation bytes per token, refined by the probe."""
    H, L, V = f.hidden, f.layers, f.vocab
    R = H * max(1, f.streams)
    return {
        "stored": L * R * 2,  # layered: one bf16 layer boundary per layer
        "layer": R * 2 * 48,  # one layer's recompute graph
        "fused": L * R * 2 * 24,  # whole graph without checkpointing
        "checkpoint": L * R * 2 + R * 2 * 48,
        "logits": V * 4 * 3,
    }


def initial(cfg, f: ModelFacts, units_mean: float, n_units: int, rows_per_unit: int,
            max_len: int = 4096) -> Plan:
    ram, budget, ws = machine_budget()
    budget, why = live_budget(budget)
    plan = Plan(budget_gb=round(budget / 1e9, 2), disk_gb_s=disk_gb_s())
    if why:
        plan.reasons["budget"] = why
    ptb = per_token_bytes(f)
    room_min = room_floor(budget)

    # mode
    if cfg.mode in ("resident", "streamed"):
        plan.mode = cfg.mode
        plan.reasons["mode"] = "set"
    elif f.total_bytes + room_min <= budget:
        plan.mode = "resident"
        plan.reasons["mode"] = (
            f"fits: {f.total_bytes / 1e9:.1f} GB of weights, "
            f"{(budget - f.total_bytes) / 1e9:.1f} GB left for training"
        )
    else:
        plan.mode = "streamed"
        plan.reasons["mode"] = (
            f"{f.total_bytes / 1e9:.1f} GB of weights against a {budget / 1e9:.1f} GB budget"
        )

    # memory left for activations
    if plan.mode == "resident":
        room = budget - f.total_bytes
    else:
        room = budget - f.backbone_bytes - room_min
    room = max(room, 2 * GiB)

    # executor
    if cfg.executor != "auto":
        plan.executor = cfg.executor
        plan.reasons["executor"] = "set"
    elif plan.mode == "streamed":
        plan.executor = "layered"
        plan.reasons["executor"] = "streamed experts: layer-major so each step reads them ~twice"
    else:
        fused_tok = int(room * 0.5 / (ptb["fused"] + ptb["logits"] / 8))
        # a fused micro-batch has to hold a few of the longest rows
        if fused_tok >= max(1024, min(4096, 4 * max_len * rows_per_unit)):
            plan.executor = "fused"
            plan.reasons["executor"] = "resident and a fused micro-batch fits"
        else:
            plan.executor = "checkpoint"
            plan.reasons["executor"] = "resident; per-layer recompute to fit memory"
    if plan.mode == "streamed" and plan.executor in ("fused", "checkpoint"):
        raise ValueError("a streamed model needs executor=layered (or auto)")

    # micro-batch tokens
    per = {
        "fused": ptb["fused"],
        "checkpoint": ptb["checkpoint"],
        "layered": ptb["layer"] + ptb["stored"] * 4,
    }[plan.executor]
    if cfg.micro_batch_tokens:
        plan.micro_tokens = int(cfg.micro_batch_tokens)
        plan.reasons["micro"] = "set"
    else:
        share = 0.45 if plan.mode == "resident" else 0.35
        top = pow2_cap(32768, plan.budget_gb, 131072)
        plan.micro_tokens = int(max(1024, min(top, room * share / per)))
        plan.micro_tokens = 1 << int(math.log2(plan.micro_tokens))
        plan.reasons["micro"] = "estimated from memory; the probe refines it"

    # head chunk: one chunk of fp32 logits (plus softmax) at ~3% of the
    # budget, 1 GB on a 48 GB Mac
    chunk_bytes = min(4 * GiB, max(GiB // 4, 0.03 * budget))
    plan.head_chunk = int(max(64, min(4096, chunk_bytes / max(1, ptb["logits"]))))

    # streamed cache: what the backbone and training room leave. This only
    # has to be safe until the probe has measured the real numbers; the
    # trainer then refits it (Trainer._fit_cache). The per-token prior runs
    # low (a 125B hybrid measured 3.0 MB/token against 1.2 MB), so it is
    # charged PRIOR_MARGIN times over, for rows as long as the data's longest.
    if plan.mode == "streamed":
        if cfg.cache_gb:
            plan.cache_gb = float(cfg.cache_gb)
        else:
            tokens = max(plan.micro_tokens, max_len * rows_per_unit)
            act = tokens * per * PRIOR_MARGIN
            cache = budget - f.backbone_bytes - act - 2 * GiB
            cache = min(cache, max_cache_bytes(f))
            plan.cache_gb = round(max(cache, min_cache_bytes(f)) / 1e9, 2)

    # step size
    if cfg.batch_size:
        plan.per_step = int(cfg.batch_size)
        plan.reasons["batch"] = "set"
    else:
        target = 16384 if plan.mode == "resident" else 32768
        rows = max(1, int(target / max(1.0, units_mean * rows_per_unit)))
        cap = max(1, n_units // 8) if n_units >= 16 else n_units
        plan.per_step = max(1, min(rows, cap, 256))
        if plan.per_step == cap and cap < rows:
            plan.reasons["batch"] = "small dataset: at least 8 steps per epoch"
        else:
            plan.reasons["batch"] = (
                f"~{target // 1024}k tokens per step"
                + (" (streamed: amortizes expert reads)" if plan.mode == "streamed" else "")
            )
    return plan


# ------------------------------------------------------------------ profiles


PROFILE_VERSION = 5  # bump when step speed or memory changes, so old probes are redone


def profile_key(f: ModelFacts, plan: Plan, cfg) -> str:
    from .. import config
    from . import fastgrad

    raw = "|".join(
        str(x)
        for x in (
            PROFILE_VERSION,
            fastgrad.enabled(),
            os.path.realpath(f.path),
            config.total_ram_bytes(),
            plan.mode,
            plan.executor,
            cfg.lora.rank,
            cfg.lora.target,
            # an auto cache is refit from the profile, which records the size
            # it was measured at, so the key does not depend on it
            plan.cache_gb if cfg.cache_gb else "auto",
        )
    )
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def load_profile(key: str) -> dict | None:
    p = train_dir() / "profiles" / f"{key}.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def save_profile(key: str, data: dict) -> None:
    d = train_dir() / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    data = dict(data, saved=time.time())
    (d / f"{key}.json").write_text(json.dumps(data, indent=2) + "\n")


# ------------------------------------------------------------------ printing


def fmt_tokens(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{n:.0f}"


def fmt_time(s: float) -> str:
    if s is None or not math.isfinite(s):
        return "?"
    s = int(s)
    if s >= 86400:
        return f"{s // 86400}d{(s % 86400) // 3600}h"
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"
