# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hardware + envelope dataclasses. No I/O, no config side effects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Hardware:
    """What the box actually is (or what the sandbox pretends it is)."""

    ram_gb: float
    perf_cores: int
    ram_bytes: int | None = None
    efficiency_cores: int | None = None
    chip: str | None = None
    hw_model: str | None = None
    gpu_cores: int | None = None
    mem_bw_gb_s: float | None = None
    disk_gb_s: float | None = None
    storage_total_gb: float | None = None
    storage_free_gb: float | None = None
    year: int | None = None
    form: str | None = None  # air / pro / max / studio / mini
    source: str = "manual"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.ram_bytes is None and self.ram_gb:
            self.ram_bytes = int(float(self.ram_gb) * (1 << 30))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hardware:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        raw = {k: v for k, v in (data or {}).items() if k in known and v is not None}
        if "ram_gb" not in raw and "ram_bytes" in raw:
            raw["ram_gb"] = int(raw["ram_bytes"]) / (1 << 30)
        if "ram_gb" not in raw:
            raise ValueError("hardware needs ram_gb")
        if "perf_cores" not in raw:
            raw["perf_cores"] = 8
        return cls(**raw)


@dataclass
class Envelope:
    """Capacity knobs derived from Hardware. Fidelity knobs are not here."""

    ram_gb: float
    ram_fraction: float
    ram_budget_gb: float
    max_cache_gb: float
    prefill_cache_gb: float
    read_threads: int
    headroom_gb: float
    activation_gb: float
    staging_cap_gb: float
    kv_fp16_ctx: int
    read_delay_us_per_mb: float = 0.0
    near_baseline: bool = False
    disk_gb_s: float | None = None
    extra_env: dict[str, str] = field(default_factory=dict)

    def to_env(self, *, simulate: bool = False) -> dict[str, str]:
        """EXPERT_STREAM_* map. ``simulate`` adds the sandbox RAM clamp."""
        env = {
            "EXPERT_STREAM_RAM_FRACTION": _fmt(self.ram_fraction),
            "EXPERT_STREAM_MAX_CACHE_GB": _fmt(self.max_cache_gb),
            "EXPERT_STREAM_PREFILL_CACHE_GB": _fmt(self.prefill_cache_gb),
            "EXPERT_STREAM_READ_THREADS": str(int(self.read_threads)),
            "EXPERT_STREAM_HEADROOM_GB": _fmt(self.headroom_gb),
            "EXPERT_STREAM_ACTIVATION_GB": _fmt(self.activation_gb),
            "EXPERT_STREAM_STAGING_CAP_GB": _fmt(self.staging_cap_gb),
            "EXPERT_STREAM_KV_FP16_CTX": str(int(self.kv_fp16_ctx)),
        }
        if simulate:
            env["EXPERT_STREAM_SIMULATE_RAM_GB"] = _fmt(self.ram_gb)
            env["EXPERT_STREAM_ENVELOPE_STRICT"] = "1"
        if self.read_delay_us_per_mb and self.read_delay_us_per_mb > 0:
            env["EXPERT_STREAM_READ_DELAY_US_PER_MB"] = _fmt(
                self.read_delay_us_per_mb
            )
        env.update(self.extra_env)
        return env

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class FitResult:
    """Whether a checkpoint fits resident, streamed, or OOMs."""

    model_id: str
    name: str
    status: str  # resident | streamed | oom | disk
    checkpoint_gb: float
    backbone_gb: float
    expert_gb: float
    ram_budget_gb: float
    cache_gb: float
    peak_gb: float
    gain_gb: float
    recommended_ctx: int
    reason: str
    backbone_source: str = "measured"
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None
    speed_basis: str | None = None
    hf: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fmt(v: float) -> str:
    if float(v) == int(v):
        return str(int(v))
    return f"{float(v):.4g}"
