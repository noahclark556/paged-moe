# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Resolve hardware (detect / yaml / catalog / sandbox) and apply envelope env.

Precedence, highest first:

  1. Explicit process env already set (EXPERT_STREAM_MAX_CACHE_GB, ...)
  2. ``machine:`` field overrides in ~/paged-moe-config.yaml
  3. Auto-detect (and optional ~/.paged_moe/host_probe.json)
  4. Shipped 48 GB baseline defaults in config.py

Sandbox apply always overwrites: it is pretending to be a different Mac.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .catalog import CATALOG, catalog_entry, match_catalog
from .detect import detect_hardware
from .scale import BASELINE_DISK_GB_S, derive_envelope, is_baseline_hardware
from .types import Envelope, Hardware

_PROVIDER: MachineProvider | None = None


def _user_machine_section() -> dict[str, Any]:
    try:
        from expert_stream.user_config import load_user_config

        cfg = load_user_config()
    except Exception:
        return {}
    raw = cfg.get("machine") if isinstance(cfg, dict) else None
    return raw if isinstance(raw, dict) else {}


def _probe_cache_path() -> Path:
    override = os.environ.get("PAGED_MOE_HOST_PROBE", "").strip()
    if override:
        return Path(override).expanduser()
    try:
        from expert_stream.user_config import default_root

        return default_root() / "host_probe.json"
    except Exception:
        return Path.home() / ".paged_moe" / "host_probe.json"


def load_host_probe() -> dict[str, Any] | None:
    path = _probe_cache_path()
    if not path.is_file():
        return None
    try:
        import json

        raw = json.loads(path.read_text())
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def save_host_probe(hw: Hardware) -> Path:
    path = _probe_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(hw.to_dict(), indent=2) + "\n")
    return path


def _merge_hardware(
    base: Hardware,
    overlay: dict[str, Any],
    *,
    source: str,
) -> Hardware:
    data = base.to_dict()
    for k, v in overlay.items():
        if k in ("auto", "profile", "id") or v is None or v == "":
            continue
        if k in Hardware.__dataclass_fields__:
            data[k] = v
            if k == "ram_gb":
                data.pop("ram_bytes", None)
    data["source"] = source
    return Hardware.from_dict(data)


class MachineProvider:
    """Single owner of 'what Mac is this' and the capacity envelope."""

    def __init__(
        self,
        hardware: Hardware,
        envelope: Envelope,
        *,
        catalog_id: str | None = None,
        simulate: bool = False,
    ):
        self.hardware = hardware
        self.envelope = envelope
        self.catalog_id = catalog_id
        self.simulate = simulate

    @classmethod
    def from_hardware(
        cls,
        hw: Hardware,
        *,
        simulate: bool = False,
        host_disk_gb_s: float | None = None,
        overrides: dict | None = None,
        catalog_id: str | None = None,
    ) -> MachineProvider:
        if simulate and host_disk_gb_s is None:
            host_disk_gb_s = _host_disk_for_sandbox()
        env = derive_envelope(
            hw,
            host_disk_gb_s=host_disk_gb_s,
            overrides=overrides,
        )
        return cls(hw, env, catalog_id=catalog_id, simulate=simulate)

    @classmethod
    def from_catalog(
        cls,
        profile_id: str,
        *,
        simulate: bool = True,
        host_disk_gb_s: float | None = None,
        overrides: dict | None = None,
    ) -> MachineProvider:
        hw = catalog_entry(profile_id)
        host_disk = host_disk_gb_s
        if host_disk is None and simulate:
            host_disk = _host_disk_for_sandbox()
        return cls.from_hardware(
            hw,
            simulate=simulate,
            host_disk_gb_s=host_disk,
            overrides=overrides,
            catalog_id=profile_id,
        )

    @classmethod
    def detect(cls, *, probe_disk: bool = False) -> MachineProvider:
        section = _user_machine_section()
        auto = section.get("auto", True)
        if auto is False or str(auto).lower() in ("0", "false", "no"):
            auto = False
        else:
            auto = True

        profile_id = section.get("profile") or section.get("id")
        if profile_id and str(profile_id) in CATALOG:
            hw = catalog_entry(str(profile_id))
            hw = _merge_hardware(hw, section, source="yaml+catalog")
            return cls.from_hardware(
                hw,
                simulate=False,
                overrides=_yaml_overrides(section),
                catalog_id=str(profile_id),
            )

        if auto:
            hw = detect_hardware(probe_disk=probe_disk)
            cached = load_host_probe()
            if cached:
                # Prefer a previously probed disk/gpu; live sysctl still wins for RAM.
                if hw.disk_gb_s is None and cached.get("disk_gb_s"):
                    hw.disk_gb_s = float(cached["disk_gb_s"])
                if hw.gpu_cores is None and cached.get("gpu_cores"):
                    hw.gpu_cores = int(cached["gpu_cores"])
                if hw.mem_bw_gb_s is None and cached.get("mem_bw_gb_s"):
                    hw.mem_bw_gb_s = float(cached["mem_bw_gb_s"])
            hw = _merge_hardware(hw, section, source="detect+yaml")
        else:
            # Manual yaml only. Missing RAM falls back to detect so we never
            # invent a tiny envelope on a 48 GB box.
            base = Hardware(
                ram_gb=float(section.get("ram_gb") or 48.0),
                perf_cores=int(section.get("perf_cores") or 16),
                source="yaml",
            )
            if not section.get("ram_gb"):
                base = detect_hardware()
            hw = _merge_hardware(base, section, source="yaml")

        if hw.disk_gb_s is None:
            cid = match_catalog(hw)
            if cid and CATALOG[cid].disk_gb_s:
                hw.disk_gb_s = CATALOG[cid].disk_gb_s
            if hw.gpu_cores is None and cid:
                hw.gpu_cores = CATALOG[cid].gpu_cores
            if hw.mem_bw_gb_s is None and cid:
                hw.mem_bw_gb_s = CATALOG[cid].mem_bw_gb_s
            catalog_id = cid
        else:
            catalog_id = match_catalog(hw)

        return cls.from_hardware(
            hw,
            simulate=False,
            overrides=_yaml_overrides(section),
            catalog_id=catalog_id,
        )

    def apply_to_environ(self, *, only_unset: bool = True) -> dict[str, str]:
        env = self.envelope.to_env(simulate=self.simulate)
        applied = {}
        for k, v in env.items():
            if only_unset and os.environ.get(k):
                continue
            os.environ[k] = str(v)
            applied[k] = str(v)
        if self.catalog_id:
            if not only_unset or not os.environ.get("EXPERT_STREAM_MACHINE_ID"):
                os.environ["EXPERT_STREAM_MACHINE_ID"] = self.catalog_id
                applied["EXPERT_STREAM_MACHINE_ID"] = self.catalog_id
        return applied

    def apply_to_config(self) -> None:
        from expert_stream import config as es_config

        es_config.sync_from_environ()

    def apply(self, *, only_unset: bool = True) -> dict[str, str]:
        applied = self.apply_to_environ(only_unset=only_unset)
        self.apply_to_config()
        if self.simulate:
            try:
                from .oom import install_metal_cap

                install_metal_cap(self.envelope)
            except Exception:
                pass
        return applied

    def summary(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "simulate": self.simulate,
            "near_baseline": self.envelope.near_baseline,
            "hardware": self.hardware.to_dict(),
            "envelope": self.envelope.to_dict(),
        }


def _yaml_overrides(section: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "ram_fraction",
        "max_cache_gb",
        "prefill_cache_gb",
        "read_threads",
        "headroom_gb",
        "activation_gb",
        "staging_cap_gb",
        "kv_fp16_ctx",
        "read_delay_us_per_mb",
        "env",
    )
    return {k: section[k] for k in keys if k in section and section[k] is not None}


def _host_disk_for_sandbox() -> float:
    cached = load_host_probe()
    if cached and cached.get("disk_gb_s"):
        try:
            return float(cached["disk_gb_s"])
        except (TypeError, ValueError):
            pass
    raw = os.environ.get("EXPERT_STREAM_SANDBOX_HOST_DISK_GB_S", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    try:
        live = detect_hardware(probe_disk=False)
        if live.disk_gb_s:
            return float(live.disk_gb_s)
        # Baseline host: the development machine this was tuned on.
        if is_baseline_hardware(live):
            return BASELINE_DISK_GB_S
    except Exception:
        pass
    return BASELINE_DISK_GB_S


def get_provider() -> MachineProvider:
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = MachineProvider.detect()
    return _PROVIDER


def reset_provider() -> None:
    global _PROVIDER
    _PROVIDER = None
    global _APPLIED
    _APPLIED = False


def set_provider(provider: MachineProvider) -> None:
    global _PROVIDER, _APPLIED
    _PROVIDER = provider
    _APPLIED = True


_APPLIED = False


def apply_machine_defaults(*, force: bool = False) -> MachineProvider:
    """Bind capacity knobs from this Mac unless they are already set.

    No-op on the measured 48 GB class so config.py defaults stay bit-for-bit.
    """
    global _APPLIED, _PROVIDER
    if _APPLIED and not force and _PROVIDER is not None:
        return _PROVIDER
    provider = get_provider() if _PROVIDER is not None else MachineProvider.detect()
    _PROVIDER = provider
    if provider.envelope.near_baseline and not force and not provider.simulate:
        _APPLIED = True
        return provider
    provider.apply(only_unset=True)
    _APPLIED = True
    return provider
