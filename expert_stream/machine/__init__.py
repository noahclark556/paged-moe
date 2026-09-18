# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Machine envelope provider for PagedMoE.

One place maps Mac hardware (RAM, cores, disk, storage) onto the *capacity*
knobs (cache ceiling, prefill cache, reader threads, headroom). Model recipes
(prune / wait / sidecar / fused prefill) stay per-checkpoint.

The 48 GB M5 Pro this engine was measured on is the baseline. Smaller Macs
scale the envelope down; larger Macs scale it up. Auto-detect is the default;
``~/paged-moe-config.yaml`` ``machine:`` can override any field.

Sandbox profiles can feed the same provider so a 16 GB envelope
OOMs on this 48 GB host instead of silently using the extra RAM.
"""

from .catalog import CATALOG, catalog_entry, list_catalog
from .detect import detect_hardware, probe_disk_gb_s, probe_host
from .ladder import LADDER, fit_model, render_ladder
from .oom import EnvelopeOOM, enforce_fit, install_metal_cap
from .provider import (
    MachineProvider,
    apply_machine_defaults,
    get_provider,
    reset_provider,
    set_provider,
)
from .scale import BASELINE, derive_envelope, read_delay_us_per_mb
from .types import Envelope, FitResult, Hardware

__all__ = [
    "BASELINE",
    "CATALOG",
    "Envelope",
    "EnvelopeOOM",
    "FitResult",
    "Hardware",
    "LADDER",
    "MachineProvider",
    "apply_machine_defaults",
    "catalog_entry",
    "derive_envelope",
    "detect_hardware",
    "enforce_fit",
    "fit_model",
    "get_provider",
    "install_metal_cap",
    "list_catalog",
    "probe_disk_gb_s",
    "probe_host",
    "read_delay_us_per_mb",
    "render_ladder",
    "reset_provider",
    "set_provider",
]
