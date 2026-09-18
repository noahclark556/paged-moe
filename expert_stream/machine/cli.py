# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""paged-moe machine | ladder | probe  (wired from mlx_hook.main)."""

from __future__ import annotations

import json
import sys
from typing import Any


def _print(obj: Any) -> None:
    if isinstance(obj, str):
        print(obj)
        return
    print(json.dumps(obj, indent=2, default=str))


def cmd_machine(argv: list[str]) -> int:
    from .catalog import list_catalog
    from .provider import MachineProvider, apply_machine_defaults, save_host_probe

    sub = (argv[0] if argv else "show").lower()
    if sub in ("-h", "--help", "help"):
        print(
            "usage: paged-moe machine [show|detect|probe|catalog|apply]\n"
            "  show     detected hardware + derived knobs (default)\n"
            "  detect   same, JSON\n"
            "  probe    sequential SSD + GPU cores, cache under ~/.paged_moe/\n"
            "  catalog  named Mac classes\n"
            "  apply    write envelope env into this process (debug)"
        )
        return 0
    if sub == "catalog":
        from .catalog import CATALOG

        rows = []
        for pid, hw in CATALOG.items():
            rows.append(
                f"{pid:28}  {hw.ram_gb:g} GB  {hw.perf_cores}P  "
                f"disk~{hw.disk_gb_s} GB/s  {hw.chip or ''}  {hw.notes or ''}"
            )
        print("\n".join(rows))
        return 0
    if sub == "probe":
        from .detect import probe_host

        hw = probe_host()
        path = save_host_probe(hw)
        _print({"wrote": str(path), "hardware": hw.to_dict()})
        return 0
    if sub == "apply":
        p = apply_machine_defaults(force=True)
        _print(p.summary())
        return 0

    p = MachineProvider.detect()
    if sub == "detect":
        _print(p.summary())
        return 0
    # human show
    h, e = p.hardware, p.envelope
    print(f"chip:          {h.chip or '?'}")
    print(f"hw.model:      {h.hw_model or '?'}")
    print(f"catalog:       {p.catalog_id or '(unmatched)'}")
    print(f"source:        {h.source}")
    print(f"ram:           {h.ram_gb:g} GB  ({h.ram_bytes} bytes)")
    print(f"perf cores:    {h.perf_cores}" + (f"  e-cores {h.efficiency_cores}" if h.efficiency_cores else ""))
    print(f"gpu cores:     {h.gpu_cores or '? (paged-moe machine probe)'}")
    print(f"mem bw:        {h.mem_bw_gb_s or '?'} GB/s")
    print(f"ssd:           {h.disk_gb_s or '? (paged-moe machine probe)'} GB/s")
    print(f"storage:       {h.storage_total_gb or '?'} GB  ({h.storage_free_gb or '?'} GB free)")
    print("--- envelope ---")
    print(f"baseline:      {'yes (48 GB class, shipped knobs kept)' if e.near_baseline else 'no (scaled from 48 GB M5 Pro)'}")
    print(f"ram fraction:  {e.ram_fraction}")
    print(f"budget:        {e.ram_budget_gb:g} GB")
    print(f"max cache:     {e.max_cache_gb:g} GB")
    print(f"prefill cache: {e.prefill_cache_gb:g} GB")
    print(f"headroom:      {e.headroom_gb:g} GB")
    print(f"activation:    {e.activation_gb:g} GB")
    print(f"staging cap:   {e.staging_cap_gb:g} GB")
    print(f"read threads:  {e.read_threads}")
    print(f"kv_fp16_ctx:   {e.kv_fp16_ctx}")
    return 0


def cmd_ladder(argv: list[str]) -> int:
    from .catalog import CATALOG
    from .ladder import render_ladder
    from .provider import MachineProvider

    profile = None
    json_out = False
    rest = []
    for a in argv:
        if a in ("-h", "--help", "help"):
            print(
                "usage: paged-moe ladder [--profile ID] [--json]\n"
                "  no flag     this Mac (detect + yaml)\n"
                "  --profile   named class (mba-m2-2023-16gb, ...)"
            )
            return 0
        if a == "--json":
            json_out = True
            continue
        if a == "--profile" and rest:
            continue
        rest.append(a)
    if "--profile" in argv:
        i = argv.index("--profile")
        if i + 1 >= len(argv):
            print("paged-moe ladder --profile needs an id", file=sys.stderr)
            return 2
        profile = argv[i + 1]
        if profile not in CATALOG:
            print(
                f"unknown profile {profile!r}. known: {', '.join(CATALOG)}",
                file=sys.stderr,
            )
            return 2
        p = MachineProvider.from_catalog(profile, simulate=True)
    else:
        p = MachineProvider.detect()

    if json_out:
        from .ladder import LADDER, fit_model

        rows = [fit_model(m, p.envelope).to_dict() for m in LADDER]
        _print({"machine": p.summary(), "ladder": rows})
        return 0
    title = p.catalog_id or (p.hardware.chip or "this Mac")
    print(render_ladder(p.envelope, title=f"PagedMoE ladder  ({title})"))
    return 0
