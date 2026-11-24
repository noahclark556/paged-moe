# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""README.md for a converted model: what it came from and how it was built."""

from __future__ import annotations

from pathlib import Path


def _gb(n) -> str:
    return f"{(n or 0) / 1e9:.2f} GB"


def text(report: dict, plan=None) -> str:
    src = report.get("source") or {}
    ref = src.get("ref") or "?"
    base = ref if src.get("kind") == "hub" else None
    lic = src.get("license")
    front = ["---"]
    if lic:
        front.append(f"license: {lic}")
    if base:
        front.append(f"base_model: {base}")
    front += ["library_name: mlx", "pipeline_tag: text-generation", "tags:", "- mlx", "- pagedmoe", "---", ""]

    name = Path(report.get("out") or "model").name
    lines = [f"# {name}", ""]
    made = f"Converted to MLX from [{ref}](https://huggingface.co/{ref})" if base else f"Converted to MLX from `{ref}`"
    if src.get("revision"):
        made += f" at revision `{str(src['revision'])[:12]}`"
    lines += [made + f" with PagedMoE {report.get('pagedmoe', '')}.", ""]
    ad = report.get("adapter")
    if ad:
        lines += [f"A {ad['kind'].upper()} adapter ({ad['modules']} modules, rank {ad.get('rank')}) "
                  "is merged into the weights.", ""]

    fc = report.get("fit_choice")
    if fc:
        lines += [f"Recipe picked for the Mac it was converted on: **{fc['picked']}** ({fc['why']}).", ""]

    lines += ["| Part | Setting | Size |", "| --- | --- | --- |"]
    for r in report.get("components") or []:
        lines.append(f"| {r['label']} | {r.get('setting') or '-'} | {_gb(r['out_bytes'])} |")
    lines += ["", f"Total {_gb(report.get('out_bytes'))}, {report.get('bits_per_weight')} bits per weight "
              f"(the float checkpoint is {_gb(report.get('float_bytes'))}).", ""]

    we = report.get("weight_error") or {}
    comps = we.get("components") or {}
    if comps:
        lines += ["Weight error from quantizing (signal to noise, higher is closer to the original):", "",
                  "| Part | SNR |", "| --- | --- |"]
        labels = {r["component"]: r["label"] for r in report.get("components") or []}
        for c, v in comps.items():
            if v.get("snr_db") is not None:
                lines.append(f"| {labels.get(c, c)} | {v['snr_db']} dB |")
        lines.append("")

    lines += ["## Use", "", "```bash", "pip install mlx-lm",
              'mlx_lm.generate --model <this folder> --prompt "hello"', "```", "",
              "Bigger than your Mac's memory? PagedMoE streams the experts from disk:", "",
              "```bash", "paged-moe serve <this folder>", "```", "",
              "## Rebuild it", "", "```bash",
              f"paged-moe convert run {ref} ./{name} --recipe pagedmoe_conversion.json", "```", ""]
    return "\n".join(front + lines)


def write(out: Path, report: dict, plan=None) -> Path:
    p = Path(out) / "README.md"
    p.write_text(text(report, plan))
    return p
