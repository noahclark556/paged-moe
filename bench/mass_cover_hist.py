#!/usr/bin/env python
# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline routing-mass histogram for a tool-result prefill.

Exact prefill. Records each MoE layer's routed ids and mixture weights for
the new chunk only, then asks whether a per-token mass budget of tau still
fits in a smaller expert union. Bytes are one expert read per kept id, the
same cost the streamed prefill pays. Nothing here changes the server default.

    python bench/mass_cover_hist.py --model ~/mlx-models/qwen38-coder-flash \
        --prefix 8192 --chunks 300 1300 2600
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Exact mixture. Sidecar and decode approximation stay out of the route.
os.environ.setdefault("PAGED_MOE_SIDECAR", "0")
os.environ.setdefault("PAGED_MOE_PRUNE", "0")
os.environ.setdefault("PAGED_MOE_WAIT_ABOVE", "0")
os.environ.setdefault("PAGED_MOE_FLOW", "0")

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from pagedmoe import load  # noqa: E402
from pagedmoe.streaming import (  # noqa: E402
    StreamedSwitchGLU,
    _route_weights_from_gate,
    _tapped_route,
    cache_only,
    exact_step,
)

TAUS = (0.80, 0.90, 0.95, 0.99, 1.0)
HEAVY = (0.50, 0.60, 0.70, 0.80)
PASS_TAU = 0.95
PASS_BYTE_RATIO = 0.60


class Capture:
    def __init__(self):
        self.on = False
        self.layers: list[dict] = []
        self._orig = StreamedSwitchGLU._forward

    def install(self) -> None:
        cap = self
        orig = self._orig

        def _forward(self, x, indices):  # noqa: N807
            lead = indices.shape
            k = lead[-1]
            ind2 = indices.reshape(-1, k)
            n = ind2.shape[0]
            if (
                cap.on
                and n > 20
                and not self._skips_final()
                and self.pred is not None
                and self.pred.gate is not None
            ):
                try:
                    w = _route_weights_from_gate(
                        self.pred.gate,
                        x.reshape(-1, x.shape[-1]),
                        ind2,
                        tapped=_tapped_route(self.pred.gate, x),
                        score_bias=self.pred.score_bias,
                    )
                except Exception:
                    w = None
                if w is not None and w.shape == ind2.shape:
                    cap.layers.append(
                        {
                            "layer": self.layer_key,
                            "nbytes": int(self.cache.expert_nbytes(self.layer_key)),
                            "ids": np.asarray(ind2).astype(np.int32, copy=True),
                            "w": np.asarray(w, dtype=np.float32),
                        }
                    )
            return orig(self, x, indices)

        StreamedSwitchGLU._forward = _forward

    def restore(self) -> None:
        StreamedSwitchGLU._forward = self._orig

    def take(self) -> list[dict]:
        layers, self.layers = self.layers, []
        return layers


def _nucleus_keep(ids: np.ndarray, w: np.ndarray, tau: float) -> np.ndarray:
    """Experts each token needs so its own top-k mass reaches tau.

    An expert's gate mass cannot be covered by a different expert, so the
    union of these per-token sets is the smallest set that meets the budget.
    """
    order = np.argsort(-w, axis=1)
    w_sorted = np.take_along_axis(w, order, axis=1)
    ids_sorted = np.take_along_axis(ids, order, axis=1)
    prev = np.cumsum(w_sorted, axis=1) - w_sorted
    keep = prev < (tau - 1e-6)
    keep[:, 0] = True
    return ids_sorted[keep]


def _token_mass(ids: np.ndarray, w: np.ndarray, kept: set[int]) -> np.ndarray:
    hit = np.isin(ids, np.fromiter(kept, dtype=np.int32, count=len(kept)))
    return (w * hit).sum(axis=1)


def _summarize(layers: list[dict]) -> dict:
    per_layer = []
    union_bytes = 0
    by_tau = {str(t): 0 for t in TAUS}
    heavy_mass_bytes = {str(f): 0.0 for f in HEAVY}
    heavy_min = {str(f): [] for f in HEAVY}
    heavy_hurt = {str(f): {"n": 0, "zero": 0, "half": 0} for f in HEAVY}
    single_token_mass = []
    row_sums = []
    top1 = []
    singleton_bytes = 0
    singleton_mass_bytes = 0.0
    for layer in layers:
        ids = layer["ids"]
        w = layer["w"]
        nbytes = int(layer["nbytes"])
        row_sums.append(w.sum(axis=1))
        flat = ids.reshape(-1)
        present = flat[flat >= 0]
        union = np.unique(present)
        union_bytes += int(union.size) * nbytes
        token_counts = {}
        mass = {}
        for e, wt in zip(ids.reshape(-1), w.reshape(-1)):
            e = int(e)
            if e < 0:
                continue
            token_counts[e] = token_counts.get(e, 0) + 1
            mass[e] = mass.get(e, 0.0) + float(wt)
        singletons = [e for e, c in token_counts.items() if c == 1]
        singleton_mass = sum(mass[e] for e in singletons)
        total_mass = sum(mass.values()) or 1.0
        singleton_bytes += len(singletons) * nbytes
        singleton_mass_bytes += (singleton_mass / total_mass) * int(union.size) * nbytes
        if singletons:
            single = set(singletons)
            single_token_mass.append(_token_mass(ids, w, single))
        order = sorted(mass, key=lambda e: mass[e], reverse=True)
        top1.append(w.max(axis=1))
        # Heaviest-first cut, ignoring which token needed the expert. This is
        # the most bytes a mass ranking can drop. It is not the pass rule.
        for frac in HEAVY:
            n_keep = max(1, int(np.ceil(frac * len(order))))
            kept = set(order[:n_keep])
            kept_mass = sum(mass[e] for e in kept)
            heavy_mass_bytes[str(frac)] += (kept_mass / total_mass) * int(union.size) * nbytes
            tm = _token_mass(ids, w, kept)
            if tm.size:
                heavy_min[str(frac)].append(float(tm.min()))
                heavy_hurt[str(frac)]["n"] += int(tm.size)
                heavy_hurt[str(frac)]["zero"] += int((tm < 0.05).sum())
                heavy_hurt[str(frac)]["half"] += int((tm < 0.50).sum())
        entry = {
            "layer": layer["layer"],
            "nbytes": nbytes,
            "tokens": int(ids.shape[0]),
            "k": int(ids.shape[1]),
            "union": int(union.size),
            "union_bytes": int(union.size) * nbytes,
            "mean_top1": float(w.max(axis=1).mean()),
            "singleton_experts": len(singletons),
            "singleton_expert_frac": len(singletons) / max(1, len(union)),
            "singleton_mass_frac": singleton_mass / total_mass,
        }
        for tau in TAUS:
            kept_ids = _nucleus_keep(ids, w, tau)
            kept = set(int(x) for x in kept_ids.tolist())
            tm = _token_mass(ids, w, kept)
            b = len(kept) * nbytes
            by_tau[str(tau)] += b
            entry[f"tau_{tau}"] = {
                "experts": len(kept),
                "byte_ratio": b / max(1, entry["union_bytes"]),
                "min_token_mass": float(tm.min()) if tm.size else 0.0,
                "mean_token_mass": float(tm.mean()) if tm.size else 0.0,
            }
        per_layer.append(entry)
    ratios = {}
    for tau in TAUS:
        ratios[str(tau)] = by_tau[str(tau)] / max(1, union_bytes)
    rs = np.concatenate(row_sums) if row_sums else np.zeros(1)
    layer_ratios_95 = [e["tau_0.95"]["byte_ratio"] for e in per_layer]
    return {
        "layers": len(per_layer),
        "union_bytes": union_bytes,
        "byte_ratio_at_tau": ratios,
        "pass_95_in_60": bool(ratios.get("0.95", 1.0) <= PASS_BYTE_RATIO),
        "layer_byte_ratio_95_min": float(min(layer_ratios_95) if layer_ratios_95 else 0),
        "layer_byte_ratio_95_median": float(np.median(layer_ratios_95) if layer_ratios_95 else 0),
        "layer_byte_ratio_95_max": float(max(layer_ratios_95) if layer_ratios_95 else 0),
        "weight_row_sum_median": float(np.median(rs)),
        "weight_row_sum_min": float(rs.min()),
        "weight_row_sum_max": float(rs.max()),
        "mean_top1": float(np.concatenate(top1).mean()) if top1 else 0.0,
        "singleton_byte_frac": singleton_bytes / max(1, union_bytes),
        "singleton_mass_frac_byte_weighted": singleton_mass_bytes / max(1, union_bytes),
        "heavy_keep_mass_frac": {
            k: v / max(1, union_bytes) for k, v in heavy_mass_bytes.items()
        },
        "heavy_keep_worst_token_mass": {
            k: (float(min(v)) if v else 0.0) for k, v in heavy_min.items()
        },
        "heavy_keep_token_frac_under_0.05": {
            k: (v["zero"] / v["n"] if v["n"] else 0.0) for k, v in heavy_hurt.items()
        },
        "heavy_keep_token_frac_under_0.50": {
            k: (v["half"] / v["n"] if v["n"] else 0.0) for k, v in heavy_hurt.items()
        },
        "singleton_token_mass_max": (
            float(max(float(a.max()) for a in single_token_mass)) if single_token_mass else 0.0
        ),
        "singleton_token_mass_mean": (
            float(np.concatenate(single_token_mass).mean()) if single_token_mass else 0.0
        ),
        "per_layer": per_layer,
    }


def _source_ids(tok, need: int) -> list[int]:
    root = ROOT / "pagedmoe"
    files = sorted(p for p in root.glob("*.py") if p.is_file())
    text = "\n".join(
        f"# ==== {p.name} ====\n{p.read_text(errors='replace')}" for p in files
    )
    ids = tok.encode(text)
    while len(ids) < need:
        ids = ids + ids
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix", type=int, default=8192)
    ap.add_argument("--chunks", type=int, nargs="+", default=[300, 1300, 2600])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model, tok = load(args.model, mode="streamed", verbose=True)
    cap = Capture()
    cap.install()
    ids = _source_ids(tok, args.prefix + sum(args.chunks) + 8)
    cache = make_prompt_cache(model)
    pos = 0
    t0 = time.perf_counter()
    with exact_step(), cache_only():
        while pos < args.prefix:
            n = min(4096, args.prefix - pos)
            mx.eval(model(mx.array([ids[pos : pos + n]]), cache=cache))
            pos += n
    seed_s = time.perf_counter() - t0
    print(f"seed {args.prefix} tok in {seed_s:.1f}s", flush=True)

    chunks_out = []
    for n in args.chunks:
        cap.layers.clear()
        cap.on = True
        t1 = time.perf_counter()
        with exact_step(), cache_only():
            mx.eval(model(mx.array([ids[pos : pos + n]]), cache=cache))
        wall = time.perf_counter() - t1
        cap.on = False
        pos += n
        layers = cap.take()
        summary = _summarize(layers)
        summary["chunk_tokens"] = n
        summary["wall_s_with_capture"] = round(wall, 2)
        chunks_out.append(summary)
        ratio = summary["byte_ratio_at_tau"].get("0.95", 1)
        print(
            f"chunk {n}: layers {summary['layers']}  "
            f"union {summary['union_bytes'] / 1e9:.2f} GB  "
            f"tau0.95 byte ratio {ratio:.3f}  "
            f"layer median {summary['layer_byte_ratio_95_median']:.3f}  "
            f"pass={summary['pass_95_in_60']}  "
            f"row_sum {summary['weight_row_sum_median']:.3f}  "
            f"wall {wall:.1f}s (capture overhead included)",
            flush=True,
        )
        for tau, r in summary["byte_ratio_at_tau"].items():
            print(f"    tau {tau}: {r:.3f} of union bytes", flush=True)
        print(
            f"    mean top1 {summary['mean_top1']:.3f}  "
            f"singleton bytes {summary['singleton_byte_frac']:.3f} "
            f"(mass {summary['singleton_mass_frac_byte_weighted']:.3f})",
            flush=True,
        )
        for frac, mass in summary["heavy_keep_mass_frac"].items():
            worst = summary["heavy_keep_worst_token_mass"][frac]
            z = summary["heavy_keep_token_frac_under_0.05"][frac]
            h = summary["heavy_keep_token_frac_under_0.50"][frac]
            print(
                f"    heaviest {float(frac):.0%} of union: {mass:.3f} of mass, "
                f"worst token {worst:.3f}, "
                f"tokens under 0.05 {z:.3f}, under 0.50 {h:.3f}",
                flush=True,
            )
        print(
            f"    singleton-expert mass on a token: "
            f"mean {summary['singleton_token_mass_mean']:.3f} "
            f"max {summary['singleton_token_mass_max']:.3f}",
            flush=True,
        )

    cap.restore()
    out = {
        "model": args.model,
        "prefix": args.prefix,
        "seed_s": round(seed_s, 2),
        "pass_rule": f"per-token mass {PASS_TAU} at or below {PASS_BYTE_RATIO} of union bytes",
        "chunks": chunks_out,
        "passed": all(c["pass_95_in_60"] for c in chunks_out),
    }
    dest = args.out or (
        ROOT / "bench" / "mass_cover_results" / f"{Path(args.model).name}.json"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Drop the bulky per-layer curves' raw nothing; per_layer is small enough.
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest}", flush=True)
    print(f"PASSED={out['passed']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
