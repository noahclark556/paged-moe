# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""LoRA on top of a (possibly streamed) model.

Adapters are mlx-lm's own LoRALinear / LoRASwitchLinear, saved in mlx-lm's
format, so a run's output loads with `mlx_lm.load(path, adapter_path=...)`
and with `pagedmoe.load(path, adapter_path=...)` for serving.

Targets are picked from the model instead of a per-architecture list:

  attention   every Linear inside an attention / linear-attention block
  all-linear  attention plus shared experts and dense MLPs (default)
  experts     all-linear plus the routed experts (resident models only:
              streamed experts are read-only pages from disk)

Routers are never targeted. A LoRA on the router changes which experts run,
which is a different kind of fine-tune than anyone asking for LoRA expects.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

_ROUTER_NAMES = {"gate", "router", "shared_expert_gate", "gate_up_router", "w_router"}
_ATTN_HINTS = ("attn", "attention", "mixer", "linear_attn", "self_attn", "mamba", "conv")


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target: str = "all-linear"  # attention | all-linear | experts | custom
    keys: list[str] | None = None  # explicit layer-relative names (target=custom)
    num_layers: int = -1  # last N decoder layers, -1 = all
    dora: bool = False

    @property
    def scale(self) -> float:
        return float(self.alpha) / float(self.rank)


@dataclass
class LoRAReport:
    keys: list[str] = field(default_factory=list)
    trainable: int = 0
    total: int = 0
    layers: int = 0

    def line(self) -> str:
        pct = 100.0 * self.trainable / max(1, self.total)
        return (
            f"LoRA {self.trainable / 1e6:.2f}M trainable ({pct:.3f}%) "
            f"on {self.layers} layers, {len(self.keys)} module kinds"
        )


def _is_router(name: str) -> bool:
    last = name.rsplit(".", 1)[-1]
    return last in _ROUTER_NAMES or "router" in last


def discover_keys(layers, target: str, streamed: bool) -> list[str]:
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear

    lin = (nn.Linear, nn.QuantizedLinear)
    switch = (SwitchLinear, QuantizedSwitchLinear)
    keys: set[str] = set()
    for layer in layers:
        for name, m in layer.named_modules():
            if not name or _is_router(name):
                continue
            if isinstance(m, switch):
                if target == "experts" and not streamed:
                    keys.add(name)
                continue
            if not isinstance(m, lin):
                continue
            if target == "attention" and not any(h in name for h in _ATTN_HINTS):
                continue
            keys.add(name)
    return sorted(keys)


def apply(model, cfg: LoRAConfig, streamed: bool) -> LoRAReport:
    """Freeze the model and add LoRA layers. Returns what was adapted."""
    from mlx_lm.tuner.utils import linear_to_lora_layers

    layers = model.layers
    n = len(layers) if cfg.num_layers in (-1, None) else int(cfg.num_layers)
    n = max(1, min(n, len(layers)))
    if cfg.target == "custom":
        if not cfg.keys:
            raise ValueError("lora.target=custom needs lora.keys")
        keys = list(cfg.keys)
    else:
        if cfg.target == "experts" and streamed:
            print(
                "[train] lora.target=experts on a streamed model: routed experts stay "
                "frozen (they page from disk); adapting the rest"
            )
        keys = discover_keys(layers[-n:], cfg.target, streamed)
    if not keys:
        raise ValueError(f"no LoRA targets found for target={cfg.target!r}")

    model.freeze()
    params = {
        "rank": int(cfg.rank),
        "scale": cfg.scale,
        "dropout": float(cfg.dropout),
        "keys": keys,
    }
    linear_to_lora_layers(model, n, params, use_dora=cfg.dora)
    object.__setattr__(
        model, "_pagedmoe_lora", {"num_layers": n, "params": params, "dora": cfg.dora}
    )

    rep = LoRAReport(keys=keys, layers=n)
    rep.trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    rep.total = sum(_logical_size(v) for _, v in tree_flatten(model.parameters()))
    return rep


def _logical_size(a: mx.array) -> int:
    # Packed quantized weights hold several values per uint32.
    if a.dtype == mx.uint32:
        return a.size * 8
    return a.size


def adapter_config(model) -> dict:
    meta = model._pagedmoe_lora
    return {
        "fine_tune_type": "dora" if meta["dora"] else "lora",
        "num_layers": meta["num_layers"],
        "lora_parameters": meta["params"],
    }


def save(model, out_dir: Path, extra: dict | None = None) -> Path:
    """adapters.safetensors + adapter_config.json, mlx-lm layout."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), weights)
    cfg = adapter_config(model)
    if extra:
        cfg.update(extra)
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return out_dir


def load_weights(model, path: Path) -> None:
    path = Path(path)
    f = path / "adapters.safetensors" if path.is_dir() else path
    model.load_weights(str(f), strict=False)
    mx.eval(model.trainable_parameters())


def _lora_modules(model):
    from mlx_lm.tuner.dora import DoRALinear
    from mlx_lm.tuner.lora import LoRALinear, LoRASwitchLinear

    if any(isinstance(m, DoRALinear) for _, m in model.named_modules()):
        # DoRA's trained magnitude survives scale 0, so that is not the base.
        raise ValueError("a reference model needs plain LoRA (lora.dora=false)")
    kinds = (LoRALinear, LoRASwitchLinear)
    return [m for _, m in model.named_modules() if isinstance(m, kinds)]


@contextlib.contextmanager
def disabled(model):
    """The base model: every adapter scaled to zero (exact, y + 0)."""
    mods = _lora_modules(model)
    saved = [m.scale for m in mods]
    try:
        for m in mods:
            m.scale = 0.0
        yield
    finally:
        for m, s in zip(mods, saved):
            m.scale = s
