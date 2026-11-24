# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run configuration.

One dataclass per method, all sharing `TrainConfig`. Anything left as None
is decided by the planner from the model, the data and this Mac (see
plan.py); set it to pin it. Configs load from YAML / JSON with the same
field names, with `lora:` as a nested block.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .lora import LoRAConfig


@dataclass
class TrainConfig:
    model: str = ""
    data: Any = None  # path, HF id, or a list of rows
    eval_data: Any = None
    output_dir: str | None = None  # default ~/.paged_moe/train/runs/<name>
    name: str | None = None

    # schedule
    epochs: float = 1.0
    max_steps: int | None = None
    batch_size: int | None = None  # rows / pairs / prompts per optimizer step
    micro_batch_tokens: int | None = None  # padded tokens per micro-batch
    max_seq_len: int | None = None

    # optimizer
    learning_rate: float | None = None
    lr_schedule: str = "cosine"  # cosine | linear | constant
    warmup_ratio: float = 0.03
    min_lr_ratio: float = 0.1
    optimizer: str = "adamw"  # adamw | adam | sgd | adafactor
    weight_decay: float = 0.0
    adam_betas: tuple[float, float] = (0.9, 0.999)
    max_grad_norm: float | None = 1.0

    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # engine
    mode: str = "auto"  # auto | resident | streamed
    executor: str = "auto"  # auto | fused | checkpoint | layered
    cache_gb: float | None = None
    probe: str = "auto"  # auto | always | never (memory / speed measurement)
    memory_guard: bool = True  # give memory back between steps when macOS runs short

    # formatting
    chat_template: bool | None = None  # None = use the tokenizer's when it has one
    completion_only: bool = True
    template_kwargs: dict = field(default_factory=dict)
    system_prompt: str | None = None

    # bookkeeping
    seed: int = 0
    save_steps: int | None = 50
    save_minutes: float | None = 20.0
    keep_checkpoints: int = 3
    resume: Any = False  # False | True / "latest" | checkpoint path
    eval_steps: int | None = None
    eval_batches: int | None = 16
    log_steps: int = 1
    report_to: list[str] = field(default_factory=list)  # "wandb"
    verbose: bool = True
    export_peft: bool = False  # also write <run>/peft/ for GPUs when the run ends

    method: str = "sft"

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        if not isinstance(self.data, (str, type(None))):
            d["data"] = f"<{type(self.data).__name__}>"
        if not isinstance(self.eval_data, (str, type(None))):
            d["eval_data"] = f"<{type(self.eval_data).__name__}>"
        return d


@dataclass
class SFTConfig(TrainConfig):
    method: str = "sft"
    packing: bool = False


@dataclass
class CPTConfig(TrainConfig):
    """Continued pretraining on raw text: packed windows, every token trained."""

    method: str = "cpt"
    packing: bool = True
    completion_only: bool = False
    chat_template: bool | None = False


@dataclass
class ORPOConfig(TrainConfig):
    method: str = "orpo"
    beta: float = 0.1  # weight of the odds-ratio term (lambda in the paper)


@dataclass
class DPOConfig(TrainConfig):
    method: str = "dpo"
    beta: float = 0.1
    loss_type: str = "sigmoid"  # sigmoid | ipo | hinge
    label_smoothing: float = 0.0


@dataclass
class SimPOConfig(TrainConfig):
    method: str = "simpo"
    beta: float = 2.0
    gamma: float = 1.0
    sft_weight: float = 0.0


@dataclass
class CPOConfig(TrainConfig):
    method: str = "cpo"
    beta: float = 0.1
    nll_weight: float = 1.0


@dataclass
class KTOConfig(TrainConfig):
    method: str = "kto"
    beta: float = 0.1
    desirable_weight: float = 1.0
    undesirable_weight: float = 1.0


@dataclass
class GRPOConfig(TrainConfig):
    method: str = "grpo"
    num_generations: int = 8
    max_completion_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    beta: float = 0.0  # KL weight; > 0 adds a reference pass per step
    epsilon: float = 0.2
    epsilon_high: float | None = 0.28
    loss_type: str = "dapo"  # dapo | grpo | dr_grpo
    scale_rewards: bool = True  # divide advantages by the group std
    iterations: int = 1  # updates per batch of rollouts
    reward_weights: list[float] | None = None
    skip_flat_groups: bool = True  # drop groups whose rewards are all equal


CONFIGS = {
    "sft": SFTConfig,
    "cpt": CPTConfig,
    "orpo": ORPOConfig,
    "dpo": DPOConfig,
    "simpo": SimPOConfig,
    "cpo": CPOConfig,
    "kto": KTOConfig,
    "grpo": GRPOConfig,
}

DEFAULT_LR = {
    "sft": 1e-4,
    "cpt": 1e-4,
    "orpo": 5e-5,
    "dpo": 2e-5,
    "simpo": 2e-5,
    "cpo": 2e-5,
    "kto": 2e-5,
    "grpo": 5e-6,
}


def _coerce(cls, data: dict):
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kw = {}
    unknown = []
    for k, v in data.items():
        k = k.replace("-", "_")
        if k not in fields:
            unknown.append(k)
            continue
        if k == "lora" and isinstance(v, dict):
            v = LoRAConfig(**{kk.replace("-", "_"): vv for kk, vv in v.items()})
        if k == "adam_betas" and isinstance(v, list):
            v = tuple(v)
        kw[k] = v
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {', '.join(sorted(unknown))}")
    return cls(**kw)


def from_dict(data: dict) -> TrainConfig:
    method = str(data.get("method", "sft")).lower()
    if method not in CONFIGS:
        raise ValueError(f"unknown method {method!r} (one of {', '.join(CONFIGS)})")
    return _coerce(CONFIGS[method], {**data, "method": method})


def load(path: str | Path) -> TrainConfig:
    p = Path(path).expanduser()
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return from_dict(data)
