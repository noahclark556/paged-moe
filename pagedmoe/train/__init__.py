# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Paged training: fine-tune MoE models whose experts stream from disk.

    from pagedmoe.train import SFTTrainer, SFTConfig

    SFTTrainer(SFTConfig(model="~/mlx-models/Qwen3.6-35B-A3B-4bit",
                         data="data/chat.jsonl")).train()

Methods: SFT, CPT, ORPO, DPO (sigmoid / IPO / hinge), SimPO, CPO, KTO, GRPO.
Usage, data formats and custom inputs: docs/paged-training.md.
"""

from .config import (
    CPOConfig,
    CPTConfig,
    DPOConfig,
    GRPOConfig,
    KTOConfig,
    ORPOConfig,
    SFTConfig,
    SimPOConfig,
    TrainConfig,
)
from .config import load as load_config
from .data import Example, Formatter, Pair, Prompt
from .executor import Batch
from .export import to_peft
from .lora import LoRAConfig
from .objectives import Custom, Objective, StepNorm
from .stats import Callback
from .trainer import (
    CPOTrainer,
    CPTTrainer,
    DPOTrainer,
    GRPOTrainer,
    KTOTrainer,
    ORPOTrainer,
    SFTTrainer,
    SimPOTrainer,
    TRAINERS,
    Trainer,
)

__all__ = [
    "Batch", "Callback", "CPOConfig", "CPOTrainer", "CPTConfig", "CPTTrainer", "Custom",
    "DPOConfig", "DPOTrainer", "Example", "Formatter", "GRPOConfig", "GRPOTrainer",
    "KTOConfig", "KTOTrainer", "LoRAConfig", "ORPOConfig", "ORPOTrainer", "Objective",
    "Pair", "Prompt", "SFTConfig", "SFTTrainer", "SimPOConfig", "SimPOTrainer",
    "StepNorm", "TRAINERS", "TrainConfig", "Trainer", "load_config", "to_peft",
]
