# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GLM-5.3-Flash / glm5_next for mlx-lm loads (text path).

mlx-lm does not ship this architecture yet. PagedMoE registers this package
as mlx_lm.models.glm5_next so the usual load + StreamedSwitchGLU path works.
Vision weights are dropped. The MTP layer is left out of the backbone and
built as a drafter head by pagedmoe.mtp (see Glm5NextMTPLayer).

The MoE blocks use mlx-lm's SwitchGLU, which is the seam PagedMoE pages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlx_lm.models.base import BaseModelArgs

from .language import LanguageModel, TextConfig


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "glm5_next"
    text_config: dict = field(default_factory=dict)
    vision_config: dict = field(default_factory=dict)
    quantization: Any = None
    image_token_id: int = 154854
    video_token_id: int = 154855
    tie_word_embeddings: bool = False
    eos_token_id: Any = None
    pad_token_id: int = 154820

    @classmethod
    def from_dict(cls, params: dict):
        params = dict(params)
        if "text_config" not in params:
            # plain text checkpoint: the whole config is the text config
            return cls(model_type=params.get("model_type", "glm5_next"), text_config=params)
        return super().from_dict(params)

    def __post_init__(self):
        tc = dict(self.text_config or {})
        tc.setdefault("model_type", "glm5_next")
        self.text = TextConfig.from_dict(tc)

    def __getattr__(self, name):
        # Engine code reads text fields (num_hidden_layers, n_routed_experts,
        # ...) straight off model.args.
        text = self.__dict__.get("text")
        if text is not None and not name.startswith("__") and hasattr(text, name):
            return getattr(text, name)
        raise AttributeError(name)


class Model(LanguageModel):
    """glm5_next text model with the mlx-lm Model surface."""

    def __init__(self, args: ModelArgs):
        super().__init__(args.text)
        self.args = args
        self.model_type = args.model_type
