# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Point cloud -> CadQuery code, with a trainable point encoder.

Each row points at a dense .npy cloud and holds the code that builds it.

    {"points_path": "points/smoke_00.npy", "solution": "import cadquery as cq\n..."}

The cloud is sampled down to N points (furthest point sampling), centered,
Fourier encoded and projected to the model's hidden size. Those N vectors
replace N placeholder tokens at the start of the sequence; the model learns
to write the code after them. The projector trains together with the LoRA
and is saved as encoder.safetensors next to the adapter.

    POINTS_ROOT=~/data/cadquery-points \\
    python -m pagedmoe.train sft \\
        --model ~/mlx-models/Qwen3.6-35B-A3B-4bit \\
        --data $POINTS_ROOT/train.jsonl \\
        --formatter pagedmoe/train/examples/points_encoder.py:format_row \\
        --encoder pagedmoe/train/examples/points_encoder.py:make

From Python:

    from pagedmoe.train import SFTConfig, SFTTrainer
    from pagedmoe.train.examples.points_encoder import PointEncoder, format_row

    SFTTrainer(SFTConfig(model=..., data=...), formatter=format_row,
               encoder=PointEncoder(hidden=2048)).train()

Swap the encoder for anything that maps rows to [P, hidden] vectors (an
image tower, an audio frontend, a frozen pretrained encoder plus a trainable
adapter) and the rest of the run stays the same.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from pagedmoe.train import Example

N_POINTS = int(os.environ.get("POINTS_N", 256))
N_FREQS = int(os.environ.get("POINTS_FREQS", 8))
PROMPT = "\nCadQuery code for this shape:\n"


def _root() -> Path:
    return Path(os.environ.get("POINTS_ROOT", ".")).expanduser()


def furthest_point_sample(pts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(pts) <= n:
        idx = np.resize(np.arange(len(pts)), n)
        return pts[idx]
    rng = np.random.default_rng(seed)
    chosen = np.empty(n, np.int64)
    chosen[0] = rng.integers(len(pts))
    dist = np.full(len(pts), np.inf)
    for i in range(1, n):
        dist = np.minimum(dist, ((pts - pts[chosen[i - 1]]) ** 2).sum(-1))
        chosen[i] = int(dist.argmax())
    return pts[chosen]


class PointEncoder(nn.Module):
    def __init__(self, hidden: int, n_points: int = N_POINTS, n_freqs: int = N_FREQS):
        super().__init__()
        self.n_points = n_points
        self.n_freqs = n_freqs
        self.proj = nn.Linear(3 * (1 + 2 * n_freqs), hidden)
        object.__setattr__(self, "_clouds", {})  # path -> sampled cloud, not a parameter

    def cloud(self, row) -> np.ndarray:
        p = Path(row["points_path"])
        p = p if p.is_absolute() else _root() / p
        key = str(p)
        if key not in self._clouds:
            pts = np.load(p).astype(np.float32)[:, :3]
            pts = furthest_point_sample(pts, self.n_points)
            self._clouds[key] = pts - pts.mean(axis=0, keepdims=True)
        return self._clouds[key]

    def __call__(self, batch) -> mx.array:
        x = mx.array(np.stack([self.cloud(r) for r in batch.rows]))  # [B, N, 3]
        feats = [x]
        for k in range(self.n_freqs):
            feats += [mx.sin(x * (2.0**k)), mx.cos(x * (2.0**k))]
        return self.proj(mx.concatenate(feats, axis=-1))  # [B, N, hidden]


def make(hidden: int) -> PointEncoder:
    return PointEncoder(hidden)


def format_row(row: dict, fmt) -> Example:
    """N placeholders (replaced by the encoder), a short prompt, then the code."""
    head = [fmt.pad_id] * N_POINTS + fmt.encode(PROMPT)
    code = fmt.encode(row["solution"]) + ([fmt.eos] if fmt.eos is not None else [])
    return Example(head + code, [0] * len(head) + [1] * len(code), embeds_at=0, row=row)
