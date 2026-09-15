# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Wrap sample bank: densify short decode into many train pairs.

The expensive part of wrap pretrain is running the MoE to get real router
labels. Once you have a *trajectory* of per-token demand, every sliding window
is a valid (history -> next demand) sample - so 32 decode tokens become dozens
of SGD steps, and a tiny corpus stops being the bottleneck.

Cross-prompt mixing of history->demand is *not* done here: routing is
model-local, and gluing history from prompt A onto demand from prompt B is
mostly noise. Mixing happens across *positions* in the same trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import FeatureSpec


def expand_trajectory(
    trajectory: list[dict[int, list[int]]],
    spec: FeatureSpec,
    *,
    history_len: int = 6,
    multi_scale: bool = True,
) -> list[tuple[np.ndarray, dict[int, list[int]]]]:
    """Turn one decode's demand frames into many (features, target) pairs.

    For each token t > 0, predict demand[t] from the preceding window. With
    ``multi_scale``, also emit shorter histories (1..history_len) so the head
    learns both cold-start and long-context regimes from the same bytes.
    """
    if len(trajectory) < 2:
        return []
    out: list[tuple[np.ndarray, dict[int, list[int]]]] = []
    h_max = max(1, int(history_len))
    scales = list(range(1, h_max + 1)) if multi_scale else [h_max]
    for t in range(1, len(trajectory)):
        target = {int(k): [int(e) for e in v] for k, v in trajectory[t].items()}
        if not target:
            continue
        for h in scales:
            start = max(0, t - h)
            hist = trajectory[start:t]
            if not hist:
                continue
            feats = spec.build_vector(hist)
            out.append((feats, target))
    return out


def expand_trajectories(
    trajectories: list[list[dict[int, list[int]]]],
    spec: FeatureSpec,
    *,
    history_len: int = 6,
    multi_scale: bool = True,
) -> list[tuple[np.ndarray, dict[int, list[int]]]]:
    samples: list[tuple[np.ndarray, dict[int, list[int]]]] = []
    for traj in trajectories:
        samples.extend(
            expand_trajectory(
                traj, spec, history_len=history_len, multi_scale=multi_scale
            )
        )
    return samples


def bank_path(root: Path, slot_id: str) -> Path:
    return Path(root) / f"{slot_id}_wrap_bank.npz"


def save_bank(
    path: Path,
    trajectories: list[list[dict[int, list[int]]]],
    *,
    meta: dict | None = None,
) -> None:
    """Persist demand trajectories (JSON-in-npz) for later fit-only runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(trajectories, separators=(",", ":")).encode("utf-8")
    meta_s = json.dumps(meta or {}, separators=(",", ":")).encode("utf-8")
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        trajectories=np.frombuffer(payload, dtype=np.uint8),
        meta=np.frombuffer(meta_s, dtype=np.uint8),
        n_traj=np.array([len(trajectories)], dtype=np.int64),
    )
    tmp.replace(path)


def load_bank(path: Path) -> tuple[list[list[dict[int, list[int]]]], dict]:
    path = Path(path)
    if not path.exists():
        return [], {}
    data = np.load(path, allow_pickle=False)
    raw = bytes(np.asarray(data["trajectories"], dtype=np.uint8).tolist())
    trajectories = json.loads(raw.decode("utf-8"))
    meta: dict = {}
    if "meta" in data.files:
        mraw = bytes(np.asarray(data["meta"], dtype=np.uint8).tolist())
        if mraw:
            meta = json.loads(mraw.decode("utf-8"))
    # Normalize keys to int (json may stringify).
    out: list[list[dict[int, list[int]]]] = []
    for traj in trajectories:
        frames = []
        for frame in traj:
            frames.append(
                {int(k): [int(e) for e in v] for k, v in frame.items()}
            )
        out.append(frames)
    return out, meta


def merge_trajectories(
    a: list[list[dict[int, list[int]]]],
    b: list[list[dict[int, list[int]]]],
    *,
    max_traj: int = 256,
) -> list[list[dict[int, list[int]]]]:
    """Keep the newest trajectories; bank stays bounded."""
    merged = list(a) + list(b)
    if len(merged) > max_traj:
        merged = merged[-max_traj:]
    return merged
