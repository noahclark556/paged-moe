# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Feature construction for the sidecar heads (v4).

A feature vector is a fixed, deterministic concatenation of four blocks:

  [0                       : hist_dim)    hashed, age-decayed (layer, expert) demand
  [hist_dim                : +stripe_dim) per-target-layer stripes for the last token
  [.. : +sketch_dim)                      signed random projection of the last hidden state
  [.. : +EXTRA_DIM)                       named scalar channels (sorted key order)

The layout is a pure function of (dim, n_layers, wrap) so a persisted weight
matrix always lines up with the features rebuilt on a later run. The sketch
block is always *allocated* even when no hidden state is available (it is then
zero, and the `has_hidden` scalar channel tells the model which regime it is
in) - otherwise enabling/disabling hidden state mid-life would silently shift
every downstream column.

Why the hidden-state block matters: the demand history alone can only express
"experts that fire together". The next token's early-layer routing is a
function of the next token's embedding, and the last hidden state is precisely
what the LM head turns into that token. Projecting it in is the difference
between modelling co-occurrence and modelling the router.
"""

from __future__ import annotations

import hashlib
import itertools
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Bump when the feature layout or weight layout changes so stale .npz files
# are ignored instead of silently mis-indexed.
SLOT_VERSION = 4

# Fixed tail reserved for named scalar channels.
EXTRA_DIM = 16

# Per-token decay of the hashed history block. Used both by the truncated-window
# builder and by the incremental accumulator, so the two agree.
HIST_DECAY = 0.6

# Canonical scalar channel names. Callers should stick to these so a slot
# trained in one build keeps meaning in the next (packing is sorted-by-key).
EXTRA_HAS_HIDDEN = "a_has_hidden"
EXTRA_DEMAND = "b_demand"
EXTRA_STEP = "c_step"
EXTRA_CHUNK = "d_chunk"
EXTRA_ENTROPY = "e_entropy"

# Keys that often embed local download paths and must not affect identity.
_CONFIG_VOLATILE_KEYS = frozenset(
    {
        "_name_or_path",
        "name_or_path",
    }
)

# Bytes sampled from each weight shard (head / mid / tail) so two checkpoints
# with the same sharding layout but different weights still diverge.
_SHARD_SAMPLE = 256 * 1024


def legacy_path_slot_id(model_path: str, n_layers: int, n_experts: int) -> str:
    """Pre-content slot id (path-based). Kept for one-shot migration only."""
    raw = f"{os.path.realpath(model_path)}|{n_layers}|{n_experts}|v{SLOT_VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _stable_config_bytes(path: Path) -> bytes:
    import json

    try:
        data = json.loads(path.read_text())
    except Exception:
        return path.read_bytes()
    if not isinstance(data, dict):
        return path.read_bytes()
    for key in _CONFIG_VOLATILE_KEYS:
        data.pop(key, None)
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()


def _shard_sample_digest(path: Path) -> bytes:
    """Size + head/mid/tail samples - cheap content id without reading the whole file."""
    h = hashlib.sha1()
    try:
        size = path.stat().st_size
    except OSError:
        return h.digest()
    h.update(str(size).encode())
    sample = _SHARD_SAMPLE
    try:
        with path.open("rb") as f:
            h.update(f.read(min(sample, size)))
            if size > sample * 2:
                f.seek(size // 2)
                h.update(f.read(sample))
            if size > sample:
                f.seek(max(0, size - sample))
                h.update(f.read(sample))
    except OSError:
        pass
    return h.digest()


def checkpoint_fingerprint(model_path: str) -> str:
    """Location-independent fingerprint of a local MLX / HF checkpoint directory.

    Combines a sanitized ``config.json``, ``model.safetensors.index.json`` when
    present, and head/mid/tail samples of every ``*.safetensors`` shard so the
    same weights resolve to the same id after move/redownload, while distinct
    fine-tunes / quants diverge even if sharding matches.
    """
    root = Path(model_path).expanduser()
    try:
        root = root.resolve()
    except OSError:
        root = Path(os.path.abspath(os.path.expanduser(model_path)))

    h = hashlib.sha1()
    cfg = root / "config.json"
    if cfg.is_file():
        h.update(b"config\0")
        h.update(_stable_config_bytes(cfg))

    index = root / "model.safetensors.index.json"
    if index.is_file():
        h.update(b"index\0")
        try:
            h.update(index.read_bytes())
        except OSError:
            pass

    shards = sorted(root.glob("*.safetensors"))
    for shard in shards:
        h.update(b"shard\0")
        h.update(shard.name.encode())
        h.update(_shard_sample_digest(shard))

    if not cfg.is_file() and not index.is_file() and not shards:
        # Last resort for odd layouts - still include geometry upstream.
        h.update(b"empty\0")
        h.update(str(root).encode())

    return h.hexdigest()


def model_slot_id(model_path: str, n_layers: int, n_experts: int) -> str:
    """Stable per-checkpoint sidecar id (path-independent).

    Hash of checkpoint fingerprint + MoE geometry + ``SLOT_VERSION``. Same
    weights in a new directory keep the same slot; a different fine-tune or
    quant gets a new one.
    """
    fp = checkpoint_fingerprint(model_path)
    raw = f"{fp}|{int(n_layers)}|{int(n_experts)}|v{SLOT_VERSION}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def migrate_legacy_slot_files(
    root: Path | str,
    model_path: str,
    n_layers: int,
    n_experts: int,
    new_id: str | None = None,
) -> tuple[str, list[str]]:
    """Rename path-based slot files to the content-based id when needed.

    Returns ``(active_slot_id, list of renamed source names)``. If the new id
    already has files, legacy files are left untouched (no overwrite).
    """
    root_p = Path(root)
    new_id = new_id or model_slot_id(model_path, n_layers, n_experts)
    old_id = legacy_path_slot_id(model_path, n_layers, n_experts)
    if old_id == new_id or not root_p.is_dir():
        return new_id, []

    moved: list[str] = []
    for src in sorted(root_p.glob(f"{old_id}*")):
        if not src.is_file():
            continue
        dest = root_p / f"{new_id}{src.name[len(old_id):]}"
        if dest.exists():
            continue
        try:
            src.rename(dest)
            moved.append(src.name)
        except OSError:
            try:
                import shutil

                shutil.copy2(src, dest)
                moved.append(src.name)
            except OSError:
                pass

    # Keep wrap-bank metadata.slot aligned with the new id when present.
    bank = root_p / f"{new_id}_wrap_bank.npz"
    if bank.is_file() and moved:
        try:
            from .bank import load_bank, save_bank

            traj, meta = load_bank(bank)
            if meta.get("slot") != new_id:
                meta = dict(meta)
                meta["slot"] = new_id
                save_bank(bank, traj, meta=meta)
        except Exception:
            pass

    return new_id, moved


def auto_feat_dim(n_layers: int, n_experts: int) -> int:
    """Feature width scaled to the model's routing space.

    Small MoEs (16-64 experts) do not need 512 hashed slots; a 512-expert model
    collides badly at 512. Powers of two keep the BLAS shapes friendly.
    """
    if n_experts <= 32:
        base = 192
    elif n_experts <= 96:
        base = 320
    elif n_experts <= 192:
        base = 512
    elif n_experts <= 384:
        base = 768
    else:
        base = 1024
    # Deep stacks carry more distinct (layer, expert) pairs to hash.
    if n_layers >= 64:
        base = int(base * 1.25)
    return int(min(1536, max(128, base)))


def auto_wrap(n_layers: int) -> int:
    """How many early layers of the *next* token to prefetch for.

    The usable lead time is the gap between finishing token N and layer i of
    token N+1 needing bytes, which grows with depth; deeper stacks can cover
    more layers before the reads stop landing in time.
    """
    return int(min(8, max(2, round(n_layers / 14.0))))


def auto_prefill_layers(n_layers: int) -> int:
    return int(min(n_layers, max(8, min(24, round(n_layers / 3.0)))))


@dataclass(frozen=True)
class FeatureSpec:
    """Deterministic feature layout + reusable hashing tables."""

    dim: int
    n_layers: int
    wrap: int
    hist_dim: int
    stripe_dim: int
    sketch_dim: int
    stripe_width: int
    table: np.ndarray  # (n_layers, n_experts) -> index into the hist block

    @staticmethod
    def build(
        dim: int,
        n_layers: int,
        wrap: int,
        n_experts: int,
        *,
        sketch_dim: int | None = None,
    ) -> "FeatureSpec":
        dim = int(max(64, dim))
        n_layers = int(max(1, n_layers))
        wrap = int(max(1, min(wrap, n_layers)))
        want_sketch = (
            int(sketch_dim) if sketch_dim is not None else _config_sketch_dim()
        )
        # Never let the sketch or stripes crowd out the hashed history.
        sk = int(max(0, min(want_sketch, (dim - EXTRA_DIM) // 4)))
        rest = dim - EXTRA_DIM - sk
        stripe_width = 24
        stripe = int(min(rest // 2, wrap * stripe_width))
        stripe_width = max(1, stripe // wrap) if stripe else 0
        stripe = stripe_width * wrap
        hist = rest - stripe
        if hist < 32:  # pathologically small dim - drop stripes first
            hist, stripe, stripe_width = rest, 0, 0
        table = _hash_table(hist, n_layers, int(max(1, n_experts)))
        return FeatureSpec(
            dim=dim,
            n_layers=n_layers,
            wrap=wrap,
            hist_dim=hist,
            stripe_dim=stripe,
            sketch_dim=sk,
            stripe_width=stripe_width,
            table=table,
        )

    # ------------------------------------------------------------------ build

    def empty(self) -> np.ndarray:
        return np.zeros(self.dim, dtype=np.float32)

    @property
    def demand_cut(self) -> int:
        return self.hist_dim + self.stripe_dim

    def hash_demand(self, demand: dict[int, list[int]]) -> np.ndarray:
        """Flat hist-block indices for one token's demand.

        Per-layer demand is ragged in practice (decode pruning drops a different
        number of experts per layer), so this flattens once and gathers once
        rather than branching on a uniform width.
        """
        layers: list[int] = []
        counts: list[int] = []
        rows: list = []
        total = 0
        for layer, ids in demand.items():
            li = int(layer)
            n = len(ids)
            if li < 0 or li >= self.n_layers or n == 0:
                continue
            layers.append(li)
            counts.append(n)
            rows.append(ids)
            total += n
        if total == 0:
            return np.empty(0, dtype=np.int64)
        experts = np.fromiter(
            itertools.chain.from_iterable(rows), dtype=np.int64, count=total
        )
        np.clip(experts, 0, self.table.shape[1] - 1, out=experts)
        layer_of = np.repeat(np.asarray(layers, dtype=np.int64), counts)
        return self.table[layer_of, experts]

    def demand_counts(self, demand: dict[int, list[int]]) -> np.ndarray:
        """One token's hashed demand as a dense hist-block vector."""
        idx = self.hash_demand(demand)
        if idx.size == 0 or self.hist_dim <= 0:
            return np.zeros(self.hist_dim, dtype=np.float32)
        return np.bincount(idx, minlength=self.hist_dim)[: self.hist_dim].astype(
            np.float32
        )

    def compose(
        self,
        hist_acc: np.ndarray,
        latest: dict[int, list[int]] | None,
        *,
        hidden_sketch: np.ndarray | None = None,
        extra: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Assemble a vector from a pre-accumulated (decayed) history block.

        The history block is an exponential moving average maintained by the
        caller, so a token costs one scaled add instead of re-hashing the whole
        window - which was ~90% of the sidecar's per-token time.
        """
        x = self.empty()
        if self.hist_dim > 0:
            x[: self.hist_dim] = hist_acc[: self.hist_dim]
        if self.stripe_dim > 0 and latest:
            base0 = self.hist_dim
            for i in range(self.wrap):
                ids = latest.get(i)
                if not ids:
                    continue
                arr = np.asarray(ids, dtype=np.int64) % self.stripe_width
                np.add.at(
                    x[base0 + i * self.stripe_width : base0 + (i + 1) * self.stripe_width],
                    arr,
                    np.float32(2.0),
                )
        cut = self.demand_cut
        n = float(np.linalg.norm(x[:cut]))
        if n > 0:
            x[:cut] /= n
        if self.sketch_dim > 0 and hidden_sketch is not None:
            m = min(self.sketch_dim, int(hidden_sketch.shape[0]))
            x[cut : cut + m] = hidden_sketch[:m]
        if extra:
            tail = self.dim - EXTRA_DIM
            for i, k in enumerate(sorted(extra.keys())[:EXTRA_DIM]):
                x[tail + i] = np.float32(extra[k])
        return x

    def build_vector(
        self,
        history: list[dict[int, list[int]]],
        *,
        hidden_sketch: np.ndarray | None = None,
        extra: dict[str, float] | None = None,
    ) -> np.ndarray:
        x = self.empty()
        if history and self.hist_dim > 0:
            idx_parts: list[np.ndarray] = []
            w_parts: list[np.ndarray] = []
            for age, demand in enumerate(reversed(history)):
                decay = np.float32(HIST_DECAY**age)
                for layer, ids in demand.items():
                    li = int(layer)
                    if li < 0 or li >= self.n_layers or len(ids) == 0:
                        continue
                    arr = np.asarray(ids, dtype=np.int64)
                    arr = arr[(arr >= 0) & (arr < self.table.shape[1])]
                    if arr.size == 0:
                        continue
                    idx_parts.append(self.table[li, arr])
                    w_parts.append(np.full(arr.size, decay, dtype=np.float32))
            if idx_parts:
                np.add.at(
                    x[: self.hist_dim],
                    np.concatenate(idx_parts),
                    np.concatenate(w_parts),
                )
        if self.stripe_dim > 0 and history:
            base0 = self.hist_dim
            latest = history[-1]
            for i in range(self.wrap):
                ids = latest.get(i)
                if not ids:
                    continue
                arr = np.asarray(ids, dtype=np.int64) % self.stripe_width
                np.add.at(
                    x[base0 + i * self.stripe_width : base0 + (i + 1) * self.stripe_width],
                    arr,
                    np.float32(2.0),
                )
        # Normalize the sparse demand blocks on their own so the (dense, already
        # scaled) sketch block cannot be swamped by a token that touched many
        # experts, or vice versa.
        cut = self.hist_dim + self.stripe_dim
        n = float(np.linalg.norm(x[:cut]))
        if n > 0:
            x[:cut] /= n
        if self.sketch_dim > 0 and hidden_sketch is not None:
            m = min(self.sketch_dim, int(hidden_sketch.shape[0]))
            x[cut : cut + m] = hidden_sketch[:m]
        if extra:
            tail = self.dim - EXTRA_DIM
            for i, k in enumerate(sorted(extra.keys())[:EXTRA_DIM]):
                x[tail + i] = np.float32(extra[k])
        return x


def _config_sketch_dim() -> int:
    from .. import config

    if not bool(getattr(config, "SIDECAR_HIDDEN", 1)):
        return 0
    return int(getattr(config, "SIDECAR_SKETCH_DIM", 64))


def _hash_table(dim: int, n_layers: int, n_experts: int) -> np.ndarray:
    """(layer, expert) -> hashed slot. Precomputed so the hot path is a gather."""
    if dim <= 0:
        return np.zeros((n_layers, n_experts), dtype=np.int64)
    layers = np.arange(n_layers, dtype=np.int64)[:, None]
    experts = np.arange(n_experts, dtype=np.int64)[None, :]
    # Multiplicative mixing with distinct odd primes per axis, then fold.
    h = (layers * np.int64(2654435761)) ^ (experts * np.int64(40503))
    h ^= h >> np.int64(13)
    return np.abs(h) % np.int64(dim)


def sketch_matrix(slot_id: str, hidden_dim: int, out_dim: int) -> np.ndarray:
    """Deterministic sparse sign projection (Achlioptas) for hidden states.

    Derived from the slot id so it is identical across restarts - a persisted
    weight matrix is meaningless if the projection it was trained through
    changes.
    """
    seed = int(
        hashlib.sha1(
            f"{slot_id}|{hidden_dim}|{out_dim}|v{SLOT_VERSION}".encode()
        ).hexdigest()[:8],
        16,
    )
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, 3, size=(int(hidden_dim), int(out_dim)))
    proj = np.zeros((int(hidden_dim), int(out_dim)), dtype=np.float32)
    proj[draw == 0] = -1.0
    proj[draw == 2] = 1.0
    proj *= np.float32(np.sqrt(3.0 / max(1, int(out_dim))))
    return proj


def project_hidden(hidden: np.ndarray, proj: np.ndarray) -> np.ndarray:
    """L2-normalize a hidden state, project, and bound the result."""
    h = np.asarray(hidden, dtype=np.float32).reshape(-1)
    if h.shape[0] != proj.shape[0]:
        m = min(h.shape[0], proj.shape[0])
        h = h[:m]
        proj = proj[:m]
    n = float(np.linalg.norm(h))
    if n > 0:
        h = h / n
    return np.tanh(h @ proj).astype(np.float32)


# ----------------------------------------------------------------- baselines


def same_layer_ceiling(
    prev: dict[int, list[int]], cur: dict[int, list[int]], wrap: int
) -> float:
    """Recall of the naive "reuse the last token's experts" prediction."""
    hit = need = 0
    for i in range(wrap):
        want = set(cur.get(i, ()))
        if not want:
            continue
        need += len(want)
        hit += len(want.intersection(prev.get(i, ())))
    return (hit / need) if need else 0.0


def union_ceiling(prev: dict[int, set[int]], cur: dict[int, set[int]]) -> float:
    """Recall if we predicted this chunk's target set by copying the last one."""
    hit = need = 0
    for i in set(prev) | set(cur):
        want = set(cur.get(i, ()))
        if not want:
            continue
        need += len(want)
        hit += len(want.intersection(prev.get(i, ())))
    return (hit / need) if need else 0.0


def hot_set(
    counts: dict[int, dict[int, int]], frac: float
) -> dict[int, list[int]]:
    """Per layer, the experts a chunk actually leaned on.

    A prefill chunk's expert *union* approaches the whole expert set, so
    predicting it is both trivial and useless. The experts worth prefetching
    are the ones many tokens in the chunk shared.
    """
    out: dict[int, list[int]] = {}
    frac = float(min(1.0, max(0.0, frac)))
    for layer, per_expert in counts.items():
        if not per_expert:
            continue
        top = max(per_expert.values())
        bar = max(2.0, top * frac)
        keep = [int(e) for e, c in per_expert.items() if c >= bar]
        if not keep:
            keep = [int(max(per_expert, key=per_expert.get))]
        out[int(layer)] = keep
    return out


# ------------------------------------------------------- back-compat helpers

_SPEC_CACHE: dict[tuple, FeatureSpec] = {}


def _spec_for(dim: int, n_layers: int, wrap: int, n_experts: int) -> FeatureSpec:
    key = (int(dim), int(n_layers), int(wrap), int(n_experts), _config_sketch_dim())
    spec = _SPEC_CACHE.get(key)
    if spec is None:
        spec = FeatureSpec.build(dim, n_layers, wrap, n_experts)
        _SPEC_CACHE[key] = spec
    return spec


def build_features(
    history: list[dict[int, list[int]]],
    dim: int,
    n_layers: int,
    wrap: int,
    *,
    extra: dict[str, float] | None = None,
    hidden_sketch: np.ndarray | None = None,
    n_experts: int = 1024,
) -> np.ndarray:
    """Layout-compatible convenience wrapper (used by tests and pretrain)."""
    spec = _spec_for(dim, n_layers, wrap, n_experts)
    return spec.build_vector(history, hidden_sketch=hidden_sketch, extra=extra)


def demand_to_features(demand, dim, n_layers, wrap: int = 4):
    return build_features([demand], dim, n_layers, wrap)
