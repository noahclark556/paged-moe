# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What a conversion does to each part of the model.

A recipe gives every component its own precision. MoE checkpoints are
almost all expert weight, and on PagedMoE the experts stream from the SSD
while everything else stays in RAM, so the two sides trade off differently:
expert bits set the disk size and how many bytes each token reads, backbone
bits set resident memory and cost very little on disk.

A component setting is one of
    4                      affine, that many bits, the recipe's group size
    "keep"                 not quantized (cast to the recipe dtype)
    "model"                whatever the architecture asks for (routers)
    {"bits": 8, "group_size": 32, "mode": "affine"}

Rules come first (regex on the module path, first match wins), then the
component, then the edge-layer bump. A module the architecture marks as not
quantizable always stays float, whatever the recipe says.

Pure Python, no mlx import, so the UI and the CLI can use it freely.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields

COMPONENTS = (
    "experts", "experts_down", "shared_experts", "attention",
    "dense_mlp", "router", "embed", "lm_head", "other",
)

LABELS = {
    "experts": "Routed experts (gate, up)",
    "experts_down": "Routed experts (down)",
    "shared_experts": "Shared experts",
    "attention": "Attention",
    "dense_mlp": "Dense MLP",
    "router": "Router",
    "embed": "Embeddings",
    "lm_head": "Output head",
    "other": "Everything else",
}

EXPERT_COMPONENTS = ("experts", "experts_down")

MODES = {
    # mode: (group size, bits) it is fixed to, None for affine
    "affine": None,
    "mxfp4": (32, 4),
    "mxfp8": (32, 8),
    "nvfp4": (16, 4),
}
AFFINE_BITS = (2, 3, 4, 5, 6, 8)
GROUP_SIZES = (32, 64, 128)
DTYPES = ("auto", "bfloat16", "float16", "float32")

PRESETS: dict[str, dict] = {
    "balanced": {
        "about": "4-bit experts, 6-bit everything else. The backbone is a small share "
                 "of the bytes, so the extra bits there are almost free on disk.",
        "experts": 4, "shared_experts": 6, "attention": 6, "dense_mlp": 6,
        "embed": 6, "lm_head": 6, "other": 6,
    },
    "standard": {
        "about": "4-bit across the board, same as a typical mlx-community build.",
        "experts": 4, "shared_experts": 4, "attention": 4, "dense_mlp": 4,
        "embed": 4, "lm_head": 4, "other": 4,
    },
    "quality": {
        "about": "6-bit experts, 8-bit backbone. Bigger on disk, closest to the original "
                 "short of 8-bit everywhere.",
        "experts": 6, "shared_experts": 8, "attention": 8, "dense_mlp": 8,
        "embed": 8, "lm_head": 8, "other": 8,
    },
    "max": {
        "about": "8-bit everything.",
        "experts": 8, "shared_experts": 8, "attention": 8, "dense_mlp": 8,
        "embed": 8, "lm_head": 8, "other": 8,
    },
    "compact": {
        "about": "3-bit experts, 4-bit backbone. Smallest that still holds up on most "
                 "big MoEs. Check the fidelity report.",
        "experts": 3, "shared_experts": 4, "attention": 4, "dense_mlp": 4,
        "embed": 4, "lm_head": 4, "other": 4,
    },
    "float": {
        "about": "No quantization. Casts to the recipe dtype, for merging an adapter "
                 "into full-precision weights.",
        "experts": "keep", "shared_experts": "keep", "attention": "keep", "dense_mlp": "keep",
        "router": "keep", "embed": "keep", "lm_head": "keep", "other": "keep",
    },
    "fit": {
        "about": "Picks the best preset that fits this Mac: fully resident if one does, "
                 "else the best quality that streams and fits the disk.",
    },
}

# best quality first, what "fit" tries
FIT_ORDER = ("max", "quality", "balanced", "standard", "compact")

# fit targets: "auto" takes resident when anything fits, else streamed
TARGETS = ("auto", "resident", "streamed")


class RecipeError(ValueError):
    pass


@dataclass
class Recipe:
    preset: str = "balanced"
    mode: str = "affine"
    group_size: int = 64
    dtype: str = "auto"
    experts: object = 4
    experts_down: object = None  # None: same as experts
    shared_experts: object = 6
    attention: object = 6
    dense_mlp: object = 6
    router: object = "model"
    embed: object = 6
    lm_head: object = 6
    other: object = 6
    # first and last N decoder layers get at least edge_bits
    edge_layers: int = 0
    edge_bits: int = 8
    edge_experts: bool = False
    rules: list = field(default_factory=list)
    # only for the "fit" preset
    target: str = "auto"

    @classmethod
    def from_preset(cls, name: str = "balanced", **overrides) -> "Recipe":
        if name not in PRESETS:
            raise RecipeError(f"unknown preset {name!r}, pick one of {', '.join(PRESETS)}")
        vals = {k: v for k, v in PRESETS[name].items() if k != "about"}
        vals.update({k: v for k, v in overrides.items() if v is not None})
        r = cls(preset=name, **vals)
        r.validate()
        return r

    @classmethod
    def from_dict(cls, d: dict | None) -> "Recipe":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = sorted(set(d) - known)
        if bad:
            raise RecipeError(f"unknown recipe keys: {', '.join(bad)}")
        preset = d.pop("preset", None) or "balanced"
        return cls.from_preset(preset, **d)

    def to_dict(self) -> dict:
        return asdict(self)

    def with_preset(self, name: str) -> "Recipe":
        """Same rules, edges, mode and dtype, components from another preset."""
        d = self.to_dict()
        for c in COMPONENTS:
            d.pop(c, None)
        d["preset"] = name
        return Recipe.from_dict(d)

    def validate(self) -> None:
        if self.mode not in MODES:
            raise RecipeError(f"mode must be one of {', '.join(MODES)}")
        if self.mode == "affine" and self.group_size not in GROUP_SIZES:
            raise RecipeError(f"group_size must be one of {GROUP_SIZES}")
        if self.dtype not in DTYPES:
            raise RecipeError(f"dtype must be one of {', '.join(DTYPES)}")
        if self.target not in TARGETS:
            raise RecipeError(f"target must be one of {', '.join(TARGETS)}")
        if self.edge_layers < 0:
            raise RecipeError("edge_layers can't be negative")
        if self.edge_bits not in AFFINE_BITS:
            raise RecipeError(f"edge_bits must be one of {AFFINE_BITS}")
        for c in COMPONENTS:
            v = getattr(self, c)
            if c == "experts_down" and v is None:
                continue
            _check_spec(v, c)
        for i, r in enumerate(self.rules):
            if not isinstance(r, dict) or not r.get("match"):
                raise RecipeError(f"rule {i + 1} needs a 'match' regex")
            try:
                re.compile(r["match"])
            except re.error as e:
                raise RecipeError(f"rule {i + 1}: bad regex {r['match']!r}: {e}") from None
            _check_spec(rule_spec(r), f"rule {i + 1}")

    def spec(self, component: str):
        v = getattr(self, component)
        if component == "experts_down" and v is None:
            v = self.experts
        return v

    def describe(self, component: str) -> str:
        return describe_spec(self.spec(component), self)


def _check_spec(v, where: str) -> None:
    if v in ("keep", "model"):
        return
    if isinstance(v, bool):
        raise RecipeError(f"{where}: use a bit count or \"keep\", not {v}")
    if isinstance(v, int):
        if v not in AFFINE_BITS:
            raise RecipeError(f"{where}: bits must be one of {AFFINE_BITS}")
        return
    if isinstance(v, dict):
        mode = v.get("mode", "affine")
        if mode not in MODES:
            raise RecipeError(f"{where}: mode must be one of {', '.join(MODES)}")
        if mode == "affine":
            if v.get("bits") not in AFFINE_BITS:
                raise RecipeError(f"{where}: bits must be one of {AFFINE_BITS}")
            if v.get("group_size", 64) not in GROUP_SIZES:
                raise RecipeError(f"{where}: group_size must be one of {GROUP_SIZES}")
        return
    raise RecipeError(f"{where}: {v!r} isn't a bit count, \"keep\", \"model\" or a dict")


def rule_spec(r: dict):
    """A rule's setting: {"match": regex, "bits": 8 | "keep" | "model"},
    optionally with "group_size", or {"match": regex, "mode": "mxfp8"}."""
    bits = r.get("bits")
    if bits in ("keep", "model"):
        return bits
    mode = r.get("mode")
    if mode and MODES.get(mode):
        return {"mode": mode}
    if bits is None:
        raise RecipeError(f"rule {r.get('match')!r} needs 'bits' (a number, \"keep\" or \"model\")")
    if "group_size" in r or mode == "affine":
        return {"bits": bits, "group_size": r.get("group_size", 64), "mode": "affine"}
    return bits


def params_for(spec, recipe: Recipe) -> dict | None:
    """Quantization params for a resolved spec, None for float."""
    if spec == "keep" or spec is None:
        return None
    if isinstance(spec, dict):
        mode = spec.get("mode", "affine")
        if MODES.get(mode):
            gs, bits = MODES[mode]
            return {"group_size": gs, "bits": bits, "mode": mode}
        return {"group_size": int(spec.get("group_size", recipe.group_size)),
                "bits": int(spec["bits"]), "mode": "affine"}
    if MODES.get(recipe.mode):
        gs, bits = MODES[recipe.mode]
        return {"group_size": gs, "bits": bits, "mode": recipe.mode}
    return {"group_size": recipe.group_size, "bits": int(spec), "mode": "affine"}


def describe_params(p: dict | None) -> str:
    if p is None:
        return "float"
    if p["mode"] != "affine":
        return p["mode"]
    return f"{p['bits']}-bit g{p['group_size']}"


def describe_spec(spec, recipe: Recipe) -> str:
    if spec == "model":
        return "model's choice"
    return describe_params(params_for(spec, recipe))


# ------------------------------------------------------------ module paths

_EMBED = {"embed_tokens", "wte", "tok_embeddings", "word_embeddings", "embeddings", "embedding"}
_HEAD = {"lm_head", "output", "embed_out"}
_ROUTER = {"gate", "router", "wg", "gate_network", "shared_expert_gate", "e_score_gate"}
_DOWN = {"down_proj", "w2", "fc2", "down"}
_ATTN = re.compile(r"(^|\.)(self_attn|attention|attn|linear_attn|mixer|token_mixer|cross_attn)(\.|$)")
_MLP = re.compile(r"(^|\.)(mlp|feed_forward|ffn|ffw|block_sparse_moe|moe)(\.|$)")
_EXPERT = re.compile(r"(^|\.)(switch_mlp|switch_glu|experts)(\.|$)")
LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)\.")


def component_of(path: str, ndim: int) -> str:
    """Which recipe component a quantizable module belongs to."""
    last = path.rsplit(".", 1)[-1]
    in_layer = LAYER_RE.search(path + ".") is not None
    if last in _HEAD and not in_layer:
        return "lm_head"
    if last in _EMBED:
        return "embed"
    if last in _ROUTER:
        return "router"
    if "shared_expert" in path:
        return "shared_experts"
    if ndim >= 3 or _EXPERT.search(path):
        return "experts_down" if last in _DOWN else "experts"
    if _ATTN.search(path):
        return "attention"
    if _MLP.search(path):
        return "dense_mlp"
    return "other"


def layer_of(path: str) -> int | None:
    m = LAYER_RE.search(path + ".")
    return int(m.group(1)) if m else None


def resolve(recipe: Recipe, path: str, ndim: int, n_layers: int, model_choice=True):
    """(params or None, component, why) for one quantizable module.

    model_choice is what the architecture's quant_predicate said: False means
    it must stay float, a dict is its own setting (usually 8-bit routers)."""
    comp = component_of(path, ndim)
    if model_choice is False:
        return None, comp, "kept float by the architecture"
    spec, why = None, None
    for r in recipe.rules:
        if re.search(r["match"], path):
            spec, why = rule_spec(r), f"rule {r['match']}"
            break
    if spec is None:
        spec = recipe.spec(comp)
        why = comp
    if spec == "model":
        if isinstance(model_choice, dict):
            p = dict(model_choice)
            p.setdefault("mode", "affine")
            return p, comp, "architecture's setting"
        spec = recipe.spec("other")
    p = params_for(spec, recipe)
    if p is not None and p["mode"] == "affine" and recipe.edge_layers > 0:
        idx = layer_of(path)
        if idx is not None and n_layers and (comp not in EXPERT_COMPONENTS or recipe.edge_experts):
            if idx < recipe.edge_layers or idx >= n_layers - recipe.edge_layers:
                if recipe.edge_bits > p["bits"]:
                    p = {**p, "bits": recipe.edge_bits}
                    why = f"{why}, edge layer"
    return p, comp, why


# ------------------------------------------------------------ sizes

_ITEM = {
    "bool": 1, "uint8": 1, "int8": 1, "uint16": 2, "int16": 2, "float16": 2, "bfloat16": 2,
    "uint32": 4, "int32": 4, "float32": 4, "uint64": 8, "int64": 8, "float64": 8,
}


def itemsize(dtype: str) -> int:
    return _ITEM[str(dtype).replace("mlx.core.", "")]


def quantized_tensors(shape: tuple, p: dict, float_dtype: str) -> dict:
    """{suffix: (shape, dtype)} that quantizing a weight of `shape` writes."""
    lead, width = tuple(shape[:-1]), int(shape[-1])
    gs, bits = p["group_size"], p["bits"]
    out = {"weight": (lead + (width * bits // 32,), "uint32")}
    if p["mode"] == "affine":
        out["scales"] = (lead + (width // gs,), float_dtype)
        out["biases"] = (lead + (width // gs,), float_dtype)
    else:
        out["scales"] = (lead + (width // gs,), "uint8")
    return out


def nbytes(shape, dtype: str) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n * itemsize(dtype)
