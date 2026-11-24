# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plan a conversion from the headers alone.

The architecture's own sanitize() runs on stand-in arrays: lazy mx.zeros with
the real shapes and dtypes, never evaluated, so nothing is read and nothing
is allocated. What comes out is exactly the tensor set mlx-lm would load,
checked strictly against the model class like a real load. That catches a
wrong or unsupported checkpoint in seconds instead of an hour into a 400 GB
download, and it gives every output tensor its shape, dtype and byte offset
before the first byte is read.

Three views of the weights show up here:
    source   names as stored in the checkpoint
    loaded   after sanitize(), in the source's quantization (if any)
    float    the loaded view with source quantization undone; recipes,
             adapters and sizes all work on this one
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import recipe as R
from .source import Source, SourceError, open_source

GiB = 1 << 30


class PlanError(RuntimeError):
    pass


@dataclass
class OutTensor:
    name: str
    shape: tuple
    dtype: str
    nbytes: int
    unit: int
    module: str | None = None  # set for the tensors of a quantized module


@dataclass
class ModulePlan:
    path: str
    kind: str  # linear | switch | embedding
    component: str
    params: dict | None  # None: stays float
    why: str
    shape: tuple  # float weight shape
    dtype: str  # dtype the weight is quantized from (or written as)
    src_quant: dict | None = None
    adapter: bool = False
    choice: object = True  # the architecture's quant_predicate answer
    quantizable: bool = True


@dataclass
class Unit:
    """One step of the conversion: a decoder layer, or the tensors outside them."""

    key: str
    floats: list  # float-view names this unit writes
    src: list = field(default_factory=list)  # source names (windowed mode)
    files: list = field(default_factory=list)  # source shards it reads (windowed mode)
    out_bytes: int = 0
    float_bytes: int = 0


@dataclass
class Plan:
    source: Source
    recipe: R.Recipe
    config: dict
    model_type: str
    arch: str
    float_dtype: str
    loaded: dict  # name -> (shape, dtype), after sanitize
    floats: dict  # name -> (shape, dtype), source quantization undone
    src_quant: dict  # module path -> params, for a quantized source
    modules: dict  # module path -> ModulePlan
    units: list
    layout: list  # OutTensor, in write order
    windowed: bool
    separable: bool
    n_layers: int
    adapter: dict | None = None
    summary_cache: dict | None = None
    cast_ok: dict = field(default_factory=dict)  # name -> may cast to float_dtype
    warnings: list = field(default_factory=list)
    out_dir: str | None = None
    ctx: int = 32768
    shard_gb: float = 5.0
    model: object = None  # skeleton from the plan, for sanitize() at run time

    def quant_config(self) -> dict | None:
        """config["quantization"] for the output, mlx-lm's layout."""
        counts: dict[tuple, int] = {}
        for m in self.modules.values():
            if m.params is not None:
                key = (m.params["group_size"], m.params["bits"], m.params["mode"])
                counts[key] = counts.get(key, 0) + math.prod(m.shape)
        if not counts:
            return None
        gs, bits, mode = max(counts, key=counts.get)
        q: dict = {"group_size": gs, "bits": bits, "mode": mode}
        for path, m in sorted(self.modules.items()):
            if m.params is None:
                continue
            if (m.params["group_size"], m.params["bits"], m.params["mode"]) != (gs, bits, mode):
                q[path] = dict(m.params)
        return q

    def summary(self) -> dict:
        if self.summary_cache is None:
            self.summary_cache = _summarize(self)
        return self.summary_cache


# ------------------------------------------------------------ model


def _text_cfg(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg


def _first(cfg: dict, *keys, default=0):
    tc = _text_cfg(cfg)
    for k in keys:
        if tc.get(k) is not None:
            return tc[k]
        if cfg.get(k) is not None:
            return cfg[k]
    return default


def _arch(model_type: str, meta_dir: Path) -> str:
    from .. import local_arch

    try:
        from mlx_lm.utils import MODEL_REMAPPING

        mt = MODEL_REMAPPING.get(model_type, model_type)
    except ImportError:
        mt = model_type
    if local_arch._mlx_provides(f"mlx_lm.models.{mt}"):
        return "mlx-lm"
    try:
        local_arch.register(meta_dir)
    except Exception as e:  # noqa: BLE001
        raise PlanError(f"the bundled {model_type} architecture failed to load: {e}") from e
    import sys

    mod = sys.modules.get(f"mlx_lm.models.{mt}")
    if mod is None:
        import mlx_lm

        raise PlanError(
            f"{model_type} isn't an architecture mlx-lm {getattr(mlx_lm, '__version__', '?')} knows, "
            "and PagedMoE has no bundled port for it. Newer mlx-lm releases add architectures "
            "often, so check for an update.")
    origin = str(getattr(mod, "__file__", "") or "")
    return "bundled port" if "/arches/" in origin else "checkpoint file"


def _model(config: dict):
    import mlx.nn as nn  # noqa: F401
    from mlx_lm.utils import _get_classes

    cfg = dict(config)
    if "quantization_config" not in cfg and "quantization_config" in (cfg.get("text_config") or {}):
        cfg["quantization_config"] = cfg["text_config"]["quantization_config"]
    try:
        model_cls, args_cls = _get_classes(cfg)
    except ValueError as e:
        raise PlanError(str(e)) from e
    try:
        return model_cls(args_cls.from_dict(cfg))
    except Exception as e:  # noqa: BLE001
        raise PlanError(f"couldn't build the {cfg.get('model_type')} model from its config: "
                        f"{type(e).__name__}: {e}") from e


def _source_quant(config: dict) -> dict | None:
    """The source's quantization in mlx-lm's form, None for float sources."""
    q = config.get("quantization")
    if q is not None:
        return q
    qc = config.get("quantization_config") or (config.get("text_config") or {}).get("quantization_config")
    if not qc:
        return None
    method = qc.get("quant_method")
    if method == "fp8" or (method is None and qc.get("fmt") in ("e4m3", "e5m2")):
        return None  # sanitize dequantizes fp8 blocks
    if method == "mxfp4":
        return {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    if method == "compressed-tensors":
        return {"group_size": 32, "bits": 4, "mode": "affine"}
    if method in ("awq", "gptq"):
        raise PlanError(f"{method.upper()} sources aren't supported yet. Use the original "
                        "(bf16 or fp8) repo of the model.")
    raise PlanError(f"source quantization {method!r} isn't supported. Use the original repo of the model.")


def _standins(src: Source) -> dict:
    import mlx.core as mx

    return {n: mx.zeros(t.shape, dtype=getattr(mx, t.dtype)) for n, t in src.tensors.items()}


def _sanitize(model, weights: dict) -> dict:
    if hasattr(model, "sanitize"):
        return model.sanitize(weights)
    return weights


def _shape_dtype(a) -> tuple:
    return tuple(int(s) for s in a.shape), str(a.dtype).replace("mlx.core.", "")


def _unit_key(name: str) -> str:
    m = R.LAYER_RE.search(name)
    return name[: m.end() - 1] if m else "other"


def _unit_order(keys) -> list:
    """'other' first (embeddings, norms, head), then layers in order."""
    keys = list(keys)
    first_seen: dict[str, int] = {}
    for k in keys:
        first_seen.setdefault(k.rsplit(".", 1)[0] if k != "other" else "", len(first_seen))

    def sort_key(k):
        if k == "other":
            return (-1, 0, k)
        prefix, _, idx = k.rpartition(".")
        return (first_seen.get(prefix, 99), int(idx) if idx.isdigit() else 0, k)

    return sorted(set(keys), key=sort_key)


# ------------------------------------------------------------ adapter


def _load_adapter(path: str) -> dict:
    p = Path(path).expanduser()
    if p.is_dir() and not (p / "adapter_config.json").exists():
        latest = p / "checkpoints" / "latest"
        if latest.exists():
            p = p / "checkpoints" / latest.read_text().strip()
    cfg_p, w_p = p / "adapter_config.json", p / "adapters.safetensors"
    if not cfg_p.exists() or not w_p.exists():
        raise PlanError(f"{p} isn't an adapter folder (adapter_config.json + adapters.safetensors)")
    cfg = json.loads(cfg_p.read_text())
    kind = cfg.get("fine_tune_type") or "lora"
    if kind not in ("lora", "dora"):
        raise PlanError(f"only LoRA and DoRA adapters merge, this one is {kind!r}")
    lp = cfg.get("lora_parameters") or {}
    scale = lp.get("scale")
    if scale is None:
        raise PlanError("adapter_config.json has no lora_parameters.scale")
    import mlx.core as mx

    weights = mx.load(str(w_p))
    mods: dict[str, dict] = {}
    for k, v in weights.items():
        mod, _, leaf = k.rpartition(".")
        if leaf in ("lora_a", "lora_b", "m"):
            mods.setdefault(mod, {})[leaf] = v
    if not mods:
        raise PlanError(f"{w_p} has no LoRA tensors")
    return {"path": str(p), "kind": kind, "scale": float(scale), "rank": lp.get("rank"),
            "modules": mods, "base_model": cfg.get("base_model") or cfg.get("model")}


def _check_adapter(adapter: dict, floats: dict, kinds: dict) -> list:
    """Adapter modules that don't match the model: wrong names or shapes."""
    bad = []
    for path, t in adapter["modules"].items():
        w = floats.get(path + ".weight")
        if w is None or "lora_a" not in t or "lora_b" not in t:
            bad.append(f"{path}: not in this model")
            continue
        shape, a, b = w[0], t["lora_a"].shape, t["lora_b"].shape
        kind = kinds.get(path)
        if kind == "linear":
            ok = len(shape) == 2 and a[0] == shape[1] and b[-1] == shape[0]
        elif kind == "switch":
            ok = len(shape) == 3 and a[-1] == shape[2] and b[0] == shape[0] and b[1] == shape[1]
        elif kind == "embedding":
            ok = len(shape) == 2 and a[0] == shape[0] and b[-1] == shape[1]
            if adapter["kind"] == "dora":
                bad.append(f"{path}: DoRA on embeddings doesn't merge yet")
                continue
        else:
            ok = False
        if not ok:
            bad.append(f"{path}: adapter shapes {tuple(a)} / {tuple(b)} don't fit weight {shape}")
    return bad


# ------------------------------------------------------------ build


def build(source, recipe: R.Recipe | dict | str | None = None, *, out_dir: str | None = None,
          adapter: str | None = None, stream: bool | None = None, revision: str | None = None,
          ctx: int = 32768, shard_gb: float = 5.0) -> Plan:
    """Everything about a conversion that can be known before reading weights."""
    import mlx.core as mx

    if isinstance(recipe, str):
        recipe = R.Recipe.from_preset(recipe)
    elif isinstance(recipe, dict) or recipe is None:
        recipe = R.Recipe.from_dict(recipe)
    recipe.validate()
    src = source if isinstance(source, Source) else open_source(str(source), revision)

    dev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        return _build(src, recipe, out_dir, adapter, stream, ctx, shard_gb)
    finally:
        mx.set_default_device(dev)


def _build(src: Source, recipe: R.Recipe, out_dir, adapter_path, stream, ctx, shard_gb) -> Plan:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.utils import dequantize_model

    config = dict(src.config)
    model_type = str(config.get("model_type") or "?")
    if config.get("model_file"):
        raise PlanError("checkpoints with a custom model_file aren't supported")
    arch = _arch(model_type, src.meta_dir)
    model = _model(config)
    sq = _source_quant(config)

    standins = _standins(src)
    try:
        loaded = _sanitize(model, dict(standins))
    except Exception as e:  # noqa: BLE001
        raise PlanError(f"the {model_type} loader rejected this checkpoint: {type(e).__name__}: {e}") from e
    loaded_meta = {k: _shape_dtype(v) for k, v in loaded.items()}

    # source quantization, the way mlx-lm's load_model decides it
    src_quant: dict[str, dict] = {}
    if sq is not None:
        qcfg = config.get("quantization") or {}

        def class_predicate(p, m):
            if p in qcfg and not isinstance(qcfg[p], bool):
                return qcfg[p]
            if p in qcfg and qcfg[p] is False:
                return False
            if not hasattr(m, "to_quantized"):
                return False
            return f"{p}.scales" in loaded_meta

        nn.quantize(model, group_size=sq["group_size"], bits=sq["bits"], mode=sq.get("mode", "affine"),
                    class_predicate=class_predicate)
        for path, m in model.named_modules():
            if hasattr(m, "bits") and hasattr(m, "group_size") and f"{path}.scales" in loaded_meta:
                src_quant[path] = {"group_size": m.group_size, "bits": m.bits,
                                   "mode": getattr(m, "mode", "affine")}

    expected = {k: _shape_dtype(v) for k, v in tree_flatten(model.parameters())}
    missing = sorted(set(expected) - set(loaded_meta))
    extra = sorted(set(loaded_meta) - set(expected))
    wrong = sorted(k for k in set(expected) & set(loaded_meta) if expected[k][0] != loaded_meta[k][0])
    if missing or extra or wrong:
        lines = []
        if missing:
            lines.append(f"{len(missing)} tensors the model needs aren't in the checkpoint, e.g. {missing[:3]}")
        if extra:
            lines.append(f"{len(extra)} checkpoint tensors the model doesn't use, e.g. {extra[:3]}")
        if wrong:
            k = wrong[0]
            lines.append(f"{len(wrong)} shapes differ, e.g. {k}: checkpoint {loaded_meta[k][0]}, "
                         f"model {expected[k][0]}")
        raise PlanError(f"this checkpoint doesn't match the {model_type} architecture: " + "; ".join(lines))

    # the float view
    if src_quant:
        model = dequantize_model(model)
    floats: dict[str, tuple] = {}
    for k, (shape, dt) in loaded_meta.items():
        mod, _, leaf = k.rpartition(".")
        if mod in src_quant and leaf in ("scales", "biases"):
            continue
        if mod in src_quant and leaf == "weight":
            q = src_quant[mod]
            scale_dt = loaded_meta[f"{mod}.scales"][1]
            fdt = scale_dt if scale_dt in ("bfloat16", "float16", "float32") else "bfloat16"
            floats[k] = (shape[:-1] + (shape[-1] * 32 // q["bits"],), fdt)
            continue
        floats[k] = (shape, dt)

    float_dtype = _float_dtype(recipe, config)
    cast_pred = getattr(model, "cast_predicate", None)
    quant_pred = getattr(model, "quant_predicate", None)
    n_layers = int(_first(config, "num_hidden_layers", "n_layers", "num_layers", default=0))

    cast_ok = {}
    for k, (shape, dt) in floats.items():
        ok = dt in ("bfloat16", "float16", "float32")
        if ok and cast_pred is not None:
            try:
                ok = bool(cast_pred(k))
            except Exception:  # noqa: BLE001
                ok = True
        cast_ok[k] = ok

    # modules the recipe can quantize
    modules: dict[str, ModulePlan] = {}
    kinds: dict[str, str] = {}
    from mlx_lm.models.switch_layers import SwitchLinear

    for path, m in model.named_modules():
        wname = f"{path}.weight"
        if wname not in floats:
            continue
        if isinstance(m, SwitchLinear):
            kind = "switch"
        elif isinstance(m, nn.Embedding):
            kind = "embedding"
        elif isinstance(m, nn.Linear):
            kind = "linear"
        else:
            continue
        kinds[path] = kind
        if not hasattr(m, "to_quantized"):
            continue
        shape, dt = floats[wname]
        choice = True
        if quant_pred is not None:
            try:
                choice = quant_pred(path, m)
            except Exception:  # noqa: BLE001
                choice = True
        modules[path] = ModulePlan(path, kind, "other", None, "", shape, dt, src_quant.get(path),
                                   choice=choice)

    adapter = None
    if adapter_path:
        adapter = _load_adapter(adapter_path)
        bad = _check_adapter(adapter, floats, kinds)
        if bad:
            raise PlanError(f"the adapter doesn't fit this model ({len(bad)} modules), e.g. {bad[0]}. "
                            "Was it trained on a different model?")
        for p in adapter["modules"]:
            if p in modules:
                modules[p].adapter = True
            else:
                shape, dt = floats[p + ".weight"]
                modules[p] = ModulePlan(p, kinds.get(p, "linear"), "other", None, "", shape, dt,
                                        src_quant.get(p), adapter=True, quantizable=False)

    # units: windowed (per source layer) when streaming a repo, else per output layer
    remote = src.kind == "hub" and src.local_dir is None
    want_stream = remote and (stream is None or bool(stream))
    separable, units = _windows(model, src, standins, loaded_meta, src_quant) if want_stream else (False, None)
    windowed = bool(want_stream and separable)
    if not windowed:
        keys = _unit_order(_unit_key(n) for n in floats)
        by_key: dict[str, list] = {k: [] for k in keys}
        for n in floats:
            by_key[_unit_key(n)].append(n)
        units = [Unit(k, sorted(by_key[k])) for k in keys]

    plan = Plan(
        source=src, recipe=recipe, config=config, model_type=model_type, arch=arch,
        float_dtype=float_dtype, loaded=loaded_meta, floats=floats, src_quant=src_quant,
        modules=modules, units=units, layout=[], windowed=windowed, separable=separable,
        n_layers=n_layers, adapter=adapter, cast_ok=cast_ok, out_dir=out_dir,
        ctx=ctx, shard_gb=shard_gb, model=model,
    )
    if want_stream and not separable:
        plan.warnings.append(
            f"the {model_type} loader needs the whole checkpoint at once, so the shards are "
            "downloaded first instead of one layer at a time")
    _assign(plan)
    return plan


def _float_dtype(recipe: R.Recipe, config: dict) -> str:
    if recipe.dtype != "auto":
        return recipe.dtype
    tdt = str(_first(config, "torch_dtype", "dtype", default="bfloat16")).replace("torch.", "")
    return tdt if tdt in ("bfloat16", "float16") else "bfloat16"


def _assign(plan: Plan) -> None:
    """Resolve the recipe onto every module and lay out the output."""
    rec = plan.recipe
    for m in plan.modules.values():
        wname = f"{m.path}.weight"
        src_dt = plan.floats[wname][1]
        m.dtype = plan.float_dtype if plan.cast_ok.get(wname) else src_dt
        m.component = R.component_of(m.path, len(m.shape))
        if not m.quantizable:
            m.params, m.why = None, "not quantizable"
            continue
        params, comp, why = R.resolve(rec, m.path, len(m.shape), plan.n_layers, m.choice)
        if params is not None and m.shape[-1] % params["group_size"]:
            params, why = None, f"width {m.shape[-1]} isn't a multiple of group size {params['group_size']}"
        if params is not None and m.dtype not in ("bfloat16", "float16", "float32"):
            params, why = None, f"{m.dtype} weight"
        m.params, m.component, m.why = params, comp, why
    plan.layout = _layout(plan)
    plan.summary_cache = None


def retarget(plan: Plan, recipe: R.Recipe) -> Plan:
    """The same checkpoint under another recipe, without rebuilding anything."""
    import copy

    p = copy.copy(plan)
    p.recipe = recipe
    p.float_dtype = _float_dtype(recipe, plan.config)
    p.modules = {k: copy.copy(m) for k, m in plan.modules.items()}
    p.units = [copy.copy(u) for u in plan.units]
    p.warnings = list(plan.warnings)
    _assign(p)
    return p


def _windows(model, src: Source, standins: dict, loaded_meta: dict, src_quant: dict):
    """Can sanitize() run one source layer at a time and give the same result?

    Only then can a repo stream: fetch a layer's shards, convert, delete. Most
    sanitize() functions look at layer 0 (or the embeddings) to decide what to
    do, so every window also carries the first group of tensors: the ones
    outside the layers plus layer 0. Those are real tensors at run time (their
    shards stay until the end), so even a sanitize that reads them gives
    exact results. A window writes only what the probe alone doesn't produce,
    and the union over windows has to equal the whole-checkpoint run."""
    groups: dict[str, list] = {}
    for n in src.tensors:
        groups.setdefault(_unit_key(n), []).append(n)
    keys = _unit_order(groups)
    layer_keys = [k for k in keys if k != "other"]
    probe_keys = [k for k in keys if k == "other"] + layer_keys[:1]
    probe = sorted(n for k in probe_keys for n in groups[k])
    seen: dict[str, str] = {}
    units = []

    def files_of(names):
        return sorted({src.tensors[n].file for n in names})

    def take(part, skip, key):
        names = []
        for name, arr in part.items():
            if name in skip:
                continue
            if name in seen or loaded_meta.get(name, (None,))[0] != _shape_dtype(arr)[0]:
                return None
            seen[name] = key
            names.append(name)
        return names

    try:
        base = _sanitize(model, {n: standins[n] for n in probe})
        names = take(base, set(), "other")
        if names is None:
            return False, None
        base_names = set(base)
        floats = sorted({_float_name(n, src_quant) for n in names} - {None})
        units.append(Unit("other", floats, probe, files_of(probe)))
        for k in keys:
            if k in probe_keys:
                continue
            src_names = sorted(set(groups[k]) | set(probe))
            part = _sanitize(model, {n: standins[n] for n in src_names})
            names = take(part, base_names, k)
            if names is None:
                return False, None
            if not names:
                continue  # dropped by the loader (vision tower, MTP layer): never downloaded
            floats = sorted({_float_name(n, src_quant) for n in names} - {None})
            units.append(Unit(k, floats, src_names, files_of(src_names)))
    except Exception:  # noqa: BLE001 - any failure means "not separable"
        return False, None
    if set(seen) != set(loaded_meta):
        return False, None
    return True, units


def _float_name(name: str, src_quant: dict) -> str | None:
    mod, _, leaf = name.rpartition(".")
    if mod in src_quant and leaf in ("scales", "biases"):
        return None
    return name


def _layout(plan: Plan) -> list:
    out = []
    for i, u in enumerate(plan.units):
        tensors = []
        for name in u.floats:
            shape, dt = plan.floats[name]
            mod, _, leaf = name.rpartition(".")
            m = plan.modules.get(mod)
            if m is not None and leaf == "weight" and m.params is not None:
                for suffix, (s, d) in R.quantized_tensors(shape, m.params, m.dtype).items():
                    tensors.append(OutTensor(f"{mod}.{suffix}", s, d, R.nbytes(s, d), i, mod))
                continue
            d = plan.float_dtype if plan.cast_ok.get(name) else dt
            tensors.append(OutTensor(name, shape, d, R.nbytes(shape, d), i))
        u.float_bytes = sum(R.nbytes(*plan.floats[n]) for n in u.floats)
        tensors.sort(key=lambda t: t.name)
        u.out_bytes = sum(t.nbytes for t in tensors)
        out.extend(tensors)
    return out


# ------------------------------------------------------------ summary


def _free_bytes(path: str | Path | None) -> int | None:
    p = Path(path).expanduser() if path else None
    while p is not None and not p.exists():
        p = p.parent if p.parent != p else None
    try:
        return shutil.disk_usage(p or Path.home()).free
    except OSError:
        return None


def _summarize(plan: Plan) -> dict:
    rec = plan.recipe
    comps: dict[str, dict] = {}
    numel_total = 0

    def row(c):
        return comps.setdefault(c, {"component": c, "label": R.LABELS.get(c, c), "modules": 0, "params": 0,
                                    "float_bytes": 0, "out_bytes": 0, "settings": {}, "kept": {}})

    by_name = {t.name: t for t in plan.layout}
    for path, m in plan.modules.items():
        r = row(m.component)
        n = math.prod(m.shape)
        r["modules"] += 1
        r["params"] += n
        r["float_bytes"] += R.nbytes(m.shape, plan.floats[path + ".weight"][1])
        if m.params is not None:
            for suffix in ("weight", "scales", "biases"):
                t = by_name.get(f"{path}.{suffix}")
                if t:
                    r["out_bytes"] += t.nbytes
            key = R.describe_params(m.params)
            r["settings"][key] = r["settings"].get(key, 0) + n
        else:
            r["out_bytes"] += by_name[path + ".weight"].nbytes
            r["kept"][m.why] = r["kept"].get(m.why, 0) + 1
            r["settings"]["float"] = r["settings"].get("float", 0) + n
        b = by_name.get(path + ".bias")
        if b:
            r["out_bytes"] += b.nbytes
    small = {"component": "small", "label": "Norms, biases and small tensors", "modules": 0, "params": 0,
             "float_bytes": 0, "out_bytes": 0, "settings": {}, "kept": {}}
    covered = set()
    for path in plan.modules:
        covered.update({f"{path}.weight", f"{path}.bias"})
    for name, (shape, dt) in plan.floats.items():
        n = math.prod(shape)
        numel_total += n
        if name in covered:
            continue
        small["params"] += n
        small["float_bytes"] += R.nbytes(shape, dt)
        small["out_bytes"] += by_name[name].nbytes if name in by_name else 0
    rows = []
    for c in R.COMPONENTS:
        if c in comps:
            r = comps[c]
            r["setting"] = " + ".join(k for k, _ in sorted(r["settings"].items(), key=lambda kv: -kv[1]))
            r["kept"] = [{"why": k, "modules": v} for k, v in r["kept"].items()]
            del r["settings"]
            rows.append(r)
    if small["params"]:
        small["setting"] = plan.float_dtype
        small["kept"] = []
        del small["settings"]
        rows.append(small)

    out_bytes = sum(t.nbytes for t in plan.layout)
    expert_out = sum(r["out_bytes"] for r in rows if r["component"] in R.EXPERT_COMPONENTS)
    float_bytes = sum(R.nbytes(*v) for v in plan.floats.values())
    cfg = plan.config
    n_exp = int(_first(cfg, "num_experts", "n_routed_experts", "num_local_experts", default=0) or 0)
    top_k = int(_first(cfg, "num_experts_per_tok", "moe_top_k", "top_k", default=0) or 0)
    active_read = int(expert_out * top_k / n_exp) if n_exp and top_k else 0

    fit = _fit(plan, out_bytes, out_bytes - expert_out, expert_out, active_read)
    disk = _disk(plan, out_bytes)
    peak = _peak(plan)

    warnings = list(plan.warnings)
    for r in rows:
        if r["component"] in ("attention", "router", "embed", "lm_head") and any(
                s in r.get("setting", "") for s in ("2-bit", "3-bit")):
            warnings.append(f"{r['label'].lower()} under 4 bits usually hurts quality more than it saves")
    if any(s in (r.get("setting") or "") for r in rows if r["component"] in R.EXPERT_COMPONENTS
           for s in ("2-bit",)):
        warnings.append("2-bit experts lose a lot on most models, check the fidelity report")
    if plan.src_quant:
        warnings.append("the source is already quantized, so this quantizes twice. A float source "
                        "(the original repo) gives a better result at the same size")
    if plan.adapter and plan.src_quant and any(m.params is not None for m in plan.modules.values()):
        warnings.append("merging into a quantized source and quantizing again loses a little of the adapter")
    if disk.get("ok") is False:
        warnings.append(disk["why"])
    if fit.get("status") in ("oom",):
        warnings.append(f"the result won't run on this Mac: {fit.get('reason')}")
    budget = fit.get("ram_budget_gb") or 0
    if budget and peak / 1e9 > 0.85 * budget:
        warnings.append(f"converting needs about {peak / 1e9:.1f} GB at its peak (the biggest tensor), "
                        f"close to this Mac's {budget:.1f} GB budget")

    return {
        "source": plan.source.describe() | {"notes": plan.source.notes},
        "model_type": plan.model_type,
        "arch": plan.arch,
        "layers": plan.n_layers,
        "experts": n_exp,
        "experts_per_token": top_k,
        "hidden": int(_first(cfg, "hidden_size", "d_model", default=0) or 0),
        "vocab": int(_first(cfg, "vocab_size", default=0) or 0),
        "params": numel_total,
        "source_quantized": bool(plan.src_quant),
        "recipe": rec.to_dict(),
        "dtype": plan.float_dtype,
        "quantization": plan.quant_config(),
        "components": rows,
        "float_bytes": float_bytes,
        "out_bytes": out_bytes,
        "expert_bytes": expert_out,
        "backbone_bytes": out_bytes - expert_out,
        "bits_per_weight": round(out_bytes * 8 / max(1, numel_total), 3),
        "active_read_bytes": active_read,
        "tensors": len(plan.layout),
        "units": [{"key": u.key, "out_bytes": u.out_bytes, "float_bytes": u.float_bytes,
                   "files": u.files} for u in plan.units],
        "windowed": plan.windowed,
        "separable": plan.separable,
        "adapter": None if not plan.adapter else {
            "path": plan.adapter["path"], "kind": plan.adapter["kind"], "rank": plan.adapter["rank"],
            "modules": len(plan.adapter["modules"]), "base_model": plan.adapter["base_model"]},
        "fit": fit,
        "disk": disk,
        "peak_bytes": peak,
        "warnings": warnings,
    }


def _fit(plan: Plan, total: int, backbone: int, expert: int, active_read: int) -> dict:
    try:
        from .. import kvmem
        from ..machine import get_provider
        from ..machine.ladder import BASELINE_DISK_GB_S, fit_model

        env = get_provider().envelope
        kv = kvmem.kv_bytes_per_token(plan.config)
        res = fit_model({"id": "convert", "name": plan.model_type, "checkpoint_gb": total / 1e9,
                         "backbone_gb": backbone / 1e9, "expert_gb": expert / 1e9,
                         "kv_kb_token": kv / 1024 if kv else 0},
                        env, ctx=getattr(plan, "ctx", 32768))
        out = res.to_dict() if hasattr(res, "to_dict") else dict(res.__dict__)
        disk_gb_s = float(env.disk_gb_s or BASELINE_DISK_GB_S)
        out["disk_gb_s"] = disk_gb_s
        out["ram_gb"] = env.ram_gb
        if out.get("status") == "streamed" and active_read:
            out["cold_tok_s"] = round(disk_gb_s / (active_read / 1e9), 2)
            out["cached_share"] = round(min(1.0, (out.get("cache_gb") or 0) * 1e9 / max(1, expert)), 3)
        return out
    except Exception as e:  # noqa: BLE001 - the preview still works without it
        return {"status": "unknown", "reason": f"{type(e).__name__}: {e}"}


def _disk(plan: Plan, out_bytes: int) -> dict:
    where = plan.out_dir
    free = _free_bytes(where)
    staging = 0
    src = plan.source
    if src.kind == "hub" and src.local_dir is None:
        if plan.windowed:
            # the probe shards for the whole run, plus the two biggest layers
            # (one converting, the next downloading)
            probe = set(plan.units[0].files) if plan.units else set()
            per = sorted((sum(src.file_sizes[f] for f in u.files if f not in probe) for u in plan.units),
                         reverse=True)
            staging = sum(src.file_sizes[f] for f in probe) + sum(per[:2])
        else:
            staging = src.total_bytes
    need = out_bytes + staging
    d = {"free": free, "out": out_bytes, "staging": staging, "need": need, "where": str(where or "")}
    if free is not None:
        d["ok"] = need <= free
        if not d["ok"]:
            d["why"] = (f"needs {need / 1e9:.0f} GB on disk ({out_bytes / 1e9:.0f} GB output"
                        f"{f', {staging / 1e9:.0f} GB source' if staging else ''}), "
                        f"{free / 1e9:.0f} GB free")
    return d


def _peak(plan: Plan) -> int:
    """Biggest tensor in float, times what's alive while it converts: the source
    pieces it's built from, the float tensor, and the quantized copy."""
    biggest = max((R.nbytes(*v) for v in plan.floats.values()), default=0)
    big_f32 = max((R.nbytes(v[0], "float32") for v in plan.floats.values()), default=0)
    return int(2 * biggest + min(big_f32, 512 << 20) + GiB)


def compare(plan: Plan, presets=R.FIT_ORDER) -> list:
    """Size, fit and speed of the same checkpoint under each preset."""
    rows = []
    for name in presets:
        s = retarget(plan, plan.recipe.with_preset(name)).summary()
        rows.append({"preset": name, "about": R.PRESETS[name]["about"], "out_bytes": s["out_bytes"],
                     "expert_bytes": s["expert_bytes"], "backbone_bytes": s["backbone_bytes"],
                     "bits_per_weight": s["bits_per_weight"], "fit": s["fit"].get("status"),
                     "cold_tok_s": s["fit"].get("cold_tok_s"), "disk_ok": s["disk"].get("ok", True)})
    return rows


def resolve_fit(base: Plan) -> tuple[Plan, dict]:
    """The best preset for this Mac and the reasoning, for preset "fit"."""
    recipe = base.recipe
    tried = []
    best_stream = None
    for name in R.FIT_ORDER:
        p = retarget(base, recipe.with_preset(name))
        s = p.summary()
        status = s["fit"].get("status")
        tried.append({"preset": name, "out_bytes": s["out_bytes"], "status": status,
                      "disk_ok": s["disk"].get("ok", True)})
        disk_ok = s["disk"].get("ok", True)
        if status == "resident" and disk_ok and recipe.target in ("auto", "resident"):
            return p, {"picked": name, "why": "best quality that runs fully in RAM", "tried": tried}
        if status == "streamed" and disk_ok and best_stream is None:
            best_stream = (name, p)
            if recipe.target == "streamed":
                return p, {"picked": name, "why": "best quality that streams and fits the disk",
                           "tried": tried}
    if best_stream and recipe.target != "resident":
        return best_stream[1], {"picked": best_stream[0],
                                "why": "nothing fits fully in RAM, best quality that streams",
                                "tried": tried}
    raise PlanError("no preset fits this Mac" + (" fully in RAM" if recipe.target == "resident" else "")
                    + ": " + ", ".join(f"{t['preset']} {t['out_bytes'] / 1e9:.0f} GB {t['status']}"
                                       for t in tried))


def make(source, recipe=None, **kw) -> Plan:
    """build(), with preset "fit" resolved to a real preset."""
    if isinstance(recipe, str):
        recipe = R.Recipe.from_preset(recipe)
    elif isinstance(recipe, dict) or recipe is None:
        recipe = R.Recipe.from_dict(recipe)
    base = build(source, recipe, **kw)
    if recipe.preset != "fit":
        return base
    p, why = resolve_fit(base)
    p.recipe.preset = why["picked"]
    p.summary()["fit_choice"] = why
    return p


__all__ = ["Plan", "PlanError", "SourceError", "build", "make", "resolve_fit", "compare", "retarget"]
