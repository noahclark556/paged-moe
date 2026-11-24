# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Model classes shipped inside a checkpoint.

Some conversions land before mlx-lm has the architecture. MiniMax-M3 ships
minimax_m3.py next to its weights and tells you to copy it into
mlx_lm/models/. Instead of patching site-packages, register that file as
mlx_lm.models.<model_type> for this process so load_model can find it.

Only used when the installed mlx-lm has no module of that name, so a release
that adds the architecture takes over without any change here.

A checkpoint that is a known architecture but omitted the module file gets
that file written beside the weights. Anything already present is not replaced.

Package ports (arches/<model_type>/) are registered from the bundled package
directly and are not copied into the checkpoint.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _qwen4_exp_config(cfg: dict) -> bool:
    """True only for the Flash-family qwen4_exp layout this port implements."""
    if cfg.get("model_type") != "qwen4_exp":
        return False
    arch = cfg.get("architectures")
    if not isinstance(arch, list) or "Qwen4ExpForConditionalGeneration" not in arch:
        return False
    text = cfg.get("text_config")
    if not isinstance(text, dict) or text.get("model_type") != "qwen4_exp_text":
        return False
    for key in (
        "hidden_size",
        "num_hidden_layers",
        "num_experts",
        "num_experts_per_tok",
        "hc_count",
        "indexer_n_heads",
        "heads_per_ngram",
    ):
        if not _positive_int(text.get(key)):
            return False
    layers = text.get("layer_types")
    if not isinstance(layers, list) or not layers:
        return False
    kinds = set(layers)
    if kinds != {"linear_attention", "full_attention"}:
        return False
    return True


def _glm5_next_config(cfg: dict) -> bool:
    """True for GLM-5.3-Flash / glm5_next hybrid MoE (text_config or flat)."""
    if cfg.get("model_type") != "glm5_next":
        return False
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    if not isinstance(text, dict):
        return False
    for key in (
        "hidden_size",
        "num_hidden_layers",
        "n_routed_experts",
        "num_experts_per_tok",
        "hc_mult",
        "index_n_heads",
        "index_topk",
    ):
        if not _positive_int(text.get(key)):
            return False
    layers = text.get("layer_types")
    if not isinstance(layers, list) or not layers:
        return False
    kinds = set(layers)
    # sparse layers are named deepseek_sparse_attention in the HF config
    if "linear_attention" not in kinds:
        return False
    if not (kinds <= {"linear_attention", "deepseek_sparse_attention", "full_attention"}):
        return False
    mlp = text.get("mlp_layer_types")
    if not isinstance(mlp, list) or not mlp:
        return False
    return True


# model_type -> predicate. Single-file ports live in arches/<model_type>.py;
# package ports live in arches/<model_type>/ with an __init__.py that exports
# Model and ModelArgs. Only types mlx-lm does not ship belong here.
_PORTS = {
    "qwen4_exp": _qwen4_exp_config,
    "glm5_next": _glm5_next_config,
}

_ARCHES = Path(__file__).resolve().parent / "arches"


def _port_text(model_type: str) -> str | None:
    path = _ARCHES / f"{model_type}.py"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if "class Model" not in text or "class ModelArgs" not in text:
        return None
    if not text.endswith("\n"):
        text += "\n"
    return text


def _port_package(model_type: str) -> Path | None:
    path = _ARCHES / model_type
    if path.is_dir() and (path / "__init__.py").is_file():
        return path
    return None


def _mlx_provides(module_name: str) -> bool:
    """True when mlx-lm itself ships this module, not a file we registered."""
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError):
        return False
    origin = getattr(spec, "origin", None) if spec is not None else None
    if not origin or origin == "namespace":
        return False
    try:
        import mlx_lm
    except ImportError:
        return False
    root = Path(mlx_lm.__file__).resolve().parent
    try:
        Path(origin).resolve().relative_to(root)
    except (OSError, ValueError):
        return False
    return True


def _register_package(name: str, pkg: Path) -> str:
    """Load arches/<type>/ as mlx_lm.models.<type> (with its submodules)."""
    init = pkg / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        name, init, submodule_search_locations=[str(pkg)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load architecture package {pkg}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        # drop any submodules that got half-registered
        prefix = name + "."
        for key in [k for k in sys.modules if k == name or k.startswith(prefix)]:
            sys.modules.pop(key, None)
        raise
    if not hasattr(module, "Model") or not hasattr(module, "ModelArgs"):
        sys.modules.pop(name, None)
        raise ImportError(f"{pkg} must export Model and ModelArgs")
    return str(pkg)


def _register_file(name: str, src: Path) -> str:
    spec = importlib.util.spec_from_file_location(name, src)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load architecture file {src}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return str(src)


def ensure(model_path) -> str | None:
    """Write a known architecture file when this checkpoint is missing one.

    Package ports are not copied into the checkpoint; register() loads them
    from the bundled package. Returns the path written, or None.
    """
    p = Path(model_path)
    try:
        cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    if not isinstance(cfg, dict):
        return None
    mt = cfg.get("model_type")
    match = _PORTS.get(mt)
    if match is None or not isinstance(mt, str) or not mt.isidentifier():
        return None
    if not match(cfg):
        return None
    if _mlx_provides(f"mlx_lm.models.{mt}"):
        return None
    # packages stay bundled; nothing to write next to the weights
    if _port_package(mt) is not None:
        return None
    try:
        dest_dir = p.resolve()
    except OSError:
        return None
    if not dest_dir.is_dir():
        return None
    dest = dest_dir / f"{mt}.py"
    try:
        if dest.is_file() and dest.stat().st_size > 0:
            return None
        if dest.is_file():
            dest.unlink()
    except OSError:
        return None
    text = _port_text(mt)
    if not text:
        return None
    try:
        fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return None
    except OSError:
        return None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
    except OSError:
        try:
            dest.unlink()
        except OSError:
            pass
        return None
    print(
        f"[paged-moe] {mt}.py was missing from the checkpoint; "
        f"wrote the bundled architecture",
        flush=True,
    )
    return str(dest)


def register(model_path) -> str | None:
    """Register a local or bundled architecture if mlx-lm lacks the type.

    Order: mlx-lm's own module, then `<model_path>/<type>.py` (after ensure),
    then a bundled package under arches/<type>/.
    """
    p = Path(model_path)
    try:
        cfg = json.loads((p / "config.json").read_text())
    except (OSError, ValueError):
        return None
    mt = cfg.get("model_type")
    if not isinstance(mt, str) or not mt.isidentifier():
        return None
    try:
        from mlx_lm.utils import MODEL_REMAPPING

        mt = MODEL_REMAPPING.get(mt, mt)
    except ImportError:
        pass

    name = f"mlx_lm.models.{mt}"
    if _mlx_provides(name):
        return None
    if name in sys.modules:
        return None

    src = p / f"{mt}.py"
    if not src.is_file():
        ensure(p)
    if src.is_file():
        path = _register_file(name, src)
        print(
            f"[paged-moe] model type {mt} is not in this mlx-lm, "
            f"using {src.name} from the checkpoint",
            flush=True,
        )
        return path

    match = _PORTS.get(mt)
    pkg = _port_package(mt)
    if match is None or pkg is None:
        return None
    if not match(cfg):
        return None
    path = _register_package(name, pkg)
    print(
        f"[paged-moe] model type {mt} is not in this mlx-lm, "
        f"using bundled arches/{mt}/",
        flush=True,
    )
    return path
