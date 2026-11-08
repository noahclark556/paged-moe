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


# model_type -> predicate. The module text lives in arches/<model_type>.py.
# Only types mlx-lm does not ship belong here. qwen3_next and qwen3_5_moe
# already load from mlx-lm, so a missing sidecar is not a problem for them.
_PORTS = {
    "qwen4_exp": _qwen4_exp_config,
}


def _port_text(model_type: str) -> str | None:
    path = Path(__file__).resolve().parent / "arches" / f"{model_type}.py"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if "class Model" not in text or "class ModelArgs" not in text:
        return None
    if not text.endswith("\n"):
        text += "\n"
    return text


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


def ensure(model_path) -> str | None:
    """Write a known architecture file when this checkpoint is missing one.

    Returns the path written. Returns None when the folder is left untouched,
    including when the file is already there or the config is not a known port.
    A failed write is ignored so load fails the way it did before.
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
    """Register `<model_path>/<model_type>.py` if mlx-lm lacks the type.

    Returns the registered file, or None when nothing was needed.
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
    src = p / f"{mt}.py"
    if not src.is_file():
        ensure(p)
    if not src.is_file():
        return None
    if name in sys.modules:
        return None

    spec = importlib.util.spec_from_file_location(name, src)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    print(
        f"[paged-moe] model type {mt} is not in this mlx-lm, "
        f"using {src.name} from the checkpoint",
        flush=True,
    )
    return str(src)
