# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Sampling defaults for a request that leaves them out.

mlx-lm's server falls back to its own CLI defaults: temperature 0, no top-k,
no repetition penalty. Greedy decoding is not what any of these models were
tuned for, and thinking models tend to loop on it. A client that sends no
sampling settings (plenty of OpenAI clients send only the messages) got that.

Here a missing setting comes from the checkpoint's generation_config.json,
which is the model's own recipe, and what that file leaves out comes from the
family's published serving recipe (FAMILIES). Anything the request sends, and
any --temp / --top-p / --top-k / --min-p given to the server, still wins. A
generation_config.json with do_sample false keeps greedy decoding.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

# Matched against model_type first, then the checkpoint folder name. First
# match wins, so specific patterns go above general ones.
#
# No presence_penalty anywhere. mlx-lm applies it to every token seen in the
# last 20, and in code and tool calls a lot of the right tokens (indentation,
# quotes, identifiers, JSON keys) repeat inside that window. Qwen's coding
# recipe is presence 0 with repetition_penalty 1.05, which still stops the
# paste loops that family falls into but barely touches syntax.
FAMILIES = (
    ("qwen", re.compile(r"qwen[34]|qwen3_next|qwen3_5|qwen4_exp", re.I),
     {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05, "repetition_context_size": 64}),
    ("deepseek", re.compile(r"deepseek", re.I), {"temperature": 0.6, "top_p": 0.95, "top_k": 40}),
    ("glm", re.compile(r"^glm|chatglm", re.I), {"temperature": 0.6, "top_p": 0.95, "top_k": 40}),
    ("kimi", re.compile(r"kimi|moonshot", re.I), {"temperature": 0.6, "top_p": 0.95, "top_k": 40}),
    ("minimax", re.compile(r"minimax", re.I), {"temperature": 0.6, "top_p": 0.95, "top_k": 40}),
    ("gpt-oss", re.compile(r"gpt[_-]?oss", re.I), {"temperature": 1.0, "top_p": 1.0}),
    ("mistral", re.compile(r"mistral|mixtral|magistral|devstral", re.I), {"temperature": 0.15, "top_p": 0.9, "top_k": 40}),
    ("gemma", re.compile(r"gemma", re.I), {"temperature": 1.0, "top_p": 0.95, "top_k": 64}),
    ("llama", re.compile(r"llama", re.I), {"temperature": 0.6, "top_p": 0.9, "top_k": 40}),
    ("phi", re.compile(r"phi[_-]?[34]|phi3|phi4", re.I), {"temperature": 0.8, "top_p": 0.95, "top_k": 40}),
    ("granite", re.compile(r"granite", re.I), {"temperature": 0.6, "top_p": 0.9, "top_k": 40}),
)

# generation_config.json keys read, with the type mlx-lm wants.
_FILE_KEYS = (("temperature", float), ("top_p", float), ("top_k", int), ("min_p", float), ("repetition_penalty", float))

# Server flags that set a default of their own.
_CLI_FLAGS = {"temperature": "--temp", "top_p": "--top-p", "top_k": "--top-k", "min_p": "--min-p"}


def family_for(model_type: str, name: str):
    for fam, rx, values in FAMILIES:
        if model_type and rx.search(model_type):
            return fam, values
    for fam, rx, values in FAMILIES:
        if name and rx.search(name):
            return fam, values
    return None, {}


def _generation_config(model_dir: Optional[str]) -> dict:
    if not model_dir:
        return {}
    try:
        gen = json.loads((Path(model_dir).expanduser() / "generation_config.json").read_text())
    except (OSError, ValueError):
        return {}
    return gen if isinstance(gen, dict) else {}


def defaults_for(model_dir: Optional[str], model_type: str = "") -> tuple:
    """(defaults, sources) for a checkpoint. sources maps each key to where it came from."""
    gen = _generation_config(model_dir)
    if gen.get("do_sample") is False:
        return {}, {}
    fam, values = family_for(model_type or "", Path(model_dir).name if model_dir else "")
    out = dict(values)
    sources = {k: f"{fam} family" for k in out}
    for key, cast in _FILE_KEYS:
        raw = gen.get(key)
        if raw is None or isinstance(raw, bool):
            continue
        try:
            value = cast(raw)
        except (TypeError, ValueError):
            continue
        if value < 0:
            continue
        out[key] = value
        sources[key] = "generation_config.json"
    if out.get("temperature", 1.0) <= 0:
        return {}, {}
    # Drop values that are mlx-lm's own no-ops, so the log shows what changes.
    for key, off in (("top_p", lambda v: v >= 1.0), ("top_k", lambda v: v <= 0), ("min_p", lambda v: v <= 0),
                     ("repetition_penalty", lambda v: v == 1.0 or v <= 0)):
        if key in out and off(out[key]):
            out.pop(key)
            sources.pop(key, None)
    if "repetition_penalty" not in out:
        out.pop("repetition_context_size", None)
        sources.pop("repetition_context_size", None)
    return out, sources


def describe(defaults: dict, sources: dict) -> str:
    if not defaults:
        return "none, the server's own defaults apply"
    by_source: dict = {}
    for key, value in defaults.items():
        if key == "repetition_context_size":
            continue
        by_source.setdefault(sources.get(key, "?"), []).append(f"{key} {value:g}")
    return "; ".join(f"{', '.join(parts)} ({src})" for src, parts in by_source.items())


def fill(handler, body: dict, defaults: dict, argv=None) -> list:
    """Set each default the request left out on the handler. Returns the keys set."""
    argv = sys.argv if argv is None else argv
    applied = []
    for key, value in defaults.items():
        if key in body:
            continue
        if key == "repetition_context_size" and "repetition_penalty" in body:
            continue
        flag = _CLI_FLAGS.get(key)
        if flag and any(a == flag or a.startswith(flag + "=") for a in argv):
            continue
        setattr(handler, key, value)
        applied.append(key)
    return applied
