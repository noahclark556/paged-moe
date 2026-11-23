# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GRPO reward functions.

Two signatures are accepted, so existing code drops in:

  TRL style   fn(prompts, completions, **columns) -> list[float]
              columns are the dataset fields, one list per field
  per sample  fn(completion, row) -> float

`completions` are decoded strings. A function may return None for a sample
to mean "no opinion" (counted as 0).

Built-ins cover the usual verifiable-reward setups; anything domain specific
(a CAD kernel, a test runner, a geometry score) is just another function.
"""

from __future__ import annotations

import inspect
import math
import re
from typing import Callable

import numpy as np

RewardFn = Callable[..., object]


def _batch_style(fn) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "completions" in params or "prompts" in params


def score(fns: list[RewardFn], prompts: list, completions: list[str], rows: list[dict],
          weights: list[float] | None = None) -> tuple[np.ndarray, dict]:
    """(combined reward per sample, {fn name: mean reward})."""
    n = len(completions)
    weights = weights or [1.0] * len(fns)
    if len(weights) != len(fns):
        raise ValueError("reward_weights must match the number of reward functions")
    total = np.zeros(n, np.float64)
    means = {}
    for fn, w in zip(fns, weights):
        name = getattr(fn, "__name__", type(fn).__name__)
        if _batch_style(fn):
            cols: dict = {}
            for r in rows:
                for k in r:
                    cols.setdefault(k, [])
            for k in cols:
                cols[k] = [r.get(k) for r in rows]
            cols.pop("prompt", None)
            out = fn(prompts=prompts, completions=completions, **cols)
        else:
            out = [fn(c, r) for c, r in zip(completions, rows)]
        vals = np.asarray([0.0 if v is None else float(v) for v in out], np.float64)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        total += w * vals
        means[name] = float(vals.mean()) if n else 0.0
    return total, means


def advantages(rewards: np.ndarray, group: int, scale: bool = True, eps: float = 1e-4):
    """Group-relative advantages and a mask of groups with any spread."""
    r = rewards.reshape(-1, group)
    mean = r.mean(axis=1, keepdims=True)
    std = r.std(axis=1, keepdims=True)
    adv = r - mean
    if scale:
        adv = adv / (std + eps)
    live = (std[:, 0] > 1e-8)
    return adv.reshape(-1), live


# ------------------------------------------------------------------ built-ins


_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_HASHES = re.compile(r"####\s*(.+)")
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str, style: str = "auto") -> str | None:
    """Final answer from a completion: \\boxed{...}, '#### x', or the last number."""
    text = text.split("</think>")[-1]
    if style in ("auto", "boxed"):
        m = _BOXED.findall(text)
        if m:
            return m[-1].strip()
        if style == "boxed":
            return None
    if style in ("auto", "hashes"):
        m = _HASHES.findall(text)
        if m:
            return m[-1].strip()
        if style == "hashes":
            return None
    m = _NUM.findall(text)
    return m[-1].replace(",", "") if m else None


def _norm(s) -> str:
    s = str(s).strip().replace(",", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if math.isfinite(f) and f == int(f) else str(f)
    except ValueError:
        return s.lower()


def exact_answer(field: str = "answer", style: str = "auto") -> RewardFn:
    """1.0 when the extracted answer equals row[field] (numbers compared as numbers)."""

    def exact_answer(completion: str, row: dict) -> float:
        want = row.get(field)
        if want is None:
            return 0.0
        want = extract_answer(str(want), "auto") if "####" in str(want) else want
        got = extract_answer(completion, style)
        return 1.0 if got is not None and _norm(got) == _norm(want) else 0.0

    return exact_answer


def regex_format(pattern: str, flags: int = re.DOTALL) -> RewardFn:
    """1.0 when the whole completion matches `pattern`."""
    rx = re.compile(pattern, flags)

    def regex_format(completion: str, row: dict) -> float:
        return 1.0 if rx.fullmatch(completion.strip()) else 0.0

    return regex_format


def think_format() -> RewardFn:
    """1.0 for '<think>...</think>' followed by a non-empty answer."""
    return regex_format(r"(?:<think>)?.*?</think>\s*\S.*")


def length_penalty(max_tokens: int, tokenizer=None, scale: float = 1.0) -> RewardFn:
    """0 up to max_tokens, then down to -scale linearly over the next max_tokens."""

    def length_penalty(completion: str, row: dict) -> float:
        n = len(tokenizer.encode(completion)) if tokenizer is not None else len(completion) / 4
        over = max(0.0, n - max_tokens)
        return -scale * min(1.0, over / max(1, max_tokens))

    return length_penalty
