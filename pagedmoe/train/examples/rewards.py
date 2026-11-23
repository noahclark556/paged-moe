# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Example GRPO rewards. Two signatures work:

    fn(completion, row) -> float                     one sample at a time
    fn(prompts, completions, **columns) -> [float]   TRL style, whole batch

Return None for "no opinion" (counted as 0). Anything goes inside: run the
code, render the CAD, call a checker.
"""

import re

_TAG = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")


def short_and_tagged(completion: str, row: dict) -> float:
    """1 for a lone <answer>N</answer>, less the longer the reply runs."""
    if not _TAG.search(completion):
        return 0.0
    return max(0.0, 1.0 - len(completion) / 400)


def answer_in_tag(prompts, completions, answer, **_) -> list[float]:
    out = []
    for c, a in zip(completions, answer):
        m = _TAG.search(c)
        out.append(1.0 if m and m.group(1) == str(a).strip() else 0.0)
    return out
