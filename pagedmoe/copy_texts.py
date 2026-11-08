# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""JSON-escaped text the verify drafter may copy, never compute from.

Tool arguments are JSON, so a file the model is writing back shows up as
``\\n`` and ``\\"`` rather than the raw tool result. The drafter matches these
runs as copy-only sources. Host-supplied ``copy_texts`` (the files on disk)
come first: a compacted tool result no longer has the bytes an edit copies.
"""

from __future__ import annotations

import json


def escaped_runs(body, *, limit_chars: int, limit_runs: int) -> list[str]:
    """Escaped forms that differ from the raw text, newest tool results last."""
    out: list[str] = []
    room = int(limit_chars)
    seen: set[str] = set()

    def push(text) -> None:
        nonlocal room
        if room < 64 or len(out) >= limit_runs or not isinstance(text, str):
            return
        raw = text[:room]
        if len(raw) < 64 or raw in seen:
            return
        esc = json.dumps(raw, ensure_ascii=False)[1:-1]
        if esc == raw:
            return
        piece = esc[:room]
        seen.add(raw)
        out.append(piece)
        room -= len(piece)

    for text in body.get("copy_texts") or []:
        push(text)
    for msg in reversed(body.get("messages") or []):
        if room < 64 or len(out) >= limit_runs:
            break
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        text = msg.get("content")
        if isinstance(text, list):
            text = "".join(
                p.get("text", "") for p in text if isinstance(p, dict)
            )
        push(text)
    return out
