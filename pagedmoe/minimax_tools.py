# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
MiniMax-M3 tool calls.

The M3 template writes calls as XML where every structural tag carries the
namespace token in front of it:

    NS<tool_call>
    NS<invoke name="read_file">NS<filePath>/a/b.dartNS</filePath>NS</invoke>
    NS</tool_call>

Nested objects are child tags and lists are repeated <item> tags. Values are
raw text, so a file body full of `<` is still unambiguous: only a tag right
after NS is structure.

The checkpoint's tokenizer_config says json_tools, which json.loads this and
drops every call. Its start marker is the bare <tool_call>, so the NS token in
front of it also leaked into content.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

NS = "]<]minimax[>["
TOOL_CALL_START = NS + "<tool_call>"
TOOL_CALL_END = NS + "</tool_call>"

_TAG_RE = re.compile(r'<(/?)([^\s<>/"]+)(?:\s+name="([^"]*)")?\s*>')


class _Frame:
    __slots__ = ("tag", "name", "children", "text")

    def __init__(self, tag: str, name: Optional[str]):
        self.tag = tag
        self.name = name
        self.children: list[tuple[str, Any]] = []
        self.text: list[str] = []

    def value(self):
        if not self.children:
            return "".join(self.text)
        if all(k == "item" for k, _ in self.children):
            return [v for _, v in self.children]
        return {k: v for k, v in self.children}


def _tool_schemas(tools) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function", t)
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            params = fn.get("parameters")
            out[fn["name"]] = params if isinstance(params, dict) else {}
    return out


def _types(schema) -> set[str]:
    if not isinstance(schema, dict):
        return set()
    t = schema.get("type")
    if isinstance(t, str):
        return {t}
    if isinstance(t, list):
        return {x for x in t if isinstance(x, str)}
    for key in ("anyOf", "oneOf"):
        alts = schema.get(key)
        if isinstance(alts, list):
            return set().union(*(_types(a) for a in alts))
    return set()


def _coerce(value, schema):
    """Cast leaf text to the schema's type. Unknown or string types stay raw."""
    types = _types(schema)
    if isinstance(value, list):
        items = schema.get("items", {}) if isinstance(schema, dict) else {}
        return [_coerce(v, items) for v in value]
    if isinstance(value, dict):
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        return {k: _coerce(v, props.get(k, {})) for k, v in value.items()}
    if not isinstance(value, str) or not types or "string" in types:
        return value
    s = value.strip()
    if "array" in types and not s:
        return []
    if "object" in types and not s:
        return {}
    if "boolean" in types and s.lower() in ("true", "false"):
        return s.lower() == "true"
    if "integer" in types:
        try:
            return int(s)
        except ValueError:
            pass
    if "number" in types:
        try:
            return float(s) if any(c in s for c in ".eE") else int(s)
        except ValueError:
            pass
    if "null" in types and s in ("", "null"):
        return None
    try:
        return json.loads(s)
    except ValueError:
        return value


def parse_tool_call(text: str, tools: Optional[list] = None) -> list[dict]:
    """Every <invoke> in one M3 tool-call block, as {name, arguments}."""
    body = (text or "").replace(TOOL_CALL_START, "").replace(TOOL_CALL_END, "")
    calls: list[dict] = []
    stack: list[_Frame] = []

    def close(frame: _Frame):
        if frame.tag == "invoke":
            args = frame.value()
            calls.append({"name": frame.name or "", "arguments": args if isinstance(args, dict) else {}})
        elif stack:
            stack[-1].children.append((frame.tag, frame.value()))

    for seg in body.split(NS):
        m = _TAG_RE.match(seg)
        if m is None:
            if stack:
                stack[-1].text.append(seg)
            continue
        closing, tag, name = m.groups()
        rest = seg[m.end():]
        if not closing:
            if tag == "tool_call":
                continue
            frame = _Frame(tag, name)
            frame.text.append(rest)
            stack.append(frame)
            continue
        if tag == "tool_call" or not any(f.tag == tag for f in stack):
            continue
        while stack:
            frame = stack.pop()
            close(frame)
            if frame.tag == tag:
                break
    # A call cut off at max_tokens keeps whatever parameters did arrive.
    while stack:
        close(stack.pop())

    calls = [c for c in calls if c["name"].strip()]
    if not calls:
        raise ValueError("no MiniMax invoke blocks found")
    schemas = _tool_schemas(tools)
    for c in calls:
        c["name"] = c["name"].strip()
        c["arguments"] = _coerce(c["arguments"], schemas.get(c["name"], {}))
    return calls


def template_uses(chat_template) -> bool:
    return isinstance(chat_template, str) and NS in chat_template and "<tool_call>" in chat_template
