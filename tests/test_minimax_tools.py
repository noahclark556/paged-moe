# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
MiniMax-M3 XML tool calls.

The round trip renders calls with the checkpoint's own chat template and
parses them back, so it skips when no MiniMax-M3 checkpoint is on disk.

    python tests/test_minimax_tools.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pagedmoe.minimax_tools import NS, TOOL_CALL_END, TOOL_CALL_START, parse_tool_call

M3_DIR = Path("~/mlx-models/MiniMax-M3-MLX-4bit").expanduser()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "startLine": {"type": "integer"},
                    "endLine": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "dryRun": {"type": "boolean"},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old": {"type": "string"},
                                "new": {"type": "string"},
                                "line": {"type": "integer"},
                            },
                        },
                    },
                },
            },
        },
    },
]


def test_logged_read_file_call():
    # Body exactly as the server logged it before this parser existed.
    body = (
        f"\n{NS}<invoke name=\"read_file\">{NS}<filePath>/Users/me/Desktop/"
        f"Development/myapp/lib/ui/shared/widgets/myapp_toolbox.dart"
        f"{NS}</filePath>{NS}<startLine>601{NS}</startLine>{NS}<endLine>1151"
        f"{NS}</endLine>{NS}</invoke>\n{NS}"
    )
    calls = parse_tool_call(body, TOOLS)
    assert calls == [
        {
            "name": "read_file",
            "arguments": {
                "filePath": "/Users/me/Desktop/Development/myapp/lib/ui/shared/widgets/myapp_toolbox.dart",
                "startLine": 601,
                "endLine": 1151,
            },
        }
    ]


def test_nested_lists_markup_values_and_multiple_calls():
    code = "if (a < b) {\n  return <Widget>[];\n}\n"
    body = (
        f"{TOOL_CALL_START}\n"
        f"{NS}<invoke name=\"replace_in_file\">{NS}<filePath>lib/a.dart{NS}</filePath>"
        f"{NS}<dryRun>false{NS}</dryRun>"
        f"{NS}<edits>{NS}<item>{NS}<old>{code}{NS}</old>{NS}<new></div>{NS}</new>"
        f"{NS}<line>12{NS}</line>{NS}</item>{NS}</edits>{NS}</invoke>\n"
        f"{NS}<invoke name=\"read_file\">{NS}<filePath>lib/b.dart{NS}</filePath>{NS}</invoke>\n"
        f"{TOOL_CALL_END}"
    )
    calls = parse_tool_call(body, TOOLS)
    assert [c["name"] for c in calls] == ["replace_in_file", "read_file"]
    args = calls[0]["arguments"]
    assert args["dryRun"] is False
    assert args["edits"] == [{"old": code, "new": "</div>", "line": 12}]
    assert calls[1]["arguments"] == {"filePath": "lib/b.dart"}


def test_truncated_call_keeps_what_arrived():
    body = f"{NS}<invoke name=\"read_file\">{NS}<filePath>lib/a.dart{NS}</filePath>{NS}<startLine>4"
    calls = parse_tool_call(body, TOOLS)
    assert calls == [{"name": "read_file", "arguments": {"filePath": "lib/a.dart", "startLine": 4}}]


def test_no_schema_keeps_strings_and_empty_raises():
    calls = parse_tool_call(f"{NS}<invoke name=\"x\">{NS}<n>7{NS}</n>{NS}</invoke>")
    assert calls[0]["arguments"] == {"n": "7"}
    try:
        parse_tool_call("no call here")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_round_trip_through_m3_template():
    if not (M3_DIR / "chat_template.jinja").is_file():
        print("skip: no MiniMax-M3 checkpoint on disk")
        return
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(M3_DIR))
    sent = [
        {"name": "read_file", "arguments": {"filePath": "lib/a.dart", "startLine": 601, "endLine": 1151}},
        {
            "name": "replace_in_file",
            "arguments": {
                "filePath": "lib/<odd>.dart",
                "dryRun": True,
                "edits": [{"old": "a < b && c > d", "new": "x", "line": 3}],
            },
        },
    ]
    messages = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": c} for c in sent],
        },
        {"role": "tool", "content": "ok"},
    ]
    text = tok.apply_chat_template(messages, tools=TOOLS, tokenize=False)
    start = text.rindex(TOOL_CALL_START)
    end = text.index(TOOL_CALL_END, start) + len(TOOL_CALL_END)
    assert parse_tool_call(text[start:end], TOOLS) == sent

    ids = tok.encode(TOOL_CALL_START, add_special_tokens=False)
    assert len(ids) == 2, ids
    assert len(tok.encode(TOOL_CALL_END, add_special_tokens=False)) == 2


if __name__ == "__main__":
    test_logged_read_file_call()
    test_nested_lists_markup_values_and_multiple_calls()
    test_truncated_call_keeps_what_arrived()
    test_no_schema_keeps_strings_and_empty_raises()
    test_round_trip_through_m3_template()
    print("OK: MiniMax tool calls")
