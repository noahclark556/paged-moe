# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tool call repair (tool_repair.py) on a generation stream.

Streams are cut into small uneven pieces so markers and names land across
pieces, then collected the way mlx-lm's handle_completion does, through the
server's patched ToolCallFormatter.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx_lm.server as mlx_server  # noqa: E402

from pagedmoe import server, tool_repair  # noqa: E402
from pagedmoe.tool_repair import ToolRepair  # noqa: E402

server._install_lenient_tool_parse()

TOOLS = [
    {"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
    for n in ("read_file", "run_in_terminal", "read")
]
NAMES = tool_repair.tool_names(TOOLS)


def gens(*segments, cuts=(1, 2, 3, 5)):
    """Response objects for (state, text) segments, cut into uneven pieces."""
    out, n = [], 0
    for state, text in segments:
        i = 0
        while i < len(text):
            step = cuts[n % len(cuts)]
            out.append(mlx_server.Response(text[i:i + step], 100 + n, state, None, 0.0, None, ()))
            i += step
            n += 1
    # The stop token.
    out.append(mlx_server.Response("", 2, None, (2,), 0.0, "stop", ()))
    return out


def collect(stream, tool_parser=None, stream_mode=False):
    """mlx-lm's handle_completion collection loop, minus the HTTP."""
    fmt = mlx_server.ToolCallFormatter(tool_parser, TOOLS, stream_mode)
    prev, finish, text, reasoning, tool_text, calls, tokens = None, "stop", "", "", "", [], 0
    made = False
    for gen in stream:
        if gen.state == "reasoning":
            reasoning += gen.text
        elif gen.state == "tool":
            tool_text += gen.text
        elif gen.state == "normal":
            if prev == "tool":
                calls.append(tool_text)
                tool_text = ""
                made = True
            text += gen.text
        tokens += 1
        if gen.finish_reason is not None:
            finish = gen.finish_reason
        prev = gen.state
    if prev == "tool" and tool_text:
        calls.append(tool_text)
        made = True
    if finish == "stop" and made:
        finish = "tool_calls"
    out = [
        (c["function"]["name"], json.loads(c["function"]["arguments"])) for c in fmt(calls)
    ]
    return {"text": text, "reasoning": reasoning, "calls": out, "finish": finish, "tokens": tokens}


def run(*segments, names=NAMES, **kw):
    src = gens(*segments, **kw)
    res = collect(ToolRepair(names).run(iter(src)))
    res["in_tokens"] = len(src)
    return res


def test_plain_text_untouched():
    for text in (
        "Hello there. Nothing to call here.",
        "a < b and c > d, <div>tag</div>, x<y",
        "Use <tool to mean nothing. Ends with <",
        "<function=not_a_tool>x</function> stays text",
        "[1] first point\n[2] second",
        "`code` and ```python\nprint(1)\n```",
    ):
        src = gens(("normal", text))
        out = list(ToolRepair(NAMES).run(iter(src)))
        res = collect(out)
        assert res["text"] == text, (text, res["text"])
        assert res["calls"] == [] and res["finish"] == "stop"
        assert len(out) == len(src)


def test_qwen_xml_in_content():
    res = run(("normal", "Let me look.\n<tool_call>\n<function=read_file>\n<parameter=path>\na/b.py\n"
                         "</parameter>\n<parameter=limit>\n40\n</parameter>\n</function>\n</tool_call>"))
    assert res["calls"] == [("read_file", {"path": "a/b.py", "limit": 40})], res
    assert res["text"] == "Let me look.\n"
    assert res["finish"] == "tool_calls"
    assert res["tokens"] == res["in_tokens"]


def test_bare_function_block():
    res = run(("normal", "<function=run_in_terminal><parameter=command>ls -la</parameter></function></tool_call>"))
    assert res["calls"] == [("run_in_terminal", {"command": "ls -la"})], res
    assert res["text"] == ""


def test_hermes_json_in_content():
    res = run(("normal", '<tool_call>\n{"name": "read_file", "arguments": {"path": "x.txt"}}\n</tool_call>'))
    assert res["calls"] == [("read_file", {"path": "x.txt"})], res
    assert res["tokens"] == res["in_tokens"]


def test_two_calls():
    block = '<tool_call>{"name": "read_file", "arguments": {"path": "%s"}}</tool_call>'
    res = run(("normal", block % "a" + "\n" + block % "b"))
    assert res["calls"] == [("read_file", {"path": "a"}), ("read_file", {"path": "b"})], res


def test_deepseek_v31_native():
    text = ("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>read_file<｜tool▁sep｜>{\"path\": \"m.py\"}"
            "<｜tool▁call▁end｜><｜tool▁call▁begin｜>run_in_terminal<｜tool▁sep｜>{\"command\": \"pwd\"}"
            "<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
    res = run(("normal", text))
    assert res["calls"] == [("read_file", {"path": "m.py"}), ("run_in_terminal", {"command": "pwd"})], res
    assert res["text"] == ""


def test_deepseek_v3_native():
    text = ("Checking.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>read_file\n"
            "```json\n{\"path\": \"z\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
    res = run(("normal", text))
    assert res["calls"] == [("read_file", {"path": "z"})], res
    assert res["text"] == "Checking."


def test_invoke_and_dsml():
    res = run(("normal", '<invoke name="read_file">\n<parameter name="path">q.md</parameter>\n</invoke>'))
    assert res["calls"] == [("read_file", {"path": "q.md"})], res
    dsml = ('<｜DSML｜function_calls>\n<｜DSML｜invoke name="read_file">\n'
            '<｜DSML｜parameter name="path" string="true">7</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="limit" string="false">7</｜DSML｜parameter>\n'
            '</｜DSML｜invoke>\n</｜DSML｜function_calls>')
    res = run(("normal", dsml))
    assert res["calls"] == [("read_file", {"path": "7", "limit": 7})], res


def test_bare_json_reply():
    res = run(("normal", '{"name": "read_file", "arguments": {"path": "j"}}'))
    assert res["calls"] == [("read_file", {"path": "j"})], res
    res = run(("normal", '\n```json\n{"name": "read_file", "arguments": "{\\"path\\": \\"k\\"}"}\n```\n'))
    assert res["calls"] == [("read_file", {"path": "k"})], res
    assert res["text"].strip() == ""
    # JSON that is not a call, and JSON after prose, stay text.
    for text in ('{"answer": 42}', 'Here: {"name": "read_file", "arguments": {}}', '{"name": "other", "arguments": {}}'):
        res = run(("normal", text))
        assert res["calls"] == [] and res["text"] == text, (text, res)


def test_name_not_offered_is_text():
    text = '<tool_call>{"name": "delete_everything", "arguments": {}}</tool_call> done'
    res = run(("normal", text))
    assert res["calls"] == [] and res["text"] == text, res


def test_name_prefix_waits_for_whole_name():
    # "read" is offered too; a stream cut after "read" must not call it.
    res = run(("normal", '<invoke name="read_file"><parameter name="path">p</parameter></invoke>'), cuts=(15, 1))
    assert res["calls"] == [("read_file", {"path": "p"})], res


def test_reasoning_call_only_when_turn_is_empty():
    call = '<tool_call>{"name": "read_file", "arguments": {"path": "r"}}</tool_call>'
    res = run(("reasoning", "I should read it. " + call), ("normal", "\n\n"))
    assert res["calls"] == [("read_file", {"path": "r"})], res
    assert res["finish"] == "tool_calls"
    assert res["tokens"] == res["in_tokens"]
    res = run(("reasoning", "I should read it. " + call), ("normal", "It says hello."))
    assert res["calls"] == [], res
    # An incomplete block in reasoning is not a call.
    res = run(("reasoning", 'maybe <tool_call>{"name": "read_file"'), ("normal", ""))
    assert res["calls"] == [], res


def test_native_tool_state_untouched():
    src = list(server._keep_open_tool_call(
        gens(("normal", "ok "), ("tool", '{"name": "read_file", "arguments": {"path": "n"}}'))
    ))
    out = list(ToolRepair(NAMES).run(iter(src)))
    assert [(g.text, g.state) for g in out] == [(g.text, g.state) for g in src]
    res = collect(out, tool_parser=mlx_server_json_parser())
    assert res["calls"] == [("read_file", {"path": "n"})], res


def mlx_server_json_parser():
    from mlx_lm.tool_parsers import json_tools

    return json_tools.parse_tool_call


def test_other_format_inside_native_tags():
    from mlx_lm.tool_parsers import qwen3_coder

    # qwen3_coder template, model wrote Hermes JSON between its tags.
    src = gens(("tool", '\n{"name": "read_file", "arguments": {"path": "h"}}\n'))
    res = collect(server._keep_open_tool_call(iter(src)), tool_parser=qwen3_coder.parse_tool_call)
    assert res["calls"] == [("read_file", {"path": "h"})], res


def test_stop_token_after_open_block():
    res = run(("normal", "<tool_call>\n<function=read_file>\n<parameter=path>\nw\n</parameter>\n"))
    assert res["calls"] == [("read_file", {"path": "w"})], res


def test_streaming_indices():
    src = gens(("normal", '<tool_call>{"name": "read", "arguments": {}}</tool_call>'))
    out = ToolRepair(NAMES).run(iter(src))
    res = collect(out, stream_mode=True)
    assert res["calls"] == [("read", {})], res


def test_opt_out_and_switch():
    class H:
        body = {"tool_call_repair": False}
        max_tokens = 10
        _pm_tool_names = NAMES

    assert server._repair_names(H, None) == set()
    H.body = {}
    assert server._repair_names(H, None) == NAMES
    H._pm_tool_names = set()
    assert server._repair_names(H, None) == set()
    assert server._repair_names(None, None) == set()


def test_legacy_deepseek_messages():
    assert tool_repair.legacy_deepseek_template("{{ '<｜tool▁calls▁begin｜>' }} {% for m in messages %}")
    assert not tool_repair.legacy_deepseek_template("{% if tools %}<｜tool▁calls▁begin｜>{% endif %}")
    msgs = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "list"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "run_in_terminal", "arguments": "{\"command\": \"ls\"}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "a b"},
        {"role": "assistant", "content": "Two files."},
        {"role": "user", "content": "read a"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "2", "type": "function", "function": {"name": "read_file", "arguments": {"path": "a"}}}]},
        {"role": "tool", "tool_call_id": "2", "content": "hi"},
    ]
    out = tool_repair.legacy_deepseek_messages(msgs, TOOLS[:1])
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user", "assistant", "tool"], out
    assert out[0]["content"].startswith("Be brief.\n\n## Tools")
    assert "`read_file`" in out[0]["content"]
    assert out[2]["content"] == (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>run_in_terminal\n```json\n"
        "{\"command\": \"ls\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"
        "<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>a b<｜tool▁output▁end｜><｜tool▁outputs▁end｜>Two files."
    )
    # The trailing result batch stays as tool messages.
    assert out[4]["content"].endswith("{\"path\": \"a\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
    assert out[5] == msgs[7]


def test_fuzz_text_and_calls():
    import random

    rng = random.Random(7)
    bits = ["word ", "<", ">", "{", "}", "`", "```", "\n", "[", "\"name\"", ": ", "<tool", "<function=",
            "<invoke name=\"nope\">", "</tool_call>", "<｜tool▁call▁begin｜>", "<｜tool▁sep｜>", "x<y ", "</div>"]
    calls = [
        '<tool_call>{"name": "read_file", "arguments": {"path": "f"}}</tool_call>',
        "<function=read_file><parameter=path>f</parameter></function>",
        '<invoke name="read_file"><parameter name="path">f</parameter></invoke>',
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>read_file<｜tool▁sep｜>{\"path\": \"f\"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
    ]
    for trial in range(400):
        text = "".join(rng.choice(bits) for _ in range(rng.randint(1, 40)))
        cuts = tuple(rng.randint(1, 6) for _ in range(5))
        src = gens(("normal", text), cuts=cuts)
        out = list(ToolRepair(NAMES).run(iter(src)))
        res = collect(out)
        if res["calls"]:
            # Random bits can spell a real call only by building JSON; never markup.
            assert all(c[0] in NAMES for c in res["calls"]), (text, res)
            continue
        assert res["text"] == text, (trial, text, res["text"])
        assert len(out) == len(src), (trial, text)

        call = rng.choice(calls)
        prose = "".join(rng.choice(bits[:1] + bits[7:8]) for _ in range(rng.randint(1, 5)))
        res = run(("normal", prose + call + " after"), cuts=cuts)
        assert res["calls"] == [("read_file", {"path": "f"})], (trial, call, cuts, res)
        assert res["text"] == prose + " after", (trial, res["text"])


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
