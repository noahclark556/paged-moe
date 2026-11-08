# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
A tool call the model ends with a stop token instead of its close tag still
reaches the client.

mlx-lm's handle_completion only parses leftover tool text when the last token's
state was "tool", and a stop token comes out of the state machine with state
None. So `<tool_call>{...}<|im_end|>` used to vanish: no text, no reasoning, no
call. This drives the real handle_completion with a scripted token stream,
first without the server's patches (the call is dropped) and then with them
(the call arrives, streamed and not).

Run:
    python tests/test_tool_stream.py
"""

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx_lm.server as S
from mlx_lm.tool_parsers import json_tools

from pagedmoe import server

BODY = ['{"name": "read_file", ', '"arguments": {"filePath": "lib/a.dart", ', '"startLine": 1}}']


def scripted(closed: bool):
    toks = [S.Response("Sure.", 10, "normal", None, 0.0, None, ())]
    toks.append(S.Response("", 11, "tool", (11,), 0.0, None, ()))
    toks += [S.Response(t, 12 + i, "tool", None, 0.0, None, ()) for i, t in enumerate(BODY)]
    if closed:
        toks.append(S.Response("", 20, "normal", (20,), 0.0, None, ()))
    toks.append(S.Response("", 21, None, (21,), 0.0, "stop", ()))
    return toks


def fake_generate(self, request, args, progress_callback=None):
    ctx = S.GenerationContext(
        has_tool_calling=True,
        has_thinking=False,
        tool_parser=json_tools.parse_tool_call,
        sequences={},
        prompt=[1, 2, 3],
    )
    return ctx, iter(self.script)


def handler(stream: bool, script):
    h = object.__new__(S.APIHandler)
    for k, v in dict(
        requested_model="m", requested_draft_model=None, adapter=None,
        temperature=0.0, top_p=1.0, top_k=0, min_p=0.0,
        xtc_probability=0.0, xtc_threshold=0.0, logit_bias=None,
        repetition_penalty=None, repetition_context_size=20,
        presence_penalty=None, presence_context_size=20,
        frequency_penalty=None, frequency_context_size=20,
        max_tokens=64, num_draft_tokens=0, logprobs=False, top_logprobs=0,
        seed=None, chat_template_kwargs=None, stream=stream, stream_options=None,
        request_id="r", system_fingerprint="f", created=0,
        object_type="chat.completion.chunk" if stream else "chat.completion",
    ).items():
        setattr(h, k, v)
    gen = object.__new__(S.ResponseGenerator)
    gen.script = script
    h.response_generator = gen
    h.wfile = io.BytesIO()
    h._set_stream_headers = h._set_completion_headers = lambda *a: None
    h.end_headers = lambda: None
    h.send_header = lambda *a: None
    return h


def calls_in(h, stream: bool):
    raw = h.wfile.getvalue().decode()
    if not stream:
        return json.loads(raw)["choices"][0]["message"].get("tool_calls", [])
    out = []
    for line in raw.splitlines():
        if line.startswith("data: {"):
            out += json.loads(line[6:])["choices"][0]["delta"].get("tool_calls", [])
    return out


def run(stream: bool, closed: bool):
    h = handler(stream, scripted(closed))
    req = S.CompletionRequest("chat", "", [], [{"type": "function"}], None)
    h.handle_completion(req, [])
    return calls_in(h, stream), h.wfile.getvalue().decode()


def main():
    S.ResponseGenerator.generate = fake_generate

    # Upstream alone: a closed call arrives, an unclosed one is dropped.
    for stream in (False, True):
        assert len(run(stream, closed=True)[0]) == 1
        assert run(stream, closed=False)[0] == [], "upstream started keeping it; drop the patch"
    print("upstream: unclosed call dropped")

    server._install_tool_stream()
    for stream in (False, True):
        for closed in (True, False):
            calls, raw = run(stream, closed)
            assert len(calls) == 1, (stream, closed, raw)
            fn = calls[0]["function"]
            assert fn["name"] == "read_file"
            assert json.loads(fn["arguments"]) == {"filePath": "lib/a.dart", "startLine": 1}
            if stream:
                frames = [
                    line[len(": ga-tool "):].split(" ", 1)
                    for line in raw.splitlines()
                    if line.startswith(": ga-tool ")
                ]
                relayed = "".join(json.loads(text) for _, text in frames)
                assert relayed == "".join(BODY), relayed
                want = 1 + len(BODY) + (0 if closed else 1)
                assert sum(int(n) for n, _ in frames) == want, frames
    print("patched: closed and unclosed calls arrive, streamed and not")
    print("all ok")


if __name__ == "__main__":
    main()
