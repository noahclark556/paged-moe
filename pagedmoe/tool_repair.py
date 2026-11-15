# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tool calls the server's parser did not catch.

mlx-lm turns a call into `tool_calls` only when the model writes the exact
start tokens of the one parser the chat template picked. Anything else ends
up in the reply as text and the client never runs it:

  - the right markup tokenized another way (`<tool_call>` as plain pieces)
  - another family's markup (Hermes JSON from a qwen3_coder model, Qwen XML
    from a json_tools model, `<invoke name=...>`)
  - DeepSeek V3 / V3.1 native calls, which mlx-lm has no parser for at all
  - a bare `{"name": ..., "arguments": ...}` reply

ToolRepair sits on the generation stream of a request that sent tools. Plain
text passes through untouched until it hits call markup. A block is only
taken as a call once its function name is one the request offered; from there
it streams as tool text, so the client sees the same thing a native call
gives. A name the request did not offer releases the text as it was.

A call written inside reasoning is taken only when the turn ends with no call
and no reply text, so a model thinking about a call does not run it.

Clients that already repair calls themselves send `"tool_call_repair": false`
and get mlx-lm's output as before.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Iterable, Iterator, Optional

# Leads the tool text of a recovered block so the formatter parses it here.
SENTINEL = "\x1epm-tool\x1e"

# Start marker -> end marker. "" ends at the end of the stream.
_BLOCKS = (
    ("<tool_call>", "</tool_call>"),
    ("<function=", "</function>"),
    ("<minimax:tool_call>", "</minimax:tool_call>"),
    ("<invoke name=", "</invoke>"),
    ("<｜tool▁calls▁begin｜>", "<｜tool▁calls▁end｜>"),
    ("<｜tool▁call▁begin｜>", "<｜tool▁call▁end｜>"),
    ("<｜DSML｜function_calls>", "</｜DSML｜function_calls>"),
    ("<｜DSML｜invoke ", "</｜DSML｜invoke>"),
)
_STARTS = tuple(s for s, _ in _BLOCKS)
_END_OF = dict(_BLOCKS)

# A block whose name has not shown up by now is not a call.
_NAME_WINDOW = 400
# A reply that opens with JSON and has no "name" this far in is not a call.
_JSON_WINDOW = 160

# Each needs the character after the name, so a name still being written does
# not read as a shorter one.
_NAME_RES = (
    re.compile(r"<function=([^>\s]+)>"),
    re.compile(r"<(?:｜DSML｜)?invoke name=[\"']?([^\"'>\s]+)[\"'>\s]"),
    re.compile(r"<｜tool▁call▁begin｜>\s*function\s*<｜tool▁sep｜>\s*([^\s`<]+)\s*\n"),
    re.compile(r"<｜tool▁call▁begin｜>\s*(?!function\s*<｜tool▁sep｜>)([^\s<]+?)\s*<｜tool▁sep｜>"),
    re.compile(r"<tool_call>\s*(?:```(?:json)?\s*)?\{\s*\"name\"\s*:\s*\"([^\"]+)\""),
)
_JSON_NAME_RE = re.compile(r"\s*(?:```(?:json)?\s*)?\[?\s*\{\s*\"name\"\s*:\s*\"([^\"]+)\"")


def tool_names(tools: Any) -> set:
    """Function names an OpenAI `tools` list offers."""
    names = set()
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name:
            names.add(name)
    return names


# ---------------------------------------------------------------------------
# Parsers. Each takes block text and returns [{"name", "arguments"}] or [].
# ---------------------------------------------------------------------------


def _value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _trim(raw: str) -> str:
    if raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\n"):
        raw = raw[:-1]
    return raw


def _unfence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _args(raw: Any) -> Optional[dict]:
    if isinstance(raw, dict):
        return raw
    if raw is None:
        return {}
    if isinstance(raw, str):
        s = _unfence(raw)
        if not s:
            return {}
        try:
            v = json.loads(s)
        except ValueError:
            return None
        return v if isinstance(v, dict) else None
    return None


def _json_calls(value: Any) -> list:
    items = value if isinstance(value, list) else [value]
    calls = []
    for item in items:
        if not isinstance(item, dict):
            return []
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            return []
        args = _args(fn.get("arguments", fn.get("parameters")))
        if args is None:
            return []
        calls.append({"name": name, "arguments": args})
    return calls


_QWEN_FN_RE = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.DOTALL)
_QWEN_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.DOTALL)


def _parse_qwen_xml(text: str) -> list:
    calls = []
    for name, body in _QWEN_FN_RE.findall(text):
        args = {k: _value(_trim(v)) for k, v in _QWEN_PARAM_RE.findall(body)}
        calls.append({"name": name, "arguments": args})
    return calls


_INVOKE_RE = re.compile(r"<invoke name=[\"']?([^\"'>\s]+)[\"']?\s*>(.*?)(?:</invoke>|$)", re.DOTALL)
_INVOKE_PARAM_RE = re.compile(r"<parameter name=[\"']?([^\"'>\s]+)[\"']?\s*>(.*?)</parameter>", re.DOTALL)


def _parse_invoke(text: str) -> list:
    calls = []
    for name, body in _INVOKE_RE.findall(text):
        args = {k: _value(_trim(v)) for k, v in _INVOKE_PARAM_RE.findall(body)}
        calls.append({"name": name, "arguments": args})
    return calls


_DSML_INVOKE_RE = re.compile(r"<｜DSML｜invoke name=\"([^\"]+)\"\s*>(.*?)(?:</｜DSML｜invoke>|$)", re.DOTALL)
_DSML_PARAM_RE = re.compile(
    r"<｜DSML｜parameter name=\"([^\"]+)\"(?:\s+string=\"(true|false)\")?\s*>(.*?)</｜DSML｜parameter>",
    re.DOTALL,
)


def _parse_dsml(text: str) -> list:
    calls = []
    for name, body in _DSML_INVOKE_RE.findall(text):
        args = {}
        for key, is_str, raw in _DSML_PARAM_RE.findall(body):
            args[key] = raw if is_str == "true" else _value(raw)
        calls.append({"name": name, "arguments": args})
    return calls


_DS_CALL_RE = re.compile(r"<｜tool▁call▁begin｜>(.*?)(?:<｜tool▁call▁end｜>|$)", re.DOTALL)


def _parse_deepseek(text: str) -> list:
    calls = []
    for body in _DS_CALL_RE.findall(text):
        if "<｜tool▁sep｜>" not in body:
            return []
        head, rest = body.split("<｜tool▁sep｜>", 1)
        if head.strip() == "function" and "\n" in rest.strip():
            # V3: function<sep>NAME\n```json\n{...}\n```
            name, raw = rest.strip().split("\n", 1)
        else:
            # V3.1: NAME<sep>{...}
            name, raw = head, rest
        name = name.strip()
        args = _args(raw)
        if not name or args is None:
            return []
        calls.append({"name": name, "arguments": args})
    return calls


def _parse_json(text: str) -> list:
    s = text.strip()
    for tag in ("<tool_call>", "</tool_call>"):
        s = s.replace(tag, "")
    s = _unfence(s)
    if not s:
        return []
    try:
        value, _ = json.JSONDecoder().raw_decode(s)
    except ValueError:
        return []
    return _json_calls(value)


def parse_block(text: str) -> list:
    """Calls in one block of tool markup, in whatever format it is written."""
    if "｜DSML｜" in text:
        return _parse_dsml(text)
    if "<｜tool▁call▁begin｜>" in text:
        return _parse_deepseek(text)
    if "<function=" in text:
        return _parse_qwen_xml(text)
    if "<invoke name=" in text:
        return _parse_invoke(text)
    return _parse_json(text)


def find_calls(text: str, names: set) -> list:
    """Complete call blocks in free text whose names are all offered tools."""
    calls, seen, pos = [], set(), 0
    while True:
        hits = [(text.find(s, pos), s) for s in _STARTS]
        hits = [(i, s) for i, s in hits if i >= 0]
        if not hits:
            return calls
        i, start = min(hits)
        j = text.find(_END_OF[start], i + len(start))
        if j < 0:
            return calls
        j += len(_END_OF[start])
        block = parse_block(text[i:j])
        if block and all(c["name"] in names for c in block):
            for c in block:
                key = (c["name"], json.dumps(c["arguments"], sort_keys=True))
                if key not in seen:
                    seen.add(key)
                    calls.append(c)
        pos = j


def parse_tool_text(tool_text: str) -> Optional[list]:
    """Calls in tool text that carries the sentinel, or None if it does not.

    Text in front of the first sentinel is what mlx-lm's own parser should
    read; it comes back as the first element (possibly "").
    """
    if SENTINEL not in tool_text:
        return None
    head, *blocks = tool_text.split(SENTINEL)
    calls = []
    for block in blocks:
        calls.extend(parse_block(block))
    return [head] + calls


# ---------------------------------------------------------------------------
# Stream filter
# ---------------------------------------------------------------------------


def _name_in(block: str) -> Optional[str]:
    for rx in _NAME_RES:
        m = rx.search(block)
        if m:
            return m.group(1)
    return None


def _partial_tail(text: str) -> int:
    """Length of the longest suffix of `text` that a start marker begins with."""
    best = 0
    for s in _STARTS:
        for n in range(min(len(s) - 1, len(text)), best, -1):
            if text.endswith(s[:n]):
                best = n
                break
    return best


class ToolRepair:
    """Recover calls from one request's generation stream (see module doc)."""

    def __init__(self, names: Iterable[str], label: str = ""):
        self.names = set(names)
        self.label = label
        self.buf = ""            # plain text held back, not yet sent
        self.carriers: list = []  # gens whose text is in buf
        self.start = None        # start marker of the block being streamed as tool
        self.block = ""          # raw text of that block so far
        self.json_block = False  # the block is a bare JSON reply
        self.trailer = ""        # closing text that may follow the block
        self.fence_open = False  # the JSON block opened a ``` fence
        self.sent = ""           # plain text sent so far
        self.out = ""            # plain text released, waiting for a token
        self.reasoning: list = []
        self.called = False
        self.prev_state = None
        self.last = None
        self.tail = None         # held stop-token gen, see run()
        self.recovered: list = []

    # -- output ---------------------------------------------------------

    def _emit(self, text: str, state: str):
        if self.carriers:
            gen = dataclasses.replace(self.carriers.pop(0), text=text, state=state)
        else:
            # More pieces than tokens: only when a whole block and the text
            # before it arrive in one detokenizer flush.
            gen = dataclasses.replace(self.last, text=text, state=state, finish_reason=None)
        self.prev_state = state
        return gen

    def _normal(self, text: str):
        """Queue plain text. It goes out as one piece, on one token."""
        self.out += text
        self.sent += text

    def _flush(self):
        if self.out:
            text, self.out = self.out, ""
            yield self._emit(text, "normal")

    def _spare(self, state: str):
        yield from self._flush()
        while self.carriers and not self.buf:
            yield self._emit("", state)

    def _tool(self, text: str):
        yield from self._flush()
        if self.block == "" and not text.startswith(SENTINEL):
            text = SENTINEL + text
        yield self._emit(text, "tool")

    # -- block handling -------------------------------------------------

    def _commit(self, name: str, start: str):
        self.start = start
        self.json_block = start == ""
        self.block = ""
        self.recovered.append(name)

    def _close(self):
        self.called = True
        self.start = None
        self.block = ""
        self.json_block = False

    def _block_end(self) -> Optional[int]:
        """Index in buf just past the end of the open block, or None."""
        if self.json_block:
            if "}" not in self.buf and "]" not in self.buf:
                return None
            joined = self.block + self.buf
            lead = len(joined) - len(joined.lstrip())
            body = joined[lead:]
            fence = 0
            if body.startswith("```"):
                fence = body.find("\n") + 1
                if fence == 0:
                    return None
            try:
                _, end = json.JSONDecoder().raw_decode(body[fence:])
            except ValueError:
                return None
            end += lead + fence
            self.fence_open = bool(fence)
            if fence:
                close = joined.find("```", end)
                if close >= 0 and not joined[end:close].strip():
                    end = close + 3
                    self.fence_open = False
            k = end - len(self.block)
            return k if k > 0 else None
        end = _END_OF[self.start]
        if self.block:
            # Only the tail can hold a marker split across the last two pieces.
            seam = self.block[-(len(end) - 1):]
            j = (seam + self.buf).find(end)
            if j < 0:
                return None
            k = j + len(end) - len(seam)
        else:
            j = self.buf.find(end, len(self.start))
            if j < 0:
                return None
            k = j + len(end)
        return k

    def _advance(self):
        while True:
            if self.start is not None:
                k = self._block_end()
                if k is None:
                    if self.buf:
                        text, self.buf = self.buf, ""
                        yield from self._tool(text)
                        self.block += text
                    yield from self._spare("tool")
                    return
                text, self.buf = self.buf[:k], self.buf[k:]
                yield from self._tool(text)
                if self.start == "<function=":
                    self.trailer = "</tool_call>"
                elif self.json_block and self.fence_open:
                    self.trailer = "```"
                self._close()
                continue

            if self.trailer:
                # A bare <function=...> block often still gets its </tool_call>,
                # and a fenced JSON call its closing fence.
                s = self.buf.lstrip()
                if s.startswith(self.trailer):
                    k = len(self.buf) - len(s) + len(self.trailer)
                    text, self.buf = self.buf[:k], self.buf[k:]
                    yield from self._flush()
                    yield self._emit(text, "tool")
                    self.trailer = ""
                    continue
                if self.trailer.startswith(s):
                    if self.buf:
                        yield from self._flush()
                    else:
                        yield from self._spare("tool")
                    return
                self.trailer = ""

            hits = [(self.buf.find(s), s) for s in _STARTS]
            hits = [(i, s) for i, s in hits if i >= 0]
            json_at = None
            if not self.sent.strip() and not self.called:
                stripped = self.buf.lstrip()
                if stripped[:1] in ("{", "[", "`"):
                    json_at = len(self.buf) - len(stripped)
            if json_at is not None and (not hits or json_at < min(hits)[0]):
                if json_at:
                    self._normal(self.buf[:json_at])
                    self.buf = self.buf[json_at:]
                m = _JSON_NAME_RE.match(self.buf)
                if m and m.group(1) in self.names:
                    self._commit(m.group(1), "")
                    continue
                if m or len(self.buf) > _JSON_WINDOW or not _could_be_json_call(self.buf):
                    self._normal(self.buf)
                    self.buf = ""
                    yield from self._spare("normal")
                yield from self._flush()
                return

            if not hits:
                keep = _partial_tail(self.buf)
                out = self.buf[: len(self.buf) - keep]
                if out:
                    self.buf = self.buf[len(out):]
                    self._normal(out)
                yield from self._spare("normal")
                return

            i, start = min(hits)
            if i:
                out, self.buf = self.buf[:i], self.buf[i:]
                self._normal(out)
            name = _name_in(self.buf[:_NAME_WINDOW + len(start)])
            if name is not None and name in self.names:
                self._commit(name, start)
                continue
            if name is None and len(self.buf) <= _NAME_WINDOW:
                yield from self._flush()
                return
            # Not a call to an offered tool: let the marker through as text.
            out, self.buf = self.buf[: len(start)], self.buf[len(start):]
            self._normal(out)

    def _settle(self, final: bool):
        """Send what is held because plain text stopped (state change or end)."""
        self.trailer = ""
        if self.start is not None:
            if self.buf:
                text, self.buf = self.buf, ""
                yield from self._tool(text)
            self._close()
            yield from self._spare("tool")
            return
        if self.buf:
            calls = parse_block(self.buf) if final and not self.sent.strip() else []
            if calls and all(c["name"] in self.names for c in calls):
                self.recovered.extend(c["name"] for c in calls)
                text, self.buf = self.buf, ""
                yield from self._flush()
                yield self._emit(SENTINEL + text, "tool")
                self.called = True
            else:
                text, self.buf = self.buf, ""
                self._normal(text)
        yield from self._spare(self.prev_state or "normal")

    # -- driver ---------------------------------------------------------

    def run(self, gens: Iterable) -> Iterator:
        for gen in gens:
            self.last = gen
            if self.tail is not None:
                yield self._pass(self.tail)
                self.tail = None
            state = getattr(gen, "state", None)
            if state == "normal":
                self.carriers.append(gen)
                self.buf += gen.text or ""
                yield from self._advance()
                continue
            if state is None and not (gen.text or ""):
                # The stop token. It carries whatever is still held, or is
                # kept so a call found in reasoning can ride on it; either
                # way the token count does not change.
                if self.buf and not self.carriers:
                    self.carriers.append(gen)
                    yield from self._settle(final=True)
                else:
                    yield from self._settle(final=True)
                    self.tail = gen
                continue
            yield from self._settle(final=state is None)
            if state == "reasoning":
                self.reasoning.append(gen.text or "")
            elif state == "tool":
                self.called = True
            yield self._pass(gen)
        yield from self._settle(final=True)
        if self.recovered:
            self._log("reply")
        tail, self.tail = self.tail, None
        if not self.called and not self.sent.strip() and self.reasoning and self.last is not None:
            calls = find_calls("".join(self.reasoning), self.names)
            if calls:
                text = SENTINEL + json.dumps(calls, ensure_ascii=False)
                if tail is not None:
                    yield dataclasses.replace(tail, text=text, state="tool")
                    tail = None
                else:
                    yield self._emit(text, "tool")
                self.called = True
                self.recovered = [c["name"] for c in calls]
                self._log("reasoning")
        if tail is not None:
            yield self._pass(tail)

    def _pass(self, gen):
        state = getattr(gen, "state", None)
        if state is None and self.prev_state == "tool" and not (gen.text or ""):
            # Stop token right after a recovered call: keep it in tool state
            # or handle_completion never closes the call.
            gen = dataclasses.replace(gen, state="tool")
            state = "tool"
        self.prev_state = state
        return gen

    def _log(self, where: str):
        names = ", ".join(self.recovered)
        print(f"[paged-moe] recovered tool call from the {where}: {names}", flush=True)
        self.recovered = []


def _could_be_json_call(text: str) -> bool:
    """`text` (opening with JSON) could still grow into {"name": "...", ...}."""
    s = text.lstrip()
    if s.startswith("`"):
        nl = s.find("\n")
        head = s if nl < 0 else s[:nl]
        if not ("```json".startswith(head) or (head.startswith("```") and head[3:].strip() in ("", "json"))):
            return False
        if nl < 0:
            return True
        s = s[nl + 1:].lstrip()
    if s.startswith("["):
        s = s[1:].lstrip()
    if not s:
        return True
    if not s.startswith("{"):
        return False
    s = s[1:].lstrip()
    target = '"name"'
    if len(s) < len(target):
        return target.startswith(s)
    if not s.startswith(target):
        return False
    s = s[len(target):].lstrip()
    if not s:
        return True
    if not s.startswith(":"):
        return False
    s = s[1:].lstrip()
    return not s or s.startswith('"')


def legacy_deepseek_template(template: Optional[str]) -> bool:
    """DeepSeek-V2 / V2.5: knows the call markup but never renders `tools`."""
    return bool(template) and "tool▁calls▁begin" in template and not re.search(r"\btools\b", template)


_DS_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
_DS_CALLS_END = "<｜tool▁calls▁end｜>"
_DS_CALL_BEGIN = "<｜tool▁call▁begin｜>"
_DS_CALL_END = "<｜tool▁call▁end｜>"
_DS_SEP = "<｜tool▁sep｜>"
_DS_EOS = "<｜end▁of▁sentence｜>"
_DS_OUTPUTS_BEGIN = "<｜tool▁outputs▁begin｜>"
_DS_OUTPUTS_END = "<｜tool▁outputs▁end｜>"
_DS_OUTPUT_BEGIN = "<｜tool▁output▁begin｜>"
_DS_OUTPUT_END = "<｜tool▁output▁end｜>"


def _legacy_tool_prompt(tools: list) -> str:
    specs = []
    for t in tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        spec = json.dumps(
            {"name": fn["name"], "description": fn.get("description") or "", "parameters": fn.get("parameters") or {}},
            ensure_ascii=False,
        )
        specs.append(f"- `{fn['name']}`:\n```json\n{spec}\n```")
    if not specs:
        return ""
    return "\n".join(["## Tools", "", "### Function", "", "You have the following functions available:", "", "\n".join(specs)])


def _legacy_call_markup(calls: list) -> str:
    parts = []
    for c in calls or []:
        fn = c.get("function") if isinstance(c, dict) else None
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        raw = fn.get("arguments")
        args = raw if isinstance(raw, str) else json.dumps(raw if raw is not None else {}, ensure_ascii=False)
        parts.append(f"{_DS_CALL_BEGIN}function{_DS_SEP}{fn['name']}\n```json\n{args}\n```{_DS_CALL_END}")
    return f"{_DS_CALLS_BEGIN}{chr(10).join(parts)}{_DS_CALLS_END}" if parts else ""


def _text(content: Any) -> str:
    return content if isinstance(content, str) else ""


def legacy_deepseek_messages(messages: list, tools: list) -> list:
    """Messages for a DeepSeek-V2 template, with tool turns written out.

    The V2 / V2.5 templates never render `tools`, and their history branch
    keeps two flags that are set once per conversation, so past the first tool
    round every call and result loses its opening markers and the model tends
    to answer in prose. Here the tool list goes into the system prompt and each
    call -> results -> reply chain becomes one assistant message in the model's
    own markup, which the template passes through. Only a trailing result batch
    stays as `tool` messages so the template closes it and hands the turn back.
    """
    out, i = [], 0
    while i < len(messages):
        m = messages[i]
        calls = m.get("tool_calls") if isinstance(m, dict) else None
        if not isinstance(m, dict) or m.get("role") != "assistant" or not calls:
            out.append(m)
            i += 1
            continue
        content, j = "", i
        while j < len(messages) and messages[j].get("role") == "assistant":
            cur = messages[j]
            cur_calls = cur.get("tool_calls") or []
            content += _text(cur.get("content")) + _legacy_call_markup(cur_calls)
            j += 1
            if not cur_calls:
                break
            k = j
            while k < len(messages) and messages[k].get("role") == "tool":
                k += 1
            if k == j or k == len(messages):
                break
            outputs = [f"{_DS_OUTPUT_BEGIN}{_text(t.get('content'))}{_DS_OUTPUT_END}" for t in messages[j:k]]
            content += f"{_DS_EOS}{_DS_OUTPUTS_BEGIN}{chr(10).join(outputs)}{_DS_OUTPUTS_END}"
            j = k
        out.append({"role": "assistant", "content": content})
        i = j

    prompt = _legacy_tool_prompt(tools)
    if not prompt:
        return out
    for n in range(len(out) - 1, -1, -1):
        if isinstance(out[n], dict) and out[n].get("role") == "system":
            base = _text(out[n].get("content"))
            out[n] = {**out[n], "content": f"{base}\n\n{prompt}" if base else prompt}
            return out
    return [{"role": "system", "content": prompt}] + out
