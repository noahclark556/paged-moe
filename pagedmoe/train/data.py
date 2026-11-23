# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Datasets, formatting and batching.

Rows are plain dicts. The format is detected per row:

  {"messages": [...], "tools": [...]?}        chat; assistant turns are trained
  {"prompt": ..., "completion": ...}          prompt / completion; completion trained
  {"text": ...}                               raw text; every token trained (CPT)
  {"prompt": ..., "chosen": ..., "rejected"}  preference pair (ORPO, DPO, SimPO, CPO)
  {"prompt": ..., "completion", "label"}      unpaired preference (KTO)
  {"prompt": ..., anything else}              GRPO prompt; extra fields go to rewards
  {"input_ids": [...], "labels" | "loss_mask"} already tokenized

`prompt`, `completion`, `chosen` and `rejected` can be strings or message
lists. Chosen / rejected can also be whole conversations with no prompt (the
shared prefix becomes the prompt).

Anything else is one function away: pass `formatter=fn(row, fmt) -> Example`
(fmt is the run's Formatter, with .encode, .render and .tok) to a trainer and the rest of the pipeline (batching, masking,
streaming, checkpoints) stays the same. An Example can carry `embeds`, a
[P, hidden] array spliced into the input at `embeds_at`, for models that
condition on something other than tokens (point clouds, images, audio
features). See docs/paged-training.md, "Custom inputs".

Sources: a list of dicts, a .jsonl / .json / .parquet / .csv file, a
directory holding train.* and valid.*, a .npy / .npz of token ids, or a
Hugging Face dataset id ("hf:org/name" or "org/name" when not a local path).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import mlx.core as mx
import numpy as np

from .executor import Batch


# --------------------------------------------------------------------- types


@dataclass
class Example:
    ids: list[int]
    loss_mask: list[int]  # per token: 1 where this token is a training target
    embeds: np.ndarray | None = None
    embeds_at: int = 0
    meta: dict = field(default_factory=dict)
    row: Any = None

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class Pair:
    chosen: Example
    rejected: Example
    row: Any = None

    def __len__(self) -> int:
        return max(len(self.chosen), len(self.rejected))


@dataclass
class Prompt:
    ids: list[int]
    prompt: Any
    row: Any = None

    def __len__(self) -> int:
        return len(self.ids)


# ------------------------------------------------------------------- loading


def _read_file(p: Path) -> list[dict]:
    suf = p.suffix.lower()
    if suf == ".jsonl":
        rows = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suf == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            for k in ("data", "rows", "train", "examples"):
                if isinstance(data.get(k), list):
                    return data[k]
            raise ValueError(f"{p}: expected a list of rows")
        return data
    if suf == ".parquet":
        import pandas as pd

        return pd.read_parquet(p).to_dict("records")
    if suf == ".csv":
        import csv

        with open(p, newline="") as f:
            return list(csv.DictReader(f))
    if suf in (".npy", ".npz"):
        return _read_tokens(p)
    raise ValueError(f"unsupported data file {p}")


def _read_tokens(p: Path) -> list[dict]:
    """Token ids from .npy (2-D, or 1-D object of sequences) or .npz.

    .npz keys: input_ids (2-D, or object array), optional lengths, optional
    loss_mask / labels (-100 = ignored), optional embeds (+ embeds_at).
    Padding in a 2-D array is cut by `lengths` when given.
    """
    if p.suffix == ".npy":
        arr = np.load(p, allow_pickle=True)
        return [{"input_ids": np.asarray(r).astype(np.int64).tolist()} for r in arr]
    z = np.load(p, allow_pickle=True)
    ids = z["input_ids"]
    lengths = z["lengths"] if "lengths" in z else None
    rows = []
    for i in range(len(ids)):
        r = np.asarray(ids[i]).astype(np.int64)
        n = int(lengths[i]) if lengths is not None else len(r)
        row: dict = {"input_ids": r[:n].tolist()}
        for key in ("loss_mask", "labels"):
            if key in z:
                row[key] = np.asarray(z[key][i])[:n].tolist()
        if "embeds" in z:
            row["embeds"] = np.asarray(z["embeds"][i])
            row["embeds_at"] = int(z["embeds_at"][i]) if "embeds_at" in z else 0
        rows.append(row)
    return rows


def load_rows(src, split: str = "train") -> list[dict]:
    if src is None:
        return []
    if isinstance(src, (list, tuple)):
        return list(src)
    if callable(src):
        return list(src())
    if not isinstance(src, (str, Path)):
        return list(src)
    s = str(src)
    if s.startswith("hf:"):
        return _hf(s[3:], split)
    p = Path(s).expanduser()
    if p.is_dir():
        names = [split] + (["valid", "validation", "val", "test"] if split != "train" else [])
        for name in names:
            for ext in (".jsonl", ".json", ".parquet", ".csv", ".npz", ".npy"):
                f = p / f"{name}{ext}"
                if f.exists():
                    return _read_file(f)
        return []
    if p.exists():
        return _read_file(p)
    files = (".jsonl", ".json", ".parquet", ".csv", ".npz", ".npy")
    if "/" in s and not s.startswith((".", "/", "~")) and not s.lower().endswith(files):
        return _hf(s, split)
    raise FileNotFoundError(f"no such data file or folder: {s} (relative paths start from {Path.cwd()})")


def _hf(name: str, split: str) -> list[dict]:
    try:
        import datasets
    except ImportError as e:
        raise ImportError("Hugging Face datasets: pip install datasets") from e
    if ":" in name:
        name, split = name.split(":", 1)
    return list(datasets.load_dataset(name, split=split))


# ---------------------------------------------------------------- formatting


class Formatter:
    """Turns rows into Examples / Pairs / Prompts for one tokenizer."""

    def __init__(
        self,
        tokenizer,
        *,
        max_len: int = 2048,
        chat: bool | None = None,
        completion_only: bool = True,
        template_kwargs: dict | None = None,
        system: str | None = None,
    ):
        self.tok = tokenizer
        self.hf = getattr(tokenizer, "_tokenizer", tokenizer)
        self.max_len = int(max_len)
        has_template = bool(getattr(self.hf, "chat_template", None))
        self.chat = has_template if chat is None else (chat and has_template)
        self.completion_only = completion_only
        self.kw = dict(template_kwargs or {})
        self.system = system
        self.pack = False
        self.eos = self._eos_id()
        self._warned: set[str] = set()

    # -- tokenizer helpers

    def _eos_id(self) -> int | None:
        eos = getattr(self.tok, "eos_token_id", None)
        if eos is None:
            ids = getattr(self.tok, "eos_token_ids", None)
            if ids:
                eos = sorted(ids)[0]
        return eos

    @property
    def pad_id(self) -> int:
        pad = getattr(self.hf, "pad_token_id", None)
        if pad is None:
            pad = self.eos if self.eos is not None else 0
        return int(pad)

    def encode(self, text: str, special: bool = False) -> list[int]:
        return list(self.hf.encode(text, add_special_tokens=special))

    def render(self, messages, gen: bool = False, tools=None) -> str:
        kw = dict(self.kw)
        if tools:
            kw["tools"] = tools
        msgs = _parsed_tool_args(messages)
        try:
            return self.hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, **kw)
        except Exception as e:
            # OpenAI rows send "content": null on tool call turns, and older
            # templates (Qwen3) call string methods on it. Templates that
            # handle null render as before; only a failing one gets "".
            if not any(isinstance(m, dict) and "content" in m and m["content"] is None for m in msgs):
                raise ValueError(f"the model's chat template failed on this row ({e})") from e
        self._warn("null-content", "chat template does not accept null message content; using empty text")
        msgs = [{**m, "content": ""} if isinstance(m, dict) and m.get("content", "") is None else m for m in msgs]
        try:
            return self.hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, **kw)
        except Exception as e:
            raise ValueError(f"the model's chat template failed on this row ({e})") from e

    def _warn(self, key: str, msg: str) -> None:
        if key not in self._warned and key not in _WARNED:
            self._warned.add(key)
            _WARNED.add(key)
            print(f"[train] {msg}")

    def _msgs(self, prompt) -> list[dict]:
        if isinstance(prompt, list):
            msgs = list(prompt)
        else:
            msgs = [{"role": "user", "content": str(prompt)}]
        if self.system and not any(m.get("role") == "system" for m in msgs):
            msgs = [{"role": "system", "content": self.system}] + msgs
        return msgs

    def _spans_to_mask(self, text: str, spans: list[tuple[int, int]]):
        """Token ids for text and a mask of tokens overlapping any char span."""
        if getattr(self.hf, "is_fast", False):
            enc = self.hf(text, add_special_tokens=False, return_offsets_mapping=True)
            ids = list(enc["input_ids"])
            mask = [0] * len(ids)
            for t, (a, b) in enumerate(enc["offset_mapping"]):
                for s, e in spans:
                    if a < e and b > s:
                        mask[t] = 1
                        break
            return ids, mask
        # Slow tokenizer: prefix encodes, assuming span edges are token edges.
        ids = self.encode(text)
        mask = [0] * len(ids)
        for s, e in spans:
            lo = len(self.encode(text[:s]))
            hi = len(self.encode(text[:e]))
            for t in range(lo, min(hi, len(ids))):
                mask[t] = 1
        return ids, mask

    def _turn_start(self, messages, k: int, full: str, tools) -> int | None:
        """Char offset in `full` where assistant turn k starts.

        Normally the end of the generation prompt. Some templates add to the
        generation prompt what the turn itself renders differently (an empty
        think block when thinking is off but the row carries reasoning), so
        then the turn starts where the two diverge, never before the history.
        """
        pre = self.render(messages[:k], gen=True, tools=tools)
        if full.startswith(pre):
            return len(pre)
        hist = self.render(messages[:k], tools=tools) if k else ""
        if not full.startswith(hist):
            return None
        n = len(hist)
        while n < len(pre) and n < len(full) and pre[n] == full[n]:
            n += 1
        return n

    # -- formats

    def chat_example(self, messages, tools=None, row=None) -> Example:
        messages = self._msgs(messages)
        full = self.render(messages, tools=tools)
        if not self.completion_only:
            ids = self.encode(full)
            return self._fit(Example(ids, [1] * len(ids), row=row))
        spans = []
        ok = True
        for k, m in enumerate(messages):
            if m.get("role") != "assistant":
                continue
            start = self._turn_start(messages, k, full, tools)
            upto = self.render(messages[: k + 1], tools=tools)
            if start is None or not full.startswith(upto) or len(upto) < start:
                ok = False
                break
            spans.append((start, len(upto)))
        if not ok:
            # The template rewrites earlier turns (some strip old reasoning),
            # so only the final assistant turn has a stable prefix.
            self._warn(
                "prefix",
                "chat template rewrites earlier turns; training the final assistant turn only",
            )
            start = self._turn_start(messages, len(messages) - 1, full, tools)
            if messages[-1].get("role") != "assistant" or start is None:
                raise ValueError("cannot find the assistant turn in the rendered chat")
            spans = [(start, len(full))]
        if not spans:
            raise ValueError("chat row has no assistant turn to train on")
        ids, mask = self._spans_to_mask(full, spans)
        return self._fit(Example(ids, mask, row=row))

    def prompt_completion(self, prompt, completion, row=None) -> Example:
        if self.chat:
            msgs = self._msgs(prompt)
            if isinstance(completion, list):
                msgs = msgs + list(completion)
            else:
                msgs = msgs + [{"role": "assistant", "content": str(completion)}]
            if self.completion_only:
                n_out = len(completion) if isinstance(completion, list) else 1
                pre = self.render(msgs[:-n_out], gen=True)
                full = self.render(msgs)
                if full.startswith(pre):
                    ids, mask = self._spans_to_mask(full, [(len(pre), len(full))])
                    return self._fit(Example(ids, mask, row=row))
            return self.chat_example(msgs, row=row)
        p = self.encode(str(prompt), special=True)
        c = self.encode(str(completion))
        if self.eos is not None:
            c = c + [self.eos]
        mask = ([0] * len(p) if self.completion_only else [1] * len(p)) + [1] * len(c)
        return self._fit(Example(p + c, mask, row=row))

    def text(self, text: str, row=None) -> Example:
        ids = self.encode(str(text), special=True)
        if self.eos is not None:
            ids = ids + [self.eos]
        ex = Example(ids, [1] * len(ids), row=row)
        # packing cuts long text into windows later, so nothing is lost
        return ex if self.pack else self._fit(ex)

    def tokens(self, row: dict) -> Example:
        ids = [int(t) for t in row["input_ids"]]
        if "loss_mask" in row:
            mask = [1 if m else 0 for m in row["loss_mask"]]
        elif "labels" in row:
            mask = [0 if int(lab) == -100 else 1 for lab in row["labels"]]
        else:
            mask = [1] * len(ids)
        emb = row.get("embeds")
        if isinstance(emb, (str, Path)):
            emb = np.load(Path(emb).expanduser())
        return self._fit(Example(ids, mask, embeds=emb, embeds_at=int(row.get("embeds_at", 0)), row=row))

    def prompt_ids(self, prompt) -> list[int]:
        if self.chat:
            return self.encode(self.render(self._msgs(prompt), gen=True))
        return self.encode(str(prompt), special=True)

    def _split_conv(self, conv):
        if isinstance(conv, list) and conv and isinstance(conv[-1], dict):
            return conv[:-1], [conv[-1]]
        return None, conv

    def pair(self, row) -> Pair:
        prompt = row.get("prompt")
        chosen, rejected = row["chosen"], row["rejected"]
        if prompt is None:
            pc, chosen = self._split_conv(chosen)
            _, rejected = self._split_conv(rejected)
            prompt = pc if pc else ""
        c = self.prompt_completion(prompt, chosen, row=row)
        r = self.prompt_completion(prompt, rejected, row=row)
        return Pair(c, r, row=row)

    def _fit(self, ex: Example) -> Example:
        """Cut to max_len, keeping the start (the prompt) and the earliest targets."""
        if len(ex.ids) > self.max_len:
            self._warn("trunc", f"rows longer than max_seq_len={self.max_len} are truncated")
            ex.ids = ex.ids[: self.max_len]
            ex.loss_mask = ex.loss_mask[: self.max_len]
        return ex

    # -- dispatch

    def kind(self, row: dict) -> str:
        if "input_ids" in row:
            return "tokens"
        if "chosen" in row and "rejected" in row:
            return "pair"
        if "messages" in row or "conversations" in row:
            return "chat"
        if "prompt" in row and "completion" in row and "label" in row:
            return "kto"
        if "prompt" in row and "completion" in row:
            return "prompt_completion"
        if "text" in row:
            return "text"
        if "prompt" in row:
            return "prompt"
        raise ValueError(f"cannot tell the format of a row with keys {sorted(row)}")

    def example(self, row: dict) -> Example:
        k = self.kind(row)
        if k == "tokens":
            return self.tokens(row)
        if k == "chat":
            msgs = row.get("messages") or _sharegpt(row["conversations"])
            return self.chat_example(msgs, tools=row.get("tools"), row=row)
        if k in ("prompt_completion", "kto"):
            ex = self.prompt_completion(row["prompt"], row["completion"], row=row)
            if k == "kto":
                ex.meta["label"] = 1 if _truthy(row["label"]) else 0
            return ex
        if k == "text":
            return self.text(row["text"], row=row)
        raise ValueError(f"a {k!r} row cannot be used as a single training example")


_WARNED: set[str] = set()  # notices already printed by any Formatter in this process


def _truthy(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "good", "desirable", "chosen")
    return bool(v)


def _parsed_tool_args(messages: list[dict]) -> list[dict]:
    """OpenAI rows carry tool call arguments as a JSON string; HF templates
    iterate them as a mapping. Copies, so the dataset rows stay as loaded."""
    out = []
    for m in messages:
        calls = m.get("tool_calls") if isinstance(m, dict) else None
        if not calls:
            out.append(m)
            continue
        fixed = []
        for c in calls:
            fn = c.get("function") if isinstance(c, dict) else None
            args = fn.get("arguments") if isinstance(fn, dict) else None
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    args = None
                if isinstance(args, dict):
                    c = {**c, "function": {**fn, "arguments": args}}
            fixed.append(c)
        out.append({**m, "tool_calls": fixed})
    return out


def _sharegpt(conv) -> list[dict]:
    roles = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
             "system": "system", "tool": "tool"}
    out = []
    for m in conv:
        role = roles.get(m.get("from", m.get("role")), m.get("from", "user"))
        out.append({"role": role, "content": m.get("value", m.get("content", ""))})
    return out


def pack(examples: Iterable[Example], max_len: int) -> list[Example]:
    """Concatenate examples into max_len windows (continued pretraining)."""
    out: list[Example] = []
    ids: list[int] = []
    mask: list[int] = []
    for ex in examples:
        ids.extend(ex.ids)
        mask.extend(ex.loss_mask)
        while len(ids) >= max_len:
            out.append(Example(ids[:max_len], mask[:max_len]))
            ids, mask = ids[max_len:], mask[max_len:]
    if len(ids) > 1:
        out.append(Example(ids, mask))
    return out


# ------------------------------------------------------------------ batching


def _round_up(n: int, m: int = 32) -> int:
    return int(math.ceil(n / m) * m)


class Schedule:
    """Deterministic steps of micro-batches, reproducible from (seed, epoch).

    A step holds `per_step` units (examples, pairs or prompts). Inside a step
    units are sorted by length and cut into micro-batches whose padded size
    (rows x longest row) stays under `micro_tokens` (see split). The optimizer
    batch is therefore always `per_step` units; micro-batching only bounds
    memory.
    """

    def __init__(self, units: list, per_step: int, micro_tokens: int, seed: int = 0,
                 rows_per_unit: int = 1, max_rows: int = 64, micro_cost: float | None = None):
        if not units:
            raise ValueError("no training data")
        self.units = units
        self.per_step = max(1, min(int(per_step), len(units)))
        self.micro_tokens = int(micro_tokens)
        self.seed = int(seed)
        self.rows_per_unit = rows_per_unit
        self.max_rows = max_rows
        self.micro_cost = float(self.MICRO_COST if micro_cost is None else micro_cost)

    @property
    def steps_per_epoch(self) -> int:
        return int(math.ceil(len(self.units) / self.per_step))

    def order(self, epoch: int) -> list[int]:
        idx = list(range(len(self.units)))
        random.Random(self.seed * 7919 + epoch).shuffle(idx)
        return idx

    def step(self, epoch: int, k: int) -> list[list]:
        idx = self.order(epoch)[k * self.per_step : (k + 1) * self.per_step]
        units = sorted((self.units[i] for i in idx), key=len, reverse=True)
        return [units[a:b] for a, b in self.split([_round_up(max(1, len(u) - 1)) for u in units])]

    # Default fixed cost of one more micro-batch, in padded tokens, when the
    # probe has not measured it. Each one restacks and dequantizes a layer's
    # experts (~1.3k tokens of work on a 235B).
    MICRO_COST = 2048

    def split(self, lens: list[int]) -> list[tuple[int, int]]:
        """Cut rows sorted longest first into micro-batches.

        Each micro-batch is padded to its first (longest) row. This picks the
        cut points with the fewest padded tokens (plus micro_cost per
        micro-batch) under the same limits as before; a greedy fill can leave
        a long row padding a group of short ones.
        """
        n = len(lens)
        r = self.rows_per_unit
        best = [0.0] + [math.inf] * n
        cut = [0] * (n + 1)
        for b in range(1, n + 1):
            for a in range(b - 1, -1, -1):
                rows = (b - a) * r
                if b - a > 1 and (rows * lens[a] > self.micro_tokens or rows > self.max_rows):
                    break
                c = best[a] + rows * lens[a] + self.micro_cost
                if c < best[b]:
                    best[b], cut[b] = c, a
        out = []
        b = n
        while b > 0:
            out.append((cut[b], b))
            b = cut[b]
        return out[::-1]


def collate(units: list, pad_id: int, kind: str = "example", extra: dict | None = None) -> Batch:
    """Units -> one padded Batch. kind: example | pair | prompt-free completion rows."""
    if kind == "pair":
        rows = [u.chosen for u in units] + [u.rejected for u in units]
    else:
        rows = list(units)
    T = _round_up(max(len(r.ids) - 1 for r in rows))
    B = len(rows)
    tok = np.full((B, T), pad_id, np.int32)
    tgt = np.full((B, T), pad_id, np.int32)
    msk = np.zeros((B, T), np.float32)
    for i, r in enumerate(rows):
        ids = np.asarray(r.ids, np.int64)
        n = len(ids) - 1
        tok[i, :n] = ids[:-1]
        tgt[i, :n] = ids[1:]
        msk[i, :n] = np.asarray(r.loss_mask[1:], np.float32)
    meta: dict = {}
    if kind == "pair":
        meta["pairs"] = len(units)
    if any("label" in r.meta for r in rows):
        meta["label"] = mx.array(np.asarray([r.meta.get("label", 1) for r in rows], np.float32))
    if any("adv" in r.meta for r in rows):
        meta["adv"] = mx.array(np.asarray([r.meta["adv"] for r in rows], np.float32))
    if any("ref_seq" in r.meta for r in rows):
        meta["ref_seq"] = mx.array(np.asarray([r.meta["ref_seq"] for r in rows], np.float32))
    if extra:
        meta.update(extra)
    embeds = None
    at = 0
    if any(r.embeds is not None for r in rows):
        shapes = {None if r.embeds is None else tuple(np.shape(r.embeds)) for r in rows}
        ats = {r.embeds_at for r in rows}
        if None in shapes or len(shapes) != 1 or len(ats) != 1:
            raise ValueError(
                "rows in one micro-batch must share embeds shape and embeds_at "
                "(pad them in the formatter)"
            )
        embeds = mx.array(np.stack([np.asarray(r.embeds, np.float32) for r in rows]))
        at = ats.pop()
    return Batch(
        tokens=mx.array(tok),
        targets=mx.array(tgt),
        mask=mx.array(msk),
        embeds=embeds,
        embeds_at=at,
        meta=meta,
        rows=[r.row for r in rows],
    )


def lengths_summary(units: list) -> dict:
    ls = np.asarray([len(u) for u in units]) if units else np.zeros(1)
    return {
        "n": int(len(units)),
        "mean": float(ls.mean()),
        "p50": float(np.percentile(ls, 50)),
        "p95": float(np.percentile(ls, 95)),
        "max": int(ls.max()),
        "tokens": int(ls.sum()),
    }


FormatterFn = Callable[[dict, Any], Any]
