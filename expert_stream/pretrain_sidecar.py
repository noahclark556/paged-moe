# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline / fast warm-start for the wrap expert-prefetch head.

Collects real router demand trajectories (short decode is enough), expands
them into many sliding-window samples that mix token positions, dense-SGD's
wrap, and saves unlocked weights under the configured SIDECAR_DIR
(default ~/.paged_moe/sidecar; override with EXPERT_STREAM_SIDECAR_DIR).

  # Fast path - small corpus, short decode, automatic token mix
  python -m expert_stream.pretrain_sidecar \\
      --model ~/mlx-models/glm-47-4bit --corpus ~/sidecar_corpus --quick

  # Fit only - no MoE generate; uses the bank filled by live use or a prior run
  python -m expert_stream.pretrain_sidecar --fit-only --epochs 12

  python -m expert_stream.pretrain_sidecar --unlock-all

Corpus:
  * directory of .txt / .md / .jsonl files
  * single .jsonl with {"prompt": "..."} or {"text": "..."} or
    {"messages": [{"role","content"}, ...]}
  * plain .txt / .md (one prompt, or blocks split on a line of ---)

Prompt trim (default --max-prompt-tokens 1024):
  For chat rows, keep system + first user task; drop tool results and
  tool-call assistant turns so prefill isn't a wall of identical HTML dumps.
  Plain text keeps the head (task), clipped to the budget.

Does not invent expert labels - only the live router produces train signal.
Live agent decode also appends to the wrap bank, so --fit-only can warm wrap
after normal use with no curated corpus at all.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MAX_PROMPT_TOKENS = 1024
# Soft caps inside the budget so system never eats the user task.
_SYSTEM_CAP = 256
_ASSIST_CAP = 128


@dataclass
class CorpusItem:
    """One pretrain example - structured chat preferred over flat text."""

    messages: list[dict] | None = None
    text: str | None = None


def _content_str(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type", "text") == "text":
                parts.append(str(c.get("text", "")))
            elif isinstance(c, str):
                parts.append(c)
        return " ".join(parts)
    return str(content)


def _n_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return max(1, (len(text) + 3) // 4)


def _clip_tokens(tokenizer, text: str, max_tok: int, *, from_tail: bool = False) -> str:
    if max_tok <= 0 or not text:
        return ""
    try:
        ids = tokenizer.encode(text)
    except Exception:
        # ~4 chars/token fallback
        n = max_tok * 4
        return text[-n:] if from_tail else text[:n]
    if len(ids) <= max_tok:
        return text
    keep = ids[-max_tok:] if from_tail else ids[:max_tok]
    try:
        return tokenizer.decode(keep)
    except Exception:
        n = max_tok * 4
        return text[-n:] if from_tail else text[:n]


def trim_messages(messages: list[dict], tokenizer, max_tokens: int) -> list[dict]:
    """Keep task signal; drop tool spam.

    Always prefers: first system (clipped) + first user task (bulk of budget).
    Optionally a short trailing plain-text assistant turn if budget remains.
    Never includes role=tool or assistant turns that are only tool_calls.
    """
    if max_tokens <= 0:
        max_tokens = DEFAULT_MAX_PROMPT_TOKENS

    systems: list[dict] = []
    users: list[dict] = []
    plain_assistants: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).lower()
        content = _content_str(m.get("content")).strip()
        tool_calls = m.get("tool_calls")
        if role == "tool":
            continue
        if role == "assistant" and (tool_calls or not content):
            # tool-call-only or empty - skip (avoids identical call spam)
            continue
        if role == "system" and content:
            systems.append({"role": "system", "content": content})
        elif role == "user" and content:
            users.append({"role": "user", "content": content})
        elif role == "assistant" and content:
            plain_assistants.append({"role": "assistant", "content": content})

    out: list[dict] = []
    budget = max_tokens

    # Reserve the majority of the budget for the user task - system is
    # boilerplate and must not crowd it out on tight caps.
    user_reserve = max(32, (budget * 2) // 3)

    if systems:
        sys_budget = min(_SYSTEM_CAP, max(32, budget - user_reserve))
        clipped = _clip_tokens(tokenizer, systems[0]["content"], sys_budget)
        if clipped.strip():
            out.append({"role": "system", "content": clipped})
            budget -= _n_tokens(tokenizer, clipped)

    if users and budget > 16:
        clipped = _clip_tokens(tokenizer, users[0]["content"], budget - 8)
        if clipped.strip():
            out.append({"role": "user", "content": clipped})
            budget -= _n_tokens(tokenizer, clipped)
    elif not users and plain_assistants and budget > 16:
        # Degenerate row - use last assistant text as a user-ish seed.
        clipped = _clip_tokens(
            tokenizer, plain_assistants[-1]["content"], budget - 8
        )
        if clipped.strip():
            out.append({"role": "user", "content": clipped})
            budget -= _n_tokens(tokenizer, clipped)

    # Optional short trailing assistant (plain text only) if room - diversity
    # without pulling in tool dumps. Prefer the last one.
    if plain_assistants and budget > 48:
        cap = min(_ASSIST_CAP, budget - 8)
        clipped = _clip_tokens(
            tokenizer, plain_assistants[-1]["content"], cap, from_tail=True
        )
        if clipped.strip() and _n_tokens(tokenizer, clipped) >= 8:
            out.append({"role": "assistant", "content": clipped})

    return out


def trim_text(text: str, tokenizer, max_tokens: int) -> str:
    """Clip flat prompts from the head (task statement lives there)."""
    if max_tokens <= 0:
        return text
    return _clip_tokens(tokenizer, text, max_tokens, from_tail=False)


def build_prompt(tokenizer, item: CorpusItem, max_prompt_tokens: int) -> str:
    if item.messages:
        msgs = trim_messages(item.messages, tokenizer, max_prompt_tokens)
        if not msgs:
            return ""
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)

    text = (item.text or "").strip()
    if not text:
        return ""
    text = trim_text(text, tokenizer, max_prompt_tokens)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return text


def load_corpus(corpus: Path) -> list[CorpusItem]:
    items: list[CorpusItem] = []
    if corpus.is_file():
        items.extend(_items_from_file(corpus))
    elif corpus.is_dir():
        for p in sorted(corpus.rglob("*")):
            if p.suffix.lower() in {".txt", ".md", ".jsonl", ".json"}:
                items.extend(_items_from_file(p))
    else:
        raise SystemExit(f"corpus not found: {corpus}")
    out = [it for it in items if _item_usable(it)]
    if not out:
        raise SystemExit(f"no usable prompts under {corpus}")
    return out


def _item_usable(item: CorpusItem) -> bool:
    if item.messages:
        # Need at least one user/system with real text after we'd drop tools.
        for m in item.messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "")).lower()
            if role in ("user", "system") and len(_content_str(m.get("content")).strip()) >= 20:
                return True
        return False
    return bool(item.text and len(item.text.strip()) >= 40)


def _items_from_file(path: Path) -> list[CorpusItem]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in {".jsonl", ".json"}:
        return _items_from_jsonish(text, path)
    blocks: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        if line.strip() == "---" and cur:
            blocks.append("\n".join(cur).strip())
            cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur).strip())
    return [CorpusItem(text=b) for b in blocks if b]


def _items_from_jsonish(text: str, path: Path) -> list[CorpusItem]:
    out: list[CorpusItem] = []
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            raise SystemExit(f"bad json {path}: {e}") from e
        rows = payload if isinstance(payload, list) else [payload]
        for item in rows:
            out.extend(_items_from_obj(item))
        return out
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise SystemExit(f"bad jsonl {path}:{i}: {e}") from e
        out.extend(_items_from_obj(obj))
    return out


def _items_from_obj(obj) -> list[CorpusItem]:
    if isinstance(obj, str):
        return [CorpusItem(text=obj)]
    if not isinstance(obj, dict):
        return []
    if isinstance(obj.get("prompt"), str):
        return [CorpusItem(text=obj["prompt"])]
    if isinstance(obj.get("text"), str):
        return [CorpusItem(text=obj["text"])]
    msgs = obj.get("messages")
    if isinstance(msgs, list) and msgs:
        # Preserve structure for smart trim (don't flatten tools in).
        return [CorpusItem(messages=list(msgs))]
    return []


# Back-compat helpers used by older tests / callers.
def _load_prompts(corpus: Path) -> list[str]:
    items = load_corpus(corpus)
    out = []
    for it in items:
        if it.text:
            out.append(it.text.strip())
        elif it.messages:
            parts = []
            for m in it.messages:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", "user")
                content = _content_str(m.get("content"))
                if content:
                    parts.append(f"{role}: {content}")
            if parts:
                out.append("\n".join(parts))
    return out


def _format_prompt(tokenizer, text: str) -> str:
    return build_prompt(tokenizer, CorpusItem(text=text), DEFAULT_MAX_PROMPT_TOKENS)


def _progress_path(slot_id: str) -> Path:
    from expert_stream import config

    return Path(config.SIDECAR_DIR) / f"{slot_id}.pretrain.json"


def _load_progress(
    slot_id: str, corpus: Path, *, max_prompt_tokens: int, max_prompts: int
) -> int:
    """Return 0-based prompt index to resume at, or 0 if none/mismatch."""
    path = _progress_path(slot_id)
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text())
    except Exception:
        return 0
    if data.get("corpus") != str(corpus.resolve()):
        return 0
    if int(data.get("max_prompt_tokens", -1)) != int(max_prompt_tokens):
        return 0
    if int(data.get("max_prompts", 0)) != int(max_prompts):
        return 0
    return max(0, int(data.get("next_index", 0)))


def _save_progress(
    slot_id: str,
    corpus: Path,
    next_index: int,
    *,
    max_prompt_tokens: int,
    max_prompts: int,
    total: int,
) -> None:
    path = _progress_path(slot_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus": str(corpus.resolve()),
        "next_index": int(next_index),
        "total": int(total),
        "max_prompt_tokens": int(max_prompt_tokens),
        "max_prompts": int(max_prompts),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _clear_progress(slot_id: str) -> None:
    path = _progress_path(slot_id)
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Offline warm-start for the expert-prefetch sidecar"
    )
    ap.add_argument("--model", default="", help="checkpoint path or HF id")
    ap.add_argument(
        "--corpus",
        default="",
        help="prompt file or directory (.txt/.md/.jsonl)",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=-1,
        help="decode tokens per prompt (default 64; --quick uses 24)",
    )
    ap.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=DEFAULT_MAX_PROMPT_TOKENS,
        help=(
            "trim input to this many tokens (default 1024). Chat rows keep "
            "system+first user and drop tool spam; plain text keeps the head."
        ),
    )
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument(
        "--chunk-prompts",
        type=int,
        default=0,
        help=(
            "process at most N prompts from the resume point, then stop and "
            "keep progress so the next run continues further in the corpus "
            "(full corpus; does not slice like --max-prompts)"
        ),
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=-1,
        help="dense SGD epochs (default 4; --quick uses 12)",
    )
    ap.add_argument("--cache-gb", type=float, default=None)
    ap.add_argument(
        "--quick",
        action="store_true",
        help=(
            "fast wrap warm-start: short decode, few prompts, many epochs, "
            "sliding-window mix of token positions (no massive corpus)"
        ),
    )
    ap.add_argument(
        "--fit-only",
        action="store_true",
        help=(
            "skip MoE generate; dense-fit wrap from the on-disk sample bank "
            "(filled by prior pretrain or live agent use). No corpus needed."
        ),
    )
    ap.add_argument(
        "--bank",
        default="",
        help="wrap bank .npz path for --fit-only (default: auto under SIDECAR_DIR)",
    )
    ap.add_argument(
        "--fresh",
        "--force",
        action="store_true",
        help="wipe slot weights + corpus progress; start from prompt 0",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="keep weights but restart corpus from prompt 0 (ignore progress file)",
    )
    ap.add_argument(
        "--start-at",
        type=int,
        default=-1,
        help="1-based prompt index to start at (overrides progress file)",
    )
    ap.add_argument("--unlock-only", action="store_true")
    ap.add_argument(
        "--unlock-all",
        action="store_true",
        help="clear lock bits on every slot under SIDECAR_DIR (no model load)",
    )
    args = ap.parse_args(argv)

    if args.quick:
        if args.max_tokens < 0:
            args.max_tokens = 24
        if args.epochs < 0:
            args.epochs = 12
        if args.max_prompts <= 0:
            args.max_prompts = 8
        if args.max_prompt_tokens == DEFAULT_MAX_PROMPT_TOKENS:
            args.max_prompt_tokens = 512
    else:
        if args.max_tokens < 0:
            args.max_tokens = 64
        if args.epochs < 0:
            args.epochs = 4

    if args.unlock_all:
        from expert_stream.sidecar import unlock_slots

        done = unlock_slots()
        print(json.dumps({"unlocked": done}), flush=True)
        return 0

    # ---- fit-only: no MoE load, densify from bank --------------------------
    if args.fit_only:
        from expert_stream import config
        from expert_stream.sidecar import ExpertSidecar
        from expert_stream.sidecar.bank import load_bank
        from expert_stream.sidecar.features import model_slot_id

        config.SIDECAR = True
        bank = Path(args.bank).expanduser() if args.bank else None
        if bank is None or not str(bank):
            root = Path(config.SIDECAR_DIR)
            banks = sorted(
                root.glob("*_wrap_bank.npz"), key=lambda p: p.stat().st_mtime
            )
            if not banks:
                raise SystemExit(
                    f"no wrap bank under {root} - run a short pretrain or "
                    f"use the model normally first so live decode fills the bank"
                )
            chosen = banks[-1]
            if args.model:
                # Match bank meta.slot to this checkpoint's slot id
                # (content-based, or legacy path-based from older runs).
                model_path = str(Path(args.model).expanduser())
                from expert_stream.sidecar.features import legacy_path_slot_id

                matched = []
                for cand in banks:
                    _traj, meta = load_bank(cand)
                    n_layers = int(meta.get("n_layers") or 0)
                    n_experts = int(meta.get("n_experts") or 0)
                    slot = str(meta.get("slot") or "")
                    if n_layers <= 0 or n_experts <= 0 or not slot:
                        continue
                    want = {
                        model_slot_id(model_path, n_layers, n_experts),
                        legacy_path_slot_id(model_path, n_layers, n_experts),
                    }
                    if slot in want:
                        matched.append(cand)
                if matched:
                    chosen = matched[-1]
                else:
                    print(
                        f"[sidecar-pretrain] warning: no bank matched "
                        f"{model_path}; using newest {chosen.name}",
                        flush=True,
                    )
            bank = chosen
        print(f"[sidecar-pretrain] fit-only bank={bank}", flush=True)
        sidecar = ExpertSidecar.from_bank(bank)
        report = sidecar.fit_pretrain(epochs=args.epochs, from_bank=True)
        print(json.dumps(report, indent=2), flush=True)
        sidecar.close()
        return 0

    if not args.model:
        raise SystemExit("--model is required (unless --unlock-all / --fit-only)")
    if not args.unlock_only and not args.corpus:
        raise SystemExit(
            "--corpus is required (unless --unlock-only / --unlock-all / --fit-only)"
        )

    os.environ["EXPERT_STREAM_SIDECAR"] = "1"
    os.environ["EXPERT_STREAM_SIDECAR_LOCK_HITS"] = "0"

    from expert_stream import config, get_stats, load, relieve_pressure
    from expert_stream.sidecar import ExpertSidecar, ModelSlot

    config.SIDECAR = True
    config.SIDECAR_LOCK_HITS = 0

    model_path = str(Path(args.model).expanduser())
    print(f"[sidecar-pretrain] loading {model_path}", flush=True)
    model, tokenizer = load(model_path, mode="streamed", cache_gb=args.cache_gb)
    ring = getattr(model, "_expert_stream_ring", None)
    sidecar = getattr(ring, "sidecar", None) if ring is not None else None
    if sidecar is None:
        raise SystemExit(
            "sidecar not attached - set EXPERT_STREAM_SIDECAR=1 and use a MoE model"
        )
    assert isinstance(sidecar, ExpertSidecar)

    if args.fresh and sidecar.slot.path and sidecar.slot.path.exists():
        sidecar.slot.path.unlink()
        print(f"[sidecar-pretrain] wiped {sidecar.slot.path}", flush=True)
        sidecar.slot = ModelSlot.create(
            sidecar.slot_id,
            sidecar.n_layers,
            sidecar.n_experts,
            sidecar.wrap_n,
            sidecar.feat_dim,
            path=sidecar.slot.path,
        )
        _clear_progress(sidecar.slot_id)
        print("[sidecar-pretrain] cleared corpus progress (--fresh/--force)", flush=True)

    if args.unlock_only:
        sidecar.unlock()
        print(json.dumps({"unlocked": True, **sidecar.stats()}), flush=True)
        return 0

    corpus_path = Path(args.corpus).expanduser().resolve()
    items = load_corpus(corpus_path)
    if args.max_prompts > 0:
        items = items[: args.max_prompts]

    if args.start_at > 0:
        start = min(len(items), args.start_at - 1)
        print(f"[sidecar-pretrain] --start-at {args.start_at} -> index {start}", flush=True)
    elif args.fresh or args.no_resume:
        start = 0
        if args.no_resume:
            _clear_progress(sidecar.slot_id)
            print("[sidecar-pretrain] --no-resume: corpus from 0, weights kept", flush=True)
    else:
        start = _load_progress(
            sidecar.slot_id,
            corpus_path,
            max_prompt_tokens=args.max_prompt_tokens,
            max_prompts=args.max_prompts,
        )
        if start > 0:
            if start >= len(items):
                print(
                    f"[sidecar-pretrain] progress says done ({start}/{len(items)}) "
                    f" -  use --fresh/--force or --no-resume to re-run",
                    flush=True,
                )
                return 0
            print(
                f"[sidecar-pretrain] resuming at prompt {start + 1}/{len(items)} "
                f"(progress {_progress_path(sidecar.slot_id).name})",
                flush=True,
            )

    chunk = max(0, int(getattr(args, "chunk_prompts", 0) or 0))
    end = len(items)
    if chunk > 0:
        end = min(len(items), start + chunk)

    print(
        f"[sidecar-pretrain] {len(items)} prompts (from {start + 1}"
        f"{f' to {end}' if chunk > 0 else ''}), "
        f"max_prompt_tokens={args.max_prompt_tokens}, "
        f"decode={args.max_tokens}, epochs={args.epochs}"
        f"{' [quick]' if args.quick else ''}"
        f"{f' [chunk={chunk}]' if chunk > 0 else ''} "
        f"(sliding-window expand + mix across token positions)",
        flush=True,
    )

    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.7, top_p=0.9)
    sidecar.begin_pretrain()

    stop = {"flag": False}

    def _on_signal(signum, _frame):
        if stop["flag"]:
            print("\n[sidecar-pretrain] second interrupt - exiting raw", flush=True)
            raise SystemExit(130)
        stop["flag"] = True
        print(
            f"\n[sidecar-pretrain] interrupt ({signum}) - finishing dense fit "
            f"on {len(sidecar._pretrain_samples)} samples collected so far...",
            flush=True,
        )

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    for idx in range(start, end):
        if stop["flag"]:
            break
        item = items[idx]
        prompt = build_prompt(tokenizer, item, args.max_prompt_tokens)
        if not prompt.strip():
            print(
                f"[sidecar-pretrain] prompt {idx + 1} empty after trim - skip",
                flush=True,
            )
            _save_progress(
                sidecar.slot_id,
                corpus_path,
                idx + 1,
                max_prompt_tokens=args.max_prompt_tokens,
                max_prompts=args.max_prompts,
                total=len(items),
            )
            continue
        prompt_toks = _n_tokens(tokenizer, prompt)
        n = 0
        try:
            for _ in stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
            ):
                n += 1
                if stop["flag"]:
                    break
        except Exception as e:
            print(f"[sidecar-pretrain] prompt {idx + 1} failed: {e!r}", flush=True)
        relieve_pressure(model)
        sidecar.end_pretrain_prompt()
        print(
            f"[sidecar-pretrain] prompt {idx + 1}/{len(items)} "
            f"prompt_toks={prompt_toks} decoded={n} "
            f"samples={len(sidecar._pretrain_samples)} "
            f"traj={len(sidecar._pretrain_trajectories)} "
            f"ceiling={sidecar.slot.mean_ceiling():.2f}",
            flush=True,
        )
        # If interrupted mid-prompt, resume on this index; else advance.
        next_i = idx if stop["flag"] else idx + 1
        _save_progress(
            sidecar.slot_id,
            corpus_path,
            next_i,
            max_prompt_tokens=args.max_prompt_tokens,
            max_prompts=args.max_prompts,
            total=len(items),
        )

    if len(sidecar._pretrain_samples) < 8:
        print(
            "[sidecar-pretrain] too few samples to fit - persisting online "
            "weights only (progress saved; re-run to continue)",
            flush=True,
        )
        sidecar.close()
        return 1 if stop["flag"] else 0

    report = sidecar.fit_pretrain(epochs=args.epochs)
    report["stats"] = get_stats(model).get("sidecar", {})
    report["interrupted"] = bool(stop["flag"])
    # Only clear progress when the full corpus (or --max-prompts slice) is
    # finished. Chunk stops keep progress so the next visit continues further.
    final_next = _load_progress(
        sidecar.slot_id,
        corpus_path,
        max_prompt_tokens=args.max_prompt_tokens,
        max_prompts=args.max_prompts,
    )
    corpus_complete = (not stop["flag"]) and final_next >= len(items)
    if corpus_complete:
        _clear_progress(sidecar.slot_id)
        report["corpus_complete"] = True
    else:
        report["corpus_complete"] = False
        report["resume_at"] = final_next + 1
        report["chunk_end"] = end
        if chunk > 0 and not stop["flag"]:
            print(
                f"[sidecar-pretrain] chunk done - next run resumes at "
                f"prompt {final_next + 1}/{len(items)}",
                flush=True,
            )
    print(json.dumps(report, indent=2), flush=True)
    sidecar.close()
    return 130 if stop["flag"] else 0


if __name__ == "__main__":
    sys.exit(main())
