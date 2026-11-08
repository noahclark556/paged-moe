# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Catch an approximate decode that loses the thread, before it is shown.

At the fast end of the speed setting (PRUNE ~0.9, WAIT_ABOVE ~0.45) a long
turn can lose the thread. GLM-5.2 wrote "Let me execute create_file calls in
2 max 1-2-3-4-5-6-7." four times over in its reasoning, and in another turn
"It looks as a bin/www file (0, but0,2,3,4,5,..." up to 146. The fidelity
guard only saw one-token runs, and only once they were on screen.

Watchers read token ids, no GPU work:
  copy     a RESCUE_GRAM run seen again within RESCUE_MAX_PERIOD tokens
  shapes   the same in prose with every bare number one symbol (counting)
  numbers  RESCUE_NUM_MIN bare numbers in RESCUE_NUM_WINDOW prose tokens,
           which is how both collapses above started
Only runs at speed slider RESCUE_UP_TO and under, from the model's kept-mass
table (prune_curve.position); before it has one, from PRUNE >=
RESCUE_MIN_PRUNE or WAIT_ABOVE >= RESCUE_MIN_WAIT.
Tokens go out RESCUE_LAG behind the decode (tool calls, think markers and EOS flush at once), and a copy
holds them until it breaks, so the start of a collapse is usually still
unshown when a watcher sees it.

Numbers ask the exact model (RESCUE_AUDIT). One pass with every routed
expert scores the unshown tokens from the first of those numbers, and decode
is cut at the first one the exact model gives less than RESCUE_AUDIT_P and
resumes exact from there. A clean audit lets the tokens go. A copy is not
audited (the exact model carries on a copy it is shown): it is dropped whole
and decoded again exact from the last shown token. In prose the first redone
token is kept off the one the loop took (RESCUE_NUDGE), and if the exact
model writes the same tokens again the pattern was real and is not held
again this turn.

So what is shown is what the approximate decode wrote, or what the exact
model writes from a point it picked. Only approximate configs run this; at an
exact config the decode loop is the plain one. An error in a watcher turns
the rescue off for the rest of the generation.
"""

from __future__ import annotations

import math
import re
import time
from collections import deque

import mlx.core as mx
from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

from . import config, fidelity, prune_curve, streaming

CLEAR, HOLD, LOOP = 0, 1, 2
# Shape id of a bare number in prose (token ids are never negative).
NUM = -1

_tokenizer = None
# Last generation: redos (loops and cuts), tokens dropped unseen, resumes
# nudged, patterns the exact redo wrote again, loops caught only after
# showing, exact audits, audits that cut.
STATS = {"trips": 0, "dropped": 0, "nudged": 0, "real": 0, "late": 0, "audits": 0, "cuts": 0}
# Tokens between recurrent-state marks for a hybrid cache rewind.
_MARK_EVERY = 16
_LETTERS = re.compile(r"[^\W\d_]")


def configure(tokenizer) -> None:
    global _tokenizer
    _tokenizer = tokenizer


def stats() -> dict:
    return dict(STATS)


def active(model) -> bool:
    if not config.RESCUE:
        return False
    info = getattr(model, "_pagedmoe_info", None) or {}
    if info.get("mode") != "streamed" or not fidelity._approximate():
        return False
    prune, wait = float(config.PRUNE), float(config.WAIT_ABOVE)
    try:
        pos = prune_curve.position(info.get("model_path"), prune, wait)
    except Exception:
        pos = None
    if pos is not None:
        return pos <= float(config.RESCUE_UP_TO) + 0.005
    return prune >= float(config.RESCUE_MIN_PRUNE) or wait >= float(config.RESCUE_MIN_WAIT)


class LoopWatch:
    """Tracks whether the newest tokens copy the ones just before them."""

    def __init__(self, gram: int = 4, max_period: int = 48, confirm: int = 12):
        self.gram = max(2, int(gram))
        self.max_period = max(1, int(max_period))
        self.confirm = max(self.gram + 1, int(confirm))
        self.allowed: set = set()
        self.reset()

    def reset(self, tokens=()) -> None:
        self.t: list[int] = []
        self.last: dict = {}
        self.forget()
        for tok in tokens:
            self.push(int(tok))
        self.forget()

    def forget(self) -> None:
        self.src = -1
        self.run = 0
        self.state = CLEAR

    @property
    def period(self) -> int:
        return len(self.t) - self.src if self.src >= 0 else 0

    def signature(self):
        p = self.period
        return (p, frozenset(self.t[-p:])) if p else None

    def push(self, tok: int) -> int:
        t = self.t
        t.append(tok)
        n = len(t)
        if self.src >= 0:
            if t[self.src] == tok:
                self.src += 1
                self.run += 1
            else:
                self.src = -1
                self.run = 0
        if n >= self.gram:
            g = tuple(t[n - self.gram :])
            if self.src < 0:
                e = self.last.get(g)
                if e is not None and n - e <= self.max_period:
                    self.src, self.run = e, self.gram
                    if self.signature() in self.allowed:
                        self.src, self.run = -1, 0
            self.last[g] = n
        if self.src < 0:
            self.state = CLEAR
        elif self.run >= max(self.confirm, self.period):
            self.state = LOOP
        else:
            self.state = HOLD
        return self.state


class Numbers:
    """Bare numbers crowding prose."""

    def __init__(self, window: int = 12, need: int = 4):
        self.window = max(1, int(window))
        self.need = max(1, int(need))
        self.cool = 0
        self.quiet = 0
        self.clear()

    def clear(self) -> None:
        self.win: deque = deque()
        self.count = 0

    def push(self, hit: bool) -> bool:
        self.win.append(hit)
        self.count += hit
        if len(self.win) > self.window:
            self.count -= self.win.popleft()
        if self.quiet > 0:
            self.quiet -= 1
            return False
        return self.count >= self.need

    def passed(self) -> None:
        """The exact model was fine with them: stay quiet longer each time."""
        self.cool = max(max(1, int(config.RESCUE_AUDIT_COOL)), 2 * self.cool)
        self.quiet = self.cool
        self.clear()


class _Text:
    """Per-token facts (decoded once per id) and where the decode is: code
    fence, inline code, tool call."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.memo: dict = {}
        self.markers = {k: v for k, v in fidelity._markers.items() if v}
        self.keep = max([len(v) for v in self.markers.values()] + [1])
        eos = getattr(tokenizer, "eos_token_ids", None)
        if eos is None:
            eos = getattr(tokenizer, "eos_token_id", None)
        self.eos = {int(eos)} if isinstance(eos, int) else {int(t) for t in (eos or ())}
        self.reset()

    def reset(self) -> None:
        self.fence = 0
        self.tick = 0
        self.tool = False
        self.recent: list[int] = []

    def _facts(self, tok: int):
        f = self.memo.get(tok)
        if f is None:
            s = self.tok.decode([tok])
            fences = s.count("```")
            f = (s.strip().isdigit(), fences, s.count("`") - 3 * fences, "\n" in s)
            self.memo[tok] = f
        return f

    def push(self, tok: int) -> tuple[bool, bool, bool]:
        """(bare number, in prose, show everything now)."""
        num, fences, ticks, newline = self._facts(tok)
        self.fence += fences
        if newline:
            self.tick = 0
        self.tick += ticks
        flush = tok in self.eos
        if self.markers:
            r = self.recent
            r.append(tok)
            del r[: -self.keep]
            for name, seq in self.markers.items():
                if len(r) >= len(seq) and tuple(r[-len(seq) :]) == seq:
                    # fidelity keys strict spans off what it is shown.
                    flush = True
                    if name == "tool_start":
                        self.tool = True
                    elif name == "tool_end":
                        self.tool = False
        prose = not self.tool and self.fence % 2 == 0 and self.tick % 2 == 0
        return num, prose, flush or self.tool


class _Ban:
    """Keeps the first step of a resumed decode off one token."""

    def __init__(self, tok: int):
        self.tok = int(tok)

    def __call__(self, tokens, logits):
        if tokens.size != 1:
            return logits
        hit = mx.arange(logits.shape[-1]) == self.tok
        return mx.where(hit, mx.array(-float("inf"), logits.dtype), logits)


def _procs(base, prefix: list[int], ban):
    """Logits processors for a resumed decode: they see the tokens before it."""
    procs = list(base or [])
    if procs and prefix:
        pre = mx.array(prefix[-2048:], dtype=mx.uint32)
        procs = [
            (lambda toks, logits, f=f: f(mx.concatenate([pre, toks.astype(mx.uint32)]), logits))
            for f in procs
        ]
    if ban is not None:
        procs.append(_Ban(ban))
    return procs or base


def _offset(cache):
    for c in cache:
        off = getattr(c, "offset", None)
        if isinstance(off, int):
            return off
    return None


class _Rewind:
    """Takes tokens back off the cache: a trim, or a mark plus a replay on a
    cache with recurrent state (same snapshot rules as verify decode)."""

    def __init__(self, model, cache, stream):
        from .lookup_decode import _needs_snapshot, _side_caches, _snapshot

        self.model, self.cache, self.stream = model, cache, stream
        self._snapshot = _snapshot
        self.plain = can_trim_prompt_cache(cache) and not any(_side_caches(c) for c in cache)
        self.start = _offset(cache)
        snap = _snapshot(cache) if not self.plain and _needs_snapshot(cache) else None
        self.hybrid = self.start is not None and snap is not None
        self.marks: list = [(self.start, snap)] if self.hybrid else []

    def mark(self, keep_from: int) -> None:
        """Take a mark every _MARK_EVERY tokens; keep what can still reach
        `keep_from` tokens past the start (the last shown token)."""
        if not self.hybrid:
            return
        off = _offset(self.cache)
        if off is not None and off - self.marks[-1][0] >= _MARK_EVERY:
            snap = self._snapshot(self.cache)
            if snap is not None:
                self.marks.append((off, snap))
        reach = [i for i, m in enumerate(self.marks) if m[0] <= self.start + keep_from]
        if reach:
            del self.marks[: reach[-1]]

    def reaches(self, fed: int) -> bool:
        """Whether the cache can go back to `fed` tokens past the start."""
        if self.plain:
            return True
        return self.hybrid and any(off <= self.start + fed for off, _ in self.marks)

    def back(self, n: int, seq: list[int]) -> None:
        """Drop the last `n` tokens fed; `seq` is every token fed since start."""
        if n <= 0:
            return
        if self.plain:
            got = trim_prompt_cache(self.cache, n)
            if got != n:
                raise RuntimeError(f"rescue could not rewind: asked {n}, trimmed {got}")
            return
        target = self.start + len(seq) - n
        off, snap = max((m for m in self.marks if m[0] <= target), key=lambda m: m[0])
        end = _offset(self.cache)
        for c, (kind, s) in zip(self.cache, snap):
            if kind == "arrays":
                c.cache = list(s)
                continue
            got = c.trim(end - off)
            if got != end - off:
                raise RuntimeError(f"rescue could not rewind: asked {end - off}, trimmed {got}")
            for side, state in s:
                side.state = state
        self.marks = [m for m in self.marks if m[0] <= off]
        self.feed(seq[off - self.start : target - self.start])

    def feed(self, tokens: list[int]) -> None:
        """Run tokens into the cache exact, a decode-sized pass at a time."""
        step = max(1, streaming._DECODE_MAX_TOKENS)
        with mx.stream(self.stream), streaming.exact_step():
            for i in range(0, len(tokens), step):
                chunk = mx.array([tokens[i : i + step]], dtype=mx.uint32)
                mx.eval(self.model(chunk, cache=self.cache))

    def score(self, tokens: list[int], want: list[int]) -> list[float]:
        """Feed `tokens` exact; exact log-probability of each of `want`."""
        with mx.stream(self.stream), streaming.exact_step():
            logits = self.model(mx.array([tokens], dtype=mx.uint32), cache=self.cache)[0]
            logits = logits.astype(mx.float32)
            ids = mx.array(want, dtype=mx.uint32)[:, None]
            lp = mx.take_along_axis(logits, ids, axis=-1)[:, 0] - mx.logsumexp(logits, axis=-1)
            return lp.tolist()


class _Guard:
    def __init__(self, start, prompt, model, cache, max_tokens, procs, stream):
        self.start = start
        self.procs = procs
        self.lead = [int(t) for t in prompt.reshape(-1).tolist()]
        self.limit = -1 if max_tokens is None or max_tokens < 0 else int(max_tokens)
        self.rw = _Rewind(model, cache, stream)
        shape = (config.RESCUE_GRAM, config.RESCUE_MAX_PERIOD, config.RESCUE_CONFIRM)
        self.watch = LoopWatch(*shape)
        try:
            self.text = _Text(_tokenizer) if _tokenizer is not None else None
        except Exception as e:
            print(f"[rescue] text watch off: {e!r}", flush=True)
            self.text = None
        self.shapes = LoopWatch(*shape) if self.text is not None and config.RESCUE_SHAPES else None
        self.loops = [w for w in (self.watch, self.shapes) if w is not None]
        self.nums = (
            Numbers(config.RESCUE_NUM_WINDOW, config.RESCUE_NUM_MIN)
            if self.text is not None
            and config.RESCUE_AUDIT
            and self.rw.plain
            and int(config.RESCUE_NUM_MIN) > 0
            else None
        )
        # Positions of the numbers in the watch's window.
        self.hits: deque = deque()
        # Without the tokenizer there is no EOS or marker to flush on.
        self.lag = max(0, int(config.RESCUE_LAG)) if self.text is not None else 0
        self.flush = False
        self.fired = self.watch
        self.audit_n = 0
        self.audit_s = 0.0
        self.toks: list[int] = []
        # (token, logprobs) not yet shown. Shown logprobs are not kept: each is
        # a vocabulary-sized array.
        self.pend: list = []
        self.shown = 0
        self.trips = 0
        self.live = True
        # (dropped tokens, watch, loop signature, index of the first redone token)
        self.shadow = None
        self.inner = start(prompt, max_tokens, [], procs)

    def _rewindable(self) -> bool:
        # The resume feeds the last shown token again.
        return self.rw.reaches(len(self.lead) + self.shown - 1)

    def _redo_matches(self) -> bool:
        """Whether the exact redo is still writing the tokens it replaced.
        A redo that writes all of them again marks the pattern as real."""
        gone, watch, sig, base = self.shadow
        i = len(self.toks) - 1 - base
        if i >= len(gone) or gone[i] != self.toks[-1]:
            self.shadow = None
            return False
        if i == len(gone) - 1:
            self._allow(watch, sig)
        return True

    def _allow(self, watch, sig) -> None:
        if sig is not None:
            watch.allowed.add(sig)
        # The other watch may be on the same pattern.
        for w in self.loops:
            if w.src >= 0:
                w.allowed.add(w.signature())
        if self.shadow is not None:
            STATS["real"] += 1
            self.shadow = None

    def _top(self):
        return max(self.loops, key=lambda w: w.state)

    def step(self, tok: int) -> str:
        """'hold', 'show', 'audit' or 'resume' for the newest token."""
        hit = False
        if self.text is not None:
            num, prose, self.flush = self.text.push(tok)
            hit = num and prose
        self.watch.push(tok)
        if self.shapes is not None:
            self.shapes.push(NUM if hit else tok)
        alarm = False
        if self.nums is not None:
            alarm = self.nums.push(hit)
            if hit:
                self.hits.append(len(self.toks) - 1)
            while self.hits and self.hits[0] < len(self.toks) - self.nums.window:
                self.hits.popleft()
        fired = self._top()
        if self.shadow is not None and self._redo_matches() and fired.state == LOOP:
            # The exact model loops here too: its own output, not a collapse.
            self._allow(fired, fired.signature())
        for w in self.loops:
            if w.state != CLEAR and w.signature() in w.allowed:
                w.forget()
        fired = self.fired = self._top()
        state = fired.state
        held = len(self.toks) - 1 > self.shown
        can_trip = self.trips < max(0, int(config.RESCUE_MAX_TRIPS))
        if state != LOOP:
            if alarm and can_trip and self._rewindable():
                return "audit"
            if state == HOLD and (held or self._rewindable()):
                return "hold"
            return "show"
        if not can_trip:
            self.live = False
            return "show"
        if not self._rewindable():
            self.trips += 1
            n, sticky = _span(self.trips)
            fidelity.force_exact(n, sticky)
            STATS["trips"] += 1
            STATS["late"] += 1
            print(f"[rescue] loop already shown, exact for the next {n} tokens", flush=True)
            return "show"
        return "resume"

    def _audit(self, skip: int = 0):
        """One exact pass over unshown tokens, from the `skip`th on.

        Returns the index (from the first unshown token) of the first one the
        exact model would not write, or None; its log-probability; and how
        many tokens of lead + toks the cache holds afterwards.
        """
        seq = self.lead + self.toks
        start = len(self.lead) + self.shown + skip
        n = min(len(seq) - start, max(1, int(config.RESCUE_AUDIT_MAX)), streaming._DECODE_MAX_TOKENS)
        self.inner.close()
        self.rw.back(len(seq) - start + 1, seq)
        t0 = time.perf_counter()
        lp = self.rw.score(seq[start - 1 : start - 1 + n], seq[start : start + n])
        self.audit_s = time.perf_counter() - t0
        self.audit_n = n
        STATS["audits"] += 1
        floor = math.log(max(1e-12, float(config.RESCUE_AUDIT_P)))
        bad = next((i for i, v in enumerate(lp) if v < floor), None)
        if bad is None:
            return None, None, start - 1 + n
        return skip + bad, lp[bad], start - 1 + n

    def _goto(self, target: int, seq: list[int], fed) -> None:
        """Leave the cache holding seq[:target]; it holds seq[:fed] (None = all)."""
        fed = len(seq) if fed is None else fed
        if fed > target:
            self.rw.back(fed - target, seq[:fed])
        elif fed < target:
            self.rw.feed(seq[fed:target])

    def _rebuild(self) -> None:
        shapes = []
        if self.text is not None:
            self.text.reset()
            for t in self.toks:
                num, prose, _ = self.text.push(t)
                shapes.append(NUM if num and prose else t)
        self.watch.reset(self.toks)
        if self.shapes is not None:
            self.shapes.reset(shapes)
        if self.nums is not None:
            self.nums.clear()
        self.hits.clear()

    def _restart(self, keep: list[int], ban=None) -> None:
        """Decode on from `keep` (lead + toks kept); the cache holds keep[:-1]."""
        prefix = keep[:-1]
        left = -1 if self.limit < 0 else max(0, self.limit - len(self.toks))
        self.inner = self.start(
            mx.array(keep[-1:], dtype=mx.uint32), left, prefix, _procs(self.procs, prefix, ban)
        )

    def _cut(self, bad: int, lp: float, fed: int, why: str) -> None:
        """Drop from the token the exact model would not write; exact from there."""
        seq = self.lead + self.toks
        keep = len(self.lead) + self.shown + bad
        self._goto(keep - 1, seq, fed)
        gone = self.toks[self.shown + bad :]
        del self.toks[self.shown + bad :]
        del self.pend[bad:]
        self._rebuild()
        self.trips += 1
        # Short and flat: exact runs ~0.5 s a token at 0.9 / 0.45 on GLM-5.2,
        # and a longer span did not stop the next collapse.
        n = max(1, int(config.RESCUE_AUDIT_TOKENS))
        sticky = self.trips >= max(1, int(config.RESCUE_STICKY))
        # The kept unshown tokens count down the exact span as they go out.
        fidelity.force_exact(n + bad, sticky)
        self.shadow = None
        STATS["trips"] += 1
        STATS["cuts"] += 1
        STATS["dropped"] += len(gone)
        print(
            f"[rescue] {why} at token {self.shown + bad}: exact p={math.exp(lp):.2g} "
            f"(audit {self.audit_n} tok {self.audit_s:.2f}s), dropped {len(gone)}, "
            f"exact for the {'rest of the turn' if sticky else f'next {n} tokens'}",
            flush=True,
        )
        self._restart(seq[:keep])

    def check(self) -> None:
        """Numbers alarm: cut where the exact model disagrees, or go on."""
        # From the token before the first number in the window.
        first = self.hits[0] if self.hits else self.shown
        bad, lp, fed = self._audit(max(0, first - 1 - self.shown))
        self.hits.clear()
        if bad is not None:
            self._cut(bad, lp, fed, "numbers")
            return
        print(
            f"[rescue] numbers at token {self.shown}: exact model agrees "
            f"(audit {self.audit_n} tok {self.audit_s:.2f}s)",
            flush=True,
        )
        self.nums.passed()
        seq = self.lead + self.toks
        self._goto(len(seq) - 1, seq, fed)
        self._restart(seq)

    def resume(self) -> None:
        """A loop: drop the held tokens and resume exact from the last shown one."""
        watch = self.fired
        period = watch.period
        sig = watch.signature()
        fed = None
        toks, shown = self.toks, self.shown
        gone = toks[shown:]
        ban = gone[0] if _nudge_ok(toks[:shown], toks[-period:]) else None
        seq = self.lead + toks
        keep = seq[: len(self.lead) + shown]
        n, sticky = _span(self.trips + 1)

        self.inner.close()
        self._goto(len(keep) - 1, seq, fed)
        self.trips += 1
        del toks[shown:]
        self.pend.clear()
        self._rebuild()
        fidelity.force_exact(n, sticky)
        self.shadow = None if ban is not None else (gone, watch, sig, shown)
        STATS["trips"] += 1
        STATS["dropped"] += len(gone)
        STATS["nudged"] += int(ban is not None)
        print(
            f"[rescue] loop (period {period}) held at token {shown}: dropped {len(gone)}, "
            f"exact for the {'rest of the turn' if sticky else f'next {n} tokens'}"
            + (", first token nudged" if ban is not None else ""),
            flush=True,
        )
        self._restart(keep, ban)

    def _release(self, lag: int = 0):
        while len(self.pend) > lag:
            item = self.pend.pop(0)
            self.shown += 1
            yield item

    def run(self):
        try:
            while True:
                restarted = False
                for tok, lp in self.inner:
                    self.toks.append(int(tok))
                    self.pend.append((tok, lp))
                    what = "show"
                    if self.live:
                        try:
                            what = self.step(int(tok))
                        except Exception as e:
                            print(f"[rescue] off for this generation: {e!r}", flush=True)
                            self.live = False
                    if what == "hold":
                        continue
                    if what == "resume":
                        self.resume()
                        restarted = True
                        break
                    if what == "audit":
                        self.check()
                        restarted = True
                        break
                    lag = self.lag if self.live and not self.flush else 0
                    yield from self._release(lag)
                    if self.live:
                        self.rw.mark(len(self.lead) + self.shown - 1)
                if not restarted:
                    break
            yield from self._release()
        except GeneratorExit:
            # The server keys the stored cache on what it was handed.
            self.inner.close()
            self.rw.back(len(self.toks) - self.shown, self.lead + self.toks)
            raise
        finally:
            self.inner.close()


def _span(trips: int) -> tuple[int, bool]:
    n = max(1, int(config.RESCUE_TOKENS)) << min(max(trips - 1, 0), 2)
    return n, trips >= max(1, int(config.RESCUE_STICKY))


def _nudge_ok(shown_ids: list[int], period_ids: list[int]) -> bool:
    """Prose only: not in a tool call, not in a code fence, the loop has words."""
    if not config.RESCUE_NUDGE or _tokenizer is None or fidelity.in_tool():
        return False
    try:
        if len(_LETTERS.findall(_tokenizer.decode(period_ids))) < 6:
            return False
        return _tokenizer.decode(shown_ids[-4096:]).count("```") % 2 == 0
    except Exception:
        return False


def guarded(start, prompt, model, cache, *, max_tokens, logits_processors, stream):
    """Decode with loop rescue.

    `start(prompt, max_tokens, prefix, procs)` returns a decode generator over
    `cache` (verify decode, or mlx-lm's), where `prefix` is what was fed
    before `prompt` since this call began. On exit the cache holds exactly
    the tokens yielded, like the generator it wraps.
    """
    for k in STATS:
        STATS[k] = 0
    guard = None
    if active(model):
        try:
            guard = _Guard(start, prompt, model, cache, max_tokens, logits_processors, stream)
        except Exception as e:
            print(f"[rescue] off for this generation: {e!r}", flush=True)
    if guard is None:
        yield from start(prompt, max_tokens, [], logits_processors)
        return
    yield from guard.run()
