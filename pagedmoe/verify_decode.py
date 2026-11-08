# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verify decode: drafted tokens are checked by the model in one pass.

Verifying several tokens in one pass is the one lever that gets more tokens
out of the same expert bytes. The draft only decides how many tokens a pass
emits, never which ones (see `lookup_decode._verify` for why equality
acceptance is exact for any sampler).

What the pass computes is config.VERIFY_FIDELITY. By default every row is
routed the way the plain step it replaces would route it (per-row selector,
same PRUNE / WAIT_ABOVE / fidelity span scale, `streaming.matched_step`), so
a pass samples from the same model plain decode does. "exact" runs it under
`streaming.exact_step()` instead: every emitted token comes from the router's
full top-k, at the price of blocking on every expert pruned decode skipped.

Drafters (the MTP head, mtp.py, is the one that needs extra weights):

* n-gram: copy what followed the last few tokens earlier in the context. Tool
  calls and edits copy file text the prompt already holds, which is where
  this hits. Also matches the JSON-escaped text of recent tool results, for
  calls that write their arguments as JSON.
* self: k cheap approximate steps (`streaming.draft_step`) from the model
  itself, then one exact pass over them. Only pays when the plain step is the
  exact one (PRUNE 0), where it replaces k exact steps with k cheap ones and
  one shared-read pass.
* mtp: the checkpoint's own multi-token-prediction head, where it has one.
  Fed the hidden state of every kept position, so it drafts in any span.

None engages unless it has been measured faster, per emitted token, than
the plain step it replaces in the same kind of span (the governor below). A
plain step here is mlx-lm's own pipelined step, op for op, so a generation
that never drafts decodes exactly as it did before this module existed.
"""

from __future__ import annotations

import contextlib
import time

import mlx.core as mx
from mlx_lm.models.cache import trim_prompt_cache

from . import activity, config, fast_replay, fidelity, gpu_crumbs, gpu_trail, streaming, turn_trace
from .lookup_decode import _needs_snapshot, _replay, _snapshot, can_rewind, unfeed

STATS = {
    "passes": 0,  # exact verify passes
    "drafted": 0,
    "accepted": 0,
    "emitted": 0,  # tokens out of verify passes
    "plain": 0,  # plain pipelined steps
    "self": 0,  # passes drafted by the model's own fast path
    "mtp": 0,  # passes drafted by the multi-token-prediction head
    "gated": 0,  # drafts the governor skipped
    "trimmed": 0,  # draft tokens cut because their rows cost more than they save
    "replays": 0,
    "fast_replays": 0,  # replays rebuilt from the pass's own outputs
    "spec": 0,  # MTP drafts made inside a pass for the next one
    "spec_hit": 0,  # of those, used because the pass was accepted in full
    "syncs": 0,  # pending tokens pulled to the CPU early to look for a draft
    # Where a pass's time goes, summed over the generation: the verify forward,
    # the replay a partial accept needs on a hybrid cache, the part of the
    # forward spent waiting on expert reads, and what it read.
    "pass_s": 0.0,
    "replay_s": 0.0,
    "wait_s": 0.0,
    "read_mb": 0.0,
}
_SINCE_LOG = 0
# The route log of the generation in progress, for tick_log.
_ROUTES = None


def reset_stats() -> None:
    global _SINCE_LOG
    for k in STATS:
        STATS[k] = 0
    _SINCE_LOG = 0


def stats() -> dict:
    if not STATS["passes"]:
        return {}
    d = STATS["drafted"]
    return {
        **STATS,
        "accept_rate": round(STATS["accepted"] / d, 4) if d else 0.0,
        "tokens_per_pass": round(STATS["emitted"] / STATS["passes"], 3),
    }


def tick_log(*, final: bool = False) -> None:
    st = stats()
    if not st:
        return
    if final:
        turn_trace.note(verify=st)
    print(
        f"[verify] {'done' if final else 'tick'} passes={st['passes']} "
        f"accept={st['accept_rate']:.2f} tok/pass={st['tokens_per_pass']:.2f} "
        f"exact_tok={st['emitted']} plain_tok={st['plain']}"
        + (f" self={st['self']}" if st["self"] else "")
        + (f" mtp={st['mtp']}" if st["mtp"] else "")
        + (f" gated={st['gated']}" if st["gated"] else "")
        + (f" trimmed={st['trimmed']}" if st["trimmed"] else "")
        + (f" replays={st['replays']}" if st["replays"] else "")
        + (f" fast={st['fast_replays']}" if st["fast_replays"] else "")
        + (f" spec={st['spec_hit']}/{st['spec']}" if st["spec"] else "")
        + (
            f" pass={st['pass_s'] * 1e3 / st['passes']:.0f}ms"
            f" wait={st['wait_s'] * 1e3 / st['passes']:.0f}ms"
            f" read={st['read_mb'] / st['passes']:.0f}MB"
            + (
                f" replay={st['replay_s'] * 1e3 / st['replays']:.0f}ms"
                if st["replays"]
                else ""
            )
        )
        + (_ROUTES.describe() if _ROUTES is not None and _ROUTES.stats["planned"] else "")
        + _GOV.describe(),
        flush=True,
    )


def _maybe_tick(n: int) -> None:
    global _SINCE_LOG
    _SINCE_LOG += n
    every = max(1, int(getattr(config, "SIDECAR_LOG_EVERY", 32) or 32))
    if _SINCE_LOG >= every:
        _SINCE_LOG = 0
        tick_log()


class _Governor:
    """Draft only where a pass has been measured cheaper per token than plain.

    One EMA of seconds per token for plain steps per span ("strict" inside
    fidelity spans, "loose" elsewhere). A drafter's cost per (span, drafter,
    match bucket) is total seconds over total emitted tokens across its recent
    passes, so a pass that emitted nine tokens counts nine times what a
    rejected one that emitted a single token does. (An average of per-pass
    ratios charged every miss as much as a hit: at 50% accept it read 135 ms
    against a true 50 ms, and gated drafts that were winning.) A bucket is
    judged only after MIN_PASSES passes. A losing one is retried after EXPLORE
    chances, and the gap doubles on every retry that still loses, so a drafter
    that never pays costs well under 1% once the gap has grown. Plain timing is
    re-taken every REFRESH passes so a warming cache cannot leave a stale
    baseline behind.
    """

    ALPHA = 0.3
    # Per drafted pass, so a bucket remembers about the last six.
    DECAY = 0.85
    EXPLORE = 16
    MAX_GAP = 512
    MIN_PLAIN = 3
    MIN_PASSES = 3
    REFRESH = 64
    STALE = 0.25  # weight left on a bucket's history when a retry starts

    def __init__(self, key=None):
        self.key = key
        self.plain: dict[str, float] = {}
        self.plain_n: dict[str, int] = {}
        # (span, key) -> [decayed seconds, decayed tokens, passes]
        self.draft: dict[tuple, list] = {}
        self.skips: dict[tuple, int] = {}
        self.gap: dict[tuple, int] = {}
        self.trial: dict[tuple, int] = {}
        self.since_plain: dict[str, int] = {}

    def ready(self, span: str) -> bool:
        return self.plain_n.get(span, 0) >= self.MIN_PLAIN and (
            self.since_plain.get(span, 0) < self.REFRESH
        )

    def cost(self, k: tuple):
        """Seconds per emitted token for (span, key), or None before any pass."""
        v = self.draft.get(k)
        if not v or v[1] <= 0:
            return None
        return v[0] / v[1]

    def want(self, span: str, key: tuple) -> bool:
        if not self.ready(span):
            return False
        k = (span, key)
        if self.trial.get(k, 0) > 0:
            return True
        d = self.cost(k)
        if d is None or self.draft[k][2] < self.MIN_PASSES or d <= self.plain[span]:
            return True
        self.skips[k] = self.skips.get(k, 0) + 1
        if self.skips[k] >= self.gap.get(k, self.EXPLORE):
            # A retry gets MIN_PASSES passes that outweigh the old ones, so a
            # drafter that started paying (warmer cache, cheaper replays) can
            # show it instead of being judged on one pass against stale history.
            self.skips[k] = 0
            self.trial[k] = self.MIN_PASSES
            v = self.draft[k]
            v[0] *= self.STALE
            v[1] *= self.STALE
            return True
        return False

    def first(self, span: str, a: tuple, b: tuple) -> bool:
        """Whether drafter `a` should be tried before `b`: the untimed one
        first, then whichever costs less per emitted token."""
        ka, kb = (span, a), (span, b)
        ta = self.cost(ka) is not None and self.draft[ka][2] >= self.MIN_PASSES
        tb = self.cost(kb) is not None and self.draft[kb][2] >= self.MIN_PASSES
        if ta != tb:
            return tb
        if not ta:
            return True
        return self.cost(ka) <= self.cost(kb)

    def pays(self, span: str, key: tuple) -> bool:
        """Untimed yet, or timed at no more per token than a plain step."""
        k = (span, key)
        d = self.cost(k)
        base = self.plain.get(span)
        return d is None or self.draft[k][2] < self.MIN_PASSES or base is None or d <= base

    def note_plain(self, span: str, sec: float) -> None:
        old = self.plain.get(span)
        self.plain[span] = sec if old is None else old + self.ALPHA * (sec - old)
        self.plain_n[span] = self.plain_n.get(span, 0) + 1
        if self.plain_n[span] >= self.MIN_PLAIN:
            self.since_plain[span] = 0

    def note_draft(self, span: str, key: tuple, sec: float, tokens: int = 1) -> None:
        """One pass (and any replay it needed): `sec` total for `tokens` emitted."""
        k = (span, key)
        old = self.draft.get(k)
        if old is None:
            self.draft[k] = [float(sec), float(max(1, tokens)), 1]
        else:
            old[0] = old[0] * self.DECAY + sec
            old[1] = old[1] * self.DECAY + max(1, tokens)
            old[2] += 1
        self.since_plain[span] = self.since_plain.get(span, 0) + 1
        if self.since_plain[span] >= self.REFRESH:
            # Fresh baseline: MIN_PLAIN new plain steps before drafting resumes.
            self.plain_n[span] = 0
        left = self.trial.get(k, 0)
        if left > 1:
            self.trial[k] = left - 1
            return
        self.trial.pop(k, None)
        base = self.plain.get(span)
        now = self.cost(k)
        if base is not None and now is not None and now > base:
            self.gap[k] = min(self.MAX_GAP, self.gap.get(k, self.EXPLORE // 2) * 2)
        else:
            self.gap.pop(k, None)

    def describe(self) -> str:
        parts = []
        for span, p in sorted(self.plain.items()):
            best = [c for k in self.draft if k[0] == span for c in [self.cost(k)] if c is not None]
            if best:
                parts.append(f"{span}={p * 1e3:.0f}/{min(best) * 1e3:.0f}ms")
        return (" plain/pass " + " ".join(parts)) if parts else ""


_GOV = _Governor()


class _SpecGate:
    """Whether an MTP pass drafts the next pass's tokens in the same graph.

    A pass kept in full then needs no separate head step and no sync before
    the next one; a pass cut short wasted the head's work. So it pays when
    P(kept in full) times the head's own draft time beats what the head adds
    to a pass. P is read per bucket of the head's confidence in the draft
    being checked, and both times are measured on this model and machine.
    The added cost is re-measured every REFRESH passes so it cannot go stale.
    """

    EDGES = (0.5, 0.8, 0.95)
    ALPHA = 0.2
    MIN = 4
    REFRESH = 256

    def __init__(self):
        self.full: dict[int, list] = {}
        self.pass_s: dict[bool, list] = {True: [0.0, 0], False: [0.0, 0]}
        self.draft_s = None
        self.n = 0
        self.last = 0
        self.bin = None

    def _ema(self, slot: list, sec: float) -> None:
        slot[0] = sec if slot[1] == 0 else slot[0] + self.ALPHA * (sec - slot[0])
        slot[1] += 1

    def want(self, conf) -> bool:
        self.n += 1
        self.bin = None if conf is None else sum(1 for e in self.EDGES if conf >= e)
        if self.bin is None:
            return False
        on, off = self.pass_s[True], self.pass_s[False]
        if off[1] < self.MIN:
            return False
        if on[1] < self.MIN or self.n - self.last >= self.REFRESH:
            self.last = self.n
            return True
        if self.draft_s is None:
            return False
        kept = self.full.get(self.bin)
        if kept is None or kept[1] < self.MIN:
            return False
        if kept[0] / kept[1] * self.draft_s > on[0] - off[0]:
            self.last = self.n
            return True
        return False

    def note_pass(self, spec: bool, sec: float, full: bool) -> None:
        self._ema(self.pass_s[spec], sec)
        if self.bin is not None:
            k = self.full.setdefault(self.bin, [0.0, 0.0])
            k[0] = k[0] * 0.97 + float(full)
            k[1] = k[1] * 0.97 + 1.0

    def note_draft(self, sec: float) -> None:
        self.draft_s = sec if self.draft_s is None else self.draft_s + self.ALPHA * (sec - self.draft_s)


_SPEC = _SpecGate()

# Match-length buckets for the governor: <5, 5-7, 8-15, 16-31, 32+ tokens.
# A longer verbatim match is a far better bet than a short one, and each bucket
# is timed (and sized, see `_width`) on its own.
_BUCKET_EDGES = (5, 8, 16, 32)


def _bucket(n: int) -> int:
    return sum(1 for e in _BUCKET_EDGES if n >= e)


def _proc_name(proc) -> str:
    """A logits processor's name for a GPU crumb label. Resumed decodes wrap
    each processor in a lambda that carries the real one as a default."""
    inner = (getattr(proc, "__defaults__", None) or (None,))[-1]
    if getattr(proc, "__name__", "") == "<lambda>" and callable(inner):
        proc = inner
    return getattr(proc, "__qualname__", None) or type(proc).__name__


def _widest(k: int) -> int:
    """Most draft tokens any bucket may use (the pass runs one more position)."""
    if not config.VERIFY_WIDE:
        return k
    return max(k, min(k + 3, streaming._DECODE_MAX_TOKENS - 1))


def _long_limit() -> int:
    """Draft tokens for a 32+ token verbatim match. Pass width is one more."""
    return max(1, min(int(config.VERIFY_LONG_TOKENS), streaming._DECODE_MAX_TOKENS - 1))


def _width(bucket: int, k: int) -> int:
    """Draft length for a match bucket. A short match is a weak bet, so it risks
    fewer positions (a rejected row is still computed); a long one is worth more.
    A 32-token match is a copy of text already seen, so it may run out to
    VERIFY_LONG_TOKENS."""
    if not config.VERIFY_WIDE:
        return k
    if bucket <= 0:
        return max(2, k // 2)
    if bucket == 1:
        return max(3, (3 * k) // 4)
    if bucket == 2:
        return k
    if bucket >= 4:
        return max(_widest(k), _long_limit())
    return _widest(k)


class _RowCost:
    """What each extra row of a pass is worth (config.VERIFY_TRUNCATE).

    Row j (j >= 1) emits a token only if the j drafts before it are all
    accepted, and then saves one plain step. Whether or not it does, it reads
    the experts it needs that nothing holds yet, and a rejected row's reads
    are pure waste. Keep row j while

        accept^j * plain_step_s  >=  row_bytes / drive_bw + ROW_FRAC * plain_step_s

    row_bytes is what recent passes read per extra row beyond what a plain step
    reads, drive_bw what the cache has measured (bytes read over time blocked),
    accept per (drafter, bucket) per drafted token. Where the cache holds the
    working set row_bytes is near 0 and only the accept odds cut a draft.
    """

    DECAY = 0.85
    # Per-token accept before a bucket has passes: <5, 5-7, 8-15, 16-31, 32+.
    PRIOR = (0.5, 0.6, 0.7, 0.8, 0.9)
    PRIOR_WEIGHT = 4.0
    # Compute a row adds beyond its reads (gather rows, attention, logits), as a
    # share of a plain step. Passes measured ~0.15 on Flash (9 rows, 140 ms vs
    # 55 ms plain) and on MiniMax-M3 (7 rows, 250 ms outside reads vs ~100 ms).
    ROW_FRAC = 0.1

    def __init__(self):
        self.acc: dict[tuple, list[float]] = {}
        self.row_bytes: float | None = None
        self.plain_bytes: float | None = None
        self.trimmed = 0

    def note_plain(self, read_bytes: float) -> None:
        b = float(max(0.0, read_bytes))
        self.plain_bytes = b if self.plain_bytes is None else self.plain_bytes + 0.3 * (b - self.plain_bytes)

    def accept(self, key) -> float:
        prior = self.PRIOR[min(int(key[1]), 4)] if key[0] == "ngram" else 0.6
        s, f = self.acc.get(key, (0.0, 0.0))
        w = self.PRIOR_WEIGHT
        return (s + w * prior) / (s + f + w)

    def note(self, key, drafted: int, accepted: int, read_bytes: float, rows: int) -> None:
        v = self.acc.setdefault(key, [0.0, 0.0])
        v[0] = v[0] * self.DECAY + accepted
        v[1] = v[1] * self.DECAY + (1.0 if accepted < drafted else 0.0)
        if rows < 2:
            return
        if self.plain_bytes is None:
            per = float(read_bytes) / rows
        else:
            per = max(0.0, float(read_bytes) - self.plain_bytes) / (rows - 1)
        self.row_bytes = per if self.row_bytes is None else self.row_bytes + 0.3 * (per - self.row_bytes)

    @staticmethod
    def drive_bw(ec) -> float:
        wait = float(getattr(ec, "disk_wait_s", 0.0) or 0.0)
        read = float(getattr(ec, "bytes_read", 0) or 0)
        if wait < 1.0 or read <= 0:
            return 10e9
        return min(20e9, max(2e9, read / wait))

    def width(self, key, n_draft: int, plain_s, ec, floor: int = 0) -> int:
        """Rows (drafts) worth running, floor..n_draft.

        floor keeps rows the governor has timed as paying (or not timed yet):
        a width of 0 runs no pass, so nothing would ever correct the estimate.
        """
        if plain_s is None or n_draft <= 0:
            return n_draft
        p = self.accept(key)
        cost = (self.row_bytes or 0.0) / self.drive_bw(ec)
        prob = 1.0
        keep = min(floor, n_draft)
        for j in range(1, n_draft + 1):
            prob *= p
            if prob * plain_s < cost + self.ROW_FRAC * plain_s:
                break
            keep = max(keep, j)
        if keep < n_draft:
            self.trimmed += n_draft - keep
        return keep


_ROWS = _RowCost()


# Newline and indent token ids, filled by configure() from the tokenizer.
# Empty until then, so the indent drafter stays quiet.
_NL_IDS: set[int] = set()
_WS_IDS: set[int] = set()


def configure(tokenizer) -> None:
    """Learn which token ids are a newline and which are pure indent after one.

    Probes only. A tokenizer that glues the indent to the next word leaves the
    sets empty and the drafter does nothing.
    """
    global _NL_IDS, _WS_IDS
    nl: set[int] = set()
    ws: set[int] = set()

    def enc(s: str) -> list[int]:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(s)
        return [int(t) for t in list(ids)]

    try:
        base = enc("\n")
        bare = enc("\nx")
    except Exception:
        _NL_IDS, _WS_IDS = set(), set()
        return
    if len(base) != 1 or not bare or bare[0] != base[0]:
        _NL_IDS, _WS_IDS = set(), set()
        return
    nl.add(base[0])
    x = bare[-1]
    for n in range(1, 13):
        for ch in (" ", "\t"):
            try:
                seq = enc("\n" + (ch * n) + "x")
            except Exception:
                continue
            if len(seq) >= 3 and seq[0] == base[0] and seq[-1] == x:
                ws.update(seq[1:-1])
    _NL_IDS, _WS_IDS = nl, ws


def draft_indent(tokens: list[int], k: int) -> list[int]:
    """Whitespace tokens that opened the previous line, when this line just started.

    `tokens` ends in a newline. The draft is the indent that followed the
    newline before that. Verified like any other draft; a miss keeps the real
    token.
    """
    if k <= 0 or not _NL_IDS or not _WS_IDS or len(tokens) < 3:
        return []
    if tokens[-1] not in _NL_IDS:
        return []
    i = len(tokens) - 2
    stop = max(-1, i - 512)
    while i > stop and tokens[i] not in _NL_IDS:
        i -= 1
    if i < 0 or tokens[i] not in _NL_IDS:
        return []
    lead: list[int] = []
    j = i + 1
    end = len(tokens) - 1
    while j < end and tokens[j] in _WS_IDS and len(lead) < k:
        lead.append(tokens[j])
        j += 1
    return lead


class _Index:
    """Incremental n-gram positions, so a draft lookup is O(matches), not O(context).

    lookup_decode.draft_ngram rescans the whole context every step (3.6 ms at
    13k tokens); this one pays that once per generation and a dict insert per
    token after.
    """

    SCAN = 64

    def __init__(self, tokens, sources=()):
        self.m = max(1, int(config.LOOKUP_NGRAM_MIN))
        self.toks: list[int] = []
        self.pos: dict[tuple, list[int]] = {}
        # Copy-only runs (see server._copy_sources): matched against, never
        # part of the sequence.
        self.src: list[list[int]] = []
        self.src_pos: dict[tuple, list[tuple[int, int]]] = {}
        self.last_n = 0
        self.last_src: int | None = None
        self._src: int | None = None
        for t in tokens:
            self.add(t)
        for run in sources:
            self.add_source(run)

    def add(self, t: int) -> None:
        self.toks.append(int(t))
        n = len(self.toks)
        if n >= self.m:
            self.pos.setdefault(tuple(self.toks[n - self.m :]), []).append(n - self.m)

    def add_source(self, run) -> None:
        run = [int(t) for t in run]
        if len(run) <= self.m:
            return
        i = len(self.src)
        self.src.append(run)
        m = self.m
        for p in range(len(run) - m):
            self.src_pos.setdefault(tuple(run[p : p + m]), []).append((i, p))

    def _best(self, seq, cands, k, best, best_score):
        """Fold the matches `cands` (start offsets into `seq`) into the best."""
        toks, m = self.toks, self.m
        n = len(toks)
        max_n = max(m, int(config.VERIFY_MATCH_CAP))
        seen = 0
        for start in reversed(cands):
            if seq is toks and start + m >= n:
                continue
            seen += 1
            if seen > self.SCAN:
                break
            length = m
            while (
                length < max_n
                and start - (length - m) - 1 >= 0
                and n - length - 1 >= 0
                and seq[start - (length - m) - 1] == toks[n - length - 1]
            ):
                length += 1
            cont = seq[start + m : start + m + k]
            if not cont:
                continue
            score = (length, len(cont) >= k)
            if best_score is None or score > best_score:
                best, best_score = cont, score
                self._src = (start + m) if seq is toks else None
        return best, best_score

    def query(self, k: int) -> list[int]:
        """Up to k tokens following the longest recent match of the suffix.

        `last_src` is the context position of the first returned token when
        the match is in the sequence itself (None for a copy-only run).
        """
        self.last_n = 0
        self.last_src = None
        self._src = None
        toks, m = self.toks, self.m
        n = len(toks)
        if k <= 0 or n < m + 1:
            return []
        key = tuple(toks[n - m :])
        best, best_score = None, None
        cands = self.pos.get(key)
        if cands:
            best, best_score = self._best(toks, cands, k, best, best_score)
        src = self.src_pos.get(key)
        if src:
            by_run: dict[int, list[int]] = {}
            for i, p in src[-self.SCAN :]:
                by_run.setdefault(i, []).append(p)
            for i, starts in by_run.items():
                best, best_score = self._best(self.src[i], starts, k, best, best_score)
        if not best:
            return []
        self.last_n = best_score[0]
        self.last_src = self._src
        return list(best)


def _pass_mode():
    """Routing for a verify pass and its replays (config.VERIFY_FIDELITY)."""
    if str(config.VERIFY_FIDELITY).strip().lower() == "exact" or streaming.decode_is_exact():
        return streaming.exact_step()
    return streaming.matched_step()


def _self_draft_on() -> bool:
    mode = str(config.SELF_DRAFT).strip().lower()
    if mode in ("0", "off", "false", ""):
        return False
    if mode in ("1", "on", "true"):
        return True
    return streaming.decode_is_exact()


_FLOW_DRAFTS = False  # this generation's self-drafts run on the flow graph


def _self_here() -> bool:
    """Whether this step may self-draft.

    Forced on/off follows SELF_DRAFT. "auto" self-drafts every step of an exact
    config, and only tool-call tokens of an approximate one, unless drafts run
    on the flow graph: then a draft is cheap enough to try in any span the
    drafter engages in, and the governor keeps it only where it pays.
    """
    mode = str(config.SELF_DRAFT).strip().lower()
    if mode in ("0", "off", "false", ""):
        return False
    if mode in ("1", "on", "true"):
        return True
    if streaming.decode_is_exact() or _FLOW_DRAFTS:
        return True
    return bool(config.SELF_DRAFT_TOOL) and fidelity.in_tool()


def _self_k(full: int) -> int:
    """Draft width. Tool-only auto uses the short width; exact, forced and flow use full."""
    mode = str(config.SELF_DRAFT).strip().lower()
    if mode in ("1", "on", "true") or streaming.decode_is_exact() or _FLOW_DRAFTS:
        return full
    return max(1, min(int(config.SELF_DRAFT_TOOL_TOKENS), streaming._DECODE_MAX_TOKENS - 1))


def _draft_routing() -> tuple[float, float]:
    """Prune / wait for a self-draft forward.

    At least DRAFT_PRUNE, and always looser than the step being checked when
    that step is a tool-call token, so the guess reads fewer experts than the
    pass that accepts or rejects it.
    """
    prune = float(config.PRUNE)
    wait = float(config.WAIT_ABOVE)
    if config.FIDELITY and fidelity.in_tool():
        prune, wait = fidelity.adjust(prune, wait)
    draft_p = max(float(config.DRAFT_PRUNE), min(0.95, prune + 0.2))
    draft_w = max(float(config.DRAFT_WAIT_ABOVE), wait)
    return draft_p, draft_w


def _span_on(span: str) -> bool:
    if streaming.decode_is_exact():
        return True  # every step is exact already; drafting is pure lookup
    scope = str(config.VERIFY_SPANS).strip().lower()
    return scope == "all" or span == "strict"


def wanted(model) -> bool:
    """Whether this decode loop can do anything for `model` right now."""
    if not config.VERIFY:
        return False
    info = getattr(model, "_pagedmoe_info", None) or {}
    if info.get("mode") == "resident":
        return getattr(model, "_pagedmoe_mtp", None) is not None or config.VERIFY_RESIDENT_NGRAM
    if info.get("mode") != "streamed":
        return False
    if streaming.decode_is_exact() or _self_draft_on():
        return True
    if getattr(model, "_pagedmoe_mtp", None) is not None:
        return True
    scope = str(config.VERIFY_SPANS).strip().lower()
    return scope == "all" or bool(config.FIDELITY)


def generate_step(
    prompt,
    model,
    *,
    max_tokens: int = 256,
    sampler=None,
    logits_processors=None,
    prompt_cache,
    history=(),
    sources=None,
    quantize_cache_fn=None,
    generation_stream=None,
):
    """mlx-lm's decode loop plus verify passes; yields (token, logprobs).

    `prompt` is the un-prefilled remainder (one token), `history` what the
    cache already holds; the n-gram drafter matches against both. `sources`,
    if given, returns extra token runs to copy from (called once, the first
    time a draft is looked up).
    """
    global _GOV, _ROWS, _ROUTES, _FLOW_DRAFTS, _SPEC
    _FLOW_DRAFTS = streaming.draft_flow_on(model)
    if not isinstance(prompt, mx.array):
        prompt = mx.array(prompt, dtype=mx.uint32)
    if sampler is None:
        sampler = lambda x: mx.argmax(x, axis=-1)  # noqa: E731
    quantize = quantize_cache_fn or (lambda _c: None)
    if generation_stream is None:
        generation_stream = mx.default_stream(mx.default_device())
    cache = prompt_cache
    limit = max_tokens if max_tokens >= 0 else float("inf")
    k_ngram = max(1, min(int(config.VERIFY_TOKENS), streaming._DECODE_MAX_TOKENS - 1))
    k_self = max(1, min(int(config.SELF_DRAFT_TOKENS), streaming._DECODE_MAX_TOKENS - 1))
    k_wide = _widest(k_ngram)
    k_long = _long_limit()
    k_mtp = max(1, min(int(config.MTP_TOKENS), streaming._DECODE_MAX_TOKENS - 1))
    mtp = getattr(model, "_pagedmoe_mtp", None)
    # A resident model has no cheap approximate step to self-draft with and no
    # expert reads for a pass to share, so only the MTP head and n-gram
    # matches draft there. Its decode is exact, so n-grams may run in any span.
    resident = (getattr(model, "_pagedmoe_info", None) or {}).get("mode") == "resident"
    ngram_resident = resident and bool(config.VERIFY_RESIDENT_NGRAM)

    def _ngram_here(bucket: int) -> bool:
        # Next to a live MTP head, a short match drafts worse than the head on
        # new text (measured -9% on a resident 27B summary), so only copies of
        # 8+ tokens take the pass there.
        if resident and mtp is not None and not mtp.dead:
            return bucket >= 2
        return True
    if mtp is not None:
        mtp.reset()
        if mtp.dead:
            mtp = None
    gov_key = (
        k_ngram, k_self, k_long, float(config.PRUNE), float(config.WAIT_ABOVE),
        str(config.VERIFY_FIDELITY).strip().lower(), bool(config.VERIFY_WIDE),
        bool(config.SELF_DRAFT_TOOL), int(config.SELF_DRAFT_TOOL_TOKENS),
        _FLOW_DRAFTS, k_mtp if mtp is not None else 0, ngram_resident,
    )
    if _GOV.key != gov_key:
        _GOV = _Governor(gov_key)
        _ROWS = _RowCost()
        _SPEC = _SpecGate()
    reset_stats()
    ec = getattr(model, "_pagedmoe_cache", None)
    routes = getattr(model, "_pagedmoe_routes", None) if config.VERIFY_REPLAY else None
    _ROUTES = routes

    seed = [int(t) for t in history] + [int(t) for t in prompt.tolist()]
    index: _Index | None = None
    proc_seq = None  # what mlx-lm's logits processors see: tokens fed since prefill

    def _index() -> _Index:
        nonlocal index
        if index is None:
            runs = ()
            if sources is not None:
                try:
                    runs = sources() or ()
                except Exception as e:
                    print(f"[verify] copy sources skipped: {e!r}", flush=True)
            index = _Index(seed, runs)
        return index

    def _emit(tok: int) -> None:
        seed.append(tok)
        if index is not None:
            index.add(tok)

    def _step(y):
        nonlocal proc_seq
        if mtp is not None:
            with mtp.capture():
                logits = model(y[None], cache=cache)[:, -1, :]
            mtp.fed(y)
        else:
            logits = model(y[None], cache=cache)[:, -1, :]
        crumbs = gpu_crumbs.ON
        if crumbs:
            gpu_crumbs.crumb(logits, "logits (final norm and LM head done)")
        if logits_processors:
            proc_seq = y if proc_seq is None else mx.concatenate([proc_seq, y])
            for proc in logits_processors:
                logits = proc(proc_seq, logits)
                if crumbs:
                    gpu_crumbs.crumb(logits, f"logits processor {_proc_name(proc)} done")
        quantize(cache)
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        if crumbs:
            gpu_crumbs.crumb(logprobs, "logprobs done (sampler input)")
            gpu_crumbs.crumb(mx.random.state[0], "random state ready (sampler input)")
        return sampler(logprobs), logprobs.squeeze(0)

    def _rewind(snap, fed: int) -> None:
        if snap is None:
            got = trim_prompt_cache(cache, fed)
            if got != fed:
                raise RuntimeError(f"verify decode could not rewind: asked {fed}, trimmed {got}")
            return
        for c, (kind, s) in zip(cache, snap):
            if kind == "arrays":
                c.cache = list(s)
                continue
            got = c.trim(fed)
            if got != fed:
                raise RuntimeError(f"verify decode could not rewind: asked {fed}, trimmed {got}")
            for side, state in s:
                side.state = state

    def _self_draft(y: int, k: int) -> list[int]:
        snap = _snapshot(cache) if _needs_snapshot(cache) else None
        out = []
        cur = y
        with streaming.draft_step(*_draft_routing()):
            for _ in range(k):
                logits = model(mx.array([[cur]], dtype=mx.uint32), cache=cache)[0, -1]
                cur = int(mx.argmax(logits).item())
                out.append(cur)
        _rewind(snap, k)
        return out

    checked = [0.0]  # fast-replay self-check seconds in the current pass

    def _full_replay(keep: list[int]) -> None:
        mx.eval(model(mx.array([keep], dtype=mx.uint32), cache=cache))

    def _pass(y: int, draft: list[int], key=None, spec_k: int = 0):
        nonlocal proc_seq
        n_draft = len(draft)
        snap = _snapshot(cache) if _needs_snapshot(cache) else None
        feed = [y] + draft
        t_pass = time.perf_counter()
        ec = getattr(model, "_pagedmoe_cache", None)
        wait0 = getattr(ec, "disk_wait_s", 0.0)
        read0 = getattr(ec, "bytes_read", 0)
        fast = fast_replay.for_model(model) if snap is not None and config.FAST_REPLAY else None
        with _pass_mode(), (mtp.capture() if mtp is not None else contextlib.nullcontext()), (
            fast.recording(len(feed)) if fast is not None else contextlib.nullcontext()
        ):
            logits = model(mx.array([feed], dtype=mx.uint32), cache=cache)[0]
        base = proc_seq
        if logits_processors:
            rows = []
            for i in range(len(feed)):
                tail = mx.array(feed[: i + 1], dtype=mx.uint32)
                seq = tail if base is None else mx.concatenate([base, tail])
                row = logits[i : i + 1]
                for proc in logits_processors:
                    row = proc(seq, row)
                rows.append(row)
            logits = mx.concatenate(rows, axis=0)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if gpu_crumbs.ON:
            gpu_crumbs.crumb(logprobs, f"verify pass {n_draft} drafted: logprobs done (sampler input)")
            gpu_crumbs.crumb(mx.random.state[0], "random state ready (sampler input)")
        picks = sampler(logprobs)
        gpu_crumbs.crumb(picks, f"verify pass {n_draft} drafted: tokens sampled")
        spec = mtp.speculate(feed, picks.reshape(-1)[-1:], spec_k) if spec_k and mtp is not None else None
        if spec is not None:
            STATS["spec"] += 1
            mx.eval(picks, logprobs, spec)
        else:
            mx.eval(picks, logprobs)
        t_done = time.perf_counter()
        STATS["pass_s"] += t_done - t_pass
        STATS["wait_s"] += max(0.0, getattr(ec, "disk_wait_s", 0.0) - wait0)
        STATS["read_mb"] += max(0, getattr(ec, "bytes_read", 0) - read0) / 1e6
        picked = [int(t) for t in picks.reshape(-1).tolist()]
        out = []
        for i, tok in enumerate(picked):
            out.append((tok, logprobs[i]))
            if i >= n_draft or tok != draft[i]:
                break
        accepted = len(out) - 1
        keep = feed[: accepted + 1]
        if key is not None and key[0] == "mtp" and config.MTP_SPECULATE:
            _SPEC.note_pass(spec is not None, t_done - t_pass, accepted == n_draft)
        if spec is not None and accepted == n_draft:
            mtp.commit()
        elif mtp is not None:
            if spec is not None:
                mtp.rollback()
            mtp.fed(mx.array(keep, dtype=mx.uint32))
        if n_draft > accepted:
            if snap is not None:
                with _pass_mode():
                    if fast is None:
                        _replay(model, cache, snap, len(feed), keep)
                    elif fast.replay(cache, snap, len(feed), keep, _full_replay):
                        STATS["fast_replays"] += 1
                    if fast is not None:
                        checked[0] += fast.take_check_s()
                STATS["replays"] += 1
                STATS["replay_s"] += time.perf_counter() - t_done - checked[0]
            else:
                _rewind(None, n_draft - accepted)
        if logits_processors:
            tail = mx.array(keep, dtype=mx.uint32)
            proc_seq = tail if base is None else mx.concatenate([base, tail])
        quantize(cache)
        STATS["passes"] += 1
        STATS["drafted"] += n_draft
        STATS["accepted"] += accepted
        STATS["emitted"] += len(out)
        return out, snap

    n = 0
    unfed = None  # yielded but not yet in the cache
    inflight = None  # (snapshot, tokens a pass left in the cache) while they are handed out
    shown = 0
    failed = False
    try:
        gpu_trail.step(f"decode first step ({int(prompt.size)} tok left from prefill)")
        with mx.stream(generation_stream):
            y, lp = _step(prompt)
            gpu_crumbs.crumb(y, "token 0 sampled")
        mx.async_eval(y, lp)
        pend_arr, pend_lp, pend_val = y, lp, None
        plain_mark = None  # perf_counter after the last plain token, for pipelined timing
        plain_read = None  # ec.bytes_read at the last plain token
        consumer = 0.0
        while n < limit:
            if activity.stopping():
                gpu_trail.mark("decode stopped for shutdown")
                break
            span = "strict" if fidelity.strict() else "loose"
            if resident:
                engaged = ngram_resident and _GOV.ready(span)
            else:
                engaged = _span_on(span) and _GOV.ready(span)
            self_ok = engaged and not resident and _self_here()
            mtp_ok = mtp is not None and not mtp.dead and _GOV.ready(span)
            indent_ok = engaged and bool(_NL_IDS) and (
                fidelity.in_tool() or streaming.decode_is_exact()
            )
            plan = None  # drafter already cleared by the governor this step
            mtp_first = not self_ok or _GOV.first(span, ("mtp", 0), ("self", 0))
            if pend_val is None and (engaged or mtp_ok) and can_rewind(cache):
                # Pull the pending token early only when a draft is likely: a
                # match ending one token back is a necessary condition for one
                # ending at the pending token.
                guess = _index().query(k_wide + 1) if engaged else None
                order = (("mtp", mtp_ok), ("self", self_ok))
                if not mtp_first:
                    order = order[::-1]
                if guess and _ngram_here(_bucket(index.last_n)) and _GOV.want(span, ("ngram", _bucket(index.last_n))):
                    plan = "ngram"
                else:
                    for name, ok in order:
                        if ok and _GOV.want(span, (name, 0)):
                            plan = name
                            break
                if plan is not None:
                    pend_val = int(pend_arr.item())
                    STATS["syncs"] += 1

            if pend_val is None:
                gpu_trail.step(f"decode plain step, token {n}")
                with mx.stream(generation_stream):
                    nxt = _step(pend_arr)
                    gpu_crumbs.crumb(nxt[0], f"token {n + 1} sampled")
                mx.async_eval(*nxt)
                tok = int(pend_arr.item())
                now = time.perf_counter()
                if plain_mark is not None:
                    _GOV.note_plain(span, max(0.0, now - plain_mark - consumer))
                if ec is not None:
                    b = getattr(ec, "bytes_read", 0)
                    if plain_read is not None:
                        _ROWS.note_plain(b - plain_read)
                    plain_read = b
                STATS["plain"] += 1
                t_y = time.perf_counter()
                yield tok, pend_lp
                consumer = time.perf_counter() - t_y
                plain_mark = now
                n += 1
                _emit(tok)
                if n % 256 == 0:
                    mx.clear_cache()
                pend_arr, pend_lp = nxt
                continue

            plain_mark = None
            plain_read = None
            tok = pend_val
            unfed = tok
            yield tok, pend_lp
            n += 1
            _emit(tok)
            if n >= limit:
                break
            room = int(min(max(k_ngram, k_long), limit - n - 1))
            draft, key = [], None
            t0 = time.perf_counter()
            if room > 0 and (engaged or mtp_ok) and can_rewind(cache):
                if engaged and plan in (None, "ngram"):
                    draft = _index().query(int(min(k_long, limit - n - 1)))
                if draft and not _ngram_here(_bucket(index.last_n)):
                    draft = []
                if draft:
                    bucket = _bucket(index.last_n)
                    draft = draft[: _width(bucket, k_ngram)]
                    key = ("ngram", bucket)
                    # The early check approved a match one token back. The real
                    # match can be a longer bucket with its own width, so the
                    # governor has to see that bucket, not the shorter one.
                    if not _GOV.want(span, key):
                        draft, key = [], None
                        STATS["gated"] += 1
                if not draft and indent_ok and tok in _NL_IDS and _GOV.want(span, ("indent", 0)):
                    ind = draft_indent(seed, min(4, room))
                    if ind:
                        draft, key = ind, ("indent", 0)
                for name in (("mtp", "self") if mtp_first else ("self", "mtp")):
                    if draft:
                        break
                    if name == "mtp" and mtp_ok and (
                        plan == "mtp" or (plan is None and _GOV.want(span, ("mtp", 0)))
                    ):
                        gpu_trail.step(f"decode mtp draft, token {n}")
                        hit = mtp.ready is not None and mtp.ready[0] == tok
                        if hit:
                            STATS["spec_hit"] += 1
                        t_d = time.perf_counter()
                        with mx.stream(generation_stream):
                            draft = mtp.draft(tok, int(min(k_mtp, room)))
                        if draft and not hit and config.MTP_SPECULATE:
                            _SPEC.note_draft(time.perf_counter() - t_d)
                        if draft:
                            key = ("mtp", 0)
                            STATS["mtp"] += 1
                    elif name == "self" and self_ok and (
                        plan == "self" or (plan is None and _GOV.want(span, ("self", 0)))
                    ):
                        gpu_trail.step(f"decode self draft, token {n}")
                        with mx.stream(generation_stream):
                            draft = _self_draft(tok, int(min(_self_k(k_self), room)))
                        key = ("self", 0)
                        STATS["self"] += 1
            plan = None
            if draft and routes is not None and key is not None and key[0] == "ngram":
                src = index.last_src if index is not None else None
                if src is not None:
                    plan = routes.plan(src - 1, [tok] + draft)
            if draft and config.VERIFY_TRUNCATE and ec is not None:
                n_keep = _ROWS.width(key, len(draft), _GOV.plain.get(span), ec,
                                     floor=1 if _GOV.pays(span, key) else 0)
                if n_keep < len(draft):
                    STATS["trimmed"] += len(draft) - n_keep
                    draft = draft[:n_keep]
            if draft and plan is not None:
                # Scored on every layer; reads start only on layers that earned it.
                routes.arm(ec, plan, len(draft) + 1)
            gpu_trail.step(
                f"decode verify pass {key[0]} {len(draft)} drafted, token {n}"
                if draft and key is not None
                else f"decode plain step after a sync, token {n}"
            )
            with mx.stream(generation_stream):
                if not draft:
                    STATS["plain"] += 1
                    pend_arr, pend_lp = _step(mx.array([tok], dtype=mx.uint32))
                    gpu_crumbs.crumb(pend_arr, f"token {n} sampled")
                    mx.async_eval(pend_arr, pend_lp)
                    pend_val = None
                    unfed = None
                    continue
                read0 = getattr(ec, "bytes_read", 0) if ec is not None else 0
                spec_k = 0
                if (
                    key is not None and key[0] == "mtp" and config.MTP_SPECULATE
                    and mtp is not None and not mtp.dead and _SPEC.want(mtp.conf)
                ):
                    spec_k = k_mtp
                checked[0] = 0.0
                got, snap = _pass(tok, draft, key, spec_k)
            if routes is not None:
                routes.end_pass()
            unfed = None
            _ROWS.note(
                key, len(draft), len(got) - 1,
                (getattr(ec, "bytes_read", 0) - read0) if ec is not None else 0,
                len(draft) + 1,
            )
            _GOV.note_draft(span, key, time.perf_counter() - t0 - checked[0], len(got))
            inflight = (snap, [tok] + [t for t, _ in got[:-1]])
            shown = 1
            for t, lp2 in got[:-1]:
                shown += 1
                yield t, lp2
                n += 1
                _emit(t)
            inflight = None
            _maybe_tick(len(got))
            pend_val, pend_lp = got[-1]
            pend_arr = None
    except GeneratorExit:
        raise
    except BaseException:
        failed = True
        raise
    finally:
        # mlx-lm has fed every token it yields by the time it yields it, and
        # the server keys the stored cache on exactly those tokens.
        if not failed and (inflight is not None or unfed is not None):
            gpu_trail.mark("decode exit: rewinding or feeding the cache")
        if not failed and inflight is not None and shown < len(inflight[1]):
            with mx.stream(generation_stream), _pass_mode():
                unfeed(model, cache, inflight[0], inflight[1], shown)
                quantize(cache)
                mx.eval([c.state for c in cache])
        elif unfed is not None and not failed:
            with mx.stream(generation_stream):
                model(mx.array([[unfed]], dtype=mx.uint32), cache=cache)
                quantize(cache)
                mx.eval([c.state for c in cache])
        tick_log(final=True)
