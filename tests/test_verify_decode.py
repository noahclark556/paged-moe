# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Exact-verify decode (verify_decode.py).

What has to hold:

  1. With no drafting, the loop is mlx-lm's generate_step: same tokens, same
     logprobs, bit for bit.
  2. With drafting forced on every step, output is still the model's own
     (greedy here), on a trimmable cache and on a hybrid one, with n-gram and
     self drafts, with a logits processor in play.
  3. On a streamed model under PRUNE 0.9 + WAIT_ABOVE, every token a verify
     pass emits is the *exact* model's pick for its prefix.
  4. The cache holds every yielded token when the loop ends, including when
     the caller stops early, because the server keys the stored cache on them.
     Same for lookup_decode.
  5. The governor backs off a losing drafter and re-takes its plain baseline.

Run:
    python tests/test_verify_decode.py
"""

import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.generate import generate_step as mlx_generate_step  # noqa: E402
from mlx_lm.models import qwen3_moe  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.sample_utils import make_logits_processors  # noqa: E402

from tests.test_lookup_hybrid import ARGS as HYBRID_ARGS  # noqa: E402
from tests.test_lookup_hybrid import build as build_hybrid  # noqa: E402
from tests.test_lookup_hybrid import check_greedy  # noqa: E402
from tests.test_stream import CONFIG, build_tiny_model  # noqa: E402

from pagedmoe import config as es_config  # noqa: E402
from pagedmoe import fidelity  # noqa: E402
from pagedmoe import get_stats, load, lookup_decode, streaming  # noqa: E402
from pagedmoe import verify_decode as vd  # noqa: E402


def build_trim():
    mx.random.seed(3)
    cfg = dict(CONFIG)
    cfg.pop("quantization", None)
    cfg.update(hidden_size=64, num_experts=8, num_experts_per_tok=2, vocab_size=128)
    model = qwen3_moe.Model(qwen3_moe.ModelArgs.from_dict(cfg))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model, cfg["vocab_size"]


def _offset(cache):
    return max(int(getattr(c, "offset", 0) or 0) for c in cache if c.is_trimmable())


def _run(model, prompt, n, *, procs=None, stop_after=None, gen_fn=None):
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([prompt[:-1]]), cache=cache))
    fn = gen_fn or vd.generate_step
    gen = fn(
        mx.array(prompt[-1:], dtype=mx.uint32),
        model,
        max_tokens=n,
        logits_processors=procs,
        prompt_cache=cache,
        history=prompt[:-1],
    )
    toks, lps = [], []
    for tok, lp in gen:
        toks.append(int(tok))
        lps.append(np.asarray(lp.astype(mx.float32)))
        if stop_after is not None and len(toks) >= stop_after:
            gen.close()
            break
    return toks, lps, cache


def _mlx(model, prompt, n, procs=None):
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([prompt[:-1]]), cache=cache))
    toks, lps = [], []
    for tok, lp in mlx_generate_step(
        mx.array(prompt[-1:], dtype=mx.uint32),
        model,
        max_tokens=n,
        logits_processors=procs,
        prompt_cache=cache,
    ):
        toks.append(int(tok))
        lps.append(np.asarray(lp.astype(mx.float32)))
    return toks, lps, cache


class _Force:
    """Drafting on every step: governor always ready and willing, drafts from `oracle`."""

    def __init__(self, oracle):
        self.oracle = oracle

    def __enter__(self):
        self.saved = (vd._Governor.ready, vd._Governor.want, vd._Index.query)
        oracle = self.oracle
        vd._Governor.ready = lambda self, span: True
        vd._Governor.want = lambda self, span, key: True

        def query(index, k):
            index.last_n = 8
            return oracle(list(index.toks), k)

        vd._Index.query = query
        return self

    def __exit__(self, *exc):
        vd._Governor.ready, vd._Governor.want, vd._Index.query = self.saved


def _spoiled(prompt, ref, vocab):
    def oracle(toks, k):
        g = len(toks) - len(prompt)
        out = list(ref[g : g + k]) if g >= 0 else []
        if not out:
            return [int(toks[-1])] * k
        bad = g % 5
        if bad < len(out):
            out[bad] = (out[bad] + 1) % vocab
        return out

    return oracle


def check_plain_matches_mlx(model, prompt, n):
    old = vd._Governor.MIN_PLAIN
    vd._Governor.MIN_PLAIN = 10**9
    try:
        a, la, ca = _run(model, prompt, n)
        b, lb, cb = _mlx(model, prompt, n)
    finally:
        vd._Governor.MIN_PLAIN = old
    assert a == b, f"plain loop diverged from mlx-lm\n  got={a}\n  ref={b}"
    worst = max(float(np.abs(x - y).max()) for x, y in zip(la, lb))
    assert worst == 0.0, f"logprobs differ from mlx-lm by {worst}"
    assert _offset(ca) == _offset(cb), f"cache offset {_offset(ca)} vs mlx-lm {_offset(cb)}"
    assert vd.STATS["passes"] == 0
    print("OK: undrafted loop is mlx-lm's generate_step, tokens and logprobs bit-identical")


def greedy_given_prefix(model, prompt, got, procs=None, margin=0.1):
    """Each token is the greedy pick for the prefix actually emitted.

    A multi-token pass does not round exactly like one-token steps (about
    0.003 in fp32 logits here), so a near-tie can flip and the two runs part
    ways from then on. Judging each token against its own prefix tests the
    rewind and the acceptance rule rather than the kernels.
    """
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([prompt[:-1]]), cache=cache))
    seq = [prompt[-1]]
    logits = model(mx.array([seq]), cache=cache)[0, -1:]
    worst = 0.0
    for i, t in enumerate(got):
        row = logits
        for proc in procs or []:
            row = proc(mx.array(seq, dtype=mx.uint32), row)
        row = row[0]
        best = int(mx.argmax(row))
        gap = float(row[best] - row[t])
        worst = max(worst, gap)
        assert gap <= margin, f"token {i} is {t}, greedy says {best} (gap {gap:.4f})"
        seq.append(t)
        logits = model(mx.array([[t]]), cache=cache)[0, -1:]
    return worst


def check_forced_trim(model, vocab, prompt, n):
    ref, _, _ = _mlx(model, prompt, n)
    with _Force(_spoiled(prompt, ref, vocab)):
        got, _, cache = _run(model, prompt, n)
    st = dict(vd.STATS)
    greedy_given_prefix(model, prompt, got)
    assert st["passes"] > 0 and st["accepted"] > 0, f"drafts never accepted: {st}"
    assert st["accepted"] < st["drafted"], "no rejection exercised"
    assert _offset(cache) == len(prompt) + n - 1 + 1, (
        f"cache offset {_offset(cache)} != prompt {len(prompt)} + {n} yielded"
    )
    print(f"OK: forced n-gram drafts on a trimmable cache keep greedy output ({st['passes']} passes, accept {st['accepted']}/{st['drafted']})")


def check_processors(model, vocab, prompt, n):
    procs = make_logits_processors(repetition_penalty=1.5, repetition_context_size=20)
    ref, _, _ = _mlx(model, prompt, n, procs)
    with _Force(_spoiled(prompt, ref, vocab)):
        got, _, _ = _run(model, prompt, n, procs=procs)
    assert vd.STATS["accepted"] > 0
    greedy_given_prefix(model, prompt, got, procs=procs)
    print("OK: logits processors see the same history as mlx-lm under drafting")


class _PieceTok:
    """Just enough tokenizer for ThinkBudget: id 127 is the think end tag."""

    think_end_id = 127

    def get_vocab(self):
        return {str(i): i for i in range(128)}

    def batch_decode(self, ids):
        return ["</think>" if i == 127 else ("." if i % 7 == 0 else "ab") for (i,) in ids]

    def encode(self, text, add_special_tokens=False):
        return [3, 5, 9, 127]


def check_think_budget(model, vocab, prompt, n):
    from pagedmoe import think_budget
    from pagedmoe.think_budget import ThinkBudget

    tok = _PieceTok()
    make = lambda: [ThinkBudget(tok, 12, "")]  # noqa: E731
    old = think_budget.SLACK_TOKENS
    think_budget.SLACK_TOKENS = 4
    try:
        _check_think_budget(model, vocab, prompt, n, make)
    finally:
        think_budget.SLACK_TOKENS = old


def _check_think_budget(model, vocab, prompt, n, make):
    ref, _, _ = _mlx(model, prompt, n, make())
    assert make()[0].closed_in(ref), f"plain decode never closed: {ref}"
    with _Force(_spoiled(prompt, ref, vocab)):
        got, _, cache = _run(model, prompt, n, procs=make())
    assert vd.STATS["accepted"] > 0
    greedy_given_prefix(model, prompt, got, procs=make())
    assert make()[0].closed_in(got), f"drafted decode never closed: {got}"
    assert _offset(cache) == len(prompt) + n
    print("OK: think budget closes the same way under drafting, cache holds every yielded token")


def check_self_draft(model, prompt, n):
    ref, _, _ = _mlx(model, prompt, n)
    old = es_config.SELF_DRAFT
    es_config.SELF_DRAFT = "1"
    try:
        with _Force(lambda toks, k: []):
            got, _, cache = _run(model, prompt, n)
    finally:
        es_config.SELF_DRAFT = old
    st = dict(vd.STATS)
    greedy_given_prefix(model, prompt, got)
    assert st["self"] > 0 and st["accepted"] > 0, f"self-draft never ran: {st}"
    assert _offset(cache) == len(prompt) + n
    print(f"OK: self-draft keeps greedy output ({st['self']} passes, accept {st['accepted']}/{st['drafted']})")


def check_early_close(model, vocab, prompt):
    ref, _, _ = _mlx(model, prompt, 30)
    for stop in (1, 2, 5, 9):
        with _Force(_spoiled(prompt, ref, vocab)):
            got, _, cache = _run(model, prompt, 30, stop_after=stop)
        assert len(got) == stop
        assert _offset(cache) == len(prompt) + stop, (
            f"stopped after {stop}: cache offset {_offset(cache)}, want {len(prompt) + stop}"
        )
    for stop in (1, 3, 7):
        _, _, cache = _run(model, prompt, 30, stop_after=stop, gen_fn=lookup_decode.generate_step)
        assert _offset(cache) == len(prompt) + stop, (
            f"lookup stopped after {stop}: offset {_offset(cache)}"
        )
    _, _, cache = _run(model, prompt, 12, gen_fn=lookup_decode.generate_step)
    assert _offset(cache) == len(prompt) + 12
    print("OK: every yielded token is in the cache at the end, early stop included (verify and lookup)")


def check_hybrid(prompt_len=48, n=36):
    model = build_hybrid()
    rng = np.random.default_rng(2)
    vocab = HYBRID_ARGS["vocab_size"]
    prompt = rng.integers(0, vocab, size=prompt_len).tolist()
    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt]), cache=cache)
    ref = []
    for _ in range(n):
        y = int(mx.argmax(logits[0, -1]))
        ref.append(y)
        logits = model(mx.array([[y]]), cache=cache)
    with _Force(_spoiled(prompt, ref, vocab)):
        got, _, c = _run(model, prompt, n)
    st = dict(vd.STATS)
    assert len(got) == n
    check_greedy(model, prompt, got)
    assert st["replays"] > 0, f"no partial accept replayed on the hybrid cache: {st}"
    attn = [x for x in c if x.is_trimmable()]
    assert all(x.offset == len(prompt) + n for x in attn), "hybrid cache offset wrong at the end"
    # Oracle drafts from the answer unspoiled, so passes accept everything and
    # an early stop lands inside a pass's accepted tokens.
    clean = lambda toks, k: list(ref[len(toks) - len(prompt) : len(toks) - len(prompt) + k])  # noqa: E731
    for stop in (2, 4, 7):
        with _Force(clean):
            got, _, c = _run(model, prompt, n, stop_after=stop)
        attn = [x for x in c if x.is_trimmable()]
        assert all(x.offset == len(prompt) + stop for x in attn), (
            f"hybrid stop after {stop}: offsets {[x.offset for x in attn]}"
        )
        fresh = make_prompt_cache(model)
        la = model(mx.array([[7]]), cache=c)
        mx.eval(model(mx.array([prompt + got]), cache=fresh))
        lb = model(mx.array([[7]]), cache=fresh)
        d = float(mx.abs(la - lb).max())
        assert d < 0.05, f"hybrid state after early stop is off by {d}"
    print(f"OK: hybrid cache drafts, replays, stays greedy, and early stops unfeed ({st['replays']} replays)")


def check_governor():
    g = vd._Governor(("t",))
    span = "strict"
    key = ("ngram", 2)
    assert not g.want(span, key), "drafted before any plain baseline"
    for _ in range(g.MIN_PLAIN):
        g.note_plain(span, 0.010)
    assert g.want(span, key), "untried drafter not tried"
    # A bucket is judged only after MIN_PASSES passes: one unlucky pass must not gate it.
    for _ in range(g.MIN_PASSES - 1):
        g.note_draft(span, key, 0.030, 1)
        assert g.want(span, key), "gated before MIN_PASSES passes"
    g.note_draft(span, key, 0.030, 1)
    gaps = []
    for _ in range(5):
        tries = 0
        while not g.want(span, key):
            tries += 1
        gaps.append(tries + 1)
        # A retry is a short trial, judged once at its end.
        for i in range(g.MIN_PASSES):
            if i:
                assert g.want(span, key), "retry trial cut short"
            g.note_draft(span, key, 0.030, 1)
    assert gaps == sorted(gaps) and gaps[-1] > gaps[0], f"backoff did not grow: {gaps}"
    # A drafter that starts paying wins its next trial despite a long losing history.
    while not g.want(span, key):
        pass
    for i in range(g.MIN_PASSES):
        g.note_draft(span, key, 0.006, 1)
    assert g.want(span, key) and g.want(span, key), "a drafter that started paying stayed gated"
    for _ in range(12):
        g.note_draft(span, key, 0.0016, 8)
    assert g.want(span, key), "winning drafter gated"
    for _ in range(g.REFRESH):
        g.note_draft(span, key, 0.0016, 8)
    assert not g.ready(span), "plain baseline never re-taken"
    for _ in range(g.MIN_PLAIN):
        g.note_plain(span, 0.010)
    assert g.ready(span)
    assert not g.ready("loose"), "spans share a baseline"
    print(f"OK: governor waits for a baseline, backs off {gaps}, refreshes plain")


def check_governor_weighting():
    """Cost is seconds over tokens emitted, not an average of per-pass ratios.

    Half the passes are rejected (1 token for 0.25 s), half accept 8 (9 tokens
    for 0.15 s). Per-pass ratios average (0.25 + 0.0167) / 2 = 133 ms, which loses
    to a 95 ms plain step. Weighted it is 0.4 s / 10 tokens = 40 ms, a win.
    """
    g = vd._Governor(("w",))
    span = "strict"
    key = ("ngram", 3)
    for _ in range(g.MIN_PLAIN):
        g.note_plain(span, 0.095)
    for _ in range(12):
        g.note_draft(span, key, 0.25, 1)
        g.note_draft(span, key, 0.15, 9)
    c = g.cost((span, key))
    assert c is not None and c < 0.095, f"mixed hits and misses read {c}"
    assert g.want(span, key), "a drafter that wins on average was gated"
    # And a drafter that only ever misses is still gated.
    bad = ("ngram", 0)
    for _ in range(8):
        g.note_draft(span, bad, 0.25, 1)
    assert g.cost((span, bad)) > 0.095
    assert not all(g.want(span, bad) for _ in range(3)), "a losing drafter was never gated"
    print(f"OK: governor weighs passes by tokens emitted ({c * 1e3:.0f} ms vs 133 ms by ratio)")


def check_indent_and_tool_self():
    saved_nl, saved_ws = set(vd._NL_IDS), set(vd._WS_IDS)
    old = (
        es_config.SELF_DRAFT,
        es_config.SELF_DRAFT_TOOL,
        es_config.PRUNE,
        es_config.WAIT_ABOVE,
        es_config.FIDELITY_TOOL_SCALE,
        fidelity._in_tool,
    )
    try:
        vd._NL_IDS, vd._WS_IDS = {10}, {32}
        assert vd.draft_indent([10, 32, 32, 7, 10], 4) == [32, 32]
        assert vd.draft_indent([10, 32, 7, 10], 4) == [32]
        assert vd.draft_indent([10, 7, 10], 4) == []
        assert vd.draft_indent([10, 32, 32, 7], 4) == []
        es_config.SELF_DRAFT = "auto"
        es_config.SELF_DRAFT_TOOL = True
        es_config.PRUNE = 0.8
        es_config.WAIT_ABOVE = 0.0
        assert not streaming.decode_is_exact()
        fidelity._in_tool = False
        assert not vd._self_here()
        fidelity._in_tool = True
        assert vd._self_here()
        assert vd._self_k(4) == es_config.SELF_DRAFT_TOOL_TOKENS
        es_config.PRUNE = 0.0
        assert streaming.decode_is_exact()
        assert vd._self_here()
        assert vd._self_k(4) == 4
        p, w = vd._draft_routing()
        assert p == es_config.DRAFT_PRUNE and w == es_config.DRAFT_WAIT_ABOVE
        es_config.PRUNE = 0.9
        es_config.FIDELITY_TOOL_SCALE = 0.6
        fidelity._in_tool = True
        p, _w = vd._draft_routing()
        assert p > 0.9 * 0.6, p
    finally:
        vd._NL_IDS, vd._WS_IDS = saved_nl, saved_ws
        (
            es_config.SELF_DRAFT,
            es_config.SELF_DRAFT_TOOL,
            es_config.PRUNE,
            es_config.WAIT_ABOVE,
            es_config.FIDELITY_TOOL_SCALE,
            fidelity._in_tool,
        ) = old
    print("OK: indent draft copies the previous line, tool self-draft stays inside tool calls")


def check_buckets():
    assert [vd._bucket(n) for n in (3, 4, 5, 7, 8, 15, 16, 31, 32, 48)] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    saved = es_config.VERIFY_WIDE
    try:
        es_config.VERIFY_WIDE = True
        w = [vd._width(b, 8) for b in range(5)]
        assert w == sorted(w) and w[2] == 8 and w[0] < 8 < w[4], w
        assert max(w) + 1 <= vd.streaming._DECODE_MAX_TOKENS, "pass wider than the decode path"
        es_config.VERIFY_WIDE = False
        assert [vd._width(b, 8) for b in range(5)] == [8] * 5
    finally:
        es_config.VERIFY_WIDE = saved
    print(f"OK: match buckets and draft widths {w}")


def check_index():
    es_config.LOOKUP_NGRAM_MIN, es_config.LOOKUP_NGRAM_MAX = 3, 8
    es_config.VERIFY_MATCH_CAP = 8
    rng = np.random.default_rng(0)
    for trial in range(200):
        toks = rng.integers(0, 6, size=int(rng.integers(4, 120))).tolist()
        k = int(rng.integers(1, 9))
        idx = vd._Index(toks)
        got = idx.query(k)
        want = lookup_decode.draft_ngram(toks, k)
        if want:
            assert got, f"index missed a match draft_ngram found: {toks} k={k}"
            assert idx.last_n == lookup_decode._LAST_MATCH, (
                f"match length {idx.last_n} vs {lookup_decode._LAST_MATCH}"
            )
        else:
            assert not got, f"index drafted where draft_ngram found nothing: {toks}"
    idx = vd._Index([1, 2, 3, 4, 5, 9, 1, 2, 3])
    assert idx.query(3) == [4, 5, 9]
    idx.add(4)
    assert idx.query(2) == [5, 9]
    print("OK: incremental n-gram index agrees with draft_ngram on match length")


def check_streamed_exact():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build_tiny_model(tmp)
        rng = np.random.default_rng(3)
        prompt = rng.integers(0, CONFIG["vocab_size"], size=50).tolist()
        n = 24
        es_config.PRUNE = 0.9
        es_config.WAIT_ABOVE = 0.3
        es_config.KEEP_FREE = False
        es_config.VERIFY_FIDELITY = "exact"
        try:
            model, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
            assert not streaming.decode_is_exact()
            assert vd.wanted(model), "streamed model under FIDELITY should want verify decode"

            def exact_greedy(prefix, k):
                with streaming.exact_step():
                    cache = make_prompt_cache(model)
                    logits = model(mx.array([prefix]), cache=cache)
                    out = []
                    for _ in range(k):
                        y = int(mx.argmax(logits[0, -1]))
                        out.append(y)
                        logits = model(mx.array([[y]]), cache=cache)
                return out

            def oracle(toks, k):
                out = exact_greedy(toks, k)
                g = len(toks)
                bad = g % 4
                if bad < len(out):
                    out[bad] = (out[bad] + 1) % CONFIG["vocab_size"]
                return out

            with _Force(oracle):
                got, _, _ = _run(model, prompt, n)
            st = dict(vd.STATS)
            assert st["passes"] > 0 and st["accepted"] > 0, st
            # Token 0 is the first plain step (it seeds the loop); every later
            # token came out of an exact pass.
            worst = 0.0
            with streaming.exact_step():
                cache = make_prompt_cache(model)
                logits = model(mx.array([prompt + got[:1]]), cache=cache)[0, -1]
                for i, t in enumerate(got[1:], start=1):
                    row = logits.astype(mx.float32)
                    best = int(mx.argmax(row))
                    gap = float(row[best] - row[t])
                    assert gap <= 0.05, (
                        f"token {i} is {t}, exact model says {best} (gap {gap:.4f}); stats {st}"
                    )
                    worst = max(worst, gap)
                    logits = model(mx.array([[t]]), cache=cache)[0, -1]
            assert get_stats(model)["cache"]["pruned_frac"] >= 0.0

            plain_ref = exact_greedy(prompt, n)
            # A stitched token counts what it pruned lazily; stats() folds it in.
            ec = model._pagedmoe_cache
            ec.stats()
            before = ec.pruned_slots
            cache = make_prompt_cache(model)
            logits = model(mx.array([prompt]), cache=cache)
            for _ in range(8):
                y = int(mx.argmax(logits[0, -1]))
                logits = model(mx.array([[y]]), cache=cache)
            ec.stats()
            pruned = ec.pruned_slots - before
            assert pruned > 0, "PRUNE 0.9 dropped nothing, so the exactness check proved nothing"
            del model
            print(f"OK: under PRUNE 0.9 every verify-pass token is the exact model's pick (worst gap {worst:.4f}, {st['passes']} passes; plain steps pruned {pruned} slots)")
            _ = plain_ref

            es_config.PRUNE = 0.0
            es_config.WAIT_ABOVE = 0.0
            # The per-layer draft route; flow drafts are covered in test_flow.
            es_config.DRAFT_FLOW = False
            model, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
            assert streaming.decode_is_exact()
            before = model._pagedmoe_cache.pruned_slots
            cache = make_prompt_cache(model)
            mx.eval(model(mx.array([prompt]), cache=cache))
            with streaming.draft_step(0.9, 0.0):
                for t in range(6):
                    mx.eval(model(mx.array([[t]]), cache=cache))
            assert model._pagedmoe_cache.pruned_slots > before, "draft_step override ignored"
            mid = model._pagedmoe_cache.pruned_slots
            for t in range(6):
                mx.eval(model(mx.array([[t]]), cache=cache))
            assert model._pagedmoe_cache.pruned_slots == mid, "draft_step leaked past its block"
            del model
            print("OK: draft_step prunes only inside its block on an exact config")
        finally:
            es_config.DRAFT_FLOW = True
            es_config.PRUNE = 0.0
            es_config.WAIT_ABOVE = 0.0
            es_config.KEEP_FREE = True
            es_config.VERIFY_FIDELITY = "plain"


def check_streamed_plain():
    """VERIFY_FIDELITY=plain: a pass routes every row like the plain step would.

    PRUNE only (WAIT_ABOVE depends on what happens to be resident, so two runs
    of plain decode can differ under it). Drafting forced on every step must
    give plain greedy decode's tokens, and the passes must have pruned.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build_tiny_model(tmp)
        rng = np.random.default_rng(4)
        prompt = rng.integers(0, CONFIG["vocab_size"], size=50).tolist()
        n = 24
        es_config.PRUNE = 0.9
        es_config.WAIT_ABOVE = 0.0
        es_config.KEEP_FREE = False
        es_config.VERIFY_FIDELITY = "plain"
        try:
            model, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
            assert not streaming.decode_is_exact()
            ref, _, _ = _mlx(model, prompt, n)

            def oracle(toks, k):
                g = len(toks) - len(prompt)
                out = list(ref[g : g + k])
                if g % 3 == 1 and out:
                    out[-1] = (out[-1] + 1) % CONFIG["vocab_size"]
                return out

            ec = model._pagedmoe_cache
            with _Force(oracle):
                ec.stats()
                before = ec.pruned_slots
                got, _, _ = _run(model, prompt, n)
                ec.stats()
                pruned = ec.pruned_slots - before
            st = dict(vd.STATS)
            assert st["passes"] > 0 and st["accepted"] > 0, st
            assert pruned > 0, "matched passes computed the full top-k"
            assert got == ref, f"plain-fidelity verify != plain decode\n ref={ref}\n got={got}"
            del model
            print(f"OK: plain-fidelity passes emit plain decode's tokens ({st['passes']} passes, {pruned} slots pruned)")
        finally:
            es_config.PRUNE = 0.0
            es_config.WAIT_ABOVE = 0.0
            es_config.KEEP_FREE = True


def check_copy_sources():
    """Copy-only runs draft where the context itself has no match."""
    es_config.LOOKUP_NGRAM_MIN, es_config.LOOKUP_NGRAM_MAX = 3, 8
    es_config.VERIFY_MATCH_CAP = 8
    ctx = [1, 2, 3, 4, 5, 6]
    src = [40, 41, 42, 43, 44, 45, 46]
    idx = vd._Index(ctx + [41, 42, 43], [src])
    assert idx.query(3) == [44, 45, 46], idx.query(3)
    idx2 = vd._Index(ctx + [41, 42, 43])
    assert idx2.query(3) == [], "no source, no draft"
    # A longer match in the context beats a shorter one in a source.
    idx3 = vd._Index([7, 41, 42, 43, 9, 9] + [7, 41, 42, 43], [[41, 42, 43, 50]])
    assert idx3.query(2) == [9, 9], idx3.query(2)
    print("OK: drafter matches copy-only sources, context wins on longer matches")


def main():
    old_spans = es_config.VERIFY_SPANS
    es_config.VERIFY_SPANS = "all"
    try:
        check_index()
        check_indent_and_tool_self()
        check_copy_sources()
        check_governor()
        check_governor_weighting()
        check_buckets()
        model, vocab = build_trim()
        rng = np.random.default_rng(5)
        prompt = rng.integers(0, vocab, size=40).tolist()
        check_plain_matches_mlx(model, prompt, 24)
        check_forced_trim(model, vocab, prompt, 30)
        check_processors(model, vocab, prompt, 30)
        check_think_budget(model, vocab, prompt, 30)
        check_self_draft(model, prompt, 20)
        check_early_close(model, vocab, prompt)
        check_hybrid()
        check_streamed_exact()
        check_streamed_plain()
    finally:
        es_config.VERIFY_SPANS = old_spans
    print("\nOK: verify decode tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
