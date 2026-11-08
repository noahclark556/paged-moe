# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Loop rescue (rescue.py).

What has to hold:

  1. The watch holds on a copy, releases when it breaks, and confirms a loop
     only after a full period. Varied tokens never trip it.
  2. A tiny random model decoding greedily falls into loops by itself. With
     the rescue forced on and no nudge, the exact redo writes the same tokens,
     so the output is plain decode's, bit for bit, with a logits processor in
     play, on a trimmable cache and on a hybrid one.
  3. With the nudge, the loop is gone and the cache still holds exactly what
     was shown, so the next turn continues from the right state.
  4. Stopping early, in or out of a hold, leaves exactly the shown tokens in
     the cache.
  5. At an exact config nothing changes and nothing is held.
  6. The lag only moves when tokens go out: same tokens, same cache on an
     early stop, and EOS goes out the step it is decoded.
  7. Garble (a stand-in forcing unlikely "number" tokens) never shows: every
     shown token is one the exact model gives at least RESCUE_AUDIT_P, and
     the cache still holds exactly what was shown.
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.sample_utils import make_logits_processors  # noqa: E402

from tests.test_lookup_hybrid import ARGS as HYBRID_ARGS  # noqa: E402
from tests.test_lookup_hybrid import build as build_hybrid  # noqa: E402
from tests.test_verify_decode import build_trim  # noqa: E402

from pagedmoe import config, fidelity, rescue  # noqa: E402
from pagedmoe import verify_decode as vd  # noqa: E402
from pagedmoe.rescue import CLEAR, HOLD, LOOP, NUM, LoopWatch, Numbers  # noqa: E402


@pytest.fixture(autouse=True)
def _approx(monkeypatch):
    monkeypatch.setattr(config, "PRUNE", 0.7)
    monkeypatch.setattr(config, "WAIT_ABOVE", 0.0)
    monkeypatch.setattr(config, "RESCUE", True)
    monkeypatch.setattr(config, "RESCUE_GRAM", 4)
    monkeypatch.setattr(config, "RESCUE_MAX_PERIOD", 48)
    monkeypatch.setattr(config, "RESCUE_CONFIRM", 12)
    monkeypatch.setattr(config, "RESCUE_TOKENS", 64)
    monkeypatch.setattr(config, "RESCUE_STICKY", 3)
    monkeypatch.setattr(config, "RESCUE_MAX_TRIPS", 6)
    monkeypatch.setattr(config, "RESCUE_NUDGE", False)
    monkeypatch.setattr(config, "RESCUE_MIN_PRUNE", 0.7)
    monkeypatch.setattr(config, "RESCUE_MIN_WAIT", 0.4)
    monkeypatch.setattr(config, "RESCUE_LAG", 12)
    monkeypatch.setattr(config, "RESCUE_SHAPES", True)
    monkeypatch.setattr(config, "RESCUE_NUM_WINDOW", 12)
    monkeypatch.setattr(config, "RESCUE_NUM_MIN", 4)
    # The test models are near flat (greedy tokens get p 0.02-0.03), so the
    # audit runs only where a test asks for it, with a floor below that.
    monkeypatch.setattr(config, "RESCUE_AUDIT", False)
    monkeypatch.setattr(config, "RESCUE_AUDIT_P", 0.01)
    monkeypatch.setattr(config, "RESCUE_AUDIT_MAX", 16)
    monkeypatch.setattr(config, "RESCUE_AUDIT_TOKENS", 16)
    monkeypatch.setattr(config, "RESCUE_AUDIT_COOL", 48)
    monkeypatch.setattr(config, "FIDELITY_GUARD", False)
    monkeypatch.setattr(rescue, "_tokenizer", None)
    monkeypatch.setattr(fidelity, "_markers", {})
    fidelity.begin_turn()
    yield
    fidelity.end_turn()


class _Tok:
    """Stand-in tokenizer: ids in `digits` read as bare numbers, the rest as words."""

    def __init__(self, digits=(), eos=()):
        self.digits = set(digits)
        self.eos_token_ids = set(eos)

    def decode(self, ids):
        return "".join(" 7" if int(i) in self.digits else " word" for i in ids)


# ------------------------------------------------------------------- watch


def _states(tokens, **kw):
    w = LoopWatch(**kw)
    return [w.push(t) for t in tokens]


def test_varied_tokens_never_hold():
    assert set(_states(range(1000, 1600))) == {CLEAR}


def test_phrase_loop_holds_then_confirms_after_one_period():
    phrase = list(range(100, 120))
    s = _states(list(range(500, 530)) + phrase + phrase + phrase[:1])
    start = 30 + 20
    assert s[: start + 3] == [CLEAR] * (start + 3), "held before GRAM tokens of the copy"
    assert s[start + 3] == HOLD
    assert LOOP in s and s.index(LOOP) == start + 19, s.index(LOOP) if LOOP in s else None


def test_copy_that_breaks_releases():
    phrase = list(range(100, 120))
    s = _states(phrase + phrase[:8] + [999] + list(range(2000, 2040)))
    assert HOLD in s and LOOP not in s
    assert s[-1] == CLEAR


def test_one_token_run_confirms_at_confirm():
    s = _states([7] * 20)
    assert s.index(LOOP) == 12


def test_far_copy_is_ignored():
    phrase = list(range(100, 120))
    s = _states(phrase + list(range(3000, 3060)) + phrase, max_period=48)
    assert LOOP not in s and HOLD not in s


def test_allowed_pattern_is_not_held_again():
    w = LoopWatch()
    phrase = list(range(100, 116))
    for t in phrase + phrase:
        st = w.push(t)
    assert st == LOOP
    w.allowed.add(w.signature())
    w.reset(phrase + phrase)
    assert [w.push(t) for t in phrase] == [CLEAR] * len(phrase)


def test_ban_only_on_first_step():
    ban = rescue._Ban(3)
    logits = mx.zeros((1, 8))
    out = ban(mx.array([5], dtype=mx.uint32), logits)
    assert float(out[0, 3]) == -float("inf") and float(out[0, 2]) == 0.0
    out = ban(mx.array([5, 6], dtype=mx.uint32), logits)
    assert float(out[0, 3]) == 0.0


def test_numbers_fire_on_a_crowd_and_back_off_after_a_clean_audit():
    n = Numbers(window=12, need=4)
    hits = [n.push(h) for h in [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1] * 4]
    assert not any(hits), "three in a window is prose with numbers"
    n = Numbers(window=12, need=4)
    hits = [n.push(h) for h in [1, 0, 1, 0, 1, 0, 1]]
    assert hits[-1] and not any(hits[:-1])
    n.passed()
    assert not any(n.push(True) for _ in range(48))
    assert n.push(True), "quiet over, the window is still full"
    n.passed()
    assert n.quiet == 96


def test_text_tracks_numbers_code_and_tool_calls(monkeypatch):
    class Tok:
        eos_token_ids = {9}
        words = {1: " the", 2: " 42", 3: "```", 4: " `x", 5: "`", 6: "\n", 7: "<tool>", 8: "</tool>", 9: "<eos>"}

        def decode(self, ids):
            return "".join(self.words[i] for i in ids)

    monkeypatch.setattr(fidelity, "_markers", {"tool_start": (7,), "tool_end": (8,)})
    t = rescue._Text(Tok())
    assert t.push(2) == (True, True, False)
    assert t.push(1) == (False, True, False)
    t.push(3)
    assert t.push(2) == (True, False, False), "inside a code fence"
    t.push(3)
    t.push(4)
    assert t.push(2)[1] is False, "inside inline code"
    t.push(5)
    assert t.push(2)[1] is True
    t.push(4)
    t.push(6)
    assert t.push(2)[1] is True, "a newline ends inline code"
    assert t.push(7) == (False, False, True)
    assert t.push(1) == (False, False, True), "no lag inside a tool call"
    assert t.push(8)[2] is True
    assert t.push(1) == (False, True, False)
    assert t.push(9)[2] is True


def test_shapes_see_counting_that_ids_do_not():
    counting = []
    for i in range(2, 40):
        counting += [1000 + i, 5]
    ids = LoopWatch()
    shapes = LoopWatch()
    assert LOOP not in [ids.push(t) for t in counting]
    s = [shapes.push(NUM if t >= 1000 else t) for t in counting]
    assert LOOP in s and s.index(LOOP) < 16


# ---------------------------------------------------------------- decoding


def _streamed(model):
    model._pagedmoe_info = {"mode": "streamed"}
    return model


def _decode(model, prompt, n, *, procs=None, guard=True, stop_after=None):
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([prompt[:-1]]), cache=cache))
    stream = mx.default_stream(mx.default_device())

    def start(p, k, prefix, ps):
        return vd.generate_step(
            p, model, max_tokens=k, logits_processors=ps, prompt_cache=cache,
            history=prompt[:-1] + prefix, generation_stream=stream,
        )

    p0 = mx.array(prompt[-1:], dtype=mx.uint32)
    gen = (
        rescue.guarded(start, p0, model, cache, max_tokens=n, logits_processors=procs, stream=stream)
        if guard
        else start(p0, n, [], procs)
    )
    out = []
    for tok, _ in gen:
        out.append(int(tok))
        if stop_after is not None and len(out) >= stop_after:
            gen.close()
            break
    return out, cache, rescue.stats()


def _attn_offsets(cache):
    return [c.offset for c in cache if c.is_trimmable() and hasattr(c, "offset")]


def _same_state(model, cache, tokens, tol=0.05):
    # Decoded and prefilled KV differ by ~2e-3 here with no rescue at all.
    fresh = make_prompt_cache(model)
    mx.eval(model(mx.array([tokens]), cache=fresh))
    a = model(mx.array([[7]]), cache=cache)
    b = model(mx.array([[7]]), cache=fresh)
    return float(mx.abs(a - b).max()) < tol


@pytest.fixture(scope="module")
def trim_model():
    model, vocab = build_trim()
    return _streamed(model), vocab


def _loopy_prompt(vocab, seed):
    return np.random.default_rng(seed).integers(0, vocab, size=24).tolist()


def test_exact_redo_is_plain_decode(trim_model):
    model, vocab = trim_model
    procs = make_logits_processors(repetition_penalty=1.05, repetition_context_size=64)
    tripped = 0
    for seed in range(6):
        prompt = _loopy_prompt(vocab, seed)
        ref, _, _ = _decode(model, prompt, 96, procs=procs, guard=False)
        got, cache, st = _decode(model, prompt, 96, procs=procs)
        tripped += st["trips"]
        assert got == ref, f"seed {seed}: rescue changed exact output"
        assert _attn_offsets(cache) == [len(prompt) + len(got)] * len(_attn_offsets(cache))
        assert _same_state(model, cache, prompt + got)
    assert tripped > 0, "no loop to rescue: the test model stopped looping"


def test_nudge_breaks_the_loop_and_keeps_the_cache(trim_model, monkeypatch):
    monkeypatch.setattr(config, "RESCUE_NUDGE", True)
    monkeypatch.setattr(rescue, "_tokenizer", type("T", (), {"decode": lambda self, ids: "words here " * 4})())
    model, vocab = trim_model
    nudged = 0
    for seed in range(6):
        prompt = _loopy_prompt(vocab, seed)
        got, cache, st = _decode(model, prompt, 96)
        nudged += st["nudged"]
        assert len(got) == 96
        assert _attn_offsets(cache) == [len(prompt) + 96] * len(_attn_offsets(cache))
        assert _same_state(model, cache, prompt + got)
    assert nudged > 0


def test_early_stop_leaves_what_was_shown(trim_model):
    model, vocab = trim_model
    prompt = _loopy_prompt(vocab, 1)
    full, _, st = _decode(model, prompt, 96)
    assert st["trips"] > 0
    for stop in range(5, 96, 7):
        got, cache, _ = _decode(model, prompt, 96, stop_after=stop)
        assert got == full[:stop]
        assert _attn_offsets(cache) == [len(prompt) + stop] * len(_attn_offsets(cache)), stop
        assert _same_state(model, cache, prompt + got), stop


def test_exact_config_passes_through(trim_model, monkeypatch):
    monkeypatch.setattr(config, "PRUNE", 0.0)
    model, vocab = trim_model
    prompt = _loopy_prompt(vocab, 0)
    ref, _, _ = _decode(model, prompt, 64, guard=False)
    got, _, st = _decode(model, prompt, 64)
    assert got == ref and st["trips"] == 0
    assert not rescue.active(model)


class _Collapse:
    """Stands in for a pruned decode losing the thread: past `after` tokens it
    writes `phrase` over and over, until decode is held exact."""

    def __init__(self, after: int, phrase: list[int]):
        self.after, self.phrase = after, phrase

    def __call__(self, tokens, logits):
        n = int(tokens.size)
        if fidelity.rescuing() or n <= self.after:
            return logits
        want = self.phrase[(n - self.after - 1) % len(self.phrase)]
        hit = mx.arange(logits.shape[-1]) == want
        return mx.where(hit, mx.array(1e4, logits.dtype), logits)


def _longest_copy(out, p):
    run = best = 0
    for j in range(p, len(out)):
        run = run + 1 if out[j] == out[j - p] else 0
        best = max(best, run)
    return best


def _phrase_run(out, phrase):
    """Longest stretch of `out` that follows `phrase` cyclically."""
    best = 0
    for i, t in enumerate(out):
        if t not in phrase:
            continue
        k, ph = 0, phrase.index(t)
        while i + k < len(out) and out[i + k] == phrase[(ph + k) % len(phrase)]:
            k += 1
        best = max(best, k)
    return best


@pytest.mark.parametrize("kind", ["trim", "hybrid"])
def test_collapse_never_shows_a_second_copy(kind, trim_model, monkeypatch):
    monkeypatch.setattr(config, "RESCUE_AUDIT", True)
    if kind == "trim":
        model, vocab = trim_model
    else:
        model, vocab = _streamed(build_hybrid()), HYBRID_ARGS["vocab_size"]
    phrase = [3, 9, 27, 81, 50, 11, 33, 99, 70, 18, 64, 5, 41, 77, 2, 60, 21, 88, 13, 45]
    phrase = [t % vocab for t in phrase]
    for seed in range(3):
        prompt = _loopy_prompt(vocab, 10 + seed)
        procs = [_Collapse(after=30, phrase=phrase)]
        fidelity.begin_turn()
        plain, _, _ = _decode(model, prompt, 120, procs=procs, guard=False)
        assert _longest_copy(plain, len(phrase)) > 40, "collapse stand-in did not loop"
        fidelity.begin_turn()
        got, cache, st = _decode(model, prompt, 120, procs=procs)
        assert st["trips"] > 0 and st["dropped"] > 0, st
        assert st["audits"] == 0, st
        # A loop the exact model writes too costs one redo, not a trip per copy.
        assert st["trips"] <= 1 + st["real"], st
        assert len(got) == 120
        # The first copy and GRAM-1 tokens of the second can show; no more.
        # (What the exact model writes after that is its own.)
        assert _phrase_run(got, phrase) < len(phrase) + config.RESCUE_GRAM, got
        assert _same_state(model, cache, prompt + got), (kind, seed)
        for stop in (35, 52, 77):
            fidelity.begin_turn()
            part, cache, _ = _decode(model, prompt, 120, procs=procs, stop_after=stop)
            assert len(part) == stop
            assert _same_state(model, cache, prompt + part), (kind, seed, stop)


def test_hybrid_exact_output_unchanged():
    model = _streamed(build_hybrid())
    vocab = HYBRID_ARGS["vocab_size"]
    for seed in range(3):
        prompt = _loopy_prompt(vocab, seed)
        ref, _, _ = _decode(model, prompt, 80, guard=False)
        got, cache, _ = _decode(model, prompt, 80)
        assert got == ref, f"seed {seed}"
        assert _same_state(model, cache, prompt + got)


def _model(kind, trim_model):
    if kind == "trim":
        return trim_model
    return _streamed(build_hybrid()), HYBRID_ARGS["vocab_size"]


@pytest.mark.parametrize("kind", ["trim", "hybrid"])
def test_lag_changes_nothing_but_timing(kind, trim_model, monkeypatch):
    monkeypatch.setattr(rescue, "_tokenizer", _Tok())
    model, vocab = _model(kind, trim_model)
    for seed in range(3):
        prompt = _loopy_prompt(vocab, seed)
        ref, _, _ = _decode(model, prompt, 80, guard=False)
        got, cache, _ = _decode(model, prompt, 80)
        assert got == ref, (kind, seed)
        assert _same_state(model, cache, prompt + got)
        for stop in (1, 5, 13, 40):
            part, cache, _ = _decode(model, prompt, 80, stop_after=stop)
            assert part == ref[:stop]
            offs = _attn_offsets(cache)
            assert offs == [len(prompt) + stop] * len(offs), (kind, seed, stop)
            assert _same_state(model, cache, prompt + part)


def test_eos_goes_out_the_step_it_is_decoded(trim_model, monkeypatch):
    model, vocab = trim_model
    prompt = _loopy_prompt(vocab, 3)
    ref, _, _ = _decode(model, prompt, 80, guard=False)
    eos = ref[30]
    k = ref.index(eos)
    monkeypatch.setattr(rescue, "_tokenizer", _Tok(eos={eos}))
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([prompt[:-1]]), cache=cache))
    stream = mx.default_stream(mx.default_device())
    made = [0]

    def start(p, n, prefix, ps):
        for item in vd.generate_step(
            p, model, max_tokens=n, logits_processors=ps, prompt_cache=cache,
            history=prompt[:-1] + prefix, generation_stream=stream,
        ):
            made[0] += 1
            yield item

    p0 = mx.array(prompt[-1:], dtype=mx.uint32)
    seen = 0
    for tok, _ in rescue.guarded(start, p0, model, cache, max_tokens=80, logits_processors=None, stream=stream):
        seen += 1
        if int(tok) == eos:
            break
    assert seen == k + 1
    assert made[0] - seen <= 1, "decode ran on past EOS"


class _Garble:
    """Stands in for a pruned decode writing junk: past `after` tokens, until
    decode is held exact, it writes the one of `ids` the model likes least."""

    def __init__(self, after: int, ids: list[int]):
        self.after, self.ids = after, list(ids)

    def __call__(self, tokens, logits):
        n = int(tokens.size)
        if fidelity.rescuing() or n <= self.after:
            return logits
        sub = logits.reshape(-1)[mx.array(self.ids)]
        want = self.ids[int(mx.argmin(sub))]
        hit = mx.arange(logits.shape[-1]) == want
        return mx.where(hit, mx.array(1e4, logits.dtype), logits)


def _exact_p(model, prompt, got):
    """Exact probability of each shown token given everything before it."""
    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt + got[:-1]]), cache=cache)[0].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tail = lp[len(prompt) - 1 :]
    return [float(mx.exp(tail[i, t])) for i, t in enumerate(got)]


def test_garble_never_shows(trim_model, monkeypatch):
    monkeypatch.setattr(config, "RESCUE_AUDIT", True)
    model, vocab = trim_model
    cuts = 0
    for seed in range(4):
        prompt = _loopy_prompt(vocab, 20 + seed)
        ref, _, _ = _decode(model, prompt, 120, guard=False)
        junk = [t for t in range(vocab) if t not in set(ref)][:16]
        monkeypatch.setattr(rescue, "_tokenizer", _Tok(digits=junk))
        procs = [_Garble(after=40, ids=junk)]
        fidelity.begin_turn()
        plain, _, _ = _decode(model, prompt, 120, procs=procs, guard=False)
        assert sum(t in junk for t in plain) > 40, "garble stand-in wrote no junk"
        fidelity.begin_turn()
        got, cache, st = _decode(model, prompt, 120, procs=procs)
        cuts += st["cuts"]
        assert len(got) == 120
        low = [(i, p) for i, p in enumerate(_exact_p(model, prompt, got)) if p < config.RESCUE_AUDIT_P]
        assert not low, (seed, low[:5], st)
        assert _same_state(model, cache, prompt + got), seed
        for stop in (30, 45, 70, 101):
            fidelity.begin_turn()
            part, cache, _ = _decode(model, prompt, 120, procs=procs, stop_after=stop)
            assert part == got[:stop]
            offs = _attn_offsets(cache)
            assert offs == [len(prompt) + stop] * len(offs), (seed, stop)
            assert _same_state(model, cache, prompt + part), (seed, stop)
    assert cuts > 0


@pytest.mark.parametrize("prune,wait,on", [(0.5, 0.0, False), (0.5, 0.42, False), (0.5, 0.43, True), (0.88, 0.0, True)])
def test_only_fast_configs_turn_it_on(trim_model, monkeypatch, prune, wait, on):
    monkeypatch.setattr(config, "RESCUE_MIN_PRUNE", 0.875)
    monkeypatch.setattr(config, "RESCUE_MIN_WAIT", 0.425)
    monkeypatch.setattr(config, "PRUNE", prune)
    monkeypatch.setattr(config, "WAIT_ABOVE", wait)
    model, _ = trim_model
    assert rescue.active(model) is on


def test_slower_configs_pass_through(trim_model, monkeypatch):
    monkeypatch.setattr(config, "RESCUE_AUDIT", True)
    monkeypatch.setattr(config, "PRUNE", 0.5)
    model, vocab = trim_model
    prompt = _loopy_prompt(vocab, 20)
    ref, _, _ = _decode(model, prompt, 120, guard=False)
    junk = [t for t in range(vocab) if t not in set(ref)][:16]
    monkeypatch.setattr(rescue, "_tokenizer", _Tok(digits=junk))
    procs = [_Garble(after=40, ids=junk)]
    fidelity.begin_turn()
    plain, _, _ = _decode(model, prompt, 120, procs=procs, guard=False)
    fidelity.begin_turn()
    got, _, st = _decode(model, prompt, 120, procs=procs)
    assert got == plain
    assert st["audits"] == 0 and st["trips"] == 0, st


def test_clean_audit_lets_the_tokens_go(trim_model, monkeypatch):
    monkeypatch.setattr(config, "RESCUE_AUDIT", True)
    model, vocab = trim_model
    audits = 0
    for seed in range(4):
        prompt = _loopy_prompt(vocab, 30 + seed)
        ref, _, _ = _decode(model, prompt, 96, guard=False)
        common = sorted(set(ref), key=ref.count, reverse=True)[:12]
        monkeypatch.setattr(rescue, "_tokenizer", _Tok(digits=common))
        got, cache, st = _decode(model, prompt, 96)
        audits += st["audits"]
        assert st["cuts"] == 0, st
        assert len(got) == 96
        assert _same_state(model, cache, prompt + got)
    assert audits > 0
