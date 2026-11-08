# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Prefill misses read into free slab slots (config.PREFILL_SLAB) or into the
prefill stage (config.PREFILL_STAGE), and the next layer's predicted reads
start early (config.PREFILL_AHEAD), past the free slots as stage or run reads
(config.PREFILL_AHEAD_LOOSE_MB).

The claim for all of them is that they only move where and when the bytes
land. Prefill logits and the decode that follows must match the run-read path
bit for bit: with room for every miss, with too few free slots (a layer mixes
slot, stage and run reads), and on a second turn whose prefill hits what
decode installed. Prefill must never evict a resident expert, and every slot must
end up accounted for exactly once (free, resident, staged or in flight).

Run:
    python tests/test_prefill_slab.py
"""

import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from tests.test_stream import CONFIG, build_tiny_model  # noqa: E402

from pagedmoe import config as es_config  # noqa: E402
from pagedmoe import get_stats, load  # noqa: E402

# (PREFILL_SLAB, PREFILL_AHEAD, PREFILL_AHEAD_LOOSE_MB, PREFILL_STAGE)
MODES = {
    "runs": (False, False, 0, False),
    "slots": (True, False, 0, False),
    "slots+ahead": (True, True, 0, False),
    "slots+ahead+loose": (True, True, 512, False),
    "stage": (False, False, 0, True),
    "slots+stage": (True, False, 0, True),
    "slots+stage+ahead": (True, True, 0, True),
}
AHEAD_MODES = ("slots+ahead", "slots+ahead+loose", "slots+stage+ahead")
SLOT_MODES = ("slots", "slots+ahead", "slots+ahead+loose", "slots+stage", "slots+stage+ahead")


def _cache_of(model):
    from pagedmoe.streaming import StreamedSwitchGLU

    for m in model.modules():
        if isinstance(m, StreamedSwitchGLU):
            return m.cache
    raise AssertionError("no streamed layer")


def run(model, prompt, n_new):
    """Full prefill logits, then greedy decode logits."""
    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt]), cache=cache)
    pre = np.asarray(logits[0].astype(mx.float32))
    tok = int(mx.argmax(logits[:, -1, :]))
    rows = []
    for _ in range(n_new):
        logits = model(mx.array([[tok]]), cache=cache)
        rows.append(np.asarray(logits[0, -1, :].astype(mx.float32)))
        tok = int(mx.argmax(logits[:, -1, :]))
    return pre, np.stack(rows)


def same(a, b, what):
    assert a.shape == b.shape, what
    assert np.array_equal(a, b), f"{what}: max |diff| {np.abs(a - b).max()}"


def slots_accounted(c):
    """Every slot is in exactly one place once reads have settled."""
    for _ in range(200):
        with c._lock:
            busy = [f for f in list(c._inflight.values()) + list(c._ahead_orphans.values()) if not f.done()]
        if not busy:
            break
        time.sleep(0.01)
    time.sleep(0.05)  # done callbacks run just after the result is set
    with c._lock:
        free = list(c._free_slots)
        assert len(free) == len(set(free)), "a slot is on the free list twice"
        used = [e["__slot__"] for e in c._lru.values() if "__slot__" in e]
        used += [r["__slot__"] for r in c._raw_ready.values() if "__slot__" in r]
        used += [c._slot_of[k] for k in c._inflight if k in c._slot_of]
        assert not set(free) & set(used), "a slot is free and in use"
        assert len(used) == len(set(used)), "two entries share a slot"
        assert len(free) + len(used) == c.slab.slots, (
            f"slots leaked: {len(free)} free + {len(used)} used != {c.slab.slots}"
        )


def guard(c):
    """Lookahead stands down on a bad precision record and re-probes once it
    decays, and the guesses it starts are all handed back."""
    layer = next(iter(c._layouts))
    ids = list(range(CONFIG["num_experts"]))
    c._ahead_n, c._ahead_u = 4 * 512.0, 0.0
    skipped = c.prefill_ahead_skipped
    assert c.prefill_ahead(layer, ids) == 0, "guard did not stand down"
    assert c.prefill_ahead_skipped == skipped + 1
    for _ in range(400):
        before = c.prefill_ahead_skipped
        c.prefill_ahead(layer, ids)
        if c.prefill_ahead_skipped == before:
            break
    assert c._ahead_n < 512, "guard never re-probed"
    c._settle_ahead(layer, [])
    slots_accounted(c)


def main():
    rng = np.random.default_rng(5)
    prompt = rng.integers(0, CONFIG["vocab_size"], size=60).tolist()
    follow = rng.integers(0, CONFIG["vocab_size"], size=24).tolist()
    n_new = 12
    old = (
        es_config.SLAB,
        es_config.PREFILL_SLAB,
        es_config.PREFILL_AHEAD,
        es_config.PREFILL_AHEAD_LOOSE_MB,
        es_config.PREFILL_STAGE,
        es_config.STITCH,
    )
    es_config.SLAB = "on"
    # A stitched slab holds every expert, so prefill has nothing to read.
    es_config.STITCH = False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            build_tiny_model(tmp)

            results = {}
            for cache_gb in (0.01, 0.0009):
                for mode, (slots, ahead, loose, stage) in MODES.items():
                    es_config.PREFILL_SLAB, es_config.PREFILL_AHEAD = slots, ahead
                    es_config.PREFILL_AHEAD_LOOSE_MB = loose
                    es_config.PREFILL_STAGE = stage
                    model, _ = load(str(tmp), mode="streamed", cache_gb=cache_gb, verbose=False)
                    first = run(model, prompt, n_new)
                    c = _cache_of(model)
                    slots_accounted(c)
                    ev0 = c.evictions
                    resident = set(c._lru)
                    # Second turn: prefill a new prompt over a slab decode filled.
                    pre2 = model(mx.array([follow]), cache=make_prompt_cache(model))
                    mx.eval(pre2)
                    assert c.evictions == ev0, "prefill evicted a resident expert"
                    assert resident <= set(c._lru), "prefill dropped a resident expert"
                    slots_accounted(c)
                    second = np.asarray(pre2[0].astype(mx.float32))
                    stats = get_stats(model)["cache"]
                    assert stats["slab_slots"] > 0, "slab did not engage"
                    results[(cache_gb, mode)] = (first, second, stats)
                    if mode == "slots+ahead+loose":
                        guard(c)
                    del model

            for cache_gb in (0.01, 0.0009):
                base = results[(cache_gb, "runs")]
                assert base[2]["prefill_slot_reads"] == 0
                assert base[2]["prefill_ahead_reads"] == 0
                assert base[2]["stage_reads"] == 0
                for mode in [m for m in MODES if m != "runs"]:
                    got = results[(cache_gb, mode)]
                    same(base[0][0], got[0][0], f"prefill logits {mode} cache={cache_gb}")
                    same(base[0][1], got[0][1], f"decode logits {mode} cache={cache_gb}")
                    same(base[1], got[1], f"2nd turn prefill {mode} cache={cache_gb}")
                    s = got[2]
                    if mode in SLOT_MODES:
                        assert s["prefill_slot_reads"] > 0, "slot reads never happened"
                    if MODES[mode][3] and (mode == "stage" or cache_gb < 0.001):
                        assert s["stage_reads"] > 0, f"{mode}: stage never read"
                    print(
                        f"cache={cache_gb} {mode}: identical; slot reads {s['prefill_slot_reads']}, "
                        f"stage reads {s['stage_reads']}, "
                        f"run fallbacks {s['prefill_slot_starved']}, ahead {s['prefill_ahead_reads']} "
                        f"used {s['prefill_ahead_used']} wasted {s['prefill_ahead_wasted_gb'] * 1e3:.2f} MB"
                    )
                for mode in AHEAD_MODES:
                    assert results[(cache_gb, mode)][2]["prefill_ahead_reads"] > 0, (
                        f"{mode}: lookahead never started a read"
                    )
            starved = results[(0.0009, "slots")][2]
            assert starved["prefill_slot_starved"] > 0, "starved case never fell back"
            tight = results[(0.0009, "slots+ahead")][2]["prefill_ahead_reads"]
            loose = results[(0.0009, "slots+ahead+loose")][2]["prefill_ahead_reads"]
            assert loose > tight, "loose tier never started a read past the free slots"

            # No slab (released for a big resumed KV, or off): lookahead still
            # runs, as host reads only, and logits do not move.
            es_config.SLAB = "off"
            es_config.PREFILL_SLAB = False
            es_config.PREFILL_AHEAD_LOOSE_MB = 512
            noslab = {}
            for ahead in (False, True):
                es_config.PREFILL_AHEAD = ahead
                model, _ = load(str(tmp), mode="streamed", cache_gb=0.01, verbose=False)
                assert _cache_of(model).slab is None
                noslab[ahead] = (run(model, prompt, n_new), get_stats(model)["cache"])
                del model
            (base, _), (got, stats) = noslab[False], noslab[True]
            same(base[0], got[0], "prefill logits, no slab")
            same(base[1], got[1], "decode logits, no slab")
            assert stats["prefill_ahead_reads"] > 0, "no-slab lookahead never started a read"
            print(
                f"no slab: identical; ahead {stats['prefill_ahead_reads']} "
                f"used {stats['prefill_ahead_used']}"
            )
    finally:
        (
            es_config.SLAB,
            es_config.PREFILL_SLAB,
            es_config.PREFILL_AHEAD,
            es_config.PREFILL_AHEAD_LOOSE_MB,
            es_config.PREFILL_STAGE,
            es_config.STITCH,
        ) = old
    print("all ok")


if __name__ == "__main__":
    main()
