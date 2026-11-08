# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Route log (route_log.py) and pass sizing (verify_decode._RowCost).

What has to hold:

  1. Rows land at their KV positions, strongest expert first, and a position
     that holds another token (or fell out of the ring) plans as unknown.
  2. Every layer of a plan is scored without reading anything; only layers
     whose precision clears the bar read ahead, and only what nothing holds.
  3. A layer that stops naming the pass's experts loses trust.
  4. Cheap rows keep a draft's full width; rows that each bring cold experts
     are cut where they stop paying.
  5. On a streamed model, prefill records every prompt position, and greedy
     output with replay + truncation on equals plain greedy decode.

Run:
    python tests/test_route_log.py
"""

import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from tests.test_stream import CONFIG, build_tiny_model  # noqa: E402

from pagedmoe import config as es_config  # noqa: E402
from pagedmoe import load, route_log  # noqa: E402
from pagedmoe import verify_decode as vd  # noqa: E402


class _FakeCache:
    def __init__(self, held):
        self.held = held  # layer_key -> set of ids
        self.batches = []
        self._prefetch_clock = 10

    def free_ids(self, lk, ids):
        return {int(e) for e in ids if int(e) in self.held.get(lk, set())}

    def prefetch_many(self, batch):
        self.batches.append(batch)


def check_record_plan():
    log = route_log.RouteLog(["a", "b"], cap=256)
    inds = np.array([[3, 1], [5, 7], [2, 4]])
    w = np.array([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]])
    log.record_tokens(10, [100, 101, 102])
    log.record("a", 10, inds, w)
    log.record("b", 10, inds[:, ::-1], None)
    assert list(log.ids[10 % 256, 0]) == [1, 3], log.ids[10 % 256, 0]
    assert list(log.ids[11 % 256, 0]) == [5, 7]
    plan = log.plan(10, [100, 101, 999])
    assert plan[2] is None, "a position holding another token plans as unknown"
    assert [list(v) for v in plan[0]] == [[1, 3], [1, 3]]
    # Kept count follows decode: one expert a row keeps only the strongest.
    for _ in range(200):
        log.note_kept("a", 1.0)
    plan = log.plan(10, [100, 101])
    assert [list(v) for v in plan[0]] == [[1], [1, 3]], plan[0]
    # The ring forgets positions it overwrote.
    log.record_tokens(10 + 256, [7])
    assert log.plan(10, [100]) is None
    print("OK: rows land by position, strongest first; stale positions plan as unknown")


def check_arm():
    log = route_log.RouteLog(["a", "b"], cap=256)
    log.record_tokens(0, [1, 2, 3])
    log.record("a", 0, np.array([[1, 2], [2, 3], [4, 5]]), None)
    log.record("b", 0, np.array([[1, 6], [1, 7], [1, 2]]), None)
    cache = _FakeCache({"a": {1}, "b": {1}})
    plan = log.plan(0, [1, 2, 3])
    assert log.arm(cache, plan, 2) == 0 and not cache.batches, "untrusted layers read nothing"
    assert log.armed == {0: {2, 3}, 1: {6, 7}}, log.armed
    log.score("a", [2, 3])
    log.end_pass()  # layer b never scored: named, nothing right
    for _ in range(log.MIN_PASSES):
        log.arm(cache, plan, 2)
        log.score("a", [2, 3])
        log.score("b", [9])
        log.end_pass()
    assert list(log.trusted()) == [True, False], log.trusted()
    n = log.arm(cache, plan, 2)
    assert n == 2 and len(cache.batches[-1]) == 1, (n, cache.batches)
    lk, ids, need_at = cache.batches[-1][0]
    assert lk == "a" and ids == [2, 3] and need_at == 11, cache.batches[-1]
    print("OK: every layer is scored; only layers that name cold experts well read ahead")


def check_gate():
    es_config.VERIFY_REPLAY_MIN_PREC = 0.5
    log = route_log.RouteLog(["a"], cap=256)
    for _ in range(log.MIN_PASSES):
        log.armed = {0: {1, 2, 3, 4}}
        log.score("a", [1])
        log.end_pass()
    assert not log.trusted()[0], "25% precision is not trusted"
    for _ in range(log.MIN_PASSES * 4):
        log.armed = {0: {1, 2}}
        log.score("a", [1, 2])
        log.end_pass()
    assert log.trusted()[0], "precise passes earn trust"
    for _ in range(30):
        log.armed = {0: {1, 2}}
        log.end_pass()
    assert not log.trusted()[0], "a layer the pass never reached counts against the plan"
    print("OK: replay is trusted only on measured precision")


def check_row_cost():
    rc = vd._RowCost()
    key = ("ngram", 4)
    ec = type("E", (), {"disk_wait_s": 0.0, "bytes_read": 0})()
    # No reads per row yet: the draft keeps its width on a good bucket.
    assert rc.width(key, 16, 0.15, ec) >= 12
    # Weak bucket: cut once the accept odds fall under the row's compute.
    assert rc.width(("ngram", 0), 8, 0.15, ec) < 8
    # Rows reading ~1.4 GB each (144 ms at 10 GB/s) against a 150 ms plain
    # step: barely the first row pays.
    rc.row_bytes = 1.44e9
    assert rc.width(key, 8, 0.15, ec) <= 1, rc.width(key, 8, 0.15, ec)
    # A bucket measured to accept everything keeps more of its rows.
    for _ in range(20):
        rc.note(key, 8, 8, 1.44e9 * 9, 9)
    assert rc.width(key, 8, 0.3, ec) >= 4, rc.width(key, 8, 0.3, ec)
    # No plain timing yet: leave the draft alone.
    assert rc.width(key, 8, None, ec) == 8
    print("OK: pass width follows row read cost and accept odds")


def check_index_src():
    es_config.LOOKUP_NGRAM_MIN = 3
    es_config.VERIFY_MATCH_CAP = 8
    idx = vd._Index([1, 2, 3, 9, 8, 7, 5, 1, 2, 3])
    assert idx.query(3) == [9, 8, 7]
    assert idx.last_src == 3, idx.last_src
    idx2 = vd._Index([5, 5, 41, 42, 43], [[41, 42, 43, 44, 45]])
    assert idx2.query(2) == [44, 45] and idx2.last_src is None
    print("OK: drafter reports the context position it copies")


def check_streamed():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build_tiny_model(tmp)
        rng = np.random.default_rng(7)
        body = rng.integers(0, CONFIG["vocab_size"], size=24).tolist()
        # A repeated run gives the n-gram drafter context positions to copy.
        prompt = body + body + body[:6]
        n = 24
        old = (es_config.VERIFY_SPANS, es_config.PRUNE, es_config.WAIT_ABOVE)
        es_config.VERIFY_SPANS = "all"
        es_config.PRUNE = 0.0
        es_config.WAIT_ABOVE = 0.0
        try:
            model, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
            log = model._pagedmoe_routes
            assert log is not None, "route log not installed"

            def run(truncate, replay):
                es_config.VERIFY_TRUNCATE = truncate
                es_config.VERIFY_REPLAY = replay
                vd._GOV = vd._Governor()
                cache = make_prompt_cache(model)
                mx.eval(model(mx.array([prompt[:-1]]), cache=cache))
                out = [
                    int(t)
                    for t, _ in vd.generate_step(
                        mx.array(prompt[-1:], dtype=mx.uint32), model,
                        max_tokens=n, prompt_cache=cache, history=prompt[:-1],
                    )
                ]
                return out

            ref = run(False, False)
            p = len(prompt) - 2
            assert log.pos[p] == p and log.tok[p] == prompt[p], (log.pos[p], log.tok[p])
            assert (log.ids[p] >= 0).all(), "prefill left a layer unrecorded"
            got = run(True, True)
            assert got == ref, f"replay/truncate changed greedy output\n ref={ref}\n got={got}"
            assert log.stats["planned"] > 0, "copied context never planned"
            print(f"OK: prefill recorded every layer; replay+truncate keep greedy output "
                  f"(planned={log.stats['planned']} named={log.stats['named']})")
        finally:
            es_config.VERIFY_SPANS, es_config.PRUNE, es_config.WAIT_ABOVE = old
            es_config.VERIFY_TRUNCATE = True
            es_config.VERIFY_REPLAY = True


def main():
    check_record_plan()
    check_arm()
    check_gate()
    check_row_cost()
    check_index_src()
    check_streamed()
    print("\nOK: route log tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
