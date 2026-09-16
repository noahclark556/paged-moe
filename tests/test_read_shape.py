# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prefill must coalesce into long sequential spans, not per-expert reads.

The run cap used to be `max(expert_nbytes, 32 MB)`. On any model whose experts
are larger than 32 MB - which is every big MoE in the catalog - that made the
cap exactly one expert, so prefill issued one independent read per expert
(~15k per pass on DeepSeek-V3.2) and never took the sequential path the run
reader was written for. These tests pin the shape, since the failure mode is
"quietly slow", not "broken".
"""

from __future__ import annotations

import unittest
from unittest import mock

from expert_stream import config
from expert_stream.cache import ExpertCache

# DeepSeek-V3.2 4-bit: ~368 GB of experts over 58 MoE layers x 256 experts.
DEEPSEEK_EXPERT_BYTES = 25 << 20
QWEN_EXPERT_BYTES = 6 << 20


def _cache(expert_bytes):
    """A cache object with only what the run-shape math touches."""
    c = ExpertCache.__new__(ExpertCache)
    c.expert_nbytes = lambda layer_key: expert_bytes
    return c


def _runs(ids, max_run):
    """Mirror of the coalescing loop in ExpertCache._begin."""
    runs = [[ids[0]]]
    for eid in ids[1:]:
        if eid == runs[-1][-1] + 1 and len(runs[-1]) < max_run:
            runs[-1].append(eid)
        else:
            runs.append([eid])
    return runs


class TestRunShape(unittest.TestCase):
    def test_decode_keeps_runs_short(self):
        c = _cache(DEEPSEEK_EXPERT_BYTES)
        self.assertEqual(c._max_run_bytes("l0", prefill=False), 32 << 20)

    def test_prefill_run_spans_many_experts(self):
        c = _cache(DEEPSEEK_EXPERT_BYTES)
        cap = c._max_run_bytes("l0", prefill=True)
        self.assertEqual(cap, config.PREFILL_RUN_BYTES)
        self.assertGreaterEqual(cap // DEEPSEEK_EXPERT_BYTES, 8)

    def test_prefill_reads_a_layer_in_a_handful_of_spans(self):
        """256 consecutive experts should become ~a dozen spans, not 256."""
        c = _cache(DEEPSEEK_EXPERT_BYTES)
        ids = list(range(256))
        before = _runs(ids, max(1, (32 << 20) // DEEPSEEK_EXPERT_BYTES))
        after = _runs(
            ids, max(1, c._max_run_bytes("l0", prefill=True) // DEEPSEEK_EXPERT_BYTES)
        )
        self.assertEqual(len(before), 256)  # the regression: no coalescing at all
        self.assertLess(len(after), 32)
        self.assertEqual(sum(len(r) for r in after), 256)

    def test_small_expert_models_still_coalesce(self):
        c = _cache(QWEN_EXPERT_BYTES)
        max_run = c._max_run_bytes("l0", prefill=True) // QWEN_EXPERT_BYTES
        self.assertGreaterEqual(max_run, 32)

    def test_run_cap_never_below_one_expert(self):
        """A single expert bigger than the cap must still be readable."""
        huge = 4 << 30
        c = _cache(huge)
        self.assertGreaterEqual(c._max_run_bytes("l0", prefill=True), huge)


class TestGroupSizing(unittest.TestCase):
    def test_group_holds_more_than_one_run(self):
        """Runs cannot span groups, so a group smaller than a run wastes it."""
        per_group = config.GROUP_BYTES // DEEPSEEK_EXPERT_BYTES
        per_run = config.PREFILL_RUN_BYTES // DEEPSEEK_EXPERT_BYTES
        self.assertGreaterEqual(per_group, per_run)

    def test_prefill_slices_are_coarser_than_decode(self):
        from expert_stream.cache import _PREFILL_SLICE_BYTES, _SLICE_BYTES

        self.assertGreater(_PREFILL_SLICE_BYTES, _SLICE_BYTES)


class TestFetchGroupsUsesPrefillShape(unittest.TestCase):
    def test_begin_is_called_with_prefill_true(self):
        """fetch_groups is the prefill entry point; decode's fetch is not."""
        c = ExpertCache.__new__(ExpertCache)
        c.expert_nbytes = lambda layer_key: DEEPSEEK_EXPERT_BYTES
        seen = []

        def fake_begin(layer_key, ids, use_slab=False, prefill=False):
            seen.append(prefill)
            return object()

        with mock.patch.object(c, "_begin", fake_begin), mock.patch.object(
            c, "_finish", lambda *a: {}
        ):
            list(c.fetch_groups("l0", list(range(64)), config.GROUP_BYTES))
        self.assertTrue(seen)
        self.assertTrue(all(seen))


if __name__ == "__main__":
    unittest.main()
