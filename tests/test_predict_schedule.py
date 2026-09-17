# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The depth-dependent prediction schedule (PrefetchRing._distances).

Measured router agreement over every (source, target) layer pair shows the
decay tracks SOURCE DEPTH, not distance: on
Qwen3-235B, precision among cache misses from layer 0 is gone by distance 4
(0.28), while from layer 24 it is still 0.73 at distance 32. PREDICT_FAR_LEAD
turns that into one extra long-range prediction per deep layer.

What has to hold, because each of these is a way to silently lose reads or
throughput:

  * off by default - the shipped uniform window is unchanged until someone
    opts in
  * silent while prediction is off, which is how the governor's OFF phase is
    expressed (predict_depth = 0). A far read during OFF would corrupt the
    very A/B that decides whether prediction pays.
  * nothing for early layers, where the matrix says prediction is worthless
  * exactly ONE far target, never a widened window: a wide speculative burst
    queues ahead of real demand misses, which is why uniformly deepening
    measured slower
  * no duplicate of a near-band target, and nothing past the last layer

Run:  python tests/test_predict_schedule.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from expert_stream import config
from expert_stream.streaming import PrefetchRing


class _Ring:
    """Just the scheduler. Building a real ring needs a cache and a model."""

    def __init__(self, depth, lead=0):
        self.predict_depth = depth
        self.lead = lead

    _distances = PrefetchRing._distances


def test_default_is_the_shipped_uniform_window():
    assert config.PREDICT_FAR_LEAD == 0, "far lead must ship off"
    ring = _Ring(depth=3)
    for src in (0, 8, 40):
        assert ring._distances(src, 94) == (2, 3, 4), (
            f"src {src}: default schedule changed"
        )


def test_far_target_is_added_only_past_the_early_layers():
    ring = _Ring(depth=3)
    n = 94
    cut = int(0.17 * n)  # 15
    with _far(16, 0.17):
        assert ring._distances(cut - 1, n) == (2, 3, 4), (
            "early layers must not far-predict: the matrix measures 0.13 "
            "miss-precision at distance 8 from layer 0"
        )
        assert ring._distances(cut, n) == (2, 3, 4, 16)
        assert ring._distances(40, n) == (2, 3, 4, 16)


def test_far_band_is_one_target_not_a_window():
    """A wide burst is the known-slower failure mode, so guard the count."""
    ring = _Ring(depth=3)
    with _far(32, 0.17):
        got = ring._distances(24, 94)
        assert len(got) == 4, f"expected near band + 1, got {got}"
        assert got[-1] == 32


def test_prediction_off_means_off():
    """The governor expresses OFF as predict_depth = 0 (streaming.py:736)."""
    ring = _Ring(depth=0)
    with _far(32, 0.0):
        assert ring._distances(40, 94) == (), (
            "far lead leaked a read while prediction was off - this would "
            "corrupt the prediction governor's own A/B measurement"
        )


def test_no_duplicates_and_nothing_past_the_last_layer():
    ring = _Ring(depth=3)
    with _far(4, 0.0):
        got = ring._distances(20, 94)
        assert got == (2, 3, 4), f"far duplicated a near target: {got}"
    with _far(32, 0.0):
        # src 80 + 32 = 112, past the end; the near band still has room.
        assert ring._distances(80, 94) == (2, 3, 4)
        got = ring._distances(91, 94)
        assert all(91 + d < 94 for d in got), f"ran past the stack: {got}"


class _far:
    def __init__(self, lead, frm):
        self.lead, self.frm = lead, frm

    def __enter__(self):
        self._old = (config.PREDICT_FAR_LEAD, config.PREDICT_FAR_FROM)
        config.PREDICT_FAR_LEAD = self.lead
        config.PREDICT_FAR_FROM = self.frm

    def __exit__(self, *a):
        config.PREDICT_FAR_LEAD, config.PREDICT_FAR_FROM = self._old
        return False


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nOK: {len(tests)} prediction-schedule tests passed")


if __name__ == "__main__":
    main()
