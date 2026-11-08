# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import unittest

from pagedmoe.prefix_cache import (
    StablePrefixPin,
    common_prefix_len,
    explicit_bounds,
    message_has_cache_control,
    pick_pin_bound,
    pin_cap,
    strip_cache_markers,
)


class TestPrefixCacheHelpers(unittest.TestCase):
    def test_common_prefix(self):
        self.assertEqual(common_prefix_len([1, 2, 3], [1, 2, 9]), 2)
        self.assertEqual(common_prefix_len([1], [2]), 0)

    def test_explicit_bounds(self):
        self.assertEqual(explicit_bounds({"cache_breaks": [10, 0, "20"]}), [10, 20])
        self.assertEqual(explicit_bounds({"cacheBreaks": [4]}), [4])
        self.assertEqual(explicit_bounds({}), [])

    def test_cache_control_and_markers(self):
        self.assertTrue(message_has_cache_control({"cache_control": {"type": "ephemeral"}}))
        self.assertFalse(message_has_cache_control({"role": "user"}))
        self.assertEqual(strip_cache_markers("hi<|cache_break|>"), "hi")

    def test_pick_pin_bound(self):
        self.assertEqual(pick_pin_bound([20, 400, 20000]), 400)
        self.assertEqual(pick_pin_bound([20]), 0)

    def test_pick_pin_bound_prefers_the_message_head(self):
        self.assertEqual(pick_pin_bound([200, 9000], max_pin=16384, prefer=9000), 9000)
        # Head past the cap: fall back to the earliest usable bound.
        self.assertEqual(pick_pin_bound([200, 9000], max_pin=1024, prefer=9000), 200)

    def test_pin_cap_scales_with_the_prompt(self):
        # A system prompt plus tool schemas is routinely larger than the old
        # flat 8192 cap, which left it unpinnable.
        self.assertEqual(pin_cap(0), 8192)
        self.assertEqual(pin_cap(4000), 8192)
        self.assertEqual(pin_cap(40000), 20000)
        self.assertEqual(pin_cap(200000), 32768)

    def test_pin_lookup_requires_match(self):
        pin = StablePrefixPin()
        self.assertIsNone(pin.lookup("m", [1, 2, 3]))
        # update needs a real cache list; empty cache_len is 0 so bound 64 fails
        self.assertFalse(pin.update("m", list(range(80)), [], 64))


if __name__ == "__main__":
    unittest.main()
