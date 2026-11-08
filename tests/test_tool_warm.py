# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tool-body expert prefetch: learn during a span, prefetch once the body starts."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pagedmoe import config  # noqa: E402
from pagedmoe import fidelity  # noqa: E402
from pagedmoe.tool_warm import ToolWarm  # noqa: E402


class _Cache:
    def __init__(self):
        self.batches = []

    def prefetch_many(self, batch):
        self.batches.append(list(batch))


def test_body_prefetch_and_hold():
    old = (
        config.SIDECAR_DIR,
        config.TOOL_WARM,
        config.TOOL_WARM_GB,
        config.TOOL_WARM_SKIP,
        config.TOOL_WARM_MIN_TOKENS,
        config.TOOL_WARM_DECAY,
        fidelity._in_tool,
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            config.SIDECAR_DIR = tmp
            config.TOOL_WARM = True
            config.TOOL_WARM_GB = 1.0
            config.TOOL_WARM_SKIP = 2
            config.TOOL_WARM_MIN_TOKENS = 3
            config.TOOL_WARM_DECAY = 1.0
            warm = ToolWarm("m", ["L0", "L1"], 4, expert_bytes=100, slot_id="t")
            cache = _Cache()
            fidelity._in_tool = True
            warm.maybe_warm(cache)
            assert cache.batches == []
            for _ in range(2 + 3):
                warm.note_demand("L0", [1])
                warm.note_demand("L1", [2])
                warm.end_token(cache)
            assert cache.batches, "body did not prefetch"
            flat = {e for _k, ids in cache.batches[-1] for e in ids}
            assert 1 in flat and 2 in flat
            fidelity._in_tool = False
            warm.maybe_warm(cache)
            assert warm._episodes == 1
            assert warm._prec_ema > 0.2
            warm.end_turn()
            assert (Path(tmp) / "t_tool_warm.npz").is_file()
    finally:
        (
            config.SIDECAR_DIR,
            config.TOOL_WARM,
            config.TOOL_WARM_GB,
            config.TOOL_WARM_SKIP,
            config.TOOL_WARM_MIN_TOKENS,
            config.TOOL_WARM_DECAY,
            fidelity._in_tool,
        ) = old


if __name__ == "__main__":
    test_body_prefetch_and_hold()
    print("OK: tool warm")
