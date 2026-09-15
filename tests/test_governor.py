# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The route-prediction governor must survive noisy agent traffic.

Pure logic: no model, no disk. Drives PrefetchRing._window_done with synthetic
windows whose throughput is drawn from a distribution as wide as the one
observed on a real host-app session (9.2 to 19.2 tok/s), and checks that the governor
converges instead of oscillating.

Run: python tests/test_governor.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from expert_stream import config  # noqa: E402
from expert_stream.streaming import PrefetchRing  # noqa: E402


def simulate(on_tps: float, off_tps: float, noise: float, windows: int, seed: int):
    """Return (decision_history, depth_history) after `windows` decode windows."""
    ring = PrefetchRing(cache=object(), depth=8, predict_depth=3, predict_mode="auto")
    rng = random.Random(seed)
    tokens = max(4, config.PREDICT_WINDOW)
    decisions: list[bool | None] = []
    depths: list[int] = []
    for _ in range(windows):
        active = ring.predict_depth > 0
        true_rate = on_tps if active else off_tps
        rate = true_rate * rng.uniform(1.0 - noise, 1.0 + noise)
        depths.append(ring.predict_depth)
        ring._window_done(tokens, tokens / rate)
        decisions.append(ring._decision)
    return decisions, depths


def flips(decisions: list[bool | None]) -> int:
    seen = [d for d in decisions if d is not None]
    return sum(1 for a, b in zip(seen, seen[1:]) if a != b)


def longest_run(depths: list[int], value: int) -> int:
    best = run = 0
    for d in depths:
        run = run + 1 if d == value else 0
        best = max(best, run)
    return best


def main() -> None:
    windows = 600

    # 1. No real difference, very noisy. Must settle off (speculative reads cost
    #    bandwidth, so a tie resolves to not doing them) and stay there.
    for seed in range(8):
        decisions, depths = simulate(14.0, 14.0, 0.45, windows, seed)
        assert decisions[-1] is False, f"seed {seed}: kept prediction on a tie"
        assert flips(decisions) <= 1, f"seed {seed}: {flips(decisions)} reversals on a tie"
        assert longest_run(depths, 0) >= config.PREDICT_HOLD_WINDOWS - 1, (
            f"seed {seed}: decision not held (longest off run {longest_run(depths, 0)})"
        )
    print("OK: noisy tie settles off without oscillating")

    # 2. The measured disk-bound win (10.6 -> 19.0 tok/s), same noise.
    for seed in range(8):
        decisions, _ = simulate(19.0, 10.6, 0.45, windows, seed)
        assert decisions[-1] is True, f"seed {seed}: missed a 79% win"
        assert flips(decisions) <= 1, f"seed {seed}: {flips(decisions)} reversals on a real win"
    print("OK: real win is found and held")

    # 3. The measured anti-win (prediction 7% slower when reads are not the
    #    bottleneck) must be rejected.
    for seed in range(8):
        decisions, _ = simulate(26.0, 28.0, 0.35, windows, seed)
        assert decisions[-1] is False, f"seed {seed}: kept prediction when it was slower"
    print("OK: regression is rejected")

    # 4. Probing is bounded: a decision arrives well inside one agent turn.
    decisions, _ = simulate(19.0, 10.6, 0.45, windows, 0)
    first = next(i for i, d in enumerate(decisions) if d is not None)
    budget = 2 * config.PREDICT_MIN_TOKENS / max(4, config.PREDICT_WINDOW)
    assert first <= budget + 2, f"first decision took {first} windows (budget {budget:.0f})"
    print(f"OK: first decision after {first} windows "
          f"({first * max(4, config.PREDICT_WINDOW)} tokens)")

    # 5. Forced modes bypass the governor entirely. Depths are the ones
    #    patch_model passes: "off" zeroes the depth at the call site.
    for mode, depth in (("on", 3), ("off", 0)):
        ring = PrefetchRing(cache=object(), depth=8, predict_depth=depth, predict_mode=mode)
        assert not ring._auto, f"mode {mode} should not run the governor"
        assert ring.predict_depth == depth
        for _ in range(4 * config.PREDICT_HOLD_WINDOWS):
            ring._window_done(8, 1.0)
        assert ring.predict_depth == depth, f"mode {mode} drifted to {ring.predict_depth}"
        assert ring._decision is None
    print("OK: explicit modes are not governed")


if __name__ == "__main__":
    main()
