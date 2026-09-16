# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The sidecar's "never slower" guarantee, tested as logic.

The sidecar's headline promise is that turning it on cannot cost throughput.
That promise rests entirely on two objects - NetGovernor, which decides whether
actuation is measurably faster, and ByteLedger, which decides whether a head's
speculative reads are paying for themselves - so both are tested here without a
model, a checkpoint, or a disk.

The noise levels are not arbitrary. Decode rate on real agent traffic swings
+/-45% window to window (tool-call JSON vs prose, cache state, a prefill
landing mid-window), which is far wider than the effect being measured, and an
earlier single-window comparison flip-flopped constantly because of it. Every
case below runs at that noise level.

Run: python tests/test_sidecar_governor.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from expert_stream import config  # noqa: E402
from expert_stream.sidecar.governor import ByteLedger, NetGovernor  # noqa: E402


def simulate(on_tps: float, off_tps: float, noise: float, windows: int, seed: int):
    """Drive a governor through `windows` windows of synthetic decode.

    Returns (decisions, actuating_history). Feeds real per-token dt through
    note_token rather than poking _window_done, so the window accounting and
    the outlier filter are exercised too.
    """
    gov = NetGovernor(name="test")
    rng = random.Random(seed)
    decisions: list[bool | None] = []
    acts: list[bool] = []
    for _ in range(windows):
        active = gov.actuating
        acts.append(active)
        rate = (on_tps if active else off_tps) * rng.uniform(1 - noise, 1 + noise)
        for _ in range(gov.window):
            gov.note_token(1.0 / rate)
        decisions.append(gov._decision)
    return decisions, acts


def flips(decisions: list[bool | None]) -> int:
    seen = [d for d in decisions if d is not None]
    return sum(1 for a, b in zip(seen, seen[1:]) if a != b)


def actuation_share(acts: list[bool], tail: int = 200) -> float:
    window = acts[-tail:]
    return sum(window) / len(window)


def exposure_report() -> None:
    """Print steady-state exposure per regime - the number that matters most.

    "Actuation is off" is not the same as "the sidecar costs nothing": the
    governor still has to sample the losing arm occasionally to notice when the
    answer changes, and every one of those windows runs in the slower
    configuration. This reports what that watching actually costs.
    """
    for name, (on, off) in {
        "regression -24%": (11.0, 14.5),
        "tie": (14.0, 14.0),
        "win +10%": (15.4, 14.0),
        "win +46%": (19.0, 13.0),
    }.items():
        shares = []
        for seed in range(16):
            _, acts = simulate(on, off, 0.45, 1200, seed)
            shares.append(actuation_share(acts, 600))
        print(
            f"    {name:16s} steady-state actuation: "
            f"min {min(shares):.1%} mean {sum(shares) / len(shares):.1%} "
            f"max {max(shares):.1%}"
        )


def main() -> None:
    windows = 800

    # 1. A large regression. Bigger than anything measured here (the old wrap
    #    policy came in at -0.3% on GLM-4.7), and deliberately so: the point of
    #    the governor is that it holds on a workload nobody has profiled, where
    #    the loss is not guaranteed to be small.
    for seed in range(8):
        decisions, acts = simulate(11.0, 14.5, 0.45, windows, seed)
        assert decisions[-1] is False, f"seed {seed}: kept a 24% regression on"
        assert actuation_share(acts) < 0.05, (
            f"seed {seed}: still actuating {actuation_share(acts):.1%} of tokens"
        )
    print("OK: large regression is switched off, exposure under 5%")

    # 2. The regression actually measured on a real bank, which is small enough
    #    to sit inside the margin. It cannot be *identified* as a loss at that
    #    size, so what matters is that it does not get identified as a win:
    #    inconclusive must resolve to off.
    for seed in range(8):
        decisions, _ = simulate(13.96, 14.0, 0.45, windows, seed)
        assert decisions[-1] is False, f"seed {seed}: kept a sub-margin loss on"
    print("OK: sub-margin loss resolves off")

    # 3. A tie must resolve off. Speculation is not free even when it is
    #    harmless on average - it spends bandwidth and adds queue depth - so
    #    "no measurable difference" is not a reason to keep doing it.
    for seed in range(8):
        decisions, acts = simulate(14.0, 14.0, 0.45, windows, seed)
        assert decisions[-1] is False, f"seed {seed}: kept actuation on a tie"
        assert flips(decisions) <= 1, f"seed {seed}: {flips(decisions)} reversals"
    print("OK: tie settles off without oscillating")

    # 4. A real win has to survive the noise, or the guarantee is worthless -
    #    a governor that only ever says no is trivially safe and useless.
    for seed in range(8):
        decisions, acts = simulate(19.0, 13.0, 0.45, windows, seed)
        assert decisions[-1] is True, f"seed {seed}: missed a 46% win"
        assert actuation_share(acts) > 0.75, (
            f"seed {seed}: only actuating {actuation_share(acts):.0%} of a win"
        )
    print("OK: real win is found, held, and actuated on most tokens")

    # 5. A win too small to distinguish from noise is refused. The margin bar
    #    exists so that a 0.5% "win" does not buy a permanent risk surface.
    for seed in range(4):
        decisions, _ = simulate(14.07, 14.0, 0.45, windows, seed)
        assert decisions[-1] is False, f"seed {seed}: accepted a sub-margin win"
    print("OK: sub-margin win is refused")

    # 6. Probing has to be cheap. Half of each probe round runs in the losing
    #    configuration, so a verdict that took an entire agent turn would cost
    #    more than it protects.
    decisions, _ = simulate(19.0, 13.0, 0.45, windows, 0)
    first = next(i for i, d in enumerate(decisions) if d is not None)
    tokens = first * NetGovernor(name="t").window
    assert tokens <= 8 * config.SIDECAR_GOV_MIN_TOKENS, (
        f"first verdict took {tokens} tokens"
    )
    print(f"OK: first verdict after {first} windows ({tokens} tokens)")

    # 7. A verdict must not be permanent - context grows, another model loads,
    #    the workload turns from prose to tool calls. The timed path is
    #    deliberately lazy once a verdict has been confirmed a few times (the
    #    watching is not free), so this is the slow route back.
    def phase(gov, windows, on_tps, off_tps, rng):
        for _ in range(windows):
            rate = (on_tps if gov.actuating else off_tps) * rng.uniform(0.6, 1.4)
            for _ in range(gov.window):
                gov.note_token(1.0 / rate)

    gov = NetGovernor(name="revisit")
    rng = random.Random(0)
    phase(gov, 400, 11.0, 14.5, rng)
    assert gov._decision is False, "did not switch off during the bad phase"
    phase(gov, 4000, 19.0, 13.0, rng)
    assert gov._decision is True, "never re-probed after conditions improved"
    print("OK: verdict is revisited when conditions change (timed path)")

    # 8. ...and the fast route back, which is the one that matters in practice.
    #    A head that trains its way to usefulness can score the reads it would
    #    have issued against the demand that followed - free, because that
    #    demand is reported either way - and ask for a fresh measurement on the
    #    strength of it. Without this, a head that became good would wait tens
    #    of thousands of tokens for a timer.
    gov = NetGovernor(name="request")
    rng = random.Random(1)
    phase(gov, 400, 11.0, 14.5, rng)
    assert gov._decision is False
    before = gov.stats()["duty_period"]
    # A request lands whenever a round is not already in flight; the head
    # retries as it accumulates evidence rather than assuming the first ask
    # succeeds.
    granted = False
    for _ in range(64):
        if gov.request_probe("head says it improved"):
            granted = True
            break
        phase(gov, 1, 11.0, 14.5, rng)
    assert granted, "probe request never accepted"
    assert gov._hold_left == 0, "request did not schedule a probe"
    assert gov.stats()["duty_period"] <= before, "request did not reset the watch"
    phase(gov, 600, 19.0, 13.0, rng)
    assert gov._decision is True, "requested probe did not find the win"
    print("OK: a head can buy a re-measurement with free evidence")

    # 9. ...but it cannot spam it. A noisy head must not be able to turn the
    #    request path into continuous 50/50 probing, which would reintroduce
    #    exactly the cost the governor removes.
    gov = NetGovernor(name="spam")
    rng = random.Random(2)
    phase(gov, 400, 11.0, 14.5, rng)
    granted = sum(1 for _ in range(50) if gov.request_probe("spam"))
    assert granted <= 1, f"{granted} back-to-back probe requests granted"
    print("OK: probe requests are rate-limited")

    # 10. Idle gaps are not decode steps. A pause while the user reads would
    #     otherwise swamp the window it lands in and hand the verdict to
    #     whichever arm happened to be unlucky.
    gov = NetGovernor(name="outlier")
    for _ in range(gov.window * 4):
        gov.note_token(0.05)          # establish the scale
    before = gov._acc[True][3] + gov._acc[False][3]
    for _ in range(gov.window * 4):
        gov.note_token(30.0)          # user went to lunch
    after = gov._acc[True][3] + gov._acc[False][3]
    assert after == before, "idle gaps were counted as decode tokens"
    print("OK: idle gaps are excluded")

    # 11. ...and the filter is relative, not a fixed bound. A checkpoint
    #     streaming 2 GB per token decodes at ~1.4 s/token, which the earlier
    #     fixed one-second cutoff rejected outright - and a governor that
    #     receives no samples never reaches a verdict, so `actuating` keeps
    #     whatever value it started with. Failing open while reporting a
    #     guarantee is worse than not having the mechanism at all.
    gov = NetGovernor(name="slow")
    rng = random.Random(3)
    for _ in range(gov.window * 200):
        gov.note_token(1.4 * rng.uniform(0.8, 1.2))
    sampled = gov._acc[True][3] + gov._acc[False][3]
    assert sampled > 0, "a slow model's tokens were all discarded as outliers"
    assert gov.decided, "no verdict reached on a slow model"
    print(f"OK: slow models are measured, not discarded ({sampled:.0f} tokens)")

    # 12. Disabled governor is transparent: actuation is unconditional and no
    #     samples are kept. This is the A/B-rig path.
    saved = config.SIDECAR_GOVERNOR
    try:
        config.SIDECAR_GOVERNOR = False
        gov = NetGovernor(name="off")
        for _ in range(gov.window * 8):
            gov.note_token(0.05)
        assert gov.actuating, "disabled governor must not gate actuation"
        assert gov._decision is None
    finally:
        config.SIDECAR_GOVERNOR = saved
    print("OK: disabled governor is transparent")

    # --- ByteLedger --------------------------------------------------------
    # The unit that costs. A head is worth running when the bytes it spends
    # come back as bytes something wanted.
    led = ByteLedger(expert_bytes=1_000_000)
    led.note_issued(100)
    led.note_used(90)
    assert led.precision == 0.9
    assert led.net_bytes == 80_000_000, led.net_bytes  # 90 used - 10 wasted
    assert led.window_net_bytes == led.net_bytes

    # The shipped wrap state on GLM-4.7: 85% of issued reads unused. Whatever
    # the wall-clock cost turns out to be on a given drive, spending six bytes
    # to place one is not a trade worth defending, and it has to read as a
    # deficit here even though `gain` called it +0.18.
    led = ByteLedger(expert_bytes=1_000_000)
    led.note_issued(1000)
    led.note_used(150)
    assert led.window_net_bytes < 0, "85% waste must show as a deficit"
    assert round(led.window_precision, 2) == 0.15

    # A window reset judges the head on what it is doing now, not on a debt
    # from before its weights changed - or on savings that would let a good
    # history fund a long bad phase.
    led.reset_window()
    assert led.window_issued == 0 and led.window_net_bytes == 0
    assert led.issued == 1000, "lifetime totals must survive a window reset"
    led.note_issued(100)
    led.note_used(80)
    assert led.window_net_bytes > 0, "recent good behaviour must clear the debt"
    print("OK: byte ledger scores the trade and windows correctly")

    print("steady-state exposure:")
    exposure_report()


if __name__ == "__main__":
    main()
