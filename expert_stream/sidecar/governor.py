# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Net-time governor: the sidecar may only actuate while it measurably helps.

Alternates actuation on/off in short windows, pools decode tok/s on each
side, and settles on only when the gap clears a margin and an error bar.
Ties settle off. Heads keep training either way; only actuation spends disk.

Same idea as the route-prediction governor in streaming.py.
"""

from __future__ import annotations

import time

from .. import config
from .slot import _log

# Multiples of the minimum sample the *first* probe round may extend to before
# calling a tie. Small: the first round is the only one that runs 50/50, so its
# samples are the expensive ones, and a tie there costs nothing to get wrong.
_FIRST_ROUND_PROBE = 4

# Longest gap accepted as a decode token before any scale is known. Generous on
# purpose: the alternative to accepting one stray sample is never measuring.
_COLD_MAX_DT = 5.0
# Once warm, a gap this many times the running token time is an idle gap, not a
# slow token. Context growth and cache pressure move token time by factors well
# under this; a user pausing moves it by orders of magnitude.
_OUTLIER_MULT = 8.0


class NetGovernor:
    """A/B's sidecar actuation against itself and keeps only measured wins.

    Fed one sample per decode token (the wall time since the previous token).
    Owns a single boolean, :attr:`actuating`, that every head consults before
    it spends a byte of disk bandwidth.
    """

    def __init__(self, *, name: str = "sidecar"):
        self.name = name
        self.enabled = bool(config.SIDECAR_GOVERNOR)
        self.window = max(4, int(config.SIDECAR_GOV_WINDOW))
        self.min_tokens = max(self.window, int(config.SIDECAR_GOV_MIN_TOKENS))
        self.margin = max(1.0, float(config.SIDECAR_GOV_MARGIN))
        self.z = float(config.SIDECAR_GOV_Z)
        # How far past the minimum sample to keep probing an inconclusive
        # result before calling it a tie. A multiple, not a share.
        self.max_probe = max(1.0, float(config.SIDECAR_GOV_MAX_PROBE))
        # Config is in tokens (the unit a reader thinks in); accounting is in
        # windows (the unit a verdict is made in).
        hold_tokens = max(self.window, int(config.SIDECAR_GOV_HOLD))
        self.hold_windows = max(1, hold_tokens // self.window)
        self.probe_duty = max(2, int(config.SIDECAR_GOV_PROBE_DUTY))

        # When the governor is off the sidecar actuates unconditionally, which
        # is the old behaviour and is only appropriate for A/B rigs.
        self._actuate = True
        self._probing = self.enabled
        self._decision: bool | None = None
        self._hold_left = 0
        # Per state: [windows, sum(rate), sum(rate^2), tokens]
        self._acc: dict[bool, list[float]] = {
            True: [0.0, 0.0, 0.0, 0.0],
            False: [0.0, 0.0, 0.0, 0.0],
        }
        self._win_tokens = 0
        self._win_time = 0.0
        self._next_check = float(self.min_tokens)
        self._duty = 0
        self._confirms = 0
        self._last_request = 0.0
        self._requests = 0
        self._last: float | None = None
        self._settled_at: float | None = None
        self._flips = 0
        self._dt_ema = 0.0

    # ----------------------------------------------------------------- state

    @property
    def actuating(self) -> bool:
        """Whether heads may spend disk bandwidth on this token."""
        return self._actuate if self.enabled else True

    @property
    def decided(self) -> bool:
        return self._decision is not None

    def rates(self) -> tuple[float, float]:
        on, _, _ = _mean_var(self._acc[True])
        off, _, _ = _mean_var(self._acc[False])
        return on, off

    # ------------------------------------------------------------- collection

    def note_token(self, dt: float | None = None) -> None:
        """Record one decode token. `dt` defaults to wall time since the last.

        Idle gaps - the user reading, a tool call running, a turn ending - are
        not decode steps, and one of them landing in a window would hand the
        verdict to whichever arm happened to be unlucky. So they are filtered
        out, but *relative* to how fast this model actually decodes rather than
        against a fixed bound.

        That distinction is load-bearing. An earlier version rejected anything
        over one second, which is wrong in both directions: on a 40 tok/s model
        it accepts 4-second idle gaps as tokens, and on a checkpoint streaming
        2 GB per token it rejects every real sample - after which the governor
        never reaches a verdict and `actuating` stays at its initial value
        forever. A safety mechanism whose failure mode is "silently stops
        measuring and leaves actuation on" is worse than none, because it
        reports a guarantee it is not providing.
        """
        if not self.enabled:
            return
        now = time.perf_counter()
        if dt is None:
            prev, self._last = self._last, now
            if prev is None:
                return
            dt = now - prev
        else:
            self._last = now
        if dt <= 0.0:
            return
        if self._dt_ema <= 0.0:
            # Cold: no scale yet. Accept anything that could plausibly be a
            # decode token, and let the EMA take over from the next one.
            if dt > _COLD_MAX_DT:
                return
            self._dt_ema = dt
        elif dt > _OUTLIER_MULT * self._dt_ema:
            return
        else:
            self._dt_ema += 0.02 * (dt - self._dt_ema)
        self._win_tokens += 1
        self._win_time += dt
        if self._win_tokens < self.window:
            return
        tokens, seconds = self._win_tokens, self._win_time
        self._win_tokens, self._win_time = 0, 0.0
        if seconds > 0.0:
            self._window_done(tokens, seconds)

    def _window_done(self, tokens: float, seconds: float) -> None:
        if not self._probing:
            self._hold_left -= 1
            if self._hold_left <= 0:
                # Re-probe, keeping half the evidence: conditions drift slowly
                # (context length, cache pressure, which model is loaded), so
                # resuming from what was already learned reaches the next
                # decision sooner without discarding a well-sampled result.
                for totals in self._acc.values():
                    for i in range(len(totals)):
                        totals[i] *= 0.5
                self._probing = True
                self._actuate = True
                # The retained evidence already sits past the first
                # checkpoint, so schedule the next verdict beyond it -
                # otherwise the round would conclude on old samples before
                # this one's conditions have been observed at all.
                carried = min(self._acc[True][3], self._acc[False][3])
                self._next_check = min(
                    self.min_tokens * self.max_probe,
                    max(self.min_tokens, 2.0 * carried),
                )
            return

        state = self._actuate
        acc = self._acc[state]
        rate = tokens / seconds
        acc[0] += 1.0
        acc[1] += rate
        acc[2] += rate * rate
        acc[3] += tokens
        self._actuate = self._next_state(state)

        on_tok, off_tok = self._acc[True][3], self._acc[False][3]
        # Test at geometric checkpoints, not every window. Re-running the same
        # comparison after each window is a multiple-comparisons trap: at a
        # 3-sigma bar and ~50 looks per round, a true tie produces a false
        # "actuation helps" verdict most rounds, purely from the number of
        # chances taken. Doubling checkpoints bound a round to ~3 looks, which
        # keeps the per-round error near the per-look error.
        if min(on_tok, off_tok) < self._next_check:
            return
        # The first round is deliberately short-sighted. It runs at 50/50, so
        # every extra sample it takes is a sample spent in a configuration that
        # may be slower - and a quick inconclusive verdict is safe, because
        # inconclusive resolves to "off". Later rounds sample the losing side
        # one window in many, so evidence there is cheap and the cap can be far
        # higher; that is what lets a genuine 10% win be resolved at all, since
        # an effect that size needs many more windows than a 46% one to clear
        # the error bar.
        cap = self.min_tokens * (
            self.max_probe if self._decision is not None else _FIRST_ROUND_PROBE
        )
        self._next_check = min(cap, max(self._next_check * 2, self.min_tokens))

        on, on_var, on_n = _mean_var(self._acc[True])
        off, off_var, off_n = _mean_var(self._acc[False])
        if on_n < 2 or off_n < 2:
            return

        stderr = ((on_var / on_n) + (off_var / off_n)) ** 0.5
        gap = on - off
        margin = (self.margin - 1.0) * off
        # Asymmetric on purpose, and this is the crux of the guarantee. The two
        # errors are not symmetric in consequence: wrongly deciding "off" costs
        # an opportunity, wrongly deciding "on" costs the user throughput on
        # every subsequent token. So switching ON must clear the full bar,
        # while switching OFF needs only half of it - cheap to stop, expensive
        # to start.
        need_on = max(margin, self.z * stderr)
        need_off = 0.5 * self.z * stderr
        sample = f"{on_tok:.0f}/{off_tok:.0f} tokens, +/-{self.z * stderr:.2f} tok/s"

        if gap > need_on:
            self._settle(True, f"{on:.2f} vs {off:.2f} tok/s ({sample})")
            return
        if -gap > need_off:
            self._settle(False, f"{on:.2f} vs {off:.2f} tok/s ({sample})")
            return
        # Inconclusive. Give it room, but not unbounded room - if the effect is
        # still inside the noise after several times the minimum sample there
        # is little to win either way, so take the side that spends no disk,
        # and hold that much longer before paying to ask again.
        if min(on_tok, off_tok) < cap:
            return
        self._settle(
            False,
            f"no measurable difference ({on:.2f} vs {off:.2f} tok/s, {sample})",
            hold_mult=4,
        )

    def _next_state(self, state: bool) -> bool:
        """Which configuration the next probe window runs in.

        The first round alternates evenly: there is no verdict yet, so neither
        side is known to be the expensive one and a fast, unbiased answer is
        worth the exposure. It is bounded - one round, then a verdict.

        Every round after that is deliberately lopsided. A 50/50 re-probe on a
        configuration already measured slower spends a quarter of all tokens in
        the state we are trying to avoid, which can easily exceed the
        regression we set out to remove. So the losing side is sampled one
        window in `probe_duty` instead: enough to notice if the answer has
        changed, cheap enough that the watching costs little. Verdicts take
        proportionally longer to revise, which is the right trade - what makes
        the answer change (context length, cache pressure, which model is
        loaded) moves on the scale of minutes, not windows.
        """
        if self._decision is None:
            return not state
        winner = self._decision
        self._duty += 1
        return (not winner) if self._duty % self._duty_period() == 0 else winner

    def _duty_period(self) -> int:
        """How many windows of the winning side per window of the other.

        Widens each time a round confirms the standing verdict: after several
        agreeing rounds, paying full price to re-ask a question that keeps
        getting the same answer is itself a regression. Any flip resets it.

        The ceiling depends on which way the verdict went, for the same reason
        the decision bars are asymmetric. While actuation is OFF, being slow to
        notice a missed opportunity costs nothing that the user can measure, so
        the watch can get very cheap - and heads have a zero-bandwidth way to
        ask for attention anyway (see `request_probe`). While actuation is ON,
        the standing verdict is the one that can silently become wrong and take
        throughput with it, so it stays under closer watch even though that
        means occasionally giving up a win we already measured.
        """
        ceiling = 3 if self._decision is False else 1
        return self.probe_duty * (2 ** min(self._confirms, ceiling))

    def _settle(self, on: bool, reason: str, *, hold_mult: int = 1) -> None:
        self._probing = False
        self._actuate = on
        self._hold_left = self.hold_windows * hold_mult
        self._settled_at = time.perf_counter()
        if on == self._decision:
            self._confirms += 1
        else:
            if self._decision is not None:
                self._flips += 1
            self._confirms = 0
            self._decision = on
            _log(f"governor {self.name} actuation {'ON' if on else 'OFF'}: {reason}")

    def request_probe(self, reason: str = "") -> bool:
        """Ask for a re-measure from free shadow evidence.

        A non-actuating head can still score the reads it would have issued
        against later demand (zero bandwidth). When that precision clears the
        bar it was switched off for, call this so the governor spends probe
        windows on a hypothesis with evidence, not a timer.

        No-op while actuation is already on. Rate-limited so noise cannot force
        continuous 50/50 probing. Resets a widened watch to a close one.
        """
        if not self.enabled or self._decision is not False:
            return False
        seen = self._acc[True][3] + self._acc[False][3]
        if seen - self._last_request < self.min_tokens:
            return False
        self._last_request = seen
        self._requests += 1
        # The watch had widened because nothing was changing, and the head is
        # reporting that something has. Back to a close watch.
        self._confirms = 0
        self._hold_left = 0
        self._probing = True
        self._actuate = True
        _log(f"governor {self.name} re-probe requested: {reason}")
        return True

    # ------------------------------------------------------------------ stats

    def stats(self) -> dict:
        on, off = self.rates()
        return {
            "enabled": self.enabled,
            "actuating": self.actuating,
            "probing": self._probing,
            "decision": self._decision,
            "tok_s_on": round(on, 3),
            "tok_s_off": round(off, 3),
            "tokens_on": int(self._acc[True][3]),
            "tokens_off": int(self._acc[False][3]),
            "flips": self._flips,
            "confirms": self._confirms,
            "probe_requests": self._requests,
            "duty_period": self._duty_period() if self.decided else 2,
        }

    def tick(self) -> str:
        on, off = self.rates()
        if not self.enabled:
            return "gov=off"
        if self._probing:
            return f"gov=probe({on:.1f}/{off:.1f})"
        return f"gov={'ON' if self._actuate else 'off'}({on:.1f}/{off:.1f})"


def _mean_var(acc: list[float]) -> tuple[float, float, float]:
    n, total, total_sq = acc[0], acc[1], acc[2]
    if n < 2:
        return (total / n if n else 0.0), float("inf"), n
    mean = total / n
    var = max(0.0, (total_sq - total * mean) / (n - 1))
    return mean, var, n


class ByteLedger:
    """Per-head speculative byte accounting (the unit that actually costs).

    On bandwidth-bound decode, value is bytes saved minus bytes spent:

      issued   speculative bytes requested
      used     of those, bytes a layer later computed with
      wasted   issued - used (bandwidth taken from demand misses)

    `net_bytes` is signed on purpose: deficit means stop actuating. `gain`
    cannot say that. Decisions use the *window*, not lifetime totals - else
    early learning debt or a long good history both distort the call. Call
    `reset_window` when the head's configuration changes.
    """

    __slots__ = (
        "expert_bytes",
        "issued",
        "issued_bytes",
        "used",
        "used_bytes",
        "w_issued",
        "w_issued_bytes",
        "w_used",
        "w_used_bytes",
    )

    def __init__(self, expert_bytes: int = 0):
        self.expert_bytes = int(expert_bytes)
        self.issued_bytes = 0
        self.used_bytes = 0
        self.issued = 0
        self.used = 0
        self.w_issued_bytes = 0
        self.w_used_bytes = 0
        self.w_issued = 0
        self.w_used = 0

    def note_issued(self, n: int, expert_bytes: int | None = None) -> None:
        per = int(expert_bytes if expert_bytes is not None else self.expert_bytes)
        n, nb = int(n), int(n) * per
        self.issued += n
        self.issued_bytes += nb
        self.w_issued += n
        self.w_issued_bytes += nb

    def note_used(self, n: int, expert_bytes: int | None = None) -> None:
        per = int(expert_bytes if expert_bytes is not None else self.expert_bytes)
        n, nb = int(n), int(n) * per
        self.used += n
        self.used_bytes += nb
        self.w_used += n
        self.w_used_bytes += nb

    def reset_window(self) -> None:
        """Start a fresh judgement window; lifetime totals are untouched."""
        self.w_issued_bytes = 0
        self.w_used_bytes = 0
        self.w_issued = 0
        self.w_used = 0

    @property
    def window_issued(self) -> int:
        return self.w_issued

    @property
    def window_precision(self) -> float:
        return (self.w_used / self.w_issued) if self.w_issued else 0.0

    @property
    def window_net_bytes(self) -> int:
        wasted = max(0, self.w_issued_bytes - self.w_used_bytes)
        return self.w_used_bytes - wasted

    @property
    def wasted_bytes(self) -> int:
        return max(0, self.issued_bytes - self.used_bytes)

    @property
    def precision(self) -> float:
        return (self.used / self.issued) if self.issued else 0.0

    @property
    def net_bytes(self) -> int:
        """Useful bytes minus wasted bytes.

        A correct speculative read costs the same bytes the demand miss would
        have cost, so it is not a saving in *bytes* - it is a saving in
        *latency*, because the read overlaps work instead of blocking a layer.
        Charging it at par against waste is therefore conservative, which is
        what we want from a guard: a head at net zero is not worth its risk.
        """
        return self.used_bytes - self.wasted_bytes

    def issued_mb_str(self) -> str:
        """Compact tick-log form: issued MB, precision, net MB."""
        return (
            f"{self.issued_bytes / 1e6:.0f}mb/p{self.precision:.2f}"
            f"/net{self.net_bytes / 1e6:+.0f}mb"
        )

    def stats(self) -> dict:
        return {
            "issued": self.issued,
            "used": self.used,
            "precision": round(self.precision, 4),
            "issued_mb": round(self.issued_bytes / 1e6, 1),
            "wasted_mb": round(self.wasted_bytes / 1e6, 1),
            "net_mb": round(self.net_bytes / 1e6, 1),
            "window_issued": self.w_issued,
            "window_precision": round(self.window_precision, 4),
            "window_net_mb": round(self.window_net_bytes / 1e6, 1),
        }
