# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The last GPU work the engine queued, so a hang names its own trigger.

A stuck Metal wait only shows the wait in the stacks, not what was sent.
Call sites `mark()` rare work as they queue it (prefill chunk, snapshot,
restore, slab rebuild, prefetch) and `step()` the per-token passes (plain
step, verify pass, self draft). They are kept apart so a few hundred tokens
of steps cannot push the prefetch or slab change before them out. Every MoE
layer fetch records its layer and when, since a fetch has to read the
router's output first and so the last layer recorded is about how far the
GPU got. `probe()` adds a line of live state (cache reads in flight, GPU
crumbs) that is read when the trail is printed.

The watchdog prints this on a stall and copies it to a small file every
poll, so it also survives a hang that takes the whole Mac down. Marks are
a tuple append; nothing here syncs, allocates on the GPU or does I/O.
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from typing import Callable

_marks: deque[tuple[float, str]] = deque(maxlen=60)
_steps: deque[tuple[float, str]] = deque(maxlen=40)
_layers: deque[tuple[float, str, bool]] = deque(maxlen=10)
_probes: dict[str, Callable[[], object]] = {}
_seq = 0
_written = -1
_LAYER_NO = re.compile(r"layers\.(\d+)\.")


def mark(what: str) -> None:
    """Rare work: prefill chunks, snapshots, restores, slab changes, prefetches."""
    global _seq
    _marks.append((time.monotonic(), what))
    _seq += 1


def step(what: str) -> None:
    """Per-token work: plain steps, verify passes, self drafts."""
    global _seq
    _steps.append((time.monotonic(), what))
    _seq += 1


def layer(key: str, prefill: bool) -> None:
    global _seq
    _layers.append((time.monotonic(), key, prefill))
    _seq += 1


def probe(name: str, fn: Callable[[], object]) -> None:
    """Register live state for the report. `fn` runs on the watchdog thread
    while the engine may be hung, so it must not take a lock or touch MLX."""
    _probes[name] = fn


def _short(key: str) -> str:
    m = _LAYER_NO.search(key)
    return f"L{m.group(1)}" if m else key


def report(now: float | None = None) -> list[str]:
    """Newest last, each with how long ago it was queued."""
    now = time.monotonic() if now is None else now
    work = sorted(list(_marks) + list(_steps))
    lines = [f"{now - t:7.1f}s ago  {what}" for t, what in work]
    layers = list(_layers)
    if layers:
        gaps = [f"{_short(k)} +{(t - p) * 1000:.0f}ms" for (p, _, _), (t, k, _) in zip(layers, layers[1:])]
        if gaps:
            lines.append(f"{'':12} layer fetch gaps, newest last: {', '.join(gaps)}")
        t, key, prefill = layers[-1]
        lines.append(f"{now - t:7.1f}s ago  last MoE layer fetched: {key}"
                     f" ({'prefill' if prefill else 'decode'})")
    for name, fn in list(_probes.items()):
        try:
            got = fn()
        except Exception as e:  # noqa: BLE001
            got = f"{name}: probe failed ({type(e).__name__}: {e})"
        for line in [got] if isinstance(got, str) else list(got or []):
            lines.append(f"{'':12} {line}")
    return lines


def write(path: str, force: bool = False) -> None:
    """Copy the trail to `path` when it changed since the last copy."""
    global _written
    if _seq == _written and not force:
        return
    seq = _seq
    lines = report()
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as f:
            f.write(f"pid {os.getpid()} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, path)
        _written = seq
    except OSError:
        pass


def rotate(path: str) -> None:
    """Keep the previous engine's trail as `<path>.prev` (a crash leaves it)."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            os.replace(path, f"{path}.prev")
    except OSError:
        pass
