# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
A slab left small by pressure grows back at the start of a turn once the
machine is calm (config.SLAB_REGROW), and only then.

Output must not move: generation after a regrow matches the resident model
token for token. The gates must hold: no regrow under WARN, before the calm
window, without room, or twice inside one window. A failed allocation at the
bigger size falls back to the old size instead of losing the slab.

Run:
    python tests/test_slab_regrow.py
"""

import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_stream import CONFIG, build_tiny_model, greedy_generate  # noqa: E402

from pagedmoe import cache as es_cache  # noqa: E402
from pagedmoe import config as es_config  # noqa: E402
from pagedmoe import load, mempressure  # noqa: E402
from pagedmoe import slab as es_slab  # noqa: E402
from pagedmoe.streaming import StreamedSwitchGLU  # noqa: E402


def cache_of(model):
    for m in model.modules():
        if isinstance(m, StreamedSwitchGLU):
            return m.cache
    raise AssertionError("no streamed layer")


def shrink(c, slots):
    """What a pressure step-down leaves behind: same slab, fewer slots."""
    c._release_slab()
    with c._lock:
        specs, _ = c._slab_pending
        assert c._build_slab_locked(specs, slots)


class Machine:
    """Stand-in for the kernel's view, so the gates can be driven directly."""

    def __init__(self):
        self.level, self.calm, self.avail = mempressure.NORMAL, 1e9, 1 << 40

    def install(self):
        mempressure.level = lambda: self.level
        mempressure.calm_s = lambda: 0.0 if self.level >= mempressure.WARN else self.calm
        mempressure.available_bytes = lambda: self.avail


def main():
    saved = (
        es_config.SLAB, mempressure.level, mempressure.calm_s,
        mempressure.available_bytes, es_cache._REGROW_MIN_BYTES,
    )
    es_config.SLAB = "on"
    es_cache._REGROW_MIN_BYTES = 0
    m = Machine()
    m.install()
    rng = np.random.default_rng(11)
    prompt = rng.integers(0, CONFIG["vocab_size"], size=40).tolist()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            build_tiny_model(tmp)
            ref_model, _ = load(str(tmp), mode="resident", verbose=False)
            ref = greedy_generate(ref_model, prompt, 10)
            del ref_model

            model, _ = load(str(tmp), mode="streamed", cache_gb=0.01, verbose=False)
            c = cache_of(model)
            full = c._slab_full_slots
            small = full // 2
            assert full >= 64, f"slab too small to test ({full})"

            # Gates: each one alone keeps the small slab.
            for label, setup in (
                ("warn", lambda: setattr(m, "level", mempressure.WARN)),
                ("not calm yet", lambda: setattr(m, "calm", 1.0)),
                ("no room", lambda: setattr(m, "avail", 0)),
            ):
                m.level, m.calm, m.avail = mempressure.NORMAL, 1e9, 1 << 40
                shrink(c, small)
                setup()
                got = greedy_generate(model, prompt, 10)
                assert c.slab.slots == small, f"{label}: regrew to {c.slab.slots}"
                assert got == ref, f"{label}: tokens moved"
                print(f"{label}: stays at {small}")
            m.level, m.calm, m.avail = mempressure.NORMAL, 1e9, 1 << 40

            # Calm and room: grows back on the next turn, output unchanged.
            shrink(c, small)
            got = greedy_generate(model, prompt, 10)
            assert c.slab.slots == full, f"did not regrow ({c.slab.slots}/{full})"
            assert got == ref, f"regrow moved tokens\n ref={ref}\n got={got}"
            assert c._pressure.scale >= 1.0 - 1e-9
            print(f"regrow: {small} -> {c.slab.slots}, tokens identical")

            # Once per calm window.
            shrink(c, small)
            greedy_generate(model, prompt, 4)
            assert c.slab.slots == small, "regrew twice inside one window"
            c._regrow_after = 0.0

            # Room vanishes at allocation time: back to the old size.
            real = es_slab.SlabStore
            calls = []

            def flaky(specs, slots):
                calls.append(slots)
                if slots > small:
                    raise RuntimeError("no room")
                return real(specs, slots)

            es_cache.SlabStore = flaky
            try:
                got = greedy_generate(model, prompt, 10)
            finally:
                es_cache.SlabStore = real
            assert calls[:2] == [full, small], calls
            assert c.slab is not None and c.slab.slots == small
            assert got == ref, "fallback moved tokens"
            print("failed grow: kept the old size")

            # A rebuild leaves the next turn's KV growth free, or that turn's
            # make_room_for releases the slab again.
            c._regrow_after = 0.0
            assert c._slots_that_fit(full) == full
            c.kv_room_bytes = c._ram_budget_bytes
            try:
                assert c._slots_that_fit(full) < full, "rebuild ignored the KV room"
                assert c._regrow_target(0) == 0, "regrow ignored the KV room"
            finally:
                c.kv_room_bytes = 0
            print("rebuild and regrow leave the KV room")
            del model
    finally:
        (
            es_config.SLAB, mempressure.level, mempressure.calm_s,
            mempressure.available_bytes, es_cache._REGROW_MIN_BYTES,
        ) = saved
    mx.clear_cache()
    print("all ok")


if __name__ == "__main__":
    main()
