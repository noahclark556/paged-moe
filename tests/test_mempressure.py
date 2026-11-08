# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Memory-pressure slab sizing (mempressure.py, ExpertCache._pressure_slots).

The kernel level has to read as one of 1/2/4 on macOS, the governor has to
step down under pressure, stop at the floor and grow back once pressure
clears, and a rebuild must never ask for more slots than it was sized for.

Run:
    python tests/test_mempressure.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pagedmoe import config, mempressure  # noqa: E402
from pagedmoe import cache as cache_mod  # noqa: E402
from pagedmoe.cache import ExpertCache  # noqa: E402


def test_level():
    lv = mempressure.level()
    if sys.platform == "darwin":
        assert lv in (1, 2, 4), lv
        assert mempressure.compressor_gb() >= 0.0
    else:
        assert lv == 0, lv
    print(f"OK: kernel pressure level {lv}")


def _calm(s=1e9, comp=1.0):
    """Patch a long calm and a flat compressor so NORMAL may grow."""
    return mock.patch.multiple(
        mempressure, calm_s=lambda: s, compressor_gb=lambda: comp
    )


def test_governor():
    with mock.patch.object(config, "MEM_PRESSURE_STEP", 0.15), mock.patch.object(
        config, "MEM_PRESSURE_FLOOR", 0.5
    ), _calm():
        g = mempressure.SlabGovernor()
        assert g.step(mempressure.NORMAL) == 1.0, "normal must not grow past 1"
        assert abs(g.step(mempressure.WARN) - 0.85) < 1e-9
        assert abs(g.step(mempressure.CRITICAL) - 0.55) < 1e-9
        assert g.step(mempressure.CRITICAL) == 0.5, "floor not held"
        assert abs(g.step(mempressure.NORMAL) - 0.575) < 1e-9
        assert abs(g.step(0) - 0.575) < 1e-9, "unknown level must hold"
        for _ in range(10):
            g.step(mempressure.NORMAL)
        assert g.scale == 1.0
    print("OK: governor steps, floor, regrow")


def test_governor_waits_for_calm_and_compressor():
    """Normal right after a warn must not grow, nor while the compressor rose."""
    with mock.patch.object(config, "MEM_PRESSURE_STEP", 0.15), mock.patch.object(
        config, "MEM_PRESSURE_FLOOR", 0.5
    ), mock.patch.object(config, "SLAB_REGROW_CALM_S", 60.0):
        g = mempressure.SlabGovernor()
        with _calm(s=0.0, comp=8.9):
            assert abs(g.step(mempressure.WARN) - 0.85) < 1e-9
        with _calm(s=30.0, comp=7.9):
            assert abs(g.step(mempressure.NORMAL) - 0.85) < 1e-9, "grew before calm"
        with _calm(s=120.0, comp=15.4):
            assert abs(g.step(mempressure.NORMAL) - 0.85) < 1e-9, "grew while compressing"
        with _calm(s=120.0, comp=9.2):
            assert abs(g.step(mempressure.NORMAL) - 0.925) < 1e-9
            for _ in range(4):
                g.step(mempressure.NORMAL)
            assert g.scale == 1.0
            assert g._compressor_gb is None, "full size must forget the shrink"
    print("OK: governor regrow waits for calm and a flat compressor")


def _fake(avail_slots=None):
    ns = SimpleNamespace(_pressure=mempressure.SlabGovernor())
    ns._system_slots = (
        (lambda w, held=0: w) if avail_slots is None
        else (lambda w, held=0: min(w, avail_slots))
    )
    return ns


def test_pressure_slots_grow_into_free_memory_only():
    fake = _fake(avail_slots=800)
    seq = iter([mempressure.WARN, mempressure.WARN, mempressure.NORMAL])
    with mock.patch.object(mempressure, "take_worst", lambda: next(seq)), _calm():
        a = ExpertCache._pressure_slots(fake, 1000)
        b = ExpertCache._pressure_slots(fake, 1000)
        c = ExpertCache._pressure_slots(fake, 1000)
    assert (a, b, c) == (850, 700, 775), (a, b, c)
    fake = _fake(avail_slots=720)
    seq = iter([mempressure.WARN, mempressure.WARN, mempressure.NORMAL])
    with mock.patch.object(mempressure, "take_worst", lambda: next(seq)), _calm():
        ExpertCache._pressure_slots(fake, 1000)
        ExpertCache._pressure_slots(fake, 1000)
        c = ExpertCache._pressure_slots(fake, 1000)
    assert c == 720, c
    assert abs(fake._pressure.scale - 0.72) < 1e-9
    fake = _fake(avail_slots=10)
    seq = iter([mempressure.WARN, mempressure.NORMAL])
    with mock.patch.object(mempressure, "take_worst", lambda: next(seq)), _calm():
        ExpertCache._pressure_slots(fake, 1000)
        c = ExpertCache._pressure_slots(fake, 1000)
    assert c == 850, "growth cap must never shrink the slab"
    print("OK: rebuild grows into free memory only")


def test_pressure_slots():
    fake = _fake()
    seq = iter([mempressure.WARN, mempressure.WARN, mempressure.NORMAL, 0])
    with mock.patch.object(mempressure, "take_worst", lambda: next(seq)), _calm():
        a = ExpertCache._pressure_slots(fake, 1000)
        b = ExpertCache._pressure_slots(fake, 1000)
        c = ExpertCache._pressure_slots(fake, 1000)
        d = ExpertCache._pressure_slots(fake, 1000)
    assert (a, b, c) == (850, 700, 775), (a, b, c)
    assert d == 1000, "no reading must leave the wanted size alone"

    fake = _fake()
    with mock.patch.object(mempressure, "take_worst", lambda: mempressure.CRITICAL):
        for _ in range(8):
            n = ExpertCache._pressure_slots(fake, 40)
    assert n == 32, n

    fake = _fake()
    with mock.patch.object(config, "MEM_PRESSURE", False), mock.patch.object(
        mempressure, "take_worst", lambda: mempressure.CRITICAL
    ):
        assert ExpertCache._pressure_slots(fake, 1000) == 1000
    print("OK: rebuild slots follow pressure, bounded")


def _sized(per_expert=100 << 20, slab=None, full=0):
    ns = SimpleNamespace(_slab_per_expert=per_expert, slab=slab, _slab_full_slots=full)
    ns._system_slots = lambda w, held=0: ExpertCache._system_slots(ns, w, held)
    return ns


def test_available():
    avail = mempressure.available_bytes()
    if sys.platform == "darwin":
        assert avail is not None and avail > 0, avail
    print(f"OK: available {((avail or 0) / 1e9):.1f} GB")


def test_system_slots():
    fake = _sized()
    gb = 1 << 30
    margin = cache_mod._SLAB_MARGIN
    with mock.patch.object(config, "MEM_PRESSURE_FLOOR", 0.5):
        with mock.patch.object(mempressure, "available_bytes", lambda: 64 * gb):
            assert ExpertCache._system_slots(fake, 100) == 100, "room must not trim"
        with mock.patch.object(mempressure, "available_bytes", lambda: margin + 7 * gb):
            assert ExpertCache._system_slots(fake, 100) == 71
            # the slab being replaced counts as room
            assert ExpertCache._system_slots(fake, 100, held=3 * gb) == 100
        with mock.patch.object(mempressure, "available_bytes", lambda: 0):
            assert ExpertCache._system_slots(fake, 100) == 50, "floor not held"
        with mock.patch.object(mempressure, "available_bytes", lambda: None):
            assert ExpertCache._system_slots(fake, 100) == 100
        with mock.patch.object(config, "MEM_PRESSURE", False), mock.patch.object(
            mempressure, "available_bytes", lambda: 0
        ):
            assert ExpertCache._system_slots(fake, 100) == 100
    print("OK: rebuild capped to available memory, floor held")


def test_rebuild_caps_only_under_pressure():
    gb = 1 << 30

    def fake():
        ns = SimpleNamespace(
            _pressure=mempressure.SlabGovernor(),
            _rebuild_worst=0,
            _slab_per_expert=100 << 20,
            _ram_budget_bytes=0,
        )
        ns._pressure_slots = lambda w: ExpertCache._pressure_slots(ns, w)
        ns._system_slots = lambda w, held=0: ExpertCache._system_slots(ns, w, held)
        return ns

    with mock.patch.object(config, "MEM_PRESSURE_FLOOR", 0.5), mock.patch.object(
        mempressure, "available_bytes", lambda: cache_mod._SLAB_MARGIN + 5 * gb
    ):
        with mock.patch.object(mempressure, "take_worst", lambda: mempressure.NORMAL):
            assert ExpertCache._slots_that_fit(fake(), 100) == 100, "normal pressure trimmed"
        with mock.patch.object(mempressure, "take_worst", lambda: 0):
            assert ExpertCache._slots_that_fit(fake(), 100) == 100, "no reading trimmed"
        with mock.patch.object(mempressure, "take_worst", lambda: mempressure.WARN):
            assert ExpertCache._slots_that_fit(fake(), 100) == 51, "warn did not cap"
        with mock.patch.object(mempressure, "take_worst", lambda: mempressure.CRITICAL):
            assert ExpertCache._slots_that_fit(fake(), 100) == 51
    print("OK: rebuild caps to available memory only after warn")


def test_slab_oversized():
    gb = 1 << 30
    slab = SimpleNamespace(slots=100, nbytes=100 * (100 << 20))
    fake = _sized(slab=slab, full=100)
    margin = cache_mod._SLAB_MARGIN
    with mock.patch.object(config, "MEM_PRESSURE_FLOOR", 0.5):
        with mock.patch.object(mempressure, "level", lambda: mempressure.WARN):
            with mock.patch.object(mempressure, "available_bytes", lambda: margin):
                assert not ExpertCache.slab_oversized(fake), "slab itself fits"
            with mock.patch.object(mempressure, "available_bytes", lambda: 0):
                assert ExpertCache.slab_oversized(fake), "3 GB short must rebuild"
        with mock.patch.object(mempressure, "level", lambda: mempressure.NORMAL), \
                mock.patch.object(mempressure, "available_bytes", lambda: 0):
            assert not ExpertCache.slab_oversized(fake), "no pressure, no rebuild"
    fake = _sized(slab=None)
    assert not ExpertCache.slab_oversized(fake)
    print("OK: short turn rebuilds only when the slab no longer fits")


def test_take_worst_resets():
    mempressure._worst = mempressure.CRITICAL
    with mock.patch.object(mempressure, "level", lambda: mempressure.NORMAL):
        assert mempressure.take_worst() == mempressure.CRITICAL
        assert mempressure.take_worst() == mempressure.NORMAL
    print("OK: worst level resets per rebuild")


def main():
    test_level()
    test_governor()
    test_governor_waits_for_calm_and_compressor()
    test_pressure_slots_grow_into_free_memory_only()
    test_pressure_slots()
    test_available()
    test_system_slots()
    test_rebuild_caps_only_under_pressure()
    test_slab_oversized()
    test_take_worst_resets()
    print("\nOK: memory pressure tests passed")


if __name__ == "__main__":
    sys.exit(main())
