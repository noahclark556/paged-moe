# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native read path: load/fallback + bit-identical pread vs os.preadv."""

from __future__ import annotations

import os
import tempfile
import threading
from concurrent.futures import Future
from pathlib import Path

import pytest

from pagedmoe.cache import _SlotLatch, _pread_full
from pagedmoe import config, native_read


def test_native_module_loads_or_falls_back_cleanly():
    # Force a fresh attempt so this test doesn't depend on import order.
    mod = native_read.get_module(force=True)
    state, detail = native_read.status()
    assert state in ("loaded", "fallback")
    assert isinstance(detail, str)
    if mod is not None:
        assert hasattr(mod, "create_pool")
        assert hasattr(mod, "submit_expert")
        assert hasattr(mod, "pread_full")


def test_pread_full_matches_os_preadv_when_native_available():
    mod = native_read.get_module(force=True)
    if mod is None:
        pytest.skip(f"native unavailable: {native_read.status()}")

    payload = os.urandom(64 * 1024 + 17)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
        f.write(payload)
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            # Split reads: head via native, mid via Python helper, tail via native.
            a = bytearray(1000)
            b = bytearray(2000)
            c = bytearray(len(payload) - 3000)
            mod.pread_full(fd, 0, memoryview(a))
            # Python path with native preferred still lands identical bytes.
            saved = config.NATIVE_READ
            try:
                config.NATIVE_READ = "on"
                _pread_full(fd, 1000, memoryview(b))
                config.NATIVE_READ = "off"
                _pread_full(fd, 3000, memoryview(c))
            finally:
                config.NATIVE_READ = saved
            assert bytes(a) + bytes(b) + bytes(c) == payload
        finally:
            os.close(fd)
    finally:
        Path(path).unlink(missing_ok=True)


def test_native_pool_submit_expert_fills_buffers():
    mod = native_read.get_module(force=True)
    if mod is None:
        pytest.skip(f"native unavailable: {native_read.status()}")

    chunks = [os.urandom(4096), os.urandom(8192), os.urandom(1024)]
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
        for c in chunks:
            f.write(c)
    offsets = []
    off = 0
    for c in chunks:
        offsets.append(off)
        off += len(c)

    pool = native_read.NativeReadPool(4, mod=mod)
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            dests = [bytearray(len(c)) for c in chunks]
            fut: Future = Future()
            fut.set_running_or_notify_cancel()
            entry = {"ok": True}
            latch = _SlotLatch(1, fut, entry)
            comps = [
                (fd, offsets[i], memoryview(dests[i])) for i in range(len(chunks))
            ]
            pool.submit_expert(comps, latch)
            assert fut.result(timeout=5) is entry
            for i, c in enumerate(chunks):
                assert bytes(dests[i]) == c
        finally:
            os.close(fd)
    finally:
        pool.shutdown()
        Path(path).unlink(missing_ok=True)


@pytest.mark.parametrize("split", [0, 4096, 64 * 1024, 1 << 20])
def test_native_pool_split_pieces_land_identical_bytes(split):
    mod = native_read.get_module(force=True)
    if mod is None or getattr(mod, "API_VERSION", 1) < 2:
        pytest.skip(f"native split unavailable: {native_read.status()}")

    sizes = [3 * 1024 * 1024 + 123, 0, 70001, 4096, 2 * 1024 * 1024]
    chunks = [os.urandom(n) for n in sizes]
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
        for c in chunks:
            f.write(c)
    offsets = [sum(sizes[:i]) for i in range(len(sizes))]

    pool = native_read.NativeReadPool(8, mod=mod, split_bytes=split)
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            jobs = []
            for n in range(24):
                dests = [bytearray(len(c)) for c in chunks]
                fut: Future = Future()
                fut.set_running_or_notify_cancel()
                entry = {"n": n}
                comps = [(fd, offsets[i], memoryview(dests[i])) for i in range(len(chunks))]
                pool.submit_expert(comps, _SlotLatch(1, fut, entry), urgent=n % 3 == 0)
                jobs.append((fut, entry, dests))
            for fut, entry, dests in jobs:
                assert fut.result(timeout=10) is entry
                for i, c in enumerate(chunks):
                    assert bytes(dests[i]) == c
        finally:
            os.close(fd)
    finally:
        pool.shutdown()
        Path(path).unlink(missing_ok=True)


def test_native_pool_split_reports_short_read_once():
    mod = native_read.get_module(force=True)
    if mod is None or getattr(mod, "API_VERSION", 1) < 2:
        pytest.skip(f"native split unavailable: {native_read.status()}")

    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
        f.write(os.urandom(10_000))
    pool = native_read.NativeReadPool(4, mod=mod, split_bytes=4096)
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            fut: Future = Future()
            fut.set_running_or_notify_cancel()
            dest = bytearray(64 * 1024)
            pool.submit_expert([(fd, 0, memoryview(dest))], _SlotLatch(1, fut, {}), urgent=True)
            with pytest.raises(OSError):
                fut.result(timeout=5)
        finally:
            os.close(fd)
    finally:
        pool.shutdown()
        Path(path).unlink(missing_ok=True)


def test_native_pool_propagates_short_read():
    mod = native_read.get_module(force=True)
    if mod is None:
        pytest.skip(f"native unavailable: {native_read.status()}")

    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
        f.write(b"hi")
    pool = native_read.NativeReadPool(2, mod=mod)
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            dest = bytearray(64)
            fut: Future = Future()
            fut.set_running_or_notify_cancel()
            latch = _SlotLatch(1, fut, {})
            pool.submit_expert([(fd, 0, memoryview(dest))], latch)
            with pytest.raises(OSError):
                fut.result(timeout=5)
        finally:
            os.close(fd)
    finally:
        pool.shutdown()
        Path(path).unlink(missing_ok=True)


def test_force_python_pool_path_still_constructs():
    """PAGED_MOE_NATIVE_READ=off must not attempt native construction."""
    from pagedmoe.cache import ExpertCache

    saved = config.NATIVE_READ
    try:
        config.NATIVE_READ = "off"
        c = ExpertCache.__new__(ExpertCache)
        c._native_reads = False
        # Bind the helper the same way enable_slabs would.
        pool = ExpertCache._make_read_pool(c, 2)
        assert type(pool).__name__ == "_ReadPool"
        assert c._native_reads is False
        pool.shutdown()
    finally:
        config.NATIVE_READ = saved


def test_parallel_submit_expert_stress():
    mod = native_read.get_module(force=True)
    if mod is None:
        pytest.skip(f"native unavailable: {native_read.status()}")

    payload = os.urandom(32 * 1024)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
        for _ in range(16):
            f.write(payload)
    pool = native_read.NativeReadPool(8, mod=mod)
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            errs: list[BaseException] = []
            done = threading.Barrier(17)

            def one(i: int) -> None:
                try:
                    dest = bytearray(len(payload))
                    fut: Future = Future()
                    fut.set_running_or_notify_cancel()
                    latch = _SlotLatch(1, fut, {"i": i})
                    pool.submit_expert(
                        [(fd, i * len(payload), memoryview(dest))], latch
                    )
                    fut.result(timeout=10)
                    assert bytes(dest) == payload
                except BaseException as e:  # noqa: BLE001
                    errs.append(e)
                finally:
                    done.wait(timeout=15)

            threads = [threading.Thread(target=one, args=(i,)) for i in range(16)]
            for t in threads:
                t.start()
            done.wait(timeout=15)
            for t in threads:
                t.join(timeout=5)
            assert not errs, errs
        finally:
            os.close(fd)
    finally:
        pool.shutdown()
        Path(path).unlink(missing_ok=True)


def test_promote_moves_queued_job_and_bytes_land_intact():
    mod = native_read.get_module(force=True)
    if mod is None or not hasattr(mod, "promote"):
        pytest.skip(f"native promote unavailable: {native_read.status()}")

    n, size = 64, 256 * 1024
    payloads = [os.urandom(size) for _ in range(n)]
    with tempfile.NamedTemporaryFile(delete=False) as f:
        path = f.name
        for p in payloads:
            f.write(p)
    pool = native_read.NativeReadPool(1, mod=mod)
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            # Not a latch the pool holds: nothing to move.
            stray = _SlotLatch(1, Future(), {})
            assert pool.promote(stray) is False
            dests, futs, latches = [], [], []
            for i in range(n):
                dest = bytearray(size)
                fut: Future = Future()
                fut.set_running_or_notify_cancel()
                latch = _SlotLatch(1, fut, {"i": i})
                pool.submit_expert([(fd, i * size, memoryview(dest))], latch)
                dests.append(dest)
                futs.append(fut)
                latches.append(latch)
            # One worker, 64 jobs queued: the last one is still waiting.
            assert pool.promote(latches[-1]) is True
            assert pool.promote(latches[-1]) is False
            for fut in futs:
                fut.result(timeout=10)
            for dest, p in zip(dests, payloads):
                assert bytes(dest) == p
        finally:
            os.close(fd)
    finally:
        pool.shutdown()
        Path(path).unlink(missing_ok=True)
