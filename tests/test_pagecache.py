# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Checkpoint reads stay out of the OS file cache (pagecache.py, MAX_IO_BYTES).

Every read loop has to split at MAX_IO_BYTES and still return the same bytes,
drop() has to actually evict a file's cached pages, and the trimmer must stay
off when NOCACHE is off.

Run:
    python tests/test_pagecache.py
"""

import ctypes
import mmap
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pagedmoe import cache, config, pagecache  # noqa: E402
from pagedmoe.safetensors_index import MAX_IO_BYTES, FilePool  # noqa: E402


def _resident(path: str) -> int:
    size = os.path.getsize(path)
    libc = ctypes.CDLL(None)
    with open(path, "rb") as f:
        m = mmap.mmap(f.fileno(), size, prot=mmap.PROT_READ)
        a = np.frombuffer(m, dtype=np.uint8)
        n = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        vec = ctypes.create_string_buffer(n)
        libc.mincore(ctypes.c_void_p(a.ctypes.data), ctypes.c_size_t(size), vec)
        res = int((np.frombuffer(vec.raw, dtype=np.uint8) & 1).sum()) * mmap.PAGESIZE
        del a
        m.close()
    return res


def _tmpfile(nbytes: int) -> str:
    fd, path = tempfile.mkstemp(suffix=".bin")
    data = np.random.default_rng(0).integers(0, 256, nbytes, dtype=np.uint8)
    os.write(fd, data.tobytes())
    os.close(fd)
    return path


def test_reads_split_and_match():
    n = 3 * MAX_IO_BYTES + 12345
    path = _tmpfile(n + 777)
    try:
        want = Path(path).read_bytes()[777 : 777 + n]
        seen = []
        real = os.preadv

        def spy(fd, bufs, off):
            seen.append(len(bufs[0]))
            return real(fd, bufs, off)

        pool = FilePool(nocache=True)
        buf = bytearray(n)
        with mock.patch.object(os, "preadv", spy):
            pool.read_into(path, 777, buf)
        assert bytes(buf) == want, "read_into bytes differ"
        assert max(seen) <= MAX_IO_BYTES, max(seen)
        assert pool.read_bytes(path, 777, n) == want, "read_bytes bytes differ"

        seen.clear()
        buf = bytearray(n)
        fd = pool._fd(path)
        with mock.patch.object(config, "NATIVE_READ", "off"), mock.patch.object(
            os, "preadv", spy
        ):
            cache._pread_full(fd, 777, memoryview(buf))
        assert bytes(buf) == want, "_pread_full bytes differ"
        assert max(seen) <= MAX_IO_BYTES, max(seen)

        buf = bytearray(n)
        cache._pread_full(fd, 777, memoryview(buf))  # native when it loads
        assert bytes(buf) == want, "native pread_full bytes differ"
        pool.close()
    finally:
        os.unlink(path)
    print("OK: reads split at MAX_IO_BYTES, same bytes")


def test_drop_evicts():
    if sys.platform != "darwin":
        print("SKIP: drop is macOS only")
        return
    path = _tmpfile(32 << 20)
    try:
        Path(path).read_bytes()  # buffered: lands in the file cache
        before = _resident(path)
        assert pagecache.drop([path]) == 1
        after = _resident(path)
        assert before > (16 << 20), before
        assert after < (1 << 20), after
        assert pagecache.drop([path + ".missing"]) == 0
    finally:
        os.unlink(path)
    print(f"OK: drop evicted {before >> 20} MB -> {after >> 20} MB")


def test_watch_gated():
    with mock.patch.object(config, "NOCACHE", False):
        assert pagecache.watch(["/nope"]) is False
    with mock.patch.object(config, "PAGECACHE_TRIM", False):
        assert pagecache.watch(["/nope"]) is False
    print("OK: trimmer off without NOCACHE / TRIM")


def main():
    test_reads_split_and_match()
    test_drop_evicts()
    test_watch_gated()
    print("\nOK: page cache tests passed")


if __name__ == "__main__":
    sys.exit(main())
