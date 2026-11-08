# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Keep the checkpoint out of the OS file cache.

F_NOCACHE is a hint, not a guarantee. A pread over 1 MB lands mostly in the
file cache anyway (16 MB reads: all of it), and even capped reads go the
buffered way once part of their range is resident, which pulls the rest in.
On Flash a two-turn run left 10.8 GB of expert pages cached after the cap,
30 GB before it. Clean file pages should be the first thing the kernel drops,
but these were just read, so they sit on the active list while the slab sits
idle through prefill. The kernel compressed 26 GB of slab to keep them and
the whole machine stalled for 16 s.

msync(MS_INVALIDATE) on a read-only mapping drops a file's clean pages. 15
GB came out in about 150 ms. A daemon thread watches the system's
file-backed page count (one sysctl) and drops the checkpoint files once it
has grown past TRIM_MB since the last drop. Nothing else of ours reads those
files through the cache, and the bytes on disk never change, so dropping
them costs at most a re-read.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os
import threading
import time

from . import config, mempressure

_PROT_READ = 0x1
_MAP_SHARED = 0x1
_MS_INVALIDATE = 0x2
_MAP_FAILED = ctypes.c_void_p(-1).value

_lock = threading.Lock()
_files: list[str] = []
_thread: threading.Thread | None = None
_libc = None
_dropped = 0
_trims = 0


def _c():
    global _libc
    if _libc is None:
        lib = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        lib.mmap.restype = ctypes.c_void_p
        lib.mmap.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_longlong,
        ]
        lib.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _libc = lib
    return _libc


def file_cache_bytes() -> int | None:
    """System-wide file-backed pages in bytes, None when not reported."""
    pages = mempressure._sysctl_int("vm.vm_page_external_count")
    if pages is None:
        return None
    return pages * mmap.PAGESIZE


def drop(paths) -> int:
    """Drop the clean cached pages of `paths`. Returns how many files it hit."""
    lib = _c()
    n = 0
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            size = os.fstat(fd).st_size
            if size <= 0:
                continue
            addr = lib.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
            if addr is None or addr == _MAP_FAILED:
                continue
            try:
                if lib.msync(addr, size, _MS_INVALIDATE) == 0:
                    n += 1
            finally:
                lib.munmap(addr, size)
        finally:
            os.close(fd)
    return n


def _poll() -> None:
    global _dropped, _trims
    period = max(0.05, float(config.PAGECACHE_TRIM_POLL_S))
    limit = max(0, int(config.PAGECACHE_TRIM_MB)) << 20
    # Pages left by an earlier run never show up as growth, so start clean.
    base = -1
    while True:
        now = file_cache_bytes()
        if now is None:
            time.sleep(period)
            continue
        if base >= 0 and now < base:
            base = now
        if base >= 0 and now - base < limit:
            time.sleep(period)
            continue
        with _lock:
            paths = list(_files)
        try:
            drop(paths)
        except Exception:  # noqa: BLE001 - never take decode down for this
            pass
        after = file_cache_bytes() or now
        _dropped += max(0, now - after)
        _trims += 1
        base = after


def watch(paths) -> bool:
    """Add checkpoint files and start the trimmer. Idempotent; False when off."""
    global _thread
    if not (config.PAGECACHE_TRIM and config.NOCACHE):
        return False
    if file_cache_bytes() is None:
        return False
    with _lock:
        for p in paths:
            p = str(p)
            if p not in _files:
                _files.append(p)
        if _thread is None:
            _thread = threading.Thread(target=_poll, name="pagecache-trim", daemon=True)
            _thread.start()
    return True


def stats() -> dict:
    return {"trims": _trims, "dropped_gb": round(_dropped / 1e9, 2)}
