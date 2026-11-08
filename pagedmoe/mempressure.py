# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Size the decode slab to what the machine can actually hold.

The RAM budget is a fraction of total RAM, fixed at load. It does not know
what else is running. With an editor's language server at 10 GB next to a
35 GB budget on a 48 GB Mac, macOS stops finding free pages and starts
compressing the coldest ones, and the slab is mostly cold: idle slots of a
24 GB allocation nobody has touched since the last turn. Measured: 28 GB in
the compressor, pressure level WARN, swap 6.5 of 7 GB, and decode steps of
1.9 s while those pages came back in. 4-bit weights barely compress (1.4x),
so the compressor burns CPU for very little room, and every app on the
machine waits on it. That is the system lag.

A compressed slot is not resident in any useful sense. Reading it costs a
decompress, and the pread of an expert into it has to fault the old page in
before it can overwrite it. So shrinking the slab while the kernel reports
pressure gives up slots that were not really there, and hands the memory back
to whoever needs it.

The kernel's own signal is `kern.memorystatus_vm_pressure_level` (1 normal,
2 warn, 4 critical). It tracks live demand, not leftover swap: it read 1 again
with 3.5 GB still swapped once the engine exited. A daemon thread samples it
and keeps the worst level since the last slab rebuild, so a pressure spike
during decode is still seen at the next rebuild after prefill.

Once the level has reached warn since the last rebuild, a rebuild also checks
what the kernel can hand out right now (`available_bytes`) and builds no
more slab than that, less a margin. Another app that grew between turns
then costs slab slots, never a compressed slab. At normal pressure that cap
is skipped: macOS keeps little free on purpose, so free + file cache reads
far below what it can give and the slab would shrink every turn. A short
turn with no prefill does not rebuild, so under warn it starts by checking
whether the slab it holds still fits next to everyone else, and rebuilds
only if not.

Only the slab size changes. Every expert a layer computes with is still the
one the router picked, read from disk if it is not resident.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import subprocess
import threading
import time

from . import config, release

NORMAL, WARN, CRITICAL = 1, 2, 4

_lock = threading.Lock()
_worst = 0
_thread: threading.Thread | None = None
_libc = None
# Monotonic time the poller last saw WARN or worse (or started). None = not polling.
_last_bad: float | None = None
# Same, but only real readings. None until the kernel has reported pressure.
_seen_bad: float | None = None


def _sysctl_int(name: str) -> int | None:
    global _libc
    try:
        if _libc is None:
            _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        val = ctypes.c_int64(0)
        size = ctypes.c_size_t(ctypes.sizeof(val))
        if _libc.sysctlbyname(name.encode(), ctypes.byref(val), ctypes.byref(size), None, 0):
            return None
        if size.value == 4:
            return ctypes.c_int32(val.value & 0xFFFFFFFF).value
        return int(val.value)
    except Exception:
        return None


def level() -> int:
    """Kernel pressure level now, 0 when the platform does not report one."""
    return _sysctl_int("kern.memorystatus_vm_pressure_level") or 0


# GPU memory macOS reports in use before this engine allocates anything.
# WindowServer, the editor and a browser sit at 1-4 GB. A crashed GPU process
# can leave far more behind in the driver with no process attached (30.6 GB
# seen after a Metal OOM followed by a kill), and only a reboot returns it.
# Another GPU-heavy app reads the same way; either way the memory is gone.
GPU_NORMAL_BYTES = 3 * 10**9
GPU_HELD_MIN_BYTES = 8 * 10**9
GPU_HELD_RAM_FRACTION = 0.15


def gpu_in_use_bytes() -> int | None:
    """IOAccelerator "In use system memory", None when unknown.

    PAGED_MOE_SIMULATE_GPU_HELD_GB stands in for the reading (tests).
    """
    sim = release.env("PAGED_MOE_SIMULATE_GPU_HELD_GB").strip()
    if sim:
        try:
            return int(float(sim) * 1e9)
        except ValueError:
            return None
    try:
        out = subprocess.run(
            ["/usr/sbin/ioreg", "-r", "-c", "IOAccelerator", "-d", "1", "-w0"],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
    except Exception:
        return None
    vals = [int(v) for v in re.findall(r'"In use system memory"=(\d+)', out)]
    return max(vals) if vals else None


def gpu_held_excess(held: int | None, ram: int) -> int:
    """Bytes of `held` above normal, 0 unless it crosses the held threshold."""
    if not held:
        return 0
    if held < max(GPU_HELD_MIN_BYTES, GPU_HELD_RAM_FRACTION * ram):
        return 0
    return max(0, int(held) - GPU_NORMAL_BYTES)


_working_set: int | None = None
_working_set_read = False


def gpu_working_set_bytes() -> int | None:
    """Metal's max recommended working set: the resident size the GPU serves
    without paging. Exceeding it with memory already wired (a crashed engine's
    orphaned residency sets, or another GPU app) is what makes a command buffer
    OOM on Metal's own completion thread and abort the process. None when mlx
    is missing or the field is absent. Read once and cached.
    """
    global _working_set, _working_set_read
    if _working_set_read:
        return _working_set
    _working_set_read = True
    sim = release.env("PAGED_MOE_SIMULATE_GPU_WORKING_SET_GB").strip()
    if sim:
        try:
            _working_set = int(float(sim) * 1e9)
        except ValueError:
            _working_set = None
        return _working_set
    try:
        import mlx.core as mx

        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        n = int(info.get("max_recommended_working_set_size") or 0)
        _working_set = n if n > 0 else None
    except Exception:
        _working_set = None
    return _working_set


def compressor_gb() -> float:
    return (_sysctl_int("vm.compressor_bytes_used") or 0) / 1e9


def available_bytes() -> int | None:
    """What the kernel can hand out without compressing anyone: free pages,
    file cache (clean, dropped on demand) and purgeable. None when unknown.
    """
    free = _sysctl_int("vm.page_free_count")
    page = _sysctl_int("hw.pagesize")
    if free is None or not page:
        return None
    files = _sysctl_int("vm.vm_page_external_count") or 0
    purge = _sysctl_int("vm.page_purgeable_count") or 0
    return (free + files + purge) * page


_libproc = None


def footprint_gb() -> float:
    """This process's phys_footprint (what Activity Monitor shows), 0 if unknown."""
    global _libproc
    try:
        if _libproc is None:
            _libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
        buf = (ctypes.c_uint64 * 64)()
        if _libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)):  # RUSAGE_INFO_V4
            return 0.0
        return buf[9] / 1e9  # ri_phys_footprint
    except Exception:
        return 0.0


def _poll() -> None:
    global _worst, _last_bad, _seen_bad
    period = max(0.05, float(config.MEM_PRESSURE_POLL_S))
    while True:
        lv = level()
        with _lock:
            if lv > _worst:
                _worst = lv
            if lv >= WARN:
                _last_bad = _seen_bad = time.monotonic()
        time.sleep(period)


def start() -> bool:
    """Begin sampling. Idempotent; False when off or unsupported."""
    global _thread, _last_bad
    if not config.MEM_PRESSURE or level() == 0:
        return False
    with _lock:
        if _thread is None:
            # Calm is only known from here on, so a fresh process starts at 0.
            _last_bad = time.monotonic()
            _thread = threading.Thread(target=_poll, name="mem-pressure", daemon=True)
            _thread.start()
    return True


def calm_s() -> float:
    """Seconds since the kernel last reported WARN or worse. 0 when not
    polling, so anything gated on calm stays off where it can't be measured.
    """
    with _lock:
        last = _last_bad
    if last is None:
        return 0.0
    if level() >= WARN:
        return 0.0
    return max(0.0, time.monotonic() - last)


def recent(window_s: float) -> bool:
    """True when the kernel reported WARN or worse in the last `window_s`.
    False when not polling, so callers keep their normal behavior there.

    Unlike `calm_s`, the poller starting is not a reading. Waiting for proven
    calm before growing is right; shrinking for the first minute of every
    process because nothing has been measured yet is not.
    """
    with _lock:
        polling = _last_bad is not None
        seen = _seen_bad
    if not polling:
        return False
    if level() >= WARN:
        return True
    return seen is not None and time.monotonic() - seen < max(0.0, float(window_s))


def take_worst() -> int:
    """Worst level since the last call, folded with the level right now."""
    global _worst
    now = level()
    with _lock:
        worst = max(_worst, now)
        _worst = 0
    return worst


class SlabGovernor:
    """Fraction of the wanted slab to build, stepped once per rebuild.

    Down a step on WARN, two on CRITICAL, back up half a step on NORMAL.
    Down fast because the machine is already paying for it, up slowly so a
    regrow does not walk straight back into the same pressure. FLOOR bounds
    what a long stretch of pressure can cost the hit rate.

    NORMAL alone is not enough to grow. The level reads normal again as soon
    as the kernel has a few free pages, with gigabytes still compressed.
    Measured on Flash next to a Dart analysis server: 85% -> 92% -> 100% over
    two turns with the compressor at 7.9 GB, then warn again at 15.4 GB and
    27.6 GB. So a step up also waits out SLAB_REGROW_CALM_S of calm and a
    compressor no bigger than it was at the last step down.
    """

    # Compressor growth since the last step down that still counts as flat.
    COMPRESSOR_SLACK_GB = 0.5

    def __init__(self) -> None:
        self.scale = 1.0
        self._compressor_gb: float | None = None

    def _may_grow(self) -> bool:
        if calm_s() < float(config.SLAB_REGROW_CALM_S):
            return False
        if self._compressor_gb is None:
            return True
        return compressor_gb() <= self._compressor_gb + self.COMPRESSOR_SLACK_GB

    def step(self, worst: int) -> float:
        st = max(0.0, float(config.MEM_PRESSURE_STEP))
        floor = min(1.0, max(0.0, float(config.MEM_PRESSURE_FLOOR)))
        if worst >= CRITICAL:
            self.scale -= 2 * st
            self._compressor_gb = compressor_gb()
        elif worst >= WARN:
            self.scale -= st
            self._compressor_gb = compressor_gb()
        elif worst == NORMAL and self.scale < 1.0 and self._may_grow():
            self.scale += st / 2
        self.scale = min(1.0, max(floor, self.scale))
        if self.scale >= 1.0:
            self._compressor_gb = None
        return self.scale


def name(lv: int) -> str:
    return {NORMAL: "normal", WARN: "warn", CRITICAL: "critical"}.get(lv, str(lv))
