# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Watchdog: a stuck engine or a drowning machine exits instead of hanging.

Three things the server could not do for itself:

* Stall. A Metal wait that never returns (seen: a prefill eval blocked on a
  Metal event, twice in a row, with 40 GB held) leaves a request open with
  no error. Every MoE layer's fetch touches `activity`; a request open for
  WATCHDOG_STALL_S with no touch is a hang. Stacks go to the log, then the
  process exits with code 70 so whoever owns the lifecycle restarts it.
* Memory. The kernel at CRITICAL pressure for WATCHDOG_PRESSURE_S straight
  while a request runs is the compressor eating the slab, which is the
  system lag and, further on, the freeze. The slab is what the machine
  needs back, so the engine drains its GPU queue and exits with code 71.
  Below that, WARN and CRITICAL are logged as they appear.
* Leftovers. Another paged-moe server on this machine whose parent is
  gone is a crashed lifecycle's orphan holding GPU memory (one Metal owner,
  or both OOM). It gets SIGTERM, then SIGKILL. One with a live parent is
  someone else's and is only reported.

Off with PAGED_MOE_WATCHDOG=0. Nothing here touches a request's values.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time

from . import activity, config, gpu_trail, mempressure

EXIT_STALL = 70
EXIT_PRESSURE = 71

# A python interpreter running the server module or script, not any command
# line that happens to mention it (a shell wrapper around this very test).
# A packaged app may start it as `python -I pagedmoe_launch.py pagedmoe.server`.
_SERVER_CMD = re.compile(
    r"^\S*python[\d.]*\s+(?:-\S+\s+)*"
    r"(?:-m\s+pagedmoe\.server\b|\S*pagedmoe/server\.py\b|\S*pagedmoe_launch\.py\s+pagedmoe\.server\b)"
)


def _log(msg: str) -> None:
    print(f"[watchdog] {msg}", flush=True)


# ------------------------------------------------------------- leftovers


def parse_ps(text: str, me: int) -> list[tuple[int, int, str]]:
    """(pid, ppid, command) rows of other paged-moe servers in `ps` output."""
    rows = []
    for line in text.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        cmd = parts[2]
        if pid == me or not _SERVER_CMD.match(cmd):
            continue
        rows.append((pid, ppid, cmd))
    return rows


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def sweep_leftovers(*, kill: bool = True, me: int | None = None, ps_text: str | None = None) -> list[int]:
    """Report other paged-moe servers; end the orphaned ones. Returns killed pids."""
    me = os.getpid() if me is None else me
    if ps_text is None:
        try:
            ps_text = subprocess.run(
                ["ps", "-axo", "pid=,ppid=,command="],
                capture_output=True, text=True, timeout=5,
            ).stdout
        except Exception as e:  # noqa: BLE001
            _log(f"leftover check skipped: {e}")
            return []
    killed: list[int] = []
    parent = os.getppid()
    for pid, ppid, cmd in parse_ps(ps_text, me):
        if pid == parent:
            continue
        orphan = ppid <= 1 or not _alive(ppid)
        if not orphan or not kill:
            _log(f"another paged-moe server is running: pid {pid} (parent {ppid}); "
                 f"{'left alone' if not orphan else 'not killing'}: {cmd[:120]}")
            continue
        _log(f"leftover engine pid {pid} (parent gone): stopping it. {cmd[:120]}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except Exception as e:  # noqa: BLE001
            _log(f"could not signal pid {pid}: {e}")
            continue
        deadline = time.monotonic() + 6.0
        while _alive(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        if _alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        killed.append(pid)
    return killed


# -------------------------------------------------------------- watchdog


def _dump_stacks() -> None:
    try:
        import faulthandler

        faulthandler.dump_traceback(all_threads=True)
    except Exception:  # noqa: BLE001
        pass


def _native_sample() -> str | None:
    """Native stacks of this process via macOS `sample`; the file path or None.

    The Python dump stops at the eval that is waiting. The frames that say
    which Metal wait it is are native, so they only show up here.
    """
    secs = int(config.WATCHDOG_SAMPLE_S)
    if secs <= 0 or sys.platform != "darwin" or not os.path.exists("/usr/bin/sample"):
        return None
    import tempfile

    root = config.INCIDENT_DIR or tempfile.gettempdir()
    try:
        os.makedirs(root, exist_ok=True)
        out = os.path.join(root, f"stall-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.txt")
        subprocess.run(["/usr/bin/sample", str(os.getpid()), str(secs), "-file", out],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=secs + 30, check=False)
    except Exception as e:  # noqa: BLE001
        _log(f"native sample failed: {e}")
        return None
    return out if os.path.exists(out) else None


# Kernel GPU driver lines worth keeping: faults, restarts, the firmware
# failing to switch away from a kernel that will not finish.
_OSLOG_PREDICATE = (
    '(process == "kernel" AND (eventMessage CONTAINS[c] "AGX" OR eventMessage CONTAINS[c] "IOGPU"'
    ' OR eventMessage CONTAINS[c] "GPU restart" OR eventMessage CONTAINS[c] "GPU fault"'
    ' OR eventMessage CONTAINS[c] "ctxswitch")) OR subsystem == "com.apple.Metal"'
    ' OR sender BEGINSWITH "AGX" OR sender BEGINSWITH "IOGPU"'
)


def _os_gpu_log(window_s: float) -> tuple[str | None, list[str]]:
    """The unified log's GPU driver lines from the last `window_s`.

    The process side of a hang is only ever a wait. Whatever the driver saw
    (an address fault, a restart, a kernel it could not preempt) is in here.
    Returns the saved file and the lines.
    """
    if not config.WATCHDOG_OSLOG or sys.platform != "darwin" or not os.path.exists("/usr/bin/log"):
        return None, []
    import tempfile

    try:
        out = subprocess.run(
            ["/usr/bin/log", "show", "--last", f"{int(window_s)}s", "--style", "compact",
             "--predicate", _OSLOG_PREDICATE],
            capture_output=True, text=True, timeout=45,
        ).stdout
    except Exception as e:  # noqa: BLE001
        _log(f"os log read failed: {e}")
        return None, []
    lines = [ln for ln in out.splitlines() if ln.strip() and not ln.startswith(("Timestamp", "Filtering"))]
    root = config.INCIDENT_DIR or tempfile.gettempdir()
    path = os.path.join(root, f"stall-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-oslog.txt")
    try:
        os.makedirs(root, exist_ok=True)
        with open(path, "w") as f:
            f.write(out)
    except OSError:
        path = None
    return path, lines


def _trail_path() -> str:
    import tempfile

    return os.path.join(config.INCIDENT_DIR or tempfile.gettempdir(), "gpu-trail.txt")


def _gpu_stats() -> str:
    """IOAccelerator utilization and in-use memory, as macOS reports them.

    A kernel that is still running reads as device utilization near 100%
    with nothing rendering. A wait that nobody signals reads idle.
    """
    if sys.platform != "darwin":
        return "n/a"
    try:
        out = subprocess.run(
            ["/usr/sbin/ioreg", "-r", "-c", "IOAccelerator", "-d", "1", "-w0"],
            capture_output=True, text=True, timeout=3,
        ).stdout
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({e})"
    got = {}
    for key in ("Device Utilization %", "Renderer Utilization %", "Tiler Utilization %",
                "In use system memory"):
        m = re.search(rf'"{re.escape(key)}"=(\d+)', out)
        if m:
            got[key] = int(m.group(1))
    if not got:
        return "unavailable"
    mem = got.get("In use system memory")
    used = f"{mem / 1e9:.1f} GB" if mem is not None else "?"
    return (f"device {got.get('Device Utilization %', '?')}%, renderer "
            f"{got.get('Renderer Utilization %', '?')}%, tiler {got.get('Tiler Utilization %', '?')}%, "
            f"in use {used}")


def _mx_memory() -> str:
    try:
        import mlx.core as mx

        return (f"active {mx.get_active_memory() / 1e9:.1f} GB, cache "
                f"{mx.get_cache_memory() / 1e9:.1f} GB, peak {mx.get_peak_memory() / 1e9:.1f} GB")
    except Exception:  # noqa: BLE001
        return "n/a"


def _thermal() -> str:
    """macOS thermal state from pmset: "ok", or the warning and speed limit."""
    if sys.platform != "darwin":
        return "n/a"
    try:
        out = subprocess.run(["/usr/bin/pmset", "-g", "therm"],
                             capture_output=True, text=True, timeout=3).stdout
    except Exception:  # noqa: BLE001
        return "n/a"
    warn = re.search(r"[Tt]hermal warning level (?:set to )?(\d+)", out)
    limit = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
    parts = []
    if warn:
        parts.append(f"warning level {warn.group(1)}")
    if limit and int(limit.group(1)) < 100:
        parts.append(f"cpu speed limit {limit.group(1)}%")
    return ", ".join(parts) or "ok"


def _mx_numbers() -> tuple[float, float]:
    try:
        import mlx.core as mx

        return mx.get_active_memory() / 1e9, mx.get_cache_memory() / 1e9
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


def _mx_active_gb() -> float:
    try:
        import mlx.core as mx

        return mx.get_active_memory() / 1e9
    except Exception:  # noqa: BLE001
        return 0.0


def _rss_gb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    except Exception:  # noqa: BLE001
        return 0.0


def _exit(code: int, drain: bool) -> None:
    # A synchronize from this thread would wait on this thread's own (empty)
    # stream. Decode drains its stream itself once activity.stop() is set.
    if drain:
        activity.stop()
        deadline = time.monotonic() + 4.0
        while activity.decoding() and time.monotonic() < deadline:
            time.sleep(0.02)
    sys.stdout.flush()
    os._exit(code)


class Watchdog:
    """Poll loop; `check` is separate from the thread so tests can drive it."""

    def __init__(self, *, stall_s: float, pressure_s: float, heartbeat_s: float,
                 exit=_exit, now=time.monotonic, level=mempressure.level,
                 thermal=_thermal, pace_ms: float | None = None):
        self.stall_s = float(stall_s)
        self.pressure_s = float(pressure_s)
        self.heartbeat_s = float(heartbeat_s)
        self._exit = exit
        self._now = now
        self._level = level
        self._thermal = thermal
        self.pace_ms = float(config.THERMAL_PACE_MS if pace_ms is None else pace_ms)
        self._thermal_at = 0.0
        self._bad_since: float | None = None
        self._last_level = 0
        self._last_beat = now()
        self._warned = False
        self._first: tuple[float, float] | None = None
        self._stop = threading.Event()

    def check(self) -> None:
        now = self._now()
        stalled = activity.stalled_for()
        if self.stall_s > 0 and stalled >= self.stall_s:
            _log(f"request open with no progress for {stalled:.0f}s "
                 f"(mx active {_mx_active_gb():.1f} GB). Stacks follow; exiting {EXIT_STALL}.")
            _log(f"state: pressure {mempressure.name(self._level())}, compressor "
                 f"{mempressure.compressor_gb():.1f} GB, peak rss {_rss_gb():.1f} GB, "
                 f"open requests {activity.open_requests()}")
            _log(f"mlx memory: {_mx_memory()}")
            _log(f"gpu: {_gpu_stats()}")
            _log("last GPU work queued (newest last):")
            for line in gpu_trail.report():
                _log(f"  {line}")
            gpu_trail.write(_trail_path(), force=True)
            _dump_stacks()
            sample = _native_sample()
            _log(f"native stacks: {sample}" if sample else "native stacks: unavailable")
            _log(f"gpu after the sample: {_gpu_stats()}")
            oslog, lines = _os_gpu_log(stalled + 60)
            if oslog:
                _log(f"gpu driver log, last {stalled + 60:.0f}s: {len(lines)} line(s), saved to {oslog}")
                for line in lines[-25:]:
                    _log(f"  {line[:240]}")
            self._exit(EXIT_STALL, False)
            return
        if self.stall_s > 0 and stalled >= min(30.0, self.stall_s / 2):
            if not self._warned:
                self._warned = True
                _log(f"no progress for {stalled:.0f}s (exit at {self.stall_s:.0f}s). gpu: {_gpu_stats()}")
                for line in gpu_trail.report():
                    _log(f"  {line}")
        else:
            self._warned = False
        if activity.open_requests():
            # Past a few seconds of no progress, keep the file current even
            # though nothing new was queued: the GPU crumbs can still move,
            # and a freeze that takes the Mac down leaves only this file.
            gpu_trail.write(_trail_path(), force=stalled >= 10)
        lv = self._level()
        if lv != self._last_level and lv >= mempressure.WARN:
            _log(f"memory pressure {mempressure.name(lv)}: mx active {_mx_active_gb():.1f} GB, "
                 f"compressor {mempressure.compressor_gb():.1f} GB")
        self._last_level = lv
        if lv >= mempressure.CRITICAL and activity.open_requests():
            if self._bad_since is None:
                self._bad_since = now
            elif self.pressure_s > 0 and now - self._bad_since >= self.pressure_s:
                _log(f"memory pressure critical for {now - self._bad_since:.0f}s during a request "
                     f"(mx active {_mx_active_gb():.1f} GB, compressor "
                     f"{mempressure.compressor_gb():.1f} GB). Exiting {EXIT_PRESSURE} to give the memory back.")
                self._exit(EXIT_PRESSURE, True)
                return
        else:
            self._bad_since = None
        self._pace(now)
        if self.heartbeat_s > 0 and now - self._last_beat >= self.heartbeat_s:
            self._last_beat = now
            self._beat(lv)

    def _pace(self, now: float) -> None:
        """Slow decode a little while macOS says the machine is too hot. pmset runs every 15 s at most."""
        if self.pace_ms <= 0:
            return
        if not activity.open_requests():
            if activity.pace():
                activity.set_pace(0.0)
            return
        if now - self._thermal_at < 15.0:
            return
        self._thermal_at = now
        state = self._thermal()
        hot = state not in ("ok", "n/a")
        if hot != bool(activity.pace()):
            activity.set_pace(self.pace_ms / 1000.0 if hot else 0.0)
            _log(f"thermal {state}: decode pace {'on' if hot else 'off'}"
                 f"{f' ({self.pace_ms:.0f} ms per token)' if hot else ''}")

    def _beat(self, lv: int) -> None:
        # One line every heartbeat_s, cheap enough for release builds. The
        # "since first" numbers show slow growth (a leak, a cache that never
        # gives back) and the gpu/thermal fields show a machine heating up,
        # which is what a hang so far has looked like from the outside.
        active, cache = _mx_numbers()
        if self._first is None:
            self._first = (active, cache)
        a0, c0 = self._first
        _log(f"alive: mx active {active:.1f} GB ({active - a0:+.1f} since first), "
             f"cache {cache:.1f} GB ({cache - c0:+.1f}), peak rss {_rss_gb():.1f} GB, "
             f"compressor {mempressure.compressor_gb():.1f} GB, "
             f"pressure {mempressure.name(lv) if lv else 'n/a'}, "
             f"gpu {_gpu_stats()}, thermal {_thermal()}, "
             f"open requests {activity.open_requests()}")

    def run(self, period_s: float = 2.0) -> None:
        while not self._stop.wait(period_s):
            try:
                self.check()
            except Exception as e:  # noqa: BLE001
                _log(f"check failed: {e}")

    def stop(self) -> None:
        self._stop.set()


_running: Watchdog | None = None


def start() -> Watchdog | None:
    """Sweep leftovers and start the poll thread (idempotent). None when off."""
    global _running
    if not config.WATCHDOG:
        return None
    if _running is not None:
        return _running
    sweep_leftovers(kill=bool(config.LEFTOVER_KILL))
    gpu_trail.rotate(_trail_path())
    _running = Watchdog(
        stall_s=config.WATCHDOG_STALL_S,
        pressure_s=config.WATCHDOG_PRESSURE_S,
        heartbeat_s=config.WATCHDOG_HEARTBEAT_S,
    )
    t = threading.Thread(target=_running.run, name="paged-moe-watchdog", daemon=True)
    t.start()
    _log(f"on: stall {config.WATCHDOG_STALL_S:.0f}s, critical pressure "
         f"{config.WATCHDOG_PRESSURE_S:.0f}s, heartbeat {config.WATCHDOG_HEARTBEAT_S:.0f}s")
    return _running
