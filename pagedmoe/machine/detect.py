# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read this Mac: RAM, cores, chip, storage, optional SSD sequential probe."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .types import Hardware

_PROBE_BYTES_DEFAULT = 64 << 20  # 64 MB: fast, enough for a sequential number


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", name],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.strip() or None
    except Exception:
        return None


def _sysctl_int(name: str) -> int | None:
    raw = _sysctl(name)
    if raw is None:
        return None
    try:
        return int(raw.split()[0])
    except ValueError:
        return None


def probe_disk_gb_s(
    path: str | Path | None = None,
    *,
    nbytes: int = _PROBE_BYTES_DEFAULT,
) -> float | None:
    """Cold-ish sequential read GB/s using F_NOCACHE when available.

    Writes a temp file on the same volume as ``path`` (default: home), then
    reads it back. Not a marketing SSD bench; it is the number the sandbox
    delay math needs.
    """
    root = Path(path).expanduser() if path else Path.home()
    try:
        root = root if root.is_dir() else root.parent
        if not root.exists():
            root = Path.home()
    except Exception:
        root = Path.home()

    data = os.urandom(min(1 << 20, nbytes))
    repeats = max(1, nbytes // len(data))
    fd = None
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix="pagedmoe-diskprobe-", dir=str(root))
        os.close(fd)
        fd = None
        with open(tmp, "wb") as f:
            for _ in range(repeats):
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # Drop page cache as much as we can.
        try:
            import fcntl

            rfd = os.open(tmp, os.O_RDONLY)
            try:
                fcntl.fcntl(rfd, fcntl.F_NOCACHE, 1)
            except Exception:
                pass
            start = time.perf_counter()
            n = 0
            while True:
                chunk = os.read(rfd, 8 << 20)
                if not chunk:
                    break
                n += len(chunk)
            dt = time.perf_counter() - start
            os.close(rfd)
        except Exception:
            start = time.perf_counter()
            with open(tmp, "rb") as f:
                n = 0
                while True:
                    chunk = f.read(8 << 20)
                    if not chunk:
                        break
                    n += len(chunk)
            dt = time.perf_counter() - start
        if dt <= 0 or n <= 0:
            return None
        return round((n / dt) / (1 << 30), 2)
    except Exception:
        return None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _gpu_cores_ioreg() -> int | None:
    try:
        out = subprocess.check_output(
            ["ioreg", "-l", "-w", "0"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )
    except Exception:
        return None
    for key in ("gpu-core-count", "IOGPUCoreCount", "gpu-core-count-driver"):
        for line in out.splitlines():
            if key in line:
                digits = "".join(ch for ch in line.split("=")[-1] if ch.isdigit())
                if digits:
                    try:
                        return int(digits)
                    except ValueError:
                        continue
    return None


def detect_hardware(*, probe_disk: bool = False, disk_path: str | None = None) -> Hardware:
    """Fast path: sysctl + shutil. Optional sequential SSD probe."""
    ram_bytes = _sysctl_int("hw.memsize")
    if ram_bytes is None:
        try:
            ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            ram_bytes = 16 << 30
    ram_gb = round(ram_bytes / (1 << 30), 2)

    perf = _sysctl_int("hw.perflevel0.physicalcpu")
    eff = _sysctl_int("hw.perflevel1.physicalcpu")
    if perf is None:
        perf = _sysctl_int("hw.physicalcpu") or 8
    chip = _sysctl("machdep.cpu.brand_string")
    hw_model = _sysctl("hw.model")

    storage_total = storage_free = None
    try:
        usage = shutil.disk_usage(str(Path.home()))
        storage_total = round(usage.total / (1 << 30), 1)
        storage_free = round(usage.free / (1 << 30), 1)
    except Exception:
        pass

    disk_gb_s = None
    if probe_disk or os.environ.get("PAGED_MOE_PROBE_DISK", "").strip() in (
        "1",
        "true",
        "yes",
    ):
        disk_gb_s = probe_disk_gb_s(disk_path)

    gpu = None
    if os.environ.get("PAGED_MOE_PROBE_GPU", "").strip() in ("1", "true", "yes"):
        gpu = _gpu_cores_ioreg()

    form = None
    model = (hw_model or "").lower()
    chip_l = (chip or "").lower()
    if "air" in chip_l:
        form = "air"
    elif "max" in chip_l:
        form = "max"
    elif "pro" in chip_l:
        form = "pro"
    elif "mac14,2" in model or "mac15,12" in model or "mac16,12" in model:
        form = "air"

    return Hardware(
        ram_gb=ram_gb,
        ram_bytes=int(ram_bytes),
        perf_cores=int(perf),
        efficiency_cores=eff,
        chip=chip,
        hw_model=hw_model,
        gpu_cores=gpu,
        disk_gb_s=disk_gb_s,
        storage_total_gb=storage_total,
        storage_free_gb=storage_free,
        form=form,
        source="detect",
    )


def probe_host(*, disk_path: str | None = None) -> Hardware:
    """Slow / exact capture: disk probe + ioreg GPU cores."""
    os.environ["PAGED_MOE_PROBE_GPU"] = "1"
    hw = detect_hardware(probe_disk=True, disk_path=disk_path)
    if hw.gpu_cores is None:
        hw.gpu_cores = _gpu_cores_ioreg()
    hw.source = "probe"
    return hw
