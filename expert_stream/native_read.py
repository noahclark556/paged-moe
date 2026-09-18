# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Load the native expert-read pool, or fall back to Python.

Tries the setuptools extension, then a JIT build under a cache dir, then
None. Same slab bytes either way; leave on and ignore compile failures.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import sysconfig
import threading
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_MOD: Any | None = None
_TRIED = False
_STATUS = "untried"
_STATUS_DETAIL = ""


def status() -> tuple[str, str]:
    """``(state, detail)`` for logs: loaded / fallback / disabled / ..."""
    return _STATUS, _STATUS_DETAIL


def _cache_dir() -> Path:
    # Prefer a host agent data root when set; else XDG / macOS cache / ~/.cache.
    # Install-time builds are nicer when they work. This path exists because
    # half the Macs I tried did not have a usable compiler in PATH for pip.
    for key in ("AGENT_DATA_DIR", "GA_DATA_DIR"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return Path(raw).expanduser() / "native"
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg:
        return Path(xdg).expanduser() / "paged-moe" / "native"
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / "paged-moe" / "native"
    return home / ".cache" / "paged-moe" / "native"


def _source_path() -> Path:
    return Path(__file__).resolve().with_name("_native_read.c")


def _ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _include_dirs() -> list[str]:
    dirs: list[str] = []
    for key in ("INCLUDEPY", "CONFINCLUDEPY"):
        p = sysconfig.get_config_var(key)
        if p and os.path.isdir(p):
            dirs.append(p)
    inc = sysconfig.get_path("include")
    if inc and os.path.isdir(inc):
        dirs.append(inc)
    # Dedup, preserve order.
    out: list[str] = []
    for d in dirs:
        if d not in out:
            out.append(d)
    return out


def _try_import_installed() -> Any | None:
    try:
        return importlib.import_module("expert_stream._native_read")
    except Exception as e:  # noqa: BLE001 - any load failure -> next strategy
        global _STATUS_DETAIL
        _STATUS_DETAIL = f"installed import failed: {type(e).__name__}: {e}"
        return None


def _load_from_path(path: Path) -> Any | None:
    name = "expert_stream._native_read"
    try:
        loader = importlib.machinery.ExtensionFileLoader(name, str(path))
        spec = importlib.util.spec_from_file_location(name, path, loader=loader)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        # Register before exec so capsule destructors during failed init
        # still see a coherent module table.
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:  # noqa: BLE001
        sys.modules.pop(name, None)
        global _STATUS_DETAIL
        _STATUS_DETAIL = f"load {path.name} failed: {type(e).__name__}: {e}"
        return None


def _jit_compile() -> Any | None:
    """Compile `_native_read.c` into a user cache with the system `cc`."""
    global _STATUS_DETAIL
    src = _source_path()
    if not src.is_file():
        _STATUS_DETAIL = f"missing source: {src}"
        return None

    try:
        cache = _cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _STATUS_DETAIL = f"cache dir unusable: {e}"
        return None

    tag = (
        f"py{sys.version_info[0]}{sys.version_info[1]}-"
        f"{sysconfig.get_platform().replace('-', '_')}"
    )
    out = cache / f"_native_read.{tag}{_ext_suffix()}"
    # Reuse a fresh-enough build of the same source.
    if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
        mod = _load_from_path(out)
        if mod is not None:
            return mod
        try:
            out.unlink()
        except OSError:
            pass

    includes = _include_dirs()
    if not includes:
        _STATUS_DETAIL = "no Python.h include path (sysconfig)"
        return None
    if not any((Path(d) / "Python.h").is_file() for d in includes):
        _STATUS_DETAIL = f"Python.h not found in {includes}"
        return None

    cc = os.environ.get("CC", "cc").strip() or "cc"
    cmd = [cc, "-O2", "-fPIC"]
    if sys.platform == "darwin":
        # macOS Python extensions: bundle + dynamic_lookup (no libpython link).
        cmd += ["-bundle", "-undefined", "dynamic_lookup"]
    else:
        cmd += ["-shared"]
    for d in includes:
        cmd += ["-I", d]
    cmd += [str(src), "-o", str(out), "-lpthread"]

    import subprocess

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError:
        _STATUS_DETAIL = f"compiler not found: {cc}"
        return None
    except subprocess.TimeoutExpired:
        _STATUS_DETAIL = "native compile timed out"
        return None
    except OSError as e:
        _STATUS_DETAIL = f"native compile spawn failed: {e}"
        return None

    if proc.returncode != 0 or not out.is_file():
        err = (proc.stderr or proc.stdout or "").strip()
        _STATUS_DETAIL = f"compile failed ({proc.returncode}): {err[:400]}"
        try:
            if out.is_file():
                out.unlink()
        except OSError:
            pass
        return None

    return _load_from_path(out)


def get_module(*, force: bool = False) -> Any | None:
    """Return the native module or None. Thread-safe; tries once unless force."""
    global _MOD, _TRIED, _STATUS, _STATUS_DETAIL
    if _TRIED and not force:
        return _MOD
    with _LOCK:
        if _TRIED and not force:
            return _MOD
        _TRIED = True
        _MOD = None
        _STATUS = "fallback"
        _STATUS_DETAIL = ""

        mod = _try_import_installed()
        if mod is not None and hasattr(mod, "create_pool") and hasattr(mod, "submit_expert"):
            _MOD = mod
            _STATUS = "loaded"
            _STATUS_DETAIL = "setuptools extension"
            return _MOD

        mod = _jit_compile()
        if mod is not None and hasattr(mod, "create_pool") and hasattr(mod, "submit_expert"):
            _MOD = mod
            _STATUS = "loaded"
            _STATUS_DETAIL = f"jit:{_cache_dir()}"
            return _MOD

        if not _STATUS_DETAIL:
            _STATUS_DETAIL = "native read unavailable"
        _STATUS = "fallback"
        _MOD = None
        return None


class NativeReadPool:
    """Drop-in for ``_ReadPool`` with ``submit_expert`` for whole-expert jobs."""

    def __init__(self, threads: int, mod: Any | None = None):
        self._mod = mod if mod is not None else get_module()
        if self._mod is None:
            raise RuntimeError("native read module unavailable")
        self.threads = max(1, int(threads))
        self._pool = self._mod.create_pool(self.threads)
        # Same observer surface as Python _ReadPool (bench/drive_idle.py).
        self.busy = self._mod.busy_view(self._pool)

    def submit_expert(self, components: list[tuple[int, int, memoryview]], latch) -> None:
        """One queue job: all component preads, then a single ``latch.done``."""
        self._mod.submit_expert(self._pool, components, latch)

    def submit(self, fd: int, offset: int, mv: memoryview, latch) -> None:
        """Single-component submit (compat with Python _ReadPool API)."""
        self._mod.submit_expert(self._pool, [(fd, offset, mv)], latch)

    def shutdown(self) -> None:
        try:
            self._mod.shutdown(self._pool)
        except Exception:  # noqa: BLE001 - best-effort on interpreter teardown
            pass


def pread_full(fd: int, offset: int, mv: memoryview) -> bool:
    """Native ``_pread_full``. Returns True if handled, False to use Python."""
    mod = get_module()
    if mod is None or not hasattr(mod, "pread_full"):
        return False
    mod.pread_full(fd, offset, mv)
    return True
