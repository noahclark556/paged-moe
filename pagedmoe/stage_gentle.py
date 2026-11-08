# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gentle background prompt staging.

A host stage (warm_prefix + stage_gentle) prefills a prompt nobody is waiting
on yet. It runs with the engine's disk reads in the background I/O class, so
the kernel serves every other app's reads first, and at each yield point
(router sync, expert group, attention sub-chunk) it yields to the user: while
there has been a click, key, scroll or gesture in any app in the last
STAGE_YIELD_S seconds (CoreGraphics per event type; pointer moves do not
count), the next step waits. So the stage runs in the user's pauses and never while they are using
the Mac. The prefill itself is unchanged: same chunks, one pass over the
expert mass, identical KV.

boost() ends all of it the moment a request is waiting on the stage (the
host calls POST /stage/boost), so a turn that lands mid-stage is never slowed.
Without a readable idle clock it falls back to a fixed pause per layer.
Every call here is best effort and never raises.
"""

from __future__ import annotations

import ctypes
import threading
import time

from . import config

_IOPOL_TYPE_DISK = 0
_IOPOL_SCOPE_PROCESS = 0
_IOPOL_DEFAULT = 0
_POLICIES = {"throttle": 3, "utility": 4}

_lock = threading.Lock()
_active = False
_policy_set = False
_setiopolicy = None
_yielded_s = 0.0
_started = 0.0


# Deliberate input only: clicks, drags, keys, modifiers, scroll, pinch and
# swipe. Pointer moves (5) and the generic trackpad touch events (gesture 29,
# pressure 34) are left out: a hand resting on the built-in trackpad sends
# them every few ms, and counting them held a stage while nobody was typing,
# clicking or scrolling.
_CG_TYPES = (1, 3, 6, 7, 10, 12, 22, 25, 27, 30, 31)
_CG_HID_STATE = 1  # kCGEventSourceStateHIDSystemState


class _InputClock:
    """Seconds since the last deliberate input, via CoreGraphics. None when unreadable."""

    def __init__(self) -> None:
        self._fn = None
        self._broken = False

    def seconds(self) -> float | None:
        if self._broken:
            return None
        try:
            if self._fn is None:
                cg = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
                fn = cg.CGEventSourceSecondsSinceLastEventType
                fn.restype = ctypes.c_double
                fn.argtypes = [ctypes.c_int, ctypes.c_uint32]
                self._fn = fn
            return min(self._fn(_CG_HID_STATE, t) for t in _CG_TYPES)
        except Exception:  # noqa: BLE001
            self._broken = True
            return None


class _IdleClock:
    """Seconds since any HID input, pointer moves included (IOHIDSystem
    HIDIdleTime). The fallback when CoreGraphics is unavailable."""

    def __init__(self) -> None:
        self._broken = False
        self._svc = 0
        self._key = None
        self._iokit = None
        self._cf = None

    def _open(self) -> bool:
        if self._broken:
            return False
        if self._svc:
            return True
        try:
            iokit = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/IOKit.framework/IOKit")
            cf = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
            iokit.IOServiceMatching.restype = ctypes.c_void_p
            iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
            iokit.IOServiceGetMatchingService.restype = ctypes.c_uint32
            iokit.IOServiceGetMatchingService.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
            iokit.IORegistryEntryCreateCFProperty.restype = ctypes.c_void_p
            iokit.IORegistryEntryCreateCFProperty.argtypes = [
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_uint32,
            ]
            cf.CFStringCreateWithCString.restype = ctypes.c_void_p
            cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
            cf.CFGetTypeID.restype = ctypes.c_ulong
            cf.CFGetTypeID.argtypes = [ctypes.c_void_p]
            cf.CFNumberGetTypeID.restype = ctypes.c_ulong
            cf.CFNumberGetValue.restype = ctypes.c_bool
            cf.CFNumberGetValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
            cf.CFRelease.argtypes = [ctypes.c_void_p]
            svc = iokit.IOServiceGetMatchingService(0, iokit.IOServiceMatching(b"IOHIDSystem"))
            key = cf.CFStringCreateWithCString(None, b"HIDIdleTime", 0x08000100)
            if not svc or not key:
                raise OSError("IOHIDSystem not found")
            self._iokit, self._cf, self._svc, self._key = iokit, cf, svc, key
            return True
        except Exception:  # noqa: BLE001
            self._broken = True
            return False

    def seconds(self) -> float | None:
        if not self._open():
            return None
        try:
            ref = self._iokit.IORegistryEntryCreateCFProperty(self._svc, self._key, None, 0)
            if not ref:
                return None
            try:
                if self._cf.CFGetTypeID(ref) != self._cf.CFNumberGetTypeID():
                    return None
                ns = ctypes.c_int64(0)
                if not self._cf.CFNumberGetValue(ref, 4, ctypes.byref(ns)):  # kCFNumberSInt64Type
                    return None
                return ns.value / 1e9
            finally:
                self._cf.CFRelease(ref)
        except Exception:  # noqa: BLE001
            self._broken = True
            return None


_input = _InputClock()
_clock = _IdleClock()


def user_idle_seconds() -> float | None:
    """Seconds since the user last clicked, typed, scrolled or gestured."""
    s = _input.seconds()
    return s if s is not None else _clock.seconds()


def _set_policy(policy: int) -> bool:
    global _setiopolicy
    try:
        if _setiopolicy is None:
            fn = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True).setiopolicy_np
            fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
            fn.restype = ctypes.c_int
            _setiopolicy = fn
        return _setiopolicy(_IOPOL_TYPE_DISK, _IOPOL_SCOPE_PROCESS, policy) == 0
    except Exception:  # noqa: BLE001 - no policy is the old behavior
        return False


def active() -> bool:
    return _active


def begin() -> None:
    """A gentle stage starts."""
    global _active, _policy_set, _yielded_s, _started
    with _lock:
        if _active:
            return
        _active = True
        _yielded_s = 0.0
        _started = time.perf_counter()
        policy = _POLICIES.get(str(config.STAGE_IOPOL).strip().lower())
        _policy_set = policy is not None and _set_policy(policy)
    idle = user_idle_seconds() if config.STAGE_YIELD_S > 0 else None
    yields = f"yields to input within {config.STAGE_YIELD_S:g}s" if idle is not None else "no idle clock"
    print(
        f"[stage] gentle: disk {config.STAGE_IOPOL if _policy_set else 'normal'}, "
        f"{config.STAGE_PACE_MS:g} ms per layer, {yields}",
        flush=True,
    )


def end() -> bool:
    """Back to full speed. True when a gentle stage was running."""
    global _active, _policy_set
    with _lock:
        if not _active:
            return False
        _active = False
        if _policy_set:
            _set_policy(_IOPOL_DEFAULT)
            _policy_set = False
        took, yielded = time.perf_counter() - _started, _yielded_s
    if yielded >= 0.5:
        print(f"[stage] done in {took:.1f}s, {yielded:.1f}s of it waiting for the user to pause", flush=True)
    return True


def boost() -> bool:
    """A request is waiting on the stage: finish it at full speed."""
    was = end()
    if was:
        print("[stage] boosted: a request is waiting", flush=True)
    return was


def pace() -> None:
    """Called once per MoE layer of a prefill, after the router sync.

    Waits while the user is active. No cap: under constant input the stage
    simply holds until they pause or send a request (boost), which is never
    worse than not staging at all.
    """
    if not _active:
        return
    if config.STAGE_PACE_MS > 0:
        time.sleep(config.STAGE_PACE_MS / 1000.0)
    wait_for_pause()


def wait_for_pause() -> None:
    """Hold while the user is active (a stage only). Also called between the
    expert groups of one layer, so a layer in progress backs off too."""
    global _yielded_s
    if not _active or config.STAGE_YIELD_S <= 0:
        return
    t0 = None
    while _active:
        idle = user_idle_seconds()
        if idle is None or idle >= config.STAGE_YIELD_S:
            break
        if t0 is None:
            t0 = time.perf_counter()
        time.sleep(min(0.25, max(0.05, config.STAGE_YIELD_S - idle)))
    if t0 is not None:
        _yielded_s += time.perf_counter() - t0
