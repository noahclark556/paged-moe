# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prefix files sealed at rest, for hosts that keep their data encrypted.

A saved prefix is the KV of the conversation it was cut from, plus its token
ids, so on disk it is as private as the chat itself. A host that encrypts its
own data hands the engine a 32-byte key on a pipe (PAGED_MOE_PREFIX_DISK_KEY_FD),
never in the environment or argv, and every prefix file is then written
sealed: AES-256-CTR with a fresh random nonce per file, then HMAC-SHA256 over
the whole file (encrypt-then-MAC), both from macOS CommonCrypto / hashlib, so
there is nothing extra to ship.

    magic (8) | nonce (16) | ciphertext | tag (32)

Writes stream through `Writer` a megabyte at a time and reads go through
`Reader`, which decrypts in place at any offset (mx.load seeks between
tensors), so neither holds a second copy of a file in RAM. `verify` checks the
tag in one pass over the file before anything in it is used; it replaces the
CRC pass a plain file gets.

A key that is asked for but cannot be read, or a Mac without CommonCrypto,
turns the prefix files off for this engine instead of writing them in the
clear. Without the knob nothing here runs.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import hmac
import os

from . import config

MAGIC = b"PMSEAL01"
_NONCE = 16
_TAG = 32
HEAD = len(MAGIC) + _NONCE
_CHUNK = 1 << 20

_state: dict | None = None
_lib = None


def _cc():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL("/usr/lib/system/libcommonCrypto.dylib")
        lib.CCCryptorCreateWithMode.argtypes = [ctypes.c_uint32] * 4 + [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.CCCryptorUpdate.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.CCCryptorRelease.argtypes = [ctypes.c_void_p]
        _lib = lib
    return _lib


def _derive(master: bytes, what: str) -> bytes:
    return hmac.new(master, f"pagedmoe prefix {what}".encode(), hashlib.sha256).digest()


def _load() -> dict:
    """{"on": bool, "ok": bool, ...keys}. Reads the key pipe once."""
    global _state
    if _state is not None:
        return _state
    fd = int(config.PREFIX_DISK_KEY_FD or -1)
    if fd < 0:
        _state = {"on": False, "ok": True}
        return _state
    master = b""
    try:
        while len(master) < 4096:
            got = os.read(fd, 4096)
            if not got:
                break
            master += got
    except OSError:
        master = b""
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        master = bytes.fromhex(master.decode("ascii").strip())
    except (UnicodeDecodeError, ValueError):
        master = b""
    ok = len(master) == 32
    if ok:
        try:
            _cc()
        except OSError:
            ok = False
    if not ok:
        print("[prefix-disk] encryption key missing or unusable; prefix files are off", flush=True)
        _state = {"on": True, "ok": False}
        return _state
    _state = {
        "on": True,
        "ok": True,
        "enc": _derive(master, "enc"),
        "mac": _derive(master, "mac"),
        "name": _derive(master, "name"),
        "id": _derive(master, "id")[:8].hex(),
    }
    return _state


def active() -> bool:
    """Files are sealed (a usable key was given)."""
    s = _load()
    return s["on"] and s["ok"]


def usable() -> bool:
    """False when sealing was asked for and cannot be done: no files at all."""
    s = _load()
    return s["ok"]


def key_id() -> str:
    return _load().get("id", "")


def stem(sha: str) -> str:
    """A file name for these ids. Keyed when sealed, so a name says nothing."""
    s = _load()
    if not active():
        return sha[:24]
    return hmac.new(s["name"], sha.encode(), hashlib.sha256).hexdigest()[:24]


def _cryptor(nonce: bytes, offset: int):
    blk, rem = divmod(offset, 16)
    iv = ((int.from_bytes(nonce, "big") + blk) % (1 << 128)).to_bytes(16, "big")
    key = _load()["enc"]
    ref = ctypes.c_void_p()
    kb = ctypes.create_string_buffer(key, len(key))
    ib = ctypes.create_string_buffer(iv, 16)
    # kCCEncrypt, kCCModeCTR, kCCAlgorithmAES, ccNoPadding, kCCModeOptionCTR_BE
    st = _cc().CCCryptorCreateWithMode(0, 4, 0, 0, ib, kb, len(key), None, 0, 0, 2, ctypes.byref(ref))
    if st != 0:
        raise OSError(f"CommonCrypto error {st}")
    if rem:
        _xor(ref, bytearray(rem))
    return ref


def _xor(ref, buf) -> None:
    """CTR in place on a writable buffer."""
    n = len(buf)
    if not n:
        return
    p = (ctypes.c_char * n).from_buffer(buf)
    moved = ctypes.c_size_t()
    st = _cc().CCCryptorUpdate(ref, p, n, p, n, ctypes.byref(moved))
    if st != 0 or moved.value != n:
        raise OSError(f"CommonCrypto error {st}")


def _release(ref) -> None:
    _cc().CCCryptorRelease(ref)


def seal_bytes(data: bytes) -> bytes:
    nonce = os.urandom(_NONCE)
    body = bytearray(data)
    ref = _cryptor(nonce, 0)
    try:
        _xor(ref, body)
    finally:
        _release(ref)
    head = MAGIC + nonce
    tag = hmac.new(_load()["mac"], head + body, hashlib.sha256).digest()
    return head + bytes(body) + tag


def open_bytes(blob: bytes) -> bytes | None:
    """The plain bytes, or None when it is not sealed under this key."""
    if len(blob) < HEAD + _TAG or not blob.startswith(MAGIC):
        return None
    tag = hmac.new(_load()["mac"], blob[:-_TAG], hashlib.sha256).digest()
    if not hmac.compare_digest(tag, blob[-_TAG:]):
        return None
    body = bytearray(blob[HEAD:-_TAG])
    ref = _cryptor(blob[len(MAGIC):HEAD], 0)
    try:
        _xor(ref, body)
    finally:
        _release(ref)
    return bytes(body)


def is_sealed(blob: bytes) -> bool:
    return blob.startswith(MAGIC)


class Writer:
    """A write-only file object for mx.save_safetensors: encrypts as it goes.
    `crc` is the CRC32 of the bytes on disk. Call `finish()` before closing."""

    closed = False

    def __init__(self, fd: int):
        import zlib

        self._zlib = zlib
        self._fd = fd
        self._nonce = os.urandom(_NONCE)
        self._ref = _cryptor(self._nonce, 0)
        self._mac = hmac.new(_load()["mac"], digestmod=hashlib.sha256)
        self._pos = 0
        self._buf = bytearray(_CHUNK)
        self.crc = 0
        self._raw(MAGIC + self._nonce)

    def _raw(self, data) -> None:
        self._mac.update(data)
        self.crc = self._zlib.crc32(data, self.crc)
        mv = memoryview(data)
        while mv:
            n = os.write(self._fd, mv)
            mv = mv[n:]

    def write(self, data) -> int:
        mv = memoryview(data).cast("B")
        total = len(mv)
        while mv:
            n = min(len(mv), _CHUNK)
            chunk = memoryview(self._buf)[:n]
            chunk[:] = mv[:n]
            _xor(self._ref, chunk)
            self._raw(chunk)
            mv = mv[n:]
        self._pos += total
        return total

    def tell(self) -> int:
        return self._pos

    def seek(self, off: int, whence: int = 0) -> int:
        if (whence, off) not in ((0, self._pos), (1, 0), (2, 0)):
            raise OSError("sealed writer only appends")
        return self._pos

    def flush(self) -> None:
        pass

    def finish(self) -> None:
        tag = self._mac.digest()
        self.crc = self._zlib.crc32(tag, self.crc)
        mv = memoryview(tag)
        while mv:
            n = os.write(self._fd, mv)
            mv = mv[n:]
        _release(self._ref)
        self._ref = None

    def abort(self) -> None:
        if self._ref is not None:
            _release(self._ref)
            self._ref = None


def verify(path) -> bool:
    """The tag matches: one F_NOCACHE pass over the file. Never raises."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return False
    try:
        if config.NOCACHE and hasattr(fcntl, "F_NOCACHE"):
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        size = os.fstat(fd).st_size
        if size < HEAD + _TAG:
            return False
        if os.pread(fd, len(MAGIC), 0) != MAGIC:
            return False
        mac = hmac.new(_load()["mac"], digestmod=hashlib.sha256)
        buf = bytearray(_CHUNK)
        mv = memoryview(buf)
        pos, end = 0, size - _TAG
        while pos < end:
            k = os.preadv(fd, [mv[: min(_CHUNK, end - pos)]], pos)
            if k <= 0:
                return False
            mac.update(mv[:k])
            pos += k
        return hmac.compare_digest(mac.digest(), os.pread(fd, _TAG, end))
    except OSError:
        return False
    finally:
        os.close(fd)


class Reader:
    """A read-only file object over a verified sealed file, for mx.load.
    Decrypts each read in place where it lands, at any offset."""

    closed = False

    def __init__(self, path):
        self._fd = os.open(str(path), os.O_RDONLY)
        if config.NOCACHE and hasattr(fcntl, "F_NOCACHE"):
            fcntl.fcntl(self._fd, fcntl.F_NOCACHE, 1)
        size = os.fstat(self._fd).st_size
        self._nonce = os.pread(self._fd, _NONCE, len(MAGIC))
        self._size = size - HEAD - _TAG
        self._pos = 0

    def readinto(self, b) -> int:
        mv = memoryview(b).cast("B")
        n = max(0, min(len(mv), self._size - self._pos))
        got = 0
        while got < n:
            k = os.preadv(self._fd, [mv[got: got + min(_CHUNK, n - got)]], HEAD + self._pos + got)
            if k <= 0:
                break
            got += k
        if got:
            ref = _cryptor(self._nonce, self._pos)
            try:
                _xor(ref, mv[:got])
            finally:
                _release(ref)
        self._pos += got
        return got

    def read(self, n: int = -1) -> bytes:
        n = self._size - self._pos if n is None or n < 0 else n
        buf = bytearray(n)
        k = self.readinto(buf)
        return bytes(buf[:k])

    def seek(self, off: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: self._size}[whence]
        self._pos = max(0, min(self._size, base + off))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def close(self) -> None:
        if not self.closed:
            os.close(self._fd)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
