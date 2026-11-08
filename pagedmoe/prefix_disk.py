# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The pinned front matter KV on disk, so a cold start skips that prefill.

The stable pin (prefix_cache.StablePrefixPin) holds the KV for the host's
front matter: system prompt, tool schemas, codebase map, memory. It lives in
RAM, so every engine start (app launch, model switch, OOM recovery) paid for
it again. On Flash that is 14k tokens and 40 s before the first new token.
This writes the pin out when it is taken and reads it back on a request whose
prompt starts with it.

A file is used only when every one of these matches, otherwise it is a miss
and the turn prefills exactly as before:

* fingerprint: weight files (name, size, mtime), config.json, the index and
  any model .py, the engine's own files, the mlx / mlx_lm versions, and every
  PAGED_MOE_* setting that can change what prefill writes (not paths,
  memory/timing knobs, or decode-only routing like PRUNE and FLOW).
* the token ids, stored in the file and compared in full.
* the cache layout: each layer's class and attributes against a fresh
  make_cache() of the loaded model, nested side caches included.
* a CRC32 of the whole file, checked before it is loaded.

A prompt that matches a file only part way (the memory block changed since it
was saved) still uses it when every layer is attention that can be cut back
(KVCache and subclasses with side caches, CacheList, a rotating cache that has
not wrapped): the file is loaded and cut to the shared tokens, side caches
included, which is what the in-memory pin does on a partial match. A
recurrent layout resumes from a checkpoint (below). Anything else needs the
whole file to match, so when the host marks where the memory block starts
(`stable_prefix_head_messages`) the server also saves a head file there,
and a memory edit resumes from it. Where the system prompt and tool schemas
end (`stable_prefix_shared_messages`, before the codebase map) it saves a
shared file, which every workspace starts with, so a new or changed
workspace resumes there. Each kind is capped separately so memory edits
never evict the head and one workspace never evicts the shared file.

A hybrid file also carries recurrent state checkpoints at fixed positions
inside it (PREFIX_DISK_CKPT_EVERY), taken inside the prefill passes that
build the file. A prompt that departs from the file part way (an update that
edits the system prompt or a tool schema) loads the file, trims attention back
to the last checkpoint before the change and puts that checkpoint's recurrent
state in, then prefills only from there. The checkpoint is checked the same way
as the file (CRC, fingerprint, the ids it was taken after, and every slot's
shape and dtype against the loaded cache); anything off is a miss and the turn
prefills from where it would have anyway. Checkpoint files are named by the
ids before them, so a newer file that starts with the same ids reuses them.

A file whose fingerprint no longer matches is deleted on the next lookup, and
the log says whether the engine, the model files or a setting changed (and
which setting), so a cold prefill after a restart is never silent. The one
exception is an update that changed only engine code (numerics_fp already
ignores comments, docstrings and log lines). Each file records how its last
prefill chunk or two ran: where they started, their sizes, any ids past the
file's end they covered, and on a recurrent layout the exact state they
started from. The first lookup that wants such a file replays those chunks
with the new engine and compares every value they write with the file. Equal
to the bit, the file is kept under the new fingerprint; any difference drops
every file from the old engine and the turn prefills as it would have. The
replay is one or two chunks, so it is only tried where it saves at least
twice that.

Save refuses anything it does not recognize (quantized KV, an attribute type
it cannot store, a side cache of the wrong length) and the pin just stays in
RAM as it always did. The KV in the file is the pin's own arrays, so a hit
resumes from the same state the in-memory pin would have given.

Reads and writes go around the file cache: written files are fsynced and
dropped, checks read with F_NOCACHE in MAX_IO_BYTES chunks, and the file is
dropped again after mx.load.

Every load, replay and save runs on generation_stream, the stream prefill and
decode run on, and a loaded cache is evaluated before it is handed back. A
lazy array built on the default stream and consumed on generation_stream is
the cross-stream wait that hangs a Metal command buffer for good (the Flash
indexer cache, cut back by a rewind, hung every replay check that way).

While a file is being loaded or saved, open.json in the slot names it. An
engine that dies in the middle (watchdog stall exit, SIGKILL, hard reboot)
leaves it behind, and the next engine drops that file, or stops saving that
prefix under that fingerprint, instead of running into the same hang again.
A SIGTERM whose GPU drain finishes (a model switch or settings restart, not a
hang) removes it first, so a restart mid-load keeps the file.

A host can also save a whole conversation (`stable_prefix_chat_messages`),
so reopening it after a restart reads the file instead of prefilling the
history. These are chat files: capped on their own (PREFIX_DISK_CHATS), tagged
with the host's chat id, and a new save of a chat replaces its older one. A
chat file is served as the request's cache, not pinned, so it is never held
twice in RAM. Tags the host lists in forget.json (a deleted chat) are removed
on the next load or save and at engine start.

With a key from the host (PAGED_MOE_PREFIX_DISK_KEY_FD) every file here is
sealed (seal.py): the KV, the token ids, checkpoints and the index, under names
keyed so they say nothing about the ids. The key id is part of the
fingerprint, so files from before encryption or under another key are dropped,
never read. A slot whose index cannot be read under this key goes at start.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import time
import zlib
from array import array
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache, _BaseCache, make_prompt_cache

from . import config, kvmem, live_config, numerics_fp, pagecache, seal
from .lookup_decode import _side_caches
from .safetensors_index import MAX_IO_BYTES

FORMAT = 1
_MIN_TOKENS = 64
# A partial hit reads the whole file, so it has to save more prefill than that.
_MIN_PARTIAL_GAIN = 256
_SUFFIX = ".safetensors"
_TOK_SUFFIX = ".tok"
_CK_PREFIX = "ck-"
_OPEN = "open.json"
# No load or save runs this long (the watchdog exits at 120 s), so an older
# marker is from a dead engine even when its pid was reused after a reboot.
_OPEN_MAX_AGE_S = 600
_NO_SAVE_MAX = 32
# A front file is the whole pinned front matter. A head file stops before the
# memory block, and a shared file before the first workspace block (map), for
# layouts that cannot trim a front file back to either.
KIND_FRONT = "front"
KIND_HEAD = "head"
KIND_SHARED = "shared"
# The system span of a request from another client (OpenAI API). Capped on
# its own so a harness's prompt never evicts the host's files.
KIND_CLIENT = "client"
# A whole conversation up to its last message, tagged with the host's chat id.
KIND_CHAT = "chat"
_KINDS = (KIND_FRONT, KIND_HEAD, KIND_SHARED, KIND_CLIENT, KIND_CHAT)
_FORGET = "forget.json"
# The kind of the file the last load served, for the caller to route it.
last_kind: str | None = None

STATS = {
    "saved": 0, "hits": 0, "partial": 0, "misses": 0, "rejected": 0, "skipped": 0,
    "verified": 0, "replay_failed": 0,
}

# Settings that cannot change the KV a prefill writes: timing, memory, paths,
# and decode-only routing. Prefill runs under streaming.exact_step(), which
# turns off PRUNE, WAIT_ABOVE, flow, ROUTE_TOP_K / TOP_P and the prune head, so
# those only shape decode. Everything else under PAGED_MOE_ goes into the
# fingerprint, so a new knob misses until it is listed here. Keep anything that
# moves prefill numerics (chunking, attention sub-chunk, KV bits) out of here.
_TIMING_ONLY = {
    "PAGED_MOE_PRUNE",
    "PAGED_MOE_WAIT_ABOVE",
    "PAGED_MOE_ROUTE_TOP_K",
    "PAGED_MOE_ROUTE_TOP_P",
    "PAGED_MOE_ROUTE_RENORM",
    "PAGED_MOE_VERIFY",
    "PAGED_MOE_VERIFY_SPANS",
    "PAGED_MOE_VERIFY_TOKENS",
    "PAGED_MOE_SELF_DRAFT",
    "PAGED_MOE_SELF_DRAFT_TOKENS",
    "PAGED_MOE_DRAFT_PRUNE",
    "PAGED_MOE_DRAFT_WAIT_ABOVE",
    "PAGED_MOE_PREFETCH_DEPTH",
    "PAGED_MOE_SPEC_INFLIGHT_MULT",
    "PAGED_MOE_READ_THREADS",
    "PAGED_MOE_READ_POOL_THREADS",
    "PAGED_MOE_CLEAR_GB",
    "PAGED_MOE_ACTIVATION_GB",
    "PAGED_MOE_PREFILL_CACHE_GB",
    "PAGED_MOE_PREFILL_SHRINK_TOKENS",
    "PAGED_MOE_PREFILL_SLAB",
    "PAGED_MOE_PREFILL_AHEAD",
    "PAGED_MOE_PREFILL_AHEAD_LOOSE_MB",
    "PAGED_MOE_PROMPT_CACHE_MOVE",
    "PAGED_MOE_KV_LAZY",
    "PAGED_MOE_KV_STORE_SLACK",
    "PAGED_MOE_MAX_CACHE_GB",
    "PAGED_MOE_CACHE_GB",
    "PAGED_MOE_HEADROOM_GB",
    "PAGED_MOE_KEEP_FREE",
    "PAGED_MOE_RAM_FRACTION",
    "PAGED_MOE_SIMULATE_RAM_GB",
    "PAGED_MOE_RESERVE_CTX",
    "PAGED_MOE_NATIVE_READ",
    "PAGED_MOE_NOCACHE",
    "PAGED_MOE_MODE",
    "PAGED_MOE_MITM",
    "PAGED_MOE_TRACE",
    "PAGED_MOE_NO_HOOK",
    "PAGED_MOE_PREDICT",
    "PAGED_MOE_SIDECAR",
    "PAGED_MOE_SIDECAR_HEAD_WRAP",
    "PAGED_MOE_SIDECAR_HEAD_PREFILL",
    "PAGED_MOE_SIDECAR_HEAD_RESIDENCY",
    "PAGED_MOE_SIDECAR_LOCK_HITS",
    "PAGED_MOE_SIDECAR_CORPUS",
    "PAGED_MOE_SIDECAR_MIN_PRECISION",
}
_TIMING_PREFIX = (
    "PAGED_MOE_PREFIX_DISK",
    "PAGED_MOE_PAGECACHE_",
    "PAGED_MOE_MEM_",
    "PAGED_MOE_HYBRID_CKPT",
    "PAGED_MOE_HYBRID_INPASS",
    "PAGED_MOE_FLOW",
    "PAGED_MOE_LOOKUP",
    "PAGED_MOE_FIDELITY",
    "PAGED_MOE_EARLY_DECODE",
    "PAGED_MOE_TTW_",
    "PAGED_MOE_PREDICT_",
    "PAGED_MOE_VERIFY_",
    "PAGED_MOE_SELF_DRAFT_",
    "PAGED_MOE_SIDECAR_",
    "PAGED_MOE_LFRU",
    "PAGED_MOE_SLAB",
    "PAGED_MOE_STAGING_",
    "PAGED_MOE_STAGE_",
    "PAGED_MOE_GPU_",
    "PAGED_MOE_WATCHDOG",
)
_TIMING_SUFFIX = ("_DIR", "_PORT", "_URL", "_ROOT", "_PYTHON", "_DEBUG", "_LOG")

_engine_fp: str | None = None
_model_fp: dict[str, dict[str, str]] = {}
_refused: set[str] = set()
_FP_PARTS = ("engine", "model", "settings")


class _Refuse(Exception):
    pass


def enabled() -> bool:
    return bool(config.PREFIX_DISK) and seal.usable()


def reset() -> None:
    """Forget cached fingerprints (tests, or after a model's files change)."""
    global _engine_fp
    _engine_fp = None
    _model_fp.clear()
    _refused.clear()


def root() -> Path:
    if config.PREFIX_DISK_DIR:
        return Path(config.PREFIX_DISK_DIR).expanduser()
    return Path(config.SIDECAR_DIR).expanduser().parent / "prefix_cache"


def _numerics_env() -> list[tuple[str, str]]:
    out = []
    for k, v in os.environ.items():
        if not k.startswith("PAGED_MOE_"):
            continue
        if k in _TIMING_ONLY or k.startswith(_TIMING_PREFIX) or k.endswith(_TIMING_SUFFIX):
            continue
        # "0" from a launcher and "0.0" from a live change are one setting.
        try:
            v = repr(float(v))
        except ValueError:
            pass
        out.append((k, v))
    return sorted(out)


def _engine_fingerprint() -> str:
    """mlx, mlx_lm and the engine code a prefill runs (numerics_fp), so an
    update that only touches decode or I/O keeps the saved files."""
    global _engine_fp
    if _engine_fp is None:
        import mlx_lm

        h = hashlib.sha256(f"mlx={mx.__version__} mlx_lm={mlx_lm.__version__}".encode())
        h.update(numerics_fp.current().encode())
        _engine_fp = h.hexdigest()
    return _engine_fp


def engine_part() -> str:
    return hashlib.sha256(f"format={FORMAT} engine={_engine_fingerprint()}".encode()).hexdigest()


STAMP = "engine.json"


def write_stamp() -> None:
    """Record the engine part of the fingerprint next to the files, so the
    host can tell which saved files this engine still serves without loading
    a model. Never raises."""
    if not enabled():
        return
    try:
        base = root()
        base.mkdir(parents=True, exist_ok=True)
        tmp = base / f"{STAMP}.tmp"
        tmp.write_text(json.dumps({"format": FORMAT, "engine": engine_part()}))
        os.replace(tmp, base / STAMP)
    except Exception as e:  # noqa: BLE001
        print(f"[prefix-disk] stamp not written: {e!r}", flush=True)


def _model_dir(model_key) -> Path | None:
    if not isinstance(model_key, tuple) or len(model_key) != 3:
        return None
    path, adapter, draft = model_key
    if adapter is not None or draft is not None:
        return None
    p = Path(str(path)).expanduser()
    return p.resolve() if p.is_dir() else None


def fingerprint_parts(model_dir: Path) -> dict[str, str]:
    """The fingerprint split by what changed it, so a miss can say why."""
    key = str(model_dir)
    parts = _model_fp.get(key)
    if parts is not None:
        return parts
    m = hashlib.sha256()
    for p in sorted(model_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix == _SUFFIX:
            st = p.stat()
            m.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}".encode())
        elif p.name in ("config.json", "model.safetensors.index.json") or p.suffix == ".py":
            m.update(p.name.encode())
            m.update(p.read_bytes())
    parts = {
        "engine": engine_part(),
        "model": m.hexdigest(),
        "settings": hashlib.sha256(json.dumps(_numerics_env()).encode()).hexdigest(),
    }
    if seal.active():
        parts["seal"] = seal.key_id()
    _model_fp[key] = parts
    return parts


def _fp_keys(parts: dict) -> tuple:
    # Plain files keep the fingerprint they always had.
    return _FP_PARTS + (("seal",) if "seal" in parts else ())


def fingerprint(model_dir: Path) -> str:
    parts = fingerprint_parts(model_dir)
    return hashlib.sha256("|".join(parts[k] for k in _fp_keys(parts)).encode()).hexdigest()


def _common_len(a: list, b: list) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _tok_sha(tokens) -> str:
    return hashlib.sha256(array("i", (int(t) for t in tokens)).tobytes()).hexdigest()


def _cls(obj) -> str:
    t = type(obj)
    return f"{t.__module__}.{t.__qualname__}"


def _is_members(v) -> bool:
    return isinstance(v, (tuple, list)) and bool(v) and all(isinstance(x, _BaseCache) for x in v)


def _members(c) -> list:
    """A layer's own caches: the members of a CacheList, else just itself."""
    inner = getattr(c, "caches", None)
    return list(inner) if _is_members(inner) else [c]


def _encode(obj, prefix: str, arrays: dict, n: int, nested: bool = False) -> dict:
    attrs: dict = {}
    kv_off = getattr(obj, "offset", None) if isinstance(obj, KVCache) else None
    if isinstance(obj, KVCache) and kv_off != n:
        raise _Refuse(f"{type(obj).__name__} offset {kv_off} != {n}")
    for name, v in vars(obj).items():
        key = f"{prefix}.{name}"
        if v is None:
            attrs[name] = {"t": "none"}
        elif isinstance(v, bool):
            attrs[name] = {"t": "bool", "v": v}
        elif isinstance(v, (int, float, str)):
            attrs[name] = {"t": type(v).__name__, "v": v}
        elif isinstance(v, mx.array):
            if kv_off is not None and name in ("keys", "values") and v.ndim == 4:
                v = v[..., :kv_off, :]
            if nested and v.ndim >= 2 and v.shape[1] != n:
                raise _Refuse(f"side cache {key} holds {v.shape[1]} positions, not {n}")
            arrays[key] = v
            attrs[name] = {"t": "arr"}
        elif isinstance(v, list) and all(x is None or isinstance(x, mx.array) for x in v):
            items = []
            for i, x in enumerate(v):
                if x is not None:
                    arrays[f"{key}.{i}"] = x
                items.append(None if x is None else "arr")
            attrs[name] = {"t": "list", "items": items}
        elif isinstance(v, _BaseCache) and not nested:
            attrs[name] = {"t": "cache", "cls": _cls(v), "attrs": _encode(v, key, arrays, n, True)["attrs"]}
        elif _is_members(v) and not nested:
            # CacheList (DSA: [latent kv, indexer keys]). Each member is a
            # whole layer cache in its own right, so it gets the layer rules.
            attrs[name] = {"t": "caches", "items": [_encode(x, f"{key}.{i}", arrays, n) for i, x in enumerate(v)]}
        else:
            raise _Refuse(f"cannot store {key} ({type(v).__name__})")
    return {"cls": _cls(obj), "attrs": attrs}


def _decode(obj, spec: dict, prefix: str, arrays: dict, used: set) -> None:
    if _cls(obj) != spec["cls"]:
        raise _Refuse(f"layer class {_cls(obj)} != {spec['cls']}")
    attrs = spec["attrs"]
    if not set(vars(obj)) <= set(attrs):
        raise _Refuse(f"{_cls(obj)} has attributes the file does not")
    for name, a in attrs.items():
        key = f"{prefix}.{name}"
        t = a["t"]
        if t == "none":
            v = None
        elif t in ("bool", "int", "float", "str"):
            v = a["v"]
        elif t == "arr":
            v = arrays[key]
            used.add(key)
        elif t == "list":
            v = []
            for i, it in enumerate(a["items"]):
                if it is None:
                    v.append(None)
                else:
                    v.append(arrays[f"{key}.{i}"])
                    used.add(f"{key}.{i}")
        elif t == "cache":
            sub = getattr(obj, name, None)
            if not isinstance(sub, _BaseCache):
                raise _Refuse(f"fresh cache has no side cache {key}")
            _decode(sub, {"cls": a["cls"], "attrs": a["attrs"]}, key, arrays, used)
            continue
        elif t == "caches":
            sub = getattr(obj, name, None)
            if not _is_members(sub) or len(sub) != len(a["items"]):
                raise _Refuse(f"fresh cache has a different member list at {key}")
            for i, (x, item) in enumerate(zip(sub, a["items"])):
                _decode(x, item, f"{key}.{i}", arrays, used)
            continue
        else:
            raise _Refuse(f"unknown attribute kind {t!r}")
        setattr(obj, name, v)


def _file_crc(path: Path) -> int:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        if config.NOCACHE and hasattr(fcntl, "F_NOCACHE"):
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        buf = bytearray(MAX_IO_BYTES)
        mv = memoryview(buf)
        crc = 0
        while True:
            k = os.readv(fd, [mv])
            if k <= 0:
                break
            crc = zlib.crc32(mv[:k], crc)
        return crc
    finally:
        os.close(fd)


def _file_ok(path: Path, crc) -> bool:
    """The file is whole: its tag when sealed, else its CRC."""
    if seal.active():
        return seal.verify(path)
    return _file_crc(path) == int(crc if crc is not None else -1)


def _save_arrays(tmp: Path, arrays: dict, meta: dict) -> int:
    """Write and fsync a safetensors file, sealed when a key is set. Returns
    the CRC32 of what is on disk."""
    if not seal.active():
        mx.save_safetensors(str(tmp), arrays, meta)
        fd = os.open(str(tmp), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return _file_crc(tmp)
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        w = seal.Writer(fd)
        try:
            mx.save_safetensors(w, arrays, meta)
            w.finish()
        finally:
            w.abort()
        os.fsync(fd)
    finally:
        os.close(fd)
    return w.crc


def _load_arrays(path: Path):
    if not seal.active():
        return mx.load(str(path), return_metadata=True)
    with seal.Reader(path) as r:
        arrays, meta = mx.load(r, format="safetensors", return_metadata=True)
        mx.eval(list(arrays.values()))
    return arrays, meta


def _read_blob(path: Path) -> bytes | None:
    """A small file (index, token ids) in plain bytes. None when it is
    missing or cannot be read under this key."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if seal.is_sealed(raw):
        return seal.open_bytes(raw) if seal.active() else None
    return None if seal.active() else raw


def _write_blob(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(seal.seal_bytes(data) if seal.active() else data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _model_slot(model_dir: Path) -> Path:
    return root() / hashlib.sha256(str(model_dir).encode()).hexdigest()[:16]


def _read_index(slot: Path) -> dict:
    raw = _read_blob(slot / "index.json")
    try:
        data = json.loads(raw) if raw is not None else None
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            return data
    except ValueError:
        pass
    return {"entries": []}


def _write_index(slot: Path, data: dict) -> None:
    _write_blob(slot / "index.json", json.dumps(data).encode())
    _gc_ckpts(slot, data)


_open_path: Path | None = None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _mark_open(slot: Path, rec: dict) -> None:
    """Name the file this engine is about to load or save. Written through
    to disk, since a hang can end in a hard reboot."""
    global _open_path
    slot.mkdir(parents=True, exist_ok=True)
    path = slot / _OPEN
    tmp = slot / f"{_OPEN}.tmp"
    body = json.dumps({**rec, "pid": os.getpid(), "at": time.time()}).encode()
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    _open_path = path


def _close_open() -> None:
    global _open_path
    path, _open_path = _open_path, None
    if path is not None:
        try:
            path.unlink()
        except OSError:
            pass


def stopped_cleanly() -> None:
    """The engine is exiting on request with its GPU work drained. A load or
    save cut short there is not a hang, so its file must not be dropped."""
    _close_open()


def _settle_open(slot: Path, data: dict) -> None:
    """A marker left by an engine that died loading or saving: drop that
    file, or stop saving that prefix under that fingerprint. A marker held
    by a live engine (not this one) is left alone."""
    path = slot / _OPEN
    try:
        rec = json.loads(path.read_text())
    except FileNotFoundError:
        return
    except (OSError, ValueError):
        rec = {}
    if not isinstance(rec, dict):
        rec = {}
    pid = int(rec.get("pid") or 0)
    fresh = time.time() - float(rec.get("at") or 0) < _OPEN_MAX_AGE_S
    if pid and pid != os.getpid() and fresh and _pid_alive(pid):
        return
    try:
        path.unlink()
    except OSError:
        pass
    if pid == os.getpid():
        return
    op = rec.get("op")
    if op == "load" and rec.get("file"):
        bad = [e for e in data["entries"] if e.get("file") == rec["file"]]
        for e in bad:
            _remove(slot, e)
        data["entries"] = [e for e in data["entries"] if e.get("file") != rec["file"]]
        _write_index(slot, data)
        STATS["rejected"] += len(bad)
        print(
            f"[prefix-disk] the last engine stopped while loading {rec['file']}; dropped it, prefilling",
            flush=True,
        )
    elif op == "save" and rec.get("tok_sha") and rec.get("fp"):
        skip = [x for x in data.get("no_save") or [] if isinstance(x, dict)]
        skip.append({"tok_sha": rec["tok_sha"], "fp": rec["fp"]})
        data["no_save"] = skip[-_NO_SAVE_MAX:]
        _write_index(slot, data)
        print(
            f"[prefix-disk] the last engine stopped while saving {str(rec['tok_sha'])[:24]}; "
            "not saving that prefix again under this engine",
            flush=True,
        )


def realize(cache) -> None:
    """Evaluate every array a cache holds, side caches and recurrent state
    included, on the current stream."""
    out: list = []

    def walk(v, depth: int) -> None:
        if depth > 8:
            return
        if isinstance(v, mx.array):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x, depth + 1)
        elif isinstance(v, _BaseCache):
            for x in vars(v).values():
                walk(x, depth + 1)

    walk(list(cache), 0)
    if out:
        mx.eval(out)


def _ck_refs(e: dict) -> list:
    ck = e.get("ckpts")
    out = [c for c in ck if isinstance(c, dict) and c.get("file")] if isinstance(ck, list) else []
    rp = _replay_ck(e)
    if rp is not None and all(c["file"] != rp["file"] for c in out):
        out.append(rp)
    return out


def _replay_ck(e: dict) -> dict | None:
    rp = e.get("replay")
    ck = rp.get("ck") if isinstance(rp, dict) else None
    return ck if isinstance(ck, dict) and ck.get("file") else None


def _entry_bytes(e: dict) -> int:
    return int(e.get("bytes", 0)) + sum(int(c.get("bytes", 0)) for c in _ck_refs(e))


def _gc_ckpts(slot: Path, data: dict) -> None:
    """Delete checkpoint and prefix files no entry names any more."""
    used = {c["file"] for e in data["entries"] for c in _ck_refs(e)}
    named = {e.get("file") for e in data["entries"]}
    try:
        for p in slot.glob(f"*{_SUFFIX}"):
            if ".part" in p.name:
                continue
            if p.name.startswith(_CK_PREFIX):
                if p.name not in used:
                    p.unlink(missing_ok=True)
            elif p.name not in named:
                p.unlink(missing_ok=True)
                _tok_path(slot, p.name).unlink(missing_ok=True)
    except OSError:
        pass


def _tok_path(slot: Path, file: str) -> Path:
    return slot / (file[: -len(_SUFFIX)] + _TOK_SUFFIX)


def _read_tokens(slot: Path, file: str) -> list | None:
    """The token ids saved next to a file. Only picks a candidate; the ids
    inside the checksummed file are what a hit is verified against."""
    raw = _read_blob(_tok_path(slot, file))
    if raw is None or len(raw) % 4:
        return None
    a = array("i")
    a.frombytes(raw)
    return a.tolist()


def _remove(slot: Path, entry: dict) -> None:
    file = entry.get("file")
    if not file:
        return
    for p in (slot / file, _tok_path(slot, file)):
        try:
            p.unlink()
        except OSError:
            pass


def _superseded_first(slot: Path, keep_file: str, ents: list) -> list:
    """`ents` (not holding keep_file) in eviction order: the one sharing the
    most tokens with the new file first, then least recently used.

    A workspace's front matter changes as its file map and memory grow, and
    each change saves a new file. The old one shares far more with the new
    file than another workspace's does (they part at the workspace path), so
    it goes first and one busy workspace cannot push out the others.
    """
    new = _read_tokens(slot, keep_file) or []

    def shared(e: dict) -> int:
        saved = _read_tokens(slot, e["file"])
        return _common_len(saved, new) if saved else 0

    return sorted(ents, key=lambda e: (-shared(e), e.get("used", 0)))


def _model_gone(model: Path) -> bool:
    """True when the model folder is missing on a disk that is there. An
    unplugged external drive is not a deleted model."""
    if model.is_dir():
        return False
    parts = model.parts
    if len(parts) >= 3 and parts[1] == "Volumes" and not Path(*parts[:3]).is_dir():
        return False
    return True


def _evict(keep_slot: Path, keep_file: str) -> None:
    base = root()
    slots = []
    for slot in base.iterdir() if base.is_dir() else []:
        if not slot.is_dir():
            continue
        if slot != keep_slot and (slot / "index.json").is_file() and _read_blob(slot / "index.json") is None:
            shutil.rmtree(slot, ignore_errors=True)
            continue
        data = _read_index(slot)
        # Keyed by path, so a moved or deleted model can never hit again.
        model = str(data.get("model") or "")
        if slot != keep_slot and model and _model_gone(Path(model)):
            shutil.rmtree(slot, ignore_errors=True)
            continue
        ents = [e for e in data["entries"] if (slot / e.get("file", "")).is_file()]
        ents.sort(key=lambda e: e.get("used", 0), reverse=True)
        # Capped per kind: every memory edit saves a new front file, and those
        # must not push out the head or shared file they all start with.
        for kind in _KINDS:
            cap = max(1, int(config.PREFIX_DISK_CHATS if kind == KIND_CHAT else config.PREFIX_DISK_PER_MODEL))
            same = [x for x in ents if x.get("kind", KIND_FRONT) == kind]
            if len(same) <= cap:
                continue
            if slot == keep_slot and any(x["file"] == keep_file for x in same):
                others = [x for x in same if x["file"] != keep_file]
                drop = _superseded_first(slot, keep_file, others)[: len(same) - cap]
            else:
                drop = same[cap:]
            for e in drop:
                _remove(slot, e)
        ents = [e for e in ents if (slot / e["file"]).is_file()]
        data["entries"] = ents
        _write_index(slot, data)
        slots.append((max((e.get("used", 0) for e in ents), default=0), slot, data))
    slots.sort(key=lambda s: s[0], reverse=True)
    n_models = max(1, int(config.PREFIX_DISK_MODELS))
    for _, slot, _ in slots[n_models:]:
        if slot != keep_slot:
            shutil.rmtree(slot, ignore_errors=True)
    cap_b = int(float(config.PREFIX_DISK_MAX_GB) * (1 << 30))
    # Each kept model's newest file of each kind goes last, so a big model's
    # second front file cannot cost a small model its only one.
    newest = set()
    for _, slot, d in slots[:n_models]:
        for kind in _KINDS:
            same = [e for e in d["entries"] if e.get("kind", KIND_FRONT) == kind]
            if same:
                newest.add((str(slot), max(same, key=lambda e: e.get("used", 0))["file"]))
    flat = [(e.get("used", 0), slot, e) for _, slot, d in slots[:n_models] for e in d["entries"]]
    flat.sort(key=lambda x: ((str(x[1]), x[2]["file"]) in newest, x[0]))
    total = sum(_entry_bytes(e) for _, _, e in flat)
    for _, slot, e in flat:
        if total <= cap_b:
            break
        if slot == keep_slot and e["file"] == keep_file:
            continue
        _remove(slot, e)
        total -= _entry_bytes(e)
        data = _read_index(slot)
        data["entries"] = [x for x in data["entries"] if x.get("file") != e["file"]]
        _write_index(slot, data)


def _label(kind: str) -> str:
    if kind == KIND_HEAD:
        return "front matter head (before memory)"
    if kind == KIND_SHARED:
        return "shared front matter (system + tools)"
    if kind == KIND_CLIENT:
        return "client system prompt"
    if kind == KIND_CHAT:
        return "conversation"
    return "front matter"


def _forgotten() -> set:
    try:
        raw = json.loads((root() / _FORGET).read_text())
    except (OSError, ValueError):
        return set()
    return {str(t) for t in raw} if isinstance(raw, list) else set()


def _apply_forget(slot: Path, data: dict, gone: set | None = None) -> None:
    """Remove files whose tag the host listed in forget.json."""
    gone = _forgotten() if gone is None else gone
    if not gone:
        return
    drop = [e for e in data["entries"] if e.get("tag") is not None and str(e["tag"]) in gone]
    if not drop:
        return
    for e in drop:
        _remove(slot, e)
    names = {e.get("file") for e in drop}
    data["entries"] = [e for e in data["entries"] if e.get("file") not in names]
    _write_index(slot, data)
    print(f"[prefix-disk] removed {len(drop)} file(s) of deleted chats", flush=True)


def housekeep() -> None:
    """At engine start: apply forget.json to every model, and drop any slot
    whose index this engine cannot read (sealed under another key, or plain
    while sealing is on). Never raises."""
    if not enabled():
        return
    try:
        gone = _forgotten()
        base = root()
        if not base.is_dir():
            return
        for slot in base.iterdir():
            if not (slot.is_dir() and (slot / "index.json").is_file()):
                continue
            if _read_blob(slot / "index.json") is None:
                shutil.rmtree(slot, ignore_errors=True)
                print(f"[prefix-disk] dropped {slot.name}: saved under another key or without encryption", flush=True)
                continue
            if gone:
                _apply_forget(slot, _read_index(slot), gone)
    except Exception as e:  # noqa: BLE001
        print(f"[prefix-disk] housekeeping skipped: {e!r}", flush=True)


def _attn_rewinds(c) -> bool:
    """Attention that can be cut back to any shorter length: KVCache (and
    subclasses with side caches), a rotating cache that has not wrapped, or a
    CacheList of those. A resumed 8-bit layer stands in for a KVCache."""
    if isinstance(c, ArraysCache):
        return False
    if isinstance(c, kvmem.ResumedKVCache):
        return True
    return all(isinstance(m, (KVCache, RotatingKVCache)) and m.is_trimmable() for m in _members(c))


def trims_exactly(cache) -> bool:
    """True when every layer is attention that rewinds (_attn_rewinds), so a
    file can serve a prompt that shares only its head with no checkpoint:
    load cuts it back to the shared tokens, side caches included."""
    return bool(cache) and all(_attn_rewinds(c) for c in cache)


def ckpt_every() -> int:
    return max(0, int(config.PREFIX_DISK_CKPT_EVERY)) if enabled() else 0


def ckpt_positions(lo: int, hi: int) -> list[int]:
    """Checkpoint positions in (lo, hi). A fixed grid, thinned to at most
    PREFIX_DISK_CKPT_MAX, so a newer file over the same ids lands on the
    same positions and reuses the files already there."""
    step = ckpt_every()
    cap = max(0, int(config.PREFIX_DISK_CKPT_MAX))
    if step <= 0 or cap <= 0 or hi <= step:
        return []
    grid = list(range(step, hi, step))
    k = -(-len(grid) // cap)
    return [p for j, p in enumerate(grid, 1) if j % k == 0 and lo < p and p >= _MIN_TOKENS]


def _recurrent_layout(cache) -> dict | None:
    """{layer: [(shape, dtype) or None per slot]} for the recurrent layers
    ({} when there are none), or None when some layer can neither rewind nor
    take a checkpoint."""
    out = {}
    for i, c in enumerate(cache):
        if isinstance(c, ArraysCache):
            if getattr(c, "left_padding", None) is not None or getattr(c, "lengths", None) is not None:
                return None
            out[i] = [None if x is None else (tuple(x.shape), str(x.dtype)) for x in c.cache]
        elif not _attn_rewinds(c):
            return None
    return out


def _slots_fit(layout: dict, slots: dict) -> bool:
    if set(slots) != set(layout):
        return False
    for i, want in layout.items():
        got = slots[i]
        if len(got) != len(want):
            return False
        for x, w in zip(got, want):
            if (x is None) != (w is None):
                return False
            if x is not None and (tuple(x.shape), str(x.dtype)) != w:
                return False
    return True


def _write_ckpt(slot: Path, fp: str, tokens: list, at: int, slots: dict, stem: str | None = None) -> dict:
    sha = _tok_sha(tokens[:at])
    stem = stem or f"{_CK_PREFIX}{seal.stem(sha)}"
    name = f"{stem}{_SUFFIX}"
    arrays = {f"L{i}.s{j}": x for i, st in slots.items() for j, x in enumerate(st) if x is not None}
    meta = {
        "format": str(FORMAT),
        "fp": fp,
        "at": str(at),
        "tok_sha": sha,
        "slots": json.dumps({str(i): [x is not None for x in st] for i, st in slots.items()}),
    }
    tmp = slot / f"{stem}.part{_SUFFIX}"
    try:
        crc = _save_arrays(tmp, arrays, meta)
        size = tmp.stat().st_size
        pagecache.drop([str(tmp)])
        os.replace(tmp, slot / name)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {"at": at, "file": name, "tok_sha": sha, "crc": crc, "bytes": size, "fp": fp}


def _save_ckpts(slot: Path, data: dict, fp: str, tokens: list, cache, ckpts: dict | None) -> list:
    """Checkpoint records for a new file over `tokens`: the ones taken this
    pass, plus any an older file already holds at a position these ids
    share. Never raises; a checkpoint that does not fit is left out."""
    n = len(tokens)
    layout = _recurrent_layout(cache)
    if not layout:
        return []
    have = {}
    for e in data["entries"]:
        if e.get("fp") != fp:
            continue
        for c in _ck_refs(e):
            if (slot / c["file"]).is_file():
                have.setdefault(c.get("tok_sha"), c)
    out: dict[int, dict] = {}
    for c in have.values():
        at = int(c.get("at", 0))
        if _MIN_TOKENS <= at < n and c.get("tok_sha") == _tok_sha(tokens[:at]):
            out[at] = dict(c)
    for at, slots in sorted((ckpts or {}).items()):
        at = int(at)
        if at in out or not (_MIN_TOKENS <= at < n) or not slots:
            continue
        if not _slots_fit(layout, slots):
            print(f"[prefix-disk] checkpoint @ {at} skipped: slot layout differs", flush=True)
            continue
        old = have.get(_tok_sha(tokens[:at]))
        if old is not None:
            out[at] = dict(old)
            continue
        try:
            out[at] = _write_ckpt(slot, fp, tokens, at, slots)
        except Exception as e:  # noqa: BLE001
            print(f"[prefix-disk] checkpoint @ {at} not saved: {e!r}", flush=True)
    return [out[a] for a in sorted(out)]


def _replay_record(slot: Path, fp: str, tokens: list, tok_sha: str, cache, replay: dict) -> dict:
    """What a later engine replays to check it still writes this file: the
    last prefill chunk(s) before its end, as run, and on a recurrent layout
    the exact state they started from. Raises _Refuse when it cannot be
    replayed exactly."""
    n = len(tokens)
    lo = int(replay["lo"])
    sizes = [int(s) for s in replay["sizes"]]
    extra = [int(t) for t in replay.get("extra") or ()]
    hi = lo + sum(sizes)
    if not sizes or min(sizes) <= 0 or not (0 <= lo < n <= hi) or len(extra) != hi - n:
        raise _Refuse("replay span does not cover the end of the file")
    if lo == 0:
        raise _Refuse("the replay would be the whole file")
    layout = _recurrent_layout(cache)
    if layout is None:
        raise _Refuse("layout cannot be rewound for a replay")
    rec: dict = {"lo": lo, "sizes": sizes, "extra": extra}
    if layout:
        if hi != n:
            raise _Refuse("recurrent state runs past the end of the file")
        slots = replay.get("slots")
        if not slots or not _slots_fit(layout, slots):
            raise _Refuse("no exact recurrent state at the replay start")
        # The state at lo depends on how the span before it was chunked, so it
        # belongs to this file only.
        rec["ck"] = _write_ckpt(slot, fp, tokens, lo, slots, stem=f"{_CK_PREFIX}{seal.stem(tok_sha)}-r")
    return rec


def _replay_check(slot: Path, entry: dict, cache, stored: list, replay_fn) -> bool:
    """Run the file's last chunk(s) again with this engine and compare what
    it writes with what the file holds, bit for bit. On a match `cache` is
    left exactly as loaded; on anything else it is garbage."""
    rec = entry["replay"]
    lo, sizes, extra = int(rec["lo"]), [int(s) for s in rec["sizes"]], [int(t) for t in rec.get("extra") or ()]
    n = len(stored)
    ids = stored[lo:n] + extra
    if lo <= 0:
        return False
    want = _positions(cache, lo, n)
    layout = _recurrent_layout(cache)
    if layout is None:
        return False
    states: dict = {}
    if layout:
        ref = _replay_ck(entry)
        if ref is None:
            return False
        at, states = _load_ckpt(slot, ref, entry["fp"], stored)
        if at != lo:
            return False
    at_end = {i: list(c.cache) for i, c in enumerate(cache) if isinstance(c, ArraysCache)}
    _rewind(cache, lo, states)
    replay_fn(cache, ids, sizes)
    got = _positions(cache, lo, n)
    if not _same_arrays(got, want):
        return False
    _rewind(cache, n, at_end)
    return True


def _rewind(cache, at: int, states: dict) -> None:
    """Rewind a loaded cache to `at`: attention (and its side caches) cut
    back, recurrent state replaced by `states` ({} on a layout with none).
    Checks everything before changing any layer; raises _Refuse when it does
    not fit."""
    layout = _recurrent_layout(cache)
    if layout is None or not _slots_fit(layout, states):
        raise _Refuse("checkpoint slots do not fit the cache")
    attn = [m for c in cache if not isinstance(c, ArraysCache) for m in _members(c)]
    for m in attn:
        if int(getattr(m, "offset", -1)) < at:
            raise _Refuse("attention holds fewer positions than the checkpoint")
        for s in _side_caches(m):
            st = s.state
            if not isinstance(st, mx.array) or st.ndim < 2 or st.shape[1] < at:
                raise _Refuse("side cache holds fewer positions than the checkpoint")
    for i, c in enumerate(cache):
        if isinstance(c, ArraysCache):
            c.cache = list(states[i])
    for m in attn:
        k = int(m.offset) - at
        if k and m.trim(k) != k:
            raise _Refuse("attention trim refused")
        for s in _side_caches(m):
            s.state = s.state[:, :at]
        if int(m.offset) != at:
            raise _Refuse(f"rewind to {at} left offset {m.offset}")
    realize(cache)


def _positions(cache, lo: int, hi: int) -> list:
    """Copies of what `cache` holds for positions [lo, hi), plus the
    recurrent state, to compare two caches by. Raises _Refuse on a layer it
    cannot read that way."""

    def own(x):
        # A slice can share the KV buffer, and then the replay's writes into
        # that buffer would copy all of it. x * 1 is exact and gets its own.
        return x * 1

    out = []
    for c in cache:
        if isinstance(c, ArraysCache):
            out.extend(c.cache)
            continue
        for m in _members(c):
            if not isinstance(m, (KVCache, RotatingKVCache)) or not m.is_trimmable():
                raise _Refuse(f"cannot compare a {type(m).__name__} by position")
            for v in (m.keys, m.values):
                if not isinstance(v, mx.array) or v.ndim != 4 or v.shape[-2] < hi:
                    raise _Refuse("attention holds fewer positions than the replay")
                out.append(own(v[..., lo:hi, :]))
            for s in _side_caches(m):
                st = s.state
                if not isinstance(st, mx.array) or st.ndim < 2 or st.shape[1] < hi:
                    raise _Refuse("side cache holds fewer positions than the replay")
                out.append(own(st[:, lo:hi]))
    mx.eval([x for x in out if x is not None])
    return out


def _same_arrays(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x is None or y is None:
            if (x is None) != (y is None):
                return False
            continue
        if x.shape != y.shape or x.dtype != y.dtype or not bool(mx.array_equal(x, y, equal_nan=True)):
            return False
    return True


def _load_ckpt(slot: Path, ref: dict, fp: str, tokens: list) -> tuple[int, dict]:
    path = slot / ref["file"]
    if not path.is_file() or path.stat().st_size != int(ref.get("bytes", -1)):
        raise _Refuse("size changed")
    if not _file_ok(path, ref.get("crc")):
        raise _Refuse("checksum mismatch")
    arrays, meta = _load_arrays(path)
    mx.eval(list(arrays.values()))
    pagecache.drop([str(path)])
    at = int(ref["at"])
    if meta.get("format") != str(FORMAT) or meta.get("fp") != ref.get("fp", fp):
        raise _Refuse("fingerprint changed")
    if meta.get("at") != str(at) or meta.get("tok_sha") != ref.get("tok_sha"):
        raise _Refuse("header does not match the index")
    if meta["tok_sha"] != _tok_sha(tokens[:at]):
        raise _Refuse("token ids differ")
    states = {}
    for key, flags in json.loads(meta["slots"]).items():
        i = int(key)
        states[i] = [arrays.pop(f"L{i}.s{j}") if f else None for j, f in enumerate(flags)]
    if arrays:
        raise _Refuse("file holds arrays the layout does not use")
    return at, states


def supports(model_key) -> bool:
    """Files can be saved for this key: enabled, no adapter or draft, and no
    save refused since start (a refused layout would cost the head stop on
    every cold prefill for nothing)."""
    return enabled() and _model_dir(model_key) is not None and str(model_key) not in _refused


def has(model_key, tokens) -> bool:
    """A current file holds exactly `tokens`. Never raises."""
    if not enabled():
        return False
    try:
        model_dir = _model_dir(model_key)
        if model_dir is None:
            return False
        fp, tok_sha = fingerprint(model_dir), _tok_sha(tokens)
        slot = _model_slot(model_dir)
        return any(
            e.get("fp") == fp and e.get("tok_sha") == tok_sha and (slot / e.get("file", "")).is_file()
            for e in _read_index(slot)["entries"]
        )
    except Exception:  # noqa: BLE001
        return False


def save(
    model_key,
    tokens,
    cache,
    model,
    budget: int,
    kind: str = KIND_FRONT,
    ckpts: dict | None = None,
    replay: dict | None = None,
    tag: str | None = None,
) -> bool:
    """Write the KV for `tokens` to disk. Never raises; False when skipped.

    `cache` must hold exactly `tokens`; it is read, not copied or changed.
    `ckpts` is {position: {layer: [slots]}}, the recurrent state after that
    many of `tokens` (hybrid layouts only, see the module doc).
    `replay` is how the end of the file was prefilled: {"lo", "sizes",
    "extra", "slots"}, the chunks from position lo, any ids past the end
    those chunks ran over, and the recurrent state at lo. A file saved
    without it is dropped on an engine change instead of checked.
    `tag` names the host's chat for a chat file; it replaces that chat's
    older file and is how forget.json finds it.
    """
    if not enabled() or live_config.kv_stale():
        return False
    if kvmem.has_resumed(cache):
        # Writing it means building the fp16 copy ResumedKVCache exists to
        # avoid. Not a refusal: the next cold prefill of this prefix saves it.
        STATS["skipped"] += 1
        return False
    try:
        with mx.stream(generation_stream):
            return _save(model_key, list(tokens), cache, model, budget, kind, ckpts, replay, tag)
    except _Refuse as e:
        STATS["skipped"] += 1
        # A chat refused for its size says nothing about the layout, and must
        # not stop the front matter from being saved.
        if kind != KIND_CHAT:
            _refused.add(str(model_key))
        print(f"[prefix-disk] not saved: {e}", flush=True)
    except Exception as e:  # noqa: BLE001 - a failed write must not touch the turn
        STATS["skipped"] += 1
        print(f"[prefix-disk] save failed: {e!r}", flush=True)
    finally:
        _close_open()
    return False


def _save(
    model_key,
    tokens: list,
    cache,
    model,
    budget: int,
    kind: str,
    ckpts: dict | None = None,
    replay: dict | None = None,
    tag: str | None = None,
) -> bool:
    if kind not in _KINDS:
        raise _Refuse(f"unknown kind {kind!r}")
    n = len(tokens)
    if n < _MIN_TOKENS:
        return False
    model_dir = _model_dir(model_key)
    if model_dir is None:
        return False
    fp = fingerprint(model_dir)
    tok_sha = _tok_sha(tokens)
    slot = _model_slot(model_dir)
    data = _read_index(slot)
    _settle_open(slot, data)
    _apply_forget(slot, data)
    if tag is not None and str(tag) in _forgotten():
        return False
    now = time.time()
    for e in data["entries"]:
        if e.get("fp") == fp and e.get("tok_sha") == tok_sha and (slot / e["file"]).is_file():
            e["used"] = now
            if tag is not None:
                e["tag"] = str(tag)
            _write_index(slot, data)
            return False
    for x in data.get("no_save") or []:
        if isinstance(x, dict) and x.get("tok_sha") == tok_sha and x.get("fp") == fp:
            STATS["skipped"] += 1
            return False
    _mark_open(slot, {"op": "save", "tok_sha": tok_sha, "fp": fp})

    fresh = make_prompt_cache(model)
    if len(fresh) != len(cache):
        raise _Refuse(f"{len(cache)} cache layers, model makes {len(fresh)}")
    arrays: dict = {}
    layers = []
    for i, (c, f) in enumerate(zip(cache, fresh)):
        if _cls(c) != _cls(f):
            raise _Refuse(f"layer {i} is {_cls(c)}, model makes {_cls(f)}")
        layers.append(_encode(c, f"L{i}", arrays, n))
    nbytes = sum(int(a.nbytes) for a in arrays.values())
    cap_b = int(float(config.PREFIX_DISK_MAX_GB) * (1 << 30))
    if nbytes > cap_b or nbytes * 2 > budget:
        raise _Refuse(f"{nbytes / 1e9:.2f} GB is over the disk cap or half the prompt budget")
    slot.mkdir(parents=True, exist_ok=True)
    ck_bytes = sum(
        int(x.nbytes) for st in (ckpts or {}).values() for xs in (st or {}).values() for x in xs if x is not None
    )
    if ck_bytes and nbytes + ck_bytes > cap_b:
        ckpts, ck_bytes = None, 0
    if shutil.disk_usage(slot).free < nbytes * 2 + ck_bytes + (2 << 30):
        raise _Refuse("not enough free disk")

    t0 = time.perf_counter()
    arrays["__tokens__"] = mx.array(tokens, dtype=mx.int32)
    meta = {
        "format": str(FORMAT),
        "fp": fp,
        "n": str(n),
        "tok_sha": tok_sha,
        "layers": json.dumps(layers),
    }
    stem = seal.stem(tok_sha)
    name = f"{stem}{_SUFFIX}"
    tmp = slot / f"{stem}.part{_SUFFIX}"
    final = slot / name
    try:
        mx.eval(list(arrays.values()))
        crc = _save_arrays(tmp, arrays, meta)
        size = tmp.stat().st_size
        pagecache.drop([str(tmp)])
        os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    try:
        _write_blob(_tok_path(slot, name), array("i", tokens).tobytes())
    except OSError:
        pass
    data = _read_index(slot)
    refs = _save_ckpts(slot, data, fp, tokens, cache, ckpts)
    rec = None
    if replay:
        try:
            rec = _replay_record(slot, fp, tokens, tok_sha, cache, replay)
        except Exception as e:  # noqa: BLE001 - the file is still good without it
            print(f"[prefix-disk] no engine check for this file: {e}", flush=True)
    data["model"] = str(model_dir)
    data["entries"] = [e for e in data["entries"] if e.get("file") != name]
    if tag is not None:
        older = [e for e in data["entries"] if e.get("kind") == kind and e.get("tag") == str(tag)]
        for e in older:
            _remove(slot, e)
        data["entries"] = [e for e in data["entries"] if e not in older]
    entry = {
        "file": name,
        "kind": kind,
        "n": n,
        "tok_sha": tok_sha,
        "fp": fp,
        "fp_file": fp,
        "parts": fingerprint_parts(model_dir),
        "settings": dict(_numerics_env()),
        "crc": crc,
        "bytes": size,
        "used": now,
    }
    if refs:
        entry["ckpts"] = refs
    if rec:
        entry["replay"] = rec
    if tag is not None:
        entry["tag"] = str(tag)
    data["entries"].append(entry)
    _write_index(slot, data)
    _evict(slot, name)
    STATS["saved"] += 1
    ck = f", {len(refs)} checkpoint(s) at {[r['at'] for r in refs]}" if refs else ""
    print(
        f"[prefix-disk] saved {_label(kind)} {n} tok ({size / 1e9:.2f} GB{ck}) "
        f"in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )
    return True


def load(model_key, tokens, model, have: int, budget: int, replay=None):
    """(snap_tokens, cache) for the file sharing the most tokens, or None.

    A file `tokens` starts with is served whole. A file that diverges part way
    is served cut back to the shared tokens (attention layouts) or to its last
    checkpoint before the change (recurrent layouts), if that beats `have`
    (what RAM already serves) by _MIN_PARTIAL_GAIN. One token is always left
    for decode. Never raises.

    `replay(cache, ids, sizes)` prefills `ids` into `cache` in chunks of
    `sizes`, exactly as the server's prefill does. With it, a file saved by
    another engine is checked by replaying its last chunk(s) before use and
    kept only if every value matches; without it such files are dropped.
    """
    global last_kind
    last_kind = None
    if not enabled():
        return None
    try:
        with mx.stream(generation_stream):
            return _load(model_key, list(tokens), model, have, budget, replay)
    except Exception as e:  # noqa: BLE001 - a bad file is a miss, never a failed turn
        STATS["rejected"] += 1
        print(f"[prefix-disk] load skipped: {e!r}", flush=True)
        return None
    finally:
        _close_open()


def _checkable(e: dict, parts: dict) -> bool:
    """Saved by another engine for the same model files and settings, with a
    record of how its end was prefilled, so the new engine can replay that and
    keep the file if it writes the same KV."""
    old = e.get("parts")
    return (
        isinstance(old, dict)
        and old.get("model") == parts["model"]
        and old.get("settings") == parts["settings"]
        and old.get("seal") == parts.get("seal")
        and isinstance(e.get("replay"), dict)
    )


def _drop_stale(slot: Path, data: dict, fp: str, parts: dict, keep_checkable: bool = False) -> None:
    """Delete files from another engine / model / settings and say which.
    With `keep_checkable`, files only the engine changed under stay for a
    replay check on first use."""
    stale = [
        e for e in data["entries"] if e.get("fp") != fp and not (keep_checkable and _checkable(e, parts))
    ]
    if not stale:
        return
    why = set()
    now_env = dict(_numerics_env())
    for e in stale:
        old = e.get("parts")
        if not isinstance(old, dict):
            why.add("saved by an older build")
            continue
        for k in _FP_PARTS + ("seal",):
            if old.get(k) == parts.get(k):
                continue
            if k == "seal":
                why.add("encryption changed")
                continue
            was = e.get("settings")
            if k == "settings" and isinstance(was, dict):
                keys = sorted(x for x in set(was) | set(now_env) if was.get(x) != now_env.get(x))
                shown = ", ".join(keys[:4]) + (", ..." if len(keys) > 4 else "")
                why.add(f"settings changed: {shown}" if keys else "settings changed")
            else:
                why.add({"engine": "engine changed", "model": "model files changed"}.get(k, f"{k} changed"))
    for e in stale:
        _remove(slot, e)
    gone = {e.get("file") for e in stale}
    data["entries"] = [e for e in data["entries"] if e.get("file") not in gone]
    _write_index(slot, data)
    STATS["rejected"] += len(stale)
    print(
        f"[prefix-disk] dropped {len(stale)} stale file(s) "
        f"({'; '.join(sorted(why)) or 'fingerprint changed'}); prefilling",
        flush=True,
    )


def _say_cold_miss(slot: Path, data: dict, tokens: list) -> None:
    """A cold start with files on disk and no hit: say how close the nearest
    one came, so a prompt that changed near the top (a template kwarg like
    reasoning effort) is not a silent full prefill."""
    best_n, best_file, best_len = -1, None, 0
    for e in data["entries"]:
        saved = _read_tokens(slot, e["file"])
        if not saved:
            continue
        k = _common_len(saved, tokens)
        if k > best_n:
            best_n, best_file, best_len = k, e["file"], len(saved)
    if best_file is None:
        return
    print(
        f"[prefix-disk] no saved prefix fits this prompt; the closest ({best_file[:12]}, "
        f"{best_len} tok) parts from it at token {best_n}; prefilling",
        flush=True,
    )


def _replay_tokens(e: dict) -> int:
    rp = e.get("replay")
    try:
        return sum(int(s) for s in rp["sizes"])
    except (TypeError, KeyError, ValueError):
        return 1 << 62


def _load(model_key, tokens: list, model, have: int, budget: int, replay=None):
    model_dir = _model_dir(model_key)
    if model_dir is None:
        return None
    slot = _model_slot(model_dir)
    data = _read_index(slot)
    _settle_open(slot, data)
    _apply_forget(slot, data)
    if not data["entries"]:
        return None
    fp = fingerprint(model_dir)
    parts = fingerprint_parts(model_dir)
    _drop_stale(slot, data, fp, parts, keep_checkable=replay is not None)
    if not data["entries"]:
        STATS["misses"] += 1
        return None
    limit = len(tokens) - 1
    floor = max(have, _MIN_TOKENS - 1)
    rewinds = trims_exactly(make_prompt_cache(model))
    best, keep, best_ck, best_score = None, 0, None, 0
    for e in data["entries"]:
        n = int(e.get("n", 0))
        current = e.get("fp") == fp
        if n <= floor or not (current or (replay is not None and _checkable(e, parts))):
            continue
        ck = None
        if n <= limit and e.get("tok_sha") == _tok_sha(tokens[:n]):
            k = n
        elif rewinds or _ck_refs(e):
            saved = _read_tokens(slot, e["file"])
            if saved is None or len(saved) != n:
                continue
            k = min(_common_len(saved, tokens), limit)
            if not rewinds:
                # The recurrent state cannot be trimmed: resume from the last
                # checkpoint at or before the first id that differs.
                fits = [c for c in _ck_refs(e) if int(c.get("at", 0)) <= k]
                if not fits:
                    continue
                ck = max(fits, key=lambda c: int(c["at"]))
                k = int(ck["at"])
            if k < _MIN_TOKENS or k - have < _MIN_PARTIAL_GAIN:
                continue
        else:
            continue
        # A file from another engine costs a replay of its last chunk(s)
        # first, so it has to save at least twice that.
        cost = 0 if current else _replay_tokens(e)
        if k - have < 2 * cost:
            continue
        if k > floor and k - cost > best_score:
            best, keep, best_ck, best_score = e, k, ck, k - cost
    if best is None:
        STATS["misses"] += 1
        if have < _MIN_TOKENS:
            _say_cold_miss(slot, data, tokens)
        return None
    path = slot / best["file"]
    n = int(best["n"])
    if int(best.get("bytes", 0)) * 2 > budget:
        STATS["skipped"] += 1
        return None

    def reject(why: str):
        STATS["rejected"] += 1
        print(f"[prefix-disk] rejected {best['file']}: {why}; prefilling", flush=True)
        _remove(slot, best)
        data["entries"] = [x for x in data["entries"] if x.get("file") != best["file"]]
        _write_index(slot, data)
        return None

    t0 = time.perf_counter()
    if not path.is_file() or path.stat().st_size != int(best.get("bytes", -1)):
        return reject("size changed")
    _mark_open(slot, {"op": "load", "file": best["file"]})
    if not _file_ok(path, best.get("crc")):
        return reject("checksum mismatch")
    arrays, meta = _load_arrays(path)
    mx.eval(list(arrays.values()))
    pagecache.drop([str(path)])
    if meta.get("format") != str(FORMAT) or meta.get("fp") != best.get("fp_file", best.get("fp")):
        return reject("fingerprint changed")
    if meta.get("n") != str(n) or meta.get("tok_sha") != best["tok_sha"]:
        return reject("header does not match the index")
    stored = arrays.pop("__tokens__", None)
    stored = None if stored is None else stored.tolist()
    if stored is None or len(stored) != n or stored[:keep] != tokens[:keep]:
        return reject("token ids differ")
    try:
        layers = json.loads(meta["layers"])
        cache = make_prompt_cache(model)
        if len(layers) != len(cache):
            raise _Refuse(f"{len(layers)} layers in the file, model makes {len(cache)}")
        used: set = set()
        for i, (c, spec) in enumerate(zip(cache, layers)):
            _decode(c, spec, f"L{i}", arrays, used)
        if used != set(arrays):
            raise _Refuse("file holds arrays the layout does not use")
        members = [x for c in cache for x in _members(c)]
        for c in members:
            if isinstance(c, KVCache) and int(c.offset) != n:
                raise _Refuse(f"attention offset {c.offset} != {n}")
    except (_Refuse, KeyError, TypeError, ValueError) as e:
        return reject(str(e))
    label = _label(best.get("kind", KIND_FRONT))
    if best.get("fp") != fp:
        old_fp = best.get("fp")
        sizes = best["replay"]["sizes"]
        print(
            f"[prefix-disk] engine changed since this {label} was saved; replaying its last "
            f"{len(sizes)} chunk(s) ({sum(sizes)} tok) to check it",
            flush=True,
        )
        t1 = time.perf_counter()
        try:
            same = _replay_check(slot, best, cache, stored, replay)
        except Exception as e:  # noqa: BLE001 - unproven is a miss, never a hit
            STATS["replay_failed"] += 1
            return reject(f"replay check failed: {e!r}")
        if not same:
            STATS["replay_failed"] += 1
            drop = [x for x in data["entries"] if x.get("fp") == old_fp]
            for x in drop:
                _remove(slot, x)
            data["entries"] = [x for x in data["entries"] if x.get("fp") != old_fp]
            _write_index(slot, data)
            STATS["rejected"] += len(drop)
            print(
                f"[prefix-disk] the new engine writes different KV; dropped {len(drop)} file(s) "
                "from the old engine; prefilling",
                flush=True,
            )
            return None
        best["fp"] = fp
        best["parts"] = dict(parts)
        best["settings"] = dict(_numerics_env())
        STATS["verified"] += 1
        print(
            f"[prefix-disk] same KV under the new engine ({time.perf_counter() - t1:.2f}s); "
            f"keeping the {label}",
            flush=True,
        )
    if keep < n and best_ck is None:
        try:
            _rewind(cache, keep, {})
        except _Refuse as e:
            # The file is fine for a whole match, only this cut does not fit.
            _write_index(slot, data)
            STATS["misses"] += 1
            print(f"[prefix-disk] partial match not served: {e}; prefilling", flush=True)
            return None
    if best_ck is not None:
        try:
            at, states = _load_ckpt(slot, best_ck, best.get("fp_file", fp), tokens)
            if at != keep:
                raise _Refuse(f"checkpoint at {at}, wanted {keep}")
            _rewind(cache, at, states)
            if any(isinstance(c, KVCache) and int(c.offset) != keep for c in members):
                raise _Refuse(f"rewind to {keep} left another offset")
        except Exception as e:  # noqa: BLE001 - only this checkpoint is bad, the file is fine
            STATS["rejected"] += 1
            print(
                f"[prefix-disk] checkpoint @ {best_ck.get('at')} of {best['file']} rejected: {e}; prefilling",
                flush=True,
            )
            bad = best_ck.get("file")
            for e in data["entries"]:
                if isinstance(e.get("ckpts"), list):
                    e["ckpts"] = [c for c in e["ckpts"] if not (isinstance(c, dict) and c.get("file") == bad)]
                rp = _replay_ck(e)
                if rp is not None and rp["file"] == bad:
                    e.pop("replay", None)
            _write_index(slot, data)
            return None
    realize(cache)
    best["used"] = time.time()
    _write_index(slot, data)
    global last_kind
    last_kind = best.get("kind", KIND_FRONT)
    STATS["hits"] += 1
    if keep < n:
        STATS["partial"] += 1
    span = f"{keep} of {n} tok" if keep < n else f"{n} tok"
    if best_ck is not None:
        span += " (recurrent checkpoint)"
    print(
        f"[prefix-disk] loaded {label} {span} from disk "
        f"({int(best['bytes']) / 1e9:.2f} GB, {time.perf_counter() - t0:.2f}s)",
        flush=True,
    )
    return tokens[:keep], cache
