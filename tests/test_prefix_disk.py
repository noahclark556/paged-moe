# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Front matter KV on disk (prefix_disk.py) on a tiny qwen3_next.

The claim is that a pin written to disk and read back is the cache a prefill
of those tokens leaves, bit for bit, side caches included, and that anything
that could make it stale is a miss: other tokens, changed weights or config,
a changed numerics setting, a corrupted file, a different cache layout.
On a plain attention model (tiny qwen3_moe) a file the prompt shares only a
head with loads trimmed to that head, again bit for bit.

Run:
    python tests/test_prefix_disk.py
"""

import contextlib
import hashlib
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import KVCache, QuantizedKVCache, make_prompt_cache  # noqa: E402

from tests.test_lookup_hybrid import ARGS, _Side, build  # noqa: E402

from pagedmoe import config as es_config  # noqa: E402
from pagedmoe import prefix_cache, prefix_disk  # noqa: E402

BUDGET = 1 << 40


class _AttnSide(KVCache):
    def __init__(self):
        super().__init__()
        self.indexer = _Side()


def _with_side(model):
    plain = getattr(model, "make_cache", None) or (lambda: [KVCache() for _ in model.layers])

    def make_cache():
        return [_AttnSide() if isinstance(c, KVCache) else c for c in plain()]

    model.make_cache = make_cache
    return model


def _fill_side(cache, n):
    for c in cache:
        if isinstance(c, _AttnSide):
            c.indexer.keys = mx.arange(n * 4, dtype=mx.float32).reshape(1, n, 4)


def _model_dir(tmp: Path, name: str = "m") -> tuple:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text('{"model_type": "qwen3_next"}')
    (d / "model-00001.safetensors").write_bytes(b"\0" * 64)
    return (str(d), None, None)


def _prefill(model, cache, toks):
    mx.eval(model(mx.array([toks]), cache=cache))


class _Env:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="prefix-disk-test-"))
        self.old_dir = es_config.PREFIX_DISK_DIR
        es_config.PREFIX_DISK_DIR = str(self.tmp / "cache")
        prefix_disk.reset()

    def close(self):
        es_config.PREFIX_DISK_DIR = self.old_dir
        prefix_disk.reset()

    def files(self):
        return sorted((self.tmp / "cache").rglob(f"*{prefix_disk._SUFFIX}"))


def _tokens(seed, n):
    return np.random.default_rng(seed).integers(0, ARGS["vocab_size"], size=n).tolist()


def check_round_trip(side: bool):
    env = _Env()
    try:
        model = build()
        if side:
            _with_side(model)
        key = _model_dir(env.tmp)
        a = _tokens(3, 120)
        p = 80
        live = make_prompt_cache(model)
        _prefill(model, live, a[:p])
        if side:
            _fill_side(live, p)
        assert prefix_disk.save(key, a[:p], live, model, BUDGET), "save refused"
        assert len(env.files()) == 1
        assert not prefix_disk.save(key, a[:p], live, model, BUDGET), "second save wrote again"
        assert len(env.files()) == 1

        prefix_disk.reset()
        got = prefix_disk.load(key, a, model, 0, BUDGET)
        assert got is not None, "load missed its own file"
        toks, cache = got
        assert toks == a[:p]
        if side:
            for c in cache:
                if isinstance(c, _AttnSide):
                    want = mx.arange(p * 4, dtype=mx.float32).reshape(1, p, 4)
                    assert bool(mx.all(c.indexer.keys == want)), "side cache content moved"

        ref = make_prompt_cache(model)
        _prefill(model, ref, a[:p])
        la = model(mx.array([a[p:]]), cache=cache)
        lb = model(mx.array([a[p:]]), cache=ref)
        d = float(mx.abs(la - lb).max())
        assert d == 0.0, f"loaded cache differs from a fresh prefill by {d}"
        # And it keeps decoding the same way.
        la = model(mx.array([[7]]), cache=cache)
        lb = model(mx.array([[7]]), cache=ref)
        assert float(mx.abs(la - lb).max()) == 0.0, "decode after load diverged"
        tag = " with a side cache" if side else ""
        print(f"OK: disk round trip is bit-identical to a fresh prefill{tag}")
    finally:
        env.close()


def check_misses():
    env = _Env()
    try:
        model = build()
        key = _model_dir(env.tmp)
        a = _tokens(4, 100)
        p = 70
        live = make_prompt_cache(model)
        _prefill(model, live, a[:p])
        assert prefix_disk.save(key, a[:p], live, model, BUDGET)

        other = list(a)
        other[10] = (other[10] + 1) % ARGS["vocab_size"]
        got, said = _quiet(prefix_disk.load, key, other, model, 0, BUDGET)
        assert got is None, "served other tokens"
        assert "parts from it at token 10" in said, f"a cold miss did not say where it parted: {said!r}"
        assert prefix_disk.load(key, a[:p], model, 0, BUDGET) is None, "left no token for decode"
        assert prefix_disk.load(key, a, model, p, BUDGET) is None, "did not beat what RAM had"
        assert prefix_disk.load(key, a, model, 0, 100) is None, "ignored the prompt budget"
        assert prefix_disk.load(("elsewhere", None, None), a, model, 0, BUDGET) is None
        assert prefix_disk.load((key[0], None, "draft"), a, model, 0, BUDGET) is None

        # Token mismatches leave the file alone.
        assert len(env.files()) == 1, "a plain miss deleted the file"

        # Weights or config change: miss, and the stale file is gone.
        cfg = Path(key[0]) / "config.json"
        cfg.write_text('{"model_type": "qwen3_next", "x": 1}')
        prefix_disk.reset()
        assert prefix_disk.load(key, a, model, 0, BUDGET) is None, "hit after config.json changed"
        assert not env.files(), "stale file kept after config.json changed"
        assert prefix_disk.save(key, a[:p], live, model, BUDGET)
        assert prefix_disk.load(key, a, model, 0, BUDGET) is not None, "fresh save after a change missed"

        w = Path(key[0]) / "model-00001.safetensors"
        os.utime(w, ns=(w.stat().st_atime_ns, w.stat().st_mtime_ns + 10**9))
        prefix_disk.reset()
        assert prefix_disk.load(key, a, model, 0, BUDGET) is None, "hit after weights changed"
        assert not env.files(), "stale file kept after weights changed"
        print("OK: other tokens, no decode token, RAM ahead, budget miss; weights and config drop the file")
    finally:
        env.close()


def check_settings():
    env = _Env()
    knobs = (
        "PAGED_MOE_PRUNE",
        "PAGED_MOE_FLOW",
        "PAGED_MOE_SIDECAR_HEAD_PRUNE",
        "PAGED_MOE_MAX_CACHE_GB",
        "PAGED_MOE_ATTN_SUB_CHUNK",
    )
    old = {k: os.environ.get(k) for k in knobs}
    try:
        model = build()
        key = _model_dir(env.tmp)
        a = _tokens(5, 100)
        live = make_prompt_cache(model)
        _prefill(model, live, a[:70])
        for k in knobs:
            os.environ.pop(k, None)
        prefix_disk.reset()
        assert prefix_disk.save(key, a[:70], live, model, BUDGET)

        os.environ["PAGED_MOE_MAX_CACHE_GB"] = "11"
        os.environ["PAGED_MOE_PRUNE"] = "0.5"
        os.environ["PAGED_MOE_FLOW"] = "1"
        os.environ["PAGED_MOE_SIDECAR_HEAD_PRUNE"] = "1"
        prefix_disk.reset()
        assert prefix_disk.load(key, a, model, 0, BUDGET) is not None, "a decode-only knob caused a miss"
        os.environ["PAGED_MOE_ATTN_SUB_CHUNK"] = "512"
        prefix_disk.reset()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert prefix_disk.load(key, a, model, 0, BUDGET) is None, "hit after a prefill knob changed"
        said = buf.getvalue()
        assert "settings changed: PAGED_MOE_ATTN_SUB_CHUNK" in said, f"miss did not say why: {said!r}"
        assert not env.files(), "stale file kept after a prefill knob changed"
        print("OK: a prefill setting misses and is named, decode-only and memory knobs do not")
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        env.close()


def check_corruption_and_layout():
    env = _Env()
    try:
        model = build()
        key = _model_dir(env.tmp)
        a = _tokens(6, 100)
        live = make_prompt_cache(model)
        _prefill(model, live, a[:70])
        assert prefix_disk.save(key, a[:70], live, model, BUDGET)
        f = env.files()[0]
        raw = bytearray(f.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        f.write_bytes(bytes(raw))
        assert prefix_disk.load(key, a, model, 0, BUDGET) is None, "loaded a corrupted file"
        assert not f.exists(), "corrupted file was left behind"

        assert prefix_disk.save(key, a[:70], live, model, BUDGET)
        side_model = _with_side(build())
        assert prefix_disk.load(key, a, side_model, 0, BUDGET) is None, "loaded into another layout"

        q = list(live)
        for i, c in enumerate(q):
            if isinstance(c, KVCache):
                q[i] = QuantizedKVCache()
                break
        skipped = prefix_disk.STATS["skipped"]
        assert not prefix_disk.save(key, a[:70], q, model, BUDGET), "saved a quantized cache"
        assert prefix_disk.STATS["skipped"] == skipped + 1, "quantized cache was not refused"
        assert not env.files()
        print("OK: corrupted files are dropped, layouts must match, quantized KV is not saved")
    finally:
        env.close()


def check_eviction():
    env = _Env()
    old = (es_config.PREFIX_DISK_MODELS, es_config.PREFIX_DISK_PER_MODEL)
    es_config.PREFIX_DISK_MODELS, es_config.PREFIX_DISK_PER_MODEL = 3, 2
    try:
        model = build()
        a = _tokens(7, 200)
        live = make_prompt_cache(model)
        _prefill(model, live, a[:70])
        keys = [_model_dir(env.tmp, f"m{i}") for i in range(4)]
        for k in keys:
            assert prefix_disk.save(k, a[:70], live, model, BUDGET)
        slots = [p for p in (env.tmp / "cache").iterdir() if p.is_dir()]
        assert len(slots) == 3, f"kept {len(slots)} models"
        assert prefix_disk.load(keys[0], a, model, 0, BUDGET) is None, "oldest model was kept"
        assert prefix_disk.load(keys[3], a, model, 0, BUDGET) is not None

        k = keys[3]
        for n in (80, 90, 100):
            c = make_prompt_cache(model)
            _prefill(model, c, a[:n])
            assert prefix_disk.save(k, a[:n], c, model, BUDGET)
        slot = prefix_disk._model_slot(Path(k[0]).resolve())
        assert len(list(slot.glob(f"*{prefix_disk._SUFFIX}"))) == 2, "per-model cap not applied"
        got = prefix_disk.load(k, a, model, 0, BUDGET)
        assert got is not None and len(got[0]) == 100, "did not pick the longest match"
        print("OK: three models, two files each, longest match wins")
    finally:
        es_config.PREFIX_DISK_MODELS, es_config.PREFIX_DISK_PER_MODEL = old
        env.close()


def check_missing_model_dropped():
    env = _Env()
    try:
        model = build()
        a = _tokens(7, 200)
        live = make_prompt_cache(model)
        _prefill(model, live, a[:70])
        gone, kept = _model_dir(env.tmp, "gone"), _model_dir(env.tmp, "kept")
        assert prefix_disk.save(gone, a[:70], live, model, BUDGET)
        gone_slot = prefix_disk._model_slot(Path(gone[0]).resolve())
        shutil.rmtree(gone[0])
        assert prefix_disk.save(kept, a[:70], live, model, BUDGET)
        assert not gone_slot.exists(), "slot for a deleted model folder was kept"
        assert prefix_disk.load(kept, a, model, 0, BUDGET) is not None
        assert prefix_disk._model_gone(Path("/Volumes/ga-not-mounted/models/x")) is False
        print("OK: a deleted model's slot goes, an unmounted drive's stays")
    finally:
        env.close()


def check_eviction_keeps_other_workspaces():
    env = _Env()
    old = es_config.PREFIX_DISK_PER_MODEL
    es_config.PREFIX_DISK_PER_MODEL = 2
    try:
        model = build()
        key = _model_dir(env.tmp)
        shared = _tokens(30, 40)
        w1 = shared + _tokens(31, 40)
        w2 = shared + _tokens(32, 30)
        saves = [w1, w2 + _tokens(33, 10), w2 + _tokens(34, 10), w2 + _tokens(35, 10)]
        for toks in saves:
            c = make_prompt_cache(model)
            _prefill(model, c, toks)
            assert prefix_disk.save(key, toks, c, model, BUDGET)
        prefix_disk.reset()
        assert prefix_disk.has(key, w1), "a growing workspace pushed out another workspace's file"
        assert prefix_disk.has(key, saves[-1]) and not prefix_disk.has(key, saves[1])
        got = prefix_disk.load(key, w1 + [1, 2, 3], model, 0, BUDGET)
        assert got is not None and got[0] == w1
        print("OK: a new save replaces its own workspace's older file, not another workspace's")
    finally:
        es_config.PREFIX_DISK_PER_MODEL = old
        env.close()


def check_cap_keeps_newest_per_model():
    env = _Env()
    old = (es_config.PREFIX_DISK_MODELS, es_config.PREFIX_DISK_PER_MODEL, es_config.PREFIX_DISK_MAX_GB)
    es_config.PREFIX_DISK_MODELS, es_config.PREFIX_DISK_PER_MODEL = 3, 2
    try:
        model = build()
        a = _tokens(41, 200)
        small, big = _model_dir(env.tmp, "small"), _model_dir(env.tmp, "big")
        c = make_prompt_cache(model)
        _prefill(model, c, a[:70])
        assert prefix_disk.save(small, a[:70], c, model, BUDGET, kind=prefix_disk.KIND_SHARED)
        size = env.files()[0].stat().st_size
        # Room for three files: big's two saves plus small's one.
        es_config.PREFIX_DISK_MAX_GB = (size * 3.5) / (1 << 30)
        for n in (90, 100):
            c = make_prompt_cache(model)
            _prefill(model, c, a[:n])
            assert prefix_disk.save(big, a[:n], c, model, BUDGET)
        c = make_prompt_cache(model)
        _prefill(model, c, a[:110])
        assert prefix_disk.save(big, a[:110], c, model, BUDGET)
        prefix_disk.reset()
        assert prefix_disk.has(small, a[:70]), "the cap took the small model's only file"
        assert prefix_disk.has(big, a[:110]), "the cap took the newest file"
        print("OK: the size cap takes older files before any model's newest one")
    finally:
        es_config.PREFIX_DISK_MODELS, es_config.PREFIX_DISK_PER_MODEL, es_config.PREFIX_DISK_MAX_GB = old
        env.close()


def check_client_files_keep_host_files():
    env = _Env()
    old = es_config.PREFIX_DISK_PER_MODEL
    es_config.PREFIX_DISK_PER_MODEL = 2
    try:
        model = build()
        key = _model_dir(env.tmp)
        host = _tokens(51, 80)
        c = make_prompt_cache(model)
        _prefill(model, c, host)
        assert prefix_disk.save(key, host, c, model, BUDGET, kind=prefix_disk.KIND_SHARED)
        clients = [_tokens(60 + i, 70 + i) for i in range(3)]
        for toks in clients:
            c = make_prompt_cache(model)
            _prefill(model, c, toks)
            assert prefix_disk.save(key, toks, c, model, BUDGET, kind=prefix_disk.KIND_CLIENT)
        prefix_disk.reset()
        assert prefix_disk.has(key, host), "a client prompt evicted the host's shared file"
        assert prefix_disk.has(key, clients[-1]) and not prefix_disk.has(key, clients[0])
        got = prefix_disk.load(key, clients[-1] + [1, 2], model, 0, BUDGET)
        assert got is not None and got[0] == clients[-1]
        print("OK: client system prompts have their own cap and never evict host files")
    finally:
        es_config.PREFIX_DISK_PER_MODEL = old
        env.close()


def check_chat_files():
    env = _Env()
    old_chats, old_per = es_config.PREFIX_DISK_CHATS, es_config.PREFIX_DISK_PER_MODEL
    es_config.PREFIX_DISK_CHATS, es_config.PREFIX_DISK_PER_MODEL = 2, 1
    try:
        model = build()
        key = _model_dir(env.tmp)
        front = _tokens(70, 80)
        c = make_prompt_cache(model)
        _prefill(model, c, front)
        assert prefix_disk.save(key, front, c, model, BUDGET)

        def chat(toks, tag):
            c = make_prompt_cache(model)
            _prefill(model, c, toks)
            return prefix_disk.save(key, toks, c, model, BUDGET, kind=prefix_disk.KIND_CHAT, tag=tag)

        a1 = front + _tokens(71, 40)
        a2 = a1 + _tokens(72, 30)
        assert chat(a1, "a") and chat(a2, "a")
        assert not prefix_disk.has(key, a1), "a chat's older file stayed next to the new one"
        got = prefix_disk.load(key, a2 + [5], model, 0, BUDGET)
        assert got is not None and got[0] == a2 and prefix_disk.last_kind == prefix_disk.KIND_CHAT

        b = front + _tokens(73, 50)
        d = front + _tokens(74, 50)
        assert chat(b, "b") and chat(d, "d")
        prefix_disk.reset()
        assert prefix_disk.has(key, front), "chats evicted the front matter file"
        assert prefix_disk.has(key, d) and prefix_disk.has(key, b) and not prefix_disk.has(key, a2)

        (env.tmp / "cache" / prefix_disk._FORGET).write_text('["b"]')
        assert prefix_disk.load(key, b + [5], model, 0, BUDGET)[0] == front, "a forgotten chat still loaded"
        assert not prefix_disk.has(key, b)
        assert not chat(b, "b"), "a forgotten chat was saved again"
        assert prefix_disk.has(key, d)
        prefix_disk.reset()
        got = prefix_disk.load(key, front + [5, 6], model, 0, BUDGET)
        assert got is not None and prefix_disk.last_kind == prefix_disk.KIND_FRONT
        print("OK: chat files have their own cap, replace their older file, and go when forgotten")
    finally:
        es_config.PREFIX_DISK_CHATS, es_config.PREFIX_DISK_PER_MODEL = old_chats, old_per
        env.close()


@contextlib.contextmanager
def _keyed(hexkey):
    """Run with the host's key on a pipe, as the app hands it over."""
    from pagedmoe import seal

    old = es_config.PREFIX_DISK_KEY_FD
    r, w = os.pipe()
    os.write(w, hexkey.encode())
    os.close(w)
    es_config.PREFIX_DISK_KEY_FD = r
    seal._state = None
    prefix_disk.reset()
    try:
        yield
    finally:
        es_config.PREFIX_DISK_KEY_FD = old
        seal._state = None
        prefix_disk.reset()


def check_sealed():
    from array import array

    from pagedmoe import seal

    env = _Env()
    old_gain = prefix_disk._MIN_PARTIAL_GAIN
    prefix_disk._MIN_PARTIAL_GAIN = 32
    try:
        model = build()
        key = _model_dir(env.tmp)
        a = _tokens(80, 300)
        p, at = 240, [64, 128, 192]
        live, ck = _with_ckpts(model, a[:p], at)
        assert prefix_disk.save(key, a[:p], live, model, BUDGET, ckpts=ck), "plain save refused"
        cache_dir = env.tmp / "cache"

        with _keyed("11" * 32):
            assert seal.active()
            _quiet(prefix_disk.housekeep)
            assert not env.files(), "plain files stayed once sealing was on"
            assert prefix_disk.save(key, a[:p], live, model, BUDGET, ckpts=ck), "sealed save refused"
            ids = array("i", a[:p]).tobytes()
            sha = prefix_disk._tok_sha(a[:p])
            for f in cache_dir.rglob("*"):
                if not f.is_file() or f.name in ("engine.json", prefix_disk._OPEN):
                    continue
                raw = f.read_bytes()
                assert raw.startswith(seal.MAGIC), f"{f.name} is not sealed"
                assert ids[:64] not in raw, f"{f.name} holds the token ids in the clear"
                assert sha[:24] not in f.name, f"{f.name} is named by the ids"
            assert len(list(cache_dir.rglob(f"{prefix_disk._CK_PREFIX}*"))) == 3

            prefix_disk.reset()
            got = prefix_disk.load(key, a, model, 0, BUDGET)
            assert got is not None and got[0] == a[:p], "sealed file missed"
            ref = _filled(model, a[:p])
            la = model(mx.array([a[p:]]), cache=got[1])
            lb = model(mx.array([a[p:]]), cache=ref)
            assert float(mx.abs(la - lb).max()) == 0.0, "sealed round trip is not bit-identical"

            b = list(a)
            b[150] = (b[150] + 1) % ARGS["vocab_size"]
            prefix_disk.reset()
            got = prefix_disk.load(key, b, model, 0, BUDGET)
            assert got is not None and got[0] == b[:128], "sealed checkpoint did not resume at 128"
            ref = make_prompt_cache(model)
            _prefill(model, ref, b[:64])
            _prefill(model, ref, b[64:128])
            la = model(mx.array([b[128:]]), cache=got[1])
            lb = model(mx.array([b[128:]]), cache=ref)
            assert float(mx.abs(la - lb).max()) == 0.0, "sealed checkpoint resume differs"

            main = [f for f in env.files() if not f.name.startswith(prefix_disk._CK_PREFIX)][0]
            raw = bytearray(main.read_bytes())
            raw[len(raw) // 2] ^= 1
            main.write_bytes(bytes(raw))
            prefix_disk.reset()
            got, said = _quiet(prefix_disk.load, key, a, model, 0, BUDGET)
            assert got is None and "checksum mismatch" in said, f"a tampered file was used: {said!r}"

            assert prefix_disk.save(key, a[:p], live, model, BUDGET), "sealed save after a reject refused"

        with _keyed("22" * 32):
            _quiet(prefix_disk.housekeep)
            assert not env.files(), "files under another key stayed"
            assert prefix_disk.load(key, a, model, 0, BUDGET) is None

        with _keyed("not a key"):
            assert not prefix_disk.enabled(), "a bad key fell back to plain files"
            assert not prefix_disk.save(key, a[:p], live, model, BUDGET)
        print("OK: sealed files round trip bit for bit, hold nothing in the clear, and go on tamper or a new key")
    finally:
        prefix_disk._MIN_PARTIAL_GAIN = old_gain
        env.close()


def check_numerics_fingerprint():
    from pagedmoe import numerics_fp

    env = _Env()
    try:
        pkg = env.tmp / "pkg"
        (pkg / "sidecar").mkdir(parents=True)
        (pkg / "streaming.py").write_text("x = 1\n")
        (pkg / "verify_decode.py").write_text("y = 1\n")
        (pkg / "sidecar" / "service.py").write_text("z = 1\n")
        base = numerics_fp.source_fingerprint(pkg)
        (pkg / "verify_decode.py").write_text("y = 2\n")
        (pkg / "sidecar" / "service.py").write_text("z = 2\n")
        assert numerics_fp.source_fingerprint(pkg) == base, "decode-only code moved the fingerprint"
        (pkg / "streaming.py").write_text("x = 2\n")
        assert numerics_fp.source_fingerprint(pkg) != base, "prefill code did not move it"
        src = (
            '"""Doc."""\n'
            "def f(x, n):\n"
            '    """Doc."""\n'
            "    # a comment\n"
            '    print(f"[prefill] chunk {len(x)} at {n / 1e9:.2f}", flush=True)\n'
            "    return x * 2\n"
        )
        (pkg / "streaming.py").write_text(src)
        base = numerics_fp.source_fingerprint(pkg)
        edits = (
            src.replace('"""Doc."""', '"""Other words."""'),
            src.replace("# a comment", "# another comment, longer"),
            src.replace("[prefill] chunk", "[prefill] one chunk of"),
            src.replace("    return x * 2", "\n    return  x*2"),
        )
        for e in edits:
            (pkg / "streaming.py").write_text(e)
            assert numerics_fp.source_fingerprint(pkg) == base, f"a non-code edit moved the fingerprint:\n{e}"
        for e in (
            src.replace("x * 2", "x * 3"),
            src.replace("len(x)", "x.pop()"),
            src.replace("{len(x)}", "{(y := len(x))}"),
        ):
            (pkg / "streaming.py").write_text(e)
            assert numerics_fp.source_fingerprint(pkg) != base, f"a code edit kept the fingerprint:\n{e}"
        (pkg / "new_module.py").write_text("w = 1\n")
        assert numerics_fp.source_fingerprint(pkg) != numerics_fp.source_fingerprint(pkg / "sidecar")
        (pkg / numerics_fp.FILE).write_text("sealed-build-hash\n")
        assert numerics_fp.current(pkg) == "sealed-build-hash", "sealed hash not read back"

        prefix_disk.write_stamp()
        import json

        stamp = json.loads((env.tmp / "cache" / prefix_disk.STAMP).read_text())
        assert stamp["engine"] == prefix_disk.engine_part()
        print("OK: only prefill code moves the engine fingerprint, not comments, docs or log text; the stamp records it")
    finally:
        env.close()


def check_partial():
    from tests.test_verify_decode import build_trim

    env = _Env()
    try:
        model, vocab = build_trim()
        key = _model_dir(env.tmp)
        rng = np.random.default_rng(11)
        a = rng.integers(0, vocab, size=560).tolist()
        k = 400
        b = a[:k] + [(t + 1) % vocab for t in a[k:]]
        p = 500
        live = make_prompt_cache(model)
        _prefill(model, live, a[:p])
        assert prefix_disk.save(key, a[:p], live, model, BUDGET)
        tok = list((env.tmp / "cache").rglob(f"*{prefix_disk._TOK_SUFFIX}"))
        assert len(tok) == 1, "no token file next to the save"

        pin = prefix_cache.StablePrefixPin()
        assert pin.put("m", a[:p], live)
        ram = pin.lookup("m", b)
        assert ram is not None and ram[2] == k

        prefix_disk.reset()
        partial = prefix_disk.STATS["partial"]
        got = prefix_disk.load(key, b, model, 0, BUDGET)
        assert got is not None, "partial match on a plain KVCache layout missed"
        toks, cache = got
        assert toks == b[:k] and prefix_disk.STATS["partial"] == partial + 1
        assert all(int(c.offset) == k for c in cache)

        la = model(mx.array([b[k:]]), cache=cache)
        lb = model(mx.array([b[k:]]), cache=ram[0])
        d = float(mx.abs(la - lb).max())
        assert d == 0.0, f"trimmed disk hit differs from the in-memory partial hit by {d}"
        ref = make_prompt_cache(model)
        _prefill(model, ref, b[:k])
        lc = model(mx.array([b[k:]]), cache=ref)
        fresh = float(mx.abs(la - lc).max())
        assert fresh == 0.0, f"trimmed disk hit differs from a fresh prefill by {fresh}"
        la = model(mx.array([[7]]), cache=cache)
        lc = model(mx.array([[7]]), cache=ref)
        assert float(mx.abs(la - lc).max()) == 0.0, "decode after a partial hit diverged"

        assert prefix_disk.load(key, b, model, k - 100, BUDGET) is None, "took a partial hit worth 100 tok"
        assert prefix_disk.load(key, a, model, 0, BUDGET)[0] == a[:p], "exact match no longer whole"

        tok[0].unlink()
        assert prefix_disk.load(key, b, model, 0, BUDGET) is None, "partial hit without a token file"
        assert prefix_disk.load(key, a, model, 0, BUDGET) is not None, "exact hit needed the token file"

        other = make_prompt_cache(model)
        _prefill(model, other, b[:p])
        assert prefix_disk.save(key, b[:p], other, model, BUDGET)
        cfg = Path(key[0]) / "config.json"
        cfg.write_text('{"model_type": "qwen3_moe", "x": 2}')
        prefix_disk.reset()
        assert prefix_disk.load(key, b, model, 0, BUDGET) is None
        left = list((env.tmp / "cache").rglob("*.tok"))
        assert not env.files() and not left, f"stale drop left {left}"
        print("OK: a partial match on plain KVCache loads trimmed, bit-identical to a fresh prefill")
    finally:
        env.close()


def check_head_file():
    from tests.test_verify_decode import build_trim

    env = _Env()
    old = es_config.PREFIX_DISK_PER_MODEL
    es_config.PREFIX_DISK_PER_MODEL = 2
    try:
        model = build()
        key = _model_dir(env.tmp)
        assert not prefix_disk.trims_exactly(make_prompt_cache(model)), "hybrid counted as trimmable"
        assert prefix_disk.trims_exactly(make_prompt_cache(build_trim()[0]))
        assert prefix_disk.supports(key) and not prefix_disk.supports((key[0], None, "draft"))

        a = _tokens(12, 200)
        h, p = 80, 150
        live = make_prompt_cache(model)
        _prefill(model, live, a[:h])
        assert prefix_disk.save(key, a[:h], live, model, BUDGET, kind=prefix_disk.KIND_HEAD)
        assert prefix_disk.has(key, a[:h]) and not prefix_disk.has(key, a[: h - 1])
        _prefill(model, live, a[h:p])
        assert prefix_disk.save(key, a[:p], live, model, BUDGET)

        ref = make_prompt_cache(model)
        _prefill(model, ref, a[:h])
        _prefill(model, ref, a[h:p])
        la = model(mx.array([a[p:]]), cache=live)
        lb = model(mx.array([a[p:]]), cache=ref)
        assert float(mx.abs(la - lb).max()) == 0.0, "saving the head changed the live prefill"

        b = a[:h] + [(t + 1) % ARGS["vocab_size"] for t in a[h:]]
        for seed in range(3):
            c = make_prompt_cache(model)
            other = a[:h] + _tokens(20 + seed, p - h)
            _prefill(model, c, other)
            assert prefix_disk.save(key, other, c, model, BUDGET)
        prefix_disk.reset()
        got = prefix_disk.load(key, b, model, 0, BUDGET)
        assert got is not None, "memory edits evicted the head file"
        toks, cache = got
        assert toks == b[:h], f"resumed at {len(toks)}, not the head"
        ref = make_prompt_cache(model)
        _prefill(model, ref, b[:h])
        la = model(mx.array([b[h:]]), cache=cache)
        lb = model(mx.array([b[h:]]), cache=ref)
        d = float(mx.abs(la - lb).max())
        assert d == 0.0, f"head file differs from a fresh prefill by {d}"

        q = [QuantizedKVCache() if isinstance(c, KVCache) else c for c in cache]
        assert not prefix_disk.save(key, _tokens(99, h), q, model, BUDGET, kind=prefix_disk.KIND_HEAD)
        assert not prefix_disk.supports(key), "a refused layout would keep paying the head stop"
        print("OK: hybrid head file resumes a memory edit bit-identically and outlives front files")
    finally:
        es_config.PREFIX_DISK_PER_MODEL = old
        env.close()


def check_shared_file():
    env = _Env()
    old = es_config.PREFIX_DISK_PER_MODEL
    es_config.PREFIX_DISK_PER_MODEL = 2
    try:
        model = build()
        key = _model_dir(env.tmp)
        s, h = 70, 100
        system = _tokens(40, s)
        live = make_prompt_cache(model)
        _prefill(model, live, system)
        assert prefix_disk.save(key, system, live, model, BUDGET, kind=prefix_disk.KIND_SHARED)
        for seed in range(3):
            ws = system + _tokens(41 + seed, h - s)
            c = make_prompt_cache(model)
            _prefill(model, c, ws)
            assert prefix_disk.save(key, ws, c, model, BUDGET, kind=prefix_disk.KIND_HEAD)
            front = ws + _tokens(50 + seed, 20)
            assert prefix_disk.save(key, front, _filled(model, front), model, BUDGET)
        prefix_disk.reset()
        assert prefix_disk.has(key, system), "head and front saves evicted the shared file"

        new_ws = system + _tokens(60, 50)
        got = prefix_disk.load(key, new_ws, model, 0, BUDGET)
        assert got is not None, "a new workspace missed the shared file"
        toks, cache = got
        assert toks == system, f"resumed at {len(toks)}, not the shared span"
        ref = make_prompt_cache(model)
        _prefill(model, ref, system)
        la = model(mx.array([new_ws[s:]]), cache=cache)
        lb = model(mx.array([new_ws[s:]]), cache=ref)
        d = float(mx.abs(la - lb).max())
        assert d == 0.0, f"shared file differs from a fresh prefill by {d}"
        print("OK: a new workspace resumes from the shared file bit-identically, and it outlives workspace files")
    finally:
        es_config.PREFIX_DISK_PER_MODEL = old
        env.close()


def _filled(model, toks):
    c = make_prompt_cache(model)
    _prefill(model, c, toks)
    return c


def _slots(cache):
    from mlx_lm.models.cache import ArraysCache

    return {i: list(c.cache) for i, c in enumerate(cache) if isinstance(c, ArraysCache)}


def _with_ckpts(model, toks, at, side=False):
    """Prefill `toks` in chunks cut at `at`, keeping the recurrent state at each."""
    c = make_prompt_cache(model)
    ck, lo = {}, 0
    for p in [*at, len(toks)]:
        _prefill(model, c, toks[lo:p])
        if p < len(toks):
            ck[p] = _slots(c)
        lo = p
    if side:
        _fill_side(c, len(toks))
    return c, ck


def check_hybrid_ckpt(side: bool):
    env = _Env()
    old_gain = prefix_disk._MIN_PARTIAL_GAIN
    prefix_disk._MIN_PARTIAL_GAIN = 32
    try:
        model = build()
        if side:
            _with_side(model)
        key = _model_dir(env.tmp)
        a = _tokens(70, 300)
        p, at = 240, [64, 128, 192]
        live, ck = _with_ckpts(model, a[:p], at, side)
        assert prefix_disk.save(key, a[:p], live, model, BUDGET, ckpts=ck)
        cks = sorted((env.tmp / "cache").rglob(f"{prefix_disk._CK_PREFIX}*"))
        assert len(cks) == 3, f"wrote {len(cks)} checkpoint files"

        # A change at 150 resumes at 128, the last checkpoint before it.
        b = list(a)
        b[150] = (b[150] + 1) % ARGS["vocab_size"]
        prefix_disk.reset()
        got = prefix_disk.load(key, b, model, 0, BUDGET)
        assert got is not None, "hybrid file with checkpoints missed a partial match"
        toks, cache = got
        assert toks == b[:128], f"resumed at {len(toks)}, not 128"
        ref = make_prompt_cache(model)
        _prefill(model, ref, b[:64])
        _prefill(model, ref, b[64:128])
        if side:
            for c in ref:
                if isinstance(c, _AttnSide):
                    c.indexer.keys = mx.arange(128 * 4, dtype=mx.float32).reshape(1, 128, 4)
            for c in cache:
                if isinstance(c, _AttnSide):
                    assert c.indexer.keys.shape[1] == 128, "side cache not cut back"
        la = model(mx.array([b[128:]]), cache=cache)
        lb = model(mx.array([b[128:]]), cache=ref)
        d = float(mx.abs(la - lb).max())
        assert d == 0.0, f"checkpoint resume differs from a fresh prefill by {d}"
        la = model(mx.array([[7]]), cache=cache)
        lb = model(mx.array([[7]]), cache=ref)
        assert float(mx.abs(la - lb).max()) == 0.0, "decode after a checkpoint resume diverged"

        # Not worth it over what RAM has, or before the first checkpoint.
        assert prefix_disk.load(key, b, model, 110, BUDGET) is None, "took a checkpoint worth 18 tok"
        early = list(a)
        early[40] = (early[40] + 1) % ARGS["vocab_size"]
        assert prefix_disk.load(key, early, model, 0, BUDGET) is None, "served a change before any checkpoint"

        # A newer file over the same ids reuses the checkpoints already there.
        c2 = list(a[:p]) + _tokens(71, 30)
        live2, ck2 = _with_ckpts(model, c2, [192], side)
        assert prefix_disk.save(key, c2, live2, model, BUDGET, ckpts=ck2)
        idx = prefix_disk._read_index(prefix_disk._model_slot(Path(key[0]).resolve()))
        ent = next(e for e in idx["entries"] if e["n"] == len(c2))
        assert [c["at"] for c in ent["ckpts"]] == at, f"carried {[c['at'] for c in ent['ckpts']]}"
        cks = sorted((env.tmp / "cache").rglob(f"{prefix_disk._CK_PREFIX}*"))
        assert len(cks) == 3, "a shared checkpoint was written twice"

        # A corrupted checkpoint is a miss that keeps the file and the others.
        f128 = next(c["file"] for c in ent["ckpts"] if c["at"] == 128)
        path = next(x for x in cks if x.name == f128)
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        path.write_bytes(bytes(raw))
        prefix_disk.reset()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert prefix_disk.load(key, b, model, 0, BUDGET) is None, "loaded a corrupted checkpoint"
        assert "checkpoint @ 128" in buf.getvalue(), buf.getvalue()
        assert prefix_disk.has(key, a[:p]), "a bad checkpoint deleted its file"
        assert not path.exists(), "the bad checkpoint file was kept"
        got = prefix_disk.load(key, a, model, 0, BUDGET)
        assert got is not None and got[0] == a[:p], "exact match broke after a bad checkpoint"
        got = prefix_disk.load(key, b, model, 0, BUDGET)
        assert got is not None and got[0] == b[:64], "the other checkpoints went with the bad one"

        # Slots that do not match the cache are not written.
        c3 = _tokens(72, 250)
        live3, ck3 = _with_ckpts(model, c3, [192], side)
        bad = {192: {i: [None for _ in s] for i, s in ck3[192].items()}}
        assert prefix_disk.save(key, c3, live3, model, BUDGET, ckpts=bad)
        idx = prefix_disk._read_index(prefix_disk._model_slot(Path(key[0]).resolve()))
        ent = next(e for e in idx["entries"] if e["n"] == len(c3))
        assert not ent.get("ckpts"), "slots that do not fit the cache were stored"

        # A stale fingerprint takes the checkpoint files with it.
        cfg = Path(key[0]) / "config.json"
        cfg.write_text('{"model_type": "qwen3_next", "ck": 1}')
        prefix_disk.reset()
        assert prefix_disk.load(key, b, model, 0, BUDGET) is None
        left = list((env.tmp / "cache").rglob(f"{prefix_disk._CK_PREFIX}*"))
        assert not left, f"stale drop left {left}"
        tag = " with a side cache" if side else ""
        print(f"OK: a hybrid file resumes from its last checkpoint before a change, bit-identically{tag}")
    finally:
        prefix_disk._MIN_PARTIAL_GAIN = old_gain
        env.close()


def check_partial_side_cache():
    """A non-recurrent layout with a side cache is served part way too."""
    from tests.test_verify_decode import build_trim

    env = _Env()
    try:
        model, vocab = build_trim()
        _with_side(model)
        assert prefix_disk.trims_exactly(make_prompt_cache(model)), "side-cache attention counted as fixed"
        key = _model_dir(env.tmp)
        rng = np.random.default_rng(13)
        a = rng.integers(0, vocab, size=560).tolist()
        k, p = 400, 500
        b = a[:k] + [(t + 1) % vocab for t in a[k:]]
        live = make_prompt_cache(model)
        _prefill(model, live, a[:p])
        _fill_side(live, p)
        assert prefix_disk.save(key, a[:p], live, model, BUDGET)
        prefix_disk.reset()
        got = prefix_disk.load(key, b, model, 0, BUDGET)
        assert got is not None and got[0] == b[:k], "side-cache layout missed a partial match"
        cache = got[1]
        for c in cache:
            if isinstance(c, _AttnSide):
                want = mx.arange(p * 4, dtype=mx.float32).reshape(1, p, 4)[:, :k]
                assert bool(mx.array_equal(c.indexer.keys, want)), "side cache not cut to the shared tokens"
        ref = make_prompt_cache(model)
        _prefill(model, ref, b[:k])
        la = model(mx.array([b[k:]]), cache=cache)
        lb = model(mx.array([b[k:]]), cache=ref)
        assert float(mx.abs(la - lb).max()) == 0.0, "partial side-cache hit differs from a fresh prefill"
        print("OK: attention with a side cache loads a partial match cut back, side cache included")
    finally:
        env.close()


def _driven(model, toks, step, log):
    from pagedmoe import server

    c = make_prompt_cache(model)
    done = server._prefill_remainder(
        model, toks, c, n_model_layers=len(c), draft_model=None, draft_cache=None,
        step_cap=step, progress=lambda *_: None, total=len(toks), processed=0,
        target=len(toks), chunk_log=log,
    )
    assert done == len(toks)
    return c


def _new_engine(tag: str):
    prefix_disk.reset()
    prefix_disk._engine_fp = hashlib.sha256(tag.encode()).hexdigest()


def _quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **kw)
    return out, buf.getvalue()


def check_engine_change_replay():
    """An engine-only change keeps a file whose last chunks replay to the same
    KV, and drops every file from the old engine when they do not."""
    import copy

    from pagedmoe import server
    from tests.test_verify_decode import build_trim

    env = _Env()
    old = es_config.ADAPTIVE_PREFILL
    es_config.ADAPTIVE_PREFILL = False
    try:
        for name, (model, vocab) in (("plain", build_trim()), ("hybrid", (build(), ARGS["vocab_size"]))):
            prefix_disk.reset()
            key = _model_dir(env.tmp, f"replay-{name}")
            rng = np.random.default_rng(21)
            a = rng.integers(0, vocab, size=420).tolist()
            p = 200
            log: list = []
            if name == "plain":
                # The pin is a fork of a pass that ran past it, like the fused pass.
                full = _driven(model, a[:260], 32, log)
                live = copy.deepcopy(full)
                for c in live:
                    c.trim(260 - p)
            else:
                live = _driven(model, a[:p], 32, log)
            spec = server._replay_spec(log, 0, a, p)
            assert spec is not None and spec["lo"] == 160, f"{name}: replay spec {spec and spec['lo']}"
            assert prefix_disk.save(key, a[:p], live, model, BUDGET, replay=spec)
            run = lambda c, ids, sizes: server._replay_chunks(model, c, ids, sizes)  # noqa: E731

            _new_engine(f"{name}-1")
            v0 = prefix_disk.STATS["verified"]
            got, said = _quiet(prefix_disk.load, key, a, model, 0, BUDGET, replay=run)
            assert got is not None and got[0] == a[:p], f"{name}: engine change lost a file that replays the same: {said}"
            assert prefix_disk.STATS["verified"] == v0 + 1 and "same KV" in said, said
            assert prefix_disk.has(key, a[:p]), f"{name}: verified file not moved to the new fingerprint"
            if name == "plain":
                ref = _driven(model, a[:260], 32, [])
                for c in ref:
                    c.trim(260 - p)
            else:
                ref = _driven(model, a[:p], 32, [])
            la = model(mx.array([a[p:]]), cache=got[1])
            lb = model(mx.array([a[p:]]), cache=ref)
            d = float(mx.abs(la - lb).max())
            assert d == 0.0, f"{name}: verified file differs from a fresh prefill by {d}"

            # Already moved: the next lookup does not replay again.
            got, said = _quiet(prefix_disk.load, key, a, model, 0, BUDGET, replay=run)
            assert got is not None and "replaying" not in said, said

            # The new engine writes something else: every old file goes.
            other = a[:p] + rng.integers(0, vocab, size=120).tolist()
            log2: list = []
            if name == "plain":
                c2 = copy.deepcopy(_driven(model, other[:300], 32, log2))
                for c in c2:
                    c.trim(40)
                n2 = 260
            else:
                c2 = _driven(model, other[:260], 32, log2)
                n2 = 260
            assert prefix_disk.save(key, other[:n2], c2, model, BUDGET, replay=server._replay_spec(log2, 0, other, n2))
            _new_engine(f"{name}-2")
            bent = lambda c, ids, sizes: run(c, [(ids[0] + 1) % vocab] + ids[1:], sizes)  # noqa: E731
            got, said = _quiet(prefix_disk.load, key, a, model, 0, BUDGET, replay=bent)
            assert got is None and "writes different KV" in said, said
            slot = prefix_disk._model_slot(Path(key[0]).resolve())
            assert not prefix_disk._read_index(slot)["entries"], f"{name}: old-engine entries kept"
            assert not list(slot.glob(f"*{prefix_disk._SUFFIX}")), f"{name}: old-engine files kept"

            # A settings change still drops, and no replay runner means no check.
            assert prefix_disk.save(key, a[:p], live, model, BUDGET, replay=spec)
            _new_engine(f"{name}-3")
            got, said = _quiet(prefix_disk.load, key, a, model, 0, BUDGET)
            assert got is None and "engine changed" in said, said
            assert prefix_disk.save(key, a[:p], live, model, BUDGET, replay=spec)
            os.environ["PAGED_MOE_ATTN_SUB_CHUNK"] = "123"
            try:
                _new_engine(f"{name}-4")
                got, said = _quiet(prefix_disk.load, key, a, model, 0, BUDGET, replay=run)
                assert got is None and "settings changed" in said, said
            finally:
                os.environ.pop("PAGED_MOE_ATTN_SUB_CHUNK", None)
            print(f"OK: {name} file survives an engine change when its last chunks replay bit-identically")
    finally:
        es_config.ADAPTIVE_PREFILL = old
        env.close()


def check_ckpt_positions():
    old = (es_config.PREFIX_DISK_CKPT_EVERY, es_config.PREFIX_DISK_CKPT_MAX)
    try:
        es_config.PREFIX_DISK_CKPT_EVERY, es_config.PREFIX_DISK_CKPT_MAX = 2048, 4
        assert prefix_disk.ckpt_positions(0, 14000) == [4096, 8192, 12288]
        assert prefix_disk.ckpt_positions(5000, 14000) == [8192, 12288]
        assert prefix_disk.ckpt_positions(0, 9000) == [2048, 4096, 6144, 8192]
        assert prefix_disk.ckpt_positions(0, 2048) == []
        es_config.PREFIX_DISK_CKPT_EVERY = 0
        assert prefix_disk.ckpt_positions(0, 14000) == []
        print("OK: checkpoint positions sit on a fixed grid, thinned to the cap")
    finally:
        es_config.PREFIX_DISK_CKPT_EVERY, es_config.PREFIX_DISK_CKPT_MAX = old


def check_pin_put():
    model = build()
    a = _tokens(8, 100)
    live = make_prompt_cache(model)
    _prefill(model, live, a[:70])
    pin = prefix_cache.StablePrefixPin()
    assert not pin.put("m", a[:69], live), "pinned a cache longer than its tokens"
    assert pin.put("m", a[:70], live)
    hit = pin.lookup("m", a)
    assert hit is not None and hit[2] == 70
    print("OK: put pins exactly the tokens the cache holds")


def _dead_pid() -> int:
    import subprocess

    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _write_open(slot: Path, rec: dict) -> None:
    import json
    import time

    (slot / prefix_disk._OPEN).write_text(json.dumps({"at": time.time(), **rec}))


def check_open_marker():
    """An engine that died mid-load or mid-save leaves open.json, and the next
    one drops that file or skips that save instead of repeating it."""
    env = _Env()
    try:
        model = build()
        key = _model_dir(env.tmp)
        slot = prefix_disk._model_slot(Path(key[0]).resolve())
        a = _tokens(11, 120)
        p = 80
        live = make_prompt_cache(model)
        _prefill(model, live, a[:p])
        assert prefix_disk.save(key, a[:p], live, model, BUDGET)
        assert not (slot / prefix_disk._OPEN).exists(), "save left its marker"
        got = prefix_disk.load(key, a, model, 0, BUDGET)
        assert got is not None and not (slot / prefix_disk._OPEN).exists(), "load left its marker"

        file = prefix_disk._read_index(slot)["entries"][0]["file"]
        _write_open(slot, {"op": "load", "file": file, "pid": os.getppid()})
        got = prefix_disk.load(key, a, model, 0, BUDGET)
        assert got is not None, "a live engine's marker dropped the file"

        _write_open(slot, {"op": "load", "file": file, "pid": _dead_pid()})
        got, said = _quiet(prefix_disk.load, key, a, model, 0, BUDGET)
        assert got is None and "stopped while loading" in said, said
        assert not env.files(), "the file a dead engine was loading is still there"
        assert not (slot / prefix_disk._OPEN).exists()

        fp = prefix_disk.fingerprint(Path(key[0]).resolve())
        _write_open(slot, {"op": "save", "tok_sha": prefix_disk._tok_sha(a[:p]), "fp": fp, "pid": _dead_pid()})
        ok, said = _quiet(prefix_disk.save, key, a[:p], live, model, BUDGET)
        assert not ok and "stopped while saving" in said, said
        assert not env.files(), "saved the prefix a dead engine was saving"
        other = make_prompt_cache(model)
        _prefill(model, other, a[:90])
        assert prefix_disk.save(key, a[:90], other, model, BUDGET), "skip spread to another prefix"

        _new_engine("open-marker-2")
        assert prefix_disk.save(key, a[:p], live, model, BUDGET), "skip outlived the engine it was for"

        file = next(e["file"] for e in prefix_disk._read_index(slot)["entries"] if e["n"] == p)
        prefix_disk._mark_open(slot, {"op": "load", "file": file})
        prefix_disk.stopped_cleanly()
        assert not (slot / prefix_disk._OPEN).exists(), "a clean stop left its marker"
        got = prefix_disk.load(key, a, model, 0, BUDGET)
        assert got is not None and got[0] == a[:p], "a clean stop mid-load dropped the file"
        print("OK: a dead engine's load drops its file, its save is not retried; live markers stay")
    finally:
        env.close()


def check_disabled():
    env = _Env()
    es_config.PREFIX_DISK = False
    try:
        model = build()
        key = _model_dir(env.tmp)
        a = _tokens(9, 100)
        live = make_prompt_cache(model)
        _prefill(model, live, a[:70])
        assert not prefix_disk.save(key, a[:70], live, model, BUDGET)
        assert not env.files()
        assert prefix_disk.load(key, a, model, 0, BUDGET) is None
        print("OK: PREFIX_DISK=0 writes and reads nothing")
    finally:
        es_config.PREFIX_DISK = True
        env.close()


def test_round_trip():
    check_round_trip(side=False)


def test_round_trip_side_cache():
    check_round_trip(side=True)


def test_misses():
    check_misses()


def test_settings():
    check_settings()


def test_corruption_and_layout():
    check_corruption_and_layout()


def test_eviction():
    check_eviction()


def test_missing_model_dropped():
    check_missing_model_dropped()


def test_eviction_keeps_other_workspaces():
    check_eviction_keeps_other_workspaces()


def test_cap_keeps_newest_per_model():
    check_cap_keeps_newest_per_model()


def test_numerics_fingerprint():
    check_numerics_fingerprint()


def test_diagnostics_do_not_move_the_settings(monkeypatch):
    from pagedmoe import prefix_disk

    base = prefix_disk._numerics_env()
    for k in ("PAGED_MOE_GPU_CRUMBS", "PAGED_MOE_WATCHDOG_OSLOG", "PAGED_MOE_WATCHDOG_STALL_S"):
        monkeypatch.setenv(k, "1")
    assert prefix_disk._numerics_env() == base
    monkeypatch.setenv("PAGED_MOE_FUSED_PREFILL", "7")
    assert prefix_disk._numerics_env() != base


def test_client_files_keep_host_files():
    check_client_files_keep_host_files()


def test_chat_files():
    check_chat_files()


def test_sealed():
    check_sealed()


def test_partial():
    check_partial()


def test_head_file():
    check_head_file()


def test_shared_file():
    check_shared_file()


def test_hybrid_ckpt():
    check_hybrid_ckpt(side=False)


def test_hybrid_ckpt_side_cache():
    check_hybrid_ckpt(side=True)


def test_ckpt_positions():
    check_ckpt_positions()


def test_partial_side_cache():
    check_partial_side_cache()


def test_engine_change_replay():
    check_engine_change_replay()


def test_pin_put():
    check_pin_put()


def test_open_marker():
    check_open_marker()


def test_disabled():
    check_disabled()


def main():
    check_round_trip(side=False)
    check_round_trip(side=True)
    check_misses()
    check_settings()
    check_corruption_and_layout()
    check_eviction()
    check_missing_model_dropped()
    check_eviction_keeps_other_workspaces()
    check_cap_keeps_newest_per_model()
    check_numerics_fingerprint()
    check_client_files_keep_host_files()
    check_chat_files()
    check_sealed()
    check_partial()
    check_head_file()
    check_shared_file()
    check_hybrid_ckpt(side=False)
    check_hybrid_ckpt(side=True)
    check_ckpt_positions()
    check_partial_side_cache()
    check_engine_change_replay()
    check_pin_put()
    check_open_marker()
    check_disabled()
    print("\nOK: prefix disk tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
