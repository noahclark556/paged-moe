# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Find, fetch and track a checkpoint's multi-token-prediction (MTP) head.

MLX conversions usually drop the MTP layers even when the base model ships
them. This module works out whether the upstream repo has them, how big they
are, and pulls only those tensors with HTTP range reads into one local
safetensors file. mtp.py turns that file into a quantized head the first time
the model loads banked, and drafts with it.

Standard library only, so the app can run it next to a live engine. Everything
it writes lives under MTP_DIR/<config key>/, never inside the model folder.

    python -m pagedmoe.mtp_fetch status --model DIR [--remote] [--refresh]
    python -m pagedmoe.mtp_fetch fetch  --model DIR [--max-bytes N]
    python -m pagedmoe.mtp_fetch remove --model DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Bump when the converted head layout or a head builder changes. A head.json
# with another version is converted again from raw (or refetched).
HEAD_VERSION = 3

# model_type -> how the checkpoint names its MTP tensors.
#   "mtp":   a separate `mtp.*` block (Qwen3-Next, Qwen3.5, Qwen4-exp)
#   "nextn": extra decoder layers past num_hidden_layers (DeepSeek-V3, GLM-4.x)
SCHEMES = {
    "qwen4_exp": "mtp",
    "qwen3_next": "mtp",
    "qwen3_5_moe": "mtp",
    "qwen3_5": "mtp",
    "deepseek_v3": "nextn",
    "kimi_k2": "nextn",
    "glm4_moe": "nextn",
}

HF = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
REMOTE_TTL_S = 7 * 24 * 3600
CHUNK = 1 << 20  # F_NOCACHE only holds for writes up to 1 MB on macOS
F_NOCACHE = 48


# ---------------------------------------------------------------- local facts


def _read_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            out = json.load(f)
        return out if isinstance(out, dict) else None
    except (OSError, ValueError):
        return None


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def text_config(cfg: dict) -> dict:
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) else cfg


def model_type(cfg: dict) -> str:
    return str(cfg.get("model_type") or text_config(cfg).get("model_type") or "")


def declared_layers(cfg: dict) -> int | None:
    """MTP depth the config declares, None when it says nothing."""
    for c in (cfg, text_config(cfg)):
        for key in ("mtp_num_hidden_layers", "num_nextn_predict_layers", "num_mtp_layers"):
            v = c.get(key)
            if isinstance(v, int):
                return v
    return None


def num_layers(cfg: dict) -> int:
    return int(text_config(cfg).get("num_hidden_layers") or cfg.get("num_hidden_layers") or 0)


def is_mtp_name(name: str, scheme: str, n_layers: int, depth: int = 1) -> bool:
    if scheme == "mtp":
        return name.startswith("mtp.") or ".mtp." in name
    if scheme == "nextn":
        m = re.match(r"^(?:model\.)?(?:language_model\.)?(?:model\.)?layers\.(\d+)\.", name)
        return bool(m) and n_layers <= int(m.group(1)) < n_layers + max(1, depth)
    return False


def local_names(model_path: Path) -> list[str]:
    idx = _read_json(model_path / "model.safetensors.index.json")
    if idx and isinstance(idx.get("weight_map"), dict):
        return list(idx["weight_map"])
    single = model_path / "model.safetensors"
    if single.exists():
        try:
            return list(_local_header(single))
        except (OSError, ValueError):
            return []
    return []


def _local_header(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def config_key(model_path: Path) -> str:
    try:
        raw = (model_path / "config.json").read_bytes()
    except OSError:
        raw = str(model_path.resolve()).encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def root_dir() -> Path:
    env = os.environ.get("PAGED_MOE_MTP_DIR", "").strip()
    return Path(env).expanduser() if env else Path.home() / ".paged_moe" / "mtp"


def head_dir(model_path: Path) -> Path:
    return root_dir() / config_key(model_path)


def _pid_alive(pid) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --------------------------------------------------------------------- status


def local_status(model_path: Path) -> dict:
    """What the engine would do with MTP for this checkpoint, without network."""
    model_path = Path(model_path).expanduser()
    cfg = _read_json(model_path / "config.json") or {}
    mt = model_type(cfg)
    scheme = SCHEMES.get(mt)
    d = head_dir(model_path)
    out: dict = {"model": str(model_path), "model_type": mt, "dir": str(d), "scheme": scheme}
    if not scheme:
        out["state"] = "unsupported"
        return out

    head = _read_json(d / "head.json") or {}
    if head.get("version") == HEAD_VERSION:
        if head.get("state") == "ready" and (d / "head.safetensors").exists():
            out.update(state="ready", bytes=int(head.get("bytes") or 0), source=head.get("source"))
            return out
        if head.get("state") == "incompatible" and not _retry_packed_failure(head, model_path, cfg, scheme, d):
            out.update(state="incompatible", error=head.get("error") or "head did not build")
            return out

    names = local_names(model_path)
    depth = declared_layers(cfg) or 1
    if any(is_mtp_name(n, scheme, num_layers(cfg), depth) for n in names):
        out["state"] = "bundled"
        return out

    raw = _read_json(d / "raw.json")
    if raw and (d / "raw.safetensors").exists():
        out.update(state="downloaded", bytes=int(raw.get("bytes") or 0), repo=raw.get("repo"))
        return out

    remote = _read_json(d / "remote.json") or {}
    fetch = _read_json(d / "fetch.json") or {}
    total = int(remote.get("bytes") or fetch.get("total") or 0)
    if fetch.get("state") == "downloading" and _pid_alive(fetch.get("pid")):
        out.update(
            state="downloading", bytes=total, done=int(fetch.get("done") or 0),
            repo=remote.get("repo"),
        )
        return out
    if fetch.get("state") in ("failed", "paused", "downloading"):
        out.update(
            state="failed" if fetch.get("state") == "failed" else "paused",
            bytes=total, done=int(fetch.get("done") or 0),
            error=fetch.get("error") or ("download stopped" if fetch.get("state") != "failed" else None),
            repo=remote.get("repo"),
        )
        return out

    if fetch.get("state") == "removed":
        out["removed"] = True
    if remote.get("state") == "available":
        out.update(state="missing", bytes=total, repo=remote.get("repo"))
    elif remote.get("state") == "none":
        out.update(state="none", reason=remote.get("reason"))
    else:
        out["state"] = "unknown"
        if remote.get("error"):
            out["error"] = remote["error"]
    return out


# --------------------------------------------------------------------- remote


def _token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    for p in (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token",
        Path.home() / ".huggingface" / "token",
    ):
        try:
            v = Path(p).read_text().strip()
            if v:
                return v
        except OSError:
            pass
    return None


def _request(url: str, rng: tuple[int, int] | None = None, timeout: float = 20.0):
    req = urllib.request.Request(url, headers={"User-Agent": "pagedmoe-mtp/1"})
    tok = _token()
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    if rng is not None:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1] - 1}")
    return urllib.request.urlopen(req, timeout=timeout)


def _get_json(url: str, timeout: float = 20.0) -> dict | None:
    try:
        with _request(url, timeout=timeout) as r:
            out = json.loads(r.read())
        return out if isinstance(out, dict) else None
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            return None
        raise


def _get_text(url: str, timeout: float = 20.0) -> str | None:
    try:
        with _request(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            return None
        raise


def _range(url: str, start: int, end: int) -> bytes:
    with _request(url, (start, end)) as r:
        data = r.read()
        if r.status == 200 and start > 0:
            raise RuntimeError("server ignored the range request")
    if len(data) != end - start:
        raise RuntimeError(f"short range read ({len(data)} of {end - start} bytes)")
    return data


_REPO = re.compile(r"^[\w.-]+/[\w.-]+$")


def _base_models(readme: str) -> list[str]:
    out: list[str] = []
    m = re.match(r"^---\s*\n(.*?)\n---", readme, re.S)
    if m:
        lines = m.group(1).splitlines()
        for i, line in enumerate(lines):
            if not line.startswith("base_model:"):
                continue
            rest = line.split(":", 1)[1].strip()
            if rest:
                out.append(rest.strip("'\""))
            for nxt in lines[i + 1:]:
                s = nxt.strip()
                if not s.startswith("- "):
                    break
                out.append(s[2:].strip().strip("'\""))
    for m2 in re.finditer(r"converted to MLX[^\[]*\[([\w.-]+/[\w.-]+)\]", readme):
        out.append(m2.group(1))
    return [r for r in out if _REPO.match(r)]


def candidate_repos(model_path: Path) -> list[str]:
    out: list[str] = []
    try:
        out += _base_models((model_path / "README.md").read_text(errors="replace"))
    except OSError:
        pass
    meta = _read_json(model_path / "ga-model.json") or {}
    if isinstance(meta.get("source"), str):
        out.append(meta["source"])
    cfg = _read_json(model_path / "config.json") or {}
    if isinstance(cfg.get("_name_or_path"), str):
        out.append(cfg["_name_or_path"])
    seen: list[str] = []
    for r in out:
        if _REPO.match(r) and r not in seen:
            seen.append(r)
    return seen


def _same_model(local: dict, upstream: dict) -> bool:
    a, b = text_config(local), text_config(upstream)
    for key in ("hidden_size", "vocab_size", "num_hidden_layers"):
        if a.get(key) is not None and b.get(key) is not None and a.get(key) != b.get(key):
            return False
    return True


def _plan_repo(repo: str, local_cfg: dict, scheme: str) -> dict | None:
    info = _get_json(f"{HF}/api/models/{repo}")
    if not info or not info.get("sha"):
        return None
    rev = info["sha"]
    base = f"{HF}/{repo}/resolve/{rev}"
    cfg = _get_json(f"{base}/config.json")
    if not cfg or not _same_model(local_cfg, cfg):
        return None
    depth = declared_layers(cfg)
    if not depth:
        return {"state": "none", "repo": repo, "reason": "upstream config declares no MTP layers"}
    L = num_layers(cfg)
    idx = _get_json(f"{base}/model.safetensors.index.json")
    if idx and isinstance(idx.get("weight_map"), dict):
        wmap = idx["weight_map"]
    else:
        wmap = None
    shards: dict[str, list[str]] = {}
    if wmap is not None:
        for name, shard in wmap.items():
            if is_mtp_name(name, scheme, L, depth):
                shards.setdefault(shard, []).append(name)
    else:
        shards["model.safetensors"] = []
    tensors = []
    for shard, names in sorted(shards.items()):
        url = f"{base}/{shard}"
        (n,) = struct.unpack("<Q", _range(url, 0, 8))
        hdr = json.loads(_range(url, 8, 8 + n))
        hdr.pop("__metadata__", None)
        want = names or [k for k in hdr if is_mtp_name(k, scheme, L, depth)]
        for name in want:
            t = hdr.get(name)
            if not t:
                continue
            s, e = t["data_offsets"]
            tensors.append({
                "name": name, "shard": shard, "start": 8 + n + int(s), "size": int(e) - int(s),
                "dtype": t["dtype"], "shape": t["shape"],
            })
    if not tensors:
        return {"state": "none", "repo": repo, "reason": "upstream checkpoint has no MTP tensors"}
    return {
        "state": "available", "repo": repo, "revision": rev, "scheme": scheme,
        "layers": L, "depth": depth, "tensors": tensors,
        "bytes": sum(t["size"] for t in tensors),
    }


def remote_plan(model_path: Path, refresh: bool = False) -> dict:
    """Upstream MTP tensors for this checkpoint, cached in remote.json."""
    model_path = Path(model_path).expanduser()
    d = head_dir(model_path)
    cached = _read_json(d / "remote.json")
    if (
        cached and not refresh and cached.get("state") in ("available", "none")
        and time.time() - float(cached.get("checked_at") or 0) < REMOTE_TTL_S
    ):
        return cached
    cfg = _read_json(model_path / "config.json") or {}
    scheme = SCHEMES.get(model_type(cfg))
    if not scheme:
        plan = {"state": "none", "reason": f"no MTP support for {model_type(cfg) or 'this model'}"}
    else:
        plan = None
        first_none = None
        err = None
        tried: list[str] = []
        queue = candidate_repos(model_path)
        while queue and len(tried) < 6:
            repo = queue.pop(0)
            if repo in tried:
                continue
            tried.append(repo)
            try:
                got = _plan_repo(repo, cfg, scheme)
                if got is None:
                    readme = _get_text(f"{HF}/{repo}/raw/main/README.md")
                    queue += [r for r in _base_models(readme or "") if r not in tried]
                    continue
            except (OSError, ValueError, RuntimeError, struct.error) as e:
                err = f"{repo}: {e}"
                continue
            if got.get("state") == "available":
                plan = got
                break
            first_none = first_none or got
            readme = None
            try:
                readme = _get_text(f"{HF}/{repo}/raw/main/README.md")
            except OSError:
                pass
            queue += [r for r in _base_models(readme or "") if r not in tried]
        if plan is None:
            if err and first_none is None:
                plan = {"state": "error", "error": err}
            elif first_none is not None:
                plan = first_none
            else:
                plan = {"state": "none", "reason": "no upstream repo found for this checkpoint"}
    plan["checked_at"] = time.time()
    if plan.get("state") != "error" or not cached:
        _write_json(d / "remote.json", plan)
    return plan


# ------------------------------------------------------------------- download


class _Stop(Exception):
    pass


def _layout(tensors: list[dict]) -> tuple[bytes, list[tuple[dict, int]]]:
    meta: dict = {"__metadata__": {"format": "pt"}}
    placed = []
    off = 0
    for t in tensors:
        meta[t["name"]] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [off, off + t["size"]]}
        placed.append((t, off))
        off += t["size"]
    hdr = json.dumps(meta, separators=(",", ":")).encode()
    hdr += b" " * ((-len(hdr)) % 8)
    return struct.pack("<Q", len(hdr)) + hdr, placed


PIECE = 32 << 20
WORKERS = 4


def _pieces(placed: list[tuple[dict, int]]) -> list[tuple[int, str, int, int, int]]:
    """(index, shard, source offset, length, output offset) in PIECE-sized ranges."""
    out = []
    for t, off in placed:
        pos = 0
        while pos < t["size"]:
            n = min(PIECE, t["size"] - pos)
            out.append((len(out), t["shard"], t["start"] + pos, n, off + pos))
            pos += n
    return out


def _get_piece(fd: int, hdr: int, base: str, piece) -> None:
    _, shard, src, n, off = piece
    tries = 0
    while True:
        got = 0
        try:
            with _request(f"{base}/{shard}", (src, src + n), timeout=60) as r:
                if r.status != 206:
                    raise RuntimeError("server ignored the range request")
                while got < n:
                    buf = r.read(min(CHUNK, n - got))
                    if not buf:
                        raise RuntimeError("connection closed early")
                    os.pwrite(fd, buf, hdr + off + got)
                    got += len(buf)
            return
        except (OSError, RuntimeError) as e:
            tries += 1
            if tries > 6:
                raise RuntimeError(f"{shard} @ {src}: {e}") from e
            time.sleep(min(30, 2 ** tries))


def _download(fd, hdr, base, todo, have: set, prog: dict, state: Path) -> None:
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    sizes = {p[0]: p[3] for p in todo}
    base_done = int(prog.get("done") or 0)
    pool = ThreadPoolExecutor(max_workers=WORKERS)
    futs = {pool.submit(_get_piece, fd, hdr, base, p): p[0] for p in todo}

    def harvest() -> None:
        for f, i in futs.items():
            if f.done() and not f.cancelled() and f.exception() is None:
                have.add(i)
        prog["done"] = base_done + sum(sizes[i] for i in have if i in sizes)
        prog["chunks"] = sorted(have)

    try:
        running = set(futs)
        last = 0.0
        while running:
            finished, running = wait(running, timeout=1.0, return_when=FIRST_COMPLETED)
            for f in finished:
                f.result()
            if time.time() - last > 1.0 or not running:
                harvest()
                _write_json(state, prog)
                last = time.time()
    finally:
        # Workers write through `fd`; the caller closes it once this returns.
        pool.shutdown(wait=True, cancel_futures=True)
        harvest()


def fetch(
    model_path: Path, max_bytes: int | None = None, quiet: bool = False, auto: bool = False,
) -> dict:
    """Download the upstream MTP tensors into raw.safetensors. Resumable.

    auto=True is the unattended path: it leaves a head the user removed alone.
    """
    model_path = Path(model_path).expanduser()
    d = head_dir(model_path)
    d.mkdir(parents=True, exist_ok=True)
    st = local_status(model_path)
    if st["state"] in ("ready", "bundled", "downloaded", "unsupported", "incompatible", "downloading"):
        return st
    if auto and st.get("removed"):
        return {**st, "skipped": "removed by the user"}
    plan = remote_plan(model_path)
    if plan.get("state") != "available":
        return local_status(model_path)
    total = int(plan["bytes"])
    if max_bytes is not None and total > max_bytes:
        out = local_status(model_path)
        out["skipped"] = f"{total} bytes is over the {max_bytes} byte auto-download limit"
        return out

    lock = d / "fetch.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            pid = int(lock.read_text().strip() or 0)
        except (OSError, ValueError):
            pid = 0
        if _pid_alive(pid):
            return local_status(model_path)
        lock.unlink(missing_ok=True)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)

    prog = {"state": "downloading", "pid": os.getpid(), "total": total, "done": 0,
            "repo": plan["repo"], "started": time.time()}

    def _stop(_sig, _frm):
        raise _Stop()

    old = signal.signal(signal.SIGTERM, _stop)
    part = d / "raw.safetensors.part"
    try:
        try:
            os.nice(10)
        except OSError:
            pass
        header, placed = _layout(plan["tensors"])
        pieces = _pieces(placed)
        sig = hashlib.sha1(header).hexdigest()
        prev = _read_json(d / "fetch.json") or {}
        have: set[int] = set()
        if part.exists() and prev.get("layout") == sig and part.stat().st_size == total + len(header):
            have = {int(i) for i in prev.get("chunks") or [] if 0 <= int(i) < len(pieces)}
        need = sum(p[3] for p in pieces if p[0] not in have) + (1 << 30)
        free = shutil.disk_usage(d).free
        if free < need:
            raise RuntimeError(f"not enough free disk ({free >> 20} MB free, {need >> 20} MB needed)")
        fd = os.open(part, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                import fcntl
                fcntl.fcntl(fd, F_NOCACHE, 1)
            except (ImportError, OSError):
                pass
            if not have:
                os.ftruncate(fd, 0)
                os.pwrite(fd, header, 0)
                os.ftruncate(fd, total + len(header))
            prog.update(layout=sig, chunks=sorted(have),
                        done=sum(p[3] for p in pieces if p[0] in have))
            _write_json(d / "fetch.json", prog)
            base = f"{HF}/{plan['repo']}/resolve/{plan['revision']}"
            _download(fd, len(header), base, [p for p in pieces if p[0] not in have],
                      have, prog, d / "fetch.json")
            os.fsync(fd)
        finally:
            os.close(fd)
        if len(have) != len(pieces) or part.stat().st_size != total + len(header):
            raise RuntimeError("download size does not match the plan")
        os.replace(part, d / "raw.safetensors")
        _write_json(d / "raw.json", {
            "repo": plan["repo"], "revision": plan["revision"], "scheme": plan["scheme"],
            "layers": plan["layers"], "depth": plan["depth"], "bytes": total,
            "finished": time.time(),
        })
        prog.update(state="done", done=total)
        _write_json(d / "fetch.json", prog)
        if not quiet:
            print(f"[mtp] downloaded {total / 1e9:.2f} GB from {plan['repo']}", flush=True)
    except _Stop:
        prog.update(state="paused", error="download stopped")
        _write_json(d / "fetch.json", prog)
    except Exception as e:
        prog.update(state="failed", error=str(e)[:400])
        _write_json(d / "fetch.json", prog)
        if not quiet:
            print(f"[mtp] download failed: {e}", flush=True)
    finally:
        signal.signal(signal.SIGTERM, old)
        lock.unlink(missing_ok=True)
    return local_status(model_path)


def remove(model_path: Path) -> dict:
    model_path = Path(model_path).expanduser()
    d = head_dir(model_path)
    fetch_state = _read_json(d / "fetch.json") or {}
    pid = fetch_state.get("pid")
    if fetch_state.get("state") == "downloading" and _pid_alive(pid) and int(pid) != os.getpid():
        try:
            os.kill(int(pid), signal.SIGTERM)
            for _ in range(50):
                if not _pid_alive(pid):
                    break
                time.sleep(0.1)
        except OSError:
            pass
    remote = _read_json(d / "remote.json")
    shutil.rmtree(d, ignore_errors=True)
    if remote and remote.get("state") in ("available", "none"):
        _write_json(d / "remote.json", remote)
    st = local_status(model_path)
    if st["state"] == "missing":
        # A removed head stays removed until the user asks for it again.
        _write_json(d / "fetch.json", {"state": "removed"})
        st = local_status(model_path)
    return st


def _rebuild_source(model_path: Path, cfg: dict, scheme: str, d: Path) -> bool:
    names = local_names(model_path)
    depth = declared_layers(cfg) or 1
    if any(is_mtp_name(name, scheme, num_layers(cfg), depth) for name in names):
        return True
    raw = _read_json(d / "raw.json")
    return bool(raw and (d / "raw.safetensors").exists())


def _retry_packed_failure(head: dict, model_path: Path, cfg: dict, scheme: str, d: Path) -> bool:
    """One retry for an incompatible stamp left by a packed head.

    Older builds loaded the head before quantizing it, so a checkpoint that
    already stored scales and biases was marked incompatible. Those stamps
    have no builder field. A failure written by this builder sets builder to
    1 and stays incompatible, which is what the settings page shows.
    """
    if int(head.get("builder") or 0) >= 1:
        return False
    if "parameters not in model" not in str(head.get("error") or ""):
        return False
    return _rebuild_source(model_path, cfg, scheme, d)


def mark_incompatible(model_path: Path, error: str) -> None:
    d = head_dir(Path(model_path))
    _write_json(d / "head.json", {
        "state": "incompatible", "version": HEAD_VERSION, "builder": 1, "error": error[:400],
    })
    (d / "head.safetensors").unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pagedmoe.mtp_fetch", description="MTP head status and download")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("status", "fetch", "remove"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--dir", default=None)
        if name == "status":
            p.add_argument("--remote", action="store_true", help="check upstream when not known yet")
            p.add_argument("--refresh", action="store_true", help="check upstream even if cached")
        if name == "fetch":
            p.add_argument("--max-bytes", type=int, default=None)
            p.add_argument("--auto", action="store_true", help="unattended: skip heads the user removed")
    a = ap.parse_args(argv)
    if a.dir:
        os.environ["PAGED_MOE_MTP_DIR"] = a.dir
    model = Path(a.model).expanduser()
    if a.cmd == "status":
        st = local_status(model)
        if st["state"] in ("unknown", "missing", "none") and (a.remote or a.refresh):
            try:
                remote_plan(model, refresh=a.refresh)
            except Exception as e:
                st["error"] = str(e)[:400]
            st = {**st, **local_status(model)}
    elif a.cmd == "fetch":
        st = fetch(model, a.max_bytes, auto=a.auto)
    else:
        st = remove(model)
    print(json.dumps(st), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
