# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where a conversion reads from: a local folder or a Hugging Face repo.

For a repo nothing big is downloaded to plan. The safetensors headers come
over HTTP range requests (huggingface_hub reads only the header bytes), the
small files (config, tokenizer, template, license) land in a meta folder, and
the shards are fetched one at a time while converting. A revision is pinned
to its commit when the source opens, so a push to the repo halfway through a
long conversion can't mix two versions.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

ST_DTYPES = {
    "BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
    "U64": "uint64", "U32": "uint32", "U16": "uint16", "U8": "uint8", "BOOL": "bool",
    # mx.load hands fp8 back as raw bytes, the architectures' sanitize dequantizes it
    "F8_E4M3": "uint8", "F8_E5M2": "uint8",
}

WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".msgpack", ".h5", ".ckpt", ".onnx", ".npz")
SMALL_FILES = (
    "*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja", "*.py", "*.md",
    "LICENSE*", "NOTICE*", "USE_POLICY*", "*.vocab", "*.spm",
)
SKIP_DIRS = ("original/", "onnx/", "gguf/", "coreml/", "metal/", ".cache/")


class SourceError(RuntimeError):
    pass


@dataclass
class TensorRef:
    file: str
    dtype: str  # mlx dtype name
    shape: tuple
    nbytes: int


@dataclass
class Source:
    kind: str  # "local" | "hub"
    ref: str  # folder or repo id
    meta_dir: Path  # config.json and the small files
    config: dict
    tensors: dict[str, TensorRef]
    file_sizes: dict[str, int]
    revision: str | None = None  # hub commit
    local_dir: Path | None = None  # where the shards already are
    license: str | None = None
    notes: list = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(self.file_sizes.values())

    @property
    def files(self) -> list[str]:
        return sorted(self.file_sizes)

    def local_path(self, file: str) -> Path | None:
        if self.local_dir is not None:
            p = self.local_dir / file
            return p if p.exists() else None
        return None

    def fetch(self, file: str, dest: Path, retries: int = 4) -> Path:
        """The shard on disk, downloading it into `dest` when it isn't local."""
        p = self.local_path(file)
        if p is not None:
            return p
        if self.kind != "hub":
            raise SourceError(f"{file} is missing from {self.ref}")
        target = Path(dest) / file
        if target.exists() and target.stat().st_size == self.file_sizes.get(file, -1):
            return target
        from huggingface_hub import hf_hub_download

        err = None
        for i in range(retries):
            try:
                got = hf_hub_download(self.ref, file, revision=self.revision, local_dir=str(dest))
                return Path(got)
            except Exception as e:  # noqa: BLE001 - network, retried below
                err = e
                time.sleep(min(30, 2 ** i))
        raise SourceError(f"couldn't download {file} from {self.ref}: {err}")

    def describe(self) -> dict:
        return {
            "kind": self.kind, "ref": self.ref, "revision": self.revision,
            "local_dir": str(self.local_dir) if self.local_dir else None,
            "files": len(self.file_sizes), "bytes": self.total_bytes,
            "tensors": len(self.tensors), "license": self.license,
            "model_type": self.config.get("model_type"),
        }


def convert_home() -> Path:
    raw = os.environ.get("PAGED_MOE_CONVERT_DIR")
    return Path(raw).expanduser() if raw else Path.home() / ".paged_moe" / "convert"


def looks_like_repo(ref: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][\w.-]*/[\w.-]+", ref)) and not Path(ref).expanduser().exists()


def open_source(ref: str, revision: str | None = None, offline_ok: bool = True) -> Source:
    ref = str(ref).strip()
    if not ref:
        raise SourceError("no source given")
    p = Path(ref).expanduser()
    if p.exists():
        return _open_local(p.resolve())
    if looks_like_repo(ref):
        return _open_hub(ref, revision, offline_ok)
    if ref.startswith(("/", "~", ".")) or ref.count("/") > 1:
        raise SourceError(f"{p} doesn't exist")
    raise SourceError(f"{ref} isn't a folder here or a Hugging Face repo id (org/name)")


# ------------------------------------------------------------ local


def _read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            raise SourceError(f"{path.name} is too short to be safetensors")
        (n,) = struct.unpack("<Q", raw)
        if n > 200 << 20:
            raise SourceError(f"{path.name} has a {n} byte header, not a safetensors file")
        return json.loads(f.read(n))


def _weight_files(d: Path) -> list[str]:
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        try:
            wm = json.loads(idx.read_text()).get("weight_map") or {}
            files = sorted(set(wm.values()))
            if files and all((d / f).exists() for f in files):
                return files
        except (OSError, ValueError):
            pass
    files = sorted(p.name for p in d.glob("model*.safetensors"))
    if not files:
        files = sorted(p.name for p in d.glob("*.safetensors") if not p.name.startswith("consolidated"))
    return files


def _tensors_from_headers(headers: dict[str, dict]) -> dict[str, TensorRef]:
    out: dict[str, TensorRef] = {}
    for file, header in headers.items():
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dt = ST_DTYPES.get(meta["dtype"])
            if dt is None:
                raise SourceError(f"{file}: {name} has dtype {meta['dtype']}, which MLX can't load")
            b, e = meta["data_offsets"]
            if name in out:
                raise SourceError(f"{name} is in both {out[name].file} and {file}")
            out[name] = TensorRef(file, dt, tuple(int(s) for s in meta["shape"]), int(e - b))
    return out


def _license(meta_dir: Path, config: dict) -> str | None:
    readme = meta_dir / "README.md"
    if readme.exists():
        try:
            text = readme.read_text(errors="replace")
            if text.startswith("---"):
                front = text.split("---", 2)[1]
                import yaml

                data = yaml.safe_load(front) or {}
                lic = data.get("license")
                if lic and lic != "other":
                    return str(lic)
                if data.get("license_name"):
                    return str(data["license_name"])
                if lic:
                    return str(lic)
        except Exception:
            pass
    return None


def _open_local(d: Path) -> Source:
    if not (d / "config.json").is_file():
        inside = sorted(c.parent.name for c in d.glob("*/config.json"))
        hint = f" Pick one inside it, e.g. {d / inside[0]}." if inside else ""
        raise SourceError(f"{d} has no config.json.{hint}")
    files = _weight_files(d)
    if not files:
        if any(d.glob("*.gguf")):
            raise SourceError(f"{d} holds GGUF files. GGUF sources aren't supported yet, use the "
                              "original Hugging Face (safetensors) repo for the model.")
        if any(d.glob("*.bin")) or any(d.glob("*.pt*")):
            raise SourceError(f"{d} only has PyTorch .bin weights. Convert it to safetensors first, "
                              "or use a repo that ships safetensors.")
        raise SourceError(f"no .safetensors files in {d}")
    headers = {f: _read_header(d / f) for f in files}
    config = json.loads((d / "config.json").read_text())
    return Source(
        kind="local", ref=str(d), meta_dir=d, config=config,
        tensors=_tensors_from_headers(headers),
        file_sizes={f: (d / f).stat().st_size for f in files},
        local_dir=d, license=_license(d, config),
    )


# ------------------------------------------------------------ hub


def _is_small(name: str) -> bool:
    if name.endswith(WEIGHT_EXT) or any(name.startswith(s) for s in SKIP_DIRS):
        return False
    base = name.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(base, pat) for pat in SMALL_FILES)


def _open_hub(repo: str, revision: str | None, offline_ok: bool) -> Source:
    try:
        from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download
    except ImportError as e:
        raise SourceError("huggingface_hub is needed for repo sources (pip install huggingface_hub)") from e
    api = HfApi()
    try:
        info = api.model_info(repo, revision=revision, files_metadata=True)
    except Exception as e:  # noqa: BLE001
        cached = _cached_snapshot(repo, revision) if offline_ok else None
        if cached is not None:
            src = _open_local(cached)
            src.kind, src.ref = "hub", repo
            src.notes.append("read from the local Hugging Face cache (the Hub didn't answer)")
            return src
        raise SourceError(f"couldn't reach {repo} on the Hub: {e}") from e
    sha = info.sha
    sizes = {s.rfilename: int(s.size or 0) for s in (info.siblings or [])}

    # already fully in the HF cache: read it from there, nothing to download
    hub_shards = {n for n in sizes if n.endswith(".safetensors") and "/" not in n
                  and not n.startswith("consolidated")}
    cached = _cached_snapshot(repo, sha)
    if cached is not None:
        try:
            src = _open_local(cached)
            if hub_shards and {n for n in hub_shards if n.startswith("model")} <= set(src.file_sizes):
                src.kind, src.ref, src.revision = "hub", repo, sha
                src.notes.append("all shards are already in the Hugging Face cache")
                return src
        except SourceError:
            pass

    meta_dir = convert_home() / "hub" / repo.replace("/", "--") / sha[:12]
    headers = _cached_headers(meta_dir, sha)
    fresh = headers is None
    if fresh:
        try:
            meta = get_safetensors_metadata(repo, revision=sha)
        except Exception as e:  # noqa: BLE001
            if any(n.endswith(".gguf") for n in sizes):
                raise SourceError(f"{repo} is a GGUF repo. GGUF sources aren't supported yet, use the "
                                  "original safetensors repo for the model.") from None
            raise SourceError(f"{repo} has no safetensors weights the Hub can describe: {e}") from e
        headers = {}
        for file, fm in meta.files_metadata.items():
            h = {}
            for name, t in fm.tensors.items():
                h[name] = {"dtype": t.dtype, "shape": list(t.shape), "data_offsets": list(t.data_offsets)}
            headers[file] = h
    files = sorted(headers)
    if any("/" in f for f in files):
        raise SourceError(f"{repo} keeps its weights in subfolders, which isn't supported yet")

    meta_dir.mkdir(parents=True, exist_ok=True)
    if fresh:
        _save_headers(meta_dir, sha, headers)
    for name in sorted(sizes):
        if not _is_small(name) or "/" in name:
            continue
        if (meta_dir / name).exists() and (meta_dir / name).stat().st_size == sizes[name]:
            continue
        try:
            hf_hub_download(repo, name, revision=sha, local_dir=str(meta_dir))
        except Exception as e:  # noqa: BLE001
            if name == "config.json":
                raise SourceError(f"couldn't download config.json from {repo}: {e}") from e
    if not (meta_dir / "config.json").exists():
        raise SourceError(f"{repo} has no config.json")
    config = json.loads((meta_dir / "config.json").read_text())
    return Source(
        kind="hub", ref=repo, meta_dir=meta_dir, config=config,
        tensors=_tensors_from_headers(headers),
        file_sizes={f: sizes.get(f) or sum(t["data_offsets"][1] for t in headers[f].values()
                                           if isinstance(t, dict) and "data_offsets" in t)
                    for f in files},
        revision=sha, license=_license(meta_dir, config),
    )


HEADERS = ".pagedmoe-headers.json"


def _cached_headers(meta_dir: Path, sha: str) -> dict | None:
    try:
        d = json.loads((meta_dir / HEADERS).read_text())
    except (OSError, ValueError):
        return None
    return d.get("headers") if d.get("sha") == sha and isinstance(d.get("headers"), dict) else None


def _save_headers(meta_dir: Path, sha: str, headers: dict) -> None:
    p = meta_dir / HEADERS
    try:
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"sha": sha, "headers": headers}))
        os.replace(tmp, p)
    except OSError:
        pass


def _cached_snapshot(repo: str, revision: str | None) -> Path | None:
    try:
        from huggingface_hub import snapshot_download

        p = Path(snapshot_download(repo, revision=revision, local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached
        return None
    if (p / "config.json").exists() and _weight_files(p):
        files = _weight_files(p)
        if all((p / f).exists() for f in files):
            return p
    return None
