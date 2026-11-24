# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run a conversion plan.

One unit at a time (the tensors outside the layers, then each decoder
layer): read it lazily, undo the source quantization, merge the adapter,
quantize per the recipe, measure what quantizing cost, write it at its
planned offset, fsync, mark it done. Memory stays around one big tensor no
matter how big the model is, and a stopped run resumes at the first unfinished
layer.

A repo that streams downloads a layer's shards while the layer before it
converts, and deletes them once no later layer needs them, so the disk only
ever holds the output plus about two layers of source.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import threading
import time
from pathlib import Path

from . import card as C
from . import plan as P
from . import recipe as R
from .merge import merge
from .writer import ShardWriter, verify as verify_headers

STATE = ".pagedmoe-convert.json"
REPORT = "pagedmoe_conversion.json"
GiB = 1 << 30


class ConvertError(RuntimeError):
    pass


class _Stop(Exception):
    pass


# ------------------------------------------------------------ events


class _Events:
    def __init__(self, on_event):
        self._cb = on_event
        self._lock = threading.Lock()

    def __call__(self, kind: str, **data) -> None:
        if self._cb is None:
            return
        with self._lock:
            try:
                self._cb({"kind": kind, **data})
            except Exception:  # noqa: BLE001 - a broken listener never stops a conversion
                pass


# ------------------------------------------------------------ downloads


class _Fetcher:
    """A repo's shards, downloaded in the order the units need them."""

    def __init__(self, src, staging: Path, needs: list, order: list, emit, keep: bool, ahead: int = 1):
        self.src, self.staging, self.needs, self.order = src, staging, needs, order
        self.emit, self.keep, self.ahead = emit, keep, ahead
        self.ready: dict[str, Path] = {}
        self.error: Exception | None = None
        self.cursor = 0  # position in order of the unit converting now
        self.cv = threading.Condition()
        self.current: tuple | None = None  # (file, size, started)
        self.done_bytes = 0
        self.total = sum(src.file_sizes[f] for f in {f for i in order for f in needs[i]})
        self._quit = False
        self._t = threading.Thread(target=self._work, name="convert-fetch", daemon=True)
        self._ticker = threading.Thread(target=self._tick, name="convert-fetch-rate", daemon=True)
        self._t.start()
        self._ticker.start()

    def _work(self) -> None:
        try:
            for pos, i in enumerate(self.order):
                with self.cv:
                    while pos > self.cursor + self.ahead and not self._quit:
                        self.cv.wait(0.5)
                    if self._quit:
                        return
                for f in self.needs[i]:
                    if f in self.ready:
                        continue
                    size = self.src.file_sizes.get(f, 0)
                    self.current = (f, size, time.time())
                    self.emit("download", file=f, bytes=size, state="start")
                    t0 = time.time()
                    path = self.src.fetch(f, self.staging)
                    secs = max(1e-6, time.time() - t0)
                    self.current = None
                    self.done_bytes += size
                    self.emit("download", file=f, bytes=size, state="done", seconds=round(secs, 2),
                              gb_s=round(size / secs / 1e9, 3), done_bytes=self.done_bytes, total_bytes=self.total)
                    with self.cv:
                        self.ready[f] = path
                        self.cv.notify_all()
        except Exception as e:  # noqa: BLE001 - surfaced to the converting thread
            with self.cv:
                self.error = e
                self.cv.notify_all()

    def _tick(self) -> None:
        part = self.staging / ".cache" / "huggingface" / "download"
        while not self._quit:
            time.sleep(1.0)
            cur = self.current
            if cur is None:
                continue
            got = 0
            try:
                for p in part.rglob("*.incomplete"):
                    got += p.stat().st_size
            except OSError:
                continue
            secs = max(1e-6, time.time() - cur[2])
            self.emit("download", file=cur[0], bytes=cur[1], state="progress", got=got,
                      gb_s=round(got / secs / 1e9, 3), done_bytes=self.done_bytes + got, total_bytes=self.total)

    def wait(self, pos: int, should_stop) -> list:
        i = self.order[pos]
        with self.cv:
            self.cursor = pos
            self.cv.notify_all()
            while True:
                if self.error is not None:
                    raise self.error
                if all(f in self.ready for f in self.needs[i]):
                    return [self.ready[f] for f in self.needs[i]]
                if should_stop():
                    raise _Stop()
                self.cv.wait(0.5)

    def release(self, pos: int) -> None:
        """Delete the shards no later unit reads."""
        if self.keep:
            return
        later = {f for i in self.order[pos + 1:] for f in self.needs[i]}
        for f in self.needs[self.order[pos]]:
            if f in later or f not in self.ready:
                continue
            p = self.ready.pop(f)
            if self.src.local_path(f) is None:
                try:
                    p.unlink()
                except OSError:
                    pass

    def close(self) -> None:
        self._quit = True
        with self.cv:
            self.cv.notify_all()


# ------------------------------------------------------------ reading


def _lazy(paths) -> dict:
    import mlx.core as mx

    out: dict = {}
    for p in paths:
        out.update(mx.load(str(p)))
    return out


class _Reader:
    """The loaded view (after sanitize) of each unit, lazily."""

    def __init__(self, plan, staging: Path, todo: list, emit, keep: bool):
        self.plan, self.emit = plan, emit
        src = plan.source
        self.fetcher = None
        self.whole = None
        if plan.windowed:
            needs = [u.files for u in plan.units]
            self.fetcher = _Fetcher(src, staging, needs, todo, emit, keep)
        elif src.local_dir is None:
            # a repo that can't stream: everything first, then convert
            self.fetcher = _Fetcher(src, staging, [src.files], [0], emit, keep, ahead=0)
        self.todo = todo
        self.paths: list = []

    def _whole(self, should_stop):
        if self.whole is None:
            src = self.plan.source
            if src.local_dir is not None:
                self.paths = [src.local_dir / f for f in src.files]
            else:
                self.paths = self.fetcher.wait(0, should_stop)
            _watch(self.paths)
            self.whole = P._sanitize(self.plan.model, _lazy(self.paths))
        return self.whole

    def unit(self, pos: int, should_stop) -> dict:
        i = self.todo[pos]
        if not self.plan.windowed:
            return self._whole(should_stop)
        paths = self.fetcher.wait(pos, should_stop)
        _watch(paths)
        self.paths = paths
        u = self.plan.units[i]
        lazy = _lazy(paths)
        return P._sanitize(self.plan.model, {n: lazy[n] for n in u.src})

    def release(self, pos: int) -> None:
        if self.plan.windowed and self.fetcher is not None:
            self.fetcher.release(pos)

    def close(self) -> None:
        if self.fetcher is not None:
            self.fetcher.close()
        self.whole = None


def _watch(paths) -> None:
    try:
        from .. import pagecache

        pagecache.watch([str(p) for p in paths])
    except Exception:  # noqa: BLE001 - only a cache hint
        pass


# ------------------------------------------------------------ math


def _pop_float(plan, loaded: dict, name: str):
    """The float view of one tensor, taken out of `loaded` so it frees once used."""
    import mlx.core as mx

    mod, _, leaf = name.rpartition(".")
    q = plan.src_quant.get(mod)
    if q is not None and leaf == "weight":
        w = loaded.pop(name)
        s = loaded.pop(f"{mod}.scales")
        b = loaded.pop(f"{mod}.biases", None)
        return mx.dequantize(w, s, b, group_size=q["group_size"], bits=q["bits"], mode=q.get("mode", "affine"))
    return loaded.pop(name)


def _error(w, parts, p: dict) -> tuple[float, float]:
    """Squared error of the quantized copy and squared size of w, in float32."""
    import mlx.core as mx

    rows = w.shape[0]
    per_row = max(1, math.prod(w.shape[1:]) * 4)
    step = max(1, min(rows, (256 << 20) // per_row))
    num = den = 0.0
    for s in range(0, rows, step):
        e = min(rows, s + step)
        b = parts[2][s:e] if len(parts) > 2 else None
        d = mx.dequantize(parts[0][s:e], parts[1][s:e], b, group_size=p["group_size"], bits=p["bits"],
                          mode=p["mode"]).astype(mx.float32)
        x = w[s:e].astype(mx.float32)
        n2, d2 = mx.sum(mx.square(d - x)), mx.sum(mx.square(x))
        mx.eval(n2, d2)
        num += n2.item()
        den += d2.item()
    return num, den


def _snr_db(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    if num <= 0:
        return 200.0
    return round(10 * math.log10(den / num), 2)


# ------------------------------------------------------------ out dir


def _fingerprint(plan, shard_bytes: int) -> str:
    src = plan.source
    files = {}
    for f, size in sorted(src.file_sizes.items()):
        p = src.local_path(f)
        files[f] = [size, p.stat().st_mtime_ns if p is not None and src.kind == "local" else 0]
    ad = None
    if plan.adapter:
        ap = Path(plan.adapter["path"]) / "adapters.safetensors"
        ad = [str(ap), ap.stat().st_size, ap.stat().st_mtime_ns]
    body = {"v": 1, "ref": src.ref, "revision": src.revision, "files": files,
            "recipe": plan.recipe.to_dict(), "dtype": plan.float_dtype, "adapter": ad,
            "layout": [[t.name, list(t.shape), t.dtype] for t in plan.layout], "shard_bytes": shard_bytes}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _save_state(out: Path, state: dict) -> None:
    tmp = out / (STATE + ".tmp")
    tmp.write_text(json.dumps(state))
    os.replace(tmp, out / STATE)


def _ours(out: Path) -> list:
    keep = []
    for p in out.iterdir():
        n = p.name
        if (n.startswith("model") and n.endswith(".safetensors")) or n in (
                "model.safetensors.index.json", STATE, STATE + ".tmp", REPORT):
            keep.append(p)
    return keep


def _prepare(out: Path, src_dir, fp: str, resume: bool, overwrite: bool, emit) -> dict | None:
    """The saved state to resume from, or None for a fresh start."""
    if src_dir is not None and out.resolve() == Path(src_dir).resolve():
        raise ConvertError("the output folder is the source folder, pick another one")
    if not out.exists():
        out.mkdir(parents=True)
        return None
    st = out / STATE
    if st.exists():
        try:
            state = json.loads(st.read_text())
        except (OSError, ValueError):
            state = None
        if state and state.get("key") == fp and resume:
            return state
        if not overwrite:
            if state and state.get("key") != fp:
                raise ConvertError(f"{out} has a stopped conversion with different settings. Use the "
                                   "same settings to resume it, or overwrite to start over")
            raise ConvertError(f"{out} has a stopped conversion. Resume it, or overwrite to start over")
    elif any(out.iterdir()):
        if (out / REPORT).exists() and not overwrite:
            raise ConvertError(f"{out} already holds a finished conversion. Overwrite it, or pick another folder")
        if not overwrite:
            raise ConvertError(f"{out} isn't empty. Pick an empty or new folder, or overwrite")
    for p in _ours(out):
        p.unlink()
    emit("phase", phase="fresh", why="starting over")
    return None


# ------------------------------------------------------------ main


def convert(source, out, recipe=None, *, plan=None, adapter: str | None = None, stream: bool | None = None,
            revision: str | None = None, shard_gb: float = 5.0, ctx: int = 32768, resume: bool = True,
            overwrite: bool = False, measure: bool = True, keep_source: bool = False,
            verify: bool = True, ignore_disk: bool = False, card: bool = True,
            on_event=None, should_stop=None) -> dict:
    """Convert a Hugging Face checkpoint (folder or repo) to MLX per a recipe.

    Returns the report (also written to <out>/pagedmoe_conversion.json), with
    status "done" or "stopped". `on_event` gets dicts with a "kind" key as it
    goes; `should_stop` is polled between tensors."""
    emit = _Events(on_event)
    should_stop = should_stop or (lambda: False)
    out = Path(out).expanduser()
    try:
        return _convert(source, out, recipe, plan, adapter, stream, revision, shard_gb, ctx, resume,
                        overwrite, measure, keep_source, verify, ignore_disk, card, emit, should_stop)
    except Exception as e:
        emit("error", message=str(e), type=type(e).__name__)
        raise


def _convert(source, out, recipe, plan, adapter, stream, revision, shard_gb, ctx, resume, overwrite,
             measure, keep_source, verify, ignore_disk, card, emit, should_stop) -> dict:
    import mlx.core as mx

    t_start = time.time()
    emit("phase", phase="plan")
    if plan is None:
        plan = P.make(source, recipe, out_dir=str(out), adapter=adapter, stream=stream, revision=revision,
                      ctx=ctx, shard_gb=shard_gb)
    summary = plan.summary()
    emit("plan", summary=summary)
    src = plan.source
    shard_bytes = int(max(0.1, float(shard_gb)) * 1e9)
    fp = _fingerprint(plan, shard_bytes)
    state = _prepare(out, src.local_dir, fp, resume, overwrite, emit)
    disk = summary["disk"]
    if state is None and disk.get("ok") is False and not ignore_disk:
        raise ConvertError(disk["why"])

    writer = ShardWriter(out, plan.layout, shard_bytes)
    if state is not None and not writer.matches_disk():
        emit("phase", phase="fresh", why="the stopped run's shards don't match, starting over")
        state = None
    if state is None:
        state = {"key": fp, "done": [], "errors": {}, "worst": [], "started": time.time(), "seconds": 0.0,
                 "units": len(plan.units), "out_bytes": summary["out_bytes"],
                 "args": {"source": src.ref, "revision": src.revision, "recipe": plan.recipe.to_dict(),
                          "adapter": plan.adapter["path"] if plan.adapter else None,
                          "shard_gb": shard_gb, "stream": plan.windowed}}
        writer.create()
        _save_state(out, state)
    else:
        emit("phase", phase="resume", done=len(state["done"]), units=len(plan.units), done_units=list(state["done"]))
    _watch(writer.paths)

    done = set(state["done"])
    todo = [i for i in range(len(plan.units)) if i not in done]
    staging = Path(out) / ".source"
    reader = _Reader(plan, staging, todo, emit, keep_source)
    total_out = sum(u.out_bytes for u in plan.units)
    written_before = sum(plan.units[i].out_bytes for i in done)
    errors: dict = state.setdefault("errors", {})  # component -> [num, den]
    worst: list = state.setdefault("worst", [])
    run_t0 = time.time()
    written = 0
    last_tick = 0.0
    adapter_mods = plan.adapter["modules"] if plan.adapter else {}
    stopped = False

    emit("phase", phase="convert", units=len(plan.units), todo=len(todo), out_bytes=total_out)
    try:
        for pos, i in enumerate(todo):
            if should_stop():
                raise _Stop()
            u = plan.units[i]
            t_unit = time.time()
            emit("unit_start", unit=i, key=u.key, units=len(plan.units), out_bytes=u.out_bytes)
            loaded = reader.unit(pos, should_stop)
            unit_err: dict = {}
            for name in u.floats:
                if should_stop():
                    raise _Stop()
                mod, _, leaf = name.rpartition(".")
                m = plan.modules.get(mod)
                if m is not None and leaf == "weight":
                    w = _pop_float(plan, loaded, name)
                    if m.adapter:
                        w = merge(m.kind, w, adapter_mods[mod], plan.adapter)
                    w = w.astype(getattr(mx, m.dtype))
                    if m.params is not None:
                        p = m.params
                        parts = mx.quantize(w, group_size=p["group_size"], bits=p["bits"], mode=p["mode"])
                        mx.eval(w, *parts) if measure else mx.eval(*parts)
                        if measure:
                            num, den = _error(w, parts, p)
                            acc = errors.setdefault(m.component, [0.0, 0.0])
                            acc[0] += num
                            acc[1] += den
                            ue = unit_err.setdefault(m.component, [0.0, 0.0])
                            ue[0] += num
                            ue[1] += den
                            rel = math.sqrt(num / den) if den > 0 else 0.0
                            worst.append([rel, mod, R.describe_params(p)])
                        del w
                        written += writer.put(f"{mod}.weight", parts[0])
                        written += writer.put(f"{mod}.scales", parts[1])
                        if len(parts) > 2:
                            written += writer.put(f"{mod}.biases", parts[2])
                        del parts
                    else:
                        mx.eval(w)
                        written += writer.put(name, w)
                        del w
                else:
                    a = _pop_float(plan, loaded, name)
                    if plan.cast_ok.get(name) and str(a.dtype).replace("mlx.core.", "") != plan.float_dtype:
                        a = a.astype(getattr(mx, plan.float_dtype))
                    mx.eval(a)
                    written += writer.put(name, a)
                    del a
                now = time.time()
                if now - last_tick > 0.5:
                    last_tick = now
                    _progress(emit, written_before + written, total_out, written, run_t0, i, len(plan.units),
                              name, mx)
            writer.sync()
            if plan.windowed:
                loaded.clear()
            mx.clear_cache()
            worst.sort(key=lambda r: -r[0])
            del worst[20:]
            state["done"].append(i)
            state["seconds"] = state.get("seconds", 0.0) + (time.time() - t_unit)
            _save_state(out, state)
            reader.release(pos)
            emit("unit_done", unit=i, key=u.key, units=len(plan.units), done=len(state["done"]),
                 out_bytes=u.out_bytes, seconds=round(time.time() - t_unit, 2),
                 snr_db={c: _snr_db(*v) for c, v in unit_err.items()})
            _progress(emit, written_before + written, total_out, written, run_t0, i, len(plan.units), None, mx)
    except _Stop:
        stopped = True
    finally:
        reader.close()
        writer.close()

    if stopped:
        _save_state(out, state)
        emit("stopped", done=len(state["done"]), units=len(plan.units), out=str(out))
        return {"status": "stopped", "out": str(out), "done": len(state["done"]), "units": len(plan.units)}

    emit("phase", phase="finish")
    writer.write_index()
    _write_config(plan, out)
    copied = _copy_small(plan, out)
    fidelity = {
        "components": {c: {"snr_db": _snr_db(*v), "rel_err": round(math.sqrt(v[0] / v[1]), 5) if v[1] > 0 else None}
                       for c, v in errors.items()},
        "worst": [{"module": m, "rel_err": round(r, 5), "setting": s} for r, m, s in worst[:10]],
    }
    from .. import __version__

    report = {
        "status": "done",
        "pagedmoe": __version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "out": str(out),
        "source": src.describe(),
        "model_type": plan.model_type,
        "recipe": plan.recipe.to_dict(),
        "fit_choice": summary.get("fit_choice"),
        "dtype": plan.float_dtype,
        "quantization": plan.quant_config(),
        "components": summary["components"],
        "out_bytes": summary["out_bytes"],
        "float_bytes": summary["float_bytes"],
        "bits_per_weight": summary["bits_per_weight"],
        "adapter": summary["adapter"],
        "weight_error": fidelity if measure else None,
        "seconds": round(state.get("seconds", 0.0), 1),
        "wall_seconds": round(time.time() - t_start, 1),
        "files": copied,
    }
    if verify:
        emit("phase", phase="verify")
        report["verify"] = _verify(plan, out, writer)
        if report["verify"]["problems"]:
            report["status"] = "failed"
    (out / REPORT).write_text(json.dumps(report, indent=2) + "\n")
    if card:
        C.write(out, report, plan)
    for p in (out / STATE, out / (STATE + ".tmp")):
        if p.exists():
            p.unlink()
    if staging.exists() and not keep_source:
        shutil.rmtree(staging, ignore_errors=True)
    emit("done", report=report)
    if report["status"] == "failed":
        raise ConvertError("the output didn't verify: " + "; ".join(report["verify"]["problems"][:3]))
    return report


def _progress(emit, done_bytes, total, written, t0, unit, units, tensor, mx) -> None:
    secs = max(1e-6, time.time() - t0)
    rate = written / secs
    left = max(0, total - done_bytes)
    emit("progress", done_bytes=done_bytes, total_bytes=total, unit=unit, units=units, tensor=tensor,
         gb_s=round(rate / 1e9, 3), eta_s=round(left / rate, 1) if rate > 0 else None,
         active_gb=round(mx.get_active_memory() / 1e9, 2), peak_gb=round(mx.get_peak_memory() / 1e9, 2))


# ------------------------------------------------------------ finish


def _write_config(plan, out: Path) -> None:
    cfg = dict(plan.config)
    cfg.pop("_name_or_path", None)
    cfg.pop("quantization", None)
    cfg.pop("quantization_config", None)
    if isinstance(cfg.get("text_config"), dict):
        tc = dict(cfg["text_config"])
        tc.pop("quantization_config", None)
        tc.pop("quantization", None)
        cfg["text_config"] = tc
    q = plan.quant_config()
    if q is not None:
        cfg["quantization"] = q
        cfg["quantization_config"] = q
    if "torch_dtype" in cfg:
        cfg["torch_dtype"] = plan.float_dtype
    cfg = dict(sorted(cfg.items()))
    (out / "config.json").write_text(json.dumps(cfg, indent=4) + "\n")


_SKIP = {"config.json", "model.safetensors.index.json", "README.md", REPORT, STATE}


def _copy_small(plan, out: Path) -> list:
    from .source import SKIP_DIRS, WEIGHT_EXT, _is_small

    meta = Path(plan.source.meta_dir)
    copied = []
    for p in sorted(meta.iterdir()):
        n = p.name
        if not p.is_file() or n in _SKIP or n.endswith(WEIGHT_EXT) or n.startswith("."):
            continue
        if not _is_small(n) or any(n.startswith(s.rstrip("/")) for s in SKIP_DIRS):
            continue
        if n.endswith(".md") and not n.upper().startswith(("LICENSE", "NOTICE", "USE_POLICY")):
            continue
        if p.stat().st_size > 64 << 20:
            continue
        shutil.copy2(p, out / n)
        copied.append(n)
    return copied


def _verify(plan, out: Path, writer: ShardWriter) -> dict:
    """Headers against the plan, then a strict structural load with mlx-lm."""
    import mlx.core as mx

    header_problems = verify_headers(out, writer)
    problems = list(header_problems)
    loads = None
    if not problems:
        dev = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            from mlx_lm.utils import load_model

            from .. import local_arch

            try:
                local_arch.register(out)
            except Exception:  # noqa: BLE001 - only needed for bundled ports
                pass
            load_model(out, lazy=True, strict=True)
            loads = True
        except Exception as e:  # noqa: BLE001
            loads = False
            problems.append(f"mlx-lm can't load it: {type(e).__name__}: {e}")
        finally:
            mx.set_default_device(dev)
    return {"problems": problems, "headers_ok": not header_problems, "mlx_lm_loads": loads}


def conversions(dirs, depth: int = 2) -> list:
    """Finished and stopped conversions in the given folders, newest first."""
    out, seen = [], set()

    def visit(p: Path, d: int):
        real = os.path.realpath(p)
        if real in seen:
            return
        seen.add(real)
        rep, st = p / REPORT, p / STATE
        try:
            if st.exists():
                s = json.loads(st.read_text())
                a = s.get("args") or {}
                out.append({"path": str(p), "name": p.name, "status": "stopped", "done": len(s.get("done") or []),
                            "units": s.get("units"), "out_bytes": s.get("out_bytes"), "source": a.get("source"),
                            "recipe": a.get("recipe"), "adapter": a.get("adapter"), "shard_gb": a.get("shard_gb"),
                            "time": st.stat().st_mtime})
                return
            if rep.exists():
                r = json.loads(rep.read_text())
                out.append({"path": str(p), "name": p.name, "status": r.get("status", "done"),
                            "source": (r.get("source") or {}).get("ref"), "source_kind": (r.get("source") or {}).get("kind"),
                            "model_type": r.get("model_type"), "recipe": r.get("recipe"), "adapter": r.get("adapter"),
                            "out_bytes": r.get("out_bytes"), "bits_per_weight": r.get("bits_per_weight"),
                            "components": r.get("components"), "weight_error": r.get("weight_error"),
                            "check": r.get("check"), "seconds": r.get("wall_seconds"), "time": rep.stat().st_mtime})
                return
        except (OSError, ValueError):
            return
        if d <= 0:
            return
        try:
            kids = sorted(k for k in p.iterdir() if k.is_dir() and not k.name.startswith("."))
        except OSError:
            return
        for k in kids:
            visit(k, d - 1)

    for d in dirs or []:
        p = Path(d).expanduser()
        if p.is_dir():
            visit(p, depth)
    out.sort(key=lambda r: -(r.get("time") or 0))
    return out


__all__ = ["convert", "conversions", "ConvertError", "STATE", "REPORT"]
