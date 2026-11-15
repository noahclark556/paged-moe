# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One bench run per mode on this Mac, with this model.

  1. read the checkpoint (autotune.inspect) and pick the mode's settings
  2. start pagedmoe.server in a clean env with fresh state (no sidecar
     training, no prefix cache from earlier runs, no paged-moe-config.yaml)
  3. replay data/session.json over the OpenAI API, streamed, timing each
     request on the client and reading the engine's per-request trace
  4. stop the server and run the loss probe (probe.py) against the exact model
  5. validate, summarize, write <out>/<model>-<mode>[-quick].json

The result file is rewritten after every step, so a crash keeps what ran.
"""

from __future__ import annotations

import http.client
import json
import os
import platform
import plistlib
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .. import autotune

DATA = Path(__file__).resolve().parent / "data"
SCHEMA = 1
DEFAULT_SPEED = 0.25
DEFAULT_CTX = 32768

# Every expert the router picks. Cache, prefetch, prefix cache and context
# quantization stay as the engine ships them. Self-draft is off: fp16 context
# plus self-draft hung the GPU on Flash once, and a killed hung engine keeps
# its GPU memory until a restart.
EXACT_ENV = {
    "PAGED_MOE_PRUNE": "0",
    "PAGED_MOE_WAIT_ABOVE": "0",
    "PAGED_MOE_ROUTE_TOP_K": "0",
    "PAGED_MOE_ROUTE_TOP_P": "0",
    "PAGED_MOE_FLOW": "0",
    "PAGED_MOE_SIDECAR_HEAD_PRUNE": "0",
    "PAGED_MOE_SELF_DRAFT": "0",
    "PAGED_MOE_STITCH": "1",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[bench] {msg}", flush=True)


def sh(cmd: list[str], **kw) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False, **kw).stdout.strip()
    except OSError:
        return ""


# --------------------------------------------------------------- machine


def disk_info(path: Path) -> dict:
    """Which volume the model is on and whether it's the internal SSD."""
    out = {"mount": None, "internal": None, "protocol": None}
    lines = sh(["df", "-P", str(path)]).splitlines()
    if len(lines) >= 2:
        out["mount"] = lines[-1].split()[-1]
    if out["mount"]:
        raw = subprocess.run(["diskutil", "info", "-plist", out["mount"]], capture_output=True, check=False).stdout
        try:
            info = plistlib.loads(raw) if raw else {}
        except Exception:
            info = {}
        out["internal"] = info.get("Internal")
        out["protocol"] = info.get("BusProtocol")
    return out


def machine_info(model_dir: Path) -> dict:
    mem = int(sh(["sysctl", "-n", "hw.memsize"]) or 0)
    return {
        "chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "model": sh(["sysctl", "-n", "hw.model"]),
        "ram_gb": round(mem / (1 << 30)),
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "model_disk": disk_info(model_dir),
    }


def versions() -> dict:
    from .. import __version__

    out = {"paged-moe": __version__}
    try:
        from importlib.metadata import version

        for name in ("mlx", "mlx-lm", "transformers"):
            try:
                out[name] = version(name)
            except Exception:
                out[name] = None
    except ImportError:
        pass
    return out


def pressure_level() -> int:
    try:
        return int(sh(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]) or 0)
    except ValueError:
        return 0


def gpu_held_gb() -> float | None:
    out = sh(["/usr/sbin/ioreg", "-r", "-c", "IOAccelerator", "-d", "1", "-w0"])
    vals = [int(v) for v in re.findall(r'"In use system memory"=(\d+)', out)]
    return max(vals) / 1e9 if vals else None


def preflight(port: int, machine: dict, allow_busy: bool) -> list[str]:
    """Refuse what would make the numbers wrong or freeze the Mac; warn on the rest."""
    warnings = []
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("the bench needs an Apple silicon Mac")
    others = sh(["pgrep", "-fl", "pagedmoe.server|mlx_lm.server|paged-moe-server|paged-moe serve"])
    if others and not allow_busy:
        raise SystemExit(
            "another model server is running. Stop it first (two models on one GPU can freeze the Mac):\n" + others
        )
    held = gpu_held_gb()
    if held is not None and held > max(8.0, machine["ram_gb"] * 0.15) and not allow_busy:
        raise SystemExit(
            f"macOS reports {held:.1f} GB of GPU memory in use with no server running (another GPU app, "
            "or memory kept after a crashed process). Quit GPU-heavy apps or restart the Mac first."
        )
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(f"port {port} is in use, pick another with --port")
    if pressure_level() >= 2:
        warnings.append("macOS reports memory pressure; close other apps for clean numbers")
    disk = machine["model_disk"]
    if disk.get("internal") is False:
        warnings.append(
            f"model is on an external drive ({disk.get('protocol') or 'unknown'}); "
            "streamed decode is usually disk bound, so expect lower numbers than the internal SSD"
        )
    return warnings


# ----------------------------------------------------------------- model


def resolve_model(raw: str) -> Path:
    p = Path(os.path.expanduser(raw))
    if p.is_dir():
        return p.resolve()
    if re.fullmatch(r"[\w.-]+/[\w.-]+", raw):
        try:
            from huggingface_hub import snapshot_download

            return Path(snapshot_download(raw, local_files_only=True))
        except Exception:
            pass
        raise SystemExit(
            f"{raw} isn't downloaded yet. Get it first, then run the bench on the folder:\n"
            f"  hf download {raw} --local-dir ~/mlx-models/{raw.split('/')[-1]}"
        )
    raise SystemExit(f"no model folder at {p}")


def checkpoint_complete(d: Path) -> bool:
    if not (d / "config.json").is_file():
        return False
    idx = d / "model.safetensors.index.json"
    if idx.is_file():
        shards = set(json.loads(idx.read_text()).get("weight_map", {}).values())
        return bool(shards) and all((d / s).is_file() for s in shards)
    return any(d.glob("*.safetensors"))


def model_card(d: Path, ck: autotune.Checkpoint) -> dict:
    cfg = json.loads((d / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    return {
        "name": d.name,
        "path": str(d),
        "model_type": ck.model_type or cfg.get("model_type"),
        "total_params_b": ck.total_b,
        "active_params_b": ck.active_b,
        "moe": ck.moe,
        "bits": ck.bits,
        "disk_gb": round(ck.disk_gb, 1),
        "layers": text.get("num_hidden_layers"),
        "experts": text.get("num_experts") or text.get("n_routed_experts") or text.get("num_local_experts"),
        "experts_per_token": text.get("num_experts_per_tok"),
        "fits_resident": ck.fits_resident,
    }


def sampling(d: Path) -> dict:
    """The checkpoint's own sampling defaults, with a fixed seed."""
    gen = {}
    try:
        gen = json.loads((d / "generation_config.json").read_text())
    except (OSError, ValueError):
        pass
    def pick(key, default, cast):
        try:
            return cast(gen[key]) if gen.get(key) is not None else default
        except (TypeError, ValueError):
            return default

    return {
        "temperature": pick("temperature", 0.6, float),
        "top_p": pick("top_p", 0.95, float),
        "top_k": pick("top_k", 20, int),
        "min_p": pick("min_p", 0.0, float),
        "repetition_penalty": pick("repetition_penalty", 1.0, float),
        "seed": 0,
    }


def template_kwargs(d: Path, think: bool) -> dict:
    """enable_thinking only when the chat template reads it."""
    text = ""
    for name in ("chat_template.jinja", "tokenizer_config.json", "chat_template.json"):
        try:
            text += (d / name).read_text(errors="replace")
        except OSError:
            pass
    return {"enable_thinking": think} if "enable_thinking" in text else {}


# --------------------------------------------------------------- settings


def mode_env(ck: autotune.Checkpoint, mode: str, speed: float | str) -> dict:
    if mode == "exact":
        return dict(EXACT_ENV)
    if speed == "auto":
        env = autotune.speed_env(ck, "auto")
    else:
        # The fixed scale, not a per-model kept-mass curve, so the same
        # --speed means the same PRUNE / WAIT_ABOVE on every Mac.
        prune, wait, one_pass = autotune._fixed_profile(float(speed))
        env = {
            "PAGED_MOE_PRUNE": f"{prune:g}",
            "PAGED_MOE_WAIT_ABOVE": f"{wait:g}",
            "PAGED_MOE_FLOW": "1" if one_pass else "0",
        }
    env["PAGED_MOE_SIDECAR_HEAD_PRUNE"] = "1" if float(env["PAGED_MOE_PRUNE"]) > 0 else "0"
    env["PAGED_MOE_STITCH"] = "1"
    return env


def engine_env(settings: dict, state: Path, trace: Path, ctx: int, extra: dict) -> dict:
    """The shell's PAGED_MOE_* and paged-moe-config.yaml never reach a run."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PAGED_MOE_")}
    env.update(settings)
    env.update(
        {
            "PAGED_MOE_HOOK": "0",
            "PAGED_MOE_TRACE": str(trace),
            "PAGED_MOE_PREFIX_DISK_DIR": str(state / "prefix_cache"),
            "PAGED_MOE_SIDECAR_DIR": str(state / "sidecar"),
            "PAGED_MOE_RESERVE_CTX": str(ctx),
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    env.update(extra)
    return env


# ---------------------------------------------------------------- server


class DoesNotFit(RuntimeError):
    """The engine refused the load: the model can't run in this Mac's memory."""


class Server:
    def __init__(self, model: Path, port: int, env: dict, log_path: Path):
        self.port = port
        self.log_path = log_path
        self.hung: str | None = None
        self._log = open(log_path, "w")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "pagedmoe.server", "--model", str(model), "--host", "127.0.0.1", "--port", str(port)],
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def wait_ready(self, timeout: float) -> float:
        """Seconds until the model is loaded. /health answers before the load
        finishes, so this also waits for the engine's load line."""
        t0 = time.perf_counter()
        healthy = False
        while time.perf_counter() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited during load (code {self.proc.returncode}), see {self.log_path}")
            if not healthy:
                try:
                    c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
                    c.request("GET", "/health")
                    healthy = c.getresponse().status == 200
                except OSError:
                    pass
            if healthy and self.load_info():
                return time.perf_counter() - t0
            text = self.log_path.read_text(errors="replace")
            oom = re.search(r"EnvelopeOOM: (.+)", text)
            if oom:
                raise DoesNotFit(oom.group(1).strip())
            if "Traceback" in text and ("in _load" in text or "in load" in text):
                raise RuntimeError(f"model load failed, see {self.log_path}")
            time.sleep(0.5)
        raise RuntimeError(f"model not loaded after {timeout:.0f}s, see {self.log_path}")

    def load_info(self) -> dict:
        for line in self.log_path.read_text(errors="replace").splitlines():
            if line.startswith("[paged-moe] {"):
                try:
                    return json.loads(line[len("[paged-moe] ") :])
                except json.JSONDecodeError:
                    pass
        return {}

    def watch(self, stall_s: float):
        """Kill the server if its log stops growing for stall_s while a request
        is out (prefill logs about once a second, decode logs ticks). Returns a
        stop callable; self.hung is set if it fired."""
        done = threading.Event()

        def run():
            last_size, last_change = -1, time.monotonic()
            while not done.wait(5.0):
                size = self.log_path.stat().st_size if self.log_path.exists() else 0
                if size != last_size:
                    last_size, last_change = size, time.monotonic()
                    continue
                if time.monotonic() - last_change < stall_s or self.proc.poll() is not None:
                    continue
                sample = self.log_path.with_name("hang_sample.txt")
                try:
                    os.kill(self.proc.pid, signal.SIGUSR1)
                    subprocess.run(
                        ["/usr/bin/sample", str(self.proc.pid), "3", "-file", str(sample)],
                        capture_output=True,
                        timeout=60,
                    )
                except Exception:
                    pass
                time.sleep(2)
                self.hung = f"no engine progress for {stall_s:.0f}s (stacks in {self.log_path})"
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return

        threading.Thread(target=run, daemon=True).start()
        return done.set

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=30)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait(timeout=10)
        self._log.close()
        for _ in range(30):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", self.port)) != 0:
                    break
            time.sleep(1)
        # Let Metal hand the memory back before the next load.
        time.sleep(5)


# --------------------------------------------------------------- scenario


def request_body(run: dict, front: dict, messages: list, max_tokens: int) -> dict:
    s = run["sample"]
    return {
        "model": "default_model",
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": run["chat_template_kwargs"],
        "max_tokens": max_tokens,
        "max_completion_tokens": max_tokens,
        "temperature": s["temperature"],
        "top_p": s["top_p"],
        "top_k": s["top_k"],
        "min_p": s["min_p"],
        "repetition_penalty": s["repetition_penalty"],
        "seed": s["seed"],
        "tools": front["tools"],
        "tool_choice": "auto",
        # System prompt + workspace note: the front matter an agent pins.
        "stable_prefix_messages": 2,
    }


def stream_chat(port: int, body: dict, timeout: float) -> dict:
    """POST and read the SSE stream. Client-side timings plus what came back."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.read()[:400]!r}")
    first = None
    text, reasoning, tool_names, tool_args = [], [], [], []
    usage, finish = {}, None
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            ev = json.loads(data)
        except json.JSONDecodeError:
            continue
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            think = d.get("reasoning") or d.get("reasoning_content")
            if d.get("content") or think or d.get("tool_calls"):
                first = first or time.perf_counter()
            if d.get("content"):
                text.append(d["content"])
            if think:
                reasoning.append(think)
            for tc in d.get("tool_calls") or []:
                fn = tc.get("function") or {}
                if fn.get("name"):
                    tool_names.append(fn["name"])
                if fn.get("arguments"):
                    tool_args.append(fn["arguments"])
            finish = ch.get("finish_reason") or finish
    end = time.perf_counter()
    return {
        "wall_s": round(end - t0, 3),
        "ttft_s": round((first or end) - t0, 3),
        "finish": finish,
        "usage": usage,
        "text": "".join(text),
        "reasoning": "".join(reasoning),
        "tool_calls": tool_names,
    }


def degenerate(text: str) -> bool:
    """A 6-word run repeated 5+ times reads as a loop, not an answer."""
    words = re.findall(r"\S+", text)
    grams: dict[tuple, int] = {}
    for i in range(len(words) - 5):
        g = tuple(words[i : i + 6])
        grams[g] = grams.get(g, 0) + 1
        if grams[g] >= 5:
            return True
    return False


def read_trace(path: Path, have: int, timeout: float = 10.0) -> dict | None:
    """The engine writes its record just after the response closes."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if path.is_file():
            lines = [l for l in path.read_text().splitlines() if l.strip()]
            if len(lines) > have:
                return json.loads(lines[have])
        time.sleep(0.25)
    return None


def step_record(label: str, client: dict, tr: dict | None) -> dict:
    tr = tr or {}
    pre = tr.get("prefill") or []
    pre_tok = sum(c["tok"] for c in pre)
    pre_s = sum(c["s"] for c in pre)
    if not pre and tr.get("prompt_tokens") is not None and tr.get("decode_start") is not None:
        # Resident models don't log prefill chunks: uncached tokens, timed
        # from the cache fetch to the start of decode.
        pre_tok = tr["prompt_tokens"] - (tr.get("cached_tokens") or 0)
        pre_s = max(0.0, tr["decode_start"] - (tr.get("fetch") or 0.0))
    dec_tok = tr.get("decode_tokens") or 0
    dec_s = tr.get("decode_s") or 0.0
    pd = tr.get("phase_decode") or {}
    pp = tr.get("phase_prefill") or {}
    out_text = client["reasoning"] + client["text"]
    return {
        "label": label,
        "prompt_tokens": tr.get("prompt_tokens") or client["usage"].get("prompt_tokens"),
        "cached_tokens": tr.get("cached_tokens"),
        "prefill_tokens": pre_tok,
        "prefill_s": round(pre_s, 3),
        "prefill_tok_s": round(pre_tok / pre_s, 1) if pre_s else None,
        "prefill_read_gb": round(sum(c["gb"] for c in pre), 2),
        "prefill_disk_wait_s": pp.get("disk_wait_s"),
        "ttft_s": client["ttft_s"],
        "decode_tokens": dec_tok,
        "decode_s": round(dec_s, 3),
        "decode_tok_s": round(dec_tok / dec_s, 2) if dec_s else None,
        "tool_tokens": tr.get("tool_tokens"),
        "strict_tokens": tr.get("strict_tokens"),
        "verify": tr.get("verify"),
        "decode_read_gb": pd.get("read_gb"),
        "decode_disk_wait_s": pd.get("disk_wait_s"),
        "decode_hit_rate": pd.get("hit_rate"),
        "stitch_tokens": (pd.get("closed_ok") or 0) + (pd.get("full_stitch_tokens") or 0),
        "stitch_replays": pd.get("closed_replay"),
        "stitch_full": tr.get("stitch_full"),
        "gb_per_token": round(pd["read_gb"] / dec_tok, 3) if pd.get("read_gb") is not None and dec_tok else None,
        "mx_peak_gb": tr.get("mx_peak_gb"),
        "engine_prune": tr.get("prune"),
        "engine_wait_above": tr.get("wait_above"),
        "events": [e.get("name") for e in tr.get("events") or []],
        "engine_error": tr.get("error"),
        "wall_s": client["wall_s"],
        "finish": client["finish"],
        "generated_tool_calls": client["tool_calls"],
        "degenerate": degenerate(out_text),
        "preview": out_text[:240],
    }


def workspace_note(sess: dict) -> str:
    # Open editor files bring the front matter to ~13k tokens, about what a
    # real coding agent sends, so prefill and decode run at agent context.
    return sess["workspace_note"] + "\n\n" + (DATA / "open_files.txt").read_text()


def front_matter_text() -> str:
    """System prompt and workspace note as plain text, for the loss probe."""
    front = json.loads((DATA / "front_matter.json").read_text())
    sess = json.loads((DATA / "session.json").read_text())
    return front["system"] + "\n\n" + workspace_note(sess)


def run_scenario(port: int, run: dict, trace: Path, quick: bool, save) -> list[dict]:
    front = json.loads((DATA / "front_matter.json").read_text())
    sess = json.loads((DATA / "session.json").read_text())
    note = workspace_note(sess)
    head = [
        {"role": "system", "content": front["system"]},
        {"role": "user", "content": note},
    ]
    messages = head + [{"role": "user", "content": sess["ask"]}]
    steps: list[dict] = []
    total = len(sess["steps"]) + 1

    def one(label, msgs, max_tokens):
        mt = max(48, max_tokens // 2) if quick else max_tokens
        have = len([l for l in trace.read_text().splitlines() if l.strip()]) if trace.is_file() else 0
        log(f"request {len(steps) + 1}/{total}: {label} (max_tokens {mt})")
        client = stream_chat(port, request_body(run, front, msgs, mt), timeout=3600.0)
        rec = step_record(label, client, read_trace(trace, have))
        steps.append(rec)
        log(
            f"  prefill {rec['prefill_tokens']} tok in {rec['prefill_s']}s, ttft {rec['ttft_s']}s, "
            f"decode {rec['decode_tokens']} tok @ {rec['decode_tok_s']} tok/s"
        )
        save(steps)

    for i, st in enumerate(sess["steps"]):
        one(st["label"], messages, st["max_tokens"])
        a = st.get("assistant")
        if not a:
            break
        call = a["call"]
        cid = f"call_{i}"
        messages = messages + [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": a.get("reasoning", ""),
                "tool_calls": [
                    {"id": cid, "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}
                ],
            },
            {"role": "tool", "tool_call_id": cid, "content": a["result"]},
        ]
    nc = sess["new_chat"]
    one(nc["label"], head + [{"role": "user", "content": nc["ask"]}], nc["max_tokens"])
    return steps


# --------------------------------------------------------------- metrics


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else round((xs[m - 1] + xs[m]) / 2, 3)


def summarize(steps: list[dict], load_s: float, load_info: dict, probe: dict | None) -> dict:
    task = [s for s in steps if not s["label"].startswith("new chat")]
    tool_rounds = task[1:]
    dec = [s for s in steps if s["decode_tokens"] and s["decode_s"]]
    tot_tok = sum(s["decode_tokens"] for s in dec)
    tot_s = sum(s["decode_s"] for s in dec)
    tool_heavy = [s for s in dec if (s["tool_tokens"] or 0) >= 0.5 * s["decode_tokens"]]
    new_chat = next((s for s in steps if s["label"].startswith("new chat")), None)
    peak = max((s["mx_peak_gb"] or 0 for s in steps), default=0)
    m = {
        "load_s": round(load_s, 1),
        "engine_mode": load_info.get("mode"),
        "expert_cache_gb": load_info.get("cache_gb"),
        "ram_budget_gb": load_info.get("ram_budget_gb"),
        "peak_memory_gb": peak or None,
        "first_turn_ttft_s": task[0]["ttft_s"] if task else None,
        "first_turn_prefill_tokens": task[0]["prefill_tokens"] if task else None,
        "first_turn_prefill_tok_s": task[0]["prefill_tok_s"] if task else None,
        "tool_result_ingest_s_median": median([s["prefill_s"] for s in tool_rounds]),
        "ttft_after_tool_s_median": median([s["ttft_s"] for s in tool_rounds]),
        "decode_tok_s": round(tot_tok / tot_s, 2) if tot_s else None,
        "decode_tok_s_median": median([s["decode_tok_s"] for s in dec]),
        "tool_call_tok_s_median": median([s["decode_tok_s"] for s in tool_heavy]),
        "gb_read_per_token_median": median([s["gb_per_token"] for s in dec]),
        "decode_hit_rate_median": median([s["decode_hit_rate"] for s in dec]),
        "stitch_tokens": sum(s.get("stitch_tokens") or 0 for s in dec),
        "stitch_full": any(s.get("stitch_full") for s in steps),
        "new_chat_ttft_s": new_chat["ttft_s"] if new_chat else None,
        "task_wall_s": round(sum(s["wall_s"] for s in task), 1),
        "decode_tokens_total": tot_tok,
    }
    if probe:
        arm = probe["arms"].get("run") or {}
        tool = probe["arms"].get("tool") or {}
        m.update(
            {
                "top1_agree": arm.get("top1_agree"),
                "nll_delta": arm.get("nll_delta"),
                "kl_mean": arm.get("kl_mean"),
                "dropped_frac": arm.get("dropped_frac"),
                "tool_span_top1_agree": tool.get("top1_agree"),
            }
        )
    return m


def validate(mode: str, settings: dict, load_info: dict, steps: list[dict], probe: dict | None) -> tuple[list, list]:
    """(errors that make the run invalid, notes worth keeping)."""
    errors, notes = [], []
    want_prune = float(settings.get("PAGED_MOE_PRUNE") or 0)
    if not load_info:
        errors.append("no engine load line in the server log")
    elif load_info.get("mode") == "streamed":
        got = float(load_info.get("prune") or 0)
        if abs(got - want_prune) > 1e-6:
            errors.append(f"engine ran PRUNE {got}, asked for {want_prune}")
        if mode == "exact" and float(load_info.get("wait_above") or 0) != 0:
            errors.append("exact run with WAIT_ABOVE on")
    for s in steps:
        if s["engine_error"]:
            errors.append(f"{s['label']}: engine error {s['engine_error']}")
        if (s["decode_tokens"] or 0) < 8:
            notes.append(f"{s['label']}: only {s['decode_tokens']} decode tokens")
        if s["degenerate"]:
            notes.append(f"{s['label']}: output repeats itself")
        for e in s["events"]:
            if e in ("pressure_critical", "metal_oom", "cold_miss"):
                notes.append(f"{s['label']}: {e}")
    for s in steps[1:-1]:
        if not s["cached_tokens"]:
            notes.append(f"{s['label']}: no prefix reuse (cold prefill)")
    if probe and mode == "exact":
        arm = probe["arms"].get("run") or {}
        if arm.get("top1_agree") != 1.0 or (arm.get("kl_mean") or 0) > 1e-6:
            errors.append(f"exact settings did not match the exact model: {arm}")
    if load_info.get("mode") == "resident" and mode == "speed":
        notes.append("the whole model fits in memory, so speed settings have nothing to drop (same as exact)")
    return errors, notes


# ------------------------------------------------------------------ run


def run(
    model: str,
    mode: str = "speed",
    *,
    speed: float | str = DEFAULT_SPEED,
    quick: bool = False,
    out_dir: str | Path | None = None,
    port: int = 18799,
    probe: bool = True,
    ctx: int = DEFAULT_CTX,
    think: bool = True,
    extra_env: dict | None = None,
    allow_busy: bool = False,
    stall_s: float = 300.0,
    keep_state: bool = False,
) -> dict:
    """Bench one mode ("speed" or "exact") and return the result document.

    The same document is written to out_dir as JSON. status is "ok",
    "invalid" (numbers recorded but a check failed), "failed" or
    "unsupported" (the model doesn't fit this Mac).
    """
    if mode not in ("speed", "exact"):
        raise ValueError(f"mode is speed or exact, not {mode!r}")
    model_dir = resolve_model(model)
    if not checkpoint_complete(model_dir):
        raise SystemExit(f"{model_dir} is missing config.json or safetensors shards (download not finished?)")
    ck = autotune.inspect(str(model_dir))
    machine = machine_info(model_dir)
    warnings = preflight(port, machine, allow_busy)
    for w in warnings:
        log(f"warning: {w}")

    out_root = Path(out_dir).expanduser() if out_dir else Path.home() / ".paged_moe" / "bench"
    out_root.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w.-]+", "-", model_dir.name)
    out = out_root / f"{slug}-{mode}{'-quick' if quick else ''}.json"
    state = Path(tempfile.mkdtemp(prefix=f"paged-moe-bench-{slug}-{mode}-"))
    trace = state / "trace.jsonl"

    extra = dict(extra_env or {})
    settings = mode_env(ck, mode, speed)
    env = engine_env(settings, state, trace, ctx, extra)
    run_cfg = {
        "sample": sampling(model_dir),
        "chat_template_kwargs": template_kwargs(model_dir, think),
    }
    doc: dict = {
        "schema": SCHEMA,
        "status": "running",
        "started": now(),
        "machine": machine,
        "versions": versions(),
        "model": model_card(model_dir, ck),
        "mode": mode,
        "speed": None if mode == "exact" else speed,
        "quick": quick,
        "settings": {
            "env": {
                k: env[k]
                for k in sorted(env)
                if k.startswith("PAGED_MOE_")
                and k not in ("PAGED_MOE_TRACE", "PAGED_MOE_PREFIX_DISK_DIR", "PAGED_MOE_SIDECAR_DIR")
            },
            **run_cfg,
        },
        "warnings": warnings + ([f"extra env: {extra}"] if extra else []),
        "state_dir": str(state),
    }

    def save(**kw):
        doc.update(kw)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2))
        tmp.replace(out)

    def cleanup():
        if not keep_state:
            for name in ("prefix_cache", "sidecar"):
                shutil.rmtree(state / name, ignore_errors=True)

    save()
    knobs = " ".join(f"{k.removeprefix('PAGED_MOE_')}={v}" for k, v in settings.items())
    log(f"{model_dir.name} | {ck.describe()}")
    log(f"{mode}{' (quick)' if quick else ''}: {knobs}")
    log(f"results -> {out}")

    server = Server(model_dir, port, env, state / "server.log")
    steps: list[dict] = []
    load_info: dict = {}
    load_s = 0.0
    try:
        load_s = server.wait_ready(timeout=1800)
        load_info = server.load_info()
        save(
            load_s=round(load_s, 1),
            engine={
                k: load_info.get(k)
                for k in (
                    "mode", "total_gb", "backbone_gb", "expert_gb", "ram_gb", "ram_budget_gb",
                    "cache_gb", "kv_reserve_gb", "reserve_ctx", "prune", "wait_above",
                )
            },
        )
        log(f"loaded in {load_s:.0f}s ({load_info.get('mode')}), expert cache {load_info.get('cache_gb')} GB")
        stop_watch = server.watch(stall_s=stall_s)
        try:
            steps = run_scenario(port, run_cfg, trace, quick, lambda s: save(steps=s))
        finally:
            stop_watch()
    except DoesNotFit as e:
        server.stop()
        cleanup()
        save(status="unsupported", error=f"does not fit on this Mac: {e}", finished=now())
        log(f"does not fit on this Mac: {e}")
        return doc
    except BaseException as e:
        why = server.hung or f"{type(e).__name__}: {e}"
        server.stop()
        cleanup()
        save(
            status="failed",
            error=why,
            steps=steps,
            finished=now(),
            server_log_tail=server.log_path.read_text(errors="replace")[-6000:],
        )
        log(f"failed: {why}")
        if server.hung:
            log("the engine stopped making progress. Restart the Mac before loading another model; "
                "a stuck GPU job keeps its memory until then.")
        if isinstance(e, KeyboardInterrupt):
            raise
        return doc
    server.stop()

    probe_doc = None
    if probe:
        n = 32 if mode == "exact" else (64 if quick else 160)
        log(f"loss probe against the exact model ({n} tokens)")
        penv = dict(env)
        penv.pop("PAGED_MOE_TRACE", None)
        ctx_file = state / "front_matter.txt"
        ctx_file.write_text(front_matter_text())
        pout = state / "probe.json"
        r = subprocess.run(
            [sys.executable, "-m", "pagedmoe.benchmark.probe", "--model", str(model_dir),
             "--context", str(ctx_file), "--out", str(pout), "--tokens", str(n)],
            env=penv,
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and pout.is_file():
            probe_doc = json.loads(pout.read_text())
        else:
            doc["warnings"].append(f"loss probe failed: {(r.stderr or r.stdout)[-800:]}")
            log("loss probe failed (see warnings in the result file)")
        save(probe=probe_doc)
    cleanup()

    errors, notes = validate(mode, settings, load_info, steps, probe_doc)
    metrics = summarize(steps, load_s, load_info, probe_doc)
    save(status="ok" if not errors else "invalid", finished=now(), metrics=metrics, errors=errors, notes=notes)
    for e in errors:
        log(f"error: {e}")
    for n_ in notes:
        log(f"note: {n_}")
    return doc


# ---------------------------------------------------------------- report


_ROWS = [
    ("decode tok/s", "decode_tok_s", "{:.2f}"),
    ("tool call tok/s", "tool_call_tok_s_median", "{:.2f}"),
    ("first turn prompt", "first_turn_prefill_tokens", "{:.0f} tok"),
    ("first turn TTFT", "first_turn_ttft_s", "{:.1f} s"),
    ("first turn prefill", "first_turn_prefill_tok_s", "{:.0f} tok/s"),
    ("tool result ingest", "tool_result_ingest_s_median", "{:.2f} s"),
    ("TTFT after tool", "ttft_after_tool_s_median", "{:.2f} s"),
    ("new chat TTFT", "new_chat_ttft_s", "{:.2f} s"),
    ("whole task", "task_wall_s", "{:.0f} s"),
    ("load", "load_s", "{:.1f} s"),
    ("expert cache", "expert_cache_gb", "{:.1f} GB"),
    ("peak memory", "peak_memory_gb", "{:.1f} GB"),
    ("GB read / token", "gb_read_per_token_median", "{:.3f}"),
    ("cache hit rate", "decode_hit_rate_median", "{:.3f}"),
    ("top-1 vs exact", "top1_agree", "{:.3f}"),
    ("NLL delta", "nll_delta", "{:+.4f}"),
]


def report(docs: list[dict]) -> str:
    if not docs:
        return ""
    m0 = docs[0]
    mc, mach = m0["model"], m0["machine"]
    lines = [
        "",
        f"{mc['name']}: {mc.get('total_params_b')}B total, {mc.get('active_params_b')}B active, "
        f"{mc.get('bits')}-bit, {mc['disk_gb']} GB",
        f"{mach['chip']}, {mach['ram_gb']} GB, macOS {mach['macos']}"
        + (", model on internal SSD" if mach["model_disk"].get("internal") else
           ", model on external drive" if mach["model_disk"].get("internal") is False else ""),
        "",
    ]
    heads = []
    for d in docs:
        label = d["mode"] if d["mode"] == "exact" else f"speed {d['speed']}"
        heads.append(label + (" quick" if d.get("quick") else ""))
    w0 = max(len(r[0]) for r in _ROWS) + 2
    cols = [max(16, len(h) + 2) for h in heads]
    lines.append("".ljust(w0) + "".join(h.rjust(c) for h, c in zip(heads, cols)))
    lines.append("".ljust(w0) + "".join(("-" * (c - 2)).rjust(c) for c in cols))
    for label, key, fmt in _ROWS:
        cells = []
        for d, c in zip(docs, cols):
            v = (d.get("metrics") or {}).get(key)
            cells.append((fmt.format(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else "-").rjust(c))
        lines.append(label.ljust(w0) + "".join(cells))
    lines.append("".ljust(w0) + "".join(d["status"].rjust(c) for d, c in zip(docs, cols)))
    return "\n".join(lines)
