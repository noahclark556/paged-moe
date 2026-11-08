# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-request breakdown from an paged-moe server log.

Rebuilds each chat request from the plain log lines the server already
prints, groups requests into sessions and prints where the wall clock went:

  python bench/log_breakdown.py server.log
  python bench/log_breakdown.py LOG --since 2026-10-01 --json out.json

A request starts at its `POST /v1/chat/completions` line. Prefill comes from
the `[prefill] N tok in Xs` lines, decode from `[paged-moe] decode:`. The gap
from one request's end to the next POST is time spent outside the engine
(tool runs, prompt building), so it is reported as `client_s`.

Log lines sometimes run together (no newline between a `[prefill]` summary and
the next INFO line), so every pattern is searched anywhere in a line.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass, field

TS_INFO = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3}) - INFO")
TS_HTTP = re.compile(r"\[(\d{2}/\w{3}/\d{4} \d{2}:\d{2}:\d{2})\] \"POST /v1/chat/completions")
START = re.compile(r"Starting httpd at")
MODEL = re.compile(r'\[paged-moe\] \{"model_path": "([^"]+)".*?"mode": "(\w+)"')
PROMPT_CACHE = re.compile(r"Prompt Cache: (\d+) sequences, ([\d.]+) GB")
PROGRESS = re.compile(r"Prompt processing progress: (\d+)/(\d+)")
CHUNK = re.compile(r"\[prefill\] chunk (\d+) tok at kv=(\d+)")
PREFILL = re.compile(
    r"\[prefill\] (\d+) tok in ([\d.]+)s · ([\d.]+) GB experts.*?· ([\d.]+)s blocked, "
    r"([\d.]+)s copy.*?, ([\d.]+)s rest"
)
PREFILL_OLD = re.compile(r"\[prefill\] (\d+) tok in ([\d.]+)s · ([\d.]+) GB experts")
DECODE = re.compile(r"\[paged-moe\] decode: (\d+) tokens in ([\d.]+)s = ([\d.]+) tok/s(?: last=(\d+))?(?: strict=(\d+))?(?: tool=(\d+))?")
TTW = re.compile(r"\[ttw\] warm turn=\d+ ms=([\d.]+) tok=(\d+)/\d+ hit=([\d.]+) miss_gb=([\d.]+) disk_s=([\d.]+)")
VERIFY = re.compile(r"\[verify\] done passes=(\d+) accept=([\d.]+) tok/pass=([\d.]+) exact_tok=(\d+) plain_tok=(\d+)")
DISK_LOAD = re.compile(r"\[prefix-disk\] loaded front matter (\d+) tok")
COLD = re.compile(r"cache miss with store|COLD MISS")
YIELD = re.compile(r"expert cache yielded for a ([\d.]+) GB")
PRESSURE = re.compile(r"memory pressure (warn|critical)")
OOM = re.compile(r"Metal OOM|Insufficient Memory")
TRACEBACK = re.compile(r"^Traceback")
WARM = re.compile(r"\[prefix\] warm_prefix=1")
THINK = re.compile(r"\[paged-moe\] think prefill: \+(\d+) tokens")
TOOL_DROP = re.compile(r"dropped unparseable tool call|tool JSON unrecoverable")
TOOL_FIX = re.compile(r"tool JSON repaired|kept with raw values|without its close tag")
SLOW_HEALTH = re.compile(r"health check slow")


def _info_ts(line: str) -> float | None:
    m = TS_INFO.search(line)
    if not m:
        return None
    t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return t.timestamp() + int(m.group(2)) / 1000


def _http_ts(line: str) -> float | None:
    m = TS_HTTP.search(line)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%d/%b/%Y %H:%M:%S").timestamp()


@dataclass
class Request:
    t0: float
    model: str
    mode: str
    prompt_tokens: int = 0
    reused_tokens: int = -1
    new_tokens: int = 0
    prefill_s: float = 0.0
    prefill_gb: float = 0.0
    blocked_s: float = 0.0
    copy_s: float = 0.0
    rest_s: float = 0.0
    tail_s: float = 0.0
    chunks: list = field(default_factory=list)
    t_prefill_end: float | None = None
    decode_tokens: int = 0
    decode_s: float = 0.0
    strict_tokens: int = 0
    warm_ms: float = 0.0
    warm_tokens: int = 0
    warm_miss_gb: float = 0.0
    warm_disk_s: float = 0.0
    verify_passes: int = 0
    verify_accept: float = 0.0
    verify_exact: int = 0
    disk_prefix: bool = False
    cold_miss: bool = False
    yields: int = 0
    pressure: int = 0
    oom: bool = False
    traceback: bool = False
    warm_request: bool = False
    think_resume: int = 0
    tool_dropped: int = 0
    tool_repaired: int = 0
    t_end: float | None = None
    client_s: float | None = None
    restart_after: bool = False

    @property
    def end(self) -> float:
        if self.t_end is not None:
            return self.t_end
        base = self.t_prefill_end or self.t0
        return base + self.decode_s

    @property
    def wall_s(self) -> float:
        return max(0.0, self.end - self.t0)


def parse(path: str, since: float | None = None) -> list[Request]:
    reqs: list[Request] = []
    model, mode = "?", "?"
    cur: Request | None = None
    last_ts: float | None = None
    with open(path, errors="replace") as f:
        for line in f:
            ts = _info_ts(line)
            if ts is not None:
                last_ts = ts
            if START.search(line):
                if cur is not None:
                    cur.restart_after = True
                if last_ts is not None and cur is not None and cur.t_end is None:
                    pass
                continue
            m = MODEL.search(line)
            if m:
                model = m.group(1).rstrip("/").split("/")[-1]
                mode = m.group(2)
                continue
            hts = _http_ts(line)
            if hts is not None:
                if cur is not None:
                    reqs.append(cur)
                cur = Request(t0=hts, model=model, mode=mode)
                continue
            if cur is None:
                continue
            if WARM.search(line):
                cur.warm_request = True
            m = PROGRESS.search(line)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                if total > 1:
                    cur.prompt_tokens = max(cur.prompt_tokens, total)
                if ts is not None:
                    cur.t_prefill_end = ts
            m = CHUNK.search(line)
            if m:
                n, kv = int(m.group(1)), int(m.group(2))
                if cur.reused_tokens < 0:
                    cur.reused_tokens = kv
                cur.chunks.append((n, kv))
            m = PREFILL.search(line)
            if m:
                n, s, gb = int(m.group(1)), float(m.group(2)), float(m.group(3))
                if n <= 8:
                    cur.tail_s += s
                else:
                    cur.prefill_s += s
                    cur.blocked_s += float(m.group(4))
                    cur.copy_s += float(m.group(5))
                    cur.rest_s += float(m.group(6))
                cur.new_tokens += n
                cur.prefill_gb += gb
            else:
                m = PREFILL_OLD.search(line)
                if m:
                    n, s = int(m.group(1)), float(m.group(2))
                    if n <= 8:
                        cur.tail_s += s
                    else:
                        cur.prefill_s += s
                    cur.new_tokens += n
                    cur.prefill_gb += float(m.group(3))
            m = DECODE.search(line)
            if m:
                cur.decode_tokens += int(m.group(1))
                cur.decode_s += float(m.group(2))
                if m.group(5):
                    cur.strict_tokens += int(m.group(5))
            m = TTW.search(line)
            if m:
                cur.warm_ms += float(m.group(1))
                cur.warm_tokens += int(m.group(2))
                cur.warm_miss_gb += float(m.group(4))
                cur.warm_disk_s += float(m.group(5))
            m = VERIFY.search(line)
            if m:
                cur.verify_passes += int(m.group(1))
                cur.verify_accept = float(m.group(2))
                cur.verify_exact += int(m.group(4))
            if DISK_LOAD.search(line):
                cur.disk_prefix = True
            if COLD.search(line):
                cur.cold_miss = True
            if YIELD.search(line):
                cur.yields += 1
            if PRESSURE.search(line):
                cur.pressure += 1
            if OOM.search(line):
                cur.oom = True
            if TRACEBACK.search(line):
                cur.traceback = True
            m = THINK.search(line)
            if m:
                cur.think_resume += int(m.group(1))
            if TOOL_DROP.search(line):
                cur.tool_dropped += 1
            if TOOL_FIX.search(line):
                cur.tool_repaired += 1
    if cur is not None:
        reqs.append(cur)
    if since is not None:
        reqs = [r for r in reqs if r.t0 >= since]
    for r in reqs:
        if r.reused_tokens >= 0:
            r.prompt_tokens = r.reused_tokens + r.new_tokens
    for a, b in zip(reqs, reqs[1:]):
        if a.model == b.model and not a.restart_after:
            a.client_s = max(0.0, b.t0 - a.end)
    return reqs


def sessions(reqs: list[Request], idle_s: float = 600.0) -> list[list[Request]]:
    out: list[list[Request]] = []
    for r in reqs:
        if out and out[-1][-1].model == r.model and (r.t0 - out[-1][-1].end) < idle_s:
            out[-1].append(r)
        else:
            out.append([r])
    return out


def _fmt(x: float) -> str:
    return f"{x:7.1f}"


def summarize(sess: list[Request]) -> dict:
    engine = sum(r.wall_s for r in sess)
    client = sum(r.client_s or 0.0 for r in sess[:-1] if (r.client_s or 0) < 600)
    pre = sum(r.prefill_s for r in sess)
    tail = sum(r.tail_s for r in sess)
    dec = sum(r.decode_s for r in sess)
    warm = sum(r.warm_ms for r in sess) / 1000
    blocked = sum(r.blocked_s for r in sess)
    toks = sum(r.decode_tokens for r in sess)
    new = sum(r.new_tokens for r in sess)
    prompt = sum(r.prompt_tokens for r in sess if r.reused_tokens >= 0)
    reused = sum(r.reused_tokens for r in sess if r.reused_tokens >= 0)
    other = max(0.0, engine - pre - tail - dec)
    return {
        "model": sess[0].model,
        "start": dt.datetime.fromtimestamp(sess[0].t0).strftime("%m-%d %H:%M"),
        "requests": len(sess),
        "wall_s": engine + client,
        "engine_s": engine,
        "client_s": client,
        "prefill_s": pre,
        "prefill_blocked_s": blocked,
        "tail_s": tail,
        "decode_s": dec,
        "decode_warm_s": warm,
        "other_engine_s": other,
        "decode_tokens": toks,
        "decode_tok_s": toks / dec if dec else 0.0,
        "steady_tok_s": (toks - sum(r.warm_tokens for r in sess)) / max(1e-9, dec - warm) if dec > warm else 0.0,
        "prompt_tokens": prompt,
        "new_tokens": new,
        "reuse_rate": reused / prompt if prompt else 0.0,
        "prefill_gb": sum(r.prefill_gb for r in sess),
        "cold_miss": sum(r.cold_miss for r in sess),
        "disk_prefix": sum(r.disk_prefix for r in sess),
        "yields": sum(r.yields for r in sess),
        "pressure": sum(r.pressure for r in sess),
        "oom": sum(r.oom for r in sess),
        "tracebacks": sum(r.traceback for r in sess),
        "restarts": sum(r.restart_after for r in sess[:-1]),
        "warm_requests": sum(r.warm_request for r in sess),
        "think_resume_tokens": sum(r.think_resume for r in sess),
        "tool_dropped": sum(r.tool_dropped for r in sess),
        "tool_repaired": sum(r.tool_repaired for r in sess),
        "verify_passes": sum(r.verify_passes for r in sess),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--since", help="YYYY-MM-DD[ HH:MM]")
    ap.add_argument("--min-requests", type=int, default=4)
    ap.add_argument("--json")
    ap.add_argument("--turns", action="store_true", help="print every request")
    args = ap.parse_args()
    since = None
    if args.since:
        fmt = "%Y-%m-%d %H:%M" if " " in args.since else "%Y-%m-%d"
        since = dt.datetime.strptime(args.since, fmt).timestamp()
    reqs = parse(args.log, since)
    ss = [s for s in sessions(reqs) if len(s) >= args.min_requests]
    rows = [summarize(s) for s in ss]
    hdr = ("start", "model", "req", "wall", "engine", "client", "prefill", "tail", "dec", "warm", "tok/s", "reuse", "new_tok", "cold", "fail")
    print(" ".join(f"{h:>8}" for h in hdr))
    for r in rows:
        fail = r["oom"] + r["tracebacks"] + r["restarts"] + r["tool_dropped"]
        print(
            f"{r['start']:>8} {r['model'][:8]:>8} {r['requests']:8d} {_fmt(r['wall_s'])} {_fmt(r['engine_s'])} "
            f"{_fmt(r['client_s'])} {_fmt(r['prefill_s'])} {_fmt(r['tail_s'])} {_fmt(r['decode_s'])} "
            f"{_fmt(r['decode_warm_s'])} {r['decode_tok_s']:8.2f} {r['reuse_rate']:8.2f} {r['new_tokens']:8d} "
            f"{r['cold_miss']:8d} {fail:8d}"
        )
    if args.turns:
        for s in ss:
            print(f"\n== {s[0].model} {dt.datetime.fromtimestamp(s[0].t0)}")
            for r in s:
                print(
                    f"  prompt={r.prompt_tokens:6d} reused={r.reused_tokens:6d} new={r.new_tokens:5d} "
                    f"prefill={r.prefill_s:5.1f}s tail={r.tail_s:4.1f}s dec={r.decode_tokens:4d}/{r.decode_s:5.1f}s "
                    f"warm={r.warm_ms/1000:4.1f}s client={(r.client_s or 0):6.1f}s"
                    + (" COLD" if r.cold_miss else "") + (" DISK" if r.disk_prefix else "")
                    + (f" yield={r.yields}" if r.yields else "") + (" WARMREQ" if r.warm_request else "")
                )
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"sessions": rows, "requests": [[asdict(r) for r in s] for s in ss]}, f, indent=1)
    if rows:
        tot = {k: sum(r[k] for r in rows) for k in ("wall_s", "engine_s", "client_s", "prefill_s", "tail_s", "decode_s", "decode_warm_s", "other_engine_s", "prefill_blocked_s")}
        w = tot["wall_s"] or 1
        print("\nshare of wall clock across these sessions:")
        for k in ("prefill_s", "tail_s", "decode_s", "decode_warm_s", "other_engine_s", "client_s"):
            print(f"  {k:16s} {tot[k]:9.1f} s  {100*tot[k]/w:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
