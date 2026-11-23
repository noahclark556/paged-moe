# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Trainers.

    from pagedmoe.train import SFTTrainer, SFTConfig

    trainer = SFTTrainer(SFTConfig(model="~/mlx-models/Qwen3.6-35B-A3B-4bit",
                                   data="data/chat.jsonl"))
    trainer.train()

Construction loads the tokenizer and data, plans the run against this Mac,
loads the model in the planned mode, adds LoRA and probes memory and speed.
`trainer.plan_text()` (printed on construction) says what was decided and
why. `train()` runs to the end, saving as it goes; Ctrl-C saves and stops,
and `resume=True` picks up from the last save with the same data order.

Every trainer is the same loop with a different objective and batch maker,
so a new method is a small subclass: see `Trainer.make_objective` and
`Trainer.next_batches`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map_with_path

from . import data as D
from . import env as train_env
from . import experts as train_experts
from . import fastgrad as train_fastgrad
from . import linear_attn as train_linear_attn
from . import lora as train_lora
from . import objectives as O
from . import plan as P
from .config import DEFAULT_LR, CONFIGS, TrainConfig
from .executor import Batch, FusedExecutor, LayeredExecutor
from .stats import Callback, Console, JsonlLog, Rate, WandbLog


def _tokenizer(path: str):
    """The tokenizer the engine would load, without loading weights."""
    from mlx_lm.utils import load_tokenizer

    from ..chat_compat import fix_think_tokens

    p = Path(path)
    cfg = json.loads((p / "config.json").read_text())
    tok = load_tokenizer(p, {}, eos_token_ids=cfg.get("eos_token_id"))
    try:
        fix_think_tokens(tok)
    except Exception:
        pass
    return tok


_HINT_SHOWN = False


def _app_hint() -> None:
    """One line per process. Paged Trainer and the bench children set
    PAGED_MOE_TRAIN_HINT=0, so it only shows from the CLI and Python."""
    global _HINT_SHOWN
    if _HINT_SHOWN or os.environ.get("PAGED_MOE_TRAIN_HINT", "1") == "0":
        return
    _HINT_SHOWN = True
    print("[train] tip: Paged Trainer is a Mac app for this same trainer, with plans, "
          "live charts, checkpoint chat and resume in one window. https://pagedmoe.com/train "
          "(PAGED_MOE_TRAIN_HINT=0 hides this)", flush=True)


class Trainer:
    method = "sft"
    rows_per_unit = 1
    unit_kind = "example"

    def __init__(
        self,
        config: TrainConfig | None = None,
        *,
        train_data=None,
        eval_data=None,
        formatter=None,
        encoder: nn.Module | None = None,
        objective: O.Objective | None = None,
        callbacks: list[Callback] | None = None,
        tokenizer=None,
        plan_only: bool = False,
        **overrides,
    ):
        cfg = config or CONFIGS[self.method]()
        if overrides:
            cfg = cfg.replace(**overrides)
        if train_data is not None:
            cfg = cfg.replace(data=train_data)
        if eval_data is not None:
            cfg = cfg.replace(eval_data=eval_data)
        if not cfg.model:
            raise ValueError("config.model is required")
        self.cfg = cfg
        self.encoder = encoder
        self.formatter_fn = formatter
        self._objective_override = objective
        self.t_setup = time.perf_counter()
        self._listeners = list(callbacks or [])
        self._ctx: dict = {}
        self._ctx_t0 = self._last_tick = self._last_print = time.perf_counter()

        train_env.apply()
        train_env.index_ops_stop_gradient()
        train_env.exact_numerics()

        self.facts = P.model_facts(cfg.model)
        self.tokenizer = tokenizer or _tokenizer(self.facts.path)
        self.out = self._output_dir()

        rows = D.load_rows(cfg.data, "train")
        eval_rows = D.load_rows(cfg.eval_data, "valid") if cfg.eval_data is not None else []
        if not rows:
            raise ValueError(f"no training rows in {cfg.data!r}")
        self.max_len = self._pick_max_len(rows)
        self.fmt = D.Formatter(
            self.tokenizer,
            max_len=self.max_len,
            chat=cfg.chat_template,
            completion_only=cfg.completion_only,
            template_kwargs=cfg.template_kwargs,
            system=cfg.system_prompt,
        )
        self.n_rows = len(rows)
        self.fmt.pack = bool(getattr(cfg, "packing", False))
        self.units = self.build_units(rows)
        self.skipped = self._skipped
        if not self.units:
            raise ValueError(f"no usable rows left to train on (all {len(rows)} were filtered: "
                             "too long for max_seq_len, or nothing to train in them)")
        self.eval_units = self.build_units(eval_rows) if eval_rows else []
        self.data_summary = D.lengths_summary(self.units)

        self.plan = P.initial(
            cfg, self.facts, self.data_summary["mean"], len(self.units), self.rows_per_unit,
            max_len=self.max_len,
        )
        need = D._round_up(self.max_len) * self.rows_per_unit
        if self.plan.micro_tokens < need and not cfg.micro_batch_tokens:
            self.plan.micro_tokens = need

        self.plan_only = plan_only
        if plan_only:
            # Headers, tokenizer and data only: what the planner decides before
            # the model loads. No probe, so no estimate.
            self.streamed = self.plan.mode == "streamed"
            self.lora_report = None
            self.est_step = None
            self.schedule = D.Schedule(
                self.units, self.plan.per_step, self.plan.micro_tokens, cfg.seed,
                rows_per_unit=self.rows_per_unit, max_rows=self.plan.max_rows,
            )
            steps = math.ceil(cfg.epochs * self.schedule.steps_per_epoch)
            self.total_steps = int(cfg.max_steps) if cfg.max_steps else max(1, steps)
            self.lr = cfg.learning_rate or DEFAULT_LR.get(self.method, 1e-4)
            return

        self._phase("load", "loading the model"
                    + (" (backbone resident, experts stay on disk)" if self.plan.mode == "streamed" else ""))
        self._load_model()
        self.objective = self._objective_override or self.make_objective()
        self.executor = self._make_executor()
        self._probe()

        self.schedule = self._schedule(self.units, self.plan.per_step, cfg.seed)
        steps = math.ceil(cfg.epochs * self.schedule.steps_per_epoch)
        self.total_steps = int(cfg.max_steps) if cfg.max_steps else max(1, steps)
        self.lr = cfg.learning_rate or DEFAULT_LR.get(self.method, 1e-4)
        self.optimizer = self._make_optimizer()
        self.enc_optimizer = self._make_optimizer() if encoder is not None else None

        self.state = {"step": 0, "epoch": 0, "k": 0, "tokens": 0, "elapsed": 0.0}
        self.callbacks: list[Callback] = [Console(cfg.log_steps), JsonlLog(self.out / "metrics.jsonl")]
        if "wandb" in (cfg.report_to or []):
            self.callbacks.append(WandbLog(name=self.out.name, config=cfg.to_dict()))
        self.callbacks += self._listeners
        self._stop = False
        self._last_save_t = time.time()
        self._saved_step = -1
        if cfg.resume:
            self._resume(cfg.resume)
        if cfg.verbose:
            print(self.plan_text(), flush=True)

    # ------------------------------------------------------------------ setup

    def _output_dir(self) -> Path:
        cfg = self.cfg
        if cfg.output_dir:
            return Path(cfg.output_dir).expanduser()
        name = cfg.name or f"{self.method}-{self.facts.name}-{time.strftime('%Y%m%d-%H%M%S')}"
        return P.train_dir() / "runs" / name

    def _pick_max_len(self, rows) -> int:
        if self.cfg.max_seq_len:
            return int(self.cfg.max_seq_len)
        probe = D.Formatter(
            self.tokenizer, max_len=1 << 20, chat=self.cfg.chat_template,
            completion_only=self.cfg.completion_only, template_kwargs=self.cfg.template_kwargs,
            system=self.cfg.system_prompt,
        )
        sample = rows[: min(len(rows), 512)]
        lens = []
        for r in sample:
            try:
                u = self.format_row(probe, r)
                lens.append(len(u))
            except Exception:
                continue
        if not lens:
            return 2048
        longest = int(np.percentile(lens, 99.5)) + 1
        extra = getattr(self.cfg, "max_completion_tokens", 0) or 0
        _ram, budget, _ws = P.machine_budget()
        top = P.pow2_cap(8192, budget / 1e9, 32768)
        return int(min(top, max(128, D._round_up(longest + extra, 64))))

    def format_row(self, fmt: D.Formatter, row):
        if self.formatter_fn is not None:
            return self.formatter_fn(row, fmt)
        return fmt.example(row)

    def build_units(self, rows) -> list:
        units = []
        bad = 0
        errors: list[str] = []
        for r in rows:
            try:
                u = self.format_row(self.fmt, r)
            except Exception as e:
                bad += 1
                if bad <= 3:
                    print(f"[train] skipping a row: {e}")
                    errors.append(str(e))
                continue
            if isinstance(u, D.Example) and sum(u.loss_mask[1:]) == 0:
                bad += 1
                if len(errors) < 3:
                    errors.append("row has no trained tokens")
                continue
            units.append(u)
        self._skipped = {"count": bad, "of": len(rows), "errors": errors}
        if bad:
            print(f"[train] skipped {bad} of {len(rows)} rows")
        if rows and not units:
            first = f"; first: {errors[0]}" if errors else ""
            raise ValueError(f"no usable rows left to train on ({bad} of {len(rows)} skipped{first})")
        if getattr(self.cfg, "packing", False) and units and isinstance(units[0], D.Example):
            units = D.pack(units, self.max_len)
        return units

    def _load_model(self):
        from .. import load

        t0 = time.perf_counter()
        kw = {"mode": self.plan.mode, "verbose": False}
        if self.plan.mode == "streamed" and self.plan.cache_gb:
            kw["cache_gb"] = self.plan.cache_gb
        self.model, _tok = load(self.facts.path, **kw)
        info = getattr(self.model, "_pagedmoe_info", {}) or {}
        self.streamed = info.get("mode", self.plan.mode) == "streamed"
        self.plan.mode = "streamed" if self.streamed else "resident"
        self.switches = train_experts.install(self.model) if self.streamed else []
        self.chunked_delta = train_linear_attn.install(self.model)
        self.fastgrad = train_fastgrad.install()
        loop = train_linear_attn.unpatched(self.model)
        if loop:
            print(
                f"[train] gated delta: {loop} layer kind(s) use mlx-lm's per-token loop; "
                "its backward holds every token's state (~10 MB/token/layer on big models)",
                flush=True,
            )
        mx.random.seed(self.cfg.seed)
        self.lora_report = train_lora.apply(self.model, self.cfg.lora, streamed=self.streamed)
        if self.facts.params:
            self.lora_report.total = self.facts.params
        self.model.train()
        self.cache = getattr(self.model, "_pagedmoe_cache", None)
        self.load_s = time.perf_counter() - t0
        bad = train_env.check()
        if bad:
            raise RuntimeError(f"engine left exact mode: {', '.join(bad)}")

    # --------------------------------------------------------------- progress

    def _phase(self, phase: str, text: str, **extra) -> None:
        """Start a progress phase: print it and tell the listeners."""
        self._ctx = {"phase": phase, "text": text, **extra}
        self._ctx_t0 = self._last_tick = self._last_print = time.perf_counter()
        if self.cfg.verbose:
            print(f"[train] {text}", flush=True)
        self._notify(dict(self._ctx, elapsed=0.0))

    def _notify(self, info: dict) -> None:
        for cb in self._listeners:
            try:
                cb.on_progress(self, info)
            except Exception:
                pass

    def _on_layer(self, part: str, i: int, L: int) -> None:
        """Executor hook, once per layer: throttled progress, and a terminal
        line every 30 s so a long step never looks hung."""
        now = time.perf_counter()
        done = L - i if part == "backward" else i + 1
        if now - self._last_tick < 1.0 and done < L:
            return
        self._last_tick = now
        elapsed = now - self._ctx_t0
        self._notify(dict(self._ctx, part=part, layer=done, layers=L, elapsed=elapsed))
        if self.cfg.verbose and now - self._last_print >= 30.0:
            self._last_print = now
            label = self._ctx.get("label") or self._ctx.get("phase", "")
            print(f"[train]   {label}: {part} layer {done}/{L}, {P.fmt_time(elapsed)} so far", flush=True)

    def _make_executor(self):
        kind = self.plan.executor
        common = {"encoder": self.encoder, "head_chunk": self.plan.head_chunk, "on_layer": self._on_layer}
        if kind == "layered":
            return LayeredExecutor(self.model, seed=self.cfg.seed, **common)
        return FusedExecutor(self.model, checkpoint=(kind == "checkpoint"), **common)

    def _make_optimizer(self):
        cfg = self.cfg
        name = cfg.optimizer.lower()
        if name == "adamw":
            return optim.AdamW(learning_rate=self.lr, betas=list(cfg.adam_betas), weight_decay=cfg.weight_decay)
        if name == "adam":
            return optim.Adam(learning_rate=self.lr, betas=list(cfg.adam_betas))
        if name == "sgd":
            return optim.SGD(learning_rate=self.lr, momentum=0.9, weight_decay=cfg.weight_decay)
        if name == "adafactor":
            return optim.Adafactor(learning_rate=self.lr)
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")

    def make_objective(self) -> O.Objective:
        return O.NLL()

    # ------------------------------------------------------------------ probe

    def _synthetic(self, tokens: int, micros: int = 1) -> list[Batch]:
        if self.encoder is not None:
            return []
        row = int(min(D._round_up(self.max_len), tokens))
        B = max(1, tokens // row)
        rng = np.random.default_rng(0)
        out = []
        for _ in range(micros):
            ids = rng.integers(0, max(2, self.facts.vocab - 1), size=(B, row + 1)).astype(np.int32)
            out.append(Batch(
                tokens=mx.array(ids[:, :-1]),
                targets=mx.array(ids[:, 1:]),
                mask=mx.ones((B, row), dtype=mx.float32),
            ))
        return out

    def _probe_once(self, tokens: int, micros: int = 1) -> dict | None:
        batches = self._synthetic(tokens, micros)
        if not batches:
            return None
        nll = O.NLL()
        nll.set_norm(O.StepNorm(tokens=float(sum(b.n_tokens for b in batches))))
        read0 = self.cache.bytes_read if self.cache is not None else 0
        mx.synchronize()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        self.executor.step(batches, nll)
        mx.synchronize()
        dt = time.perf_counter() - t0
        read = (self.cache.bytes_read - read0) if self.cache is not None else 0
        return {"tokens": int(sum(b.mask.size for b in batches)), "s": dt,
                "peak": int(mx.get_peak_memory()), "read": int(read)}

    def _probe_hint(self, n: int) -> str:
        """How long the probe should take, from the expert bytes a step reads."""
        f = self.facts
        if self.plan.mode != "streamed" or not f.expert_bytes:
            return ""
        disk = (self.plan.disk_gb_s or 6.0) * 1e9 * 0.5
        cache = (self.plan.cache_gb or 0) * 1e9
        step = 2 * max(f.expert_bytes - cache, f.expert_bytes * 0.25) / disk
        total = step * (n + 0.5)
        if total < 45:
            return ""
        return f" (~{P.fmt_time(total)}: each step reads the {f.expert_bytes / 1e9:.0f} GB of experts about twice)"

    def _probe_step(self, tokens: int, what: str, micros: int = 1) -> dict:
        k = self._ctx.get("k", 0) + 1
        n = max(self._probe_n, k)
        label = f"probe {k}/{n}"
        self._ctx = {"phase": "probe", "label": label, "what": what, "k": k, "n": n, "tokens": tokens * micros}
        self._ctx_t0 = self._last_tick = self._last_print = time.perf_counter()
        self._notify(dict(self._ctx, elapsed=0.0))
        r = self._probe_once(tokens, micros)
        if self.cfg.verbose and r is not None:
            read = f", read {r['read'] / 1e9:.0f} GB" if r["read"] else ""
            print(f"[train]   {label} ({what}): {r['tokens']} tokens in {r['s']:.1f}s, "
                  f"peak {r['peak'] / 1e9:.1f} GB{read}", flush=True)
        return r

    def _probe(self):
        cfg, plan = self.cfg, self.plan
        key = P.profile_key(self.facts, plan, cfg)
        prof = P.load_profile(key) if cfg.probe == "auto" else None
        if cfg.probe == "never" or self.encoder is not None:
            plan.probe = {"source": "skipped"}
            self._apply_batch_estimate(None)
            return
        if prof is None:
            small = int(max(D._round_up(min(self.max_len, 1024)), min(2048, plan.micro_tokens // 4)))
            n = 4 if plan.executor == "layered" else 3
            self._probe_n = n
            hint = self._probe_hint(n)
            self._phase("probe", f"probing memory and speed: {n} short steps on synthetic batches, "
                        f"nothing is trained{hint}", k=0, n=n)
            self._bound_buffers(start=True)
            self._probe_step(small, "warm-up")  # warm kernels and the cache
            a = self._probe_step(small, "memory and speed")
            b = self._probe_step(small * 2, "memory and speed")
            pts = [a, b]
            slope_mem = max(1.0, (b["peak"] - a["peak"]) / max(1, b["tokens"] - a["tokens"]))
            base_mem = a["peak"] - slope_mem * a["tokens"]
            slope_s = max(1e-7, (b["s"] - a["s"]) / max(1, b["tokens"] - a["tokens"]))
            base_s = max(0.0, a["s"] - slope_s * a["tokens"])
            prof = {"base_mem": base_mem, "slope_mem": slope_mem, "base_s": base_s,
                    "slope_s": slope_s, "read_per_step": b["read"], "probe_tokens": [a["tokens"], b["tokens"]]}
            if plan.executor == "layered":
                # b's tokens as two micro-batches: what one more micro-batch
                # costs (expert restacks, host syncs), which real steps pay
                c = self._probe_step(small, "cost of a micro-batch", micros=2)
                prof["micro_s"] = max(0.0, c["s"] - b["s"])
                # A layered step holds every layer boundary of its tokens, so
                # memory grows at least that fast. A smaller slope means the
                # expert cache moved between the probe points and hid it.
                floor = (self.facts.layers + 1) * self._boundary_bytes()
                if prof["slope_mem"] < floor:
                    prof["slope_mem"] = float(floor)
                    prof["base_mem"] = max(p["peak"] - floor * p["tokens"] for p in pts)
            prof["boundary_bytes"] = self._boundary_bytes()
            mt = self._micro_from(prof, measured=False)
            if not cfg.micro_batch_tokens and mt > 2 * b["tokens"]:
                # The line is extrapolated well past what was measured and
                # memory grows faster than linear at large sizes. Walk up in
                # doublings, each predicted from the steepest slope seen so
                # far, and stop before a size that would not fit: running the
                # extrapolated size directly is what spikes past the budget.
                limit = plan.budget_gb * 1e9 * P.FIT
                slope, last = prof["slope_mem"], b
                size = 2 * b["tokens"]
                while size <= mt:
                    if last["peak"] + 1.25 * slope * (size - last["tokens"]) > limit:
                        break
                    self._probe_n += 1
                    v = self._probe_step(size, "step up toward the planned micro-batch")
                    slope = max(slope, (v["peak"] - last["peak"]) / max(1, v["tokens"] - last["tokens"]))
                    pts.append(v)
                    last, size = v, size * 2
                if len(pts) > 2:
                    prof["slope_mem"] = slope
                    prof["base_mem"] = max(p["peak"] - slope * p["tokens"] for p in pts)
                    prof["slope_s"] = max(1e-7, (last["s"] - a["s"]) / max(1, last["tokens"] - a["tokens"]))
                    prof["base_s"] = max(0.0, a["s"] - prof["slope_s"] * a["tokens"])
                    prof["probe_tokens"] += [p["tokens"] for p in pts[2:]]
            if self.cache is not None:
                # what of base_mem was expert cache; the most it held is the
                # safe figure to take back out when the cache is refit
                prof["cache_bytes"] = int(max(self.cache.resident_bytes, self.cache.peak_resident_bytes))
            P.save_profile(key, prof)
            src = "measured"
        else:
            src = "cached"
            if cfg.verbose:
                print("[train] memory and speed from an earlier probe of this model on this Mac "
                      "(probe=always measures again)", flush=True)
        plan.probe = dict(prof, source=src)
        self._prof, self._prof_key = prof, key
        if not cfg.micro_batch_tokens:
            plan.micro_tokens = self._micro_from(prof)
            plan.reasons["micro"] = (
                f"probe: {prof['slope_mem'] / 1e6:.2f} MB/token over "
                f"{prof['base_mem'] / 1e9:.1f} GB base"
            )
        self._apply_batch_estimate(prof)
        self._fit_cache(prof)

    def _micro_from(self, prof, measured: bool = True) -> int:
        """Largest power-of-two micro-batch the measured memory line allows.

        measured=False is what the line alone says, before the probe has
        stepped up toward it."""
        plan = self.plan
        budget = plan.budget_gb * 1e9 * P.FIT
        fit = (budget - prof["base_mem"]) / prof["slope_mem"]
        need = D._round_up(self.max_len) * self.rows_per_unit
        cap = P.pow2_cap(32768 if plan.executor != "fused" else 16384, plan.budget_gb, 131072)
        seen = max(prof.get("probe_tokens") or [0]) if measured else 0
        if seen:
            # never more than one doubling past a measured size
            cap = min(cap, 2 * seen)
        mt = int(max(need, min(cap, fit)))
        return max(need, 1 << int(math.log2(max(1, mt))))

    def _fit_cache(self, prof):
        """Streamed: give the expert cache what the measured step leaves.

        base_mem was measured with the cache holding cache_bytes, and expert
        arrays are plain allocations on the training path (no slab), so peak
        memory moves one for one with the cache budget. The cache can grow
        into room the plan did not know about (weights that stay on disk) or
        shrink when long rows force a bigger micro-batch than the probe ran.
        """
        cfg, plan, f = self.cfg, self.plan, self.facts
        if plan.mode != "streamed" or self.cache is None or "cache_bytes" not in prof:
            return
        budget = plan.budget_gb * 1e9
        extra = self._stored_bytes() * max(0.0, plan.per_step * self.unit_tokens() - plan.micro_tokens)
        other = (prof["base_mem"] - prof["cache_bytes"] + prof.get("peak_fix", 0.0)
                 + prof["slope_mem"] * plan.micro_tokens + extra)
        floor = P.min_cache_bytes(f)
        if cfg.cache_gb:
            cache = int(cfg.cache_gb * 1e9)
        else:
            cache = int(min(P.max_cache_bytes(f), max(floor, budget * P.FIT - other)))
            was = self.cache.budget_bytes
            if abs(cache - was) > 64 << 20:
                with self.cache._lock:
                    self.cache.budget_bytes = cache
                    self.cache._evict_to_budget()
                mx.clear_cache()
                plan.reasons["cache"] = f"probe: {was / 1e9:.1f} -> {cache / 1e9:.1f} GB to fit the measured step"
            plan.cache_gb = round(cache / 1e9, 2)
        peak = other + cache
        plan.probe["step_peak"] = int(peak)
        if peak <= budget:
            return
        fits = (budget * P.FIT - (other - prof["slope_mem"] * plan.micro_tokens) - floor) / prof["slope_mem"]
        fits = int(max(0, fits) / max(1, self.rows_per_unit)) // 256 * 256
        msg = (
            f"a {plan.micro_tokens}-token micro-batch is projected at {peak / 1e9:.1f} GB, over the "
            f"{budget / 1e9:.1f} GB budget even with the expert cache at {cache / 1e9:.1f} GB. "
            f"max_seq_len {fits} or lower fits"
        )
        ram, _budget, ws = P.machine_budget()
        if ws and peak > ws - P.GiB and not cfg.micro_batch_tokens:
            raise RuntimeError(f"{msg}; past the GPU working set ({ws / 1e9:.1f} GB) a step can freeze the Mac")
        print(f"[train] warning: {msg}; expect memory pressure", flush=True)

    def _apply_batch_estimate(self, prof):
        """Streamed: grow the step until compute covers the fixed expert reads."""
        cfg, plan = self.cfg, self.plan
        mean = max(1.0, self.unit_tokens())
        if prof is not None:
            self.est_step = lambda tok, micros=1: (
                prof["base_s"] + prof["slope_s"] * tok + prof.get("micro_s", 0.0) * (micros - 1)
            )
        else:
            self.est_step = None
        if cfg.batch_size or prof is None or plan.mode != "streamed":
            return
        tokens = prof["base_s"] / max(1e-9, prof["slope_s"])
        # Every step pays a fixed cost per expert it touches, and a few
        # thousand tokens touch nearly all of them, so tokens per step is
        # worth more than expert cache: on a model many times the budget the
        # cache hits ~15% at best. Step tokens may take the cache down to its
        # floor to reach the target; _fit_cache gives the cache what is left.
        cache_now = prof.get("cache_bytes", 0)
        if cfg.cache_gb:
            keep = int(cfg.cache_gb * 1e9)
        else:
            keep = P.min_cache_bytes(self.facts) if cache_now else 0
        fixed = (prof["base_mem"] - cache_now + keep + prof.get("peak_fix", 0.0)
                 + prof["slope_mem"] * plan.micro_tokens)
        room = max(0.0, plan.budget_gb * 1e9 * P.FIT - fixed)
        top = 131072 * P.size_scale(plan.budget_gb)
        tokens = min(max(tokens, 16384), top, plan.micro_tokens + room / self._stored_bytes())
        cap = max(1, len(self.units) // 4)
        want = tokens / mean
        plan.per_step = int(max(1, min(cap, want)))
        # base_s fits to ~0 when the reads scale with tokens too (every
        # expert touched at any size), so only name it when it means something
        reads = f" to cover {prof['base_s']:.0f}s of expert reads" if prof["base_s"] >= 1 else ""
        if cap < want:
            plan.reasons["batch"] = (
                f"small dataset: wants ~{tokens / 1024:.0f}k tokens/step{reads}, capped at 4 steps per epoch"
            )
        else:
            plan.reasons["batch"] = f"~{tokens / 1024:.0f}k tokens/step{reads or ' to amortize expert reads'}"

    def _measured_boundary(self) -> int | None:
        ex = getattr(self, "executor", None)
        prof = getattr(self, "_prof", None) or {}
        b = getattr(ex, "boundary_bytes", None) or prof.get("boundary_bytes")
        return int(b) if b else None

    def _boundary_bytes(self) -> int:
        """Bytes per token at one layer boundary. Measured from what the model
        passes its layers when it can be: hyper-connection models carry several
        hidden states there, which hidden_size alone undercounts."""
        return self._measured_boundary() or self.facts.hidden * max(1, self.facts.streams) * 2

    def _stored_bytes(self) -> float:
        """Bytes a streamed step keeps per token beyond one micro-batch: every
        layer's input plus the gradient carried back. From the config alone it
        gets a margin, since the width there is a guess."""
        if self._measured_boundary():
            return (self.facts.layers + 2) * self._boundary_bytes()
        return self.facts.layers * self._boundary_bytes() * 1.25

    def _check_fit(self, peak: float) -> str | None:
        """Streamed: after a real step, hold the measured peak to the plan.

        The probe's memory line can miss by a few GB on some models (what the
        expert cache held at the instant of the peak is not observable). The
        first real step measures it. An overshoot comes out of the expert
        cache, and the correction is saved with the probe profile so the next
        run on this model plans it in from the start.
        """
        plan, c = self.plan, self.cache
        if plan.mode != "streamed" or c is None or self.cfg.cache_gb:
            return None
        target = plan.budget_gb * 1e9 * P.FIT
        over = peak - target
        if over < 256 << 20:
            return None
        floor = P.min_cache_bytes(self.facts)
        was = c.budget_bytes
        new = int(max(floor, was - over - (256 << 20)))
        if new >= was - (64 << 20):
            return None
        with c._lock:
            c.budget_bytes = new
            c._evict_to_budget()
        mx.clear_cache()
        plan.cache_gb = round(new / 1e9, 2)
        msg = (f"step peaked at {peak / 1e9:.1f} GB, planned {target / 1e9:.1f}: "
               f"expert cache {was / 1e9:.1f} -> {new / 1e9:.1f} GB")
        plan.reasons["cache"] = msg
        prof, key = getattr(self, "_prof", None), getattr(self, "_prof_key", None)
        if prof is not None and key:
            prof["peak_fix"] = float(prof.get("peak_fix", 0.0)) + float(was - new)
            P.save_profile(key, prof)
        print(f"[train] {msg}. Saved for the next run on this model.", flush=True)
        return msg

    # ------------------------------------------------------------- plan text

    def unit_tokens(self) -> float:
        """Expected trained tokens per unit, for the plan's estimate."""
        return self.data_summary["mean"] * self.rows_per_unit

    def _schedule(self, units, per_step: int, seed: int, rows_per_unit: int | None = None) -> D.Schedule:
        p = self.plan
        cost = None
        pr = p.probe or {}
        if pr.get("micro_s") and pr.get("slope_s"):
            # what one more micro-batch costs, in tokens of compute
            cost = min(16384.0, pr["micro_s"] / pr["slope_s"])
        return D.Schedule(units, per_step, p.micro_tokens, seed,
                          rows_per_unit=self.rows_per_unit if rows_per_unit is None else rows_per_unit,
                          max_rows=p.max_rows, micro_cost=cost)

    def _step_shape(self) -> tuple[float, float]:
        """Mean padded tokens and micro-batches per step, from the real schedule."""
        p = self.plan
        sched = self._schedule(self.units, p.per_step, self.cfg.seed)
        n = min(sched.steps_per_epoch, 64)
        tok = mic = 0
        for k in range(n):
            for m in sched.step(0, k):
                rows = len(m) * self.rows_per_unit
                tok += rows * D._round_up(max(1, max(len(u) for u in m) - 1))
                mic += 1
        if not n or not mic:
            t = p.per_step * self.unit_tokens()
            return t, max(1, math.ceil(t / p.micro_tokens))
        return tok / n, mic / n

    def estimate(self) -> dict | None:
        """Probe-based speed estimate for the planned step, or None without a probe."""
        if self.est_step is None:
            return None
        p, f = self.plan, self.facts
        per_step_tokens, micro = self._step_shape()
        step_s = self.est_step(per_step_tokens, micro)
        if p.executor != "layered":
            step_s = self.est_step(per_step_tokens / micro) * micro
        out = {
            "step_tokens": per_step_tokens,
            "step_s": step_s,
            "tok_s": per_step_tokens / max(1e-6, step_s),
            "total_s": step_s * self.total_steps,
            "source": p.probe.get("source"),
        }
        if p.mode == "streamed" and p.probe.get("read_per_step"):
            out["read_gb"] = p.probe["read_per_step"] / 1e9
            out["passes"] = p.probe["read_per_step"] / max(1, f.expert_bytes)
        return out

    def plan_info(self) -> dict:
        """The plan as data: what plan_text prints, for tools and UIs."""
        ram, budget, ws = P.machine_budget()
        lora = None
        if self.lora_report is not None:
            r = self.lora_report
            lora = {"trainable": r.trainable, "total": r.total, "layers": r.layers,
                    "keys": list(r.keys), "line": r.line()}
        return {
            "method": self.method,
            "unit_kind": self.unit_kind,
            "rows_per_unit": self.rows_per_unit,
            "facts": dict(self.facts.__dict__, backbone_bytes=self.facts.backbone_bytes),
            "machine": {"ram": ram, "budget": budget, "working_set": ws},
            "plan": self.plan.to_dict(),
            "plan_only": self.plan_only,
            "data": dict(self.data_summary, rows=self.n_rows, skipped=self.skipped,
                         eval_units=len(self.eval_units)),
            "max_seq_len": self.max_len,
            "steps_per_epoch": self.schedule.steps_per_epoch,
            "total_steps": self.total_steps,
            "lr": self.lr,
            "lora": lora,
            "estimate": self.estimate(),
            "output_dir": str(self.out),
            "text": self.plan_text(),
        }

    def plan_text(self) -> str:
        f, p, s = self.facts, self.plan, self.data_summary
        per_step_tokens = p.per_step * self.unit_tokens()
        lines = [
            "",
            "paged training plan",
            f"  method    {self.method}",
            f"  model     {f.name} ({f.model_type}), {f.layers} layers, "
            f"{f.experts} experts / {f.active} active",
            f"  size      {f.total_bytes / 1e9:.1f} GB (experts {f.expert_bytes / 1e9:.1f} GB, "
            + (
                f"backbone {f.backbone_bytes / 1e9:.1f} GB, {f.skip_bytes / 1e9:.1f} GB never resident)"
                if f.skip_bytes and p.mode == "streamed"
                else f"backbone {(f.total_bytes - f.expert_bytes) / 1e9:.1f} GB)"
            ),
            f"  memory    {p.budget_gb:.1f} GB training budget"
            + (f"  ({p.reasons['budget']})" if p.reasons.get("budget") else ""),
            f"  mode      {p.mode}  ({p.reasons.get('mode', '')})",
            f"  executor  {p.executor}  ({p.reasons.get('executor', '')})",
        ]
        if p.mode == "streamed":
            why = p.reasons.get("cache")
            lines.append(f"  cache     {p.cache_gb} GB of experts held between reads" + (f"  ({why})" if why else ""))
        lines += [
            f"  lora      {self.lora_report.line() if self.lora_report else 'added when the model loads'}",
            f"  data      {s['n']} {self.unit_kind}s, mean {s['mean']:.0f} tokens "
            f"(p95 {s['p95']:.0f}, max {s['max']}), max_seq_len {self.max_len}",
            f"  batch     {p.per_step} {self.unit_kind}s/step "
            f"({'up to ' if self.method == 'grpo' else '~'}{P.fmt_tokens(per_step_tokens)} tokens), "
            f"micro-batches <= {p.micro_tokens} tokens  ({p.reasons.get('batch', '')})",
            f"  steps     {self.schedule.steps_per_epoch}/epoch, {self.total_steps} total, lr {self.lr:g} "
            f"{self.cfg.lr_schedule}",
        ]
        est = self.estimate()
        if est is not None:
            lines.append(
                f"  estimate  ~{P.fmt_tokens(est['tok_s'])} tok/s, ~{est['step_s']:.1f} s/step, "
                f"~{P.fmt_time(est['total_s'])} total  (probe: {est['source']})"
            )
            if "read_gb" in est:
                lines.append(
                    f"  reads     ~{est['read_gb']:.0f} GB/step "
                    f"({est['passes']:.1f}x the expert mass)"
                )
        if self.method == "grpo":
            lines.append("  note      estimate assumes every completion runs to max_completion_tokens; "
                         "generation time is measured live")
        lines.append(f"  output    {self.out}")
        return "\n".join(lines) + "\n"

    # ---------------------------------------------------------------- batches

    def next_batches(self, micro_units: list) -> list[Batch] | None:
        """Micro-batches for one step. Return None to skip the step."""
        kind = "pair" if self.unit_kind == "pair" else "example"
        return [D.collate(m, self.fmt.pad_id, kind) for m in micro_units]

    def step_norm(self, batches: list[Batch]) -> O.StepNorm:
        tokens = rows = pairs = chosen = 0.0
        max_len = 1.0
        for b in batches:
            m = b.mask
            tokens += float(m.sum().item())
            rows += float(m.shape[0])
            per_row = m.sum(axis=-1)
            max_len = max(max_len, float(per_row.max().item()))
            p = int(b.meta.get("pairs", 0))
            if p:
                pairs += p
                chosen += float(per_row[:p].sum().item())
        return O.StepNorm(
            tokens=max(1.0, tokens), rows=max(1.0, rows), pairs=max(1.0, pairs),
            chosen_tokens=max(1.0, chosen), max_len=max_len,
        )

    # ------------------------------------------------------------------ loop

    def lr_at(self, step: int) -> float:
        cfg = self.cfg
        warm = int(cfg.warmup_ratio * self.total_steps)
        if warm and step < warm:
            return self.lr * (step + 1) / warm
        if cfg.lr_schedule == "constant":
            return self.lr
        t = (step - warm) / max(1, self.total_steps - warm)
        lo = self.lr * cfg.min_lr_ratio
        if cfg.lr_schedule == "linear":
            return lo + (self.lr - lo) * (1 - t)
        return lo + (self.lr - lo) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    def update(self, batches: list[Batch], lr: float):
        """One optimizer update over `batches`. Returns (loss, metrics, grad norm)."""
        self.objective.set_norm(self.step_norm(batches))
        loss, grads, enc_grads, metrics = self.executor.step(batches, self.objective)
        if self.cfg.max_grad_norm:
            grads, gn = optim.clip_grad_norm(grads, self.cfg.max_grad_norm)
        else:
            gn = mx.sqrt(sum((g * g).sum() for _, g in tree_flatten(grads)))
        self.optimizer.learning_rate = lr
        self.optimizer.update(self.model, grads)
        mx.eval(self.model.trainable_parameters(), self.optimizer.state)
        if self.enc_optimizer is not None and enc_grads:
            from mlx.utils import tree_unflatten

            self.enc_optimizer.learning_rate = lr
            self.enc_optimizer.update(self.encoder, tree_unflatten(list(enc_grads.items())))
            mx.eval(self.encoder.trainable_parameters(), self.enc_optimizer.state)
        if hasattr(self.objective, "end_step"):
            self.objective.end_step()
        return loss, metrics, float(gn.item() if hasattr(gn, "item") else gn)

    def _write_run(self, summary: dict | None) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        rec = {"config": self.cfg.to_dict(), "plan": self.plan.to_dict(), "summary": summary,
               "facts": self.facts.__dict__, "cli": getattr(self, "cli_extras", {})}
        (self.out / "run.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")

    def train(self):
        cfg, st = self.cfg, self.state
        self._write_run(None)
        self._guard_reset()
        self._install_signals()
        for cb in self.callbacks:
            cb.on_train_begin(self)
        self.before_train()
        _app_hint()
        rate = Rate()
        t_run = time.perf_counter()
        logs: dict = {}
        try:
            while st["step"] < self.total_steps and not self._stop:
                if st["k"] >= self.schedule.steps_per_epoch:
                    st["epoch"] += 1
                    st["k"] = 0
                micro = self.schedule.step(st["epoch"], st["k"])
                t0 = time.perf_counter()
                read0, cstats0 = self._cache_marks()
                train_experts.COUNTERS.reset()
                mx.reset_peak_memory()
                lr = self.lr_at(st["step"])
                batches = self.next_batches(micro)
                st["k"] += 1
                if batches is None:
                    st["step"] += 1
                    continue
                n = st["step"] + 1
                self._ctx = {"phase": "step", "label": f"step {n}/{self.total_steps}", "step": n,
                             "total": self.total_steps}
                self._ctx_t0 = self._last_tick = self._last_print = time.perf_counter()
                try:
                    loss, metrics, gn = self.update(batches, lr)
                except KeyboardInterrupt:
                    st["k"] -= 1  # this batch was not trained; the next run starts with it
                    raise
                extra = self.after_update(batches) or {}
                dt = time.perf_counter() - t0
                rate.add(dt)
                ntok = int(sum(b.n_tokens for b in batches))
                self._last_batches = batches
                st["tokens"] += ntok
                st["step"] += 1
                st["elapsed"] += dt
                logs = self._step_logs(loss, metrics, gn, lr, dt, ntok, rate, read0, cstats0)
                logs.update(extra)
                self._bound_buffers()
                refit = self._check_fit(logs.get("peak_gb", 0.0) * 1e9)
                if refit:
                    logs["memory_refit"] = refit
                guard = self._guard_check()
                if guard:
                    logs["memory_guard"] = guard
                for cb in self.callbacks:
                    cb.on_step_end(self, st["step"], logs)
                if cfg.eval_steps and self.eval_units and st["step"] % cfg.eval_steps == 0:
                    self.evaluate()
                if self._save_due():
                    self.save_checkpoint()
        except KeyboardInterrupt:
            print("\n[train] interrupted; saving", flush=True)
            self._stop = True
        finally:
            self._restore_signals()
        if self._saved_step == st["step"]:
            path = self.out / "checkpoints" / f"step-{st['step']:06d}"
        else:
            path = self.save_checkpoint(final=not self._stop)
        if self.eval_units and not self._stop:
            logs["eval"] = self.evaluate()
        summary = {
            "steps": st["step"],
            "tokens": st["tokens"],
            "seconds": round(time.perf_counter() - t_run, 1),
            "adapter": str(self.out),
            "stopped_early": self._stop,
        }
        self._write_run(summary)
        for cb in self.callbacks:
            cb.on_train_end(self, summary)
        print(f"[train] done: {st['step']} steps, {P.fmt_tokens(st['tokens'])} loss tokens, "
              f"adapter in {self.out} (last checkpoint {path.name})", flush=True)
        return summary

    # ---------------------------------------------------------- memory guard

    def _guard_reset(self) -> None:
        from .. import mempressure

        self._guard = {"comp0": mempressure.compressor_gb(), "cool": 0, "stuck": False}
        self._bound_buffers(start=True)

    def _bound_buffers(self, start: bool = False) -> None:
        """Keep MLX's pool of freed buffers inside the budget.

        Resident loads leave the pool uncapped (right for inference). Training
        frees and reallocates big activations of changing shapes every step,
        so the pool grows on top of the step's peak and pushes the Mac into
        the compressor and swap while the step itself is inside the budget.
        The pool gets what the measured peak leaves. Streamed loads already
        cap it in the loader.
        """
        if self.plan.mode != "resident" or not hasattr(mx, "set_cache_limit"):
            return
        budget = self.plan.budget_gb * 1e9
        used = budget * P.FIT if start else mx.get_peak_memory()
        try:
            mx.set_cache_limit(int(max(1 << 30, budget - used)))
        except Exception:
            pass

    def _guard_check(self) -> str | None:
        """Between steps: give memory back when the Mac runs short.

        The plan fits the budget it measured, but other apps can grow during a
        long run. When the kernel reports pressure, or the compressor has grown
        well past where it was, the expert cache shrinks first (streamed), then
        the micro-batch halves. Neither changes what a step computes: the same
        rows per step, the same gradients, only more reads or more passes.
        """
        if not self.cfg.memory_guard or os.environ.get("PAGED_MOE_SIMULATE_RAM_GB"):
            return None
        from .. import mempressure

        g = self._guard
        if g["cool"] > 0:
            g["cool"] -= 1
            return None
        lv = mempressure.level()
        comp = mempressure.compressor_gb()
        grew = comp - g["comp0"]
        p = self.plan
        limit = max(4.0, 0.12 * p.budget_gb)
        if not (lv >= mempressure.CRITICAL or grew >= limit or (lv >= mempressure.WARN and grew >= 1.5)):
            return None
        give = max(grew, 0.1 * p.budget_gb) * 1e9
        msg = ""
        c = self.cache
        if self.streamed and c is not None and not self.cfg.cache_gb:
            was = c.budget_bytes
            new = int(max(P.min_cache_bytes(self.facts), was - give))
            if was - new > 256 << 20:
                with c._lock:
                    c.budget_bytes = new
                    c._evict_to_budget()
                p.cache_gb = round(new / 1e9, 2)
                msg = f"expert cache {was / 1e9:.1f} -> {new / 1e9:.1f} GB"
        if not msg:
            need = D._round_up(self.max_len) * self.rows_per_unit
            if p.micro_tokens // 2 >= need:
                old = p.micro_tokens
                p.micro_tokens //= 2
                self.schedule.micro_tokens = p.micro_tokens
                msg = f"micro-batches {old} -> {p.micro_tokens} tokens"
        mx.clear_cache()
        g["comp0"], g["cool"] = comp, 2
        why = f"memory pressure ({mempressure.name(lv)}, compressor +{grew:.1f} GB)"
        if msg:
            p.reasons["guard"] = msg
            print(f"[train] {why}: {msg}. Same rows per step, same gradients.", flush=True)
            return msg
        if not g["stuck"]:
            g["stuck"] = True
            print(f"[train] warning: {why} and nothing left to give back. Close other apps "
                  "or lower max_seq_len.", flush=True)
        return None

    def before_train(self) -> None:
        """Hook: runs once before the first step (reference scoring, etc)."""

    def after_update(self, batches) -> dict | None:
        """Hook: extra per-step logs."""
        return None

    def _cache_marks(self):
        if self.cache is None:
            return 0, None
        return self.cache.bytes_read, (self.cache.hits, self.cache.misses, self.cache.disk_wait_s)

    def _step_logs(self, loss, metrics, gn, lr, dt, ntok, rate, read0, c0) -> dict:
        # tok/s counts every position processed (what the estimate predicts);
        # loss_tokens is how many of them carried loss
        seq = sum(b.shape[0] * b.shape[1] for b in self._last_batches)
        logs = {"loss": float(loss), "lr": float(lr), "grad_norm": gn, "step_s": dt,
                "tokens": seq, "loss_tokens": ntok, "tok_s": seq / max(1e-9, dt),
                "peak_gb": mx.get_peak_memory() / 1e9}
        logs.update({k: float(v) for k, v in metrics.items()})
        remaining = self.total_steps - self.state["step"]
        logs["eta_s"] = (rate.value or dt) * remaining
        logs["timing"] = {k: round(v, 2) for k, v in self.executor.timing.items()}
        if self.cache is not None and c0 is not None:
            read = self.cache.bytes_read - read0
            hits = self.cache.hits - c0[0]
            miss = self.cache.misses - c0[1]
            logs["read_gb"] = read / 1e9
            logs["passes"] = read / max(1, self.facts.expert_bytes)
            logs["cache_hit"] = hits / max(1, hits + miss)
            logs["disk_wait_s"] = self.cache.disk_wait_s - c0[2]
            logs["experts"] = train_experts.COUNTERS.snapshot()
        return logs

    def _save_due(self) -> bool:
        cfg, st = self.cfg, self.state
        if cfg.save_steps and st["step"] % cfg.save_steps == 0:
            return True
        if cfg.save_minutes and time.time() - self._last_save_t >= cfg.save_minutes * 60:
            return True
        return False

    # ----------------------------------------------------------- checkpoints

    def save_checkpoint(self, final: bool = False) -> Path:
        st = self.state
        ck = self.out / "checkpoints" / f"step-{st['step']:06d}"
        extra = {"base_model": self.facts.path, "method": self.method}
        train_lora.save(self.model, ck, extra=extra)
        mx.save_safetensors(str(ck / "optimizer.safetensors"), dict(tree_flatten(self.optimizer.state)))
        if self.encoder is not None:
            mx.save_safetensors(str(ck / "encoder.safetensors"), dict(tree_flatten(self.encoder.parameters())))
            if self.enc_optimizer is not None and self.enc_optimizer.state:
                mx.save_safetensors(str(ck / "encoder_optimizer.safetensors"),
                                    dict(tree_flatten(self.enc_optimizer.state)))
        state = dict(st, total_steps=self.total_steps, seed=self.cfg.seed,
                     per_step=self.plan.per_step, micro_tokens=self.plan.micro_tokens,
                     data_hash=self._data_hash(), data_key=self._data_key(), time=time.time())
        if hasattr(self.objective, "z0"):
            state["objective"] = {"z0": float(self.objective.z0)}
        (ck / "trainer_state.json").write_text(json.dumps(state, indent=2) + "\n")
        (self.out / "checkpoints" / "latest").write_text(ck.name + "\n")
        train_lora.save(self.model, self.out, extra=extra)
        if self.encoder is not None:
            mx.save_safetensors(str(self.out / "encoder.safetensors"),
                                dict(tree_flatten(self.encoder.parameters())))
        self._prune_checkpoints(keep_path=ck)
        self._last_save_t = time.time()
        self._saved_step = st["step"]
        for cb in self.callbacks:
            cb.on_save(self, st["step"], ck)
        return ck

    def _prune_checkpoints(self, keep_path: Path | None = None):
        """Keep the newest saves by save time, not by step number: after a
        resume from an older checkpoint the fresh save has the lower step."""
        keep = max(1, int(self.cfg.keep_checkpoints))
        root = self.out / "checkpoints"

        def saved(p: Path) -> float:
            st = p / "trainer_state.json"
            return st.stat().st_mtime if st.exists() else p.stat().st_mtime

        cks = sorted((p for p in root.glob("step-*") if p.is_dir()), key=saved)
        import shutil

        for p in cks[:-keep]:
            if keep_path is None or p.resolve() != Path(keep_path).resolve():
                shutil.rmtree(p, ignore_errors=True)

    def _data_hash(self) -> str:
        """Short fingerprint kept for runs saved before data_key existed."""
        h = hashlib.sha1()
        for u in self.units[:2000]:
            ids = u.ids if hasattr(u, "ids") else u.chosen.ids + u.rejected.ids
            h.update(np.asarray(ids[:64], np.int64).tobytes())
        h.update(str(len(self.units)).encode())
        return h.hexdigest()[:16]

    def _data_key(self) -> str:
        """Every token and mask of the train and eval units, plus what shaped
        them. Reference scores and resume checks key on this."""
        if getattr(self, "_key", None):
            return self._key
        h = hashlib.sha1()
        c = self.cfg
        h.update(json.dumps([self.max_len, c.completion_only, c.chat_template, c.template_kwargs,
                             c.system_prompt, self.method], sort_keys=True, default=str).encode())
        for part in (self.units, self.eval_units):
            h.update(str(len(part)).encode())
            for u in part:
                exs = [u.chosen, u.rejected] if isinstance(u, D.Pair) else [u]
                for ex in exs:
                    h.update(np.asarray(ex.ids, np.int64).tobytes())
                    mask = getattr(ex, "loss_mask", None)
                    if mask is not None:
                        h.update(np.asarray(mask, np.int8).tobytes())
                    if "label" in getattr(ex, "meta", {}):
                        h.update(str(ex.meta["label"]).encode())
        self._key = h.hexdigest()[:20]
        return self._key

    def _resume(self, spec):
        root = self.out / "checkpoints"
        if spec is True or spec == "latest":
            latest = root / "latest"
            if not latest.exists():
                print(f"[train] nothing to resume in {root}; starting fresh")
                return
            ck = root / latest.read_text().strip()
        else:
            ck = Path(spec).expanduser()
            if (ck / "checkpoints" / "latest").exists():
                self._move_out(ck)
                ck = ck / "checkpoints" / (ck / "checkpoints" / "latest").read_text().strip()
        st = json.loads((ck / "trainer_state.json").read_text())
        same = st["data_key"] == self._data_key() if st.get("data_key") else st.get("data_hash") == self._data_hash()
        if not same:
            print("[train] warning: data changed since this checkpoint; the order will differ")
        self._check_adapter(ck)
        train_lora.load_weights(self.model, ck)
        # The micro-batch is not restored: it only bounds memory, and this
        # run's probe sized it (and the expert cache) for this Mac as it is now.
        self.plan.per_step = st.get("per_step", self.plan.per_step)
        self.schedule = self._schedule(self.units, self.plan.per_step, self.cfg.seed)
        self.total_steps = st.get("total_steps", self.total_steps) if not self.cfg.max_steps else self.total_steps
        self._load_optimizer(self.optimizer, self.model, ck / "optimizer.safetensors")
        if self.encoder is not None and (ck / "encoder.safetensors").exists():
            self.encoder.load_weights(str(ck / "encoder.safetensors"))
            if self.enc_optimizer is not None:
                self._load_optimizer(self.enc_optimizer, self.encoder, ck / "encoder_optimizer.safetensors")
        for k in ("step", "epoch", "k", "tokens", "elapsed"):
            self.state[k] = st.get(k, self.state[k])
        if hasattr(self.objective, "z0") and "z0" in (st.get("objective") or {}):
            self.objective.z0 = float(st["objective"]["z0"])
        self._trim_metrics(self.state["step"])
        print(f"[train] resumed from {ck} at step {self.state['step']}", flush=True)

    def _move_out(self, out: Path) -> None:
        """Point the run (and its metrics log) at another folder."""
        self.out = Path(out)
        for cb in self.callbacks:
            if isinstance(cb, JsonlLog):
                cb.path = self.out / "metrics.jsonl"

    def _check_adapter(self, ck: Path) -> None:
        try:
            saved = json.loads((Path(ck) / "adapter_config.json").read_text())
        except (OSError, ValueError):
            return
        now = train_lora.adapter_config(self.model)
        a, b = saved.get("lora_parameters") or {}, now.get("lora_parameters") or {}
        diff = [k for k in ("rank", "keys") if k in a and a.get(k) != b.get(k)]
        if saved.get("num_layers") != now.get("num_layers"):
            diff.append("num_layers")
        if saved.get("fine_tune_type", "lora") != now.get("fine_tune_type", "lora"):
            diff.append("dora")
        if diff:
            raise ValueError(f"{ck} was saved with a different LoRA ({', '.join(diff)}); "
                             "resume with the run's original lora settings")

    def _trim_metrics(self, step: int) -> None:
        """Drop steps logged after the checkpoint being resumed (they are
        about to be trained again), so the log has one entry per step."""
        p = self.out / "metrics.jsonl"
        if not p.exists():
            return
        keep = []
        for line in p.read_text().splitlines():
            try:
                if int(json.loads(line).get("step", 0)) <= step:
                    keep.append(line)
            except (ValueError, TypeError, AttributeError):
                keep.append(line)
        p.write_text("".join(x + "\n" for x in keep))

    @staticmethod
    def _load_optimizer(opt, module, path: Path):
        if not path.exists():
            return
        saved = mx.load(str(path))
        opt.init(module.trainable_parameters())

        def pick(p, v):
            return saved.get(p, v)

        opt.state = tree_map_with_path(pick, opt.state)
        mx.eval(opt.state)

    # ------------------------------------------------------------------ eval

    def score_batches(self, batches: list[Batch], group_tokens: int | None = None) -> list[mx.array]:
        """Token logprobs for many batches, grouped so streamed reads amortize."""
        cap = group_tokens or int((262144 if self.streamed else 65536) * P.size_scale(self.plan.budget_gb))
        out: list[mx.array] = []
        group: list[Batch] = []
        size = 0
        for b in batches:
            n = int(b.mask.size)
            if group and size + n > cap:
                out += self.executor.logps_many(group)
                group, size = [], 0
            group.append(b)
            size += n
        if group:
            out += self.executor.logps_many(group)
        return out

    def eval_batches(self) -> list[Batch]:
        sched = self._schedule(self.eval_units, len(self.eval_units), 0)
        micro = sched.step(0, 0)
        if self.cfg.eval_batches:
            micro = micro[: int(self.cfg.eval_batches)]
        return self.next_eval_batches(micro)

    def next_eval_batches(self, micro):
        return DefaultBatches.collate(self, micro)

    def evaluate(self) -> dict:
        batches = self.eval_batches()
        if not batches:
            return {}
        self._ctx = {"phase": "eval", "label": "eval"}
        self._ctx_t0 = self._last_tick = self._last_print = time.perf_counter()
        self.model.eval()
        try:
            lps = self.score_batches(batches)
        finally:
            self.model.train()
        self.objective.set_norm(self.step_norm(batches))
        total = 0.0
        sums: dict = {}
        # running statistics (KTO's z0) learn from training batches only
        acc = list(self.objective._acc) if hasattr(self.objective, "_acc") else None
        for lp, b in zip(lps, batches):
            loss, m = self.objective(lp, b)
            total += float(loss.item())
            for k, v in m.items():
                sums[k] = sums.get(k, 0.0) + float(v.item())
        if acc is not None:
            self.objective._acc = acc
        logs = {"loss": total, **sums}
        for cb in self.callbacks:
            cb.on_evaluate(self, self.state["step"], logs)
        return logs

    # --------------------------------------------------------------- signals

    def _install_signals(self):
        self._old_term = signal.getsignal(signal.SIGTERM)

        def on_term(signum, frame):
            print("\n[train] SIGTERM; saving after this step", flush=True)
            self._stop = True

        try:
            signal.signal(signal.SIGTERM, on_term)
        except ValueError:
            pass

    def _restore_signals(self):
        try:
            signal.signal(signal.SIGTERM, self._old_term)
        except (ValueError, TypeError):
            pass


class DefaultBatches:
    @staticmethod
    def collate(trainer: Trainer, micro):
        kind = "pair" if trainer.unit_kind == "pair" else "example"
        return [D.collate(m, trainer.fmt.pad_id, kind) for m in micro]


# ===================================================================== methods


class SFTTrainer(Trainer):
    """Supervised fine-tuning: chat, prompt / completion, raw text or tokens."""

    method = "sft"


class CPTTrainer(Trainer):
    """Continued pretraining on raw text, packed into max_seq_len windows."""

    method = "cpt"


class _PairTrainer(Trainer):
    rows_per_unit = 2
    unit_kind = "pair"

    def format_row(self, fmt, row):
        if self.formatter_fn is not None:
            return self.formatter_fn(row, fmt)
        return fmt.pair(row)

    def build_units(self, rows):
        units = super().build_units(rows)
        return [u for u in units if sum(u.chosen.loss_mask[1:]) and sum(u.rejected.loss_mask[1:])]


class ORPOTrainer(_PairTrainer):
    """ORPO: SFT on the chosen answer plus an odds-ratio preference term. No reference."""

    method = "orpo"

    def make_objective(self):
        return O.ORPO(lam=self.cfg.beta)


class SimPOTrainer(_PairTrainer):
    method = "simpo"

    def make_objective(self):
        c = self.cfg
        return O.SimPO(beta=c.beta, gamma=c.gamma, sft_weight=c.sft_weight)


class CPOTrainer(_PairTrainer):
    method = "cpo"

    def make_objective(self):
        return O.CPO(beta=self.cfg.beta, nll_weight=self.cfg.nll_weight)


class _ReferenceMixin:
    """Scores every training row once under the base model (LoRA off)."""

    def _ref_rows(self, unit):
        if isinstance(unit, D.Pair):
            return [unit.chosen, unit.rejected]
        return [unit]

    def before_train(self):
        units = list(self.units) + list(self.eval_units)
        rows = [r for u in units for r in self._ref_rows(u)]
        todo = [r for r in rows if "ref_seq" not in r.meta]
        if not todo:
            return
        cache = self.out / "reference_logps.npy"
        key = self._data_key()
        meta = self.out / "reference_logps.json"
        if cache.exists() and meta.exists() and json.loads(meta.read_text()).get("key") == key:
            vals = np.load(cache)
            if len(vals) == len(rows):
                for r, v in zip(rows, vals):
                    r.meta["ref_seq"] = float(v)
                print(f"[train] reference logprobs loaded ({len(rows)} rows)", flush=True)
                return
        t0 = time.perf_counter()
        self._phase("reference", f"scoring {len(rows)} rows under the base model (once) ...")
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]), reverse=True)
        micro: list[list[int]] = []
        cur: list[int] = []
        cur_max = 0
        for i in order:
            n = D._round_up(len(rows[i]) - 1)
            if cur and ((len(cur) + 1) * max(cur_max, n) > self.plan.micro_tokens or len(cur) >= self.plan.max_rows):
                micro.append(cur)
                cur, cur_max = [], 0
            cur.append(i)
            cur_max = max(cur_max, n)
        if cur:
            micro.append(cur)
        batches = [D.collate([rows[i] for i in m], self.fmt.pad_id) for m in micro]
        self.model.eval()
        try:
            with train_lora.disabled(self.model):
                lps = self.score_batches(batches)
        finally:
            self.model.train()
        vals = np.zeros(len(rows), np.float32)
        for m, b, lp in zip(micro, batches, lps):
            s = np.asarray((lp * b.mask).sum(axis=-1).astype(mx.float32))
            for i, v in zip(m, s):
                vals[i] = v
        for r, v in zip(rows, vals):
            r.meta["ref_seq"] = float(v)
        self.out.mkdir(parents=True, exist_ok=True)
        np.save(cache, vals)
        meta.write_text(json.dumps({"key": key}) + "\n")
        print(f"[train] reference scored in {P.fmt_time(time.perf_counter() - t0)}", flush=True)


class DPOTrainer(_ReferenceMixin, _PairTrainer):
    method = "dpo"

    def make_objective(self):
        c = self.cfg
        return O.DPO(beta=c.beta, loss_type=c.loss_type, label_smoothing=c.label_smoothing)


class KTOTrainer(_ReferenceMixin, Trainer):
    method = "kto"

    def make_objective(self):
        c = self.cfg
        return O.KTO(beta=c.beta, desirable_weight=c.desirable_weight,
                     undesirable_weight=c.undesirable_weight)


class GRPOTrainer(Trainer):
    """GRPO: sample a group per prompt, score with reward functions, update.

    reward_funcs: callables, see rewards.py for the two accepted signatures.
    """

    method = "grpo"
    unit_kind = "prompt"

    def __init__(self, config=None, *, reward_funcs=None, **kw):
        if not reward_funcs:
            raise ValueError("GRPOTrainer needs reward_funcs")
        self.reward_funcs = list(reward_funcs)
        cfg = config
        ng = kw.get("num_generations", cfg.num_generations if cfg is not None else 8)
        self.rows_per_unit = int(ng)
        super().__init__(config, **kw)
        self._roll_logs: dict = {}

    def format_row(self, fmt, row):
        if self.formatter_fn is not None:
            return self.formatter_fn(row, fmt)
        if "prompt" not in row:
            raise ValueError("GRPO rows need a 'prompt'")
        return D.Prompt(fmt.prompt_ids(row["prompt"]), row["prompt"], row=row)

    def build_units(self, rows):
        return [u for u in super().build_units(rows) if len(u.ids) < self.max_len]

    def _pick_max_len(self, rows) -> int:
        if self.cfg.max_seq_len:
            return int(self.cfg.max_seq_len)
        fmt = D.Formatter(self.tokenizer, max_len=1 << 20, chat=self.cfg.chat_template,
                          template_kwargs=self.cfg.template_kwargs, system=self.cfg.system_prompt)
        lens = [len(fmt.prompt_ids(r["prompt"])) for r in rows[:512] if "prompt" in r]
        longest = int(max(lens)) if lens else 512
        return int(D._round_up(longest + self.cfg.max_completion_tokens + 1, 64))

    def make_objective(self):
        c = self.cfg
        return O.GRPO(beta=c.beta, epsilon=c.epsilon, epsilon_high=c.epsilon_high, loss_type=c.loss_type)

    def unit_tokens(self) -> float:
        # upper bound: every completion runs to max_completion_tokens
        return (self.data_summary["mean"] + self.cfg.max_completion_tokens) * self.rows_per_unit

    def _probe(self):
        super()._probe()
        # The schedule counts prompts; each expands to num_generations rows.
        if not self.cfg.batch_size:
            self.plan.per_step = max(1, min(len(self.units), 64 // max(1, self.rows_per_unit) or 1))
            self.plan.reasons["batch"] = f"{self.plan.per_step} prompts x {self.rows_per_unit} generations"

    def next_batches(self, micro_units):
        cfg = self.cfg
        G = int(cfg.num_generations)
        prompts = [u for m in micro_units for u in m]
        ids = [p.ids for p in prompts for _ in range(G)]
        from . import rewards as R
        from .rollout import generate

        roll = generate(
            self.model, self.tokenizer, ids, max_tokens=cfg.max_completion_tokens,
            temperature=cfg.temperature, top_p=cfg.top_p, streamed=self.streamed,
        )
        # rewards see the text without the stop token; training keeps it
        texts = [self.tokenizer.decode(t[:-1] if done and t else t)
                 for t, done in zip(roll.tokens, roll.finished)]
        rows = [p.row for p in prompts for _ in range(G)]
        ptexts = [p.prompt for p in prompts for _ in range(G)]
        rewards, per_fn = R.score(self.reward_funcs, ptexts, texts, rows, cfg.reward_weights)
        adv, live = R.advantages(rewards, G, scale=cfg.scale_rewards)
        self._roll_logs = {
            "reward": float(rewards.mean()),
            "reward_std": float(rewards.reshape(-1, G).std(axis=1).mean()),
            "completion_len": float(np.mean([len(t) for t in roll.tokens])),
            "finished": float(np.mean(roll.finished)),
            "gen_s": roll.seconds,
            "gen_tok_s": roll.tok_s,
            "live_groups": float(live.mean()),
            **{f"r_{k}": v for k, v in per_fn.items()},
        }
        examples = []
        for i, (p_ids, comp) in enumerate(zip(ids, roll.tokens)):
            g = i // G
            if not comp or (cfg.skip_flat_groups and not live[g]):
                continue
            seq = (p_ids + comp)[: self.max_len]
            mask = ([0] * len(p_ids) + [1] * len(comp))[: self.max_len]
            examples.append(D.Example(seq, mask, meta={"adv": float(adv[i])}, row=rows[i]))
        if not examples:
            sample = texts[0].strip().replace("\n", " ")[:80] if texts else ""
            print(f"[train] every group scored flat (reward {rewards.mean():.2f}, "
                  f"{roll.gen_tokens} tokens in {roll.seconds:.1f}s); skipping. "
                  f"sample: {sample!r}", flush=True)
            return None
        sched = self._schedule(examples, len(examples), 0, rows_per_unit=1)
        batches = [D.collate(m, self.fmt.pad_id) for m in sched.step(0, 0)]
        need_old = cfg.iterations > 1
        if need_old or cfg.beta > 0:
            self.model.eval()
            try:
                if need_old:
                    for b, lp in zip(batches, self.score_batches(batches)):
                        b.meta["old_lp"] = lp
                if cfg.beta > 0:
                    with train_lora.disabled(self.model):
                        for b, lp in zip(batches, self.score_batches(batches)):
                            b.meta["ref_lp"] = lp
            finally:
                self.model.train()
        return batches

    def update(self, batches, lr):
        out = None
        for _ in range(max(1, int(self.cfg.iterations))):
            out = super().update(batches, lr)
        return out

    def after_update(self, batches):
        return dict(self._roll_logs)

    def next_eval_batches(self, micro):
        return []


TRAINERS = {
    "sft": SFTTrainer,
    "cpt": CPTTrainer,
    "orpo": ORPOTrainer,
    "dpo": DPOTrainer,
    "simpo": SimPOTrainer,
    "cpo": CPOTrainer,
    "kto": KTOTrainer,
    "grpo": GRPOTrainer,
}
