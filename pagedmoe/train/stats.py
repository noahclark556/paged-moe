# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Progress lines, metrics log and callbacks."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .plan import fmt_time, fmt_tokens


class Callback:
    """Hooks into a run. Override what you need."""

    def on_train_begin(self, trainer) -> None: ...
    def on_progress(self, trainer, info: dict) -> None:
        """Setup and in-step progress (load, probe, layer k of L), about once a second."""
    def on_step_end(self, trainer, step: int, logs: dict) -> None: ...
    def on_evaluate(self, trainer, step: int, logs: dict) -> None: ...
    def on_save(self, trainer, step: int, path: Path) -> None: ...
    def on_train_end(self, trainer, logs: dict) -> None: ...


class JsonlLog(Callback):
    """metrics.jsonl in the run dir: one line per step and per eval."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _write(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def on_step_end(self, trainer, step, logs):
        self._write({"kind": "step", "step": step, "time": time.time(), **logs})

    def on_evaluate(self, trainer, step, logs):
        self._write({"kind": "eval", "step": step, "time": time.time(), **logs})


class WandbLog(Callback):
    def __init__(self, project: str = "paged-training", name: str | None = None, config: dict | None = None):
        import wandb

        self.wandb = wandb
        self.run = wandb.init(project=project, name=name, config=config or {})

    def on_step_end(self, trainer, step, logs):
        self.wandb.log({f"train/{k}": v for k, v in logs.items() if isinstance(v, (int, float))}, step=step)

    def on_evaluate(self, trainer, step, logs):
        self.wandb.log({f"eval/{k}": v for k, v in logs.items() if isinstance(v, (int, float))}, step=step)

    def on_train_end(self, trainer, logs):
        self.run.finish()


class Rate:
    """EMA of seconds per step, for the ETA."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.value: float | None = None

    def add(self, x: float) -> None:
        self.value = x if self.value is None else (1 - self.alpha) * self.value + self.alpha * x


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s


class Console(Callback):
    """One line per logged step, a short breakdown every `detail_every` steps."""

    def __init__(self, log_steps: int = 1, detail_every: int = 25):
        self.log_steps = max(1, int(log_steps))
        self.detail_every = detail_every

    def on_step_end(self, trainer, step, logs):
        if step % self.log_steps and step != trainer.total_steps:
            return
        parts = [
            _c("1", f"step {step}/{trainer.total_steps}"),
            f"loss {logs['loss']:.4f}",
        ]
        for k in ("acc", "margin", "reward", "kl", "clip_frac"):
            if k in logs:
                parts.append(f"{k} {logs[k]:.3f}")
        parts += [
            f"lr {logs['lr']:.1e}",
            f"gn {logs['grad_norm']:.2f}",
            f"{fmt_tokens(logs['tok_s'])} tok/s",
            f"{logs['step_s']:.1f}s",
        ]
        if logs.get("read_gb"):
            parts.append(f"read {logs['read_gb']:.1f} GB ({logs['passes']:.1f}x)")
        parts.append(f"peak {logs['peak_gb']:.1f} GB")
        parts.append(_c("2", f"eta {fmt_time(logs.get('eta_s'))}"))
        print(" | ".join(parts), flush=True)
        if self.detail_every and step % self.detail_every == 0:
            t = logs.get("timing", {})
            if t:
                br = ", ".join(f"{k} {v:.1f}s" for k, v in t.items())
                print(_c("2", f"  time: {br}"), flush=True)
            if logs.get("experts"):
                e = logs["experts"]
                print(_c("2", f"  experts: {e.get('experts', 0)} fetched, "
                              f"ahead hit {e.get('ahead_hit')}, "
                              f"cache hit {logs.get('cache_hit', 0):.0%}, "
                              f"disk wait {logs.get('disk_wait_s', 0):.1f}s"), flush=True)

    def on_evaluate(self, trainer, step, logs):
        body = ", ".join(f"{k} {v:.4f}" for k, v in logs.items() if isinstance(v, float))
        print(_c("36", f"eval @ {step}: {body}"), flush=True)

    def on_save(self, trainer, step, path):
        print(_c("2", f"  saved {path}"), flush=True)
