# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPU breadcrumbs: how far the GPU really got when a Metal wait hangs.

gpu_trail records what the CPU queued. Once the GPU wedges, every later CPU
wait blocks too, so the wait in the stacks is a victim: the stalls on record
were stuck in a logprob read, a snapshot save and a router sync, all behind
the same hung GPU. A breadcrumb is a one-thread kernel queued right after a
point of interest (an MoE layer's input or output, a sampled token). It takes
that array as an input, so the GPU cannot run it before the array is done,
and it writes its sequence number into a small buffer whose memory the CPU
mapped at start. The watchdog reads that memory directly, no MLX call and no
sync: the last number there is the last point the GPU finished.

Each crumb is pushed with its own async_eval, which closes the command buffer
it sits in, and a finished command buffer's writes are visible to the CPU.
So a crumb shows up as soon as the work before it is done, and nothing
queued after it can hold it back.

Side branch only: no compute reads a crumb's output, so values are untouched
(bit-identical). Costs a dispatch and a command buffer commit per crumb.
PAGED_MOE_GPU_CRUMBS: 1 (default) marks the steps of each token, 2 adds every
MoE layer's input and output, 0 turns it off.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from . import config, gpu_trail

ON = int(config.GPU_CRUMBS) >= 1
LAYERS = int(config.GPU_CRUMBS) >= 2

_SOURCE = """
    uint v = tag[0];
    (void)after;
    ((device uint*)crumb)[0] = v;
    out[0] = v;
"""

_kernel = None
_buf = None
_view: np.ndarray | None = None
_tag = 0
_queued: deque[tuple[int, float, str]] = deque(maxlen=512)
# Crumb outputs nobody reads; held so their buffers outlive the kernel.
_keep: deque = deque(maxlen=64)


def _say(msg: str) -> None:
    print(f"[gpu-crumbs] {msg}", flush=True)


def _launch(x, tag: int):
    import mlx.core as mx

    return _kernel(
        inputs=[x, mx.array([tag], dtype=mx.uint32), _buf],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(1,)],
        output_dtypes=[mx.uint32],
    )[0]


def _init() -> bool:
    """Build the kernel and the mapped buffer, then prove a crumb lands."""
    global ON, LAYERS, _kernel, _buf, _view
    try:
        import mlx.core as mx

        _kernel = mx.fast.metal_kernel(
            name="pagedmoe_gpu_crumb",
            input_names=["after", "tag", "crumb"],
            output_names=["out"],
            source=_SOURCE,
            ensure_row_contiguous=False,
        )
        # MLX binds tiny inputs in the read-only constant address space; past
        # a handful of elements they are plain device memory the kernel can write.
        _buf = mx.zeros((256,), dtype=mx.uint32)
        mx.eval(_buf)
        view = np.asarray(_buf)
        if not view.flags.writeable or not view.flags.c_contiguous:
            raise RuntimeError("MLX did not hand back a view of the crumb buffer")
        _view = view
        probe = mx.arange(8, dtype=mx.bfloat16)
        out = _launch(probe, 0x5EED)
        mx.eval(out)
        if int(_view[0]) != 0x5EED or int(out.item()) != 0x5EED:
            raise RuntimeError(f"self test read {int(_view[0]):#x}, wanted 0x5eed")
        _view[0] = 0
    except Exception as e:  # noqa: BLE001 - a diagnostic must never stop decode
        ON = LAYERS = False
        _kernel = _buf = _view = None
        _say(f"off, self test failed: {type(e).__name__}: {e}")
        return False
    gpu_trail.probe("gpu crumbs", report)
    _say("on: each token's logits, processors, logprobs and sample"
         + (", plus every MoE layer's input and output" if LAYERS else ""))
    return True


def crumb(x, label: str) -> None:
    """Queue a crumb that runs once `x` is computed. A no-op when off."""
    global _tag
    if not ON:
        return
    if _kernel is None and not _init():
        return
    try:
        import mlx.core as mx

        if x.size == 0:
            return
        tag = _tag + 1 if _tag < 0xFFFFFFF0 else 1
        out = _launch(x, tag)
        mx.async_eval(out)
    except Exception:  # noqa: BLE001 - inside a trace, odd dtype: skip this one
        return
    _tag = tag
    _queued.append((tag, time.monotonic(), label))
    _keep.append(out)


def report(now: float | None = None) -> list[str]:
    """Where the GPU stopped, as the crumbs on either side of it."""
    view = _view
    if not ON or view is None:
        return []
    done = int(view[0])
    queued = list(_queued)
    now = time.monotonic() if now is None else now
    if not queued:
        return [f"gpu crumbs: none queued yet (buffer holds {done})"]
    at = next((i for i in range(len(queued) - 1, -1, -1) if queued[i][0] == done), None)
    if at is None:
        if done == 0:
            return [f"gpu crumbs: the GPU has not finished any of the {len(queued)} queued"]
        return [f"gpu crumbs: the GPU last finished #{done}, older than the "
                f"{len(queued)} kept, so it is far behind the CPU"]
    lines = []
    for tag, t, label in queued[max(0, at - 2): at + 1]:
        lines.append(f"gpu crumbs: finished #{tag} {label} (queued {now - t:.1f}s ago)")
    pending = queued[at + 1:]
    if not pending:
        lines.append("gpu crumbs: the GPU finished every crumb queued, so the stuck work comes after "
                     "the last one (unmarked work, or a read on another stream)")
        return lines
    tag, t, label = pending[0]
    lines.append(f"gpu crumbs: STUCK before #{tag} {label} (queued {now - t:.1f}s ago); "
                 f"{len(pending)} crumb(s) not finished")
    for tag, t, label in pending[1:4]:
        lines.append(f"gpu crumbs:   then #{tag} {label}")
    return lines
