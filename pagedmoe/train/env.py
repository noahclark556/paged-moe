# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Engine settings for training.

Training wants the exact model: every token computes with the experts its
router picked. So everything that trades fidelity for decode speed is off,
and so is everything that only exists for serving (slab, stitch, fused
prefill, verify replay, sidecar, learned margin). The training expert path
fetches through the plain LRU, which hands back ordinary arrays that stay
valid for the whole backward pass.

Cache size, read threads and RAM fraction are left alone: those are timing
and capacity knobs, and the planner sizes them.
"""

from __future__ import annotations

import os

# env name -> (config attribute, value)
EXACT = {
    "PAGED_MOE_PRUNE": ("PRUNE", 0.0),
    "PAGED_MOE_WAIT_ABOVE": ("WAIT_ABOVE", 0.0),
    "PAGED_MOE_ROUTE_TOP_K": ("ROUTE_TOP_K", 0),
    "PAGED_MOE_ROUTE_TOP_P": ("ROUTE_TOP_P", 0.0),
    "PAGED_MOE_FLOW": ("FLOW", False),
    "PAGED_MOE_DRAFT_FLOW": ("DRAFT_FLOW", False),
    "PAGED_MOE_SLAB": ("SLAB", "off"),
    "PAGED_MOE_STITCH": ("STITCH", False),
    "PAGED_MOE_FUSED_PREFILL": ("FUSED_PREFILL", False),
    "PAGED_MOE_VERIFY": ("VERIFY", False),
    "PAGED_MOE_SIDECAR": ("SIDECAR", False),
    "PAGED_MOE_PREDICT": ("PREDICT", "off"),
    "PAGED_MOE_WS_MARGIN_LEARN": ("WS_MARGIN_LEARN", False),
    "PAGED_MOE_PREFILL_AHEAD": ("PREFILL_AHEAD", False),
    "PAGED_MOE_TOOL_WARM": ("TOOL_WARM", False),
    "PAGED_MOE_TTW_LOG": ("TTW_LOG", False),
}


def _env_text(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)


def apply() -> None:
    """Force exact mode in the environment and in an already imported config."""
    for name, (_attr, value) in EXACT.items():
        os.environ[name] = _env_text(value)
    os.environ["PAGED_MOE_MITM"] = "0"
    from .. import config

    for _name, (attr, value) in EXACT.items():
        if hasattr(config, attr):
            setattr(config, attr, value)


_ARG_FNS = ("argpartition", "argsort", "argmax", "argmin")
_saved_arg_fns: dict = {}


def index_ops_stop_gradient() -> None:
    """Make mx.arg* return stop_gradient'ed indices.

    MoE routers do take_along_axis(scores, argpartition(scores)). MLX then
    tries to differentiate through the integer indices and refuses
    ("Cannot calculate VJP with respect to indices"), so stock mlx-lm cannot
    take a gradient through most MoE blocks. Indices never carry a gradient,
    so cutting them is exact: the same values, and the gradient still flows
    through the gathered scores.
    """
    import mlx.core as mx

    if _saved_arg_fns:
        return
    for name in _ARG_FNS:
        fn = getattr(mx, name, None)
        if fn is None:
            continue
        _saved_arg_fns[name] = fn

        def wrapped(*args, __fn=fn, **kwargs):
            return mx.stop_gradient(__fn(*args, **kwargs))

        setattr(mx, name, wrapped)


def restore_index_ops() -> None:
    import mlx.core as mx

    for name, fn in _saved_arg_fns.items():
        setattr(mx, name, fn)
    _saved_arg_fns.clear()


def exact_numerics() -> None:
    """Turn off mx.compile while training.

    A compiled function fuses differently when it runs plain than when it is
    traced for a gradient (gated delta's step function in qwen3_5 is one), so
    a no-grad forward and the same forward under value_and_grad disagree by
    more than rounding. Training needs them identical: the layered recompute,
    reference logprobs and GRPO's old-policy ratio all assume it. Rollouts
    turn it back on with compiled().
    """
    import mlx.core as mx

    mx.disable_compile()


class compiled:
    """Re-enable mx.compile for a block that is never differentiated."""

    def __enter__(self):
        import mlx.core as mx

        mx.enable_compile()

    def __exit__(self, *exc):
        import mlx.core as mx

        mx.disable_compile()


def check() -> list[str]:
    """Settings that drifted from exact mode since apply(). Empty when clean."""
    from .. import config

    bad = []
    for _name, (attr, value) in EXACT.items():
        if hasattr(config, attr) and getattr(config, attr) != value:
            bad.append(f"{attr}={getattr(config, attr)!r}")
    return bad
