# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Prefill stage (config.PREFILL_STAGE): prefill misses read into a Metal slot
store instead of host buffers.

What has to hold:

  1. Stage on and off give the same prefill logits and decode, bit for bit,
     with the slab kept and with the slab released for the chunk.
  2. Every stage slot is back after a prefill, the stage stays up across
     prefill chunks and is gone once decode starts.
  3. No room for it: no stage, the run-read path, same output.
  4. make_room_for does not count the stage against the chunk it was
     admitted for (it used to hand the whole slab back mid-prefill).

Run:
    python tests/test_prefill_stage.py
"""

import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from tests.test_stream import CONFIG, build_tiny_model  # noqa: E402

from pagedmoe import config as es_config  # noqa: E402
from pagedmoe import get_stats, load  # noqa: E402
from pagedmoe.cache import _SLAB_MARGIN  # noqa: E402


def run(model, prompt, n_new):
    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt]), cache=cache)
    pre = np.asarray(logits[0].astype(mx.float32))
    tok = int(mx.argmax(logits[:, -1, :]))
    rows = []
    for _ in range(n_new):
        logits = model(mx.array([[tok]]), cache=cache)
        rows.append(np.asarray(logits[0, -1, :].astype(mx.float32)))
        tok = int(mx.argmax(logits[:, -1, :]))
    return pre, np.stack(rows)


def same(a, b, what):
    assert np.array_equal(a, b), f"{what}: max |diff| {np.abs(a - b).max()}"


def main():
    rng = np.random.default_rng(9)
    prompt = rng.integers(0, CONFIG["vocab_size"], size=64).tolist()
    old = (es_config.SLAB, es_config.PREFILL_STAGE, es_config.PREFILL_SLAB, es_config.PREFILL_SHRINK_TOKENS)
    es_config.SLAB = "on"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            build_tiny_model(tmp)
            # (label, cache_gb, PREFILL_SLAB, SHRINK): kept starved, kept no slot
            # reads, slab released for the chunk.
            cases = (("starved", 0.0009, True, 8192), ("stage only", 0.01, False, 8192), ("released", 0.01, True, 16))
            for label, cache_gb, slots, shrink in cases:
                es_config.PREFILL_SLAB, es_config.PREFILL_SHRINK_TOKENS = slots, shrink
                out = {}
                for stage in (False, True):
                    es_config.PREFILL_STAGE = stage
                    model, _ = load(str(tmp), mode="streamed", cache_gb=cache_gb, verbose=False)
                    c = model._pagedmoe_cache
                    out[stage] = run(model, prompt, 8)
                    assert c.stage is None, f"{label}: stage still up after decode"
                    st = get_stats(model)["cache"]
                    if stage:
                        assert st["stage_reads"] > 0, f"{label}: stage never read"
                        assert st["stage_builds"] >= 1
                        # Two chunks, then check mid-prefill: one stage, all slots back.
                        cache = make_prompt_cache(model)
                        mx.eval(model(mx.array([prompt[:32]]), cache=cache))
                        builds = c.stage_builds
                        mx.eval(model(mx.array([prompt[32:-1]]), cache=cache))
                        assert c.stage is not None and c.stage_builds == builds, "stage rebuilt between chunks"
                        assert len(c._stage_free) == c.stage.slots, (
                            f"{label}: {c.stage.slots - len(c._stage_free)} stage slots leaked"
                        )
                        c.leave_prefill()
                        assert c.stage is None
                    else:
                        assert st["stage_reads"] == 0
                    del model
                same(out[False][0], out[True][0], f"prefill logits {label}")
                same(out[False][1], out[True][1], f"decode logits {label}")
                print(f"OK: {label}: stage on == off, slots all returned")

            # No room: the stage is refused and the run-read path takes over.
            es_config.PREFILL_SLAB, es_config.PREFILL_SHRINK_TOKENS = True, 8192
            es_config.PREFILL_STAGE = True
            model, _ = load(str(tmp), mode="streamed", cache_gb=0.0009, verbose=False)
            c = model._pagedmoe_cache
            c._ram_budget_bytes = int(mx.get_active_memory()) + _SLAB_MARGIN + 1
            tight = run(model, prompt, 8)
            assert get_stats(model)["cache"]["stage_reads"] == 0, "stage built with no room"
            same(tight[0], out[False][0], "prefill logits, stage refused")
            print("OK: no room, no stage, same output")

            # A landed stage read still in flight when the stage goes, and no
            # longer in the tracking list (_stage_return prunes finished ones):
            # decode used to claim it and compute from a freed stage.
            from concurrent.futures import Future

            c._ram_budget_bytes = 1 << 40
            cache = make_prompt_cache(model)
            mx.eval(model(mx.array([prompt[:-1]]), cache=cache))
            assert c.stage is not None
            layer = next(iter(c._layouts))
            stale = Future()
            stale.set_result({"__nbytes__": 1, "__stage__": 0, "__gen__": c._stage_gen})
            with c._lock:
                c._inflight[(layer, 0)] = stale
                c._stage_futs = []
            c.leave_prefill()
            assert (layer, 0) not in c._inflight, "landed stage read survived the release"
            # And one that slips past anyway is re-read, not computed from.
            with c._lock:
                c._inflight[(layer, 0)] = stale
            got = c.fetch(layer, [0])
            assert "__stage__" not in got[0], "decode computed from a stage entry"
            # The silent form: a later prefill (new stage) claiming that read
            # would compute from whatever the new stage holds in that slot.
            assert c.ensure_stage()
            with c._lock:
                c._inflight[(layer, 0)] = stale
            b = c._begin(layer, [0], prefill=True)
            assert all(f is not stale for _, f in b.futs), "prefill claimed a read from an older stage"
            c._finish(layer, b, "soft")
            c.leave_prefill()
            for t in prompt[-1:] + prompt[:4]:
                mx.eval(model(mx.array([[t]]), cache=cache))
            print("OK: stage reads never reach decode")

            # Admitted stage does not count against its own chunk.
            c._ram_budget_bytes = 1 << 40
            assert c.ensure_stage()
            stage_bytes = c.stage.nbytes
            need = 64 << 20
            c._ram_budget_bytes = int(mx.get_active_memory()) + _SLAB_MARGIN + need - stage_bytes // 2
            assert not c.make_room_for(need), "slab handed back over the stage's own memory"
            assert c.slab is not None
            c.leave_prefill()
            del model
            print("OK: make_room_for credits the admitted stage")
    finally:
        es_config.SLAB, es_config.PREFILL_STAGE, es_config.PREFILL_SLAB, es_config.PREFILL_SHRINK_TOKENS = old
    print("\nOK: prefill stage tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
