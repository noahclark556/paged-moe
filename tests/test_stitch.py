# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stitch: one-graph decode when every expert fits in the slab.

The gather reads the same slots as the per-layer path, so greedy tokens match
a run with stitch forced off. A cache too small to hold the set must decline
and still match.

    python tests/test_stitch.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from tests.test_stream import CONFIG, build_tiny_model, greedy_generate

from pagedmoe import config, load


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build_tiny_model(tmp)
        prompt = list(range(2, 42))
        n_new = 12
        n_layers = CONFIG["num_hidden_layers"]

        config.STITCH = False
        config.PRUNE = 0.0
        config.FLOW = False
        plain, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
        assert not plain._pagedmoe_cache.stitched
        ref = greedy_generate(plain, prompt, n_new)
        del plain

        config.STITCH = True
        config.STITCH_PARTIAL = True
        stitched, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
        cache = stitched._pagedmoe_cache
        assert cache.stitched, "slab holds the tiny model; stitch should be on"
        got = greedy_generate(stitched, prompt, n_new)
        assert got == ref, f"stitch diverged\n  ref={ref}\n  got={got}"
        # The prompt is one prefill. Each later token is one decode step.
        decode_steps = n_new - 1
        assert cache.stitch_forwards == decode_steps * n_layers, cache.stitch_forwards
        assert cache.full_stitch_tokens == decode_steps, cache.full_stitch_tokens
        # A full set is not evictable. Asking for a slot must not drop one.
        before = len(cache._stitch_keys)
        cache._free_slots.clear()
        with cache._lock:
            assert cache._alloc_slot(evict_ok=True) is None
        assert len(cache._stitch_keys) == before
        print(f"OK: stitch matches ({decode_steps} decode steps, {cache.stitch_forwards} layers)")

        # Drop the full-set flag. The experts stay resident, so the closed
        # path (live table, rewind on a miss) should accept every armed token.
        cache.stitched = False
        cache.closed_ok = 0
        cache.closed_replay = 0
        stitched._pagedmoe_ring.closed_armed = True
        got_closed = greedy_generate(stitched, prompt, n_new)
        assert got_closed == ref, f"closed stitch diverged\n  ref={ref}\n  got={got_closed}"
        # The prompt disarms it; four clean per-layer tokens arm it again.
        assert cache.closed_ok == decode_steps - 4, (cache.closed_ok, cache.closed_replay)
        assert cache.closed_replay == 0
        print(f"OK: closed stitch accepted {cache.closed_ok} tokens")

        # A big prompt gives the slab back while the closed path is armed. The
        # next decode token must rebuild it on the per-layer path, not gather
        # from a slab that is gone.
        cache.stitched = False
        pc = make_prompt_cache(stitched)
        logits = stitched(mx.array([prompt]), cache=pc)
        tok = int(mx.argmax(logits[:, -1, :]))
        got_released = [tok]
        cache.enter_prefill(1 << 30)
        assert cache.slab is None
        stitched._pagedmoe_ring.closed_armed = True
        for _ in range(n_new - 1):
            logits = stitched(mx.array([[tok]]), cache=pc)
            tok = int(mx.argmax(logits[:, -1, :]))
            got_released.append(tok)
        config.STITCH_PARTIAL = False
        assert got_released == ref, f"released slab diverged\n  got={got_released}"
        assert cache.slab is not None
        print("OK: released slab rebuilds before any one-graph token")
        del stitched

        # Partial stitch is opt-in. A default load does not install it.
        config.STITCH = True
        default, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
        assert not default._pagedmoe_info.get("stitch_closed")
        del default

        # Slider prune on a model that fits. KEEP_FREE puts every expert back,
        # so stitch must stay on and match the per-layer path token for token.
        config.PRUNE = 0.5
        config.WAIT_ABOVE = 0.2
        config.STITCH = True
        pruned_fast, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
        assert pruned_fast._pagedmoe_cache.stitched
        config.STITCH = False
        pruned_slow, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
        assert not pruned_slow._pagedmoe_cache.stitched
        fast_ids = greedy_generate(pruned_fast, prompt, n_new)
        slow_ids = greedy_generate(pruned_slow, prompt, n_new)
        assert fast_ids == slow_ids, f"prune+stitch diverged\n  fast={fast_ids}\n  slow={slow_ids}"
        print("OK: prune 0.5 + wait 0.2 matches with the full set resident")
        del pruned_fast, pruned_slow
        config.PRUNE = 0.0
        config.WAIT_ABOVE = 0.0
        config.STITCH = True

        config.STITCH = True
        small, _ = load(str(tmp), mode="streamed", cache_gb=0.00015, verbose=False)
        assert not small._pagedmoe_cache.stitched, "a starved slab must not stitch"
        got_small = greedy_generate(small, prompt, n_new)
        assert got_small == ref, f"starved path diverged\n  got={got_small}"
        print("OK: starved cache stays on the per-layer path")
        del small


if __name__ == "__main__":
    main()
