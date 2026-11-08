# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Cache-only prefill skips the final layer's experts (config.PREFILL_SKIP_FINAL),
and kept-slab prefill chunks price their memory from measured peaks
(ExpertCache.prefill_need).

What has to hold:

  1. A prefill under streaming.cache_only() leaves the KV cache bit-identical
     to a normal prefill, and the decode that follows gives the same logits.
  2. It really skipped: the final layer read nothing, and the counter moved.
  3. Outside cache_only(), or with the knob off, nothing is skipped.
  4. _is_final_layer only says yes for the last entry of a layer list.
    5. prefill_need: prior before any sample; one sample bounds shorter chunks
     and scales up for longer ones; with two sizes a line that covers every
     sample; never under the floor.

Run:
    python tests/test_prefill_skip_final.py
"""

import sys
import tempfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from tests.test_stream import CONFIG, build_tiny_model  # noqa: E402

from pagedmoe import cache as es_cache  # noqa: E402
from pagedmoe import config as es_config  # noqa: E402
from pagedmoe import load, streaming  # noqa: E402


def _prefill_then_decode(model, prompt, n_new, *, cache_only):
    cache = make_prompt_cache(model)
    with streaming.exact_step():
        if cache_only:
            with streaming.cache_only():
                model(mx.array([prompt[:-1]]), cache=cache)
        else:
            model(mx.array([prompt[:-1]]), cache=cache)
    mx.eval([c.state for c in cache])
    state = [np.asarray(a.astype(mx.float32)) for c in cache for a in c.state]
    tok = prompt[-1]
    rows = []
    for _ in range(n_new):
        logits = model(mx.array([[tok]]), cache=cache)
        rows.append(np.asarray(logits[0, -1, :].astype(mx.float32)))
        tok = int(mx.argmax(logits[:, -1, :]))
    return state, np.stack(rows)


def check_skip_is_lossless():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build_tiny_model(tmp)
        rng = np.random.default_rng(11)
        prompt = rng.integers(0, CONFIG["vocab_size"], size=64).tolist()
        model, _ = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
        ring = model._pagedmoe_ring
        ec = model._pagedmoe_cache
        assert ring.final_is_moe, "tiny model's last layer is MoE"
        ref_state, ref_rows = _prefill_then_decode(model, prompt, 6, cache_only=False)
        assert ec.prefill_layers_skipped == 0, "skipped outside cache_only()"

        final = ring.layers[-1].layer_key
        reads = {}
        orig = ec.fetch_groups

        def counting(layer_key, ids, *a, **k):
            reads[layer_key] = reads.get(layer_key, 0) + len(ids)
            return orig(layer_key, ids, *a, **k)

        ec.fetch_groups = counting
        try:
            got_state, got_rows = _prefill_then_decode(model, prompt, 6, cache_only=True)
        finally:
            ec.fetch_groups = orig
        assert ec.prefill_layers_skipped == 1, ec.prefill_layers_skipped
        assert final not in reads, f"final layer still fetched {reads.get(final)} experts"
        assert len(reads) == len(ring.layers) - 1, reads
        assert len(ref_state) == len(got_state)
        for i, (a, b) in enumerate(zip(ref_state, got_state)):
            assert np.array_equal(a, b), f"cache array {i}: max |diff| {np.abs(a - b).max()}"
        assert np.array_equal(ref_rows, got_rows), "decode after a skipped prefill differs"

        es_config.PREFILL_SKIP_FINAL = False
        try:
            _prefill_then_decode(model, prompt, 1, cache_only=True)
            assert ec.prefill_layers_skipped == 1, "knob off still skipped"
        finally:
            es_config.PREFILL_SKIP_FINAL = True
        del model
    print("OK: cache-only prefill skips the final layer, KV and decode bit-identical")


def check_is_final_layer():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Linear(2, 2)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Block() for _ in range(3)]

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    m = Outer()
    assert streaming._is_final_layer(m, "model.layers.2.mlp")
    assert not streaming._is_final_layer(m, "model.layers.1.mlp")
    assert not streaming._is_final_layer(m, "model.mlp")
    assert not streaming._is_final_layer(m, "model.nope.2.mlp")
    print("OK: _is_final_layer only for the last entry of the layer list")


def check_prefill_need():
    fake = SimpleNamespace(_prefill_peaks=deque(maxlen=16))
    need = lambda n: es_cache.ExpertCache.prefill_need(fake, n)  # noqa: E731
    floor = es_cache._PREFILL_NEED_FLOOR
    margin = es_cache._PREFILL_NEED_MARGIN
    assert need(10) == floor, "tiny chunk under the floor"
    assert need(4096) == int(4096 * es_cache._PREFILL_TOKEN_PRIOR * margin)
    fake._prefill_peaks.append((1300, 1_700_000_000))
    assert need(1000) == int(1_700_000_000 * margin), "one sample bounds a shorter chunk"
    assert need(2600) == int(1_700_000_000 * 2 * margin), "one sample scales up"
    fake._prefill_peaks.extend([(300, 700_000_000), (900, 1_400_000_000)])
    for m, b in fake._prefill_peaks:
        assert need(m) >= int(b * margin), f"line under the {m}-token sample"
    assert need(300) < need(1300) < need(2600), "need grows with the chunk"
    assert need(300) < int(1_700_000_000 * margin), "small chunk priced like the big one"
    print("OK: prefill_need uses measured peaks, covers every sample, keeps the floor")


def main():
    check_is_final_layer()
    check_prefill_need()
    check_skip_is_lossless()
    print("\nOK: prefill skip-final tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
