# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Decode starts a layer's resident experts while its misses read.

The rows computed early and the rows computed after the reads land have to
come out exactly as one gather over the whole layer would, or the overlap is
changing output instead of timing.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_stream import CONFIG, build_tiny_model, greedy_generate  # noqa: E402

from pagedmoe import config as es_config  # noqa: E402
from pagedmoe import load, streaming  # noqa: E402


def _first_glu(model):
    for layer in model.model.layers:
        glu = getattr(getattr(layer, "mlp", None), "switch_mlp", None)
        if isinstance(glu, streaming.StreamedSwitchGLU):
            return glu
    raise AssertionError("no streamed MoE layer")


class TestHitsEarly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        build_tiny_model(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _load(self, cache_gb=1.0):
        model, _ = load(self.tmp.name, mode="streamed", cache_gb=cache_gb, verbose=False)
        return model

    def test_split_matches_one_gather(self):
        model = self._load()
        glu = _first_glu(model)
        k = CONFIG["num_experts_per_tok"]
        rng = np.random.default_rng(0)
        inds = np.stack([rng.choice(CONFIG["num_experts"], k, replace=False) for _ in range(3)])
        ids = sorted({int(e) for e in inds.reshape(-1)})
        experts = glu.cache.fetch(glu.layer_key, ids, install="lru")
        slots = {e: ent["__slot__"] for e, ent in experts.items()}
        x = mx.random.normal((3, CONFIG["hidden_size"])).astype(mx.bfloat16)
        keep = rng.random(inds.shape) > 0.3
        for mask in (None, keep):
            ref = glu._run_slab(x, inds, slots, keep=mask)
            flat = inds.reshape(-1)
            pos = np.arange(flat.size) if mask is None else np.nonzero(mask.reshape(-1))[0]
            done = pos[np.isin(flat[pos], ids[: len(ids) // 2])]
            y = glu._slab_rows(x, done // k, [slots[int(e)] for e in flat[done]])
            got = glu._run_slab(x, inds, slots, keep=mask, early=(done, y))
            self.assertTrue(mx.array_equal(ref, got), f"keep={mask is not None}")

    def test_decode_output_unchanged(self):
        rng = np.random.default_rng(3)
        prompt = rng.integers(0, CONFIG["vocab_size"], size=3).tolist()
        calls = []
        real = streaming.StreamedSwitchGLU._run_hits_early

        def spy(self, *a, **kw):
            out = real(self, *a, **kw)
            calls.append(out is not None)
            return out

        outs = {}
        for on in (False, True):
            # PRUNE / WAIT_ABOVE would make the output depend on residency, and a
            # stitched slab holds every expert, so decode would never miss.
            with mock.patch.object(es_config, "HITS_EARLY", on), mock.patch.object(
                es_config, "PRUNE", 0.0
            ), mock.patch.object(es_config, "WAIT_ABOVE", 0.0), mock.patch.object(
                es_config, "STITCH", False
            ), mock.patch.object(
                streaming.StreamedSwitchGLU, "_run_hits_early", spy
            ):
                # Short prompt, cold slab: decode keeps meeting experts it never read.
                model = self._load(cache_gb=0.002)
                outs[on] = greedy_generate(model, prompt, 16)
                del model
        self.assertEqual(outs[False], outs[True])
        self.assertTrue(any(calls), "no decode layer ever started hits early")


if __name__ == "__main__":
    unittest.main()
