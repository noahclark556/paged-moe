# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Missing architecture files are filled only for a known, matching config."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pagedmoe import local_arch


def _flash_config(**over):
    text = {
        "model_type": "qwen4_exp_text",
        "hidden_size": 2560,
        "num_hidden_layers": 48,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "hc_count": 4,
        "indexer_n_heads": 4,
        "heads_per_ngram": 8,
        "layer_types": ["linear_attention", "full_attention"],
    }
    text.update(over.pop("text", {}))
    cfg = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "text_config": text,
    }
    cfg.update(over)
    return cfg


class EnsureArchTests(unittest.TestCase):
    def test_writes_bundled_port_when_the_file_is_missing(self):
        bundled = local_arch._port_text("qwen4_exp")
        if bundled is None:
            self.skipTest("qwen4_exp port is not bundled in this tree")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(_flash_config()))
            with mock.patch.object(local_arch, "_mlx_provides", return_value=False):
                written = local_arch.ensure(root)
            self.assertEqual(written, str((root / "qwen4_exp.py").resolve()))
            self.assertEqual((root / "qwen4_exp.py").read_text(), bundled)
            with mock.patch.object(local_arch, "_mlx_provides", return_value=False):
                self.assertIsNone(local_arch.ensure(root))
            self.assertEqual((root / "qwen4_exp.py").read_text(), bundled)

    def test_leaves_an_existing_file_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(_flash_config()))
            (root / "qwen4_exp.py").write_text("# checkpoint port\n")
            with mock.patch.object(local_arch, "_mlx_provides", return_value=False):
                self.assertIsNone(local_arch.ensure(root))
            self.assertEqual((root / "qwen4_exp.py").read_text(), "# checkpoint port\n")

    def test_does_not_touch_other_model_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps({"model_type": "qwen3_next", "architectures": ["Qwen3NextForCausalLM"]})
            )
            self.assertIsNone(local_arch.ensure(root))
            self.assertFalse((root / "qwen3_next.py").exists())
            self.assertFalse((root / "qwen4_exp.py").exists())

    def test_refuses_a_qwen4_exp_config_that_does_not_match(self):
        bad = [
            _flash_config(architectures=["SomeOtherModel"]),
            _flash_config(text={"model_type": "qwen3"}),
            _flash_config(text={"layer_types": ["full_attention"]}),
            _flash_config(text={"hidden_size": True}),
            {"model_type": "qwen4_exp"},
        ]
        for cfg in bad:
            with self.subTest(cfg=cfg), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "config.json").write_text(json.dumps(cfg))
                with mock.patch.object(local_arch, "_mlx_provides", return_value=False):
                    self.assertIsNone(local_arch.ensure(root))
                self.assertFalse((root / "qwen4_exp.py").exists())

    def test_does_not_write_when_mlx_lm_already_ships_the_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(_flash_config()))
            with mock.patch.object(local_arch, "_mlx_provides", return_value=True):
                self.assertIsNone(local_arch.ensure(root))
            self.assertFalse((root / "qwen4_exp.py").exists())


if __name__ == "__main__":
    unittest.main()
