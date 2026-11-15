# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sampling defaults for requests that leave them out (sampling.py)."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pagedmoe import sampling, server  # noqa: E402


def ckpt(name, gen=None):
    d = Path(tempfile.mkdtemp()) / name
    d.mkdir()
    if gen is not None:
        (d / "generation_config.json").write_text(json.dumps(gen))
    return str(d)


class Handler:
    """The attributes mlx-lm's do_POST sets before handle_completion."""

    def __init__(self):
        self.temperature, self.top_p, self.top_k, self.min_p = 0.0, 1.0, 0, 0.0
        self.repetition_penalty, self.repetition_context_size = 0.0, 20


def test_family_match_order():
    assert sampling.family_for("qwen3_5_moe", "whatever")[0] == "qwen"
    assert sampling.family_for("qwen4_exp", "")[0] == "qwen"
    assert sampling.family_for("deepseek_v32", "")[0] == "deepseek"
    assert sampling.family_for("glm4_moe", "")[0] == "glm"
    assert sampling.family_for("gpt_oss", "")[0] == "gpt-oss"
    # model_type wins over the folder name, the name is the fallback.
    assert sampling.family_for("minimax_m3", "Qwen-ish-name")[0] == "minimax"
    assert sampling.family_for("custom", "GLM-5.2-4bit")[0] == "glm"
    assert sampling.family_for("custom", "my-model")[0] is None


def test_checkpoint_file_wins_family_fills():
    d = ckpt("PagedMoE-35B", {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "do_sample": True})
    out, src = sampling.defaults_for(d, "qwen3_5_moe")
    assert out == {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                   "repetition_penalty": 1.05, "repetition_context_size": 64}, out
    assert src["temperature"] == "generation_config.json" and src["repetition_penalty"] == "qwen family"
    d = ckpt("Qwen3-235B-Instruct", {"temperature": 0.7, "top_p": 0.8, "top_k": 20})
    assert sampling.defaults_for(d, "qwen3_moe")[0]["temperature"] == 0.7
    # The file's own repetition_penalty replaces the family's.
    d = ckpt("q", {"temperature": 0.7, "repetition_penalty": 1.1})
    assert sampling.defaults_for(d, "qwen3")[0]["repetition_penalty"] == 1.1


def test_greedy_and_missing():
    assert sampling.defaults_for(ckpt("cad", {"do_sample": False}), "qwen3") == ({}, {})
    assert sampling.defaults_for(ckpt("z", {"temperature": 0}), "qwen3") == ({}, {})
    # No file: family only. No family either: nothing.
    out, _ = sampling.defaults_for(ckpt("x"), "deepseek_v3")
    assert out == {"temperature": 0.6, "top_p": 0.95, "top_k": 40}, out
    assert sampling.defaults_for(ckpt("my-model"), "custom") == ({}, {})
    assert sampling.defaults_for(None, "") == ({}, {})


def test_noops_and_bad_values_dropped():
    d = ckpt("gpt-oss-120b", {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0})
    out, _ = sampling.defaults_for(d, "gpt_oss")
    assert out == {"temperature": 1.0}, out
    d = ckpt("m", {"temperature": "hot", "top_k": True, "top_p": -1})
    out, _ = sampling.defaults_for(d, "minimax")
    assert out == {"temperature": 0.6, "top_p": 0.95, "top_k": 40}, out


def test_fill_only_what_is_missing():
    defaults = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05,
                "repetition_context_size": 64}
    h = Handler()
    assert sampling.fill(h, {}, defaults, argv=[]) == list(defaults)
    assert (h.temperature, h.top_k, h.repetition_context_size) == (0.6, 20, 64)
    h = Handler()
    h.temperature = 0.0
    applied = sampling.fill(h, {"temperature": 0.0, "repetition_penalty": 1.2}, defaults, argv=[])
    assert h.temperature == 0.0 and "temperature" not in applied
    # The client set its own penalty: its context size is not ours to pick.
    assert "repetition_context_size" not in applied and h.repetition_context_size == 20


def test_server_flags_win():
    defaults = {"temperature": 0.6, "top_k": 20, "top_p": 0.95}
    h = Handler()
    applied = sampling.fill(h, {}, defaults, argv=["--model", "m", "--temp", "0.2", "--top-k=5"])
    assert applied == ["top_p"], applied
    assert h.temperature == 0.0 and h.top_k == 0


def test_server_lookup_and_log():
    d = ckpt("DeepSeek-V3.1-4bit", {"temperature": 0.6, "top_p": 0.95})

    class Args:
        model_type = "deepseek_v3"

    class Model:
        _pagedmoe_info = {"model_path": d}
        args = Args()

    out, src = server._sampling_for(Model())
    assert out == {"temperature": 0.6, "top_p": 0.95, "top_k": 40}, out
    assert server._sampling_for(Model()) is server._sampling_for(Model())
    assert "top_k 40 (deepseek family)" in sampling.describe(out, src)
    assert server._sampling_for(None) == ({}, {})


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
