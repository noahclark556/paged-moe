# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
PEFT export of a run's adapter (pagedmoe.train.export):

  names map across the multimodal split (language_model.model.layers.N in
  MLX, model.language_model.layers.N in HF) and ignore the mtp stack
  lora_a / lora_b transpose into PEFT's lora_A / lora_B, alpha = scale * r
  DoRA magnitudes come along, routed-expert LoRA is skipped
  a wrong base model (shapes or missing layers) stops the export
  the base repo is read off a model card
  export_peft=True on a real tiny run writes <run>/peft at the end, and the
  CLI export command does the same for a checkpoint

When torch and peft are installed it also loads the export with PEFT.

    python tests/test_train_export.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
import numpy as np

from pagedmoe.train.export import ExportError, base_repo_of, to_peft

IN, OUT, R = 32, 48, 4


def fake_base(d: Path, prefix="model.language_model.", q_out=OUT):
    """An HF-style (unquantized) checkpoint: only names and shapes matter."""
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model_type": "toy"}))
    w = {}
    for i in range(2):
        w[f"{prefix}layers.{i}.self_attn.q_proj.weight"] = mx.zeros((q_out, IN))
        w[f"{prefix}layers.{i}.mlp.shared_expert.down_proj.weight"] = mx.zeros((OUT, IN))
    w["mtp.layers.0.self_attn.q_proj.weight"] = mx.zeros((OUT, IN))
    mx.save_safetensors(str(d / "model.safetensors"), w)
    return d


def fake_adapter(d: Path, base: Path, dora=False, experts=True):
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    w = {}
    for i in range(2):
        for mod in ("self_attn.q_proj", "mlp.shared_expert.down_proj"):
            k = f"language_model.model.layers.{i}.{mod}"
            w[f"{k}.lora_a"] = mx.array(rng.standard_normal((IN, R)).astype(np.float32))
            w[f"{k}.lora_b"] = mx.array(rng.standard_normal((R, OUT)).astype(np.float32))
            if dora:
                w[f"{k}.m"] = mx.array(rng.standard_normal((OUT,)).astype(np.float32))
        if experts:
            k = f"language_model.model.layers.{i}.mlp.switch_mlp.down_proj"
            w[f"{k}.lora_a"] = mx.zeros((4, IN, R))
            w[f"{k}.lora_b"] = mx.zeros((4, R, OUT))
    mx.save_safetensors(str(d / "adapters.safetensors"), w)
    (d / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "dora" if dora else "lora", "num_layers": 2, "base_model": str(base),
        "lora_parameters": {"rank": R, "scale": 2.0, "dropout": 0.05}}))
    return d, w


def check_mapping(tmp: Path):
    base = fake_base(tmp / "base")
    src, w = fake_adapter(tmp / "run", base)
    logs = []
    info = to_peft(src, base=str(base), log=logs.append)
    out = Path(info["out"])
    assert out == src / "peft", out
    assert info["modules"] == 4 and info["skipped_experts"] == 2 and info["verified"], info
    assert any("routed-expert" in s for s in logs), logs
    t = mx.load(str(out / "adapter_model.safetensors"))
    key = "base_model.model.model.language_model.layers.1.self_attn.q_proj"
    assert t[f"{key}.lora_A.weight"].shape == (R, IN)
    assert t[f"{key}.lora_B.weight"].shape == (OUT, R)
    src_a = w["language_model.model.layers.1.self_attn.q_proj.lora_a"]
    assert np.array_equal(np.asarray(t[f"{key}.lora_A.weight"]), np.asarray(src_a).T)
    assert not any("mtp" in k or "switch_mlp" in k for k in t), list(t)
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert cfg["peft_type"] == "LORA" and cfg["r"] == R and cfg["lora_alpha"] == 8.0
    assert cfg["lora_dropout"] == 0.05 and not cfg["use_dora"]
    assert cfg["target_modules"][0] == "model.language_model.layers.0.mlp.shared_expert.down_proj"
    assert len(cfg["target_modules"]) == 4
    assert (out / "README.md").exists()
    print("mapping: ok")


def check_dora(tmp: Path):
    base = fake_base(tmp / "base-d")
    src, w = fake_adapter(tmp / "run-d", base, dora=True, experts=False)
    info = to_peft(src, tmp / "out-d", base=str(base), log=lambda s: None)
    t = mx.load(str(Path(info["out"]) / "adapter_model.safetensors"))
    mags = [k for k in t if k.endswith("lora_magnitude_vector")]
    assert len(mags) == 4, mags
    assert json.loads((Path(info["out"]) / "adapter_config.json").read_text())["use_dora"]
    print("dora: ok")


def check_errors(tmp: Path):
    src, _ = fake_adapter(tmp / "run-e", fake_base(tmp / "base-e"))
    wrong_shape = fake_base(tmp / "base-shape", q_out=OUT + 8)
    try:
        to_peft(src, tmp / "x1", base=str(wrong_shape), log=lambda s: None)
        raise AssertionError("shape mismatch exported")
    except ExportError as e:
        assert "shapes don't match" in str(e), e
    other = tmp / "base-other"
    other.mkdir()
    (other / "config.json").write_text("{}")
    mx.save_safetensors(str(other / "model.safetensors"),
                        {"model.layers.0.self_attn.k_proj.weight": mx.zeros((OUT, IN))})
    try:
        to_peft(src, tmp / "x2", base=str(other), log=lambda s: None)
        raise AssertionError("unrelated base exported")
    except ExportError as e:
        assert "none of the adapted layers" in str(e), e
    print("errors: ok")


def check_base_repo(tmp: Path):
    a = tmp / "card-a"
    a.mkdir()
    (a / "README.md").write_text("---\nbase_model:\n- Qwen/Qwen3.6-35B-A3B\nlibrary_name: mlx\n---\n# x\n")
    b = tmp / "card-b"
    b.mkdir()
    (b / "README.md").write_text("This model was converted to MLX format from [zai-org/GLM-4.7](https://x) using mlx-lm.")
    assert base_repo_of(a) == "Qwen/Qwen3.6-35B-A3B"
    assert base_repo_of(b) == "zai-org/GLM-4.7"
    assert base_repo_of("org/name") == "org/name"
    assert base_repo_of(None) is None
    print("base repo: ok")


def check_run(tmp: Path):
    """export_peft on a real tiny run, then the CLI on its checkpoint."""
    from test_stream import build_tiny_model
    from test_train_methods import ByteTok, common, rows_chat

    from pagedmoe.train import SFTConfig, SFTTrainer
    from pagedmoe.train.cli import main as cli

    (tmp / "model").mkdir()
    build_tiny_model(tmp / "model")
    cfg = SFTConfig(**common(tmp, "sft", data=rows_chat(), export_peft=True, max_steps=2))
    summary = SFTTrainer(cfg, tokenizer=ByteTok()).train()
    peft = tmp / "sft" / "peft"
    assert summary.get("peft") == str(peft), summary
    pcfg = json.loads((peft / "adapter_config.json").read_text())
    assert pcfg["target_modules"] and all(".layers." in m for m in pcfg["target_modules"]), pcfg
    ck = next((tmp / "sft" / "checkpoints").glob("step-*"))
    assert cli(["export", str(tmp / "sft"), "--checkpoint", ck.name, "--out", str(tmp / "ck-peft")]) == 0
    assert (tmp / "ck-peft" / "adapter_model.safetensors").exists()
    print(f"run: ok ({len(pcfg['target_modules'])} modules)")


def check_peft_loads(tmp: Path):
    """Optional: PEFT itself reads the export onto a matching torch model."""
    try:
        import torch
        from peft import PeftModel
    except ImportError:
        print("peft load: skipped (torch / peft not installed)")
        return

    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(IN, OUT, bias=False)

    class Mlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_expert = torch.nn.Module()
            self.shared_expert.down_proj = torch.nn.Linear(IN, OUT, bias=False)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn, self.mlp = Attn(), Mlp()

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Module()
            self.model.language_model.layers = torch.nn.ModuleList([Layer(), Layer()])

        def prepare_inputs_for_generation(self, *args, **kw):
            return kw

    base = fake_base(tmp / "base-p")
    src, w = fake_adapter(tmp / "run-p", base, experts=False)
    info = to_peft(src, base=str(base), log=lambda s: None)
    m = PeftModel.from_pretrained(Toy(), info["out"])
    lin = m.base_model.model.model.language_model.layers[0].self_attn.q_proj
    got = lin.lora_A["default"].weight.detach().numpy()
    want = np.asarray(w["language_model.model.layers.0.self_attn.q_proj.lora_a"]).T
    assert np.allclose(got, want), "PEFT loaded different weights"
    assert lin.scaling["default"] == 2.0
    print("peft load: ok")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        check_mapping(tmp)
        check_dora(tmp)
        check_errors(tmp)
        check_base_repo(tmp)
        check_run(tmp)
        check_peft_loads(tmp)
    print("OK")


def test_train_export():
    main()


if __name__ == "__main__":
    main()
