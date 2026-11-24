# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
pagedmoe.convert against mlx-lm on a tiny Qwen3-MoE saved in the Hugging Face
layout (one tensor per expert, bf16), so sanitize() has real work to do.

Checks: a uniform 4-bit recipe gives exactly what mlx-lm's quantize gives,
a mixed recipe writes a config mlx-lm loads strictly, a stopped run resumes
to the same bytes, a per-layer streamed run matches a whole-checkpoint run,
an adapter merges exactly like mlx-lm's fuse, and bad input fails early.

    python tests/test_convert.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CFG = {
    "model_type": "qwen3_moe", "hidden_size": 64, "num_hidden_layers": 3, "intermediate_size": 128,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 256,
    "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "max_position_embeddings": 512,
    "tie_word_embeddings": False, "num_experts": 8, "num_experts_per_tok": 2,
    "decoder_sparse_step": 1, "mlp_only_layers": [], "moe_intermediate_size": 64,
    "norm_topk_prob": True, "torch_dtype": "bfloat16",
}


def make_source(d: Path, per_layer_shards: bool = False) -> Path:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3_moe

    d.mkdir(parents=True, exist_ok=True)
    mx.random.seed(0)
    model = qwen3_moe.Model(qwen3_moe.ModelArgs.from_dict(CFG))
    hf = {}
    for k, v in tree_flatten(model.parameters()):
        v = v.astype(mx.bfloat16)
        if ".switch_mlp." in k:
            pre, _, rest = k.partition(".switch_mlp.")
            for e in range(v.shape[0]):
                hf[f"{pre}.experts.{e}.{rest}"] = v[e]
        else:
            hf[k] = v
    mx.eval(hf)
    (d / "config.json").write_text(json.dumps(CFG))
    (d / "tokenizer_config.json").write_text("{}")
    if not per_layer_shards:
        mx.save_safetensors(str(d / "model.safetensors"), hf)
        return d
    groups: dict[str, dict] = {}
    for k, v in hf.items():
        key = ".".join(k.split(".")[:3]) if k.startswith("model.layers.") else "other"
        groups.setdefault(key, {})[k] = v
    names = sorted(groups)
    wm = {}
    for i, key in enumerate(names):
        fn = f"model-{i + 1:05d}-of-{len(names):05d}.safetensors"
        mx.save_safetensors(str(d / fn), groups[key])
        wm.update({k: fn for k in groups[key]})
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    return d


def _tensors(out: Path) -> dict:
    import mlx.core as mx

    w = {}
    for p in sorted(out.glob("model*.safetensors")):
        w.update(mx.load(str(p)))
    return w


def _same(a: dict, b: dict) -> None:
    import mlx.core as mx

    assert set(a) == set(b), (sorted(set(a) ^ set(b)))[:5]
    for k in a:
        assert a[k].dtype == b[k].dtype, (k, a[k].dtype, b[k].dtype)
        assert a[k].shape == b[k].shape, (k, a[k].shape, b[k].shape)
        assert mx.array_equal(a[k], b[k]).item(), k


def test_standard_matches_mlx_lm(tmp_path=None):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model, quantize_model

    from pagedmoe.convert import convert

    tmp = Path(tmp_path or tempfile.mkdtemp())
    src = make_source(tmp / "src")
    rep = convert(src, tmp / "out", "standard", measure=False)
    assert rep["status"] == "done" and rep["verify"]["mlx_lm_loads"], rep.get("verify")

    model, config = load_model(src, lazy=True)
    model, qcfg = quantize_model(model, config, 64, 4)
    ref = dict(tree_flatten(model.parameters()))
    _same(_tensors(tmp / "out"), ref)
    cfg = json.loads((tmp / "out" / "config.json").read_text())
    assert cfg["quantization"]["bits"] == 4 and cfg["quantization"]["group_size"] == 64
    # the router stays 8-bit, the architecture asks for it
    assert cfg["quantization"]["model.layers.0.mlp.gate"]["bits"] == 8
    mx.clear_cache()


def test_mixed_recipe_loads(tmp_path=None):
    import mlx.core as mx
    from mlx_lm.utils import load_model

    from pagedmoe.convert import Recipe, convert

    tmp = Path(tmp_path or tempfile.mkdtemp())
    src = make_source(tmp / "src")
    rec = Recipe.from_preset("balanced", edge_layers=1, experts_down=5,
                             rules=[{"match": r"layers\.1\.self_attn\.o_proj", "bits": "keep"},
                                    {"match": r"lm_head", "mode": "mxfp8"}])
    rep = convert(src, tmp / "out", rec)
    assert rep["status"] == "done"
    cfg = json.loads((tmp / "out" / "config.json").read_text())
    q = cfg["quantization"]

    def at(path):
        return q.get(path) or {k: q[k] for k in ("group_size", "bits", "mode")}

    assert at("model.layers.1.mlp.switch_mlp.down_proj")["bits"] == 5
    assert at("model.layers.1.mlp.switch_mlp.up_proj")["bits"] == 4
    assert at("model.layers.1.self_attn.q_proj")["bits"] == 6
    assert at("model.layers.0.self_attn.q_proj")["bits"] == 8  # edge layer
    assert at("lm_head")["mode"] == "mxfp8"
    w = _tensors(tmp / "out")
    assert w["model.layers.1.self_attn.o_proj.weight"].dtype == mx.bfloat16
    assert "model.layers.1.self_attn.o_proj.scales" not in w

    ref, _ = load_model(src)
    got, _ = load_model(tmp / "out")
    x = mx.array([[1, 5, 9, 33, 200, 17, 4, 99]])
    a, b = ref(x).astype(mx.float32), got(x).astype(mx.float32)
    cos = (mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b))).item()
    assert cos > 0.98, cos
    snr = rep["weight_error"]["components"]
    assert snr["attention"]["snr_db"] > snr["experts"]["snr_db"]  # 6-bit beats 4-bit


def test_resume_gives_same_bytes(tmp_path=None):
    from pagedmoe.convert import convert
    from pagedmoe.convert.run import STATE

    tmp = Path(tmp_path or tempfile.mkdtemp())
    src = make_source(tmp / "src")
    convert(src, tmp / "clean", "balanced")
    polls = {"n": 0}

    def stop_soon():
        polls["n"] += 1
        return polls["n"] > 12

    rep = convert(src, tmp / "out", "balanced", should_stop=stop_soon)
    assert rep["status"] == "stopped" and (tmp / "out" / STATE).exists()
    assert 0 < rep["done"] < rep["units"]
    try:
        convert(src, tmp / "out", "compact")
        raise AssertionError("a different recipe must not resume someone else's run")
    except RuntimeError as e:
        assert "different settings" in str(e)
    rep = convert(src, tmp / "out", "balanced")
    assert rep["status"] == "done" and not (tmp / "out" / STATE).exists()
    assert (tmp / "out" / "model.safetensors").read_bytes() == (tmp / "clean" / "model.safetensors").read_bytes()


def test_streamed_units_match_whole(tmp_path=None):
    from pagedmoe.convert import convert
    from pagedmoe.convert import plan as P
    from pagedmoe.convert.source import open_source

    tmp = Path(tmp_path or tempfile.mkdtemp())
    src_dir = make_source(tmp / "src", per_layer_shards=True)
    convert(src_dir, tmp / "whole", "balanced")

    fetched = []
    src = open_source(str(src_dir))
    src.kind, src.local_dir = "hub", None

    def fetch(file, dest, retries=4):
        Path(dest).mkdir(parents=True, exist_ok=True)
        fetched.append(file)
        return Path(shutil.copy(src_dir / file, Path(dest) / file))

    src.fetch = fetch
    plan = P.make(src, "balanced", out_dir=str(tmp / "streamed"))
    assert plan.windowed and plan.separable
    assert [u.key for u in plan.units] == ["other", "model.layers.1", "model.layers.2"]
    rep = convert(src, tmp / "streamed", plan=plan)
    assert rep["status"] == "done"
    assert sorted(fetched) == sorted(src.files)
    assert not (tmp / "streamed" / ".source").exists()
    _same(_tensors(tmp / "streamed"), _tensors(tmp / "whole"))


def test_adapter_merges_like_fuse(tmp_path=None):
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import load_model

    from pagedmoe.convert import convert

    tmp = Path(tmp_path or tempfile.mkdtemp())
    src = make_source(tmp / "src")
    model, _ = load_model(src)
    model.freeze()
    lp = {"rank": 4, "scale": 10.0, "dropout": 0.0,
          "keys": ["self_attn.q_proj", "self_attn.v_proj", "mlp.switch_mlp.up_proj"]}
    linear_to_lora_layers(model, 2, lp)
    mx.random.seed(1)
    for name, m in model.named_modules():
        if hasattr(m, "lora_b"):
            m.lora_b = (0.05 * mx.random.normal(m.lora_b.shape)).astype(m.lora_b.dtype)
    ad = tmp / "adapter"
    ad.mkdir()
    mx.save_safetensors(str(ad / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    (ad / "adapter_config.json").write_text(json.dumps(
        {"fine_tune_type": "lora", "num_layers": 2, "lora_parameters": lp}))

    fused = [(n, m.fuse()) for n, m in model.named_modules() if hasattr(m, "fuse")]
    model.update_modules(tree_unflatten(fused))
    ref = dict(tree_flatten(model.parameters()))

    rep = convert(src, tmp / "out", "float", adapter=str(ad))
    assert rep["adapter"]["modules"] == 6
    _same(_tensors(tmp / "out"), ref)


def test_quantized_source_requantizes(tmp_path=None):
    import mlx.core as mx
    from mlx_lm.utils import load_model

    from pagedmoe.convert import convert

    tmp = Path(tmp_path or tempfile.mkdtemp())
    src = make_source(tmp / "src")
    convert(src, tmp / "q8", "max", measure=False)
    rep = convert(tmp / "q8", tmp / "q4", "standard")
    assert rep["status"] == "done" and rep["verify"]["mlx_lm_loads"]
    convert(src, tmp / "direct", "standard", measure=False)
    x = mx.array([[3, 1, 4, 1, 5, 9, 2, 6]])
    a = load_model(src)[0](x).astype(mx.float32)

    def cos(path):
        b = load_model(path)[0](x).astype(mx.float32)
        return (mx.sum(a * b) / (mx.linalg.norm(a) * mx.linalg.norm(b))).item()

    # 8-bit first, then 4-bit: about as close as going straight to 4-bit
    assert cos(tmp / "q4") > cos(tmp / "direct") - 0.01


def test_bad_input_fails_early(tmp_path=None):
    from pagedmoe.convert import PlanError, Recipe, RecipeError, make_plan
    from pagedmoe.convert.source import SourceError

    tmp = Path(tmp_path or tempfile.mkdtemp())
    src = make_source(tmp / "src")
    cfg = dict(CFG, num_hidden_layers=4)
    (src / "config.json").write_text(json.dumps(cfg))
    try:
        make_plan(src, "balanced")
        raise AssertionError("a checkpoint missing a layer must not plan")
    except PlanError as e:
        assert "doesn't match" in str(e)
    for bad in ({"experts": 7}, {"group_size": 48}, {"rules": [{"match": "("}]}, {"exprts": 4}):
        try:
            Recipe.from_dict(bad)
            raise AssertionError(f"{bad} must not validate")
        except RecipeError:
            pass
    try:
        make_plan(tmp / "nope", "balanced")
        raise AssertionError("a missing folder must not plan")
    except SourceError as e:
        assert "doesn't exist" in str(e)


def test_plan_sizes_and_fit(tmp_path=None):
    from pagedmoe.convert import compare, make_plan

    tmp = Path(tmp_path or tempfile.mkdtemp())
    src = make_source(tmp / "src")
    p = make_plan(src, "fit", out_dir=str(tmp / "o"))
    s = p.summary()
    assert s["fit_choice"]["picked"] == "max"  # tiny, everything fits
    sizes = {r["preset"]: r["out_bytes"] for r in compare(p)}
    assert sizes["max"] > sizes["quality"] > sizes["balanced"] > sizes["standard"] > sizes["compact"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
