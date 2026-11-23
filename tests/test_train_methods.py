# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Every paged-training method runs end to end through the public Trainer API
on a tiny random Qwen3-MoE (4-bit) with a byte-level stand-in tokenizer:

  sft / cpt / orpo / dpo / simpo / cpo / kto / grpo, resident
  sft and orpo streamed (starved cache, layered executor)
  resume continues from the saved step with the same weights
  the saved adapter loads back through mlx_lm and pagedmoe.load
  custom inputs: per-row embeds and a trainable encoder

    python tests/test_train_methods.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from test_stream import CONFIG, build_tiny_model

from pagedmoe.train import (  # noqa: E402
    CPOConfig,
    CPOTrainer,
    CPTConfig,
    CPTTrainer,
    DPOConfig,
    DPOTrainer,
    Example,
    GRPOConfig,
    GRPOTrainer,
    KTOConfig,
    KTOTrainer,
    LoRAConfig,
    ORPOConfig,
    ORPOTrainer,
    SFTConfig,
    SFTTrainer,
    SimPOConfig,
    SimPOTrainer,
)

H = CONFIG["hidden_size"]


class ByteTok:
    """Byte-level ids 2..257, eos 1, pad 0, with a toy chat template."""

    eos_token_id = 1
    eos_token_ids = {1}
    pad_token_id = 0
    is_fast = False
    chat_template = "toy"

    def encode(self, text, add_special_tokens=False):
        return [2 + b for b in text.encode()]

    def decode(self, ids):
        return bytes(i - 2 for i in ids if 2 <= i < 258).decode(errors="ignore")

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        s = "".join(f"<{m['role']}>{m['content']}|" for m in messages)
        return s + ("<assistant>" if add_generation_prompt else "")


def rows_chat(n=24):
    return [{"messages": [{"role": "user", "content": f"add {i} and {i}"},
                          {"role": "assistant", "content": f"{2 * i}"}]} for i in range(n)]


def rows_pairs(n=16):
    return [{"prompt": f"say yes {i}", "chosen": "yes", "rejected": "no way"} for i in range(n)]


def rows_kto(n=16):
    return [{"prompt": f"q{i}", "completion": "good" if i % 2 else "bad", "label": bool(i % 2)}
            for i in range(n)]


GLM_CONFIG = {
    "model_type": "glm4_moe", "vocab_size": 512, "hidden_size": 128, "intermediate_size": 256,
    "max_position_embeddings": 2048, "moe_intermediate_size": 64, "norm_topk_prob": True,
    "num_attention_heads": 4, "n_group": 1, "head_dim": 32, "topk_group": 1,
    "n_shared_experts": 1, "n_routed_experts": 16, "routed_scaling_factor": 1.0,
    "num_experts_per_tok": 4, "first_k_dense_replace": 1, "num_hidden_layers": 4,
    "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "rope_scaling": None,
    "use_qk_norm": True, "tie_word_embeddings": False, "attention_bias": False,
    "partial_rotary_factor": 0.5, "quantization": {"group_size": 32, "bits": 4},
}


def build_tiny_glm(dest: Path):
    """GLM-4 MoE: its model.layers is a fresh slice on every access (mlx-lm's
    pipeline models, like DeepSeek V3 and Kimi-K2)."""
    from mlx_lm.models import glm4_moe

    mx.random.seed(11)
    model = glm4_moe.Model(glm4_moe.ModelArgs.from_dict(GLM_CONFIG))
    nn.quantize(model, group_size=32, bits=4)
    mx.save_safetensors(str(dest / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (dest / "config.json").write_text(json.dumps(GLM_CONFIG))
    # the loader always loads a tokenizer, and transformers has no file-less
    # default for glm4_moe (the trainer here uses ByteTok anyway)
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel({"<unk>": 0, **{chr(97 + i): 1 + i for i in range(26)}}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(dest / "tokenizer.json"))
    (dest / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}))


def common(tmp, name, **kw):
    base = dict(model=str(tmp / "model"), output_dir=str(tmp / name), probe="never",
                max_steps=3, batch_size=4, learning_rate=5e-3, save_steps=None,
                save_minutes=None, lora=LoRAConfig(rank=4, alpha=8), verbose=False, warmup_ratio=0.0)
    base.update(kw)
    return base


def lora_np(trainer):
    return {k: np.asarray(v) for k, v in tree_flatten(trainer.model.trainable_parameters())}


def run(cls, cfg_cls, tmp, name, data, **kw):
    extra = {k: kw.pop(k) for k in list(kw) if k in ("reward_funcs", "formatter", "encoder")}
    cfg = cfg_cls(**common(tmp, name, data=data, **kw))
    t = cls(cfg, tokenizer=ByteTok(), **extra)
    before = lora_np(t)
    summary = t.train()
    after = lora_np(t)
    moved = max(float(np.abs(after[k] - before[k]).max()) for k in before)
    lines = [json.loads(x) for x in (Path(tmp) / name / "metrics.jsonl").read_text().splitlines()]
    losses = [r["loss"] for r in lines if r["kind"] == "step"]
    assert all(np.isfinite(losses)), losses
    assert moved > 0, f"{name}: LoRA did not move"
    assert (Path(tmp) / name / "adapters.safetensors").exists()
    print(f"{name}: {summary['steps']} steps, losses {[round(x, 4) for x in losses]}, "
          f"plan {t.plan.mode}/{t.plan.executor}")
    return t


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "model").mkdir()
        build_tiny_model(tmp / "model")

        run(SFTTrainer, SFTConfig, tmp, "sft", rows_chat())
        run(CPTTrainer, CPTConfig, tmp, "cpt", [{"text": "the quick brown fox " * 20}] * 8,
            max_seq_len=64)
        run(ORPOTrainer, ORPOConfig, tmp, "orpo", rows_pairs())
        run(DPOTrainer, DPOConfig, tmp, "dpo", rows_pairs())
        run(DPOTrainer, DPOConfig, tmp, "ipo", rows_pairs(), loss_type="ipo")
        run(SimPOTrainer, SimPOConfig, tmp, "simpo", rows_pairs())
        run(CPOTrainer, CPOConfig, tmp, "cpo", rows_pairs())
        run(KTOTrainer, KTOConfig, tmp, "kto", rows_kto())

        def count_a(completions, prompts, **cols):
            return [c.count("a") / 10.0 for c in completions]

        def longish(completion, row):
            return min(1.0, len(completion) / 8.0)

        run(GRPOTrainer, GRPOConfig, tmp, "grpo", [{"prompt": f"write {i}"} for i in range(8)],
            reward_funcs=[count_a, longish], num_generations=4, max_completion_tokens=12,
            batch_size=2, beta=0.04, iterations=2)

        # Pipeline-style layer list (GLM-4, DeepSeek V3, Kimi-K2).
        (tmp / "glm").mkdir()
        build_tiny_glm(tmp / "glm")
        run(SFTTrainer, SFTConfig, tmp, "sft_glm", rows_chat(), model=str(tmp / "glm"))
        t = run(SFTTrainer, SFTConfig, tmp, "sft_glm_streamed", rows_chat(), model=str(tmp / "glm"),
                mode="streamed", cache_gb=0.0003)
        assert t.plan.executor == "layered" and t.streamed

        # Streamed: starved cache, so the layered executor reads inside backward.
        t = run(SFTTrainer, SFTConfig, tmp, "sft_streamed", rows_chat(), mode="streamed",
                cache_gb=0.0003)
        assert t.plan.executor == "layered" and t.streamed
        run(ORPOTrainer, ORPOConfig, tmp, "orpo_streamed", rows_pairs(), mode="streamed",
            cache_gb=0.0003)
        run(GRPOTrainer, GRPOConfig, tmp, "grpo_streamed", [{"prompt": f"write {i}"} for i in range(8)],
            reward_funcs=[longish], num_generations=4, max_completion_tokens=8, batch_size=2,
            mode="streamed", cache_gb=0.0003)

        # Streamed with an auto cache: the probe refits it to the measured
        # step (here every expert fits), and a cached profile refits the same.
        import os

        from pagedmoe.train import plan as P

        os.environ["PAGED_MOE_TRAIN_DIR"] = str(tmp / "train_dir")
        t = run(SFTTrainer, SFTConfig, tmp, "sft_auto_cache", rows_chat(), mode="streamed", probe="always")
        assert "cache_bytes" in t.plan.probe and t.plan.probe["source"] == "measured", t.plan.probe
        lo, hi = P.min_cache_bytes(t.facts), P.max_cache_bytes(t.facts)
        assert lo <= t.cache.budget_bytes <= hi, (lo, t.cache.budget_bytes, hi)
        assert t.plan.probe["step_peak"] <= t.plan.budget_gb * 1e9
        t2 = run(SFTTrainer, SFTConfig, tmp, "sft_auto_cache2", rows_chat(), mode="streamed", probe="auto")
        assert t2.plan.probe["source"] == "cached" and t2.cache.budget_bytes == t.cache.budget_bytes
        del os.environ["PAGED_MOE_TRAIN_DIR"]
        print(f"auto cache: {t.cache.budget_bytes / 1e6:.1f} MB of {t.facts.expert_bytes / 1e6:.1f} MB experts")

        # Resume: 2 steps with a save each step, then continue to 4.
        cfg = SFTConfig(**common(tmp, "resume", data=rows_chat(), max_steps=2, save_steps=1))
        a = SFTTrainer(cfg, tokenizer=ByteTok())
        a.train()
        saved = lora_np(a)
        cfg2 = SFTConfig(**common(tmp, "resume", data=rows_chat(), max_steps=4, save_steps=1,
                                  resume=True))
        b = SFTTrainer(cfg2, tokenizer=ByteTok())
        assert b.state["step"] == 2, b.state
        now = lora_np(b)
        assert all(np.array_equal(saved[k], now[k]) for k in saved), "resume lost weights"
        b.train()
        assert b.state["step"] == 4
        print("resume: ok")

        # The adapter loads with stock mlx-lm and with the engine loader.
        from mlx_lm.utils import load_model

        from mlx_lm.tuner.utils import load_adapters

        from pagedmoe import load

        m, _ = load_model(tmp / "model")
        load_adapters(m, str(tmp / "resume"))
        got = dict(tree_flatten(m.trainable_parameters()))
        assert got, "no adapter weights after mlx_lm load_adapters"
        ref = mx.load(str(tmp / "resume" / "adapters.safetensors"))
        k0 = next(iter(ref))
        assert np.allclose(np.asarray(got[k0]), np.asarray(ref[k0]))
        load(str(tmp / "model"), adapter_path=str(tmp / "resume"), verbose=False)
        print("adapter round trip: ok")

        # Custom inputs: a fixed per-row prefix, and a trainable encoder.
        def fmt_embeds(row, f):
            ids = [0] * 4 + f.encode(row["target"]) + [1]
            mask = [0] * 4 + [1] * (len(ids) - 4)
            return Example(ids, mask, embeds=np.full((4, H), row["v"], np.float32), embeds_at=0, row=row)

        data = [{"target": "abc", "v": 0.1}, {"target": "xyz", "v": -0.1}] * 8
        run(SFTTrainer, SFTConfig, tmp, "embeds", data, formatter=fmt_embeds)

        class Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(3, H)

            def __call__(self, batch):
                pts = mx.array(np.stack([np.full((4, 3), r["v"], np.float32) for r in batch.rows]))
                return self.proj(pts)

        def fmt_enc(row, f):
            ids = [0] * 4 + f.encode(row["target"]) + [1]
            mask = [0] * 4 + [1] * (len(ids) - 4)
            return Example(ids, mask, embeds_at=0, row=row)

        enc = Enc()
        w0 = np.asarray(enc.proj.weight)
        run(SFTTrainer, SFTConfig, tmp, "encoder", data, formatter=fmt_enc, encoder=enc)
        assert not np.allclose(w0, np.asarray(enc.proj.weight)), "encoder did not train"
        t = run(SFTTrainer, SFTConfig, tmp, "encoder_streamed", data, formatter=fmt_enc,
                encoder=Enc(), mode="streamed", cache_gb=0.0003)
        print("custom inputs: ok")
    print("OK")


def test_train_methods():
    main()


if __name__ == "__main__":
    main()
