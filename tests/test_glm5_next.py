# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
glm5_next (GLM-5.3-Flash) loads through the bundled arches port and pages
experts through StreamedSwitchGLU.

Uses the public tiny HF fixture (downloaded once to /tmp/glm53-tiny-hf),
skips without it. From it this builds a checkpoint laid out like
mlx-community/GLM-5.3-Flash-4bit: per-expert tensors under
model.language_model., 4-bit with 5-bit expert down and 6-bit shared experts
keyed on model.layers.N, bf16 linear attention / kv_b_proj / mHC fn, an
extra MTP decoder layer and vision weights that must be dropped.

Checks: resident and streamed agree, fused prefill (mHC sub-chunks) matches
one-shot prefill, and the quantized model stays close to the bf16 one.

    python tests/test_glm5_next.py
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURE = Path("/tmp/glm53-tiny-hf")
REPO = "inference-optimization/GLM-5.3-Flash-0.1B-A0.1B"
FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "chat_template.jinja"]


def _ensure_fixture() -> Path | None:
    if (FIXTURE / "config.json").is_file() and (FIXTURE / "model.safetensors").is_file():
        return FIXTURE
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(REPO, local_dir=str(FIXTURE), allow_patterns=FILES)
    except Exception as e:
        print(f"skip: could not get {REPO} ({e})")
        return None
    return FIXTURE if (FIXTURE / "model.safetensors").is_file() else None


# (pattern on the bare text key, bits); first match wins, None = keep bf16
_QUANT = [
    (r"mlp\.experts\.\d+\.down_proj$", 5),
    (r"mlp\.experts\.\d+\.(gate|up)_proj$", 4),
    (r"mlp\.shared_experts\.\w+_proj$", 6),
    (r"^layers\.[0-2]\.mlp\.down_proj$", 5),
    (r"^layers\.[0-2]\.mlp\.(gate|up)_proj$", 4),
    (r"self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa)$", 4),
]


def _build_mlx_layout(src: Path, dst: Path) -> None:
    import mlx.core as mx

    cfg = json.loads((src / "config.json").read_text())
    tc = cfg["text_config"]
    n = tc["num_hidden_layers"]
    w = mx.load(str(src / "model.safetensors"))
    sparse = {i for i, t in enumerate(tc["layer_types"]) if t != "linear_attention"}
    out, qcfg = {}, {"group_size": 64, "bits": 4}
    for k, v in w.items():
        if not k.startswith("model.language_model."):
            out[k] = v
            continue
        bare = k[len("model.language_model."):]
        mod, _, leaf = bare.rpartition(".")
        bits = None
        if leaf == "weight" and v.ndim == 2:
            for pat, b in _QUANT:
                if re.search(pat, mod):
                    bits = b
                    break
            # mlx-community quantizes o_proj only on the sparse (MLA) layers
            m = re.match(r"layers\.(\d+)\.self_attn\.o_proj$", mod)
            if m and int(m.group(1)) in sparse:
                bits = 4
        if bits is None:
            out[k] = v
            continue
        qw, s, b = mx.quantize(v, group_size=64, bits=bits)
        prefix = k[: -len(".weight")]
        out[prefix + ".weight"], out[prefix + ".scales"], out[prefix + ".biases"] = qw, s, b
        if bits != 4:
            key = "model." + re.sub(r"experts\.\d+\.", "switch_mlp.", mod)
            qcfg[key] = {"group_size": 64, "bits": bits}
    # The nextn block is stored as one more decoder layer, which the backbone
    # sanitize must drop. Like layer 45 of the real checkpoint it is a sparse
    # attention MoE layer with no mHC weights, plus the nextn projections.
    proxy = max(i for i in sparse if tc["mlp_layer_types"][i] == "sparse")
    src_pre, mtp_pre = f"model.language_model.layers.{proxy}.", f"model.language_model.layers.{n}."
    for k in [k for k in out if k.startswith(src_pre) and ".hc_" not in k]:
        out[mtp_pre + k[len(src_pre):]] = out[k]
    for k, b in list(qcfg.items()):
        if isinstance(b, dict) and k.startswith(f"model.layers.{proxy}."):
            qcfg[k.replace(f".layers.{proxy}.", f".layers.{n}.")] = b
    H = tc["hidden_size"]
    mx.random.seed(7)
    out[mtp_pre + "eh_proj.weight"] = (0.05 * mx.random.normal((H, 2 * H))).astype(mx.bfloat16)
    for name in ("enorm", "hnorm", "shared_head.norm"):
        out[mtp_pre + name + ".weight"] = mx.ones((H,), dtype=mx.bfloat16)
    dst.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst / "model.safetensors"), out)
    cfg["quantization"] = cfg["quantization_config"] = qcfg
    (dst / "config.json").write_text(json.dumps(cfg))
    for name in FILES[2:]:
        if (src / name).is_file():
            (dst / name).write_bytes((src / name).read_bytes())
    index = {
        "metadata": {"total_size": int(sum(v.nbytes for v in out.values()))},
        "weight_map": {k: "model.safetensors" for k in out},
    }
    (dst / "model.safetensors.index.json").write_text(json.dumps(index))


def _rel(a, b) -> float:
    import mlx.core as mx

    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float(mx.max(mx.abs(a - b)) / (mx.max(mx.abs(b)) + 1e-9))


def main():
    import mlx.core as mx
    import numpy as np

    from pagedmoe import load, prefill_fused, config
    from pagedmoe.streaming import StreamedSwitchGLU, find_switch_glus

    src = _ensure_fixture()
    if src is None:
        return
    ids = mx.array(np.random.default_rng(0).integers(100, 150000, size=(1, 150)))
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "mlx"
        _build_mlx_layout(src, dst)

        bf16, _ = load(str(src), mode="resident")
        base = bf16(ids)
        mx.eval(base)
        del bf16

        res, _ = load(str(dst), mode="resident")
        layer = res.model.layers[3].mlp
        assert layer.switch_mlp.down_proj.bits == 5, layer.switch_mlp.down_proj.bits
        assert layer.shared_experts.down_proj.bits == 6
        assert res.model.layers[3].mlp.switch_mlp.activation.limit == 10.0
        r_logits = res(ids)
        mx.eval(r_logits)
        q_err = _rel(r_logits, base)
        assert q_err < 0.25, f"quantized drifted from bf16: {q_err}"
        del res

        model, tok = load(str(dst), lazy=True, mode="streamed", cache_gb=0.05)
        assert find_switch_glus(model) == []
        streamed = [m for _, m in model.named_modules() if isinstance(m, StreamedSwitchGLU)]
        assert len(streamed) == 2, len(streamed)
        s_logits = model(ids)
        mx.eval(s_logits)
        s_err = _rel(s_logits, r_logits)
        assert s_err < 1e-2, f"streamed vs resident: {s_err}"

        # fused prefill: attention in 40-row sub-chunks, mHC on each slice
        prefill_fused.uninstall()
        was = config.ATTN_SUB_CHUNK
        config.ATTN_SUB_CHUNK = 40
        try:
            assert prefill_fused.install(model)
            cache = model.make_cache()
            f_logits = model(ids, cache=cache)
            mx.eval(f_logits)
            f_err = _rel(f_logits, s_logits)
            assert f_err < 1e-2, f"fused prefill vs one-shot: {f_err}"
            nxt = mx.argmax(f_logits[:, -1:], axis=-1)
            step = model(nxt, cache=cache)
            mx.eval(step)
            assert bool(mx.all(mx.isfinite(step)))
        finally:
            config.ATTN_SUB_CHUNK = was
            prefill_fused.uninstall()

        print(f"glm5_next: q-vs-bf16 {q_err:.3g}, streamed-vs-resident {s_err:.3g}, "
              f"fused-vs-oneshot {f_err:.3g}")
        del model
        mtp_drafts(dst, Path(tmp))
        train(dst, Path(tmp))
    print("OK")


def indexer_after_trim(model, topk: int) -> None:
    """A verify pass that is cut short trims the sparse attention cache, then
    plain decode goes on one token at a time. That step has to match a cache
    that never saw the rejected tokens, past index_topk where the indexer's
    reused pools come into play."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    i = next(i for i, l in enumerate(model.layers) if not l.is_linear)
    attn = model.layers[i].self_attn
    H = attn.hidden_size
    mx.random.seed(11)
    # (n + 3) // kpool > (n + 1) // kpool, so a stale pool would hold rejected keys
    n = topk + 21
    prefix = mx.random.normal((1, n, H))
    kept, rejected = mx.random.normal((1, 1, H)), mx.random.normal((1, 2, H))
    step = mx.random.normal((1, 1, H))

    picked = []
    cls = type(attn.indexer)
    orig = cls.__call__

    def spy(self, *a, **k):
        out = orig(self, *a, **k)
        picked.append(out)
        return out

    def feed(cache, x):
        mask = create_attention_mask(x, cache[0], return_array=True) if x.shape[1] > 1 else None
        return attn(x, mask, cache)

    cls.__call__ = spy
    try:
        trimmed = model.mtp_cache()
        feed(trimmed, prefix)
        feed(trimmed, mx.concatenate([kept, rejected], axis=1))
        trimmed.trim(2)
        got = feed(trimmed, step)
        got_sel = picked[-1]
        fresh = model.mtp_cache()
        feed(fresh, mx.concatenate([prefix, kept], axis=1))
        # Same full pooling the trimmed cache falls back to.
        fresh[1]._pool = None
        want = feed(fresh, step)
        want_sel = picked[-1]
    finally:
        cls.__call__ = orig
    # The kept token's latent was computed in a different batch shape, so the
    # attention output only matches to rounding. The selection has to be exact.
    assert mx.array_equal(got_sel, want_sel), "indexer picked different positions after trim"
    err = _rel(got, want)
    assert err < 1e-4, f"step after trim off by {err:.3g}"


def mtp_drafts(model_path: Path, tmp: Path) -> None:
    """The nextn head builds from the checkpoint and drafting never changes
    output. The prompt runs past index_topk so the indexer selects, and the
    drafts are half wrong so verify passes trim the main and head caches."""
    import os

    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import make_prompt_cache

    from pagedmoe import config, load, mtp
    from pagedmoe import verify_decode as vd

    os.environ["PAGED_MOE_MTP_DIR"] = str(tmp / "heads")
    est = mtp.estimate_bytes(model_path)
    model, _ = load(str(model_path), mode="resident")
    drafter = getattr(model, "_pagedmoe_mtp", None)
    assert drafter is not None, "nextn head did not attach"
    assert est == drafter.nbytes == mtp.estimate_bytes(model_path), (est, drafter.nbytes)

    # The arch's own head budget applies unless the env var is set.
    from pagedmoe.loader import _mtp_max_frac

    frac = _mtp_max_frac(model)
    assert frac == max(config.MTP_MAX_FRAC, model.mtp_max_frac), frac
    # A cache the head can't fit in skips it before anything is read.
    built = mtp.prepare
    mtp.prepare = lambda *a, **k: (_ for _ in ()).throw(AssertionError("head built anyway"))
    try:
        small, _ = load(str(model_path), mode="streamed",
                        cache_gb=0.5 * est / frac / 1e9)
    finally:
        mtp.prepare = built
    assert getattr(small, "_pagedmoe_mtp", None) is None
    del small
    assert type(drafter.head.layers[0]).__name__ == "Glm5NextMTPLayer"
    topk = json.loads((model_path / "config.json").read_text())["text_config"]["index_topk"]
    prompt = np.random.default_rng(1).integers(100, 150000, size=topk + 20).tolist()

    def run(n):
        cache = make_prompt_cache(model)
        mx.eval(model(mx.array([prompt[:-1]]), cache=cache))
        return [int(t) for t, _ in vd.generate_step(
            mx.array(prompt[-1:], dtype=mx.uint32), model, max_tokens=n,
            prompt_cache=cache, history=prompt[:-1],
        )]

    model._pagedmoe_mtp = None
    ref = run(40)
    model._pagedmoe_mtp = drafter
    orig, made, fed = mtp.Drafter.draft, [], []

    def draft(self, tok, k):
        # The real head runs; its tokens are swapped for the answer, with
        # every other draft spoiled at the end.
        at = len(fed) + len(self.pend_t) + 1
        fed.extend(int(t) for t in self.pend_t)
        fed.append(int(tok))
        guess = orig(self, tok, k)
        if not guess:
            return guess
        made.append(len(guess))
        out = list(ref[at:at + len(guess)]) or guess
        if len(made) % 2 == 0:
            out[-1] = (out[-1] + 1) % 150000
        return out

    saved = (vd._Governor.ready, vd._Governor.want, vd._Index.query,
             config.MTP_SPECULATE, config.MTP_CATCHUP)
    vd._Governor.ready = lambda self, span: True
    # Every third chance is a plain step, so single-token decode follows trims.
    turns = iter(range(10**6))
    vd._Governor.want = lambda self, span, key: key[0] == "mtp" and next(turns) % 3 != 2
    vd._Index.query = lambda index, k: []
    config.MTP_SPECULATE, config.MTP_CATCHUP = False, 10**6
    mtp.Drafter.draft = draft
    try:
        got = run(40)
        st = dict(vd.STATS)
    finally:
        mtp.Drafter.draft = orig
        (vd._Governor.ready, vd._Governor.want, vd._Index.query,
         config.MTP_SPECULATE, config.MTP_CATCHUP) = saved
    assert fed == got[:len(fed)], "head pairs out of step with the output"
    assert int(drafter.cache.offset) == len(fed), (drafter.cache.offset, len(fed))
    indexer_after_trim(model, topk)
    assert made, "head never drafted"
    assert 0 < st["accepted"] < st["drafted"], f"need both accepts and rejects: {st}"
    assert got == ref, f"drafting changed output\n  ref={ref}\n  got={got}"
    print(f"glm5_next mtp: {len(made)} drafts, accept {st['accepted']}/{st['drafted']}, output identical")


def train(model_path: Path, tmp: Path) -> None:
    """LoRA SFT resident and streamed, and GRPO (generate, then train)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_train_methods import rows_chat, run

    from pagedmoe.train import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

    t = run(SFTTrainer, SFTConfig, tmp, "glm5_sft", rows_chat(), model=str(model_path),
            mode="resident")
    keys = t.lora_report.keys
    assert not any("indexer" in k for k in keys), keys
    assert any("self_attn.q_proj" in k for k in keys), keys  # KDA projections
    assert any("self_attn.q_b_proj" in k for k in keys), keys  # MLA projections
    assert t.chunked_delta

    # PEFT names come out in the HF layout (model.language_model.layers.N...)
    from pagedmoe.train.export import to_peft

    to_peft(tmp / "glm5_sft", base=str(FIXTURE), log=lambda *_: None)
    import mlx.core as mx

    peft = mx.load(str(tmp / "glm5_sft" / "peft" / "adapter_model.safetensors"))
    assert peft and all(".language_model.layers." in k for k in peft), list(peft)[:3]

    t = run(SFTTrainer, SFTConfig, tmp, "glm5_sft_streamed", rows_chat(),
            model=str(model_path), mode="streamed", cache_gb=0.0003)
    assert t.streamed and t.plan.executor == "layered"
    # 4 mHC streams cross every layer boundary
    hidden = json.loads((model_path / "config.json").read_text())["text_config"]["hidden_size"]
    assert t.executor.boundary_bytes == 4 * hidden * 2, t.executor.boundary_bytes

    def longish(completion, row):
        return min(1.0, len(completion) / 8.0)

    run(GRPOTrainer, GRPOConfig, tmp, "glm5_grpo", [{"prompt": f"write {i}"} for i in range(4)],
        model=str(model_path), reward_funcs=[longish], num_generations=2,
        max_completion_tokens=6, batch_size=2, mode="streamed", cache_gb=0.0003, max_steps=2)


def test_glm5_next():
    # train() puts the whole process in training mode (see conftest.py), so
    # under pytest this runs as its own process and the serving tests after
    # it start clean.
    import subprocess

    subprocess.run([sys.executable, str(Path(__file__).resolve())], check=True)


if __name__ == "__main__":
    main()
