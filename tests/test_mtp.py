# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
MTP head drafting (mtp.py) and fetching (mtp_fetch.py).

What has to hold:

  1. A head converts from HF-layout tensors (per-expert or fused experts,
     zero-centered norms) into exactly the weights the head module wants, is
     saved once, and loads back from the saved file bit for bit.
  2. The head is fed one (hidden, next token) pair per kept position, in
     order, with no gaps or repeats, across plain steps and verify passes.
  3. Drafting with the head never changes output: every token is the greedy
     pick for its prefix. Drafting inside the pass gives the same output and
     leaves the head cache as drafting after it does.
  4. A head that will not build marks itself incompatible and the model loads
     without it; a saved head that no longer loads is dropped.
  5. The fetch writes a loadable safetensors file of just the planned tensors,
     resumes after an interruption, and status / remove report what is on disk.

Run:
    python tests/test_mtp.py
"""

import importlib.util
import io
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from tests.test_lookup_hybrid import ARGS, build, check_greedy  # noqa: E402

from pagedmoe import config as es_config  # noqa: E402
from pagedmoe import mtp, mtp_fetch  # noqa: E402
from pagedmoe import verify_decode as vd  # noqa: E402

FLASH_PORTS = [
    Path.home() / "Desktop/models/Qwen3.8-Flash-Next-MLX-6bit/qwen4_exp.py",
    Path.home() / "mlx-models/qwen38-coder-flash/qwen4_exp.py",
]


def _model_dir(tmp: Path, cfg: dict, name="m") -> Path:
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg))
    return d


def _hf_layout(head_params: dict, fused: bool) -> dict:
    """Head weights as an HF checkpoint stores them: mtp.*, experts split, norms centered."""
    out = {}
    for k, v in head_params.items():
        m = k.split(".switch_mlp.")
        if len(m) == 2:
            proj, kind = m[1].split(".")
            if fused:
                continue
            for e in range(v.shape[0]):
                out[f"mtp.{m[0]}.experts.{e}.{proj}.{kind}"] = v[e]
            continue
        if v.ndim == 1 and k.endswith(".weight") and "norm" in k.rsplit(".", 2)[-2]:
            v = v - 1.0
        out["mtp." + k] = v
    if fused:
        for k, v in head_params.items():
            if k.endswith("switch_mlp.gate_proj.weight"):
                base = k[: -len("switch_mlp.gate_proj.weight")]
                up = head_params[base + "switch_mlp.up_proj.weight"]
                out[f"mtp.{base}experts.gate_up_proj"] = mx.concatenate([v, up], axis=1)
                out[f"mtp.{base}experts.down_proj"] = head_params[base + "switch_mlp.down_proj.weight"]
    return out


def _random_head(model, cfg, seed=1):
    parts = mtp._Parts(model, cfg)
    mx.random.seed(seed)
    head = mtp._build(parts)
    # Trained norms sit well above 1 (Flash's q_norm stores ~2.7 unshifted),
    # so the shift has to come from the source, not from the values.
    head.update(nn.utils.tree_map(
        lambda p: (1.5 + 2.0 * mx.random.uniform(shape=p.shape)) if p.ndim == 1 else p * 3.0,
        head.parameters(),
    ))
    mx.eval(head.parameters())
    return dict(tree_flatten(head.parameters()))


def _write_raw(d: Path, tensors: dict, layers: int) -> None:
    d.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(d / "raw.safetensors"), tensors)
    (d / "raw.json").write_text(json.dumps({"repo": "x/y", "layers": layers, "bytes": 1}))


def _same_params(a: dict, b: dict) -> float:
    assert set(a) == set(b), f"param names differ: {set(a) ^ set(b)}"
    return max(float(mx.abs(a[k].astype(mx.float32) - b[k].astype(mx.float32)).max()) for k in a)


def check_convert_roundtrip(tmp: Path):
    model = build()
    cfg = {**ARGS, "model_type": "qwen3_next"}
    os.environ["PAGED_MOE_MTP_DIR"] = str(tmp / "heads")
    for fused in (False, True):
        mdir = _model_dir(tmp, {**cfg, "fused": fused}, f"rt{int(fused)}")
        want = _random_head(model, cfg)
        d = mtp_fetch.head_dir(mdir)
        hf = _hf_layout(want, fused)
        for name in ("pre_fc_norm_hidden.weight", "pre_fc_norm_embedding.weight", "norm.weight"):
            assert mx.allclose(hf["mtp." + name] + 1.0, want[name]), f"{name} not stored centered"
        _write_raw(d, hf, ARGS["num_hidden_layers"])
        assert mtp_fetch.local_status(mdir)["state"] == "downloaded"
        prep = mtp.prepare(model, cfg, mdir)
        assert prep is not None, "head did not build"
        got = dict(tree_flatten(prep.head.parameters()))
        worst = _same_params(want, got)
        assert worst < 1e-6, f"converted head differs by {worst}"
        st = mtp_fetch.local_status(mdir)
        assert st["state"] == "ready", st
        assert not (d / "raw.safetensors").exists(), "raw tensors kept after conversion"
        again = mtp.prepare(model, cfg, mdir)
        worst = _same_params(got, dict(tree_flatten(again.head.parameters())))
        assert worst == 0.0, f"saved head reloads {worst} off"
    print("OK: per-expert and fused HF layouts convert to the head's weights, saved head reloads bit-identical")


def _quantize_like_checkpoint(model):
    nn.quantize(
        model, group_size=32, bits=4,
        class_predicate=lambda p, m: (
            not p.endswith("mlp.gate") and hasattr(m, "to_quantized") and m.weight.shape[-1] % 32 == 0
        ),
    )


def check_float_head_quantizes_after_load(tmp: Path):
    """A float head (the downloaded 6-bit case) still loads, then takes the backbone quant."""
    model = build()
    _quantize_like_checkpoint(model)
    cfg = {**ARGS, "model_type": "qwen3_next", "quantization": {"group_size": 32, "bits": 4}}
    os.environ["PAGED_MOE_MTP_DIR"] = str(tmp / "heads")
    parts = mtp._Parts(model, cfg)
    mx.random.seed(2)
    fresh = mtp._build(parts)
    fresh.update(nn.utils.tree_map(
        lambda p: mx.ones(p.shape) if p.ndim == 1 else p,
        fresh.parameters(),
    ))
    mx.eval(fresh.parameters())
    floats = dict(tree_flatten(fresh.parameters()))
    assert mtp._packed_quant_map(fresh, floats) is None
    mdir = _model_dir(tmp, {**cfg, "n": "floathead"}, "floathead")
    _write_raw(mtp_fetch.head_dir(mdir), _hf_layout(floats, False), ARGS["num_hidden_layers"])
    prep = mtp.prepare(model, cfg, mdir)
    assert prep is not None, "float head did not build"
    ref = mtp._build(parts)
    converted = mtp._convert(model, parts, _hf_layout(floats, False), ARGS["num_hidden_layers"], raw_hf=True)
    ref.load_weights(list(converted.items()), strict=True)
    mtp._apply_quant(ref, mtp._quant_map(parts, ref))
    mx.eval(ref.parameters())
    worst = _same_params(dict(tree_flatten(ref.parameters())), dict(tree_flatten(prep.head.parameters())))
    assert worst < 1e-6, f"float head diverged from load-then-quantize by {worst}"
    assert isinstance(prep.head.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(prep.head.layers[0].mlp.gate, nn.Linear)
    assert mtp_fetch.local_status(mdir)["state"] == "ready"
    print("OK: float heads still load before quantize and follow the backbone")


def check_packed_head(tmp: Path):
    """A head stored already quantized loads by quantizing first. A bad pack still marks incompatible."""
    model = build()
    _quantize_like_checkpoint(model)
    cfg = {**ARGS, "model_type": "qwen3_next", "quantization": {"group_size": 32, "bits": 4}}
    os.environ["PAGED_MOE_MTP_DIR"] = str(tmp / "heads")
    parts = mtp._Parts(model, cfg)
    mx.random.seed(3)
    fresh = mtp._build(parts)
    fresh.update(nn.utils.tree_map(
        lambda p: mx.ones(p.shape) if p.ndim == 1 else p,
        fresh.parameters(),
    ))
    # A small MLX norm must be kept as stored. The float download path is
    # what recenters; a packed checkpoint is already in backbone form.
    fresh.pre_fc_norm_embedding.weight = mx.full(fresh.pre_fc_norm_embedding.weight.shape, 0.2)
    qmap = mtp._quant_map(parts, fresh)
    mtp._apply_quant(fresh, qmap)
    mx.eval(fresh.parameters())
    packed = dict(tree_flatten(fresh.parameters()))
    assert any(k.endswith(".scales") for k in packed)
    recipe = mtp._packed_quant_map(mtp._build(parts), packed)
    assert recipe["layers.0.self_attn.q_proj"] == {"bits": 4, "group_size": 32, "mode": "affine"}, recipe
    assert recipe["layers.0.mlp.gate"] is None
    assert recipe["fc"]["bits"] == 4
    mdir = _model_dir(tmp, {**cfg, "n": "packed"}, "packed")
    tensors = {"mtp." + k: v for k, v in packed.items()}
    mx.save_safetensors(str(mdir / "model.safetensors"), tensors)
    (mdir / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {k: "model.safetensors" for k in tensors},
    }))
    # The stamp a previous load left on this kind of checkpoint.
    mtp_fetch._write_json(mtp_fetch.head_dir(mdir) / "head.json", {
        "state": "incompatible", "version": mtp_fetch.HEAD_VERSION,
        "error": "ValueError('Received 44 parameters not in model.')",
    })
    assert mtp_fetch.local_status(mdir)["state"] == "bundled"
    prep = mtp.prepare(model, cfg, mdir)
    assert prep is not None, "packed head did not build"
    worst = _same_params(packed, dict(tree_flatten(prep.head.parameters())))
    assert worst == 0.0, f"packed head reloaded {worst} off"
    assert isinstance(prep.head.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(prep.head.layers[0].mlp.gate, nn.Linear)
    assert mtp_fetch.local_status(mdir)["state"] == "ready"
    again = mtp.prepare(model, cfg, mdir)
    assert _same_params(packed, dict(tree_flatten(again.head.parameters()))) == 0.0

    bad = _model_dir(tmp, {**cfg, "n": "badpack"}, "badpack")
    junk = dict(tensors)
    junk["mtp.fc.scales"] = mx.zeros((3, 3))
    junk["mtp.fc.biases"] = mx.zeros((3, 3))
    mx.save_safetensors(str(bad / "model.safetensors"), junk)
    (bad / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {k: "model.safetensors" for k in junk},
    }))
    assert mtp.prepare(model, cfg, bad) is None
    st = mtp_fetch.local_status(bad)
    assert st["state"] == "incompatible", st
    assert "Not usable" not in (st.get("error") or "")
    # A second look does not retry a failure this builder just wrote.
    assert mtp_fetch.local_status(bad)["state"] == "incompatible"
    saved = json.loads((mtp_fetch.head_dir(bad) / "head.json").read_text())
    assert saved.get("builder") == 1
    print("OK: packed heads quantize before load; a bad pack stays incompatible")


def check_quant_mirror(tmp: Path):
    model = build()
    nn.quantize(model, group_size=32, bits=4, class_predicate=lambda p, m: not p.endswith("mlp.gate") and hasattr(m, "to_quantized") and m.weight.shape[-1] % 32 == 0)
    cfg = {**ARGS, "model_type": "qwen3_next", "quantization": {"group_size": 32, "bits": 4}}
    parts = mtp._Parts(model, cfg)
    head = mtp._build(parts)
    qmap = mtp._quant_map(parts, head)
    assert qmap["layers.0.mlp.gate"] is None, "router gate must stay float like the backbone"
    assert qmap["layers.0.self_attn.q_proj"] == {"bits": 4, "group_size": 32, "mode": "affine"}, qmap
    assert qmap["fc"] == {"bits": 4, "group_size": 32, "mode": "affine"}, "no analog: config default"
    mtp._apply_quant(head, qmap)
    assert isinstance(head.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(head.layers[0].mlp.gate, nn.Linear)
    print("OK: head quantization mirrors the backbone module by module, router stays float")


def _attached(tmp: Path, name: str):
    model = build()
    cfg = {**ARGS, "model_type": "qwen3_next"}
    os.environ["PAGED_MOE_MTP_DIR"] = str(tmp / "heads")
    mdir = _model_dir(tmp, {**cfg, "n": name}, name)
    _write_raw(mtp_fetch.head_dir(mdir), _hf_layout(_random_head(model, cfg, 4), False), ARGS["num_hidden_layers"])
    prep = mtp.prepare(model, cfg, mdir)
    drafter = mtp.attach(model, prep)
    return model, drafter


def _run(model, prompt, n):
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array([prompt[:-1]]), cache=cache))
    out = []
    for tok, _ in vd.generate_step(
        mx.array(prompt[-1:], dtype=mx.uint32), model, max_tokens=n,
        prompt_cache=cache, history=prompt[:-1],
    ):
        out.append(int(tok))
    return out, cache


class _Force:
    def __enter__(self):
        self.saved = (vd._Governor.ready, vd._Governor.want, vd._Index.query)
        vd._Governor.ready = lambda self, span: True
        vd._Governor.want = lambda self, span, key: key[0] == "mtp"
        vd._Index.query = lambda index, k: []
        return self

    def __exit__(self, *exc):
        vd._Governor.ready, vd._Governor.want, vd._Index.query = self.saved


def check_pairs_and_output(tmp: Path):
    model, drafter = _attached(tmp, "gen")
    rng = np.random.default_rng(3)
    prompt = rng.integers(0, ARGS["vocab_size"], size=30).tolist()
    model._pagedmoe_mtp = None
    ref, _ = _run(model, prompt, 48)
    model._pagedmoe_mtp = drafter
    fed = []
    orig = mtp.Drafter.draft

    def draft(self, tok, k):
        # The real head runs (its state is what is under test); the tokens it
        # returns are swapped for the answer with every third one spoiled, so
        # passes keep several rows and also reject.
        g = len(fed) + len(self.pend_t) + 1
        fed.extend(int(t) for t in mx.concatenate(self.pend_t + [mx.array([tok], dtype=mx.uint32)]).tolist())
        guess = orig(self, tok, k)
        if not guess:
            return guess
        out = list(ref[g:g + len(guess)])
        if len(fed) % 3 == 0 and out:
            out[-1] = (out[-1] + 1) % ARGS["vocab_size"]
        return out or guess

    mtp.Drafter.draft = draft
    old_cap, old_spec = es_config.MTP_CATCHUP, es_config.MTP_SPECULATE
    es_config.MTP_CATCHUP = 10**6
    es_config.MTP_SPECULATE = False
    try:
        with _Force():
            got, _ = _run(model, prompt, 40)
        st = dict(vd.STATS)
    finally:
        mtp.Drafter.draft = orig
        es_config.MTP_CATCHUP, es_config.MTP_SPECULATE = old_cap, old_spec
    assert st["mtp"] > 0 and st["passes"] > 0, f"head never drafted: {st}"
    assert 0 < st["accepted"] < st["drafted"], f"need both accepts and rejects: {st}"
    assert fed == got[: len(fed)], (
        f"head pairs out of step with the output\n  fed={fed}\n  got={got[:len(fed)]}"
    )
    assert len(fed) >= len(got) - 8, f"head fell behind: fed {len(fed)} of {len(got)}"
    check_greedy(model, prompt, got)
    off = int(drafter.cache.offset)
    assert off == len(fed), f"head cache holds {off} positions, {len(fed)} real pairs went in"
    print(f"OK: head sees every kept position once, in order ({st['mtp']} mtp passes, accept {st['accepted']}/{st['drafted']})")


def check_speculate(tmp: Path):
    """Drafting inside the pass leaves output and the head cache as drafting after it."""
    model, drafter = _attached(tmp, "spec")
    rng = np.random.default_rng(5)
    prompt = rng.integers(0, ARGS["vocab_size"], size=30).tolist()
    model._pagedmoe_mtp = None
    ref, _ = _run(model, prompt, 48)
    model._pagedmoe_mtp = drafter
    orig = mtp.Drafter.draft
    emitted = []

    def draft(self, tok, k):
        guess = orig(self, tok, k)
        if not guess:
            return guess
        g = len(emitted)
        out = list(ref[g:g + len(guess)])
        if g % 3 == 0 and out:
            out[-1] = (out[-1] + 1) % ARGS["vocab_size"]
        return out or guess

    def run(spec: bool):
        es_config.MTP_SPECULATE = spec
        emitted.clear()
        cache = make_prompt_cache(model)
        mx.eval(model(mx.array([prompt[:-1]]), cache=cache))
        for tok, _ in vd.generate_step(
            mx.array(prompt[-1:], dtype=mx.uint32), model, max_tokens=40,
            prompt_cache=cache, history=prompt[:-1],
        ):
            emitted.append(int(tok))
        off = int(drafter.cache.offset)
        keys = drafter.cache.keys[..., :off, :]
        mx.eval(keys)
        return list(emitted), off, keys, dict(vd.STATS)

    mtp.Drafter.draft = draft
    old = (es_config.MTP_CATCHUP, es_config.MTP_SPECULATE, vd._SpecGate.want)
    es_config.MTP_CATCHUP = 10**6
    vd._SpecGate.want = lambda self, conf: True
    try:
        with _Force():
            plain, off0, keys0, _ = run(False)
            got, off1, keys1, st = run(True)
    finally:
        mtp.Drafter.draft = orig
        es_config.MTP_CATCHUP, es_config.MTP_SPECULATE, vd._SpecGate.want = old
    assert st["spec"] > 0 and st["spec_hit"] > 0, f"never drafted in the pass: {st}"
    assert st["spec_hit"] < st["spec"], f"need passes cut short too: {st}"
    assert got == plain, f"in-pass drafting changed output\n  {got}\n  {plain}"
    check_greedy(model, prompt, got)
    assert off1 == off0, f"head cache holds {off1} positions, {off0} without in-pass drafting"
    gap = float(mx.abs(keys1.astype(mx.float32) - keys0.astype(mx.float32)).max())
    assert gap < 1e-4, f"head cache differs by {gap}"
    print(f"OK: in-pass drafts match drafting after the pass (used {st['spec_hit']} of {st['spec']}), head cache the same")


def check_spec_gate():
    """The gate speculates only where kept-in-full odds times draft time beat its cost."""
    g = vd._SpecGate()
    seen = []
    for i in range(400):
        conf = 0.99 if i % 2 else 0.3
        spec = g.want(conf)
        seen.append((conf, spec))
        full = conf > 0.9 or i % 10 == 0
        g.note_pass(spec, 0.030 if spec else 0.024, full)
        g.note_draft(0.010)
    late = seen[100:]
    hi = sum(s for c, s in late if c > 0.9) / sum(1 for c, _ in late if c > 0.9)
    lo = sum(s for c, s in late if c < 0.9) / sum(1 for c, _ in late if c < 0.9)
    assert hi > 0.95, f"confident drafts that are kept in full should speculate ({hi:.2f})"
    assert lo < 0.02, f"unconfident drafts should not ({lo:.2f})"
    g = vd._SpecGate()
    for i in range(400):
        spec = g.want(0.99)
        g.note_pass(spec, 0.040 if spec else 0.024, True)
        g.note_draft(0.010)
    assert sum(g.want(0.99) for _ in range(100)) <= 1, "speculated where it costs more than a draft"
    print("OK: in-pass drafting gate follows measured odds and costs")


def check_no_head_is_plain(tmp: Path):
    model, drafter = _attached(tmp, "plain")
    rng = np.random.default_rng(4)
    prompt = rng.integers(0, ARGS["vocab_size"], size=20).tolist()
    old = vd._Governor.MIN_PLAIN
    vd._Governor.MIN_PLAIN = 10**9
    try:
        a, _ = _run(model, prompt, 20)
        model._pagedmoe_mtp = None
        b, _ = _run(model, prompt, 20)
    finally:
        vd._Governor.MIN_PLAIN = old
    assert a == b, "capturing hidden states changed plain decode"
    assert vd.STATS["passes"] == 0
    print("OK: a head that never drafts leaves plain decode untouched")


def check_incompatible(tmp: Path):
    model = build()
    cfg = {**ARGS, "model_type": "qwen3_next"}
    os.environ["PAGED_MOE_MTP_DIR"] = str(tmp / "heads")
    mdir = _model_dir(tmp, {**cfg, "bad": 1}, "bad")
    d = mtp_fetch.head_dir(mdir)
    _write_raw(d, {"mtp.fc.weight": mx.zeros((3, 3))}, ARGS["num_hidden_layers"])
    assert mtp.prepare(model, cfg, mdir) is None
    st = mtp_fetch.local_status(mdir)
    assert st["state"] == "incompatible", st

    mdir2 = _model_dir(tmp, {**cfg, "broken": 1}, "broken")
    _write_raw(mtp_fetch.head_dir(mdir2), _hf_layout(_random_head(model, cfg), False), ARGS["num_hidden_layers"])
    assert mtp.prepare(model, cfg, mdir2) is not None
    hd = mtp_fetch.head_dir(mdir2)
    (hd / "head.safetensors").write_bytes(b"not a safetensors file")
    assert mtp.prepare(model, cfg, mdir2) is None
    assert not (hd / "head.json").exists(), "broken saved head kept"
    assert mtp_fetch.local_status(mdir2)["state"] != "ready"
    print("OK: unbuildable heads mark incompatible, broken saved heads are dropped, model loads without")


class _Resp(io.BytesIO):
    status = 206

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def check_fetch(tmp: Path):
    os.environ["PAGED_MOE_MTP_DIR"] = str(tmp / "heads")
    mdir = _model_dir(tmp, {"model_type": "qwen3_next", "num_hidden_layers": 2}, "fetch")
    upstream = tmp / "up" / "model-00001.safetensors"
    upstream.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    src = {
        "model.layers.0.mlp.gate.weight": mx.array(rng.standard_normal((4, 8)).astype(np.float32)),
        "mtp.fc.weight": mx.array(rng.standard_normal((8, 16)).astype(np.float32)),
        "mtp.norm.weight": mx.array(rng.standard_normal((8,)).astype(np.float32)),
        "mtp.layers.0.mlp.experts.0.up_proj.weight": mx.array(rng.standard_normal((300, 300)).astype(np.float32)),
    }
    mx.save_safetensors(str(upstream), src)
    blob = upstream.read_bytes()
    (n,) = struct.unpack("<Q", blob[:8])
    hdr = json.loads(blob[8:8 + n])
    tensors = []
    for name, t in hdr.items():
        if name == "__metadata__" or not mtp_fetch.is_mtp_name(name, "mtp", 2):
            continue
        s, e = t["data_offsets"]
        tensors.append({"name": name, "shard": upstream.name, "start": 8 + n + s, "size": e - s,
                        "dtype": t["dtype"], "shape": t["shape"]})
    tensors.sort(key=lambda t: t["size"])
    plan = {"state": "available", "repo": "a/b", "revision": "r", "scheme": "mtp", "layers": 2,
            "depth": 1, "tensors": tensors, "bytes": sum(t["size"] for t in tensors),
            "checked_at": 1e18}
    d = mtp_fetch.head_dir(mdir)
    mtp_fetch._write_json(d / "remote.json", plan)
    assert mtp_fetch.local_status(mdir)["state"] == "missing"

    calls = {"n": 0, "fail_at": 2, "dead": None}
    orig = mtp_fetch._request
    orig_piece = mtp_fetch.PIECE
    mtp_fetch.PIECE = 64 << 10  # several pieces per tensor here

    def fake(url, rng_=None, timeout=20.0):
        calls["n"] += 1
        a, b = rng_
        if calls["n"] == calls["fail_at"] or calls["dead"] == a:
            raise OSError("connection reset")
        return _Resp(blob[a:b])

    mtp_fetch._request = fake
    old_sleep = mtp_fetch.time.sleep
    mtp_fetch.time.sleep = lambda s: None
    try:
        st = mtp_fetch.fetch(mdir, quiet=True)
        assert st["state"] == "downloaded", st
        got = mx.load(str(d / "raw.safetensors"))
        assert set(got) == {t["name"] for t in tensors}
        for k in got:
            assert mx.array_equal(got[k], src[k]), k

        want_bytes = (d / "raw.safetensors").read_bytes()

        # One range never answers: the download fails, keeps what it got, and
        # a second run reads only that range.
        (d / "raw.safetensors").unlink()
        (d / "raw.json").unlink()
        big = tensors[-1]
        calls.update(n=0, fail_at=10**9, dead=big["start"] + mtp_fetch.PIECE)
        st = mtp_fetch.fetch(mdir, quiet=True)
        assert st["state"] == "failed", st
        assert 0 < st["done"] < plan["bytes"], st
        n_pieces = len(mtp_fetch._pieces(mtp_fetch._layout(plan["tensors"])[1]))
        kept = len(mtp_fetch._read_json(d / "fetch.json")["chunks"])
        assert 0 < kept < n_pieces, (kept, n_pieces)
        calls.update(n=0, dead=None)
        st = mtp_fetch.fetch(mdir, quiet=True)
        assert st["state"] == "downloaded", st
        assert (d / "raw.safetensors").read_bytes() == want_bytes, "resumed file differs"
        assert calls["n"] == n_pieces - kept, (
            f"resume read {calls['n']} ranges, expected the {n_pieces - kept} not yet on disk")

        # Too big for the unattended limit.
        (d / "raw.safetensors").unlink()
        (d / "raw.json").unlink()
        st = mtp_fetch.fetch(mdir, max_bytes=10, quiet=True)
        assert st["state"] == "missing" and "skipped" in st, st

        # Removed by the user: status keeps the size, auto fetch leaves it alone.
        st = mtp_fetch.remove(mdir)
        assert st["state"] == "missing" and st.get("removed"), st
        st = mtp_fetch.fetch(mdir, auto=True, quiet=True)
        assert st.get("skipped") == "removed by the user", st
        st = mtp_fetch.fetch(mdir, quiet=True)
        assert st["state"] == "downloaded", "an explicit download after remove must work"
    finally:
        mtp_fetch._request = orig
        mtp_fetch.time.sleep = old_sleep
        mtp_fetch.PIECE = orig_piece
    print("OK: fetch pulls only the planned tensors, resumes a partial file, honors the size limit and removal")


def check_bundled_mtp_stays_out_of_the_backbone():
    """language_model.mtp.* is a head, not a backbone weight.

    The 4-bit native-MTP quant stores the head under that prefix. The backbone
    sanitize has to drop it or load_weights rejects the file. The head loader
    still sees the original names.
    """
    from pagedmoe.arches.qwen4_exp import Model

    class _W:
        ndim = 1
        shape = (1,)

    weights = {
        "language_model.mtp.fc_embedding.weight": _W(),
        "language_model.mtp.layers.0.mlp.gate.weight": _W(),
        "mtp.pre_fc_norm_hidden.weight": _W(),
        "language_model.model.embed_tokens.weight": _W(),
        "language_model.lm_head.weight": _W(),
    }
    out = Model.sanitize(Model.__new__(Model), weights)
    assert "model.embed_tokens.weight" in out, out
    assert "lm_head.weight" in out, out
    assert not any("mtp" in k for k in out), sorted(out)
    assert mtp_fetch.is_mtp_name("language_model.mtp.fc_embedding.weight", "mtp", 48)
    assert mtp_fetch.is_mtp_name("mtp.layers.0.mlp.gate.weight", "mtp", 48)
    print("OK: language_model.mtp weights stay out of the backbone and remain a head")


def check_status_and_names(tmp: Path):
    os.environ["PAGED_MOE_MTP_DIR"] = str(tmp / "heads")
    m = _model_dir(tmp, {"model_type": "llama"}, "llama")
    assert mtp_fetch.local_status(m)["state"] == "unsupported"
    m = _model_dir(tmp, {"model_type": "glm4_moe", "num_hidden_layers": 2}, "glm")
    (m / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.layers.1.mlp.gate.weight": "a", "model.layers.2.eh_proj.weight": "b"}}))
    assert mtp_fetch.local_status(m)["state"] == "bundled"
    assert mtp_fetch.is_mtp_name("model.layers.92.enorm.weight", "nextn", 92)
    assert not mtp_fetch.is_mtp_name("model.layers.91.mlp.gate.weight", "nextn", 92)
    assert mtp_fetch.is_mtp_name("mtp.layers.0.mlp.gate.weight", "mtp", 48)
    # A working head stays ready. An old packed-load failure is tried again
    # only while the checkpoint still has the tensors. Anything this builder
    # writes, and any other failure, stays incompatible.
    ready = _model_dir(tmp, {"model_type": "qwen3_next", "num_hidden_layers": 2, "n": "ready"}, "ready")
    (ready / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "mtp.fc.weight": "model.safetensors"}}))
    rd = mtp_fetch.head_dir(ready)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "head.safetensors").write_bytes(b"ok")
    (rd / "head.json").write_text(json.dumps({
        "state": "ready", "version": mtp_fetch.HEAD_VERSION, "bytes": 1, "source": "converted",
    }))
    assert mtp_fetch.local_status(ready)["state"] == "ready"
    stuck = _model_dir(tmp, {"model_type": "qwen3_next", "num_hidden_layers": 2, "n": "stuck"}, "stuck")
    (stuck / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "mtp.fc.weight": "model.safetensors"}}))
    sd = mtp_fetch.head_dir(stuck)
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "head.json").write_text(json.dumps({
        "state": "incompatible", "version": mtp_fetch.HEAD_VERSION, "builder": 1,
        "error": "ValueError('Received 4 parameters not in model.')",
    }))
    assert mtp_fetch.local_status(stuck)["state"] == "incompatible"
    other = _model_dir(tmp, {"model_type": "qwen3_next", "num_hidden_layers": 2, "n": "other"}, "other")
    (other / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "mtp.fc.weight": "model.safetensors"}}))
    od = mtp_fetch.head_dir(other)
    od.mkdir(parents=True, exist_ok=True)
    (od / "head.json").write_text(json.dumps({
        "state": "incompatible", "version": mtp_fetch.HEAD_VERSION,
        "error": "Incompatible('nextn layer tensors are incomplete')",
    }))
    assert mtp_fetch.local_status(other)["state"] == "incompatible"
    retry = _model_dir(tmp, {"model_type": "qwen3_next", "num_hidden_layers": 2, "n": "retry"}, "retry")
    (retry / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "mtp.layers.0.self_attn.q_proj.weight": "model.safetensors",
        "mtp.layers.0.self_attn.q_proj.scales": "model.safetensors",
    }}))
    assert mtp_fetch.local_status(retry)["state"] == "bundled"
    mtp_fetch._write_json(mtp_fetch.head_dir(retry) / "head.json", {
        "state": "incompatible", "version": mtp_fetch.HEAD_VERSION,
        "error": "ValueError('Received 44 parameters not in model.')",
    })
    assert mtp_fetch.local_status(retry)["state"] == "bundled", "old packed failure should rebuild"
    nowhere = _model_dir(tmp, {"model_type": "qwen3_next", "num_hidden_layers": 2, "n": "nowhere"}, "nowhere")
    mtp_fetch._write_json(mtp_fetch.head_dir(nowhere) / "head.json", {
        "state": "incompatible", "version": mtp_fetch.HEAD_VERSION,
        "error": "ValueError('Received 4 parameters not in model.')",
    })
    assert mtp_fetch.local_status(nowhere)["state"] == "incompatible"
    plain = _model_dir(tmp, {"model_type": "llama"}, "plain")
    mtp_fetch._write_json(mtp_fetch.head_dir(plain) / "head.json", {
        "state": "incompatible", "version": mtp_fetch.HEAD_VERSION,
        "error": "ValueError('Received 4 parameters not in model.')",
    })
    assert mtp_fetch.local_status(plain)["state"] == "unsupported"
    readme = "---\nlicense: mit\nbase_model:\n- Qwen/Qwen3.5-35B-A3B\n- other/thing\ntags: []\n---\nconverted to MLX format from [Org/Model](https://x)\n"
    assert mtp_fetch._base_models(readme) == ["Qwen/Qwen3.5-35B-A3B", "other/thing", "Org/Model"]
    assert mtp_fetch._base_models("---\nbase_model: Qwen/Qwen3.8-Flash-Next\n---\n") == ["Qwen/Qwen3.8-Flash-Next"]
    print("OK: status states and MTP tensor names per layout, base_model parsing")


def _flash_module():
    port = next((p for p in FLASH_PORTS if p.exists()), None)
    if port is None:
        return None
    spec = importlib.util.spec_from_file_location("qwen4_exp", str(port))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_qwen4_hc(tmp: Path):
    mod = _flash_module()
    if mod is None:
        print("skip: qwen4_exp port not found")
        return
    text = dict(
        hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, vocab_size=128, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=32, shared_expert_intermediate_size=32,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, hc_count=2, hc_lowrank=16, indexer_n_heads=2,
        indexer_head_dim=16, indexer_budget=64, indexer_compress_ratio=4,
        ngram_vocab_size_base=1000, make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4, heads_per_ngram=2, ple_embed_dim=64, ple_layer_ids=[2],
        mtp_num_hidden_layers=1,
    )
    mx.random.seed(0)
    model = mod.Model(mod.ModelArgs(model_type="qwen4_exp", text_config=text))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    cfg = {"model_type": "qwen4_exp", "text_config": text}
    os.environ["PAGED_MOE_MTP_DIR"] = str(tmp / "heads")
    mdir = _model_dir(tmp, cfg, "q4")
    want = _random_head(model, cfg)
    _write_raw(mtp_fetch.head_dir(mdir), _hf_layout(want, True), 4)
    prep = mtp.prepare(model, cfg, mdir)
    assert prep is not None, "qwen4 hc head did not build"
    assert _same_params(want, dict(tree_flatten(prep.head.parameters()))) < 1e-6
    drafter = mtp.attach(model, prep)
    rng = np.random.default_rng(1)
    prompt = rng.integers(0, 128, size=24).tolist()
    with _Force():
        got, _ = _run(model, prompt, 24)
    assert vd.STATS["mtp"] > 0, dict(vd.STATS)
    assert int(drafter.cache.offset) > 0
    indexer = drafter.cache.indexer.keys
    assert indexer is not None and int(indexer.shape[1]) == int(drafter.cache.offset), (
        "indexer keys not trimmed with the head KV"
    )
    check_greedy(model, prompt, got)
    print(f"OK: qwen4_exp hyper-connection head builds, drafts ({vd.STATS['mtp']} passes), output greedy")


def check_dense_head(tmp: Path):
    """A dense checkpoint (Qwen3.5 27B) has no MoE layer; the head clones a dense one."""
    from mlx_lm.models import qwen3_5

    text = {
        **{k: ARGS[k] for k in (
            "hidden_size", "num_hidden_layers", "intermediate_size", "num_attention_heads",
            "linear_num_value_heads", "linear_num_key_heads", "linear_key_head_dim",
            "linear_value_head_dim", "linear_conv_kernel_dim", "rms_norm_eps", "vocab_size",
            "num_key_value_heads", "max_position_embeddings", "head_dim", "full_attention_interval",
        )},
        "model_type": "qwen3_5_text",
        "num_experts": 0,
        "mtp_num_hidden_layers": 1,
    }
    cfg = {"model_type": "qwen3_5", "text_config": text}
    mx.random.seed(7)
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(cfg))
    model.set_dtype(mx.float32)
    model.update(nn.utils.tree_map(lambda p: p * 3.0 if p.ndim == 2 else p, model.parameters()))
    mx.eval(model.parameters())

    parts = mtp._Parts(model, cfg)
    assert parts.dense and parts.proxy == 3, (parts.dense, parts.proxy)
    os.environ["PAGED_MOE_MTP_DIR"] = str(tmp / "heads")
    mdir = _model_dir(tmp, cfg, "dense")
    want = _random_head(model, cfg, 9)
    assert any(k.startswith("layers.0.mlp.gate_proj") for k in want), sorted(want)
    _write_raw(mtp_fetch.head_dir(mdir), _hf_layout(want, False), text["num_hidden_layers"])
    prep = mtp.prepare(model, cfg, mdir)
    assert prep is not None, "dense head did not build"
    worst = _same_params(want, dict(tree_flatten(prep.head.parameters())))
    assert worst < 1e-6, f"converted dense head differs by {worst}"
    drafter = mtp.attach(model, prep)
    model._pagedmoe_info = {"mode": "resident"}
    assert vd.wanted(model), "resident model with a head should take verify decode"

    rng = np.random.default_rng(5)
    prompt = rng.integers(0, text["vocab_size"], size=24).tolist()
    with _Force():
        got, _ = _run(model, prompt, 32)
    st = dict(vd.STATS)
    assert st["mtp"] > 0, f"head never drafted: {st}"
    check_greedy(model, prompt, got)
    assert int(drafter.cache.offset) <= len(prompt) + len(got)
    from pagedmoe import fast_replay

    rep = fast_replay.for_model(model)
    assert not rep.off, f"fast replay turned off: {rep.why}"
    assert "dense" in rep.role.values(), "dense MLPs should replay from their recorded rows"
    assert rep.fast > 0, "no fast replay ran past the checks"
    print(f"OK: dense head builds on a dense layer, drafts ({st['mtp']} passes), output greedy")

    # Without a head, a resident model still drafts copies from its context.
    model._pagedmoe_mtp = None
    assert vd.wanted(model), "resident model should take verify decode for n-gram drafts"
    prompt = rng.integers(0, text["vocab_size"], size=24).tolist()
    junk = rng.integers(0, text["vocab_size"], size=8).tolist()
    saved = (vd._Governor.ready, vd._Governor.want, vd._Index.query)
    vd._Governor.ready = lambda self, span: True
    vd._Governor.want = lambda self, span, key: key[0] == "ngram"
    # A tiny random model never copies, so stand in for a match that misses.
    vd._Index.query = lambda index, k: junk[:k]
    try:
        got, _ = _run(model, prompt, 24)
    finally:
        vd._Governor.ready, vd._Governor.want, vd._Index.query = saved
    st = dict(vd.STATS)
    assert st["passes"] > 0 and st["mtp"] == 0, f"no n-gram pass on a resident model: {st}"
    check_greedy(model, prompt, got)
    old = es_config.VERIFY_RESIDENT_NGRAM
    es_config.VERIFY_RESIDENT_NGRAM = False
    try:
        assert not vd.wanted(model), "resident model with no head and n-grams off is plain decode"
    finally:
        es_config.VERIFY_RESIDENT_NGRAM = old
    print(f"OK: resident n-gram drafts ({st['passes']} passes, {st['accepted']} accepted), output greedy")


def main():
    old = (es_config.VERIFY_SPANS, os.environ.get("PAGED_MOE_MTP_DIR"))
    es_config.VERIFY_SPANS = "all"
    try:
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            check_bundled_mtp_stays_out_of_the_backbone()
            check_status_and_names(tmp)
            check_fetch(tmp)
            check_convert_roundtrip(tmp)
            check_quant_mirror(tmp)
            check_float_head_quantizes_after_load(tmp)
            check_packed_head(tmp)
            check_incompatible(tmp)
            check_no_head_is_plain(tmp)
            check_pairs_and_output(tmp)
            check_speculate(tmp)
            check_spec_gate()
            check_qwen4_hc(tmp)
            check_dense_head(tmp)
    finally:
        es_config.VERIFY_SPANS = old[0]
        if old[1] is None:
            os.environ.pop("PAGED_MOE_MTP_DIR", None)
        else:
            os.environ["PAGED_MOE_MTP_DIR"] = old[1]
    print("\nOK: mtp tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
