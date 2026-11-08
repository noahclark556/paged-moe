# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-token-prediction head as a verify drafter.

Models trained with MTP (Qwen3-Next, Qwen3.5, Qwen4-exp, DeepSeek-V3, GLM-4.x)
carry one extra decoder layer that predicts the token after next from the
main model's last hidden state. MLX conversions drop it. When the head is
around (bundled in the checkpoint, or fetched by mtp_fetch), this builds it
next to the backbone and hands verify_decode k-token drafts from it.

The head never touches the output: every draft goes through a verify pass
like the n-gram and self drafts, and the governor drops it where it loses.
Its experts are small and resident, so a draft is one short layer stack plus
the LM head per token, with no expert reads.

Three layouts:

* "mtp": `fc(cat[norm_e(embed), norm_h(hidden)])`, one layer, final norm
  (Qwen3-Next, Qwen3.5). `hidden` is the main model's final-norm output.
* "mtp" with hyper-connections (Qwen4-exp): separate fc_embedding / fc_hidden,
  the embedding added to every residual stream, the head's own stream mixer.
  `hidden` is the stream stack the main model's mixer reads.
* "nextn": `eh_proj(cat[enorm(embed), hnorm(hidden)])`, one decoder layer,
  shared_head.norm (DeepSeek-V3, GLM-4.x).

The converted head is saved once per checkpoint (head.safetensors, quantized
like the matching backbone modules) and loaded from there afterwards.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import os
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from . import config, mtp_fetch

# Families whose checkpoints store RMSNorm weights zero-centered (x * (1 + w)).
CENTERED = {"qwen4_exp", "qwen3_next", "qwen3_5_moe", "qwen3_5"}

_PROJ = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
_EXPERT = re.compile(
    r"^(?P<p>.*\.mlp)\.experts\.(?P<e>\d+)\.(?P<proj>gate_proj|up_proj|down_proj|w1|w2|w3)"
    r"\.(?P<k>weight|scales|biases|weight_scale_inv)$"
)
_LAYER = re.compile(r"^(?:.*\.)?layers\.\d+\.")


class Incompatible(Exception):
    """The head cannot be built for this checkpoint (layout or shapes)."""


# ------------------------------------------------------------------ the model


class _Tap:
    __slots__ = ("on", "value")

    def __init__(self):
        self.on = False
        self.value = None


def _install_tap(module, tap: _Tap, use_input: bool) -> None:
    """Record `module`'s input or output while tap.on, by swapping its class."""
    base = type(module)

    def __call__(self, x, *args, **kwargs):
        out = base.__call__(self, x, *args, **kwargs)
        if tap.on:
            tap.value = x if use_input else out
        return out

    module.__class__ = type(base.__name__, (base,), {"__call__": __call__})


class _Shared(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm


class _Head(nn.Module):
    pass


class _Parts:
    """Where the backbone keeps what a head needs."""

    def __init__(self, model, cfg: dict):
        self.model_type = mtp_fetch.model_type(cfg)
        self.scheme = mtp_fetch.SCHEMES.get(self.model_type)
        if not self.scheme:
            raise Incompatible(f"no MTP layout for {self.model_type or 'this model'}")
        inner = None
        for path in (("language_model", "model"), ("model",), ("language_model",)):
            obj = model
            for a in path:
                obj = getattr(obj, a, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "layers") and hasattr(obj, "embed_tokens"):
                inner = obj
                break
        if inner is None:
            raise Incompatible("backbone has no text model with layers and embed_tokens")
        self.inner = inner
        owner = getattr(model, "language_model", None) or model
        lm_head = getattr(owner, "lm_head", None)
        self.lm_head = lm_head if lm_head is not None else inner.embed_tokens.as_linear
        self.embed = inner.embed_tokens
        args = None
        top = getattr(model, "args", None)
        for cand in (
            getattr(inner, "args", None), getattr(owner, "args", None),
            getattr(top, "text", None), getattr(top, "text_config", None), top,
        ):
            if cand is not None and not isinstance(cand, dict) and hasattr(cand, "hidden_size"):
                args = cand
                break
        if args is None:
            raise Incompatible("backbone args not found")
        self.args = args
        self.hidden = int(args.hidden_size)
        self.eps = float(getattr(args, "rms_norm_eps", 1e-6))
        self.hc = int(getattr(args, "hc_count", 0) or 0)

        # A full-attention MoE layer to clone. Picked before patch_model swaps
        # the expert blocks, so its class and quantization are the stock ones.
        # A dense checkpoint (no MoE layer at all) clones a full-attention
        # dense layer instead; its head has a dense MLP too.
        self.proxy = None
        dense = None
        for i in range(len(inner.layers) - 1, -1, -1):
            layer = inner.layers[i]
            if getattr(layer, "self_attn", None) is None:
                continue
            if getattr(layer, "ple", None) is not None:
                continue
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            if getattr(mlp, "switch_mlp", None) is None:
                if dense is None:
                    dense = i
                continue
            self.proxy = i
            break
        self.dense = False
        if self.proxy is None and dense is not None and not any(
            getattr(getattr(l, "mlp", None), "switch_mlp", None) is not None for l in inner.layers
        ):
            self.proxy = dense
            self.dense = True
        if self.proxy is None:
            raise Incompatible("no full-attention layer to model the head on")
        self.layer_cls = type(inner.layers[self.proxy])
        call = getattr(self.layer_cls, "__wrapped__", None) or self.layer_cls.__call__
        params = list(inspect.signature(getattr(call, "__wrapped__", call)).parameters)[1:]
        if len(params) < (8 if self.hc else 3):
            raise Incompatible(f"{self.layer_cls.__name__} is called with {params}")
        from mlx_lm.models.cache import KVCache

        self.cache_cls = KVCache
        try:
            cls = type(model.make_cache()[self.proxy])
            cls()
            self.cache_cls = cls
        except Exception:
            pass
        self.quant = _quant_lookup(model)
        self.default_quant = _default_quant(cfg)
        # Classes come off live instances: a checkpoint's own model file is
        # not always importable by name.
        if self.hc:
            q_norm = getattr(inner.layers[self.proxy].self_attn, "q_norm", None)
            mixer = getattr(inner, "hyper_connection_mixer", None)
            if q_norm is None or mixer is None:
                raise Incompatible("hyper-connection backbone without q_norm or a stream mixer")
            self.norm_cls = type(q_norm)
            self.mixer_cls = type(mixer)
        else:
            final = getattr(inner, "norm", None)
            self.norm_cls = type(final) if final is not None else nn.RMSNorm
            self.mixer_cls = None

    def norm(self, dims: int, group: int | None = None):
        if self.hc:
            return self.norm_cls(dims, group_size=group, eps=self.eps)
        try:
            return self.norm_cls(dims, eps=self.eps)
        except TypeError:
            return nn.RMSNorm(dims, eps=self.eps)


def _quant_lookup(model) -> list[tuple[str, dict | None]]:
    out = []
    for path, m in model.named_modules():
        if not hasattr(m, "weight") or not hasattr(m, "to_quantized") and not hasattr(m, "bits"):
            continue
        q = None
        if getattr(m, "bits", None) is not None and getattr(m, "group_size", None) is not None:
            q = {"bits": int(m.bits), "group_size": int(m.group_size),
                 "mode": str(getattr(m, "mode", "affine"))}
        out.append((_LAYER.sub("", path) if _LAYER.match(path) else path, q))
    return out


def _default_quant(cfg: dict) -> dict | None:
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    if not isinstance(q, dict) or not q.get("bits"):
        return None
    return {"bits": int(q["bits"]), "group_size": int(q.get("group_size") or 64),
            "mode": str(q.get("mode") or "affine")}


_EXPERT_ARGS = ("num_experts", "n_routed_experts", "num_local_experts")


def _head_experts(weights: dict[str, mx.array]) -> int | None:
    for k, v in weights.items():
        if k.endswith(("switch_mlp.gate_proj.weight", "switch_mlp.up_proj.weight")):
            return int(v.shape[0])
    return None


def _build(parts: _Parts, experts: int | None = None) -> _Head:
    H = parts.hidden
    head = _Head()
    args = parts.args
    # A pruned backbone has fewer routed experts than the head, which was
    # never pruned. The head layer is built at its own count.
    names = [a for a in _EXPERT_ARGS if isinstance(getattr(args, a, None), int)]
    if experts and names and getattr(args, names[0]) != experts:
        args = copy.copy(args)
        for a in names:
            setattr(args, a, experts)
    layer = parts.layer_cls(args, parts.proxy)
    head.layers = [layer]
    if parts.scheme == "mtp" and parts.hc:
        mixer_cls = parts.mixer_cls
        head.fc_embedding = nn.Linear(H, H, bias=False)
        head.fc_hidden = nn.Linear(H, H, bias=False)
        head.pre_fc_norm_embedding = parts.norm(H)
        head.pre_fc_norm_hidden = parts.norm(parts.hc * H, group=H)
        head.hyper_connection_mixer = mixer_cls(parts.args, use_combine=False)
    elif parts.scheme == "mtp":
        head.fc = nn.Linear(2 * H, H, bias=False)
        head.pre_fc_norm_embedding = parts.norm(H)
        head.pre_fc_norm_hidden = parts.norm(H)
        head.norm = parts.norm(H)
    else:
        head.eh_proj = nn.Linear(2 * H, H, bias=False)
        head.enorm = parts.norm(H)
        head.hnorm = parts.norm(H)
        head.shared_head = _Shared(parts.norm(H))
    return head


def _quant_map(parts: _Parts, head: _Head) -> dict[str, dict | None]:
    out: dict[str, dict | None] = {}
    for path, m in head.named_modules():
        if not hasattr(m, "to_quantized") or not hasattr(m, "weight"):
            continue
        key = _LAYER.sub("", path) if _LAYER.match(path) else path
        q = parts.default_quant
        for main, mq in parts.quant:
            if main == key or main.endswith("." + key):
                q = mq
                break
        if q is not None and int(m.weight.shape[-1]) % int(q["group_size"]) != 0:
            q = None
        out[path] = q
    return out


_PACKED_BITS = (2, 3, 4, 5, 6, 8)


def _quant_spec(full, packed, scales, biases) -> dict | None:
    """bits / group of one linear already stored as packed weight + scales + biases.

    `full` is the unquantized module weight. A shape that is not that packing
    returns None, and the strict load then fails closed.
    """
    if not full or not packed or not scales or not biases or len(full) < 2:
        return None
    if tuple(packed[:-1]) != tuple(full[:-1]) or tuple(scales[:-1]) != tuple(full[:-1]):
        return None
    if tuple(biases) != tuple(scales):
        return None
    full_in = int(full[-1])
    packed_in = int(packed[-1])
    groups = int(scales[-1])
    if full_in <= 0 or packed_in <= 0 or groups <= 0 or full_in % groups != 0:
        return None
    if (packed_in * 32) % full_in != 0:
        return None
    bits = packed_in * 32 // full_in
    group = full_in // groups
    if bits not in _PACKED_BITS or full_in % group != 0:
        return None
    return {"bits": int(bits), "group_size": int(group), "mode": "affine"}


def _packed_quant_map(head: _Head, weights: dict[str, mx.array]) -> dict[str, dict | None] | None:
    """Quant recipe already packed into `weights`, or None when they are plain floats.

    Only modules that carry `.scales` and `.biases` are quantized, and only
    with the group and bits those tensors imply. A float head (the usual
    download) has no scales, so this stays out of that path.
    """
    if not any(k.endswith(".scales") for k in weights):
        return None
    out: dict[str, dict | None] = {}
    for path, m in head.named_modules():
        if not hasattr(m, "to_quantized") or not hasattr(m, "weight"):
            continue
        w = weights.get(f"{path}.weight")
        s = weights.get(f"{path}.scales")
        b = weights.get(f"{path}.biases")
        if w is None or s is None or b is None:
            out[path] = None
            continue
        out[path] = _quant_spec(tuple(m.weight.shape), tuple(w.shape), tuple(s.shape), tuple(b.shape))
    return out


def _apply_quant(head: _Head, qmap: dict[str, dict | None]) -> None:
    if not any(qmap.values()):
        return

    def pred(path, module):
        q = qmap.get(path)
        return dict(q) if q else False

    nn.quantize(head, class_predicate=pred)


# --------------------------------------------------------------- conversion


def _dequant_fp8(w: mx.array, scale_inv: mx.array) -> mx.array:
    w = mx.from_fp8(w, dtype=mx.bfloat16)
    bs = 128
    m, n = w.shape
    pb, ps = (-m) % bs, (-n) % bs
    w = mx.pad(w, ((0, pb), (0, ps)))
    w = w.reshape((m + pb) // bs, bs, (n + ps) // bs, bs)
    w = (w * scale_inv[:, None, :, None]).reshape(m + pb, n + ps)
    return w[:m, :n].astype(mx.bfloat16)


def _common(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """FP8 dequant and expert stacking for head-relative names."""
    out = {}
    for k, v in weights.items():
        if k.endswith("weight_scale_inv"):
            continue
        sk = k[: -len("weight")] + "weight_scale_inv" if k.endswith("weight") else None
        if sk is not None and sk in weights:
            v = _dequant_fp8(v, weights[sk])
        out[k] = v
    groups: dict[tuple, dict[int, mx.array]] = {}
    rest = {}
    for k, v in out.items():
        m = _EXPERT.match(k)
        if m:
            proj = _PROJ.get(m["proj"], m["proj"])
            groups.setdefault((m["p"], proj, m["k"]), {})[int(m["e"])] = v
            continue
        if k.endswith(".mlp.experts.gate_up_proj"):
            base = k[: -len("experts.gate_up_proj")]
            mid = v.shape[-2] // 2
            rest[base + "switch_mlp.gate_proj.weight"] = v[..., :mid, :]
            rest[base + "switch_mlp.up_proj.weight"] = v[..., mid:, :]
            continue
        if k.endswith(".mlp.experts.down_proj"):
            rest[k[: -len("experts.down_proj")] + "switch_mlp.down_proj.weight"] = v
            continue
        rest[k] = v
    for (p, proj, kind), by_e in groups.items():
        if sorted(by_e) != list(range(len(by_e))):
            raise Incompatible(f"{p}: expert ids are not 0..{len(by_e) - 1}")
        rest[f"{p}.switch_mlp.{proj}.{kind}"] = mx.stack([by_e[e] for e in range(len(by_e))])
    return rest


def _center(weights: dict[str, mx.array], raw_hf: bool) -> dict[str, mx.array]:
    for k, v in list(weights.items()):
        mod = k.rsplit(".", 2)[-2] if k.count(".") else ""
        if v.ndim != 1 or not k.endswith(".weight") or "norm" not in mod or "linear_attn" in k:
            continue
        # Upstream HF tensors are always unshifted, and trained ones can
        # average well above 1, so only a local checkpoint gets the guess.
        if raw_hf or float(mx.mean(v.astype(mx.float32)).item()) < 0.6:
            weights[k] = v + 1.0
    return weights


def _convert(
    model, parts: _Parts, raw: dict[str, mx.array], n_layers: int, raw_hf: bool = True,
) -> dict[str, mx.array]:
    if parts.scheme == "mtp":
        weights = {}
        for k, v in raw.items():
            if k.startswith("mtp."):
                weights[k[4:]] = v
            elif ".mtp." in k:
                weights[k.split(".mtp.", 1)[1]] = v
        weights = _common(weights)
        # Scales mean an MLX quant already stored the norms the way the
        # backbone stores them. The float path still recenters.
        if parts.model_type in CENTERED and not any(k.endswith(".scales") for k in weights):
            weights = _center(weights, raw_hf)
        return weights

    pre = re.compile(rf"^(?:model\.)?(?:language_model\.)?(?:model\.)?layers\.{n_layers}\.")
    own, extra = {}, {}
    P = parts.proxy
    for k, v in raw.items():
        m = pre.match(k)
        if not m:
            continue
        rel = k[m.end():]
        if rel.startswith(("enorm.", "hnorm.", "eh_proj.", "shared_head.norm.")):
            extra[rel] = v
        elif rel.startswith(("embed_tokens.", "shared_head.head.")):
            continue
        else:
            own[f"model.layers.{P}.{rel}"] = v
    if not own or not extra:
        raise Incompatible("nextn layer tensors are incomplete")
    sanitized = model.sanitize(dict(own))
    lead = f"model.layers.{P}."
    weights = {"layers.0." + k[len(lead):]: v for k, v in sanitized.items() if k.startswith(lead)}
    for k, v in _common(extra).items():
        weights[k] = v
    return weights


def _load_source(model_path: Path, d: Path, state: str, scheme: str, cfg: dict) -> tuple[dict, int]:
    if state == "downloaded":
        meta = mtp_fetch._read_json(d / "raw.json") or {}
        return mx.load(str(d / "raw.safetensors")), int(meta.get("layers") or mtp_fetch.num_layers(cfg))
    L = mtp_fetch.num_layers(cfg)
    depth = mtp_fetch.declared_layers(cfg) or 1
    idx = mtp_fetch._read_json(model_path / "model.safetensors.index.json") or {}
    wmap = idx.get("weight_map") or {}
    by_file: dict[str, list[str]] = {}
    if wmap:
        for name, shard in wmap.items():
            if mtp_fetch.is_mtp_name(name, scheme, L, depth):
                by_file.setdefault(shard, []).append(name)
    else:
        names = [n for n in mtp_fetch.local_names(model_path) if mtp_fetch.is_mtp_name(n, scheme, L, depth)]
        by_file["model.safetensors"] = names
    out = {}
    for shard, names in by_file.items():
        tensors = mx.load(str(model_path / shard))
        for n in names:
            out[n] = tensors[n]
    return out, L


def _params(head) -> dict[str, mx.array]:
    return dict(tree_flatten(head.parameters()))


class Prepared:
    def __init__(self, head: _Head, parts: _Parts, nbytes: int, source: str):
        self.head = head
        self.parts = parts
        self.nbytes = nbytes
        self.source = source


def prepare(model, cfg: dict, model_path) -> Prepared | None:
    """Build the head if this checkpoint has one on disk. None otherwise.

    Runs before the expert cache is sized, so the caller can take the head's
    bytes out of it. Never raises: a head that will not build leaves the model
    decoding exactly as it would without one.
    """
    if not config.MTP:
        return None
    model_path = Path(model_path)
    try:
        st = mtp_fetch.local_status(model_path)
    except Exception as e:
        print(f"[mtp] status check failed, no head: {e!r}", flush=True)
        return None
    state = st.get("state")
    if state not in ("ready", "bundled", "downloaded"):
        return None
    d = Path(st["dir"])
    t0 = time.perf_counter()
    try:
        parts = _Parts(model, cfg)
        if state == "ready":
            meta = mtp_fetch._read_json(d / "head.json") or {}
            head = _build(parts, meta.get("experts"))
            qmap = meta.get("quant") or {}
            _apply_quant(head, qmap)
            head.load_weights(str(d / "head.safetensors"), strict=True)
            mx.eval(head.parameters())
            source = meta.get("source") or "converted"
        else:
            raw, n_layers = _load_source(model_path, d, state, parts.scheme, cfg)
            weights = _convert(model, parts, raw, n_layers, raw_hf=state == "downloaded")
            del raw
            experts = _head_experts(weights)
            head = _build(parts, experts)
            # A checkpoint that already quantized the head (scales and biases
            # next to the packed weight) has to be quantized before load.
            # Plain float weights, including a downloaded head, load first and
            # then take the backbone's quant, same as before.
            packed = _packed_quant_map(head, weights)
            if packed is not None:
                qmap = packed
                _apply_quant(head, qmap)
                head.load_weights(list(weights.items()), strict=True)
            else:
                head.load_weights(list(weights.items()), strict=True)
                qmap = _quant_map(parts, head)
                _apply_quant(head, qmap)
            del weights
            mx.eval(head.parameters())
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / "head.tmp.safetensors"
            mx.save_safetensors(str(tmp), _params(head))
            os.replace(tmp, d / "head.safetensors")
            nbytes = sum(v.nbytes for v in _params(head).values())
            source = "bundled" if state == "bundled" else (st.get("repo") or "downloaded")
            mtp_fetch._write_json(d / "head.json", {
                "state": "ready", "version": mtp_fetch.HEAD_VERSION, "bytes": nbytes,
                "quant": qmap, "source": source, "model_type": parts.model_type,
                "experts": experts, "converted": time.time(),
            })
            if state == "downloaded":
                (d / "raw.safetensors").unlink(missing_ok=True)
            mx.clear_cache()
        nbytes = sum(v.nbytes for v in _params(head).values())
        print(
            f"[mtp] head ready ({nbytes / 1e9:.2f} GB, {parts.scheme}"
            f"{' hc' if parts.hc else ''}, from {source}) in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )
        return Prepared(head, parts, nbytes, source)
    except Exception as e:
        mx.clear_cache()
        if state == "ready":
            # A saved head that no longer loads is rebuilt from upstream.
            print(f"[mtp] saved head did not load, dropping it: {e!r}", flush=True)
            for name in ("head.safetensors", "head.json"):
                (d / name).unlink(missing_ok=True)
        else:
            print(f"[mtp] head did not build, decoding without it: {e!r}", flush=True)
            mtp_fetch.mark_incompatible(model_path, repr(e))
        return None


# ------------------------------------------------------------------ drafting


class Drafter:
    """The head, its own KV cache, and the (hidden, next token) pairs it owes.

    Every forward of the main model that verify_decode keeps hands the head
    one pair per kept position: the hidden state at t and the token at t+1.
    Pairs queue up while the head is not drafting and go in as one batch on
    the next draft. A draft adds the pending pair for the token just emitted,
    chains k head steps, then trims the k-1 guessed positions off the head's
    cache again, so the cache only ever holds real positions.
    """

    def __init__(self, prep: Prepared, tap: _Tap):
        self.head = prep.head
        self.parts = prep.parts
        self.tap = tap
        self.nbytes = prep.nbytes
        self.fails = 0
        self.dead = False
        self.drafts = 0
        self.reset()

    def reset(self) -> None:
        self.cache = self.parts.cache_cls()
        self.pend_h: list[mx.array] = []
        self.pend_t: list[mx.array] = []
        self.last_h = None
        self.pre = None
        self.spec = None
        self.ready = None
        self.conf = None
        self.dead = self.fails >= 3

    @contextlib.contextmanager
    def capture(self):
        self.tap.on = True
        self.tap.value = None
        try:
            yield
        finally:
            self.tap.on = False

    def fed(self, tokens: mx.array) -> None:
        """The main model just kept `tokens`; tap.value holds their hidden states."""
        h = self.tap.value
        self.tap.value = None
        self.ready = None
        if self.dead:
            return
        if h is None:
            self.last_h = None
            self.pre = None
            return
        toks = tokens.reshape(-1)
        S = int(toks.shape[0])
        h = h[:, :S]
        if self.pre is None and self.last_h is not None:
            self.pend_h.append(self.last_h)
            self.pend_t.append(toks[0:1])
        self.pre = None
        for i in range(1, S):
            self.pend_h.append(h[:, i - 1:i])
            self.pend_t.append(toks[i:i + 1])
        self.last_h = h[:, S - 1:S]
        cap = max(1, int(config.MTP_CATCHUP))
        if len(self.pend_h) > cap:
            del self.pend_h[:-cap], self.pend_t[:-cap]

    def _fwd(self, hid: mx.array, ids: mx.array):
        p = self.parts
        head = self.head
        emb = p.embed(ids)
        S = int(ids.shape[1])
        mask = "causal" if S > 1 else None
        layer = head.layers[0]
        if p.scheme == "mtp" and p.hc:
            e = head.fc_embedding(head.pre_fc_norm_embedding(emb))
            hn = head.pre_fc_norm_hidden(hid).reshape(*hid.shape[:-1], p.hc, p.hidden)
            x = (head.fc_hidden(hn) + e[..., None, :]).reshape(*hid.shape[:-1], -1)
            idx = getattr(self.cache, "indexer", None)
            x = layer(x, p.inner.rope, mask, None, self.cache, idx, None, None)
            return p.lm_head(head.hyper_connection_mixer(x)), x
        if p.scheme == "mtp":
            x = head.fc(mx.concatenate(
                [head.pre_fc_norm_embedding(emb), head.pre_fc_norm_hidden(hid)], axis=-1))
            x = layer(x, mask, self.cache)
            out = head.norm(x)
            return p.lm_head(out), out
        x = head.eh_proj(mx.concatenate([head.enorm(emb), head.hnorm(hid)], axis=-1))
        x = layer(x, mask, self.cache)
        return p.lm_head(head.shared_head.norm(x)), x

    def _trim(self, n: int) -> None:
        if n <= 0:
            return
        self.cache.trim(n)
        idx = getattr(self.cache, "indexer", None)
        keys = getattr(idx, "keys", None)
        if keys is not None:
            idx.keys = keys[:, : max(0, int(keys.shape[1]) - n)]

    def speculate(self, feed: list[int], bonus: mx.array, k: int):
        """Queue the draft that follows `feed` if the pass keeps all of it.

        Mirrors fed(feed) then draft(bonus) without touching the queued pairs,
        so commit() or rollback() can settle it once the pass is evaluated.
        Returns the lazy draft tokens, or None when there is nothing to do.
        """
        self.spec = None
        h = self.tap.value
        T = len(feed)
        if self.dead or h is None or k <= 0 or T <= 0:
            return None
        try:
            hs, ts = list(self.pend_h), list(self.pend_t)
            if self.pre is None and self.last_h is not None:
                hs.append(self.last_h)
                ts.append(mx.array(feed[:1], dtype=mx.uint32))
            for i in range(1, T):
                hs.append(h[:, i - 1:i])
                ts.append(mx.array(feed[i:i + 1], dtype=mx.uint32))
            cap = max(1, int(config.MTP_CATCHUP))
            if len(hs) > cap:
                hs, ts = hs[-cap:], ts[-cap:]
            last = h[:, T - 1:T]
            hs.append(last)
            ts.append(bonus.reshape(-1)[:1].astype(mx.uint32))
            offset = int(getattr(self.cache, "offset", 0) or 0)
            if offset + len(hs) + k > int(config.MTP_CTX):
                return None
            hid = mx.concatenate(hs, axis=1) if len(hs) > 1 else hs[0]
            toks, conf = self._chain(hid, mx.concatenate(ts)[None], k)
            self.spec = (len(hs), k, last, ts[-1], toks, conf)
            return [toks, conf]
        except Exception as e:
            self.fails += 1
            self.dead = True
            print(f"[mtp] in-pass draft failed, head off for this generation: {e!r}", flush=True)
            return None

    def commit(self) -> None:
        """The pass was kept in full: its pairs are in the head, the draft is ready."""
        n, k, last, bonus, toks, conf = self.spec
        self.spec = None
        self.tap.value = None
        self._trim(k - 1)
        self.pend_h, self.pend_t = [], []
        self.last_h = last
        self.pre = int(bonus.item())
        self.drafts += 1
        self.ready = (self.pre, [int(t) for t in toks.tolist()], float(conf.item()))

    def rollback(self) -> None:
        """The pass was cut short: take the speculative positions back off the head."""
        n, k = self.spec[0], self.spec[1]
        self.spec = None
        self._trim(n + k - 1)

    def _chain(self, hid: mx.array, ids: mx.array, k: int):
        """k greedy head steps, lazy. Also the head's lowest top probability
        over them, which says how likely the whole draft is to be kept."""
        logits, h = self._fwd(hid, ids)
        last = logits[:, -1]
        cur = mx.argmax(last, axis=-1)
        out = [cur]
        confs = [mx.max(mx.softmax(last.astype(mx.float32), axis=-1), axis=-1)]
        h = h[:, -1:]
        for _ in range(k - 1):
            logits, h = self._fwd(h, cur[None])
            last = logits[:, -1]
            cur = mx.argmax(last, axis=-1)
            out.append(cur)
            confs.append(mx.max(mx.softmax(last.astype(mx.float32), axis=-1), axis=-1))
        return mx.concatenate(out), mx.min(mx.concatenate(confs))

    def draft(self, tok: int, k: int) -> list[int]:
        if self.dead or self.last_h is None or k <= 0:
            return []
        ready, self.ready = self.ready, None
        if ready is not None and ready[0] == tok and self.pre == tok:
            self.conf = ready[2]
            return ready[1][:k]
        try:
            hs = self.pend_h + [self.last_h]
            ts = self.pend_t + [mx.array([tok], dtype=mx.uint32)]
            self.pend_h, self.pend_t = [], []
            offset = int(getattr(self.cache, "offset", 0) or 0)
            if offset + len(hs) + k > int(config.MTP_CTX):
                self.cache = self.parts.cache_cls()
            hid = mx.concatenate(hs, axis=1) if len(hs) > 1 else hs[0]
            out, conf = self._chain(hid, mx.concatenate(ts)[None], k)
            mx.eval(out, conf)
            toks = [int(t) for t in out.tolist()]
            self.conf = float(conf.item())
            self._trim(k - 1)
            self.pre = tok
            self.drafts += 1
            return toks
        except Exception as e:
            self.fails += 1
            self.dead = True
            print(f"[mtp] draft failed, head off for this generation: {e!r}", flush=True)
            return []


def attach(model, prep: Prepared) -> Drafter:
    """Install the hidden-state tap and hang the drafter on the model."""
    p = prep.parts
    tap = _Tap()
    if p.scheme == "mtp" and p.hc:
        target = getattr(p.inner, "hyper_connection_mixer", None)
        use_input = True
    else:
        target = getattr(p.inner, "norm", None)
        use_input = False
    if target is None:
        raise Incompatible("backbone has no final norm to read the hidden state from")
    _install_tap(target, tap, use_input)
    drafter = Drafter(prep, tap)
    model._pagedmoe_mtp = drafter
    return drafter


def describe(model) -> dict | None:
    d = getattr(model, "_pagedmoe_mtp", None)
    if d is None:
        return None
    return {"gb": round(d.nbytes / 1e9, 2), "scheme": d.parts.scheme, "drafts": d.drafts,
            "off": d.dead}

