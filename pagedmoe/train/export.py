# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export a run's adapter for GPUs: Hugging Face PEFT format.

    to_peft(run_or_checkpoint)  ->  <dir>/peft/
        adapter_model.safetensors
        adapter_config.json
        README.md

PEFT is what transformers, vLLM and SGLang load, and llama.cpp's
convert_lora_to_gguf.py reads it too. The export only touches the adapter
file and the base model's tensor names, never the weights, so it takes
seconds on any size of model.

mlx-lm and HF name the same layer differently on some models (a multimodal
Qwen3.5 is language_model.model.layers.N in MLX and model.language_model.
layers.N in HF), so names are matched on the part from `layers.N.` on and
checked against the original model's tensor list and shapes. Anything that
doesn't line up stops the export instead of writing an adapter that would
quietly do nothing.

Routed-expert LoRA (lora.target=experts on a resident model) is skipped:
MLX stacks the experts into one tensor and HF has no one standard layout
for that. Attention, shared experts and dense MLPs export.
"""

from __future__ import annotations

import json
import re
import struct
from collections import Counter
from pathlib import Path

_LAYER = re.compile(r"^(.*?)(layers\.\d+\..+)$")


class ExportError(ValueError):
    pass


# ------------------------------------------------------------------ inputs


def find_adapter(path: str | Path, checkpoint: str | None = None) -> Path:
    """A run dir, a checkpoint dir, or a run dir plus a checkpoint id."""
    p = Path(path).expanduser()
    if checkpoint and checkpoint not in ("final", "latest"):
        p = p / "checkpoints" / checkpoint
    if not (p / "adapters.safetensors").is_file():
        raise ExportError(f"no adapters.safetensors in {p}")
    if not (p / "adapter_config.json").is_file():
        raise ExportError(f"no adapter_config.json in {p}")
    return p


def base_repo_of(base: str | Path | None) -> str | None:
    """The original HF repo a local MLX checkpoint was converted from.

    mlx-community and PagedMoE model cards say it in the front matter
    (base_model:) or in "converted to MLX format from [org/name]".
    """
    if not base:
        return None
    p = Path(str(base)).expanduser()
    if not p.is_dir():
        return str(base) if "/" in str(base) and not str(base).startswith((".", "~", "/")) else None
    readme = p / "README.md"
    if readme.is_file():
        text = readme.read_text(errors="ignore")
        front = text.split("---", 2)[1] if text.startswith("---") else ""
        m = re.search(r"^base_model:\s*(?:\n\s*-\s*)?['\"]?([\w.-]+/[\w.-]+)", front, re.M)
        if m:
            return m.group(1)
        m = re.search(r"converted to MLX format from \[([\w.-]+/[\w.-]+)\]", text)
        if m:
            return m.group(1)
    try:
        name = json.loads((p / "config.json").read_text()).get("_name_or_path", "")
        if re.fullmatch(r"[\w.-]+/[\w.-]+", name or ""):
            return name
    except Exception:
        pass
    return None


def _local_shapes(d: Path) -> dict[str, tuple]:
    """name -> shape from the safetensors headers in a folder (no weights read)."""
    out: dict[str, tuple] = {}
    for f in sorted(d.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            head = json.loads(fh.read(n))
        for k, v in head.items():
            if k != "__metadata__":
                out[k] = tuple(v.get("shape") or ())
    return out


def _hub_shapes(repo: str) -> dict[str, tuple]:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars

    quiet = are_progress_bars_disabled()
    disable_progress_bars()
    try:
        meta = HfApi().get_safetensors_metadata(repo)
    finally:
        if not quiet:
            enable_progress_bars()
    out: dict[str, tuple] = {}
    for fm in meta.files_metadata.values():
        for k, t in fm.tensors.items():
            out[k] = tuple(t.shape)
    return out


def reference_names(base_repo: str | None, base_ckpt: str | None, log=print):
    """(name -> shape, where they came from, verified).

    The original model's tensors are the truth. Without them (offline, no
    repo found) the MLX checkpoint's own names stand in, unverified.
    """
    if base_repo:
        d = Path(base_repo).expanduser()
        try:
            if d.is_dir() and not _is_mlx(d):
                return _local_shapes(d), str(d), True
            if not d.is_dir():
                return _hub_shapes(base_repo), f"huggingface.co/{base_repo}", True
            base_ckpt = str(d)
        except Exception as e:
            log(f"[export] couldn't read {base_repo}'s tensor list ({type(e).__name__}); "
                "falling back to the MLX checkpoint's names")
    if base_ckpt and Path(base_ckpt).expanduser().is_dir():
        shapes = _local_shapes(Path(base_ckpt).expanduser())
        # the one known MLX/HF naming split: multimodal wrappers
        fixed = {re.sub(r"^language_model\.model\.", "model.language_model.", k): v for k, v in shapes.items()}
        return fixed, f"{base_ckpt} (MLX names, unverified)", False
    gone = f"the model folder {base_ckpt} isn't there anymore" if base_ckpt else "the adapter doesn't say which model it was trained on"
    raise ExportError(f"can't tell which model this adapter is for: {gone}. "
                      "Point it at the model folder or the original HF repo (--base org/name)")


def _is_mlx(d: Path) -> bool:
    try:
        c = json.loads((d / "config.json").read_text())
    except Exception:
        return False
    return bool(c.get("quantization") or (c.get("text_config") or {}).get("quantization"))


# ------------------------------------------------------------------ mapping


def _modules(weights: dict) -> tuple[dict, list[str], list[str]]:
    """module path -> {a, b, m}; plus skipped expert modules and unknown keys."""
    mods: dict[str, dict] = {}
    other: list[str] = []
    for k, v in weights.items():
        path, _, leaf = k.rpartition(".")
        if leaf in ("lora_a", "lora_b", "m"):
            mods.setdefault(path, {})[{"lora_a": "a", "lora_b": "b", "m": "m"}[leaf]] = v
        else:
            other.append(k)
    experts = sorted(p for p, t in mods.items() if "a" in t and t["a"].ndim == 3)
    for p in experts:
        mods.pop(p)
    broken = [p for p, t in mods.items() if "a" not in t or "b" not in t]
    if broken:
        raise ExportError(f"adapter is missing lora_a / lora_b for {broken[0]}")
    return mods, experts, other


def map_names(mods: dict, ref: dict[str, tuple], check_shapes: bool) -> dict[str, str]:
    """MLX module path -> HF module path, or ExportError with what didn't fit."""
    by_rest: dict[str, set[str]] = {}
    for name in ref:
        if not name.endswith(".weight"):
            continue
        m = _LAYER.match(name[: -len(".weight")])
        if m:
            by_rest.setdefault(m.group(2), set()).add(m.group(1))
    votes: Counter = Counter()
    rests = {}
    for path in mods:
        m = _LAYER.match(path)
        if not m:
            raise ExportError(f"{path} is not inside a decoder layer; nothing to map it to")
        rests[path] = m.group(2)
        votes.update(by_rest.get(m.group(2), ()))
    if not votes:
        sample = next(iter(rests.values()), "?")
        raise ExportError(f"none of the adapted layers exist in the base model (first: {sample}). "
                          "Is this the model the run was trained on?")
    # the main decoder stack covers every layer; side stacks (mtp, vision) only some
    prefix = max(votes, key=lambda p: (votes[p], "mtp" not in p, "model" in p))
    out, missing, shape_bad = {}, [], []
    for path, rest in rests.items():
        if prefix not in by_rest.get(rest, ()):
            missing.append(rest)
            continue
        hf = prefix + rest
        out[path] = hf
        if check_shapes:
            want = ref.get(hf + ".weight")
            a, b = mods[path]["a"], mods[path]["b"]
            got = (b.shape[1], a.shape[0])
            if want and len(want) == 2 and tuple(want) != got:
                shape_bad.append(f"{hf} is {tuple(want)} in the base model, the adapter is {got}")
    if missing:
        raise ExportError(f"{len(missing)} adapted layers have no match in the base model, "
                          f"e.g. {prefix}{missing[0]}. The HF model names this layer differently "
                          "(a fused or split projection), so a PEFT adapter can't target it.")
    if shape_bad:
        raise ExportError("shapes don't match the base model: " + shape_bad[0])
    return out


def _natural(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


# ------------------------------------------------------------------ export


def to_peft(path: str | Path, out: str | Path | None = None, *, checkpoint: str | None = None,
            base: str | None = None, log=print) -> dict:
    """Write a PEFT copy of a run's adapter. Returns what was written."""
    import mlx.core as mx

    src = find_adapter(path, checkpoint)
    cfg = json.loads((src / "adapter_config.json").read_text())
    lp = cfg.get("lora_parameters") or {}
    rank = int(lp.get("rank") or 0)
    scale = float(lp.get("scale") or 0)
    if not rank or not scale:
        raise ExportError("adapter_config.json has no rank / scale")
    dora = cfg.get("fine_tune_type") == "dora"
    if cfg.get("fine_tune_type") not in (None, "lora", "dora"):
        raise ExportError(f"only LoRA / DoRA adapters export, this is {cfg.get('fine_tune_type')}")

    weights = mx.load(str(src / "adapters.safetensors"))
    mods, experts, other = _modules(weights)
    if experts:
        log(f"[export] skipping routed-expert LoRA on {len(experts)} layers "
            "(stacked experts have no standard PEFT layout); everything else exports")
    if other:
        log(f"[export] skipping {len(other)} tensors that aren't LoRA (e.g. {other[0]})")
    if not mods:
        raise ExportError("nothing to export: the adapter only has routed-expert LoRA")
    if (src / "encoder.safetensors").exists() or (src.parent.parent / "encoder.safetensors").exists():
        log("[export] the trained input encoder isn't part of a PEFT adapter and is not included")

    base_ckpt = cfg.get("base_model")
    picked = Path(str(base)).expanduser() if base else None
    if picked and picked.is_dir() and _is_mlx(picked):
        # an MLX copy stands in for a moved model folder; GPU users still need the HF repo
        base_ckpt, base = str(picked), None
    repo = base or base_repo_of(base_ckpt) or cfg.get("base_repo")
    if base_ckpt and (Path(base_ckpt).expanduser() / "prune.json").is_file():
        log(f"[export] the run was trained on a pruned copy of {repo or 'the base model'}. The adapter "
            "loads on the full model, which also has the experts the pruned copy dropped")
    ref, where, verified = reference_names(repo, base_ckpt, log=log)
    names = map_names(mods, ref, check_shapes=verified)

    tensors = {}
    for path_, hf in names.items():
        t = mods[path_]
        key = f"base_model.model.{hf}"
        tensors[f"{key}.lora_A.weight"] = mx.contiguous(t["a"].T)
        tensors[f"{key}.lora_B.weight"] = mx.contiguous(t["b"].T)
        if dora:
            if "m" not in t:
                raise ExportError(f"DoRA adapter has no magnitude for {path_}")
            tensors[f"{key}.lora_magnitude_vector"] = t["m"]

    dst = Path(out).expanduser() if out else src / "peft"
    dst.mkdir(parents=True, exist_ok=True)
    mx.eval(list(tensors.values()))
    mx.save_safetensors(str(dst / "adapter_model.safetensors"), tensors, metadata={"format": "pt"})
    peft_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": repo or "",
        "r": rank,
        "lora_alpha": scale * rank,
        "lora_dropout": float(lp.get("dropout") or 0.0),
        # full paths, not a regex: on transformers 5 MoE models PEFT turns a
        # regex into leaf names, so shared_expert.gate_proj would also wrap
        # the fused routed experts
        "target_modules": sorted(names.values(), key=_natural),
        "bias": "none",
        "use_dora": dora,
        "use_rslora": False,
        "fan_in_fan_out": False,
        "init_lora_weights": True,
        "inference_mode": True,
        "modules_to_save": None,
    }
    (dst / "adapter_config.json").write_text(json.dumps(peft_cfg, indent=2) + "\n")
    _readme(dst, repo, len(names), verified, where)

    info = {"out": str(dst), "base": repo, "modules": len(names), "tensors": len(tensors),
            "skipped_experts": len(experts), "checked_against": where, "verified": verified}
    if verified:
        note = ""
    elif repo:
        note = " (names not checked, couldn't reach the original model; export again online to check)"
    else:
        note = " (names not checked, no original model found; pass --base org/name to check)"
    log(f"[export] PEFT adapter for {repo or 'the base model'}: {len(names)} layers, "
        f"r={rank}, alpha={scale * rank:g} -> {dst}{note}")
    return info


def _readme(dst: Path, repo: str | None, n: int, verified: bool, where: str) -> None:
    base = repo or "<the original model>"
    lines = [
        "---",
        "library_name: peft",
        *([f"base_model: {repo}"] if repo and "/" in repo and not Path(repo).expanduser().is_dir() else []),
        "tags:",
        "- pagedmoe",
        "- lora",
        "---",
        "",
        "# LoRA adapter (PEFT)",
        "",
        f"Trained with PagedMoE on an MLX copy of {base} and exported to PEFT.",
        f"{n} adapted layers. Names {'checked against ' + where if verified else 'not checked (exported offline)'}.",
        "",
        "Load it on top of the original (full precision) model:",
        "",
        "```python",
        "from peft import PeftModel",
        "from transformers import AutoModelForCausalLM",
        "",
        f'model = AutoModelForCausalLM.from_pretrained("{base}", torch_dtype="auto", device_map="auto")',
        'model = PeftModel.from_pretrained(model, "path/to/this/folder")',
        "```",
        "",
        "vLLM: `vllm serve <base> --enable-lora --lora-modules mine=path/to/this/folder`.",
        "llama.cpp: `python convert_lora_to_gguf.py path/to/this/folder --base <base folder>`.",
        "",
        "It was trained against the 4-bit MLX weights. On the full precision model it",
        "behaves like a QLoRA adapter: the same fine-tune, with small numeric drift.",
        "",
    ]
    (dst / "README.md").write_text("\n".join(lines))
