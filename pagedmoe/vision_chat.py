# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Image parts for resident chat.

mlx-lm rejects any content list that is not all text, and its Qwen loader
drops ``vision_tower.*``. The chat template already knows ``image_url``.
This keeps those parts, runs only the vision tower from the checkpoint, and
writes the features onto ``<|image_pad|>`` embeddings inside the model that
is already loaded.

Every image of the same size expands to the same pad tokens, so a short
content tag goes in front of each image. Without it the prompt cache could
hand back KV built from a different picture.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
import traceback
from collections import OrderedDict

import numpy as np

_tls = threading.local()
_tower = None
_tower_key = None
_processor = None
_processor_key = None
_installed = False

_PAD = "<|image_pad|>"
_START = "<|vision_start|>"
_FEATURE_SLOTS = 8
_features: "OrderedDict[str, tuple]" = OrderedDict()


def install() -> None:
    global _installed
    if _installed:
        return
    import mlx.nn as nn
    import mlx_lm.server as mlx_server

    _allow_image_content(mlx_server)
    _hook_embeddings(nn)
    _hook_tokenize(mlx_server)
    _hook_stream(mlx_server)
    _installed = True


def _allow_image_content(mlx_server) -> None:
    import json as _json

    def process_message_content(messages):
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                if not any(_image_part(part) for part in content):
                    texts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            texts.append(part.get("text") or "")
                        else:
                            raise ValueError("Only 'text' content type is supported.")
                    message["content"] = "".join(texts)
            elif content is None:
                message["content"] = ""

            tool_calls = message.get("tool_calls")
            if tool_calls:
                for tool_call in tool_calls:
                    func = tool_call.get("function") if isinstance(tool_call, dict) else None
                    if not func:
                        continue
                    args = func.get("arguments")
                    if isinstance(args, str) and args:
                        func["arguments"] = _json.loads(args)

    mlx_server.process_message_content = process_message_content


def _hook_embeddings(nn) -> None:
    from mlx.nn.layers.quantized import QuantizedEmbedding

    def wrap(cls):
        if getattr(cls, "_pagedmoe_vision", False):
            return
        orig = cls.__call__

        def wrapped(self, inputs, *args, **kwargs):
            out = orig(self, inputs, *args, **kwargs)
            state = getattr(_tls, "state", None)
            if not state or state.get("done"):
                return out
            return _scatter(out, inputs, state)

        wrapped._pagedmoe_vision = True
        cls.__call__ = wrapped
        cls._pagedmoe_vision = True

    wrap(nn.Embedding)
    wrap(QuantizedEmbedding)


def _hook_tokenize(mlx_server) -> None:
    gen = mlx_server.ResponseGenerator
    if getattr(gen._tokenize, "_pagedmoe_vision", False):
        return
    orig = gen._tokenize

    def _tokenize(self, tokenizer, request, args):
        _tls.state = None
        messages = getattr(request, "messages", None) or []
        if getattr(request, "request_type", None) != "chat" or not _has_images(messages):
            return orig(self, tokenizer, request, args)
        try:
            return _vision_tokenize(self, tokenizer, request, args)
        except Exception:
            print("[vision] image prepare failed", flush=True)
            traceback.print_exc()
            _tls.state = None
            raise

    _tokenize._pagedmoe_vision = True
    gen._tokenize = _tokenize


def _hook_stream(mlx_server) -> None:
    """Skip the features of image tokens the prompt cache already holds.

    stream_generate gets only the uncached tail of the prompt, so the first
    image token the embedding sees may belong to the second image.
    """
    orig = mlx_server.stream_generate
    if getattr(orig, "_pagedmoe_vision", False):
        return

    def stream_generate(*args, **kwargs):
        state = getattr(_tls, "state", None)
        rest = kwargs.get("prompt")
        if state and not state.get("done") and rest is not None:
            try:
                left = int(np.count_nonzero(np.asarray(rest).reshape(-1) == state["image_id"]))
                total = int(state["feats"].shape[0])
                state["cursor"] = max(0, total - left)
                if left == 0:
                    state["done"] = True
                    state["feats"] = None
            except Exception:
                pass
        return orig(*args, **kwargs)

    stream_generate._pagedmoe_vision = True
    mlx_server.stream_generate = stream_generate


def _image_part(part) -> bool:
    if not isinstance(part, dict):
        return False
    kind = part.get("type")
    return kind in ("image", "image_url") or "image" in part or "image_url" in part


def _has_images(messages) -> bool:
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list) and any(_image_part(part) for part in content):
            return True
    return False


def _data_images(messages):
    from PIL import Image

    images = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not _image_part(part):
                continue
            url = _part_url(part)
            if not url or not str(url).startswith("data:") or "," not in str(url):
                raise ValueError("image content must be a data URL")
            raw = base64.b64decode(str(url).split(",", 1)[1])
            digest = hashlib.sha1(raw).hexdigest()[:12]
            images.append((digest, Image.open(io.BytesIO(raw)).convert("RGB")))
    if not images:
        raise ValueError("no image bytes in the request")
    return images


def _part_url(part) -> str:
    if part.get("type") == "image_url":
        image_url = part.get("image_url") or {}
        if isinstance(image_url, str):
            return image_url
        return image_url.get("url") or ""
    return part.get("image") or part.get("url") or ""


def _vision_tokenize(gen, tokenizer, request, args):
    import mlx.core as mx
    import mlx_lm.server as mlx_server

    mlx_server.process_message_content(request.messages)
    images = _data_images(request.messages)
    model_path = gen.model_provider.cli_args.model
    image_id, features, grids, merge = _encode_images(model_path, images)
    digests = [digest for digest, _ in images]

    chat_template_args = dict(getattr(gen.model_provider.cli_args, "chat_template_args", None) or {})
    extra = getattr(args, "chat_template_kwargs", None) or {}
    if extra:
        chat_template_args.update(extra)
    template_kwargs = dict(tools=getattr(request, "tools", None), **chat_template_args)
    template_kwargs["tokenize"] = False
    rendered = tokenizer.apply_chat_template(
        request.messages,
        add_generation_prompt=True,
        **template_kwargs,
    )
    if not isinstance(rendered, str):
        rendered = tokenizer.decode(rendered)

    expanded = _expand_pads(rendered, grids, merge, digests)
    prompt = tokenizer.encode(expanded, add_special_tokens=False)
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    prompt = [int(t) for t in prompt]

    n_img = sum(1 for tok in prompt if tok == image_id)
    n_feat = int(features.shape[0])
    if n_img != n_feat:
        raise RuntimeError(f"image tokens {n_img} != vision features {n_feat}")

    _tls.state = {
        "image_id": int(image_id),
        "feats": features,
        "cursor": 0,
        "done": False,
        "grids": grids,
        "merge": merge,
        "digests": digests,
    }
    print(f"[vision] {len(images)} image(s), {n_feat} tokens", flush=True)

    initial_state = "normal"
    if getattr(tokenizer, "has_thinking", False):
        think_start = tokenizer.rfind_think_start(prompt)
        think_end = tokenizer.rfind_think_end(prompt)
        if think_start > think_end:
            initial_state = "reasoning"
    return prompt, [prompt], ["user"], initial_state


def expand_rendered(text: str) -> str:
    """Rendered chat text with its image slots expanded like this request's prompt.

    Anything that re-renders the messages after _tokenize (replay splice) gets
    one pad per image from the template. Without this it would hand back a
    prompt with a single image token per picture. Slots match images in
    message order, so a render of the first k messages takes the first images.
    """
    state = getattr(_tls, "state", None)
    if not state or _PAD not in text:
        return text
    n = text.count(_PAD)
    grids = state["grids"]
    if n > len(grids):
        raise RuntimeError(f"rendered text has {n} image slots for {len(grids)} images")
    return _expand_pads(text, grids[:n], state["merge"], state["digests"][:n])


def image_tokens(ids) -> int | None:
    """Image pad ids in `ids` for this request, None when it has no pictures."""
    state = getattr(_tls, "state", None)
    if not state:
        return None
    return int(np.count_nonzero(np.asarray(list(ids)).reshape(-1) == state["image_id"]))


def _tag(part: str, digest: str) -> str:
    """Put the image's content tag just ahead of its vision start token."""
    if not digest:
        return part
    tag = f"[image {digest[:8]}]"
    if part.endswith(_START):
        return part[: -len(_START)] + tag + _START
    return part + tag


def _expand_pads(rendered: str, grids, merge: int, digests=None) -> str:
    parts = rendered.split(_PAD)
    if len(parts) - 1 != len(grids):
        raise RuntimeError(
            f"chat template has {len(parts) - 1} image slots for {len(grids)} images"
        )
    digests = list(digests or [])
    chunks = []
    unit = max(int(merge), 1) ** 2
    for i, grid in enumerate(grids):
        n = int(np.prod(grid) // unit)
        if n < 1:
            raise RuntimeError(f"image grid {grid} produced no tokens")
        chunks.append(_tag(parts[i], digests[i] if i < len(digests) else ""))
        chunks.append(_PAD * n)
    chunks.append(parts[-1])
    return "".join(chunks)


def _encode_one(model_path, digest, image):
    """Vision features for one image. Tool rounds resend the same picture, so
    the last few are kept by content hash."""
    import mlx.core as mx

    key = f"{model_path}:{digest}"
    hit = _features.get(key)
    if hit is not None:
        _features.move_to_end(key)
        return hit
    tower, _image_id, _merge = _tower_for(model_path)
    processor = _processor_for(model_path)
    processed = processor.image_processor([image])
    pixels = mx.array(processed["pixel_values"]).astype(tower.patch_embed.proj.weight.dtype)
    grid = mx.array(processed["image_grid_thw"])
    hidden, _deep = tower(pixels, grid)
    mx.eval(hidden)
    out = (hidden, np.asarray(processed["image_grid_thw"]).tolist()[0])
    _features[key] = out
    while len(_features) > _FEATURE_SLOTS:
        _features.popitem(last=False)
    return out


def _encode_images(model_path, images):
    import mlx.core as mx

    _tower, image_id, merge = _tower_for(model_path)
    feats, grids = [], []
    for digest, image in images:
        hidden, grid = _encode_one(model_path, digest, image)
        feats.append(hidden)
        grids.append(grid)
    hidden = feats[0] if len(feats) == 1 else mx.concatenate(feats, axis=0)
    return image_id, hidden, grids, merge


def _tower_for(model_path):
    global _tower, _tower_key
    model_path = os.path.expanduser(model_path)
    if _tower is not None and _tower_key == model_path:
        return _tower
    import mlx.core as mx
    from mlx_vlm.models.qwen3_vl.config import VisionConfig
    from mlx_vlm.models.qwen3_vl.vision import VisionModel

    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as handle:
        cfg = json.load(handle)
    image_id = int(cfg.get("image_token_id") or 0)
    if not image_id:
        raise RuntimeError("checkpoint has no image_token_id")
    vision_cfg = cfg.get("vision_config") or {}
    if not vision_cfg:
        raise RuntimeError("checkpoint has no vision tower")
    merge = int(vision_cfg.get("spatial_merge_size") or 2)
    tower = VisionModel(VisionConfig.from_dict(vision_cfg))

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    shards = sorted({name for key, name in weight_map.items() if key.startswith("vision_tower.")})
    if not shards:
        raise RuntimeError("no vision_tower weights in the checkpoint")

    weights = {}
    for name in shards:
        print(f"[vision] loading {name}", flush=True)
        shard = mx.load(os.path.join(model_path, name))
        for key, value in shard.items():
            if key.startswith("vision_tower."):
                weights[key[len("vision_tower.") :]] = value
        del shard
    tower.load_weights(list(tower.sanitize(weights).items()), strict=True)
    mx.eval(tower.parameters())
    _tower = (tower, image_id, merge)
    _tower_key = model_path
    return _tower


def _processor_for(model_path):
    global _processor, _processor_key
    model_path = os.path.expanduser(model_path)
    if _processor is not None and _processor_key == model_path:
        return _processor
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    _processor = Qwen3VLProcessor.from_pretrained(model_path, local_files_only=True)
    _processor_key = model_path
    return _processor


def _scatter(embeds, inputs, state):
    import mlx.core as mx

    ids = np.asarray(inputs).reshape(-1)
    pos = np.flatnonzero(ids == state["image_id"])
    if pos.size == 0:
        return embeds
    feats = state["feats"]
    cursor = int(state["cursor"])
    if feats is None or cursor + int(pos.size) > int(feats.shape[0]):
        state["done"] = True
        state["feats"] = None
        print("[vision] feature cursor ran out, leaving the rest as text tokens", flush=True)
        return embeds
    piece = feats[cursor : cursor + int(pos.size)]
    state["cursor"] = cursor + int(pos.size)
    if state["cursor"] >= int(feats.shape[0]):
        state["done"] = True
        state["feats"] = None
    flat = embeds.reshape(-1, embeds.shape[-1])
    piece = piece.astype(flat.dtype)
    flat[mx.array(pos.astype(np.uint32))] = piece
    return flat.reshape(embeds.shape)
