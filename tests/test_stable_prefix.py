# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Host-declared stable front matter (`stable_prefix_messages`)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from pagedmoe import prefix_cache

TOKENIZER_DIR = Path(
    os.environ.get(
        "PAGED_MOE_TEST_TOKENIZER",
        str(Path.home() / "mlx-models" / "Tiel-Coder-35B-A3B-MLX-oQ6e"),
    )
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"filePath": {"type": "string"}},
                "required": ["filePath"],
            },
        },
    }
]

FRONT = [
    {"role": "system", "content": "You are a coding agent. " * 200},
    {"role": "user", "content": "[CODEBASE_MAP] map.md: src/a.ts src/b.ts"},
    {"role": "assistant", "content": "Noted the codebase map."},
    {"role": "user", "content": "[WORKSPACE_PROJECT_MEMORY] notes"},
    {"role": "assistant", "content": "Noted the workspace project memory."},
]


def _tokens(tok, messages):
    out = tok.apply_chat_template(
        messages, tools=TOOLS, add_generation_prompt=True, tokenize=True
    )
    if isinstance(out, dict) or hasattr(out, "input_ids"):
        out = out["input_ids"]
    return list(out)


@unittest.skipUnless((TOKENIZER_DIR / "tokenizer_config.json").exists(), "no test tokenizer")
class TestStablePrefixBound(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mlx_lm.utils import load_tokenizer

        cls.tok = load_tokenizer(TOKENIZER_DIR)

    def _bound(self, messages, n):
        from pagedmoe import server

        server._pending_stable = (
            [dict(m) for m in messages[:n]],
            TOOLS,
            {},
        )
        try:
            return server._stable_prefix_bound(self.tok, _tokens(self.tok, messages))
        finally:
            server._pending_stable = None

    def test_bound_survives_a_history_rewrite(self):
        before = FRONT + [
            {"role": "user", "content": "Add a toggle to settings.ts"},
            {"role": "assistant", "content": "Reading it now."},
            {"role": "user", "content": "also persist it"},
        ]
        after = FRONT + [
            {"role": "user", "content": "[SUMMARY_BLOCK_handoff] 14 earlier message(s) compacted."},
            {"role": "assistant", "content": "Noted the handoff."},
            {"role": "user", "content": "also persist it"},
        ]
        a = _tokens(self.tok, before)
        b = _tokens(self.tok, after)
        bound = self._bound(before, len(FRONT))
        self.assertGreater(bound, 0)
        self.assertEqual(bound, self._bound(after, len(FRONT)))
        # The pin key is a prefix of both prompts, so the rewritten one hits it.
        self.assertGreaterEqual(prefix_cache.common_prefix_len(a, b), bound)
        # And it covers the front matter, not just the system message.
        from pagedmoe.server import _first_message_boundary

        self.assertGreater(bound, _first_message_boundary(self.tok, a))

    def test_no_spec_means_no_bound(self):
        from pagedmoe import server

        server._pending_stable = None
        self.assertEqual(server._stable_prefix_bound(self.tok, list(range(5000))), 0)


class TestHybridPin(unittest.TestCase):
    def _cache(self, n):
        c = ArraysCache(1)
        c[0] = mx.zeros((1, n, 4))
        c.offset = n
        return [c]

    def test_full_match_hits_partial_match_misses(self):
        pin = prefix_cache.StablePrefixPin()
        toks = list(range(3000))
        cache = self._cache(2000)
        # ArraysCache cannot trim: the pin must be taken with KV exactly at the bound.
        self.assertTrue(pin.update("m", toks, cache, 2000))
        hit = pin.lookup("m", toks[:2000] + [9, 9, 9])
        self.assertIsNotNone(hit)
        self.assertEqual(hit[2], 2000)
        # Rewritten inside the pinned span: a hybrid cache cannot serve it.
        self.assertIsNone(pin.lookup("m", toks[:1500] + [7] * 1000))

    def test_pin_refused_when_kv_is_past_the_bound(self):
        pin = prefix_cache.StablePrefixPin()
        self.assertFalse(pin.update("m", list(range(3000)), self._cache(2500), 2000))


class _Tok:
    eos_token_ids = {0}
    has_thinking = False


class _ChatTok:
    """im_end=1, im_start=2, endoftext=3; the server adds im_start as a stop."""

    has_thinking = False

    def __init__(self):
        self.eos_token_ids = {1}

    def get_added_vocab(self):
        return {"<|im_end|>": 1, "<|im_start|>": 2, "<|endoftext|>": 3}


class TestMessageEnds(unittest.TestCase):
    def test_turn_openers_stop_replies_but_are_not_message_ends(self):
        from pagedmoe import server

        tok = _ChatTok()
        server._stop_on_endoftext(tok)
        self.assertEqual(tok.eos_token_ids, {1, 2, 3})
        self.assertEqual(server._message_ends(tok), {1, 3})
        # system, user, assistant, user "OK", then the generation prompt.
        ids = [2, 9, 9, 1, 7, 2, 9, 9, 1, 7, 2, 9, 9, 1, 7, 2, 9, 5, 1, 7, 2, 9, 7]
        self.assertEqual(server._last_message_start(tok, ids), 14)
        self.assertEqual(server._first_message_boundary(tok, ids), 4)


class TestHybridSnapshotPath(unittest.TestCase):
    """_snapshot_user_segment on a non-trimmable cache with a stable bound."""

    def _run(self, tokens, cached, stable_at, prev=None, head_at=0, on_disk=False, shared_at=0, ck=False):
        from unittest import mock
        from pagedmoe import inpass_snap, prefix_disk, server

        cache = [ArraysCache(1)]
        cache[0][0] = mx.zeros((1, max(cached, 1), 4))
        cache[0].offset = cached
        stops = []

        def fake_prefill(model, prompt, c, *, processed, target, ckpt_at=None, ckpt_out=None, **_):
            if ckpt_out is not None:
                for r in ckpt_at or ():
                    if processed <= r < target:
                        ckpt_out[r] = {0: [mx.zeros((1, 4))]}
            stops.append(cached + target)
            c[0][0] = mx.zeros((1, cached + target, 4))
            c[0].offset = cached + target
            return target

        store = mock.Mock()
        store.max_bytes = 1 << 40
        server._req_ctx.tokens = list(tokens)
        server._req_ctx.cached = cached
        server._req_ctx.store = store
        server._req_ctx.model_key = "m"
        server._recent_prompts.clear()
        if prev is not None:
            server._recent_prompts.append(list(prev))
        model = mock.Mock()
        model.layers = [None]
        head_spec = ("head",) if head_at else None
        shared_spec = ("shared",) if shared_at else None
        self.saved = []

        def bound(_tok, _toks, spec="pending"):
            if spec is head_spec and head_spec:
                return head_at
            if spec is shared_spec and shared_spec:
                return shared_at
            return stable_at

        def has(_key, toks):
            return on_disk is True or (isinstance(on_disk, set) and len(toks) in on_disk)

        self.ckpts = {}

        def save(key, toks, c, _model, _budget, kind=prefix_disk.KIND_FRONT, ckpts=None, replay=None):
            self.saved.append((kind, len(toks), c[0].offset))
            self.ckpts[kind] = sorted(ckpts or ())
            return True

        with mock.patch.object(inpass_snap, "supports", return_value=ck), mock.patch.object(
            server, "_prefill_remainder", fake_prefill
        ), mock.patch.object(
            server, "_stable_prefix_bound", side_effect=bound
        ), mock.patch.object(server, "_pending_stable_head", head_spec), mock.patch.object(
            server, "_pending_stable_shared", shared_spec
        ), mock.patch.object(
            server, "_served_model", model
        ), mock.patch.object(prefix_disk, "supports", return_value=True), mock.patch.object(
            prefix_disk, "has", side_effect=has
        ), mock.patch.object(prefix_disk, "save", side_effect=save):
            server._snapshot_user_segment(
                model,
                {
                    "tokenizer": _Tok(),
                    "prompt": mx.array(tokens[cached:]),
                    "prompt_cache": cache,
                },
            )
        return stops, cache

    def test_head_file_saved_before_memory_once(self):
        tokens = list(range(1, 6001))
        stops, cache = self._run(tokens, cached=0, stable_at=3000, head_at=2000)
        self.assertEqual(stops[:2], [2000, 3000])
        self.assertEqual(stops[-1], 5999)
        self.assertIn(("head", 2000, 2000), self.saved, "head file not written at the head")
        self.assertIn(("front", 3000, 3000), self.saved)
        self.assertEqual(cache[0].offset, 5999)

        stops, _ = self._run(tokens, cached=0, stable_at=3000, head_at=2000, on_disk=True)
        self.assertNotIn(2000, stops, "stopped again with the head file on disk")
        self.assertNotIn("head", [k for k, _, _ in self.saved])

    def test_disk_checkpoints_add_no_stops(self):
        from pagedmoe import config as es_config
        from pagedmoe import server

        tokens = list(range(1, 20001))
        old = (es_config.PREFIX_DISK_CKPT_EVERY, es_config.PREFIX_DISK_CKPT_MAX)
        es_config.PREFIX_DISK_CKPT_EVERY, es_config.PREFIX_DISK_CKPT_MAX = 2048, 4
        try:
            es_config.PREFIX_DISK_CKPT_EVERY = 0
            server._stable_pin._pins.clear()
            plain, _ = self._run(tokens, cached=0, stable_at=14000, head_at=9000, ck=True)
            self.assertEqual(self.ckpts["front"], [])
            es_config.PREFIX_DISK_CKPT_EVERY = 2048
            server._stable_pin._pins.clear()
            stops, _ = self._run(tokens, cached=0, stable_at=14000, head_at=9000, ck=True)
            self.assertEqual(stops, plain, "checkpoints changed where prefill stops")
            self.assertEqual(self.ckpts["head"], [4096, 8192])
            self.assertEqual(self.ckpts["front"], [4096, 8192, 12288])
            server._stable_pin._pins.clear()
            stops, _ = self._run(tokens, cached=6000, stable_at=14000, ck=True)
            self.assertEqual(self.ckpts["front"], [8192, 12288], "took a checkpoint it resumed past")
        finally:
            es_config.PREFIX_DISK_CKPT_EVERY, es_config.PREFIX_DISK_CKPT_MAX = old
            server._stable_pin._pins.clear()

    def test_shared_file_saved_before_the_head(self):
        tokens = list(range(1, 6001))
        stops, cache = self._run(tokens, cached=0, stable_at=3000, head_at=2000, shared_at=1500)
        self.assertEqual(stops[:3], [1500, 2000, 3000])
        self.assertEqual(stops[-1], 5999)
        self.assertIn(("shared", 1500, 1500), self.saved)
        self.assertIn(("head", 2000, 2000), self.saved)
        self.assertEqual(cache[0].offset, 5999)

        stops, _ = self._run(tokens, cached=0, stable_at=3000, head_at=2000, shared_at=1500, on_disk={1500})
        self.assertEqual(stops[:2], [2000, 3000], "stopped at the shared span with its file on disk")
        self.assertNotIn("shared", [k for k, _, _ in self.saved])

        stops, _ = self._run(tokens, cached=1500, stable_at=3000, head_at=2000, shared_at=1500)
        self.assertNotIn(1500, stops, "stopped at the shared span it resumed from")
        self.assertEqual(stops[:2], [2000, 3000])

        stops, _ = self._run(tokens, cached=0, stable_at=3000, shared_at=1500)
        self.assertEqual(stops[:2], [1500, 3000], "no memory block: shared then front")

        stops, _ = self._run(tokens, cached=0, stable_at=3000, head_at=2000, shared_at=2000)
        self.assertNotIn("shared", [k for k, _, _ in self.saved], "shared span not shorter than the head")
        self.assertEqual(stops.count(2000), 1)

    def test_mid_snap_lands_before_a_repeated_reminder_tail(self):
        # Dual research: the host drops last turn's reminder and appends it
        # again after the new tool result. The snapshot belongs right before
        # the new copy, not at the end and not where the old copy sat.
        hist = list(range(1, 3000)) + [0]
        new = list(range(5000, 7000)) + [0]
        tail = [8] * 200
        stops, _ = self._run(hist + new + tail, cached=0, stable_at=0, prev=hist + tail)
        self.assertIn(len(hist + new), stops, f"mid-snap not before the reminder: {stops}")
        self.assertNotIn(len(hist), stops)

    def test_mid_snap_before_the_reminder_when_the_old_tail_had_a_nudge(self):
        hist = list(range(1, 3000)) + [0]
        nudge = [7] * 60 + [0]
        new = list(range(5000, 7000)) + [0]
        tail = [8] * 200
        stops, _ = self._run(hist + new + tail, cached=0, stable_at=0, prev=hist + nudge + tail)
        self.assertIn(len(hist + new), stops, f"mid-snap not before the reminder: {stops}")
        self.assertNotIn(len(hist), stops)

    def test_mid_snap_at_the_previous_end_when_the_tail_is_new(self):
        hist = list(range(1, 3000)) + [0]
        stops, _ = self._run(hist + list(range(5000, 7200)), cached=0, stable_at=0, prev=hist + [9] * 200)
        self.assertIn(len(hist), stops, f"mid-snap not where the previous prompt diverged: {stops}")

    def test_no_head_stop_when_resumed_past_it(self):
        tokens = list(range(1, 6001))
        stops, _ = self._run(tokens, cached=2500, stable_at=3000, head_at=2000, prev=tokens[:5000])
        self.assertNotIn(2000, stops)
        self.assertNotIn("head", [k for k, _, _ in self.saved])

    def test_head_ignored_unless_shorter_than_front(self):
        tokens = list(range(1, 6001))
        stops, _ = self._run(tokens, cached=0, stable_at=3000, head_at=3000)
        self.assertEqual(stops.count(3000), 1)
        self.assertNotIn("head", [k for k, _, _ in self.saved])

    def setUp(self):
        from pagedmoe import server

        server._stable_pin.clear()

    def test_cold_prompt_stops_at_front_matter_then_finishes(self):
        from pagedmoe import server

        tokens = list(range(1, 6001))
        stops, cache = self._run(tokens, cached=0, stable_at=3000)
        self.assertEqual(stops[0], 3000)
        self.assertEqual(stops[-1], 5999)
        pin = server._stable_pin.get("m")
        self.assertIsNotNone(pin)
        self.assertEqual(len(pin[0]), 3000)
        # Live cache still ends where decode expects it.
        self.assertEqual(cache[0].offset, 5999)

    def test_rewritten_history_resumes_from_pin(self):
        from pagedmoe import server

        tokens = list(range(1, 6001))
        self._run(tokens, cached=0, stable_at=3000)
        rewritten = tokens[:3000] + [9] * 1500
        hit = server._stable_pin.lookup("m", rewritten)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[2], 3000)

    def test_no_extra_stop_when_already_past_bound(self):
        tokens = list(range(1, 6001))
        stops, _ = self._run(tokens, cached=3000, stable_at=3000, prev=tokens[:5000])
        self.assertNotIn(3000, stops)

    def test_served_head_pin_does_not_grow_past_bound(self):
        from pagedmoe import server

        tokens = list(range(1, 20001))
        self._run(tokens, cached=0, stable_at=3000)
        # A later hit at 5000 (inside history) must not replace the front matter pin.
        self._run(tokens, cached=5000, stable_at=3000, prev=tokens[:15000])
        self.assertEqual(len(server._stable_pin.get("m")[0]), 3000)


if __name__ == "__main__":
    unittest.main()
