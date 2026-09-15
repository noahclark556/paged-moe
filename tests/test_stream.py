# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Correctness test: streamed inference must produce IDENTICAL tokens to
fully-resident inference.

We build a tiny random Qwen3-MoE checkpoint (standard MLX format, 4-bit
quantized) in a temp dir, load it both ways, and compare greedy generations:

  1. resident (plain mlx-lm path)
  2. streamed with a generous cache
  3. streamed with a starvation-level cache (forces constant eviction)

All three must emit the same token ids. Run:

    python tests/test_stream.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from mlx_lm.models import qwen3_moe
from mlx_lm.models.cache import make_prompt_cache

from expert_stream import get_stats, load
from expert_stream.streaming import _nucleus_mask

CONFIG = {
    "model_type": "qwen3_moe",
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "intermediate_size": 256,
    "num_attention_heads": 4,
    "num_experts": 16,
    "num_experts_per_tok": 4,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "moe_intermediate_size": 64,
    "rms_norm_eps": 1e-6,
    "vocab_size": 512,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "max_position_embeddings": 2048,
    "norm_topk_prob": True,
    "quantization": {"group_size": 32, "bits": 4},
}


def build_tiny_model(dest: Path):
    mx.random.seed(7)
    args = qwen3_moe.ModelArgs.from_dict(CONFIG)
    model = qwen3_moe.Model(args)
    nn.quantize(model, group_size=32, bits=4)
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(dest / "model.safetensors"), weights)
    with open(dest / "config.json", "w") as f:
        json.dump(CONFIG, f)


def greedy_generate(model, prompt_ids, n_new, want_logits=False):
    cache = make_prompt_cache(model)
    y = mx.array([prompt_ids])
    out = []
    rows = []
    logits = model(y, cache=cache)
    tok = int(mx.argmax(logits[:, -1, :]))
    out.append(tok)
    for _ in range(n_new - 1):
        logits = model(mx.array([[tok]]), cache=cache)
        rows.append(np.asarray(logits[0, -1, :].astype(mx.float32)))
        tok = int(mx.argmax(logits[:, -1, :]))
        out.append(tok)
    return (out, np.stack(rows)) if want_logits else out


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        build_tiny_model(tmp)

        # Long-ish prompt so prefill exercises the stacked gather path.
        rng = np.random.default_rng(3)
        prompt = rng.integers(0, CONFIG["vocab_size"], size=50).tolist()
        n_new = 20

        from expert_stream import config as es_config

        print("== resident ==")
        model_r, _tok = load(str(tmp), mode="resident", verbose=True)
        ref = greedy_generate(model_r, prompt, n_new)
        del model_r

        # Slabs are on by default, so pin them off here: the per-expert path
        # is still used for prefill and for models slabs can't address, and it
        # needs its own coverage regardless of what the default happens to be.
        es_config.SLAB = "off"

        print("== streamed (roomy cache) ==")
        model_s, _tok = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=True)
        got = greedy_generate(model_s, prompt, n_new)
        stats = get_stats(model_s)
        print("cache stats:", json.dumps(stats["cache"]))
        assert got == ref, f"streamed != resident\n  ref={ref}\n  got={got}"
        assert stats["cache"]["misses"] > 0, "expected disk reads"
        del model_s

        print("== streamed (starved cache: 5 experts' worth) ==")
        # Each expert here is ~30 KB; 0.00015 GB ≈ 5 experts resident max.
        model_t, _tok = load(
            str(tmp), mode="streamed", cache_gb=0.00015, verbose=True
        )
        got2 = greedy_generate(model_t, prompt, n_new)
        stats2 = get_stats(model_t)
        print("cache stats:", json.dumps(stats2["cache"]))
        assert got2 == ref, f"starved streamed != resident\n  ref={ref}\n  got={got2}"
        assert stats2["cache"]["evictions"] > 0, "expected evictions"
        del model_t

        print("== streamed + prune (epsilon threshold: must stay identical) ==")
        es_config.PRUNE = 1e-9  # keeps every slot; only exercises the code path
        try:
            model_p, _tok = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=True)
            got3 = greedy_generate(model_p, prompt, n_new)
            assert got3 == ref, f"prune eps changed output\n  ref={ref}\n  got={got3}"
            del model_p

            print("== streamed + prune 0.9 (aggressive: must run, may differ) ==")
            es_config.PRUNE = 0.9
            es_config.KEEP_FREE = False  # isolate the selector from cache state
            model_q, _tok = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=True)
            got4 = greedy_generate(model_q, prompt, n_new)
            stats4 = get_stats(model_q)
            assert len(got4) == n_new
            assert stats4["cache"]["pruned_frac"] > 0, "expected pruned slots"
            print(f"pruned_frac={stats4['cache']['pruned_frac']}")
            del model_q

            # With KEEP_FREE on and a cache big enough to hold everything,
            # there are no bytes left to save, so the approximation should
            # switch itself off and the output should return to exact.
            es_config.KEEP_FREE = True
            model_q2, _tok = load(
                str(tmp), mode="streamed", cache_gb=1.0, verbose=False
            )
            greedy_generate(model_q2, prompt, n_new)  # warm the cache
            got4b = greedy_generate(model_q2, prompt, n_new)
            stats4b = get_stats(model_q2)
            print(f"keep-free: pruned_frac={stats4b['cache']['pruned_frac']}, "
                  f"rescued={stats4b['cache']['kept_free']}")
            assert got4b == ref, (
                "prune 0.9 with a fully resident cache should cost nothing, "
                f"but output moved\n  ref={ref}\n  got={got4b}"
            )
            del model_q2
        finally:
            es_config.PRUNE = 0.0
            es_config.KEEP_FREE = True

        print("== streamed top-k cap == resident at num_experts_per_tok=cap ==")
        # With renorm on, TOP_K=c is not an approximation of anything: it is
        # exactly the same model run with num_experts_per_tok=c.
        #
        #   parent (K slots):  sum_i softmax_all(top_K)_i * y_i / sum(top_K)
        #   cap keeps top c:   same sum restricted to the c strongest
        #   renorm divides by  sum(softmax_all(top_c)) / sum(softmax_all(top_K))
        #   =>                 sum_i softmax_all(top_c)_i * y_i / sum(top_c)
        #
        # which is the definition of a norm_topk_prob top-c mixture. So the
        # reference here is a *resident* load of the same weights with the
        # config's top-k lowered, and the two must agree token for token.
        # The reference caps *decode only*, exactly as the engine does -
        # prefill computes the full mixture so prompt states stay exact.
        # (Lowering num_experts_per_tok in config.json instead would also cap
        # prefill, and on a 50-token prompt that dominates the divergence.)
        cap = 2  # of CONFIG["num_experts_per_tok"] == 4
        orig_call = qwen3_moe.Qwen3MoeSparseMoeBlock.__call__

        def decode_capped_call(self, x):
            gates = mx.softmax(self.gate(x), axis=-1, precise=True)
            k = cap if x.reshape(-1, x.shape[-1]).shape[0] == 1 else self.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if self.norm_topk_prob:
                scores = scores / mx.sum(scores, axis=-1, keepdims=True)
            y = self.switch_mlp(x, inds)
            return (y * scores[..., None]).sum(axis=-2)

        # This tiny model's argmax is degenerate (one token dominates), so the
        # comparison is on logits, not tokens: that measures the mixture math
        # the cap changes rather than whatever survives an argmax.
        model_u, _tok = load(str(tmp), mode="resident", verbose=False)
        _, logits_full = greedy_generate(model_u, prompt, n_new, want_logits=True)
        del model_u

        qwen3_moe.Qwen3MoeSparseMoeBlock.__call__ = decode_capped_call
        try:
            model_c, _tok = load(str(tmp), mode="resident", verbose=False)
            ref_cap, logits_cap = greedy_generate(
                model_c, prompt, n_new, want_logits=True
            )
            del model_c
        finally:
            qwen3_moe.Qwen3MoeSparseMoeBlock.__call__ = orig_call

        spread = float(np.abs(logits_full).max())
        cap_effect = float(np.abs(logits_cap - logits_full).max())
        assert cap_effect > 1e-3 * spread, (
            f"capping decode to top-{cap} barely moved the logits "
            f"({cap_effect:.4g} vs scale {spread:.4g}); the test cannot "
            "distinguish a working cap from a no-op"
        )

        es_config.ROUTE_TOP_K = cap
        es_config.ROUTE_RENORM = True
        es_config.KEEP_FREE = False  # residency rescue would defeat the cap here
        try:
            model_k, _tok = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=True)
            got9, logits_got = greedy_generate(
                model_k, prompt, n_new, want_logits=True
            )
            stats9 = get_stats(model_k)
            print("cache stats:", json.dumps(stats9["cache"]))
            err = float(np.abs(logits_got - logits_cap).max())
            print(f"cap logits: |Δ vs top-{cap} reference|={err:.3g}, "
                  f"cap effect={cap_effect:.3g}, scale={spread:.3g}")
            assert err < 0.01 * cap_effect, (
                f"capped streamed != resident top-{cap}: max logit delta "
                f"{err:.4g}, which is not negligible next to the cap's own "
                f"effect ({cap_effect:.4g})"
            )
            assert got9 == ref_cap, (
                f"capped streamed tokens != resident top-{cap}"
                f"\n  ref={ref_cap}\n  got={got9}"
            )
            assert stats9["cache"]["pruned_frac"] > 0, "cap dropped no slots"
            del model_k

            # Renorm is load-bearing, not cosmetic: without it the block
            # returns a mixture whose mass is short by whatever the dropped
            # experts carried, so it must NOT match the top-c reference.
            es_config.ROUTE_RENORM = False
            model_n, _tok = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
            _, logits_noren = greedy_generate(
                model_n, prompt, n_new, want_logits=True
            )
            noren = float(np.abs(logits_noren - logits_cap).max())
            assert noren > 10 * err, (
                "cap without renorm matched the renormalized reference "
                f"({noren:.4g} vs {err:.4g}); renorm is not being applied"
            )
            print(f"renorm off: |Δ vs reference|={noren:.3g} (expected large)")
            del model_n

            # KEEP_FREE puts cache-resident experts back after the cap dropped
            # them, so with a roomy cache the mixture must move back *toward*
            # the full one (and away from the strict top-c reference).
            es_config.ROUTE_RENORM = True
            es_config.KEEP_FREE = True
            model_f, _tok = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=False)
            _, logits_free = greedy_generate(
                model_f, prompt, n_new, want_logits=True
            )
            stats_f = get_stats(model_f)
            assert stats_f["cache"]["kept_free"] > 0, "no experts were rescued"
            to_full = float(np.abs(logits_free - logits_full).max())
            assert to_full < cap_effect, (
                "residency rescue did not move the mixture toward the full "
                f"one ({to_full:.4g} vs the cap's {cap_effect:.4g})"
            )
            print(f"keep-free: rescued {stats_f['cache']['kept_free']} slots, "
                  f"|Δ vs full|={to_full:.3g} (cap alone: {cap_effect:.3g})")
            del model_f
        finally:
            es_config.ROUTE_TOP_K = 0
            es_config.ROUTE_RENORM = False
            es_config.KEEP_FREE = True

        print("== streamed + nucleus routing (TOP_P) ==")
        # p at 1.0 needs every expert to reach the mass target, so the mixture
        # is the full one and output must not move a bit.
        es_config.ROUTE_TOP_P = 1.0 - 1e-9
        es_config.KEEP_FREE = False  # isolate the selector from cache state
        try:
            model_p1, _tok = load(
                str(tmp), mode="streamed", cache_gb=1.0, verbose=False
            )
            got11 = greedy_generate(model_p1, prompt, n_new)
            assert got11 == ref, f"top_p~1 changed output\n  ref={ref}\n  got={got11}"
            del model_p1

            # A real threshold must drop mass on some tokens but never empty a
            # row, and must land between "full" and "one expert" in effect.
            es_config.ROUTE_TOP_P = 0.6
            model_p2, _tok = load(
                str(tmp), mode="streamed", cache_gb=1.0, verbose=True
            )
            _, logits_p = greedy_generate(
                model_p2, prompt, n_new, want_logits=True
            )
            stats_p = get_stats(model_p2)
            print("cache stats:", json.dumps(stats_p["cache"]))
            assert stats_p["cache"]["pruned_frac"] > 0, "top_p dropped no slots"
            assert np.isfinite(logits_p).all(), "top_p produced non-finite logits"
            del model_p2
        finally:
            es_config.ROUTE_TOP_P = 0.0
            es_config.KEEP_FREE = True

        # The nucleus mask itself, on cases that are easy to get wrong.
        w = np.array([
            [0.7, 0.2, 0.05, 0.05],   # peaked: 0.7 alone clears p=0.6
            [0.25, 0.25, 0.25, 0.25],  # flat: needs 3 of 4 for p=0.6
            [1.0, 0.0, 0.0, 0.0],      # degenerate
        ])
        m = _nucleus_mask(w, 0.6)
        assert m[0].tolist() == [True, False, False, False], m[0]
        assert m[1].sum() == 3, m[1]
        assert m[2].tolist() == [True, False, False, False], m[2]
        # Never empty, and never keeps more than it must.
        assert (_nucleus_mask(w, 1e-9).sum(axis=-1) == 1).all()
        assert (_nucleus_mask(w, 1.0).sum(axis=-1) >= 1).all()
        print("OK: nucleus mask unit cases")

        print("== streamed + wait gate (WAIT_ABOVE) ==")
        # The gate only skips experts that would *block on a read*, so on a warm
        # cache there is nothing to skip and output must stay exact even at the
        # most aggressive setting. PRUNE at epsilon keeps every slot; it is only
        # there because the gate rides the same mask.
        es_config.PRUNE = 1e-9
        es_config.WAIT_ABOVE = 1.01
        try:
            model_w, _tok = load(
                str(tmp), mode="streamed", cache_gb=1.0, verbose=False
            )
            greedy_generate(model_w, prompt, n_new)  # warm every expert in
            got12 = greedy_generate(model_w, prompt, n_new)
            stats_w = get_stats(model_w)
            assert stats_w["cache"]["skipped_waits"] == 0, (
                "skipped a read on a fully warm cache: "
                f"{stats_w['cache']['skipped_waits']}"
            )
            assert got12 == ref, f"wait gate moved warm output\n  got={got12}"
            del model_w

            # Starved cache: now misses are unavoidable, so the gate must
            # actually fire and the model must still produce tokens.
            model_w2, _tok = load(
                str(tmp), mode="streamed", cache_gb=0.00015, verbose=True
            )
            got13 = greedy_generate(model_w2, prompt, n_new)
            stats_w2 = get_stats(model_w2)
            print("cache stats:", json.dumps(stats_w2["cache"]))
            assert len(got13) == n_new
            assert stats_w2["cache"]["skipped_waits"] > 0, "gate never fired"
            print(f"wait gate: skipped {stats_w2['cache']['skipped_waits']} reads "
                  f"on a starved cache")
            del model_w2
        finally:
            es_config.PRUNE = 0.0
            es_config.WAIT_ABOVE = 0.0

        print("== streamed + slab (gather_qmm over slots) ==")
        # Slabs change both how experts are read (pread straight into
        # Metal-backed slot memory) and how they are computed (one gather_qmm
        # per projection instead of three calls per expert). Output must not
        # move a bit.
        es_config.SLAB = "on"
        try:
            model_z, _tok = load(
                str(tmp), mode="streamed", cache_gb=0.01, verbose=True
            )
            got6 = greedy_generate(model_z, prompt, n_new)
            stats6 = get_stats(model_z)
            print("cache stats:", json.dumps(stats6["cache"]))
            assert stats6["cache"]["slab_slots"] > 0, "slab did not engage"
            assert got6 == ref, f"slab != resident\n  ref={ref}\n  got={got6}"

            # Second turn on the same model: prefill now takes cache *hits*
            # that the first turn's decode installed, and those are slot-backed
            # entries carrying a slot number rather than arrays. This is every
            # turn after the first in a real session, and it crashed the server
            # with KeyError: 'up_proj.weight' until _run learned to convert.
            got6b = greedy_generate(model_z, prompt, n_new)
            assert got6b == ref, f"slab 2nd turn != resident\n  got={got6b}"
            del model_z

            print("== streamed + slab, starved (fewer slots than experts) ==")
            # Slots are reused constantly here, which is the dangerous case:
            # reuse overwrites bytes in place, so if a slot were ever rewritten
            # while a pending graph still read it, tokens would diverge.
            model_y, _tok = load(
                str(tmp), mode="streamed", cache_gb=0.0007, verbose=True
            )
            got7 = greedy_generate(model_y, prompt, n_new)
            stats7 = get_stats(model_y)
            print("cache stats:", json.dumps(stats7["cache"]))
            assert stats7["cache"]["slab_slots"] > 0, "slab did not engage"
            assert stats7["cache"]["slot_starved"] == 0, "ran out of slots"
            assert got7 == ref, f"starved slab != resident\n  ref={ref}\n  got={got7}"
            del model_y

            print("== streamed + slab, prefill releases and rebuilds slab ==")
            # A big prefill hands the slab's memory to activations, and decode
            # rebuilds it afterwards (see ExpertCache.enter_prefill). Lower the
            # threshold so this 50-token prompt takes that path.
            shrink = es_config.PREFILL_SHRINK_TOKENS
            es_config.PREFILL_SHRINK_TOKENS = 40
            try:
                model_w, _tok = load(
                    str(tmp), mode="streamed", cache_gb=0.01, verbose=True
                )
                got8 = greedy_generate(model_w, prompt, n_new)
                stats8 = get_stats(model_w)
                print("cache stats:", json.dumps(stats8["cache"]))
                # Rebuilt for decode, so the slab is live again at the end.
                assert stats8["cache"]["slab_slots"] > 0, "slab was not rebuilt"
                assert got8 == ref, f"slab rebuild != resident\n  got={got8}"
                del model_w
            finally:
                es_config.PREFILL_SHRINK_TOKENS = shrink
        finally:
            es_config.SLAB = "auto"

        print("== streamed + speculative decoding (self-draft, greedy) ==")
        # The model drafts for itself: acceptance is ~100%, and greedy
        # speculative output must be IDENTICAL to plain greedy decoding.
        # Exercises the batched verify path (num_draft_tokens+1 positions
        # through the decode path - see _DECODE_MAX_TOKENS) plus KV-cache
        # trims on rejection.
        from mlx_lm.generate import speculative_generate_step

        model_v, _tok = load(str(tmp), mode="streamed", cache_gb=1.0, verbose=True)
        draft, _tok2 = load(str(tmp), mode="resident", verbose=True)
        spec_cache = make_prompt_cache(model_v) + make_prompt_cache(draft)
        got5 = [
            tok
            for tok, _lp, _fd in speculative_generate_step(
                mx.array(prompt, mx.uint32),
                model_v,
                draft,
                num_draft_tokens=4,
                max_tokens=n_new,
                prompt_cache=spec_cache,
            )
        ]
        assert got5 == ref, f"speculative != resident\n  ref={ref}\n  got={got5}"
        del model_v, draft

        print(f"\nOK: all modes emitted identical tokens: {ref[:10]}...")


if __name__ == "__main__":
    main()
