# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-head sidecar smoke tests: features, heads, façade, disabled path."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from expert_stream import config
from expert_stream.sidecar import (
    ExpertSidecar,
    ModelSlot,
    build_features,
    demand_to_features,
    model_slot_id,
    same_layer_ceiling,
)


def test_model_slot_id_stable_and_geometry_sensitive():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "config.json").write_text(
            '{"model_type":"toy","num_hidden_layers":4,"num_experts":8}'
        )
        (root / "model.safetensors.index.json").write_text(
            '{"metadata":{"total_size":1},"weight_map":{"a":"model.safetensors"}}'
        )
        shard = root / "model.safetensors"
        shard.write_bytes(b"x" * 4096)
        a = model_slot_id(str(root), 92, 160)
        b = model_slot_id(str(root), 92, 160)
        c = model_slot_id(str(root), 92, 128)
        assert a == b and a != c


def test_model_slot_id_path_independent():
    """Same checkpoint bytes in two directories share a slot id."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        def write_ckpt(folder: Path, payload: bytes):
            folder.mkdir(parents=True)
            (folder / "config.json").write_text(
                '{"model_type":"toy","_name_or_path":"/old/path","num_experts":8}'
            )
            (folder / "model.safetensors.index.json").write_text(
                '{"metadata":{"total_size":%d},"weight_map":{"w":"model.safetensors"}}'
                % len(payload)
            )
            (folder / "model.safetensors").write_bytes(payload)

        a_dir, b_dir = td / "a", td / "b"
        write_ckpt(a_dir, b"weight-bytes-aaaa" * 200)
        write_ckpt(b_dir, b"weight-bytes-aaaa" * 200)
        # Volatile _name_or_path differs if we rewrite - already stripped.
        (b_dir / "config.json").write_text(
            '{"model_type":"toy","_name_or_path":"/other","num_experts":8}'
        )
        assert model_slot_id(str(a_dir), 4, 8) == model_slot_id(str(b_dir), 4, 8)

        write_ckpt(td / "c", b"weight-bytes-BBBB" * 200)
        assert model_slot_id(str(a_dir), 4, 8) != model_slot_id(str(td / "c"), 4, 8)


def test_migrate_legacy_slot_files():
    from expert_stream.sidecar.features import (
        legacy_path_slot_id,
        migrate_legacy_slot_files,
        model_slot_id,
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        model = td / "model"
        model.mkdir()
        (model / "config.json").write_text('{"model_type":"toy"}')
        (model / "model.safetensors").write_bytes(b"z" * 8192)
        slot_root = td / "sidecar"
        slot_root.mkdir()
        old = legacy_path_slot_id(str(model), 4, 8)
        new = model_slot_id(str(model), 4, 8)
        assert old != new
        (slot_root / f"{old}_wrap.npz").write_bytes(b"wrap")
        (slot_root / f"{old}_prune.npz").write_bytes(b"prune")
        active, moved = migrate_legacy_slot_files(
            slot_root, str(model), 4, 8, new_id=new
        )
        assert active == new
        assert sorted(moved) == sorted([f"{old}_wrap.npz", f"{old}_prune.npz"])
        assert (slot_root / f"{new}_wrap.npz").read_bytes() == b"wrap"
        assert not (slot_root / f"{old}_wrap.npz").exists()
        # Second pass is a no-op (already migrated).
        _, moved2 = migrate_legacy_slot_files(slot_root, str(model), 4, 8, new_id=new)
        assert moved2 == []


def test_demand_features_normed_and_deterministic():
    d = {0: [1, 2, 3], 1: [4]}
    x1 = demand_to_features(d, 64, 4)
    x2 = demand_to_features(d, 64, 4)
    assert np.allclose(x1, x2)
    assert abs(float(np.linalg.norm(x1)) - 1.0) < 1e-5


def test_history_features_change_with_age():
    d0 = {0: [1, 2], 1: [3]}
    d1 = {0: [7, 8], 1: [9]}
    one = build_features([d1], 128, 4, wrap=2)
    two = build_features([d0, d1], 128, 4, wrap=2)
    assert not np.allclose(one, two)


def test_same_layer_ceiling():
    prev = {0: [1, 2, 3], 1: [4, 5]}
    cur = {0: [1, 9], 1: [4, 5, 6]}
    assert abs(same_layer_ceiling(prev, cur, 2) - 0.6) < 1e-6


def test_slot_learns_a_fixed_pattern():
    rng = np.random.RandomState(0)
    feat_dim, n_experts, wrap = 64, 32, 2
    slot = ModelSlot.create("t", n_layers=4, n_experts=n_experts, wrap=wrap, feat_dim=feat_dim)
    features = rng.randn(feat_dim).astype(np.float32)
    features /= np.linalg.norm(features)
    demand = {0: [1, 5, 9], 1: [2, 6, 10]}

    def recall_of(weights):
        saved = slot.live
        slot.live = weights
        try:
            pred = slot.predict_topk(features, 3)
        finally:
            slot.live = saved
        hit = need = 0
        for i, want in demand.items():
            need += len(want)
            hit += len(set(want).intersection(pred[i]))
        return hit / need

    before = recall_of(slot.shadow)
    for _ in range(120):
        slot.train_shadow(features, demand, lr=0.3)
    after = recall_of(slot.shadow)
    assert after > before + 0.3, (before, after)


def test_promote_requires_precision():
    slot = ModelSlot.create("t", 4, 16, 2, 32)
    slot.shadow_recalls.extend([0.5] * 20)
    slot.shadow_precisions.extend([0.05] * 20)
    assert slot.promote_if_better(0.35, 0.22) is False
    slot.shadow_precisions.clear()
    slot.shadow_precisions.extend([0.3] * 20)
    assert slot.promote_if_better(0.35, 0.22) is True
    assert slot.prefetch_enabled


def test_promote_and_lock():
    slot = ModelSlot.create("t", 4, 16, 2, 32)
    slot.prefetch_enabled = True
    slot.tokens_seen = 5000
    slot.recalls.extend([0.7] * 20)
    slot.precisions.extend([0.4] * 20)
    slot.ceilings.extend([0.3] * 20)
    assert slot.maybe_lock(0.35, 0.22, 0) is False
    assert slot.maybe_lock(0.35, 0.22, 3) is False
    assert slot.maybe_lock(0.35, 0.22, 3) is False
    assert slot.maybe_lock(0.35, 0.22, 3) is True
    assert slot.locked


def test_lock_refuses_near_ceiling():
    slot = ModelSlot.create("t", 4, 16, 2, 32)
    slot.prefetch_enabled = True
    slot.tokens_seen = 5000
    slot.recalls.extend([0.30] * 20)
    slot.precisions.extend([0.30] * 20)
    slot.ceilings.extend([0.28] * 20)
    for _ in range(10):
        assert slot.maybe_lock(0.25, 0.22, 3) is False
    assert not slot.locked


def test_disabled_config_default():
    assert config.SIDECAR is False or bool(
        __import__("os").environ.get("EXPERT_STREAM_SIDECAR")
    )


class _FakeCache:
    def __init__(self):
        self.batches = []

    def prefetch_many(self, batch):
        self.batches.append(batch)


def test_sidecar_multi_head_constructs():
    saved = (
        config.SIDECAR_HEAD_WRAP,
        config.SIDECAR_HEAD_PREFILL,
        config.SIDECAR_HEAD_RESIDENCY,
        config.SIDECAR_HEAD_PRUNE,
        config.SIDECAR_LOG_EVERY,
    )
    config.SIDECAR_HEAD_WRAP = True
    config.SIDECAR_HEAD_PREFILL = True
    config.SIDECAR_HEAD_RESIDENCY = True
    config.SIDECAR_HEAD_PRUNE = True
    config.SIDECAR_LOG_EVERY = 1000
    try:
        with tempfile.TemporaryDirectory() as td:
            config.SIDECAR_DIR = td
            keys = [f"layer.{i}.mlp" for i in range(8)]
            sc = ExpertSidecar("/tmp/fake-multi", keys, n_experts=16)
            assert sc.wrap.enabled and sc.prefill.enabled
            assert sc.residency.enabled and sc.prune.enabled
            cache = _FakeCache()
            sc.begin_prefill_chunk(cache)
            for i, key in enumerate(keys):
                sc.note_prefill_demand(key, [i % 16, (i + 1) % 16])
            sc.end_prefill_chunk()
            assert sc.prefill.slot.tokens_seen >= 1
            # Regression: drop_token mid-chunk must not be the normal prefill
            # path (streaming used to call it on every MoE layer).
            sc.begin_prefill_chunk(cache)
            sc.note_prefill_demand(keys[0], [0, 1])
            sc.drop_token()
            sc.note_prefill_demand(keys[1], [2, 3])
            sc.end_prefill_chunk()
            # aborted chunk does not train again
            assert sc.prefill.slot.tokens_seen == 1
            for t in range(20):
                for i, key in enumerate(keys):
                    base = (i * 3) % 16
                    sc.note_demand(key, [base, (base + 1) % 16])
                sc.end_token(cache)
            st = sc.stats()
            assert "wrap" in st and "prefill" in st
            assert "residency" in st and "prune" in st
            sc.close()
    finally:
        (
            config.SIDECAR_HEAD_WRAP,
            config.SIDECAR_HEAD_PREFILL,
            config.SIDECAR_HEAD_RESIDENCY,
            config.SIDECAR_HEAD_PRUNE,
            config.SIDECAR_LOG_EVERY,
        ) = saved


def test_feature_layout_is_deterministic_and_blocked():
    from expert_stream.sidecar.features import EXTRA_DIM, FeatureSpec

    spec = FeatureSpec.build(512, 89, 6, 160)
    assert spec.hist_dim + spec.stripe_dim + spec.sketch_dim + EXTRA_DIM == 512
    again = FeatureSpec.build(512, 89, 6, 160)
    assert np.array_equal(spec.table, again.table)
    hist = [{0: [1, 2], 1: [3]}]
    a = spec.build_vector(hist)
    b = spec.build_vector(hist)
    assert np.allclose(a, b)
    # The sketch block only fills when a hidden state is supplied, and it must
    # not disturb the demand blocks.
    cut = spec.hist_dim + spec.stripe_dim
    sk = np.full(spec.sketch_dim, 0.5, dtype=np.float32)
    c = spec.build_vector(hist, hidden_sketch=sk)
    assert np.allclose(a[:cut], c[:cut])
    assert np.allclose(c[cut : cut + spec.sketch_dim], 0.5)


def test_hash_demand_handles_ragged_layers():
    """Decode pruning drops a different count per layer, so demand is ragged."""
    from expert_stream.sidecar.features import FeatureSpec

    spec = FeatureSpec.build(256, 6, 3, 32)
    ragged = {0: [1, 2, 3], 1: [4], 2: [5, 6], 3: [], 4: [7, 8, 9, 10]}
    idx = spec.hash_demand(ragged)
    assert idx.size == 10
    assert idx.min() >= 0 and idx.max() < spec.hist_dim
    counts = spec.demand_counts(ragged)
    assert abs(float(counts.sum()) - 10.0) < 1e-5
    # Out-of-range layers and experts are dropped or clamped, never raised.
    assert spec.hash_demand({99: [1], -1: [2]}).size == 0
    assert spec.hash_demand({0: [10**6]}).size == 1


def test_incremental_history_matches_windowed_build():
    from expert_stream.sidecar.features import HIST_DECAY, FeatureSpec

    spec = FeatureSpec.build(256, 6, 3, 32)
    tokens = [
        {0: [1, 2], 1: [3]},
        {0: [2], 1: [4, 5]},
        {0: [7], 1: [8], 2: [9]},
    ]
    acc = np.zeros(spec.hist_dim, dtype=np.float32)
    for tok in tokens:
        acc *= np.float32(HIST_DECAY)
        acc += spec.demand_counts(tok)
    incremental = spec.compose(acc, tokens[-1])
    windowed = spec.build_vector(tokens)
    assert np.allclose(incremental, windowed, atol=1e-5)


def test_hidden_sketch_is_stable_across_restarts():
    from expert_stream.sidecar.features import project_hidden, sketch_matrix

    p1 = sketch_matrix("slot-abc", 128, 16)
    p2 = sketch_matrix("slot-abc", 128, 16)
    assert np.array_equal(p1, p2)
    assert not np.array_equal(p1, sketch_matrix("slot-xyz", 128, 16))
    h = np.random.RandomState(1).randn(128).astype(np.float32)
    s = project_hidden(h, p1)
    assert s.shape == (16,) and np.all(np.abs(s) <= 1.0)


def test_hidden_state_makes_a_learnable_pattern_learnable():
    """History alone cannot separate two demands that share the same history."""
    from expert_stream.sidecar.features import FeatureSpec

    spec = FeatureSpec.build(256, 4, 2, 32)
    slot = ModelSlot.create("h", 4, 32, 2, 256)
    hist = [{0: [1, 2], 1: [3, 4]}]
    a = np.zeros(spec.sketch_dim, dtype=np.float32)
    a[: spec.sketch_dim // 2] = 1.0
    b = np.zeros(spec.sketch_dim, dtype=np.float32)
    b[spec.sketch_dim // 2 :] = 1.0
    fa = spec.build_vector(hist, hidden_sketch=a)
    fb = spec.build_vector(hist, hidden_sketch=b)
    da, db = {0: [5], 1: [6]}, {0: [20], 1: [21]}
    for _ in range(150):
        slot.train_shadow(fa, da, lr=0.3)
        slot.train_shadow(fb, db, lr=0.3)
    pa = slot.predict_topk(fa, 2, use_shadow=True)
    pb = slot.predict_topk(fb, 2, use_shadow=True)
    assert 5 in pa[0] and 20 in pb[0], (pa, pb)


def test_margin_scoring_credits_only_what_baseline_missed():
    slot = ModelSlot.create("m", 4, 32, 2, 64)
    baseline = {0: [1, 2], 1: [3]}
    demand = {0: [1, 9], 1: [3, 7]}
    predicted = {0: [1, 9], 1: [3, 7]}
    gain, waste = slot.score_margin(predicted, baseline, demand)
    # extras are {9} and {7}; both wanted -> 2 of 4 needed, nothing wasted.
    assert abs(gain - 0.5) < 1e-6 and abs(waste) < 1e-6
    # Re-deriving the baseline exactly earns nothing.
    gain2, _ = slot.score_margin(baseline, baseline, demand)
    assert gain2 == 0.0


def test_promote_gates_on_marginal_gain():
    slot = ModelSlot.create("g", 4, 16, 2, 32)
    slot.shadow_recalls.extend([0.9] * 20)
    slot.shadow_precisions.extend([0.5] * 20)
    # High recall, but every hit was already covered by the baseline.
    slot.shadow_gains.extend([0.0] * 20)
    assert slot.promote_if_better(0.35, 0.22, min_gain=0.02) is False
    assert not slot.prefetch_enabled
    slot.shadow_gains.clear()
    slot.shadow_gains.extend([0.08] * 20)
    assert slot.promote_if_better(0.35, 0.22, min_gain=0.02) is True
    assert slot.prefetch_enabled


def test_slot_roundtrips_weights_and_earned_state():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "rt.npz"
        a = ModelSlot.create("rt", 4, 16, 2, 32, path=path)
        feats = np.random.RandomState(2).randn(32).astype(np.float32)
        for _ in range(30):
            a.train_shadow(feats, {0: [1, 2], 1: [3]}, lr=0.2)
        a.prefetch_enabled = True
        a.recalls.extend([0.44] * 8)
        a.save()
        b = ModelSlot.create("rt", 4, 16, 2, 32, path=path)
        assert np.allclose(a.shadow, b.shadow)
        assert np.allclose(a.freq, b.freq)
        assert b.train_steps == a.train_steps
        # A head that had earned actuation keeps it across a restart.
        assert b.prefetch_enabled
        assert abs(b.mean_recall() - 0.44) < 1e-5


def test_residency_reuse_is_measured_in_tokens_not_layer_calls():
    """Regression: horizon shorter than the layer count froze every score.

    note_demand used to append one reuse window per MoE layer, so on a model
    with more layers than the horizon a key's own next appearance always fell
    outside the window and no score ever left its initial value.
    """
    from expert_stream.sidecar.heads.residency import ResidencyHead

    with tempfile.TemporaryDirectory() as td:
        saved = config.SIDECAR_HEAD_RESIDENCY, config.SIDECAR_RESIDENCY_HORIZON
        config.SIDECAR_HEAD_RESIDENCY = True
        config.SIDECAR_RESIDENCY_HORIZON = 8
        try:
            h = ResidencyHead("res", 16, Path(td))
            n_layers = 40  # deliberately more layers than the horizon
            key_to_idx = {f"l{i}": i for i in range(n_layers)}
            for _ in range(60):
                for i in range(n_layers):
                    h.note_demand(i, [i % 16])
                h.end_token()
            assert h.protect_score("l0", 0, key_to_idx) > 0.5
            assert h.should_protect("l0", 0, key_to_idx)
            # An expert nothing ever routed to is not protected.
            assert not h.should_protect("l0", 15, key_to_idx)
        finally:
            config.SIDECAR_HEAD_RESIDENCY, config.SIDECAR_RESIDENCY_HORIZON = saved


def test_residency_protect_is_rate_limited():
    from expert_stream.sidecar.heads.residency import ResidencyHead

    with tempfile.TemporaryDirectory() as td:
        saved = config.SIDECAR_HEAD_RESIDENCY
        config.SIDECAR_HEAD_RESIDENCY = True
        try:
            h = ResidencyHead("res2", 16, Path(td))
            key_to_idx = {f"l{i}": i for i in range(4)}
            for _ in range(80):
                for i in range(4):
                    h.note_demand(i, list(range(16)))
                h.end_token()
            protected = 0
            for _ in range(400):
                h.on_scan()
                if h.should_protect("l0", 0, key_to_idx):
                    h.on_protect()
                    protected += 1
            assert 0 < protected < 400 * config.SIDECAR_RESIDENCY_MAX_PROTECT + 70
        finally:
            config.SIDECAR_HEAD_RESIDENCY = saved


def test_prefill_targets_hot_set_not_union():
    from expert_stream.sidecar.features import hot_set

    # 100 tokens touched expert 0 constantly and 5..9 once each: only 0 is hot.
    counts = {0: {0: 100, 1: 60, 5: 1, 6: 1, 7: 1}}
    hot = hot_set(counts, 0.5)
    assert set(hot[0]) == {0, 1}
    # Degenerate layer still yields its single best expert.
    assert hot_set({1: {3: 1}}, 0.5)[1] == [3]


def test_expand_trajectory_mixes_token_positions():
    from expert_stream.sidecar.bank import expand_trajectory
    from expert_stream.sidecar.features import FeatureSpec

    spec = FeatureSpec.build(128, 4, 2, 16)
    # 5 tokens -> many windows when multi_scale is on.
    traj = [
        {0: [1, 2], 1: [3]},
        {0: [2, 4], 1: [5]},
        {0: [7], 1: [8, 9]},
        {0: [1], 1: [3]},
        {0: [4, 5], 1: [6]},
    ]
    samples = expand_trajectory(traj, spec, history_len=3, multi_scale=True)
    assert len(samples) > len(traj) - 1  # multi-scale densifies
    # Single-scale: exactly one sample per target token after the first.
    plain = expand_trajectory(traj, spec, history_len=3, multi_scale=False)
    assert len(plain) == 4
    feats, demand = plain[0]
    assert feats.shape == (128,)
    assert 0 in demand and 1 in demand


def test_bank_roundtrip_and_fit_from_bank():
    from expert_stream.sidecar import ExpertSidecar
    from expert_stream.sidecar.bank import bank_path, expand_trajectories, load_bank, save_bank
    from expert_stream.sidecar.features import FeatureSpec

    with tempfile.TemporaryDirectory() as td:
        config.SIDECAR_DIR = td
        config.SIDECAR_HEAD_WRAP = True
        traj = [
            [{0: [1, 2], 1: [3]}, {0: [2], 1: [4]}, {0: [5, 6], 1: [7]}]
            for _ in range(3)
        ]
        meta = {
            "slot": "banktest",
            "n_layers": 4,
            "n_experts": 16,
            "feat_dim": 192,
            "wrap": 2,
        }
        path = bank_path(Path(td), "banktest")
        save_bank(path, traj, meta=meta)
        # from_bank must not double-count when fit_pretrain(from_bank=True).
        loaded, _ = load_bank(path)
        spec = FeatureSpec.build(192, 4, 2, 16)
        expected = len(
            expand_trajectories(loaded, spec, history_len=6, multi_scale=True)
        )
        sc = ExpertSidecar.from_bank(path)
        report = sc.fit_pretrain(epochs=2, from_bank=True)
        assert report["trajectories"] == 3, report
        assert report["samples"] == expected, (report, expected)
        assert report["holdout_recall"] >= 0.0
        sc.close()


def test_live_bank_note_does_not_touch_disk_inline():
    """Hot-path bank append must not flush (decode must stay exception-safe)."""
    with tempfile.TemporaryDirectory() as td:
        config.SIDECAR_DIR = td
        config.SIDECAR_HEAD_WRAP = True
        config.SIDECAR_HEAD_PREFILL = False
        config.SIDECAR_HEAD_RESIDENCY = False
        config.SIDECAR_HEAD_PRUNE = False
        keys = [f"l{i}" for i in range(4)]
        sc = ExpertSidecar("/tmp/bank-hot", keys, n_experts=16)
        for t in range(80):
            sc._bank_note({0: [t % 16], 1: [(t + 1) % 16]})
        from expert_stream.sidecar.bank import bank_path

        assert not bank_path(Path(td), sc.slot_id).exists()
        sc._flush_bank_traj()
        assert bank_path(Path(td), sc.slot_id).exists()
        sc.close()


def test_defaults_are_tight():
    assert config.SIDECAR_TOPK <= 4
    assert config.SIDECAR_MIN_PRECISION >= 0.15
    assert config.SIDECAR_LOCK_HITS == 0
    assert config.SIDECAR_HEAD_WRAP is True or config.SIDECAR_HEAD_WRAP == 1


def test_prune_head_propose_floors_at_base():
    from expert_stream.sidecar.heads.prune import PrunePolicyHead

    with tempfile.TemporaryDirectory() as td:
        saved = config.SIDECAR_HEAD_PRUNE, config.PRUNE, config.WAIT_ABOVE
        config.SIDECAR_HEAD_PRUNE = True
        config.PRUNE = 0.8
        config.WAIT_ABOVE = 0.2
        try:
            h = PrunePolicyHead("x", Path(td))
            wnp = np.array([[0.9, 0.05, 0.03, 0.02]], dtype=np.float32)
            t, w = h.propose(wnp)  # cold - should return base
            assert t == 0.8 and w == 0.2
            for _ in range(40):
                h.observe(wnp, 0.8)
            t2, _ = h.propose(wnp)
            assert t2 >= 0.8
        finally:
            config.SIDECAR_HEAD_PRUNE, config.PRUNE, config.WAIT_ABOVE = saved


def test_prune_mass_gate_is_relative_to_the_shipped_threshold():
    """PRUNE=0.7 itself keeps <half the mass, so an absolute bar vetoes forever."""
    from expert_stream.sidecar.heads.prune import PrunePolicyHead

    with tempfile.TemporaryDirectory() as td:
        saved = config.SIDECAR_HEAD_PRUNE, config.PRUNE, config.WAIT_ABOVE
        config.SIDECAR_HEAD_PRUNE = True
        config.PRUNE = 0.7
        config.WAIT_ABOVE = 0.2
        try:
            h = PrunePolicyHead("mass", Path(td))
            # Mass spread the way a real top-8 router spreads it: the 0.7
            # baseline keeps well under half of it.
            wnp = np.array([[0.40, 0.22, 0.14, 0.09, 0.06, 0.04, 0.03, 0.02]], np.float32)
            for _ in range(40):
                h.observe(wnp, 0.7)
            assert h.mean_base_mass() < 0.6, h.mean_base_mass()
            # Relative to that baseline the head is not losing mass, so it may
            # go live; against an absolute 0.85 bar it never could.
            assert h.mass_ratio() >= 0.98, (h.mean_mass(), h.mean_base_mass())
        finally:
            config.SIDECAR_HEAD_PRUNE, config.PRUNE, config.WAIT_ABOVE = saved


def test_topk_backs_off_when_extra_reads_are_wasted():
    from expert_stream.sidecar.heads.wrap import WrapHead

    with tempfile.TemporaryDirectory() as td:
        saved = config.SIDECAR_HEAD_WRAP, config.SIDECAR_TOPK
        config.SIDECAR_HEAD_WRAP = True
        config.SIDECAR_TOPK = 0  # auto
        try:
            h = WrapHead("tk", 8, 32, [f"l{i}" for i in range(8)], Path(td))
            assert h.auto_topk
            h.slot.demand_sizes.extend([6] * 32)
            h.slot.wastes.extend([0.2] * 32)
            h._retune_topk()
            paying = h.topk
            h.slot.wastes.clear()
            h.slot.wastes.extend([0.99] * 32)
            for _ in range(3):
                h._retune_topk()
            assert h.topk < paying, (paying, h.topk)
            assert h.topk >= 1
        finally:
            config.SIDECAR_HEAD_WRAP, config.SIDECAR_TOPK = saved


def test_disabled_sidecar_does_no_work_on_the_decode_path():
    """SIDECAR=0 must leave the engine exactly as it was before any of this."""
    from expert_stream.streaming import PrefetchRing

    class _Cache:
        def __init__(self):
            self._residency_advisor = "sentinel"

    cache = _Cache()
    ring = PrefetchRing.__new__(PrefetchRing)
    ring.cache = cache
    ring.sidecar = None
    ring._token_open = False
    # attach_sidecar(None) must clear the advisor, not leave a stale one.
    PrefetchRing.attach_sidecar(ring, None)
    assert cache._residency_advisor is None
    # Every hook is a None-check and nothing more.
    PrefetchRing.note_demand(ring, "layer.0.mlp", [1, 2, 3])
    assert ring._token_open is False
    PrefetchRing.end_decode_token(ring)
    PrefetchRing.end_decode_token(ring, hidden=object())


def test_eviction_without_advisor_matches_plain_lru():
    """The reprieve list must not change victim choice when no advisor is set."""
    import collections

    from expert_stream.cache import ExpertCache

    cache = ExpertCache.__new__(ExpertCache)
    cache._lru = collections.OrderedDict((("l0", e), {}) for e in range(5))
    cache._freq = {}
    cache._slot_of = {}
    cache._pinned = set()
    cache.resident_bytes = 100
    cache.budget_bytes = 40
    evicted = []

    def _evict_key(key):
        evicted.append(key)
        cache._lru.pop(key)
        cache.resident_bytes -= 20

    cache._evict_key = _evict_key
    ExpertCache._evict_to_budget(cache)
    # Plain LRU order: oldest first, until back under budget.
    assert evicted == [("l0", 0), ("l0", 1), ("l0", 2)], evicted


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"OK: sidecar ({len(tests)} tests)")
