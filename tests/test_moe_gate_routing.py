# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for MoEGate-family routing helpers.

The full streamed correctness suite (tests/test_stream.py) covers Linear
routers via a tiny Qwen3-MoE. These tests pin the GLM/DeepSeek path, which
returns `(inds, scores)` instead of logits, without needing a 200 GB
checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from expert_stream.streaming import (
    _align_scores_to_indices,
    _nucleus_mask,
    _selector_mask,
    _unit_mass,
)


def test_unit_mass_cancels_routed_scaling():
    # MoEGate multiplies by routed_scaling_factor after normalizing; selectors
    # must see share-of-mixture, not the scaled values.
    scaled = np.array([[0.5 * 2.5, 0.3 * 2.5, 0.2 * 2.5]], dtype=np.float32)
    u = _unit_mass(scaled)
    assert np.allclose(u.sum(axis=-1), 1.0)
    assert np.allclose(u, [[0.5, 0.3, 0.2]])


def test_align_scores_permutes_to_switch_order():
    # argpartition order from a re-run of the gate need not match the parent's.
    gate_inds = np.array([[3, 1, 7, 2]], dtype=np.int32)
    gate_scores = np.array([[0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
    ind2 = np.array([[1, 2, 3, 7]], dtype=np.int32)  # parent order
    aligned = _align_scores_to_indices(gate_inds, gate_scores, ind2)
    assert np.allclose(aligned, [[0.3, 0.1, 0.4, 0.2]])


def test_selector_mask_shared_across_families():
    w = np.array(
        [
            [0.55, 0.20, 0.15, 0.10],  # peaked
            [0.30, 0.28, 0.22, 0.20],  # flat
        ],
        dtype=np.float32,
    )
    # PRUNE 0.5: keep within half of the winner.
    m = _selector_mask(w, prune=0.5)
    assert m[0].tolist() == [True, False, False, False]
    # Flat row: 0.5 * 0.30 = 0.15, so every slot clears the bar.
    assert m[1].tolist() == [True, True, True, True]

    # Harder prune separates the flat row.
    m = _selector_mask(w, prune=0.8)
    assert m[1].tolist() == [True, True, False, False]

    # Cap 2 of 4.
    m = _selector_mask(w, cap=2)
    assert m[0].sum() == 2 and m[0][0] and m[0][1]
    assert m[1].sum() == 2

    # Nucleus 0.7: peaked needs winner+second (0.55 alone is short of 0.7);
    # flat needs 3.
    m = _selector_mask(w, top_p=0.7)
    assert m[0].tolist() == [True, True, False, False]
    assert m[1].sum() == 3

    m = _selector_mask(w, top_p=0.5)
    assert m[0].tolist() == [True, False, False, False]


def test_wait_threshold_in_unit_mass_units():
    # A 0.2 WAIT_ABOVE gate must mean "20% of the mixture" on both families.
    # After unit-mass, a scaled MoEGate row and a Linear softmax row with the
    # same shares produce the same skip set.
    shares = np.array([[0.45, 0.25, 0.18, 0.12]], dtype=np.float32)
    linear = shares.copy()
    moe_gate = shares * 2.5
    assert np.array_equal(linear < 0.2, _unit_mass(moe_gate) < 0.2)


def test_nucleus_never_empties():
    w = np.array([[1.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]])
    assert (_nucleus_mask(w, 1e-9).sum(axis=-1) == 1).all()
    assert (_nucleus_mask(w, 1.0).sum(axis=-1) >= 1).all()


if __name__ == "__main__":
    test_unit_mass_cancels_routed_scaling()
    test_align_scores_permutes_to_switch_order()
    test_selector_mask_shared_across_families()
    test_wait_threshold_in_unit_mass_units()
    test_nucleus_never_empties()
    print("OK: MoEGate routing helpers")
