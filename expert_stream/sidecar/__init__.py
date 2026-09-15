# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Online multi-head expert sidecar.

Import ExpertSidecar from here. When EXPERT_STREAM_SIDECAR=0 the package is
never constructed by patch_model.
"""

from .features import (
    SLOT_VERSION,
    build_features,
    demand_to_features,
    legacy_path_slot_id,
    migrate_legacy_slot_files,
    model_slot_id,
    same_layer_ceiling,
)
from .service import ExpertSidecar, unlock_slots
from .slot import ExpertHeadSlot, ModelSlot

__all__ = [
    "ExpertSidecar",
    "ExpertHeadSlot",
    "ModelSlot",
    "SLOT_VERSION",
    "build_features",
    "demand_to_features",
    "legacy_path_slot_id",
    "migrate_legacy_slot_files",
    "model_slot_id",
    "same_layer_ceiling",
    "unlock_slots",
]
