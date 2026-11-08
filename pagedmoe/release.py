# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Source checkout or release build.

A release ships this module compiled (.so or .pyc), never as .py, so there
is no marker file to delete. In a release the switches below are ignored:
they load code from outside the package or fake the machine for benches.
"""

from __future__ import annotations

import os

SEALED = not __file__.endswith(".py")

DEV_ONLY = frozenset(
    {
        "PAGED_MOE_MITM",
        "PAGED_MOE_HOOK",
        "PAGED_MOE_ENABLE_ALL",
        "PAGED_MOE",
        "PAGED_MOE_SIMULATE_RAM_GB",
        "PAGED_MOE_SIMULATE_GPU_HELD_GB",
        "PAGED_MOE_SIMULATE_GPU_WORKING_SET_GB",
        "PAGED_MOE_ENVELOPE_STRICT",
        "PAGED_MOE_READ_DELAY_US_PER_MB",
        "PAGED_MOE_HOST_PROBE",
        "PAGED_MOE_SANDBOX_HOST_DISK_GB_S",
    }
)


def env(name: str, default: str = "") -> str:
    """os.environ.get that returns `default` for a DEV_ONLY name in a release."""
    if SEALED and name in DEV_ONLY:
        return default
    return os.environ.get(name, default)
