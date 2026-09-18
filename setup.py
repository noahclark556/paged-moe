# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Optional build hook for the native expert-read extension.

A failed compile must not fail ``pip install``: decode falls back to the
pure-Python read pool (see expert_stream.native_read).
"""

from __future__ import annotations

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class optional_build_ext(build_ext):
    def build_extension(self, ext):  # noqa: D102 - setuptools override
        try:
            super().build_extension(ext)
        except Exception as exc:  # noqa: BLE001 - install must still succeed
            self.warn(
                f"PagedMoE native read extension skipped ({ext.name}): {exc}. "
                "Decode will use the Python read pool or a JIT build at runtime."
            )


setup(
    ext_modules=[
        Extension(
            "expert_stream._native_read",
            sources=["expert_stream/_native_read.c"],
            extra_compile_args=["-O2"],
        )
    ],
    cmdclass={"build_ext": optional_build_ext},
)
