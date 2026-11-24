# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Upload a finished conversion to the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

from .run import REPORT, STATE


def upload(folder, repo: str, *, private: bool = True, token: str | None = None) -> str:
    folder = Path(folder).expanduser()
    if (folder / STATE).exists() or not (folder / REPORT).exists():
        raise RuntimeError(f"{folder} isn't a finished conversion")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_large_folder(repo_id=repo, folder_path=str(folder), repo_type="model",
                            ignore_patterns=[STATE, STATE + ".tmp", ".source/**", ".cache/**"])
    return f"https://huggingface.co/{repo}"
