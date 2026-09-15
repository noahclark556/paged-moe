# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Drop-in hook: route selected ``mlx_lm.load`` calls through PagedMoE.

Opt-in (any one is enough)
--------------------------
  1. ``~/paged-moe-config.yaml`` ``models:`` list (or ``enable_all: true``)
  2. Marker in the checkpoint dir: ``paged_moe.yaml`` or ``.paged_moe``
  3. ``PAGED_MOE=1`` for every load

Debug / kill switch
-------------------
  ``PAGED_MOE_HOOK=0`` - leave loads alone
  ``PAGED_MOE_DEBUG=1`` - print stream / passthrough decisions
  ``paged-moe status|install|uninstall``
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

from .user_config import (
    config_path,
    default_root,
    ensure_sample_config,
    load_user_config,
)

_ORIG_LOAD: Callable[..., Any] | None = None
_INSTALLED = False
_PTH_NAME = "zz_paged_moe_hook.pth"


def _debug(msg: str) -> None:
    if _debug_enabled():
        print(f"[paged-moe] {msg}", file=sys.stderr, flush=True)


def _debug_enabled() -> bool:
    if os.environ.get("PAGED_MOE_DEBUG", "").strip() in ("1", "true", "yes"):
        return True
    cfg = load_user_config()
    return bool(cfg.get("debug"))


def _hook_disabled() -> bool:
    return os.environ.get("PAGED_MOE_HOOK", "1").strip() in ("0", "false", "no", "off")


def _enable_all() -> bool:
    if os.environ.get("PAGED_MOE", "").strip() in ("1", "true", "yes", "all"):
        return True
    if os.environ.get("EXPERT_STREAM_HOOK", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "all",
    ):
        return True
    return bool(load_user_config().get("enable_all"))


def _realpath(path: str | Path) -> str:
    try:
        return str(Path(path).expanduser().resolve())
    except Exception:
        return str(Path(path).expanduser())


def _marker_env(model_dir: Path) -> dict[str, str] | None:
    """Return {} if marker present, env dict if paged_moe.yaml has env, else None."""
    yaml_marker = model_dir / "paged_moe.yaml"
    dot_marker = model_dir / ".paged_moe"
    if yaml_marker.is_file():
        try:
            import yaml

            raw = yaml.safe_load(yaml_marker.read_text()) or {}
        except Exception:
            return {}
        if isinstance(raw, dict):
            env = raw.get("env") or {}
            if isinstance(env, dict):
                return {str(k): str(v) for k, v in env.items()}
            return {}
        return {}
    if dot_marker.exists():
        return {}
    return None


def _match_entry(path_or_repo: str) -> dict[str, Any] | None:
    """Return the matching models[] entry, a synthetic marker entry, or None."""
    cfg = load_user_config()
    want = _realpath(path_or_repo) if Path(path_or_repo).expanduser().exists() else str(path_or_repo)
    want_name = Path(want).name

    for entry in cfg.get("models") or []:
        if not isinstance(entry, dict):
            continue
        entry_path = entry.get("path") or entry.get("model_path") or ""
        if not entry_path:
            continue
        ep = _realpath(entry_path) if Path(entry_path).expanduser().exists() else str(entry_path)
        names = entry.get("names") or entry.get("aliases") or []
        if not isinstance(names, list):
            names = [names]
        name_hit = want_name in {str(n) for n in names} or str(path_or_repo) in {
            str(n) for n in names
        }
        if ep == want or Path(ep).name == want_name or name_hit:
            return entry

    # Marker file beside a local checkpoint
    local = Path(path_or_repo).expanduser()
    if local.is_dir():
        marker_env = _marker_env(local)
        if marker_env is not None:
            return {"path": str(local), "env": marker_env, "_source": "marker"}

    return None


def should_stream(path_or_repo: str) -> bool:
    if _enable_all():
        return True
    return _match_entry(path_or_repo) is not None


def _merged_env(path_or_repo: str) -> dict[str, str]:
    cfg = load_user_config()
    out: dict[str, str] = {}
    defaults = cfg.get("defaults") or {}
    if isinstance(defaults, dict):
        out.update({str(k): str(v) for k, v in defaults.items()})
    entry = _match_entry(path_or_repo)
    if entry:
        env = entry.get("env") or {}
        if isinstance(env, dict):
            out.update({str(k): str(v) for k, v in env.items()})
    return out


def _apply_env(env: dict[str, str]) -> None:
    for k, v in env.items():
        os.environ[str(k)] = str(v)
    # Refresh already-imported engine knobs (SIDECAR_DIR, PRUNE, ...).
    try:
        from . import config as es_config

        es_config.sync_from_environ()
    except Exception as e:
        _debug(f"sync_from_environ failed: {e!r}")


def _patched_load(path_or_repo: str, *args, **kwargs):
    use = should_stream(path_or_repo)
    if not use:
        _debug(f"passthrough mlx_lm.load {path_or_repo!r}")
        assert _ORIG_LOAD is not None
        return _ORIG_LOAD(path_or_repo, *args, **kwargs)

    env = _merged_env(path_or_repo)
    if env:
        _apply_env(env)
        _debug(f"env overrides: {sorted(env)}")
    _debug(f"stream via PagedMoE {path_or_repo!r}")

    # Local import so installing the hook does not pull Metal until a load.
    from .loader import load as es_load
    from .sidecar.slot import _debug_enabled

    # Quiet by default for drop-in users; opt in with PAGED_MOE_DEBUG /
    # EXPERT_STREAM_SIDECAR_DEBUG (or pass verbose= explicitly).
    kwargs.setdefault("verbose", _debug_enabled())
    return es_load(path_or_repo, *args, **kwargs)


def install() -> bool:
    """Patch ``mlx_lm.utils.load`` (and ``mlx_lm.load``) in this process."""
    global _ORIG_LOAD, _INSTALLED
    if _hook_disabled():
        _debug("hook disabled (PAGED_MOE_HOOK=0)")
        return False
    if _INSTALLED and _ORIG_LOAD is not None:
        return True
    try:
        import mlx_lm
        import mlx_lm.utils as utils
    except Exception as e:
        _debug(f"mlx_lm not importable: {e!r}")
        return False

    if getattr(utils.load, "__paged_moe_wrapped__", False):
        _ORIG_LOAD = getattr(utils.load, "__paged_moe_orig__", utils.load)
        _INSTALLED = True
        return True

    _ORIG_LOAD = utils.load
    wrapped = _patched_load
    wrapped.__paged_moe_wrapped__ = True  # type: ignore[attr-defined]
    wrapped.__paged_moe_orig__ = _ORIG_LOAD  # type: ignore[attr-defined]
    utils.load = wrapped
    try:
        mlx_lm.load = wrapped
    except Exception:
        pass
    _INSTALLED = True
    _debug(f"installed (config={config_path()})")
    return True


def uninstall() -> bool:
    """Restore the original ``mlx_lm.utils.load`` in this process."""
    global _ORIG_LOAD, _INSTALLED
    if _ORIG_LOAD is None:
        _INSTALLED = False
        return False
    try:
        import mlx_lm
        import mlx_lm.utils as utils

        utils.load = _ORIG_LOAD
        try:
            mlx_lm.load = _ORIG_LOAD
        except Exception:
            pass
    except Exception:
        return False
    _INSTALLED = False
    _debug("uninstalled")
    return True


def is_installed() -> bool:
    try:
        import mlx_lm.utils as utils

        if getattr(utils.load, "__paged_moe_wrapped__", False):
            return True
    except Exception:
        pass
    return _INSTALLED


def site_packages_dir() -> Path | None:
    """Prefer the venv site-packages that owns mlx_lm."""
    try:
        import mlx_lm

        p = Path(mlx_lm.__file__).resolve().parent.parent
        if p.name == "site-packages" or p.name.endswith("site-packages"):
            return p
        # editable / unusual layouts - fall back to site.getsitepackages
    except Exception:
        pass
    try:
        import site

        for sp in site.getsitepackages():
            path = Path(sp)
            if path.is_dir():
                return path
    except Exception:
        pass
    return None


def pth_path() -> Path | None:
    sp = site_packages_dir()
    return (sp / _PTH_NAME) if sp else None


def install_sitecustomize_pth(models_dir: str | Path | None = None) -> Path:
    """Write a ``.pth`` so every interpreter using this env auto-installs the hook."""
    sp = site_packages_dir()
    if sp is None:
        raise RuntimeError("could not locate site-packages for mlx_lm")
    path = sp / _PTH_NAME
    # site.py executes lines that start with "import "
    path.write_text(
        "import expert_stream._autoload as _paged_moe_boot; _paged_moe_boot.autoload()\n"
    )
    ensure_sample_config(models_dir)
    return path


def uninstall_sitecustomize_pth() -> bool:
    path = pth_path()
    if path and path.is_file():
        path.unlink()
        return True
    return False


def status() -> dict[str, Any]:
    cfg = load_user_config()
    pth = pth_path()
    return {
        "hook_installed_in_process": is_installed(),
        "hook_env_disabled": _hook_disabled(),
        "enable_all": _enable_all(),
        "config_path": str(config_path()),
        "config_exists": config_path().is_file(),
        "paged_moe_home": str(default_root()),
        "models_listed": len(cfg.get("models") or []),
        "pth_path": str(pth) if pth else None,
        "pth_present": bool(pth and pth.is_file()),
        "debug": _debug_enabled(),
    }


def _pop_models_dir(argv: list[str]) -> str | None:
    """Pull ``--models-dir PATH`` out of argv; leave remaining args in place."""
    out: list[str] = []
    models_dir: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--models-dir" and i + 1 < len(argv):
            models_dir = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--models-dir="):
            models_dir = arg.split("=", 1)[1]
            i += 1
            continue
        out.append(arg)
        i += 1
    argv[:] = out
    return models_dir


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    models_dir = _pop_models_dir(argv)
    cmd = (argv[0] if argv else "status").lower()

    if cmd in ("-h", "--help", "help"):
        print(
            "usage: paged-moe "
            "[status|install|uninstall|ensure-config|which <model-path>] "
            "[--models-dir PATH]"
        )
        return 0

    if cmd == "ensure-config":
        path = ensure_sample_config(models_dir)
        print(f"wrote/kept {path}")
        return 0

    if cmd == "install":
        path = install_sitecustomize_pth(models_dir)
        install()
        print(f"installed auto-hook -> {path}")
        print(f"config -> {config_path()}")
        return 0

    if cmd == "uninstall":
        ok = uninstall_sitecustomize_pth()
        uninstall()
        print("removed auto-hook" if ok else "no auto-hook pth found")
        return 0

    if cmd == "which":
        if len(argv) < 2:
            print("usage: paged-moe which <model-path>", file=sys.stderr)
            return 2
        target = argv[1]
        hit = should_stream(target)
        print(f"{'STREAM' if hit else 'passthrough'}  {target}")
        if hit:
            env = _merged_env(target)
            if env:
                for k in sorted(env):
                    print(f"  {k}={env[k]}")
        return 0

    # status (default)
    import json

    print(json.dumps(status(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
