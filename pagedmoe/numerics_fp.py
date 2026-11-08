# Copyright (C) 2026 Noah Clark
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hash of the engine code that can change what a prefill writes.

Saved front matter KV (prefix_disk) is keyed on this, so an update that only
touches decode, I/O, telemetry or the sidecar keeps every saved file. The
list below is what is left out; everything else in the package counts,
including any module added later. Leave a module out only when it can never
run during a prefill or change its numbers.

Each file is hashed as its syntax tree, not its bytes, so comments,
formatting, docstrings and log lines do not count. A print or log call is
left out only when nothing in its arguments can do anything but read: no
calls except a short list of pure ones, no walrus, no lambda. Anything else
in a log line counts like code.

A sealed build has no source at runtime, so the build writes the hash into
the package as FILE and current() reads it back.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

FILE = "_numerics_fp.txt"

EXCLUDE_MODULES = frozenset(
    {
        "activity",
        "copy_texts",
        "dev_mitm",
        "early_decode",
        "fidelity",
        "lookup_decode",
        "mempressure",
        "minimax_tools",
        "mlx_hook",
        "_autoload",
        "native_read",
        "_native_read",
        "pagecache",
        "pretrain_sidecar",
        "release",
        "route_log",
        "stage_gentle",
        "think_budget",
        "time_to_warm",
        "tool_warm",
        "turn_trace",
        "user_config",
        "verify_decode",
        "watchdog",
    }
)
EXCLUDE_DIRS = frozenset({"sidecar", "pretrained_sidecar", "__pycache__"})

_cached: str | None = None


def _counted(pkg: Path, path: Path) -> bool:
    rel = path.relative_to(pkg)
    if any(part in EXCLUDE_DIRS for part in rel.parts[:-1]):
        return False
    return path.suffix == ".py" and path.stem not in EXCLUDE_MODULES


_PURE_CALLS = frozenset(
    {
        "abs", "all", "any", "bool", "dict", "enumerate", "float", "format", "getattr",
        "hasattr", "hex", "id", "int", "isinstance", "len", "list", "max", "min", "repr",
        "round", "set", "sorted", "str", "sum", "tuple", "type", "zip",
    }
)
_PURE_METHODS = frozenset(
    {
        "endswith", "format", "hexdigest", "item", "join", "lower", "lstrip", "replace",
        "rstrip", "split", "startswith", "strip", "tolist", "upper",
    }
)
_LOGGERS = frozenset({"log", "logger", "_log", "LOG", "logging"})
_LOG_LEVELS = frozenset({"debug", "info", "warning", "warn", "error", "exception", "critical"})


def _reads_only(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, (ast.NamedExpr, ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom)):
            return False
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id in _PURE_CALLS:
                continue
            if isinstance(f, ast.Attribute) and f.attr in _PURE_METHODS:
                continue
            return False
    return True


def _is_log(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return False
    call = stmt.value
    f = call.func
    if isinstance(f, ast.Name) and f.id == "print":
        pass
    elif (
        isinstance(f, ast.Attribute)
        and f.attr in _LOG_LEVELS
        and isinstance(f.value, ast.Name)
        and f.value.id in _LOGGERS
    ):
        pass
    else:
        return False
    return all(_reads_only(a) for a in call.args) and all(_reads_only(k.value) for k in call.keywords)


class _Strip(ast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and all(isinstance(s, ast.stmt) for s in body):
            if (
                isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            node.body = [s for s in body if not _is_log(s)]
        for name in ("orelse", "finalbody"):
            block = getattr(node, name, None)
            if isinstance(block, list) and block and all(isinstance(s, ast.stmt) for s in block):
                setattr(node, name, [s for s in block if not _is_log(s)])
        return node


def _normalized(src: bytes) -> bytes:
    """The code a file runs, without comments, docstrings or inert log lines.
    Raw bytes when it does not parse."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return b"raw\0" + src
    tree = _Strip().visit(tree)
    return ast.dump(tree, annotate_fields=False, include_attributes=False).encode()


def source_fingerprint(pkg: Path) -> str | None:
    """Hash of the counted .py files under `pkg`, or None without source."""
    files = sorted((p for p in pkg.rglob("*.py") if _counted(pkg, p)), key=lambda p: str(p.relative_to(pkg)))
    if not files:
        return None
    h = hashlib.sha256(b"numerics-v2")
    for p in files:
        h.update(str(p.relative_to(pkg)).encode())
        h.update(b"\0")
        h.update(_normalized(p.read_bytes()))
        h.update(b"\0")
    return h.hexdigest()


def _compiled_fingerprint(pkg: Path) -> str:
    """Last resort for a tree with neither source nor FILE: every module."""
    h = hashlib.sha256(b"compiled-v1")
    for p in sorted(pkg.rglob("*"), key=lambda p: str(p.relative_to(pkg))):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(pkg)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def current(pkg: Path | None = None) -> str:
    global _cached
    if _cached is not None and pkg is None:
        return _cached
    root = pkg or Path(__file__).resolve().parent
    fp = None
    stamp = root / FILE
    if stamp.is_file():
        fp = stamp.read_text().strip() or None
    if fp is None:
        fp = source_fingerprint(root)
    if fp is None:
        fp = _compiled_fingerprint(root)
    if pkg is None:
        _cached = fp
    return fp


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    fp = source_fingerprint(target)
    if fp is None:
        sys.exit("no source to fingerprint")
    print(fp)
