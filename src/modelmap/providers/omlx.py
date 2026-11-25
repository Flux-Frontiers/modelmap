"""The oMLX model directories.

From oMLX's ``omlx/settings.py`` (commit 30a896a) and a real install:

- The base directory is ``OMLX_BASE_PATH``, else the path in the one-line
  file ``~/Library/Application Support/oMLX/base-path`` that the macOS app
  writes when the user moves its data, else ``~/.omlx``.
- Model directories are ``OMLX_MODEL_DIR`` (comma separated), else
  ``model.model_dirs`` in ``<base>/settings.json``, else the deprecated
  ``model.model_dir``, else ``<base>/models``.

Each model is a directory, optionally under an organization directory:
``mlx-community/Qwen3-8B-4bit/`` with ``config.json`` and ``*.safetensors``.
"""

from __future__ import annotations

import json
from pathlib import Path

from modelmap.providers.base import Context
from modelmap.providers.generic import scan_tree
from modelmap.records import ModelRecord, ScanRoot

NAME = "omlx"


def _base(ctx: Context) -> tuple[Path, str]:
    override = ctx.first_env("OMLX_BASE_PATH")
    if override:
        return ctx.expand(override[1]), override[0]
    bootstrap = ctx.home / "Library" / "Application Support" / "oMLX" / "base-path"
    try:
        moved = bootstrap.read_text(encoding="utf-8").strip()
    except OSError:
        moved = ""
    if moved:
        return ctx.expand(moved), "base-path"
    return ctx.home / ".omlx", "default"


def roots(ctx: Context) -> list[ScanRoot]:
    """The oMLX model directories, by the precedence in the module docstring."""
    override = ctx.first_env("OMLX_MODEL_DIR")
    if override:
        dirs = [d.strip() for d in override[1].split(",") if d.strip()]
        return [ctx.root(NAME, ctx.expand(d), override[0]) for d in dirs]

    base, source = _base(ctx)
    settings = base / "settings.json"
    try:
        model = json.loads(settings.read_text(encoding="utf-8")).get("model")
    except (OSError, ValueError, AttributeError) as exc:
        if settings.exists():
            ctx.warn(settings, f"unreadable settings: {exc}")
        model = None
    if isinstance(model, dict):
        dirs = model.get("model_dirs") or ([model["model_dir"]] if model.get("model_dir") else [])
        if isinstance(dirs, list) and dirs:
            return [ctx.root(NAME, ctx.expand(str(d)), "settings.json") for d in dirs]
    return [ctx.root(NAME, base / "models", source)]


def scan(root: ScanRoot, ctx: Context) -> list[ModelRecord]:
    """Every model directory under an oMLX model directory."""
    return scan_tree(root.resolved, ctx, provider=NAME, max_depth=3, config_marks_model=True)
