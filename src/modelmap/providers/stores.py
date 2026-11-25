"""Tools whose store is a plain tree of model files.

Each needs only a rule for where the tree is; the generic tree scanner
does the rest. Sources for the locations:

- LM Studio: ``findLMStudioHome.ts`` in lmstudio-js and ``lmstudioPaths.ts``
  and ``importCmd.ts`` in the ``lms`` CLI. The home is the path in
  ``~/.lmstudio-home-pointer``, else ``~/.cache/lm-studio`` if it exists
  (installs older than 0.3.6), else ``~/.lmstudio``. Models are in the
  ``downloadsFolder`` of ``<home>/settings.json``, else ``<home>/models``,
  laid out ``publisher/model/file.gguf``.
- llama.cpp: ``fs_get_cache_directory`` in ``common/common.cpp`` and
  ``get_cache_directory`` in ``common/hf-cache.cpp``. ``LLAMA_CACHE`` is
  used as given. Otherwise flat files live in ``%LOCALAPPDATA%\\llama.cpp``,
  ``~/Library/Caches/llama.cpp`` or ``$XDG_CACHE_HOME/llama.cpp``. Since
  2026-03 ``-hf`` downloads go to the Hugging Face hub cache instead, which
  the ``hf`` provider reads; under ``LLAMA_CACHE`` the same hub layout
  appears in this directory.
- GPT4All: the desktop settings documentation (download path) and
  ``gpt4all.py`` in the Python bindings (``~/.cache/gpt4all``).
- Jan: the data-folder documentation and ``resolve_jan_data_folder`` in
  ``src-tauri``: ``JAN_DATA_FOLDER``, else ``data_folder`` in
  ``<data dir>/Jan/settings.json``, else ``<data dir>/Jan/data``. Models are
  in ``llamacpp/models`` and ``mlx/models`` under the data folder.
- Whisper: ``load_model`` in ``whisper/__init__.py``.

Only llama.cpp's flat cache rule was checked against a local checkout. No
store for LM Studio, GPT4All, Jan or Whisper was on the development machine,
so these rest on the sources above and on fixtures.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from modelmap.providers import hf
from modelmap.providers.base import Context
from modelmap.providers.generic import scan_tree
from modelmap.records import ModelRecord, ScanRoot

Found = list[tuple[Path, str]]


class TreeStore:
    """A provider whose models are found by walking a directory tree.

    :param name: Provider name.
    :param find: Returns ``(path, source)`` for each place to look.
    :param options: Extra arguments for :func:`scan_tree`.
    """

    def __init__(self, name: str, find: Callable[[Context], Found], **options: Any) -> None:
        self.NAME = name
        self._find = find
        self._options = options

    def roots(self, ctx: Context) -> list[ScanRoot]:
        return [ctx.root(self.NAME, path, source) for path, source in self._find(ctx)]

    def scan(self, root: ScanRoot, ctx: Context) -> list[ModelRecord]:
        return scan_tree(root.resolved, ctx, provider=self.NAME, **self._options)


def _data_dir(ctx: Context) -> Path:
    """The per-user application data directory of the platform."""
    if ctx.platform == "darwin":
        return ctx.home / "Library" / "Application Support"
    if ctx.platform == "win32":
        return ctx.expand(ctx.env.get("APPDATA") or str(ctx.home / "AppData" / "Roaming"))
    return ctx.expand(ctx.env.get("XDG_DATA_HOME") or str(ctx.home / ".local" / "share"))


def _cache_dir(ctx: Context) -> Path:
    return ctx.expand(ctx.env.get("XDG_CACHE_HOME") or str(ctx.home / ".cache"))


def _lmstudio(ctx: Context) -> Found:
    pointer = ctx.home / ".lmstudio-home-pointer"
    try:
        pointed = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        pointed = ""
    if pointed:
        home, source = ctx.expand(pointed), pointer.name
    else:
        legacy = ctx.home / ".cache" / "lm-studio"
        home, source = (
            (legacy, "default (legacy)")
            if legacy.is_dir()
            else (
                ctx.home / ".lmstudio",
                "default",
            )
        )
    try:
        folder = json.loads((home / "settings.json").read_text(encoding="utf-8"))["downloadsFolder"]
    except (OSError, ValueError, KeyError, TypeError):
        folder = None
    if isinstance(folder, str) and folder:
        return [(ctx.expand(folder), "settings.json")]
    return [(home / "models", source)]


def _llamacpp(ctx: Context) -> Found:
    override = ctx.first_env("LLAMA_CACHE")
    if override:
        return [(ctx.expand(override[1]), override[0])]
    if ctx.platform == "darwin":
        return [(ctx.home / "Library" / "Caches" / "llama.cpp", "default")]
    if ctx.platform == "win32":
        local = ctx.env.get("LOCALAPPDATA") or str(ctx.home / "AppData" / "Local")
        return [(ctx.expand(local) / "llama.cpp", "default")]
    return [(_cache_dir(ctx) / "llama.cpp", "default")]


def _gpt4all(ctx: Context) -> Found:
    if ctx.platform == "win32":
        local = ctx.env.get("LOCALAPPDATA") or str(ctx.home / "AppData" / "Local")
        desktop = ctx.expand(local) / "nomic.ai" / "GPT4All"
    else:
        desktop = _data_dir(ctx) / "nomic.ai" / "GPT4All"
    return [(desktop, "default"), (ctx.home / ".cache" / "gpt4all", "default (python)")]


def _jan(ctx: Context) -> Found:
    app = _data_dir(ctx) / "Jan"
    override = ctx.first_env("JAN_DATA_FOLDER")
    if override:
        data, source = ctx.expand(override[1]), override[0]
    else:
        try:
            folder = json.loads((app / "settings.json").read_text(encoding="utf-8"))["data_folder"]
        except (OSError, ValueError, KeyError, TypeError):
            folder = None
        if isinstance(folder, str) and folder:
            data, source = ctx.expand(folder), "settings.json"
        else:
            data, source = app / "data", "default"
    return [(data / "llamacpp" / "models", source), (data / "mlx" / "models", source)]


def _whisper(ctx: Context) -> Found:
    return [(_cache_dir(ctx) / "whisper", "default")]


class _LlamaCpp(TreeStore):
    """llama.cpp's cache: loose GGUF files, and hub-layout repositories."""

    def scan(self, root: ScanRoot, ctx: Context) -> list[ModelRecord]:
        return hf.scan_hub(root.resolved, ctx, self.NAME) + super().scan(root, ctx)


lmstudio = TreeStore("lmstudio", _lmstudio, max_depth=3, config_marks_model=True)
llamacpp = _LlamaCpp("llamacpp", _llamacpp, max_depth=0)
gpt4all = TreeStore("gpt4all", _gpt4all, max_depth=1)
jan = TreeStore("jan", _jan, max_depth=3, config_marks_model=True)
whisper = TreeStore("whisper", _whisper, max_depth=0, min_size=0)
