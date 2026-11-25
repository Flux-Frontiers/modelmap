"""Find models in any directory tree by what the files are.

Used for paths the user names, for the home sweep, and by providers whose
store is a plain tree of model files (oMLX, LM Studio, llama.cpp).

Three shapes are recognized:

- a model directory: ``config.json`` beside ``*.safetensors``, or a
  diffusers ``model_index.json``. The whole directory is one record.
- a GGUF or safetensors file on its own. One record per file, with
  numbered shards (``-00001-of-00004``) joined. A multimodal projector
  (``mmproj``) is kind ``auxiliary``; a GGUF with no tensors is a
  tokenizer stub and is skipped.
- a large checkpoint known only by its extension (``.pt``, ``.bin``...).
  It is listed with an issue saying it was not verified.
"""

from __future__ import annotations

import os
from collections.abc import Collection

from modelmap.headers import sniff
from modelmap.providers.base import PRUNE, Context, Tally, files_under, listdir
from modelmap.providers.describe import (
    _OTHER_FORMATS,
    PICKLE_EXTENSIONS,
    Description,
    describe_directory,
    describe_gguf,
    guess_family,
    is_projector,
    shard_base,
)
from modelmap.records import ModelRecord

NAME = "generic"
MIN_UNVERIFIED_SIZE = 50_000_000
# With sniff_all, a file with an unknown extension is opened only above this.
_SNIFF_MIN_SIZE = 16_000_000


def _name(provider: str, root: str, path: str) -> str:
    """Path under the root for a tool's store; parent and name for a loose file.

    A sweep finds many files called ``last`` or ``model``; the parent
    directory tells them apart without printing a whole path.
    """
    if path == root:
        return os.path.basename(path)
    if provider == NAME:
        return "/".join(path.split(os.sep)[-2:])
    return os.path.relpath(path, root).replace(os.sep, "/")


def _is_torch_file(path: str) -> bool:
    """True for a zip archive or a pickle stream, the two ``torch.save`` forms."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
    except OSError:
        return False
    return head == b"PK\x03\x04" or head[:1] == b"\x80"


def _record(
    ctx: Context,
    provider: str,
    name: str,
    path: str,
    files: list[str],
    description: Description,
    kind: str = "model",
) -> ModelRecord:
    tally = Tally()
    tally.charge(ctx, f"{provider}:{name}", files)
    return tally.record(
        issues=description.issues,
        name=name,
        provider=provider,
        format=description.format,
        path=path,
        kind=kind,
        family=guess_family(name, description.architecture),
        quant=description.quant,
        params=description.params,
        params_exact=description.params_exact,
    )


def _scan_files(
    ctx: Context,
    provider: str,
    root: str,
    directory: str,
    files: list[os.DirEntry[str]],
    min_size: int,
    sniff_all: bool,
) -> list[ModelRecord]:
    """Records for the loose model files directly inside one directory."""
    groups: dict[tuple[str, str], list[str]] = {}
    records = []
    for entry in files:
        ext = os.path.splitext(entry.name)[1].lower()
        kind = {".gguf": "gguf", ".safetensors": "safetensors"}.get(ext)
        if kind is None and (ext in PICKLE_EXTENSIONS or ext in _OTHER_FORMATS):
            if entry.stat().st_size >= min_size:
                # Inside a tool's own store the location vouches for the file;
                # elsewhere a checkpoint must at least start like one.
                unverified = (
                    ext in PICKLE_EXTENSIONS and provider == NAME and not _is_torch_file(entry.path)
                )
                description = Description(
                    format=_OTHER_FORMATS.get(ext, "pytorch"),
                    issues=["unverified: identified by extension only"] if unverified else [],
                )
                name = _name(provider, root, os.path.splitext(entry.path)[0])
                records.append(_record(ctx, provider, name, entry.path, [entry.path], description))
            continue
        if kind is None and sniff_all and entry.stat().st_size >= _SNIFF_MIN_SIZE:
            kind = sniff(entry.path)
        if kind:
            groups.setdefault((kind, shard_base(entry.name)), []).append(entry.path)

    for (kind, base), paths in sorted(groups.items()):
        name = _name(provider, root, os.path.join(directory, base))
        if kind == "gguf":
            description = describe_gguf(paths)
            if description.vocab_only:
                continue
        else:
            description = describe_directory(directory, paths)
        role = "auxiliary" if is_projector(base) else "model"
        path = paths[0] if len(paths) == 1 else directory
        records.append(_record(ctx, provider, name, path, paths, description, role))
    return records


def scan_tree(
    root: str,
    ctx: Context,
    *,
    provider: str = NAME,
    max_depth: int | None = None,
    claimed: Collection[str] = (),
    min_size: int = MIN_UNVERIFIED_SIZE,
    config_marks_model: bool = False,
    sniff_all: bool = False,
) -> list[ModelRecord]:
    """Find every model under a directory.

    :param root: Directory to walk; pass a resolved path.
    :param ctx: The scan context.
    :param provider: Provider name to put on the records.
    :param max_depth: Levels below ``root`` to descend; None for no limit.
    :param claimed: Resolved directories another provider owns; not entered.
    :param min_size: Smallest file to list when only its extension says it
        is a model.
    :param config_marks_model: Treat a directory holding ``config.json`` as
        a model even with no weight files in it. True inside a tool's own
        store, where such a directory is a broken download; false in a
        sweep, where ``config.json`` is everywhere.
    :param sniff_all: Open large files with unknown extensions to check for
        a GGUF or safetensors header.
    :return: Records in directory order.
    """
    records: list[ModelRecord] = []
    stack = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        entries = listdir(ctx, directory)
        names = {e.name for e in entries}
        files = [e for e in entries if e.is_file()]
        has_safetensors = any(e.name.endswith(".safetensors") for e in files)
        is_model_dir = "model_index.json" in names or (
            "config.json" in names and (has_safetensors or config_marks_model)
        )
        if is_model_dir and directory != root:
            contents = files_under(ctx, directory)
            present = [f for f in contents if os.path.exists(f)]
            description = describe_directory(directory, present)
            name = _name(provider, root, directory)
            records.append(_record(ctx, provider, name, directory, contents, description))
            continue

        records += _scan_files(ctx, provider, root, directory, files, min_size, sniff_all)
        if max_depth is not None and depth >= max_depth:
            continue
        subdirs = [
            e.path
            for e in entries
            if e.is_dir(follow_symlinks=False) and e.name not in PRUNE and e.path not in claimed
        ]
        stack.extend((path, depth + 1) for path in reversed(subdirs))
    return records
