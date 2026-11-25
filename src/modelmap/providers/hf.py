"""The Hugging Face hub cache and the Xet cache beside it.

Layout (``huggingface_hub`` ``constants.py``, the manage-cache guide, and a
real cache):

- ``<hub>/models--org--name/`` per repository, also ``datasets--`` and
  ``spaces--``. Inside: ``blobs/<hash>`` holds the content,
  ``snapshots/<revision>/<path>`` are symlinks into ``blobs``, ``refs/main``
  names the current revision, and ``.no_exist/`` holds empty markers for
  files the Hub does not have.
- ``<hub>/blobs/<2 hex>/<xet hash>`` is a store shared between
  repositories (``huggingface_hub`` 1.32 and later). A repository's
  ``blobs/<hash>`` is then itself a symlink into it. Each shared blob has
  ``.refs`` and ``.lock`` files beside it.
- ``<HF_HOME>/xet`` holds Xet shards and upload staging, not model files.
  It can be deleted and is rebuilt.

A repository is charged through its ``blobs`` entries, following links, so
content in the shared store is attributed to the repositories that use it.
Snapshots give the file names and headers only.
"""

from __future__ import annotations

import os

from modelmap.headers import sniff
from modelmap.providers.base import Context, Tally, files_under, is_dir, listdir
from modelmap.providers.describe import describe_directory, describe_gguf, guess_family
from modelmap.records import ModelRecord, ScanRoot

NAME = "hf"
_XET = "hf-xet"
_KINDS = {"models": "model", "datasets": "dataset", "spaces": "other"}
_SHARED_BOOKKEEPING = (".refs", ".lock")


def roots(ctx: Context) -> list[ScanRoot]:
    """The hub cache and the Xet cache.

    Hub precedence: ``HF_HUB_CACHE``, ``HUGGINGFACE_HUB_CACHE``,
    ``HF_HOME/hub``, ``XDG_CACHE_HOME/huggingface/hub``,
    ``~/.cache/huggingface/hub``. Xet: ``HF_XET_CACHE``, else ``HF_HOME/xet``.
    """
    home_var = ctx.first_env("HF_HOME")
    xdg = ctx.first_env("XDG_CACHE_HOME")
    if home_var:
        hf_home, home_source = ctx.expand(home_var[1]), "HF_HOME"
    elif xdg:
        hf_home, home_source = ctx.expand(xdg[1]) / "huggingface", "XDG_CACHE_HOME"
    else:
        hf_home, home_source = ctx.home / ".cache" / "huggingface", "default"

    direct = ctx.first_env("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")
    if direct:
        hub = ctx.root(NAME, ctx.expand(direct[1]), direct[0])
    else:
        hub = ctx.root(NAME, hf_home / "hub", home_source)
    xet_var = ctx.first_env("HF_XET_CACHE")
    if xet_var:
        xet = ctx.root(_XET, ctx.expand(xet_var[1]), xet_var[0])
    else:
        xet = ctx.root(_XET, hf_home / "xet", home_source)
    return [hub, xet]


def _current_snapshot(ctx: Context, repo: str) -> str | None:
    """The snapshot ``refs/main`` names, else the newest one."""
    snapshots = os.path.join(repo, "snapshots")
    try:
        with open(os.path.join(repo, "refs", "main"), encoding="utf-8") as fh:
            named = os.path.join(snapshots, fh.read().strip())
        if os.path.isdir(named):
            return named
    except OSError:
        pass
    if not os.path.isdir(snapshots):
        return None
    dated = []
    for entry in listdir(ctx, snapshots):
        try:
            if entry.is_dir():
                dated.append((entry.stat().st_mtime, entry.name, entry.path))
        except OSError:
            continue
    return max(dated)[2] if dated else None


def _scan_repo(ctx: Context, repo: os.DirEntry[str], provider: str) -> ModelRecord | None:
    prefix, _, rest = repo.name.partition("--")
    name = rest.replace("--", "/")
    snapshot = _current_snapshot(ctx, repo.path)
    files = files_under(ctx, snapshot) if snapshot else []
    present = [f for f in files if os.path.exists(f)]
    # Blobs are named by hash; the current snapshot gives them their names.
    named = {os.path.realpath(f): os.path.relpath(f, snapshot) for f in present} if snapshot else {}

    tally = Tally()
    blobs = os.path.join(repo.path, "blobs")
    if os.path.isdir(blobs):
        entries = listdir(ctx, blobs)
        labels = {}
        for entry in entries:
            label = named.get(os.path.realpath(entry.path))
            if entry.name.endswith(".incomplete"):
                labels[entry.path] = (entry.name[:16], "incomplete download")
            elif label is None:
                labels[entry.path] = (entry.name[:16], "not in the current revision")
            else:
                labels[entry.path] = (label, "")
        tally.charge(ctx, f"{provider}:{name}", [e.path for e in entries], labels)
        if any(e.name.endswith(".incomplete") for e in entries):
            tally.issues.append("incomplete download")
    if not tally.files:
        # Only refs or .no_exist markers: the Hub was asked, nothing is stored.
        return None

    kind = _KINDS[prefix]
    issues: list[str] = []
    if snapshot:
        issues += [
            f"broken link: {os.path.relpath(f, snapshot)}" for f in files if f not in present
        ]
    description = describe_directory(snapshot or repo.path, present)
    if description.format == "unknown":
        # Config and tokenizer files only, or a dataset: nothing to describe.
        description.format, description.issues = "files", []
        if kind == "model":
            kind = "other"
    return tally.record(
        issues=issues + description.issues,
        name=name,
        provider=provider,
        format=description.format,
        path=repo.path,
        kind=kind,
        family=guess_family(name, description.architecture) if kind == "model" else None,
        quant=description.quant,
        params=description.params,
        params_exact=description.params_exact,
    )


def _scan_shared(ctx: Context, hub: str, provider: str) -> list[ModelRecord]:
    """Shared blobs that no repository links to."""
    shared = os.path.join(hub, "blobs")
    if not os.path.isdir(shared):
        return []
    records = []
    rest = Tally()
    for path in files_under(ctx, shared):
        base = os.path.basename(path)
        if path.endswith(_SHARED_BOOKKEEPING) or base.startswith("."):
            continue
        if ctx.ledger.owner(path) is not None:
            continue
        if sniff(path) == "gguf":
            description = describe_gguf([path])
            name = f"shared blob {base[:12]}"
            tally = Tally()
            tally.charge(ctx, f"{provider}:{name}", [path])
            records.append(
                tally.record(
                    issues=["no repository refers to this blob", *description.issues],
                    name=name,
                    provider=provider,
                    format="gguf",
                    path=path,
                    family=guess_family(name, description.architecture),
                    quant=description.quant,
                    params=description.params,
                )
            )
        else:
            rest.charge(ctx, f"{provider}:unreferenced shared blobs", [path])
    if rest.files:
        records.append(
            rest.record(
                name="unreferenced shared blobs",
                provider=provider,
                format="files",
                path=shared,
                kind="unattributed",
            )
        )
    return records


def scan_hub(hub: str, ctx: Context, provider: str = NAME) -> list[ModelRecord]:
    """Every repository in a hub-layout cache, plus unreferenced shared blobs.

    :param hub: Resolved cache directory.
    :param ctx: The scan context.
    :param provider: Provider name for the records; llama.cpp uses the same
        layout under ``LLAMA_CACHE``.
    """
    records = []
    for entry in listdir(ctx, hub):
        prefix = entry.name.partition("--")[0]
        if prefix in _KINDS and "--" in entry.name and is_dir(entry):
            record = _scan_repo(ctx, entry, provider)
            if record:
                records.append(record)
    return records + _scan_shared(ctx, hub, provider)


def scan(root: ScanRoot, ctx: Context) -> list[ModelRecord]:
    """The hub cache, or one ``cache`` record for the Xet directory."""
    if root.provider != _XET:
        return scan_hub(root.resolved, ctx)
    tally = Tally()
    tally.charge(ctx, f"{NAME}:xet cache", files_under(ctx, root.resolved))
    if not tally.files:
        return []
    return [
        tally.record(
            name="xet cache", provider=NAME, format="files", path=root.resolved, kind="cache"
        )
    ]
