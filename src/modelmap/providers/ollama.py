"""The Ollama model store.

Layout, from Ollama's source (``manifest/paths.go``, ``manifest/layer.go``,
``envconfig/config.go`` at commit 924cc28) and a real 0.40 store:

- ``manifests/<host>/<namespace>/<model>/<tag>`` is a Docker v2 manifest:
  JSON with a ``config`` descriptor and a ``layers`` list, each entry
  ``{mediaType, digest, size}``. ``manifests-v2/`` is a newer tree with the
  same shape that Ollama reads first; an entry there may be a manifest list
  whose ``manifests`` are child manifests, inline or by digest.
- ``blobs/sha256-<hex>`` holds the content of digest ``sha256:<hex>``.
- ``metadata/`` caches GGUF header fields per blob and is regenerated on
  demand.
- Weights are the ``model`` layers (several for a split GGUF) or, for a
  safetensors model, one ``tensor`` layer per tensor. ``projector``,
  ``adapter`` and ``draft`` layers are further weight files of the same
  model. Template, params, license and system layers are small.

Several tags often point at one set of weights (``qwen3:8b`` and
``qwen3:latest``; a tag made only to change the context length). Such tags
are one record, named for the first tag, with the rest as aliases, so the
map shows one tile per model actually on disk.
"""

from __future__ import annotations

import json
import os
from typing import Any

from modelmap.headers import sniff
from modelmap.providers.base import Context, Tally, files_under, listdir
from modelmap.providers.describe import Description, describe_gguf, guess_family
from modelmap.records import ModelRecord, ScanRoot

NAME = "ollama"
_PREFIX = "application/vnd.ollama.image."
_PRIMARY = (_PREFIX + "model", _PREFIX + "tensor")
_DEFAULT_HOSTS = ("registry.ollama.ai", "ollama.com")
# Read in this order; a tag found in both trees is taken from the first.
_MANIFEST_TREES = ("manifests-v2", "manifests")


def roots(ctx: Context) -> list[ScanRoot]:
    """``OLLAMA_MODELS`` if set, else the per-user and service defaults.

    The Linux installer runs Ollama as a service user whose home is
    ``/usr/share/ollama``, so its models are not under the caller's home.
    """
    override = ctx.first_env("OLLAMA_MODELS")
    if override:
        return [ctx.root(NAME, ctx.expand(override[1]), override[0])]
    found = [ctx.root(NAME, ctx.home / ".ollama" / "models", "default")]
    if ctx.platform == "linux":
        found.append(ctx.root(NAME, "/usr/share/ollama/.ollama/models", "default (service)"))
    return found


def tag_name(parts: tuple[str, ...]) -> str:
    """Turn a manifest path into the name ``ollama list`` shows.

    ``registry.ollama.ai/library/qwen3/8b`` is ``qwen3:8b``; another
    namespace keeps it (``user/model:tag``); another host keeps the host.
    """
    *repo, tag = parts
    if repo:
        # manifests-v2 writes a host's port separator as %3A.
        repo[0] = repo[0].replace("%3A", ":").replace("%3a", ":")
    if repo and repo[0].lower() in _DEFAULT_HOSTS:
        repo = repo[1:]
        if repo[:1] == ["library"]:
            repo = repo[1:]
    return f"{'/'.join(repo)}:{tag}"


def _blob(store: str, digest: str) -> str:
    return os.path.join(store, "blobs", digest.replace(":", "-"))


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _layers(store: str, manifest: Any) -> list[dict[str, Any]]:
    """Every blob descriptor a manifest, or a manifest list, refers to."""
    found: list[dict[str, Any]] = []
    for child in manifest.get("manifests") or []:
        if "layers" not in child and "digest" in child:
            # A child held by reference: the manifest itself is a blob.
            found.append(child)
            try:
                child = _load_json(_blob(store, child["digest"]))
            except (OSError, ValueError):
                continue
        found += _layers(store, child)
    found += [layer for layer in manifest.get("layers") or [] if "digest" in layer]
    config = manifest.get("config")
    if isinstance(config, dict) and config.get("digest"):
        found.append(config)
    return found


def _read_manifests(ctx: Context, store: str) -> dict[str, list[tuple[str, list[dict[str, Any]]]]]:
    """Group tags by their weights: ``{weights key: [(tag, descriptors)]}``."""
    groups: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {}
    seen: set[str] = set()
    for tree in _MANIFEST_TREES:
        base = os.path.join(store, tree)
        if not os.path.isdir(base):
            continue
        for path in files_under(ctx, base):
            if os.path.basename(path).startswith("."):
                continue
            name = tag_name(tuple(os.path.relpath(path, base).split(os.sep)))
            if name in seen:
                continue
            try:
                layers = _layers(store, _load_json(path))
            except (OSError, ValueError, AttributeError, TypeError) as exc:
                ctx.warn(path, f"unreadable manifest: {exc}")
                continue
            seen.add(name)
            weights = [d["digest"] for d in layers if d.get("mediaType") in _PRIMARY]
            key = "+".join(weights) if weights else f"manifest:{name}"
            groups.setdefault(key, []).append((name, layers))
    return groups


def _describe(store: str, layers: list[dict[str, Any]]) -> tuple[Description, str | None]:
    """Describe a model from its primary weight layers; also its main file."""
    models = [_blob(store, d["digest"]) for d in layers if d.get("mediaType") == _PRIMARY[0]]
    present = [p for p in models if os.path.exists(p)]
    if present and all(sniff(p) == "gguf" for p in present):
        return describe_gguf(present), present[0]
    if any(d.get("mediaType") == _PRIMARY[1] for d in layers):
        return Description(format="safetensors"), None
    kind = sniff(present[0]) if present else None
    return Description(format=kind or "unknown"), None


def scan(root: ScanRoot, ctx: Context) -> list[ModelRecord]:
    """One record per unique set of weights, plus any unreferenced blobs."""
    store = root.resolved
    records = []
    groups = _read_manifests(ctx, store)
    for tags in sorted(groups.values(), key=lambda tags: min(tag for tag, _ in tags)):
        tags.sort(key=lambda tag: tag[0])
        name, first_layers = tags[0]
        issues: list[str] = []
        paths: list[str] = []
        for _, layers in tags:
            for layer in layers:
                path = _blob(store, layer["digest"])
                if path in paths:
                    continue
                if os.path.exists(path):
                    paths.append(path)
                elif layer.get("mediaType", "").startswith(_PREFIX):
                    issues.append(
                        f"missing blob {layer['digest'][:19]} "
                        f"({layer.get('size', 'unknown')} bytes declared)"
                    )
        tally = Tally()
        tally.charge(ctx, f"{NAME}:{name}", paths)
        description, main = _describe(store, first_layers)
        records.append(
            tally.record(
                issues=issues + description.issues,
                name=name,
                provider=NAME,
                format=description.format,
                path=main or store,
                family=guess_family(name, description.architecture),
                quant=description.quant,
                params=description.params,
                params_exact=description.params_exact,
                aliases=[tag for tag, _ in tags[1:]],
            )
        )

    orphans = Tally()
    blobs = os.path.join(store, "blobs")
    if os.path.isdir(blobs):
        for entry in listdir(ctx, blobs):
            if ctx.ledger.owner(entry.path) is None:
                orphans.charge(ctx, f"{NAME}:unreferenced blobs", [entry.path])
    if orphans.files:
        records.append(
            orphans.record(
                name="unreferenced blobs",
                provider=NAME,
                format="files",
                path=blobs,
                kind="unattributed",
            )
        )
    return records
