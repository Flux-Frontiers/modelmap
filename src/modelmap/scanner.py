"""Run the providers and the tree scans and return one :class:`Scan`.

Order of work, which is also the order files are charged in:

1. Each provider's stores, in the order of ``PROVIDERS``.
2. Paths the user named.
3. With no paths named, a sweep of the home directory for loose models.

A directory is scanned once. A root that is the same directory as one
already scanned, or lies inside one, is skipped: the ledger would keep the
totals right, but every model would be listed a second time at size 0.
"""

from __future__ import annotations

import os
import socket
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from modelmap import __version__
from modelmap.providers import PROVIDERS, hf, ollama
from modelmap.providers.base import Context, finish
from modelmap.providers.generic import MIN_UNVERIFIED_SIZE, scan_file, scan_tree
from modelmap.records import ModelRecord, Scan, ScanRoot

SWEEP_DEPTH = 5
# Application data: full of files, and every store inside it is a provider root.
_SWEEP_SKIP = {"darwin": ("Library",), "win32": ("AppData",)}


@dataclass
class Options:
    """What to scan.

    :param paths: Files or directories the user named.
    :param only: Provider names to run; None for all.
    :param defaults: Scan the known stores.
    :param sweep: Sweep the home directory for loose models. None means
        yes when neither paths nor providers are named.
    :param excludes: Glob patterns skipped in tree scans.
    :param min_size: Smallest file to list when only its extension says it
        is a model.
    :param max_depth: Levels to descend below a named path; None for all.
    """

    paths: Sequence[str] = ()
    only: Sequence[str] | None = None
    defaults: bool = True
    sweep: bool | None = None
    excludes: Sequence[str] = ()
    min_size: int = MIN_UNVERIFIED_SIZE
    max_depth: int | None = None


def _is_ollama_store(directory: str, names: set[str]) -> bool:
    """``manifests`` beside ``blobs/sha256-<hex>`` files.

    Docker's model store has the same two directory names but keeps its
    blobs one level down, in ``blobs/sha256/``.
    """
    if "blobs" not in names or not names & {"manifests", "manifests-v2"}:
        return False
    try:
        with os.scandir(os.path.join(directory, "blobs")) as it:
            return any(e.name.startswith("sha256-") for e in it)
    except OSError:
        return False


def _is_hub_cache(directory: str, names: set[str]) -> bool:
    """A ``models--org--name`` style directory that holds ``blobs`` or ``snapshots``."""
    for name in sorted(names):
        if name.startswith(("models--", "datasets--", "spaces--")):
            repo = os.path.join(directory, name)
            if os.path.isdir(os.path.join(repo, "blobs")) or os.path.isdir(
                os.path.join(repo, "snapshots")
            ):
                return True
    return False


@dataclass
class _Run:
    ctx: Context
    options: Options
    records: list[ModelRecord] = field(default_factory=list)
    roots: list[ScanRoot] = field(default_factory=list)
    done: list[str] = field(default_factory=list)

    def covered(self, resolved: str) -> bool:
        """True if a root already scanned is this directory or contains it."""
        return any(resolved == d or resolved.startswith(d + os.sep) for d in self.done)

    def claim(self, directory: str, names: set[str]) -> list[ModelRecord] | None:
        """Hand a directory that is some tool's store to that tool's provider."""
        if _is_ollama_store(directory, names):
            root = self.ctx.root(ollama.NAME, directory, "layout")
            found = ollama.scan(root, self.ctx)
        elif _is_hub_cache(directory, names):
            root = self.ctx.root(hf.NAME, directory, "layout")
            found = hf.scan_hub(directory, self.ctx)
        else:
            return None
        self.roots.append(root)
        self.done.append(directory)
        return found

    def stores(self) -> None:
        for provider in PROVIDERS:
            if self.options.only is not None and provider.NAME not in self.options.only:
                continue
            for root in provider.roots(self.ctx):
                self.roots.append(root)
                if root.exists and not self.covered(root.resolved):
                    self.records += provider.scan(root, self.ctx)
                    self.done.append(root.resolved)

    def paths(self) -> None:
        for path in self.options.paths:
            root = self.ctx.root("user", self.ctx.expand(path), "argument")
            root.exists = os.path.exists(root.resolved)
            self.roots.append(root)
            if not root.exists:
                self.ctx.warn(path, "no such file or directory")
            elif self.covered(root.resolved):
                continue
            elif os.path.isdir(root.resolved):
                self.records += self.tree(root.resolved, self.options.max_depth, sniff_all=True)
                self.done.append(root.resolved)
            else:
                self.records += scan_file(root.resolved, self.ctx, min_size=0)

    def sweep(self) -> None:
        home = os.path.realpath(self.ctx.home)
        root = self.ctx.root("sweep", home, f"home, depth {SWEEP_DEPTH}")
        self.roots.append(root)
        skip = [os.path.join(home, name) for name in _SWEEP_SKIP.get(self.ctx.platform, ())]
        self.records += self.tree(home, SWEEP_DEPTH, sniff_all=False, skip=skip)

    def tree(
        self, root: str, max_depth: int | None, sniff_all: bool, skip: Sequence[str] = ()
    ) -> list[ModelRecord]:
        return scan_tree(
            root,
            self.ctx,
            max_depth=max_depth,
            claimed={*self.done, *skip},
            min_size=self.options.min_size,
            sniff_all=sniff_all,
            excludes=self.options.excludes,
            claim=self.claim,
            root_may_be_model=True,
        )


def scan(ctx: Context | None = None, options: Options | None = None) -> Scan:
    """Scan a machine.

    :param ctx: The machine to scan; the running one by default.
    :param options: What to scan; the known stores and a home sweep by
        default.
    :return: Everything found. If the scan is interrupted, what was found
        so far, with a warning.
    """
    ctx = ctx or Context.current()
    options = options or Options()
    run = _Run(ctx, options)
    start = time.perf_counter()
    try:
        if options.defaults:
            run.stores()
        run.paths()
        # Naming providers or paths says where to look; the sweep is for
        # the question "what is on this machine".
        wanted = not options.paths and options.only is None
        sweep = options.sweep if options.sweep is not None else wanted
        if sweep and options.defaults:
            run.sweep()
    except KeyboardInterrupt:
        ctx.warn("", "interrupted: the results are partial")
    # A file named on the command line is found again by the sweep. Keep the
    # first record of each (provider, path); the ledger charged it already.
    seen: set[tuple[str, str]] = set()
    records = []
    for record in run.records:
        key = (record.provider, os.path.realpath(record.path))
        if key not in seen:
            seen.add(key)
            records.append(record)
    finish(records)
    return Scan(
        records=records,
        roots=run.roots,
        warnings=ctx.warnings,
        elapsed_s=round(time.perf_counter() - start, 3),
        hostname=socket.gethostname(),
        tool_version=__version__,
    )


def redact(scan: Scan, home: str | os.PathLike[str]) -> Scan:
    """Remove what identifies the machine and its user, in place.

    The home directory becomes ``~`` wherever it appears: in paths, in
    warning and issue text, and in the names of loose records, which start
    with the name of their parent directory. The host name is dropped. Use
    before sending an inventory to someone else.

    :param scan: The scan to change.
    :param home: The home directory to hide, as configured and as resolved.
    :return: The same scan.
    """
    prefixes = sorted({str(home), os.path.realpath(home)}, key=len, reverse=True)
    users = {os.path.basename(prefix) for prefix in prefixes}

    def hide(text: str) -> str:
        for prefix in prefixes:
            text = text.replace(prefix, "~")
        return text

    def hide_name(text: str) -> str:
        # "user/model", or an owner such as "generic:user/model".
        owner, colon, name = text.rpartition(":") if ":" in text.split("/")[0] else ("", "", text)
        first, slash, rest = name.partition("/")
        if slash and first in users:
            name = f"~/{rest}"
        return hide(f"{owner}{colon}{name}")

    scan.hostname = None
    for record in scan.records:
        record.path = hide(record.path)
        for entry in record.file_list:
            entry.path = hide(entry.path)
            entry.used_by = [hide_name(user) for user in entry.used_by]
        record.name = hide_name(record.name)
        record.aliases = [hide_name(a) for a in record.aliases]
        record.shared_with = [hide_name(o) for o in record.shared_with]
        record.issues = [hide(i) for i in record.issues]
    for root in scan.roots:
        root.path, root.resolved = hide(root.path), hide(root.resolved)
    for warning in scan.warnings:
        warning.path, warning.message = hide(warning.path), hide(warning.message)
    return scan
