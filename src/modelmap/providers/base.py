"""What every provider shares: the scan context, roots and charging.

A provider is a module with two functions. ``roots(ctx)`` returns the
directories the tool stores models in on this machine. ``scan(root, ctx)``
returns the records found under one of them. Nothing here reads the real
environment: the home directory, environment variables and platform come
in through :class:`Context`, so tests can describe any machine.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from modelmap.records import FileEntry, ModelRecord, ScanRoot, ScanWarning
from modelmap.sizing import SizeLedger

# A file smaller than this that two records share is not worth a note:
# license and template blobs are shared by unrelated models.
SHARE_NOTE_MIN = 1_000_000

# Directory names never worth descending into.
PRUNE = frozenset(
    {
        ".git",
        "node_modules",
        "site-packages",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".Trash",
    }
)


@dataclass
class Context:
    """The machine being scanned and the state of the scan.

    :param home: The user's home directory.
    :param env: Environment variables.
    :param platform: ``sys.platform`` value: ``darwin``, ``linux``, ``win32``.
    :param ledger: Charges each file once across every provider.
    :param warnings: Paths that could not be read.
    :param uses: For each file, every record that uses it, in scan order.
    """

    home: Path
    env: Mapping[str, str]
    platform: str
    ledger: SizeLedger = field(default_factory=SizeLedger)
    warnings: list[ScanWarning] = field(default_factory=list)
    uses: dict[tuple[int, int], list[str]] = field(default_factory=dict)

    @classmethod
    def current(cls) -> Context:
        """The context of the running process."""
        return cls(home=Path.home(), env=os.environ, platform=sys.platform)

    def warn(self, path: str | os.PathLike[str], message: str) -> None:
        self.warnings.append(ScanWarning(str(path), message))

    def expand(self, value: str) -> Path:
        """Turn a configured path into a Path, expanding a leading ``~``."""
        if value == "~" or value.startswith(("~/", "~\\")):
            return self.home / value[2:]
        return Path(value)

    def first_env(self, *names: str) -> tuple[str, str] | None:
        """The first of ``names`` that is set and non-empty, with its value."""
        for name in names:
            value = self.env.get(name)
            if value:
                return name, value
        return None

    def root(self, provider: str, path: str | os.PathLike[str], source: str) -> ScanRoot:
        """Describe a root, resolving it even when it is itself a symlink."""
        resolved = os.path.realpath(path)
        return ScanRoot(
            provider=provider,
            path=str(path),
            resolved=resolved,
            source=source,
            exists=os.path.isdir(resolved),
        )


@dataclass
class Tally:
    """Sizes of one record's files as they are charged to the ledger."""

    size: int = 0
    disk: int = 0
    files: int = 0
    mtime: float | None = None
    shared_with: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    entries: list[FileEntry] = field(default_factory=list)

    def charge(
        self,
        ctx: Context,
        owner: str,
        paths: Iterable[str | os.PathLike[str]],
        labels: Mapping[str, tuple[str, str]] | None = None,
    ) -> None:
        """Charge files to ``owner``, noting sharing and unreadable files.

        :param ctx: The scan context, whose ledger decides who pays.
        :param owner: Name of the record the files belong to.
        :param paths: Files to charge. Symlinks are followed.
        :param labels: For a path, its ``(label, note)`` in the file list;
            a path not listed is labeled with its file name.
        """
        for path in paths:
            try:
                charge = ctx.ledger.charge(path, owner)
            except OSError as exc:
                kind = "broken link" if os.path.islink(path) else "unreadable"
                self.issues.append(f"{kind}: {os.path.basename(path)}")
                if kind == "unreadable":
                    ctx.warn(path, exc.strerror or str(exc))
                continue
            file = charge.file
            assert file is not None
            self.size += charge.apparent
            self.disk += charge.disk
            self.files += 1
            if self.mtime is None or file.mtime > self.mtime:
                self.mtime = file.mtime
            other = charge.shared_with
            if other and file.apparent >= SHARE_NOTE_MIN and other not in self.shared_with:
                self.shared_with.append(other)
            users = ctx.uses.setdefault(file.key, [])
            if owner not in users:
                users.append(owner)
            label, note = (labels or {}).get(str(path), (os.path.basename(path), ""))
            self.entries.append(
                FileEntry(
                    path=os.path.realpath(path),
                    label=label,
                    size_bytes=file.apparent,
                    disk_bytes=file.disk,
                    charged=other is None,
                    note=note,
                    # The shared list: complete only when the scan ends.
                    used_by=users,
                )
            )

    def record(self, issues: Iterable[str] = (), **fields: Any) -> ModelRecord:
        """Build a record from the tally plus the given fields.

        :param issues: Issues found outside charging, listed first.
        :param fields: The remaining :class:`ModelRecord` fields.
        """
        return ModelRecord(
            size_bytes=self.size,
            disk_bytes=self.disk,
            files=self.files,
            file_list=self.entries,
            mtime=self.mtime,
            shared_with=self.shared_with,
            issues=[*issues, *self.issues],
            **fields,
        )


def finish(records: Iterable[ModelRecord]) -> None:
    """Complete the file lists once every record has been scanned.

    While a scan runs, each file entry points at one shared list of the
    records using that file. Give each entry its own list, without the
    record it belongs to. A file may be the first one charged and still be
    used by a record scanned later; only now is that known.
    """
    for record in records:
        owner = f"{record.provider}:{record.name}"
        for entry in record.file_list:
            entry.used_by = [user for user in entry.used_by if user != owner]


def is_file(entry: os.DirEntry[str]) -> bool:
    """``entry.is_file()``, false when the entry cannot be examined."""
    try:
        return entry.is_file()
    except OSError:
        return False


def is_dir(entry: os.DirEntry[str], *, follow_symlinks: bool = True) -> bool:
    """``entry.is_dir()``, false when the entry cannot be examined."""
    try:
        return entry.is_dir(follow_symlinks=follow_symlinks)
    except OSError:
        return False


def file_size(path: str) -> int | None:
    """Size of a file, or None when it cannot be examined.

    A directory that may be listed but not searched yields names whose
    files cannot be stat'ed.
    """
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def listdir(ctx: Context, path: str | os.PathLike[str]) -> list[os.DirEntry[str]]:
    """Entries of a directory in name order; a warning and nothing on error."""
    try:
        with os.scandir(path) as it:
            return sorted(it, key=lambda entry: entry.name)
    except OSError as exc:
        ctx.warn(path, exc.strerror or str(exc))
        return []


def files_under(ctx: Context, path: str | os.PathLike[str]) -> list[str]:
    """Every file below a directory, in a fixed order.

    Symlinked files are included, since a store may link to its blobs;
    symlinked directories are not followed, so a loop cannot trap the walk.
    """
    found: list[str] = []
    stack = [str(path)]
    while stack:
        current = stack.pop()
        subdirs = []
        for entry in listdir(ctx, current):
            if is_dir(entry, follow_symlinks=False):
                if entry.name not in PRUNE:
                    subdirs.append(entry.path)
            elif not entry.is_symlink() or not os.path.isdir(entry.path):
                found.append(entry.path)
        stack.extend(reversed(subdirs))
    return found
