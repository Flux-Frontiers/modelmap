"""Removing a record: what would go, and doing it once confirmed.

Nothing leaves the disk that the plan did not list. A plan is one of:

- ``tool``: the store's own tool removes the record. In Ollama and the
  Hugging Face cache the model is the unit of deletion, and deleting a blob
  by hand leaves manifests or links pointing at nothing. The syntax was read
  from ``ollama rm --help`` (Ollama 0.40) and ``hf cache rm --help`` (hf
  2.4.0), and the ``hf`` forms were run with ``--dry-run`` on a real cache.
- ``trash``: the record's files, or its folder, move to the Trash. A folder
  moves whole only when every entry in it is one of the record's files;
  otherwise the listed files move one by one and the rest stays.
- ``refuse``: nothing is done, and the note says why.

Paths to move are built from the record's folder and each file's label,
never from a resolved path: a file reached through a symlink must not take
the other store's file with it. A record whose files are used by another
record is refused for the same reason.

``ollama rm`` has no store argument; it acts on the running Ollama. It is
offered only for records in the Ollama provider's own store, and only for
tags the running Ollama lists.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field

from modelmap.providers import ollama
from modelmap.records import ModelRecord

_HUB_TYPES = {"models": "model", "datasets": "dataset", "spaces": "space"}
TRASH_NOTE = "Moved to the Trash: the space is freed when the Trash is emptied."


@dataclass
class Plan:
    """What removing a record would do.

    :param record: The record.
    :param action: ``tool``, ``trash`` or ``refuse``.
    :param note: One sentence for a person: what happens, or why not.
    :param argv: The command a ``tool`` plan runs.
    :param paths: The files or folder a ``trash`` plan moves.
    :param variants: Ollama tags removed with the record that share its
        weights but differ in their settings; some may have been made with
        ``ollama create``, and those settings cannot be pulled again.
    """

    record: ModelRecord
    action: str
    note: str
    argv: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)


def _refuse(record: ModelRecord, note: str) -> Plan:
    return Plan(record, "refuse", note)


def _shared_store(path: str) -> str | None:
    """The hub cache a shared-blob path belongs to, or None."""
    parts = path.split(os.sep)
    if "blobs" in parts[-3:] and not any("--" in part for part in parts[-4:]):
        return os.sep.join(parts[: len(parts) - parts[::-1].index("blobs") - 1])
    return None


def _which(command: str, env: Mapping[str, str] | None) -> str | None:
    """A command on the ``PATH`` of ``env``; never the process's own PATH."""
    path = (env or {}).get("PATH")
    return shutil.which(command, path=path) if path else None


def trash_command(env: Mapping[str, str] | None) -> list[str] | None:
    """The command that moves files to the Trash, or None if there is none.

    macOS has ``/usr/bin/trash``; on Linux the ``trash`` of trash-cli or
    ``gio trash`` from GLib.
    """
    found = _which("trash", env)
    if found:
        return [found]
    gio = _which("gio", env)
    return [gio, "trash"] if gio else None


def _ollama_tags(env: Mapping[str, str] | None) -> set[str] | None:
    """Tags the running Ollama lists, or None if it cannot be asked."""
    ollama = _which("ollama", env)
    if not ollama:
        return None
    try:
        result = subprocess.run(
            [ollama, "list"], env=dict(env or {}), capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()[1:]
    return {line.split()[0] for line in lines if line.strip()}


def _inside(path: str, folder: str) -> bool:
    return os.path.commonpath([path, folder]) == folder


def _unlisted(folder: str, listed: set[str]) -> str | None:
    """The first entry under ``folder`` that is not a listed file, or None."""
    blocked: list[str] = []

    def note_error(error: OSError) -> None:
        blocked.append(str(error.filename))

    for dirpath, dirnames, filenames in os.walk(folder, onerror=note_error):
        dirnames.sort()
        for name in dirnames:
            if os.path.islink(os.path.join(dirpath, name)):
                return os.path.join(dirpath, name)  # a link to a folder: not walked
        for name in sorted(filenames):
            if os.path.join(dirpath, name) not in listed:
                return os.path.join(dirpath, name)
        if blocked:
            return blocked[0]
    return blocked[0] if blocked else None


def _is_model_folder(record: ModelRecord) -> bool:
    """A record that is a whole model folder, not a group of loose files.

    A model folder's own files include its ``config.json`` or
    ``model_index.json``; a group of loose shards never does.
    """
    return any(e.label in ("config.json", "model_index.json") for e in record.file_list)


def _trash_paths(record: ModelRecord, home: str) -> tuple[list[str], str]:
    """The paths a ``trash`` plan moves, or an empty list and the reason.

    A model folder moves whole or not at all: moving its listed files out of
    a folder that holds more (a git checkout, code) would leave a broken
    working tree. Loose files move one by one.
    """
    path = record.path
    is_folder = os.path.isdir(path) and not os.path.islink(path)
    base = path if is_folder else os.path.dirname(path)
    if is_folder and base in (os.sep, home, os.path.realpath(home)):
        return [], "The record's folder is the home folder or the disk's root."
    listed = []
    for entry in record.file_list:
        target = os.path.normpath(os.path.join(base, entry.label))
        if not _inside(target, base) or target == base:
            return [], f"The file {entry.label} is not inside {base}."
        if not os.path.lexists(target):
            return [], f"The file {target} is no longer there. Scan again."
        if os.path.isdir(target) and not os.path.islink(target):
            return [], f"{target} is a folder, not one of the record's files."
        listed.append(target)
    if is_folder and _is_model_folder(record):
        extra = _unlisted(base, set(listed))
        if extra:
            return [], (
                f"The folder also holds {extra}, which is not part of the model. Move the "
                "model's files yourself, or the folder if you mean all of it."
            )
        return [base], ""
    return listed, ""


def plan(
    record: ModelRecord,
    ollama_stores: Collection[str] = (),
    *,
    env: Mapping[str, str] | None = None,
    check_disk: bool = True,
    home: str = "~",
) -> Plan:
    """Work out how a record would be removed.

    :param record: The record.
    :param ollama_stores: Resolved directories of the running Ollama's
        stores, as the Ollama provider found them.
    :param env: Environment whose ``PATH`` is searched for ``hf``,
        ``ollama`` and the Trash command.
    :param check_disk: Look at the disk and ask the tools. False to describe
        the plan only, as in a page made from an inventory.
    :param home: The home folder of the machine scanned; never moved whole.
    """
    name, path = record.name, record.path
    # A store's own tool handles sharing inside the store. A user in another
    # store (a loose file linked into this one) would be left broken.
    outside = sorted(
        {
            user
            for entry in record.file_list
            for user in entry.used_by
            if user.partition(":")[0] != record.provider
        }
    )
    if record.kind == "cache":
        return _refuse(record, "A cache: the tool that owns it rebuilds it when needed.")
    hub = _shared_store(path) if record.provider in ("hf", "llamacpp") else None
    if hub:
        prune = shlex.join(["hf", "cache", "prune", "--cache-dir", hub])
        return _refuse(
            record,
            f"No repository refers to this. `{prune}` removes it, together with everything "
            "else in the cache that nothing refers to; run it with --dry-run first.",
        )
    if record.kind == "unattributed":
        return _refuse(record, "Nothing refers to these files. modelmap does not remove them.")

    base = os.path.basename(path)
    repo_type = _HUB_TYPES.get(base.partition("--")[0]) if "--" in base else None
    if outside and (repo_type or record.provider == "ollama"):
        return _refuse(
            record,
            f"{', '.join(outside)} {'uses' if len(outside) == 1 else 'use'} files of this one "
            "through a link, and would break.",
        )
    if repo_type:
        argv = ["hf", "cache", "rm", f"{repo_type}/{name}", "--cache-dir", os.path.dirname(path)]
        if check_disk and not _which("hf", env):
            return _refuse(record, "The hf command is needed to remove this, and it is not found.")
        return Plan(
            record,
            "tool",
            f"Runs `{shlex.join(argv)}`: removes the repository from the Hugging Face cache.",
            [*argv, "--yes"],
        )

    if record.provider == "ollama":
        live = any(path == s or path.startswith(s + os.sep) for s in ollama_stores)
        if not live:
            return _refuse(
                record,
                "This Ollama store is not the one the Ollama on this machine uses, and "
                "`ollama rm` acts only on that one. Delete the store's folder yourself "
                "if you no longer need it.",
            )
        tags = [name, *record.aliases]
        argv = ["ollama", "rm", *tags]
        own: list[str] = []
        if check_disk:
            listed = _ollama_tags(env)
            if listed is None:
                return _refuse(record, "Ollama is not running, or the ollama command is not found.")
            missing = [tag for tag in tags if tag not in listed]
            if missing:
                return _refuse(
                    record,
                    f"The running Ollama does not list {', '.join(missing)}. It may use another "
                    "store than the one scanned.",
                )
            store = next(s for s in ollama_stores if path == s or path.startswith(s + os.sep))
            own = [tag for tag in tags if tag in ollama.variants(store, name) or tag == name]
            own = own if len(own) > 1 else []
        what = "every tag of these weights" if record.aliases else "the model"
        note = f"Runs `{shlex.join(argv)}`: removes {what} from Ollama."
        if own:
            note += (
                f" These tags share the weights but differ in their settings: {', '.join(own)}. "
                "Settings made with `ollama create` are lost and cannot be pulled again."
            )
        return Plan(record, "tool", note, argv, variants=own)

    shared = sorted({user for entry in record.file_list for user in entry.used_by})
    if shared and not any(entry.charged for entry in record.file_list):
        return _refuse(
            record,
            f"This is another name for files of {', '.join(shared)}: removing it would free "
            "nothing. Remove that instead.",
        )
    if shared:
        return _refuse(
            record,
            f"Some of its files are also used by {', '.join(shared)}. Removing them would "
            "break those too.",
        )
    if not record.file_list:
        return _refuse(record, "The record has no files.")
    if not check_disk:
        return Plan(record, "trash", "Moves its files to the Trash.")
    if not trash_command(env):
        return _refuse(record, "No command to move files to the Trash was found.")
    paths, reason = _trash_paths(record, os.path.expanduser(home))
    if not paths:
        return _refuse(record, reason)
    if paths == [path] and os.path.isdir(path):
        note = f"Moves the folder, with its {len(record.file_list)} files, to the Trash."
    else:
        count = len(paths)
        note = f"Moves {'the file' if count == 1 else f'{count} files'} to the Trash."
    return Plan(record, "trash", note, paths=paths)


def find(records: Sequence[ModelRecord], target: str) -> list[ModelRecord]:
    """The records a target names: ``provider:name``, a path, or a name.

    An Ollama tag names the record it is an alias of. Several matches mean
    the target is ambiguous.
    """
    exact = [
        r
        for r in records
        if target in (f"{r.provider}:{r.name}", *(f"{r.provider}:{a}" for a in r.aliases))
    ]
    if exact:
        return exact
    if os.path.lexists(os.path.expanduser(target)):
        wanted = os.path.realpath(os.path.expanduser(target))
        by_path = [r for r in records if os.path.realpath(r.path) == wanted]
        if by_path:
            return by_path
    return [r for r in records if target == r.name or target in r.aliases]


def target_of(record: ModelRecord, records: Sequence[ModelRecord]) -> str:
    """The shortest target that names this record and no other."""
    label = f"{record.provider}:{record.name}"
    clashes = [r for r in records if r is not record and find([r], label)]
    return record.path if clashes else label


def execute(plan: Plan, env: Mapping[str, str] | None) -> tuple[bool, str]:
    """Carry out a ``tool`` or ``trash`` plan and check that it worked.

    :return: Whether the record's files are gone, and what went wrong if not.
    """
    if plan.action == "tool":
        command = [_which(plan.argv[0], env) or plan.argv[0], *plan.argv[1:]]
    elif plan.action == "trash":
        trash = trash_command(env)
        if not trash:
            return False, "No command to move files to the Trash was found."
        command = [*trash, *plan.paths]
    else:
        return False, plan.note
    try:
        result = subprocess.run(
            command, env=dict(env or {}), capture_output=True, text=True, timeout=600
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    output = (result.stderr or result.stdout).strip().splitlines()
    detail = output[-1] if output else f"exit status {result.returncode}"
    if result.returncode != 0:
        return False, detail

    if plan.action == "trash":
        left = [path for path in plan.paths if os.path.lexists(path)]
        return (not left, f"Still there: {', '.join(left)}" if left else "")
    if plan.argv[0] == "ollama":
        tags = _ollama_tags(env) or set()
        left = [tag for tag in plan.argv[2:] if tag in tags]
        return (not left, f"Ollama still lists {', '.join(left)}" if left else "")
    gone = not os.path.lexists(plan.record.path)
    return (gone, "" if gone else f"Still there: {plan.record.path}")
