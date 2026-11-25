"""File sizes that count each file once.

Two numbers per file. Apparent size (``st_size``) is what the model is.
Disk size (allocated blocks) is what it costs. They differ for sparse
files. Tools that share one file between names, as the Hugging Face cache
does with symlinks and Ollama does with layers used by several tags, would
be counted twice by a walk that adds up names; :class:`SizeLedger` charges
each inode to the first owner that claims it.

Copy-on-write clones (APFS, Btrfs, XFS reflinks) have separate inodes that
share blocks. That sharing cannot be seen through ``stat``, so disk totals
overstate on volumes that use clones.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class FileSize:
    """Sizes and identity of one file.

    :param apparent: ``st_size`` in bytes.
    :param disk: Bytes allocated on disk.
    :param key: ``(st_dev, st_ino)``, the same for every name of one file.
    :param mtime: Modification time as a POSIX time.
    """

    apparent: int
    disk: int
    key: tuple[int, int]
    mtime: float = 0.0


@dataclass(frozen=True)
class Charge:
    """The result of charging a file to an owner.

    :param apparent: Apparent bytes charged; 0 when already charged.
    :param disk: Disk bytes charged; 0 when already charged.
    :param shared_with: The owner charged earlier, or None on first charge.
    :param file: The file's full sizes, whoever was charged for it.
    """

    apparent: int
    disk: int
    shared_with: str | None = None
    file: FileSize | None = None


def stat_size(path: str | os.PathLike[str]) -> FileSize:
    """Stat a file, following symlinks.

    :param path: File to measure.
    :return: Its sizes and inode identity.
    :raises OSError: If the path cannot be read or is a broken link.
    """
    st = os.stat(path)
    # st_blocks counts 512-byte units on every POSIX system. Windows has no
    # st_blocks, so the apparent size stands in.
    blocks = getattr(st, "st_blocks", None)
    disk = blocks * 512 if blocks is not None else st.st_size
    return FileSize(apparent=st.st_size, disk=disk, key=(st.st_dev, st.st_ino), mtime=st.st_mtime)


class SizeLedger:
    """Charges each file to one owner, however many names point at it.

    Use one ledger for a whole scan so a file reachable from two providers
    is counted once in the grand total.
    """

    def __init__(self) -> None:
        self._owners: dict[tuple[int, int], str] = {}

    def owner(self, path: str | os.PathLike[str]) -> str | None:
        """Who was charged for a file, or None if nobody was or it is unreadable."""
        try:
            return self._owners.get(stat_size(path).key)
        except OSError:
            return None

    def charge(self, path: str | os.PathLike[str], owner: str) -> Charge:
        """Charge a file to ``owner`` unless another owner already has it.

        A second charge by the same owner costs nothing and reports no
        sharing.

        :param path: File to charge.
        :param owner: Name of the record claiming the file.
        :return: The bytes charged, and who holds the file if not ``owner``.
        :raises OSError: If the path cannot be read.
        """
        size = stat_size(path)
        first = self._owners.get(size.key)
        if first is None:
            self._owners[size.key] = owner
            return Charge(size.apparent, size.disk, file=size)
        if first == owner:
            return Charge(0, 0, file=size)
        return Charge(0, 0, shared_with=first, file=size)
