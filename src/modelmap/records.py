"""The scan result: what was found, where, and what went wrong.

A :class:`Scan` serializes to JSON so an inventory made on one machine can
be mapped on another. ``schema_version`` changes when a field is removed or
changes meaning. Adding a field does not change it: a reader ignores keys it
does not know, so an older modelmap can map a newer inventory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

SCHEMA_VERSION = 1


def _load(cls: Any, data: dict[str, Any]) -> Any:
    """Build a dataclass from a dict, ignoring keys it does not declare."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ModelRecord:
    """One model, dataset or cache entry found on disk.

    :param name: Display name, for example ``mlx-community/Qwen3-8B-4bit``.
    :param provider: Tool that owns the files: ``hf``, ``ollama``, ``omlx``,
        ``lmstudio``, ``llamacpp`` or ``generic``.
    :param format: ``gguf``, ``safetensors``, ``mlx``, ``pytorch`` and so on.
    :param path: Directory or file that holds the model.
    :param size_bytes: Apparent size, with shared files charged once.
    :param disk_bytes: Bytes allocated on disk, with shared files charged once.
    :param kind: ``model``, ``auxiliary`` (a file that cannot run alone, such
        as a multimodal projector), ``dataset``, ``cache`` or ``unattributed``.
    :param family: Model family guessed from the name, such as ``qwen``.
    :param quant: Quantization label, such as ``Q4_K_M`` or ``4bit``.
    :param params: Parameter count, or None when unknown.
    :param params_exact: False when ``params`` is an estimate.
    :param mtime: Newest modification time among the files, as a POSIX time.
    :param files: Number of files.
    :param shared_with: Names of records that share a file with this one.
    :param issues: Problems with this entry, such as a truncated header.
    """

    name: str
    provider: str
    format: str
    path: str
    size_bytes: int
    disk_bytes: int
    kind: str = "model"
    family: str | None = None
    quant: str | None = None
    params: int | None = None
    params_exact: bool = True
    mtime: float | None = None
    files: int = 1
    shared_with: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass
class ScanRoot:
    """A directory the scan visited.

    :param provider: Provider the root belongs to, or ``user`` for a path
        given on the command line.
    :param path: The path as configured, before symlinks are resolved.
    :param resolved: The real path that was walked.
    :param source: What produced the path: an environment variable name,
        ``default``, or ``argument``.
    :param exists: False when the path is absent.
    """

    provider: str
    path: str
    resolved: str
    source: str
    exists: bool = True


@dataclass
class ScanWarning:
    """Something the scan could not read or understand.

    :param path: Where it happened.
    :param message: What went wrong.
    """

    path: str
    message: str


@dataclass
class Scan:
    """The result of one scan.

    :param records: Everything found.
    :param roots: Directories visited.
    :param warnings: Paths that could not be read.
    :param elapsed_s: Wall-clock scan time in seconds.
    :param hostname: Machine name, or None when redacted.
    :param tool_version: Version of modelmap that made the scan.
    :param schema_version: Version of this JSON layout.
    """

    records: list[ModelRecord] = field(default_factory=list)
    roots: list[ScanRoot] = field(default_factory=list)
    warnings: list[ScanWarning] = field(default_factory=list)
    elapsed_s: float = 0.0
    hostname: str | None = None
    tool_version: str = ""
    schema_version: int = SCHEMA_VERSION

    @property
    def size_bytes(self) -> int:
        """Total apparent size of every record."""
        return sum(r.size_bytes for r in self.records)

    @property
    def disk_bytes(self) -> int:
        """Total allocated size of every record."""
        return sum(r.disk_bytes for r in self.records)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scan:
        """Rebuild a scan from :meth:`to_dict` output.

        :param data: Parsed JSON object.
        :raises ValueError: If the schema version is not the one this
            version of modelmap reads.
        """
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"inventory has schema version {version!r}; "
                f"this modelmap reads version {SCHEMA_VERSION}"
            )
        scan = _load(cls, data)
        scan.records = [_load(ModelRecord, r) for r in data.get("records", [])]
        scan.roots = [_load(ScanRoot, r) for r in data.get("roots", [])]
        scan.warnings = [_load(ScanWarning, w) for w in data.get("warnings", [])]
        return scan

    @classmethod
    def from_json(cls, text: str) -> Scan:
        return cls.from_dict(json.loads(text))
