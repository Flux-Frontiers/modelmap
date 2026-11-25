"""Build a fake home directory full of model files.

Every model file has a real GGUF or safetensors header and a sparse
payload, so a "70 GB" model costs a few kilobytes on disk and the whole
tree builds in well under a second. Tests run real modelmap code against
this tree; nothing is mocked.

The layout mirrors what was observed on real machines:

- a Hugging Face hub cache with two revisions that share a blob,
- an Ollama store outside the home directory, reached through a symlink,
  with two tags that share a weights layer,
- an oMLX model directory with a packed 4-bit model and a sharded model,
- LM Studio, llama.cpp and loose files,
- hostile cases: unreadable directory, symlink loop, empty, truncated and
  lying files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GB = 1000**3
MB = 1000**2

# GGML tensor types used by the fixtures, and their bytes per element.
GGML_F32 = 0
GGML_F16 = 1
_GGML_BYTES = {GGML_F32: 4, GGML_F16: 2}
_GGUF_ALIGNMENT = 32

_ST_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "U32": 4, "I64": 8, "U8": 1}

Tensor = tuple[str, tuple[int, ...], int]


def _gguf_string(text: str, endian: str) -> bytes:
    raw = text.encode()
    return struct.pack(endian + "Q", len(raw)) + raw


def _gguf_value(value: Any, endian: str) -> bytes:
    """Encode a Python value as a GGUF type code followed by the value."""
    if isinstance(value, bool):
        return struct.pack(endian + "I?", 7, value)
    if isinstance(value, int):
        return struct.pack(endian + "II", 4, value)
    if isinstance(value, float):
        return struct.pack(endian + "If", 6, value)
    if isinstance(value, str):
        return struct.pack(endian + "I", 8) + _gguf_string(value, endian)
    if isinstance(value, list):
        head = struct.pack(endian + "I", 9)
        if all(isinstance(v, str) for v in value):
            body = b"".join(_gguf_string(v, endian) for v in value)
            return head + struct.pack(endian + "IQ", 8, len(value)) + body
        if all(isinstance(v, int) for v in value):
            body = struct.pack(f"{endian}{len(value)}i", *value)
            return head + struct.pack(endian + "IQ", 5, len(value)) + body
        if all(isinstance(v, float) for v in value):
            body = struct.pack(f"{endian}{len(value)}f", *value)
            return head + struct.pack(endian + "IQ", 6, len(value)) + body
    raise TypeError(f"no GGUF encoding for {value!r}")


def gguf_header(
    *,
    tensors: tuple[Tensor, ...] = (),
    metadata: dict[str, Any] | None = None,
    version: int = 3,
    endian: str = "<",
    tensor_count: int | None = None,
    kv_count: int | None = None,
) -> bytes:
    """Return a GGUF header, padded to the data alignment.

    :param tensors: ``(name, dims, ggml_type)`` for each tensor.
    :param metadata: Key-value pairs; str, bool, int, float and flat lists.
    :param version: GGUF version to declare.
    :param endian: ``"<"`` or ``">"``.
    :param tensor_count: Override the declared tensor count, to lie.
    :param kv_count: Override the declared metadata count, to lie.
    """
    metadata = metadata or {}
    out = b"GGUF" + struct.pack(
        endian + "IQQ",
        version,
        len(tensors) if tensor_count is None else tensor_count,
        len(metadata) if kv_count is None else kv_count,
    )
    for key, value in metadata.items():
        out += _gguf_string(key, endian) + _gguf_value(value, endian)
    offset = 0
    for name, dims, ggml_type in tensors:
        out += _gguf_string(name, endian)
        out += struct.pack(f"{endian}I{len(dims)}Q", len(dims), *dims)
        out += struct.pack(endian + "IQ", ggml_type, offset)
        size = math.prod(dims) * _GGML_BYTES[ggml_type]
        offset += -(-size // _GGUF_ALIGNMENT) * _GGUF_ALIGNMENT
    return out + b"\0" * (-len(out) % _GGUF_ALIGNMENT)


def write_gguf(path: Path, *, payload: int | None = None, **header: Any) -> int:
    """Write a GGUF file with a sparse payload and return its size.

    :param path: File to create; parent directories are made.
    :param payload: Payload size in bytes. Default: what the tensors need.
    :param header: Arguments for :func:`gguf_header`.
    """
    head = gguf_header(**header)
    if payload is None:
        payload = sum(
            math.prod(dims) * _GGML_BYTES[ggml_type]
            for _, dims, ggml_type in header.get("tensors", ())
        )
    return _write_sparse(path, head, len(head) + payload)


def safetensors_header(
    tensors: dict[str, tuple[str, tuple[int, ...]]], metadata: dict[str, str] | None = None
) -> tuple[bytes, int]:
    """Return a safetensors length prefix plus header, and the payload size.

    :param tensors: ``{name: (dtype, shape)}``.
    :param metadata: Optional ``__metadata__`` object.
    """
    table: dict[str, Any] = {}
    if metadata is not None:
        table["__metadata__"] = metadata
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = math.prod(shape) * _ST_BYTES[dtype]
        table[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(table, separators=(",", ":")).encode()
    return struct.pack("<Q", len(raw)) + raw, offset


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...]]],
    metadata: dict[str, str] | None = None,
) -> int:
    """Write a safetensors file with a sparse payload and return its size."""
    head, payload = safetensors_header(tensors, metadata)
    return _write_sparse(path, head, len(head) + payload)


def _write_sparse(path: Path, head: bytes, size: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(head)
        fh.truncate(size)
    return size


def _write(path: Path, data: bytes | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        data = data.encode()
    path.write_bytes(data)
    return path


def _link(link: Path, target: str | Path) -> Path:
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, link)
    return link


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@dataclass
class FakeMachine:
    """Paths into a built fixture tree.

    :param root: Directory everything lives under.
    :param home: The fake home directory.
    :param hf_hub: Hugging Face hub cache.
    :param ollama_store: The real Ollama store, outside ``home``.
    :param ollama_link: ``~/.ollama/models``, a symlink to the store.
    :param omlx_models: oMLX model directory.
    :param hostile: Directory of files and links meant to break a scanner.
    :param files: Named files that tests refer to.
    :param skipped: Hostile cases this filesystem or user could not create.
    """

    root: Path
    home: Path
    hf_hub: Path
    ollama_store: Path
    ollama_link: Path
    omlx_models: Path
    hostile: Path
    files: dict[str, Path] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)

    def cleanup(self) -> None:
        """Make the unreadable directory removable again."""
        denied = self.hostile / "denied"
        if denied.exists():
            denied.chmod(0o755)


def _build_hf(m: FakeMachine) -> None:
    tiny = m.hf_hub / "models--acme--tiny"
    weights = tiny / "blobs" / _digest("tiny weights")
    write_safetensors(weights, {"w": ("F16", (1000, 1000, 1000))})  # 2 GB
    m.files["hf_weights"] = weights
    for rev, label in (("rev1", "tiny config 1"), ("rev2", "tiny config 2")):
        config = _write(tiny / "blobs" / _digest(label), json.dumps({"rev": rev}))
        snapshot = tiny / "snapshots" / rev
        _link(snapshot / "model.safetensors", f"../../blobs/{weights.name}")
        _link(snapshot / "config.json", f"../../blobs/{config.name}")
        m.files[f"hf_config_{rev}"] = config
        m.files[f"hf_snapshot_{rev}"] = snapshot
    _link(tiny / "snapshots" / "rev2" / "broken.bin", "../../blobs/missing")
    _write(tiny / "refs" / "main", "rev2")

    big = m.hf_hub / "models--acme--big-GGUF"
    blob = big / "blobs" / _digest("big gguf")
    write_gguf(
        blob,
        tensors=(("blk.0.weight", (35 * 1000, 1000 * 1000), GGML_F16),),  # 70 GB
        metadata={"general.architecture": "llama", "general.name": "Big", "general.file_type": 1},
    )
    _link(big / "snapshots" / "r1" / "big.F16.gguf", f"../../blobs/{blob.name}")
    m.files["hf_big_gguf"] = blob

    data = m.hf_hub / "datasets--acme--words"
    blob = _write(data / "blobs" / _digest("words"), b"word\n" * 2000)
    _link(data / "snapshots" / "r1" / "words.txt", f"../../blobs/{blob.name}")

    # Seen on a real machine: blobs at the top of the hub, in no repo.
    m.files["hf_unattributed"] = _write(m.hf_hub / "blobs" / "06" / _digest("stray"), b"\0" * 4096)


def _build_ollama(m: FakeMachine) -> None:
    blobs = m.ollama_store / "blobs"

    def blob(label: str, data: bytes | None = None, **gguf: Any) -> dict[str, Any]:
        digest = _digest(label)
        path = blobs / f"sha256-{digest}"
        if data is None:
            write_gguf(path, **gguf)
        else:
            _write(path, data)
        m.files[f"ollama_{label}"] = path
        return {"digest": f"sha256:{digest}", "size": path.stat().st_size}

    weights = blob(
        "weights",
        # 5e9 parameters, labeled Q4_K_M, in a 5 GB file.
        tensors=(("blk.0.weight", (5000, 1000 * 1000), GGML_F32),),
        metadata={"general.architecture": "qwen3", "general.file_type": 15},
        payload=5 * GB,
    )
    template = blob("template", b"{{ .Prompt }}")
    params_a = blob("params_a", b'{"num_ctx":2048}')
    params_b = blob("params_b", b'{"num_ctx":4096}')
    blob("orphan", tensors=(("w", (1000, 1000), GGML_F32),), payload=1 * GB)

    def manifest(relpath: str, layers: list[tuple[str, dict[str, Any]]]) -> None:
        body = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "layers": [
                {"mediaType": f"application/vnd.ollama.image.{kind}", **layer}
                for kind, layer in layers
            ],
        }
        m.files[f"ollama_manifest_{relpath.rsplit('/', 1)[-1]}"] = _write(
            m.ollama_store / "manifests" / relpath, json.dumps(body)
        )

    library = "registry.ollama.ai/library/tiny"
    manifest(f"{library}/1b", [("model", weights), ("template", template), ("params", params_a)])
    manifest(f"{library}/1b-4k", [("model", weights), ("template", template), ("params", params_b)])
    manifest(
        "hf.co/acme/gone-GGUF/Q4_K_M",
        [("model", {"digest": f"sha256:{_digest('never downloaded')}", "size": 3 * GB})],
    )
    _link(m.ollama_link, m.ollama_store)


def _build_omlx(m: FakeMachine) -> None:
    omlx = m.omlx_models.parent
    _write(
        omlx / "settings.json",
        json.dumps(
            {"model": {"model_dirs": [str(m.omlx_models)], "model_dir": str(m.omlx_models)}}
        ),
    )

    # 4-bit MLX model: a 64x64 weight packed into U32 as 64x8.
    packed = m.omlx_models / "mlx-community" / "Tiny-4bit"
    _write(packed / "config.json", json.dumps({"quantization": {"group_size": 64, "bits": 4}}))
    m.files["mlx_packed"] = packed / "model.safetensors"
    write_safetensors(
        m.files["mlx_packed"],
        {
            "layers.0.proj.weight": ("U32", (64, 8)),
            "layers.0.proj.scales": ("BF16", (64, 1)),
            "layers.0.proj.biases": ("BF16", (64, 1)),
            "layers.0.proj.bias": ("BF16", (64,)),
            "layers.0.norm.weight": ("BF16", (64,)),
        },
        metadata={"format": "mlx"},
    )

    sharded = m.omlx_models / "acme" / "Sharded"
    _write(sharded / "config.json", json.dumps({"model_type": "tiny"}))
    for index, name in enumerate(("a", "b"), start=1):
        shard = sharded / f"model-0000{index}-of-00002.safetensors"
        write_safetensors(shard, {f"{name}.weight": ("F16", (2000, 1000, 1000))})  # 4 GB
        m.files[f"mlx_shard_{index}"] = shard


def _build_loose(m: FakeMachine) -> None:
    home = m.home
    lmstudio = home / ".lmstudio" / "models" / "pub" / "vision"
    m.files["lmstudio_model"] = lmstudio / "vision-Q4_K_M.gguf"
    write_gguf(
        m.files["lmstudio_model"],
        tensors=(("w", (4000, 1000 * 1000), GGML_F32),),
        metadata={"general.architecture": "llava", "general.file_type": 15},
        payload=2 * GB,
    )
    m.files["mmproj"] = lmstudio / "mmproj-vision-f16.gguf"
    write_gguf(
        m.files["mmproj"],
        tensors=(("v.proj", (1000, 1000), GGML_F16),),
        metadata={"general.architecture": "clip", "general.file_type": 1},
    )
    m.files["llamacpp"] = home / "Library" / "Caches" / "llama.cpp" / "acme_tiny-Q8_0.gguf"
    write_gguf(
        m.files["llamacpp"],
        tensors=(("w", (1000, 1000), GGML_F32),),
        metadata={"general.file_type": 7},
        payload=1 * GB,
    )
    m.files["loose"] = home / "models" / "loose.gguf"
    write_gguf(m.files["loose"], tensors=(("w", (500, 1000 * 1000), GGML_F16),))  # 1 GB
    # Below the 50 MB floor that applies to unverifiable files only.
    m.files["small"] = home / ".kgrag" / "small-Q8_0.gguf"
    write_gguf(
        m.files["small"],
        tensors=(("w", (33, 1000 * 1000), GGML_F32),),
        metadata={"general.architecture": "bert", "general.file_type": 7},
        payload=37 * MB,
    )
    m.files["vocab"] = home / ".unsloth" / "llama.cpp" / "models" / "ggml-vocab-test.gguf"
    write_gguf(
        m.files["vocab"],
        metadata={
            "general.architecture": "llama",
            "tokenizer.ggml.tokens": ["<s>", "</s>", "hello", "world"],
            "tokenizer.ggml.scores": [0.0, 0.0, -1.0, -2.0],
            "tokenizer.ggml.token_type": [3, 3, 1, 1],
        },
    )
    m.files["pickle"] = home / "ckpt" / "model.pt"
    _write_sparse(m.files["pickle"], b"PK\x03\x04", 100 * MB)


def _build_hostile(m: FakeMachine) -> None:
    hostile = m.hostile
    good = gguf_header(tensors=(("w", (4, 4), GGML_F16),), metadata={"general.name": "x"})

    m.files["empty"] = _write(hostile / "empty.gguf", b"")
    m.files["truncated"] = _write(hostile / "truncated.gguf", good[:20])
    m.files["lying_count"] = _write(hostile / "lying-count.gguf", gguf_header(tensor_count=2**60))
    # Declares a 10,000-byte header in a 58-byte file.
    m.files["lying_length"] = _write(
        hostile / "lying.safetensors", struct.pack("<Q", 10_000) + b"{" + b" " * 49
    )
    m.files["huge_length"] = _write(
        hostile / "huge.safetensors", struct.pack("<Q", 2**62) + b'{"a":1}'
    )
    _link(hostile / "loop", ".")

    denied = hostile / "denied"
    write_gguf(denied / "hidden.gguf", tensors=(("w", (4, 4), GGML_F16),))
    denied.chmod(0o000)
    if os.access(denied, os.R_OK):
        m.skipped["denied"] = "running as a user that ignores directory permissions"

    try:
        (hostile / "caf\udcff.gguf").write_bytes(good)
    except OSError as exc:
        # APFS refuses names that are not valid UTF-8.
        m.skipped["non_utf8_name"] = f"{sys.platform} filesystem refused the name: {exc}"


def build(root: Path) -> FakeMachine:
    """Build the fixture tree under ``root`` and return its paths.

    :param root: An empty directory, normally pytest's ``tmp_path``.
    """
    home = root / "home"
    m = FakeMachine(
        root=root,
        home=home,
        hf_hub=home / ".cache" / "huggingface" / "hub",
        ollama_store=root / "shared" / "ollama",
        ollama_link=home / ".ollama" / "models",
        omlx_models=home / ".omlx" / "models",
        hostile=home / "hostile",
    )
    _build_hf(m)
    _build_ollama(m)
    _build_omlx(m)
    _build_loose(m)
    _build_hostile(m)
    return m
