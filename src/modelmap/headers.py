"""Read GGUF and safetensors headers without reading any weights.

Both readers return an info object and never raise on a bad file: a
truncated file, a header that declares more than the file holds, or an
unreadable path becomes an entry in ``issues``. A reader consumes at most
``HEADER_CAP`` bytes, in chunks, and checks every declared length and count
against the file size before acting on it.

GGUF layout and enums follow llama.cpp's ``gguf-py/gguf/constants.py`` and
``gguf_reader.py``. Safetensors layout: an 8-byte little-endian header
length, then a JSON object mapping tensor names to ``dtype``, ``shape`` and
``data_offsets``.
"""

from __future__ import annotations

import json
import math
import os
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, BinaryIO

HEADER_CAP = 64 * 1024 * 1024
_CHUNK = 1024 * 1024

GGUF_MAGIC = b"GGUF"
_GGUF_VERSIONS = (2, 3)
# GGUFValueType: struct codes for the fixed-size types.
_SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}
_STRING = 8
_ARRAY = 9
_MAX_ARRAY_DEPTH = 4
_MAX_DIMS = 8  # GGML allows 4; anything above 8 is a corrupt header

# LlamaFileType, with the MOSTLY_/ALL_ prefix dropped.
FILE_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4_MOE",
    39: "NVFP4",
    40: "Q1_0",
}

# Bit width of the unsigned safetensors dtypes MLX packs quantized weights into.
_PACKED_DTYPES = {"U8": 8, "U16": 16, "U32": 32, "U64": 64}


class _HeaderError(Exception):
    """A header that cannot be parsed; its message becomes an issue."""


@dataclass
class GGUFInfo:
    """What a GGUF header says about its file.

    :param version: GGUF version, or None when the file is too short to say.
    :param tensor_count: Declared number of tensors.
    :param metadata: Scalar and string key-value pairs. Arrays are skipped.
    :param param_count: Sum of tensor element counts, exact for any
        quantization. None unless every tensor info was read.
    :param issues: Problems found; empty for a well-formed header.
    """

    version: int | None = None
    tensor_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    param_count: int | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def architecture(self) -> str | None:
        return self.metadata.get("general.architecture")

    @property
    def name(self) -> str | None:
        return self.metadata.get("general.name")

    @property
    def quant(self) -> str | None:
        """Quantization label from ``general.file_type``, or None when absent."""
        file_type = self.metadata.get("general.file_type")
        if file_type is None:
            return None
        return FILE_TYPES.get(file_type, f"file_type {file_type}")

    @property
    def is_vocab_only(self) -> bool:
        """True for a tokenizer-only stub such as llama.cpp's ggml-vocab files."""
        return self.tensor_count == 0


@dataclass
class SafetensorsInfo:
    """What a safetensors header says about its file.

    :param tensor_count: Number of tensors in the header.
    :param metadata: The ``__metadata__`` object, if any.
    :param param_count: Parameter count, or None when it cannot be known
        (a packed weight with no bit width given, or a bad header).
    :param params_exact: False when the count was derived from packed
        quantized weights and their configured bit widths.
    :param packed: True when the file holds MLX-style packed integer weights.
    :param issues: Problems found; empty for a well-formed header.
    """

    tensor_count: int = 0
    metadata: dict[str, str] = field(default_factory=dict)
    param_count: int | None = None
    params_exact: bool = True
    packed: bool = False
    issues: list[str] = field(default_factory=list)


class _Reader:
    """Chunked forward reader that refuses to pass the file size or the cap."""

    def __init__(self, fh: BinaryIO, size: int, cap: int) -> None:
        self._fh = fh
        self._data = bytearray()
        self._size = size
        self._cap = cap
        self.pos = 0
        self.set_endian("<")

    def set_endian(self, endian: str) -> None:
        self.endian = endian
        self._u32 = struct.Struct(endian + "I")
        self._u64 = struct.Struct(endian + "Q")

    @property
    def remaining(self) -> int:
        """Bytes between the read position and the end of the file."""
        return self._size - self.pos

    def _need(self, n: int) -> None:
        end = self.pos + n
        if end <= len(self._data):
            return
        if end > self._size:
            raise _HeaderError(
                f"truncated: header needs {n} bytes at offset {self.pos}, "
                f"file is {self._size} bytes"
            )
        if end > self._cap:
            raise _HeaderError(f"header is larger than the {self._cap} byte cap")
        limit = min(self._size, self._cap)
        while len(self._data) < end:
            want = min(max(end - len(self._data), _CHUNK), limit - len(self._data))
            chunk = self._fh.read(want)
            if not chunk:
                raise _HeaderError(f"truncated: file ended at offset {len(self._data)}")
            self._data += chunk

    def u32(self) -> int:
        self._need(4)
        (value,) = self._u32.unpack_from(self._data, self.pos)
        self.pos += 4
        return value

    def u64(self) -> int:
        self._need(8)
        (value,) = self._u64.unpack_from(self._data, self.pos)
        self.pos += 8
        return value

    def scalar(self, code: str) -> Any:
        fmt = struct.Struct(self.endian + code)
        self._need(fmt.size)
        (value,) = fmt.unpack_from(self._data, self.pos)
        self.pos += fmt.size
        return value

    def skip(self, n: int) -> None:
        self._need(n)
        self.pos += n

    def string(self) -> str:
        n = self.u64()
        self._need(n)
        raw = bytes(self._data[self.pos : self.pos + n])
        self.pos += n
        return raw.decode("utf-8", "replace")

    def check_count(self, count: int, min_item_bytes: int, what: str) -> None:
        """Reject a declared count the rest of the file cannot hold."""
        if count * min_item_bytes > self.remaining:
            raise _HeaderError(
                f"declared {count} {what}, more than the remaining {self.remaining} bytes can hold"
            )


def _skip_array(r: _Reader, depth: int) -> None:
    item_type = r.u32()
    count = r.u64()
    if item_type in _SCALARS:
        r.skip(count * struct.calcsize(_SCALARS[item_type]))
    elif item_type == _STRING:
        r.check_count(count, 8, "array strings")
        for _ in range(count):
            r.skip(r.u64())
    elif item_type == _ARRAY:
        if depth >= _MAX_ARRAY_DEPTH:
            raise _HeaderError("arrays nested too deeply")
        r.check_count(count, 12, "nested arrays")
        for _ in range(count):
            _skip_array(r, depth + 1)
    else:
        raise _HeaderError(f"unknown array item type {item_type}")


def _parse_gguf(r: _Reader, info: GGUFInfo) -> None:
    r._need(4)
    if bytes(r._data[:4]) != GGUF_MAGIC:
        raise _HeaderError("not a GGUF file: bad magic")
    r.pos = 4
    version = r.u32()
    if version & 0xFFFF == 0:
        # A big-endian file read as little-endian has an empty low word.
        r.set_endian(">")
        r.pos = 4
        version = r.u32()
    info.version = version
    if version not in _GGUF_VERSIONS:
        raise _HeaderError(f"unsupported GGUF version {version}")

    tensor_count = r.u64()
    kv_count = r.u64()
    info.tensor_count = tensor_count
    r.check_count(kv_count, 13, "metadata entries")

    for _ in range(kv_count):
        key = r.string()
        value_type = r.u32()
        if value_type in _SCALARS:
            info.metadata[key] = r.scalar(_SCALARS[value_type])
        elif value_type == _STRING:
            info.metadata[key] = r.string()
        elif value_type == _ARRAY:
            _skip_array(r, 1)
        else:
            raise _HeaderError(f"unknown value type {value_type} for key {key!r}")

    r.check_count(tensor_count, 24, "tensors")
    params = 0
    for _ in range(tensor_count):
        r.skip(r.u64())  # name
        n_dims = r.u32()
        if n_dims > _MAX_DIMS:
            raise _HeaderError(f"tensor declares {n_dims} dimensions")
        elements = 1
        for _ in range(n_dims):
            elements *= r.u64()
        r.skip(12)  # dtype (u32) and data offset (u64)
        params += elements
    info.param_count = params


def read_gguf(path: str | os.PathLike[str], *, cap: int = HEADER_CAP) -> GGUFInfo:
    """Read a GGUF header.

    :param path: File to read.
    :param cap: Most bytes to consume from the file.
    :return: Header facts; ``issues`` is non-empty when the header is bad.
    """
    info = GGUFInfo()
    try:
        with open(path, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            _parse_gguf(_Reader(fh, size, cap), info)
    except (OSError, _HeaderError) as exc:
        info.issues.append(str(exc))
    return info


def _tensor_elements(name: str, entry: Any) -> int:
    shape = entry.get("shape") if isinstance(entry, dict) else None
    if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
        raise _HeaderError(f"tensor {name!r} has no valid shape")
    return math.prod(shape)


def _layer_bits(quantization: Mapping[str, Any] | None, stem: str) -> int | None:
    """Bit width of one packed layer: its own entry, else the model default."""
    if not quantization:
        return None
    layer = quantization.get(stem)
    bits = layer.get("bits") if isinstance(layer, Mapping) else None
    if bits is None:
        bits = quantization.get("bits")
    return bits if isinstance(bits, int) and bits > 0 else None


def _count_safetensors(
    tensors: dict[str, Any], quantization: Mapping[str, Any] | None, info: SafetensorsInfo
) -> None:
    """Fill the parameter fields of ``info`` from the tensor table.

    MLX stores a quantized weight as a packed unsigned-integer tensor named
    ``<stem>.weight`` beside float tensors ``<stem>.scales`` and
    ``<stem>.biases``. The packed tensor's shape is not the weight's shape:
    each element holds ``width / bits`` weights. The scales and biases are
    quantization data, not parameters. Mixed-precision models give some
    layers their own bit width, keyed by ``<stem>`` in ``quantization``.
    """
    packed_stems = set()
    for name, entry in tensors.items():
        stem, dot, leaf = name.rpartition(".")
        if (
            dot
            and leaf == "weight"
            and isinstance(entry, dict)
            and entry.get("dtype") in _PACKED_DTYPES
            and f"{stem}.scales" in tensors
        ):
            packed_stems.add(stem)
    info.packed = bool(packed_stems)

    params = 0
    bits_known = True
    for name, entry in tensors.items():
        stem, _, leaf = name.rpartition(".")
        elements = _tensor_elements(name, entry)
        if stem not in packed_stems:
            params += elements
        elif leaf == "weight":
            bits = _layer_bits(quantization, stem)
            if bits is None:
                bits_known = False
            else:
                params += elements * _PACKED_DTYPES[entry["dtype"]] // bits
        elif leaf not in ("scales", "biases"):
            params += elements
    info.params_exact = not info.packed
    info.param_count = params if bits_known else None


def read_safetensors(
    path: str | os.PathLike[str],
    *,
    quantization: Mapping[str, Any] | None = None,
    cap: int = HEADER_CAP,
) -> SafetensorsInfo:
    """Read a safetensors header.

    :param path: File to read.
    :param quantization: The ``quantization`` object from the model's
        ``config.json``: a default ``bits`` plus optional per-layer entries.
        Needed to count the parameters of MLX packed weights.
    :param cap: Largest header to accept, in bytes.
    :return: Header facts; ``issues`` is non-empty when the header is bad.
    """
    info = SafetensorsInfo()
    try:
        with open(path, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            prefix = fh.read(8)
            if len(prefix) < 8:
                raise _HeaderError("truncated: file is shorter than the 8-byte length prefix")
            (length,) = struct.unpack("<Q", prefix)
            if length > size - 8:
                raise _HeaderError(f"declared header length {length} exceeds the file size {size}")
            if length > cap:
                raise _HeaderError(f"header is larger than the {cap} byte cap")
            raw = fh.read(length)
        header = json.loads(raw)
        if not isinstance(header, dict):
            raise _HeaderError("header is not a JSON object")
        metadata = header.pop("__metadata__", None)
        if isinstance(metadata, dict):
            info.metadata = metadata
        info.tensor_count = len(header)
        _count_safetensors(header, quantization, info)
    except (OSError, _HeaderError, ValueError, RecursionError) as exc:
        info.issues.append(str(exc))
    return info


def sniff(path: str | os.PathLike[str]) -> str | None:
    """Identify a model file by its first bytes.

    :param path: File to inspect.
    :return: ``"gguf"``, ``"safetensors"``, or None for anything else,
        including a file that cannot be opened.
    """
    try:
        with open(path, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            head = fh.read(9)
    except OSError:
        return None
    if head[:4] == GGUF_MAGIC:
        return "gguf"
    if len(head) == 9 and head[8:9] == b"{":
        (length,) = struct.unpack("<Q", head[:8])
        if 2 <= length <= size - 8:
            return "safetensors"
    return None
