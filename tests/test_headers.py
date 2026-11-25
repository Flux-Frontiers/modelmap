"""Header readers: exact facts from good files, issues from bad ones."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from modelmap.headers import read_gguf, read_safetensors, sniff
from tests.fakemachine import (
    GGML_F16,
    GGML_F32,
    FakeMachine,
    gguf_header,
    safetensors_header,
    write_gguf,
    write_safetensors,
)

TENSORS = (("blk.0.weight", (4, 8), GGML_F16), ("blk.0.bias", (8,), GGML_F32))
METADATA = {
    "general.architecture": "llama",
    "general.name": "Tiny",
    "general.file_type": 15,
    "llama.context_length": 2048,
    "llama.rope.scale": 1.5,
    "tokenizer.add_bos": True,
    "tokenizer.ggml.tokens": ["<s>", "</s>", "hi"],
    "tokenizer.ggml.token_type": [3, 3, 1],
    "tokenizer.ggml.scores": [0.0, -1.0, -2.0],
}


def _bytes(tmp_path: Path, data: bytes, name: str = "f.gguf") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------- GGUF: good


def test_gguf_reads_params_quant_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "tiny.gguf"
    write_gguf(path, tensors=TENSORS, metadata=METADATA)

    info = read_gguf(path)

    assert info.issues == []
    assert info.version == 3
    assert info.tensor_count == 2
    assert info.param_count == 4 * 8 + 8
    assert info.architecture == "llama"
    assert info.name == "Tiny"
    assert info.quant == "Q4_K_M"
    assert info.metadata["llama.context_length"] == 2048
    assert info.metadata["llama.rope.scale"] == 1.5
    assert info.metadata["tokenizer.add_bos"] is True
    assert not info.is_vocab_only


def test_gguf_skips_arrays_and_keeps_what_follows(tmp_path: Path) -> None:
    path = tmp_path / "tiny.gguf"
    write_gguf(path, tensors=TENSORS, metadata={**METADATA, "after.arrays": "still here"})

    info = read_gguf(path)

    assert "tokenizer.ggml.tokens" not in info.metadata
    assert info.metadata["after.arrays"] == "still here"


@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize("endian", ["<", ">"])
def test_gguf_versions_and_byte_orders(tmp_path: Path, version: int, endian: str) -> None:
    path = tmp_path / "tiny.gguf"
    write_gguf(path, tensors=TENSORS, metadata=METADATA, version=version, endian=endian)

    info = read_gguf(path)

    assert info.issues == []
    assert info.version == version
    assert info.param_count == 40
    assert info.quant == "Q4_K_M"


def test_gguf_without_file_type_has_no_quant(tmp_path: Path) -> None:
    path = tmp_path / "tiny.gguf"
    write_gguf(path, tensors=TENSORS)
    assert read_gguf(path).quant is None


def test_gguf_unknown_file_type_is_reported_not_guessed(tmp_path: Path) -> None:
    path = tmp_path / "tiny.gguf"
    write_gguf(path, tensors=TENSORS, metadata={"general.file_type": 999})
    assert read_gguf(path).quant == "file_type 999"


@pytest.mark.parametrize(
    ("file_type", "quant"), [(41, "Q2_0"), (42, "F8_E4M3"), (1024, None), (32, "BF16")]
)
def test_gguf_newer_file_types(tmp_path: Path, file_type: int, quant: str | None) -> None:
    # 1024 is LlamaFileType.GUESSED: the file does not say.
    path = tmp_path / "tiny.gguf"
    write_gguf(path, tensors=TENSORS, metadata={"general.file_type": file_type})
    assert read_gguf(path).quant == quant


def test_gguf_vocab_only_file(machine: FakeMachine) -> None:
    info = read_gguf(machine.files["vocab"])
    assert info.issues == []
    assert info.is_vocab_only
    assert info.param_count == 0


def test_gguf_nested_array_is_skipped(tmp_path: Path) -> None:
    # One KV: an array of two arrays of uint8.
    inner = struct.pack("<IQ", 0, 3) + b"\x01\x02\x03"
    nested = struct.pack("<I", 9) + struct.pack("<IQ", 9, 2) + inner + inner
    key = struct.pack("<Q", 1) + b"k"
    data = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + key + nested

    info = read_gguf(_bytes(tmp_path, data))

    assert info.issues == []
    assert info.param_count == 0


def test_gguf_reads_only_the_header_of_a_70gb_file(machine: FakeMachine) -> None:
    path = machine.files["hf_big_gguf"]
    assert path.stat().st_size == pytest.approx(70e9, rel=0.01)
    header_len = len(
        gguf_header(
            tensors=(("blk.0.weight", (35 * 1000, 1000 * 1000), GGML_F16),),
            metadata={
                "general.architecture": "llama",
                "general.name": "Big",
                "general.file_type": 1,
            },
        )
    )

    # A cap equal to the header length proves no payload byte is needed.
    info = read_gguf(path, cap=header_len)

    assert info.issues == []
    assert info.param_count == 35_000_000_000
    assert info.quant == "F16"


# ----------------------------------------------------------------- GGUF: bad


def test_gguf_every_truncation_gives_an_issue_and_no_params(tmp_path: Path) -> None:
    whole = gguf_header(tensors=TENSORS, metadata=METADATA)
    # The header ends with up to 31 bytes of alignment padding that the
    # reader never needs, so only shorter cuts are sure to lose content.
    for cut in range(len(whole) - 31):
        info = read_gguf(_bytes(tmp_path, whole[:cut]))
        assert info.issues, f"no issue for a file cut at {cut}"
        assert info.param_count is None


def test_gguf_header_over_the_cap(tmp_path: Path) -> None:
    path = tmp_path / "tiny.gguf"
    write_gguf(path, tensors=TENSORS, metadata=METADATA)

    info = read_gguf(path, cap=64)

    assert info.param_count is None
    assert "cap" in info.issues[0]


def test_gguf_bad_magic(tmp_path: Path) -> None:
    info = read_gguf(_bytes(tmp_path, b"GGML" + b"\0" * 64))
    assert "bad magic" in info.issues[0]
    assert info.version is None


@pytest.mark.parametrize("version", [1, 4, 99])
def test_gguf_unsupported_version_is_an_issue(tmp_path: Path, version: int) -> None:
    info = read_gguf(_bytes(tmp_path, gguf_header(tensors=TENSORS, version=version)))
    assert info.version == version
    assert info.param_count is None
    assert f"unsupported GGUF version {version}" in info.issues[0]


@pytest.mark.parametrize("lie", ["tensor_count", "kv_count"])
def test_gguf_lying_counts_fail_at_once(tmp_path: Path, lie: str) -> None:
    data = gguf_header(tensors=TENSORS, metadata=METADATA, **{lie: 2**60})
    info = read_gguf(_bytes(tmp_path, data))
    assert info.param_count is None
    assert info.issues


def test_gguf_lying_string_length(tmp_path: Path) -> None:
    data = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2**61) + b"key"
    info = read_gguf(_bytes(tmp_path, data + b"\0" * 64))
    assert "truncated" in info.issues[0]


def test_gguf_lying_array_counts(tmp_path: Path) -> None:
    key = struct.pack("<Q", 1) + b"k"
    for item_type in (4, 8, 9):  # uint32, string, array
        array = struct.pack("<I", 9) + struct.pack("<IQ", item_type, 2**59)
        data = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + key + array + b"\0" * 64
        info = read_gguf(_bytes(tmp_path, data))
        assert info.issues, f"no issue for a lying array of type {item_type}"


def test_gguf_unknown_value_and_item_types(tmp_path: Path) -> None:
    key = struct.pack("<Q", 1) + b"k"
    head = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + key
    value = read_gguf(_bytes(tmp_path, head + struct.pack("<I", 77) + b"\0" * 64))
    item = read_gguf(_bytes(tmp_path, head + struct.pack("<IIQ", 9, 77, 1) + b"\0" * 64))
    assert "unknown value type 77" in value.issues[0]
    assert "unknown array item type 77" in item.issues[0]


def test_gguf_arrays_nested_too_deeply(tmp_path: Path) -> None:
    key = struct.pack("<Q", 1) + b"k"
    nest = struct.pack("<IQ", 9, 1)
    data = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + key + struct.pack("<I", 9) + nest * 6
    info = read_gguf(_bytes(tmp_path, data + b"\0" * 64))
    assert "nested too deeply" in info.issues[0]


def test_gguf_tensor_with_too_many_dimensions(tmp_path: Path) -> None:
    name = struct.pack("<Q", 1) + b"w"
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, 0) + name + struct.pack("<I", 500)
    info = read_gguf(_bytes(tmp_path, data + b"\0" * 64))
    assert "500 dimensions" in info.issues[0]


def test_gguf_unreadable_paths(tmp_path: Path, machine: FakeMachine) -> None:
    assert read_gguf(tmp_path / "missing.gguf").issues
    assert read_gguf(tmp_path).issues  # a directory
    assert read_gguf(machine.files["empty"]).issues
    assert read_gguf(machine.files["truncated"]).issues
    assert read_gguf(machine.files["lying_count"]).issues


# ---------------------------------------------------------------- safetensors


def test_safetensors_counts_parameters(tmp_path: Path) -> None:
    path = tmp_path / "m.safetensors"
    write_safetensors(
        path,
        {"a.weight": ("F16", (10, 20)), "a.bias": ("F32", (20,)), "scalar": ("F32", ())},
        metadata={"format": "pt"},
    )

    info = read_safetensors(path)

    assert info.issues == []
    assert info.tensor_count == 3
    assert info.param_count == 10 * 20 + 20 + 1
    assert info.params_exact
    assert not info.packed
    assert info.metadata == {"format": "pt"}


def test_safetensors_packed_mlx_weights_need_the_bit_width(machine: FakeMachine) -> None:
    path = machine.files["mlx_packed"]

    without = read_safetensors(path)
    with_bits = read_safetensors(path, quantization={"group_size": 64, "bits": 4})

    assert without.packed and without.param_count is None and not without.params_exact
    # A 64x64 weight (packed as 64x8 uint32), its bias and one norm weight.
    # The scales and biases beside the packed weight are not parameters.
    assert with_bits.param_count == 64 * 64 + 64 + 64
    assert not with_bits.params_exact
    assert with_bits.issues == []


def test_safetensors_per_layer_bit_width_overrides_the_default(machine: FakeMachine) -> None:
    # A mixed-precision model: this layer is 8-bit, so 64x8 uint32 hold 64x32.
    quantization = {"bits": 4, "layers.0.proj": {"bits": 8, "group_size": 64}}
    info = read_safetensors(machine.files["mlx_packed"], quantization=quantization)
    assert info.param_count == 64 * 32 + 64 + 64


@pytest.mark.parametrize("quantization", [{}, {"bits": 0}, {"bits": "4"}, {"group_size": 64}])
def test_safetensors_unusable_bit_width_gives_no_count(
    machine: FakeMachine, quantization: dict[str, object]
) -> None:
    info = read_safetensors(machine.files["mlx_packed"], quantization=quantization)
    assert info.param_count is None
    assert info.issues == []


def test_safetensors_unsigned_tensor_without_scales_is_not_packed(tmp_path: Path) -> None:
    path = tmp_path / "m.safetensors"
    write_safetensors(path, {"ids.weight": ("U32", (10, 4))})
    info = read_safetensors(path)
    assert not info.packed
    assert info.param_count == 40


def test_safetensors_reads_only_the_header_of_a_large_file(machine: FakeMachine) -> None:
    path = machine.files["hf_weights"]
    assert path.stat().st_size > 1.9e9
    info = read_safetensors(path)
    assert info.issues == []
    assert info.param_count == 1_000_000_000


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"\x05\x00", "shorter than the 8-byte length prefix"),
        (struct.pack("<Q", 10_000) + b"{}", "exceeds the file size"),
        (struct.pack("<Q", 2**62) + b"{}", "exceeds the file size"),
        (struct.pack("<Q", 5) + b"{oops", ""),
        (struct.pack("<Q", 2) + b"[]", "not a JSON object"),
        (struct.pack("<Q", 3) + b"\xff\xfe\xfd", ""),
    ],
)
def test_safetensors_bad_headers_give_issues(tmp_path: Path, data: bytes, expected: str) -> None:
    info = read_safetensors(_bytes(tmp_path, data, "m.safetensors"))
    assert info.param_count is None
    assert len(info.issues) == 1
    assert expected in info.issues[0]


@pytest.mark.parametrize("entry", [{"dtype": "F32"}, {"shape": "big"}, {"shape": [2, -1]}, 7])
def test_safetensors_bad_tensor_entries_give_issues(tmp_path: Path, entry: object) -> None:
    raw = json.dumps({"w": entry}).encode()
    info = read_safetensors(_bytes(tmp_path, struct.pack("<Q", len(raw)) + raw, "m.safetensors"))
    assert info.param_count is None
    assert "no valid shape" in info.issues[0]


def test_safetensors_header_over_the_cap(tmp_path: Path) -> None:
    path = tmp_path / "m.safetensors"
    write_safetensors(path, {"w": ("F32", (2, 2))})
    info = read_safetensors(path, cap=4)
    assert "cap" in info.issues[0]


def test_safetensors_deeply_nested_json_is_an_issue(tmp_path: Path) -> None:
    raw = b"[" * 100_000
    info = read_safetensors(_bytes(tmp_path, struct.pack("<Q", len(raw)) + raw, "m.safetensors"))
    assert info.issues


def test_safetensors_unreadable_and_hostile_paths(tmp_path: Path, machine: FakeMachine) -> None:
    assert read_safetensors(tmp_path / "missing.safetensors").issues
    assert read_safetensors(machine.files["lying_length"]).issues
    assert read_safetensors(machine.files["huge_length"]).issues


# ----------------------------------------------------------------------- sniff


def test_sniff_identifies_files_by_content_not_name(machine: FakeMachine, tmp_path: Path) -> None:
    head, _ = safetensors_header({"w": ("F32", (2,))})
    disguised = _bytes(tmp_path, head, "weights.bin")

    assert sniff(machine.files["ollama_weights"]) == "gguf"  # no extension at all
    assert sniff(machine.files["hf_weights"]) == "safetensors"
    assert sniff(disguised) == "safetensors"
    assert sniff(machine.files["pickle"]) is None
    assert sniff(machine.files["empty"]) is None
    assert sniff(machine.files["lying_length"]) is None
    assert sniff(tmp_path / "missing") is None
    assert sniff(tmp_path) is None
