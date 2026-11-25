"""Per-file entries: what each record is made of, and who else uses it."""

from __future__ import annotations

import os

from modelmap.providers.base import Context
from modelmap.records import ModelRecord, Scan
from modelmap.scanner import Options, redact, scan
from tests.fakemachine import FakeMachine


def _by_name(result: Scan) -> dict[str, ModelRecord]:
    return {f"{r.provider}:{r.name}": r for r in result.records}


def test_charged_files_add_up_to_each_record(machine: FakeMachine, ctx: Context) -> None:
    result = scan(ctx)
    assert result.records
    for record in result.records:
        charged = [f for f in record.file_list if f.charged]
        assert sum(f.size_bytes for f in charged) == record.size_bytes, record.name
        assert sum(f.disk_bytes for f in charged) == record.disk_bytes, record.name
        assert len(record.file_list) == record.files, record.name
        for entry in record.file_list:
            assert os.path.isabs(entry.path) and os.path.exists(entry.path)
            assert entry.label


def test_hugging_face_blobs_are_named_from_the_current_snapshot(
    machine: FakeMachine, ctx: Context
) -> None:
    tiny = _by_name(scan(ctx))["hf:acme/tiny"]
    by_label = {f.label: f for f in tiny.file_list}
    assert by_label["model.safetensors"].path == os.path.realpath(machine.files["hf_weights"])
    assert by_label["model.safetensors"].note == ""
    assert by_label["config.json"].path == os.path.realpath(machine.files["hf_config_rev2"])
    # The config of the older revision has no name in the current one.
    (old,) = [f for f in tiny.file_list if f.note]
    assert old.note == "not in the current revision"
    assert old.path == os.path.realpath(machine.files["hf_config_rev1"])
    assert old.label == machine.files["hf_config_rev1"].name[:16]


def test_blob_in_the_shared_store_is_listed_at_its_real_path(
    machine: FakeMachine, ctx: Context
) -> None:
    (entry,) = _by_name(scan(ctx))["hf:acme/shared"].file_list
    assert entry.path == os.path.realpath(machine.files["hf_shared_payload"])
    assert entry.label == "shared.F16.gguf"


def test_ollama_blobs_are_named_by_layer(machine: FakeMachine, ctx: Context) -> None:
    tiny = _by_name(scan(ctx))["ollama:tiny:1b"]
    assert sorted(f.label for f in tiny.file_list) == [
        "config",
        "config",
        "params",
        "params",
        "template",
        "weights",
    ]
    weights = next(f for f in tiny.file_list if f.label == "weights")
    assert weights.path == os.path.realpath(machine.files["ollama_weights"])
    assert weights.size_bytes > 4_000_000_000


def test_model_directory_files_are_named_by_relative_path(
    machine: FakeMachine, ctx: Context
) -> None:
    sharded = _by_name(scan(ctx))["omlx:acme/Sharded"]
    assert [f.label for f in sharded.file_list] == [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    loose = _by_name(scan(Context(machine.home, {}, "darwin")))["generic:models/loose"]
    assert [f.label for f in loose.file_list] == ["loose.gguf"]


def test_first_owner_learns_of_later_users(machine: FakeMachine) -> None:
    twin = machine.home / ".lmstudio" / "models" / "acme" / "big" / "big.F16.gguf"
    twin.parent.mkdir(parents=True)
    os.link(machine.files["hf_big_gguf"], twin)

    found = _by_name(scan(Context(machine.home, {}, "darwin")))

    (original,) = found["hf:acme/big-GGUF"].file_list
    (copy,) = found["lmstudio:acme/big/big.F16"].file_list
    # The record scanned first paid for the file and did not know of the
    # second one at the time. It is the one a deletion would break.
    assert (original.charged, original.used_by) == (True, ["lmstudio:acme/big/big.F16"])
    assert (copy.charged, copy.used_by) == (False, ["hf:acme/big-GGUF"])
    assert copy.size_bytes == original.size_bytes > 0


def test_a_file_used_by_one_record_lists_nobody(machine: FakeMachine, ctx: Context) -> None:
    for record in scan(ctx).records:
        for entry in record.file_list:
            assert entry.used_by == [], (record.name, entry.label)


def test_inventory_keeps_the_file_lists(machine: FakeMachine, ctx: Context) -> None:
    result = scan(ctx)
    again = Scan.from_json(result.to_json())
    assert again == result
    assert again.records[0].file_list[0].label == result.records[0].file_list[0].label


def test_redact_hides_file_paths_and_users(machine: FakeMachine) -> None:
    twin = machine.home / "copies" / "loose.gguf"
    twin.parent.mkdir()
    os.link(machine.files["loose"], twin)
    result = scan(Context(machine.home, {}, "darwin"), Options(sweep=True))
    text = redact(result, machine.home).to_json()
    assert str(machine.home) not in text
    assert os.path.realpath(machine.home) not in text
    paths = [f.path for r in result.records for f in r.file_list]
    assert any(p.startswith("~/") for p in paths)
