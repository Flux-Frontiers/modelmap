"""The Hugging Face provider: root precedence, attribution, shared blobs."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from modelmap.providers import hf
from modelmap.providers.base import Context
from modelmap.records import ModelRecord
from tests.fakemachine import GB, FakeMachine

HOME = Path("/home/u")


def _roots(env: dict[str, str]) -> tuple[tuple[str, str], tuple[str, str]]:
    hub, xet = hf.roots(Context(home=HOME, env=env, platform="linux"))
    return (hub.path, hub.source), (xet.path, xet.source)


@pytest.mark.parametrize(
    ("env", "hub", "xet"),
    [
        ({}, ("/home/u/.cache/huggingface/hub", "default"), "/home/u/.cache/huggingface/xet"),
        ({"XDG_CACHE_HOME": "/x"}, ("/x/huggingface/hub", "XDG_CACHE_HOME"), "/x/huggingface/xet"),
        ({"HF_HOME": "/h", "XDG_CACHE_HOME": "/x"}, ("/h/hub", "HF_HOME"), "/h/xet"),
        (
            {"HUGGINGFACE_HUB_CACHE": "/old", "HF_HOME": "/h"},
            ("/old", "HUGGINGFACE_HUB_CACHE"),
            "/h/xet",
        ),
        (
            {"HF_HUB_CACHE": "/new", "HUGGINGFACE_HUB_CACHE": "/old", "HF_HOME": "/h"},
            ("/new", "HF_HUB_CACHE"),
            "/h/xet",
        ),
        ({"HF_HUB_CACHE": "~/hub", "HF_XET_CACHE": "/xc"}, ("/home/u/hub", "HF_HUB_CACHE"), "/xc"),
        (
            {"HF_HOME": ""},
            ("/home/u/.cache/huggingface/hub", "default"),
            "/home/u/.cache/huggingface/xet",
        ),
    ],
)
def test_root_precedence(env: dict[str, str], hub: tuple[str, str], xet: str) -> None:
    found_hub, found_xet = _roots(env)
    assert found_hub == hub
    assert found_xet[0] == xet


def _scan(ctx: Context) -> dict[str, ModelRecord]:
    hub, _ = hf.roots(ctx)
    return {r.name: r for r in hf.scan(hub, ctx)}


def test_finds_every_repository_and_nothing_else(machine: FakeMachine, ctx: Context) -> None:
    records = _scan(ctx)
    orphan = machine.files["hf_shared_orphan"].name[:12]
    assert set(records) == {
        "acme/tiny",
        "acme/big-GGUF",
        "acme/words",
        "acme/shared",
        "acme/config-only",
        f"shared blob {orphan}",
        "unreferenced shared blobs",
    }
    # A repository with only a ref and a .no_exist marker stores nothing.
    assert "acme/stub" not in records
    assert ctx.warnings == []


def test_revisions_sharing_a_blob_are_counted_once(machine: FakeMachine, ctx: Context) -> None:
    tiny = _scan(ctx)["acme/tiny"]
    weights = machine.files["hf_weights"].stat().st_size
    configs = sum(machine.files[f"hf_config_{rev}"].stat().st_size for rev in ("rev1", "rev2"))
    assert tiny.size_bytes == weights + configs
    assert tiny.files == 3
    assert (tiny.kind, tiny.format, tiny.params, tiny.params_exact) == (
        "model",
        "safetensors",
        1_000_000_000,
        True,
    )
    assert tiny.issues == ["broken link: broken.bin"]


def test_gguf_repository_is_described_from_its_header(ctx: Context) -> None:
    big = _scan(ctx)["acme/big-GGUF"]
    assert (big.format, big.quant, big.params, big.family) == (
        "gguf",
        "F16",
        35_000_000_000,
        "llama",
    )
    assert big.size_bytes == pytest.approx(70 * GB, rel=0.01)


def test_blob_in_the_shared_store_is_charged_to_its_repository(
    machine: FakeMachine, ctx: Context
) -> None:
    records = _scan(ctx)
    shared = records["acme/shared"]
    assert shared.size_bytes == machine.files["hf_shared_payload"].stat().st_size
    assert (shared.format, shared.quant, shared.family) == ("gguf", "F16", "qwen")
    assert shared.issues == []


def test_unreferenced_shared_blobs_are_listed(machine: FakeMachine, ctx: Context) -> None:
    records = _scan(ctx)
    orphan_path = machine.files["hf_shared_orphan"]
    orphan = records[f"shared blob {orphan_path.name[:12]}"]
    assert (orphan.kind, orphan.format, orphan.quant) == ("model", "gguf", "Q4_K_M")
    assert orphan.size_bytes == orphan_path.stat().st_size
    assert orphan.issues == ["no repository refers to this blob"]

    rest = records["unreferenced shared blobs"]
    # The stray file only: not the .refs, .lock or marker files.
    assert (rest.kind, rest.files, rest.size_bytes) == ("unattributed", 1, 4096)


def test_datasets_and_config_only_repositories_are_not_models(ctx: Context) -> None:
    records = _scan(ctx)
    assert (records["acme/words"].kind, records["acme/words"].format) == ("dataset", "files")
    config_only = records["acme/config-only"]
    assert (config_only.kind, config_only.format, config_only.issues) == ("other", "files", [])


def test_total_equals_the_content_on_disk(machine: FakeMachine, ctx: Context) -> None:
    bookkeeping = (".refs", ".lock", ".huggingface-shared-blobs")
    expected = 0
    for dirpath, _, filenames in os.walk(machine.hf_hub):
        if "/refs" in dirpath or "/.no_exist" in dirpath:
            continue
        for name in filenames:
            path = os.path.join(dirpath, name)
            if not os.path.islink(path) and not name.endswith(bookkeeping):
                expected += os.path.getsize(path)
    assert sum(r.size_bytes for r in _scan(ctx).values()) == expected


def test_incomplete_download_is_an_issue(machine: FakeMachine, ctx: Context) -> None:
    blobs = machine.hf_hub / "models--acme--config-only" / "blobs"
    (blobs / "abc.incomplete").write_bytes(b"x" * 100)
    assert _scan(ctx)["acme/config-only"].issues == ["incomplete download"]


def test_xet_cache_is_one_cache_record(machine: FakeMachine, ctx: Context) -> None:
    _, xet = hf.roots(ctx)
    (record,) = hf.scan(xet, ctx)
    assert (record.name, record.kind, record.provider, record.size_bytes) == (
        "xet cache",
        "cache",
        "hf",
        8192,
    )


def test_newest_snapshot_is_used_without_refs(machine: FakeMachine, ctx: Context) -> None:
    (machine.hf_hub / "models--acme--tiny" / "refs" / "main").unlink()
    tiny = _scan(ctx)["acme/tiny"]
    assert tiny.format == "safetensors"
    assert tiny.params == 1_000_000_000


def test_two_repositories_linking_one_shared_blob(machine: FakeMachine, ctx: Context) -> None:
    payload = machine.files["hf_shared_payload"]
    fork = machine.hf_hub / "models--zeta--fork"
    (fork / "blobs").mkdir(parents=True)
    os.symlink(f"../../blobs/{payload.parent.name}/{payload.name}", fork / "blobs" / "etag1")
    (fork / "snapshots" / "r1").mkdir(parents=True)
    os.symlink("../../blobs/etag1", fork / "snapshots" / "r1" / "shared.F16.gguf")

    records = _scan(ctx)

    # The first repository in name order pays; the second says who did.
    assert records["acme/shared"].size_bytes == payload.stat().st_size
    fork_record = records["zeta/fork"]
    assert (fork_record.size_bytes, fork_record.shared_with) == (0, ["hf:acme/shared"])
    assert (fork_record.format, fork_record.quant, fork_record.files) == ("gguf", "F16", 1)
