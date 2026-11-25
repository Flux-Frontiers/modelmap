"""Sizing: two numbers per file, and each file charged once."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from modelmap.sizing import SizeLedger, stat_size
from tests.fakemachine import GB, FakeMachine


def test_apparent_and_disk_size_differ_for_a_sparse_file(machine: FakeMachine) -> None:
    size = stat_size(machine.files["hf_big_gguf"])
    if size.disk >= size.apparent:
        pytest.skip("this filesystem does not store sparse files sparsely")
    assert size.apparent == pytest.approx(70 * GB, rel=0.01)
    assert size.disk < 1_000_000


def test_symlink_and_hardlink_share_one_key(tmp_path: Path) -> None:
    real = tmp_path / "real.bin"
    real.write_bytes(b"x" * 5000)
    os.symlink(real, tmp_path / "soft")
    os.link(real, tmp_path / "hard")
    other = tmp_path / "other.bin"
    other.write_bytes(b"x" * 5000)

    assert stat_size(tmp_path / "soft").key == stat_size(real).key
    assert stat_size(tmp_path / "hard").key == stat_size(real).key
    assert stat_size(other).key != stat_size(real).key
    assert stat_size(tmp_path / "soft").apparent == 5000


def test_ledger_charges_a_file_once(tmp_path: Path) -> None:
    real = tmp_path / "real.bin"
    real.write_bytes(b"x" * 5000)
    os.link(real, tmp_path / "hard")
    ledger = SizeLedger()

    first = ledger.charge(real, "model-a")
    again = ledger.charge(tmp_path / "hard", "model-a")
    shared = ledger.charge(tmp_path / "hard", "model-b")

    assert (first.apparent, first.shared_with) == (5000, None)
    assert first.disk > 0
    assert (again.apparent, again.disk, again.shared_with) == (0, 0, None)
    assert (shared.apparent, shared.disk, shared.shared_with) == (0, 0, "model-a")


def test_ledger_raises_on_a_broken_link(machine: FakeMachine) -> None:
    broken = machine.files["hf_snapshot_rev2"] / "broken.bin"
    assert broken.is_symlink()
    with pytest.raises(OSError):
        SizeLedger().charge(broken, "acme/tiny")


def test_hf_revisions_sharing_a_blob_are_counted_once(machine: FakeMachine) -> None:
    ledger = SizeLedger()
    total = 0
    by_names = 0
    for rev in ("rev1", "rev2"):
        for name in ("model.safetensors", "config.json"):
            path = machine.files[f"hf_snapshot_{rev}"] / name
            total += ledger.charge(path, "acme/tiny").apparent
            by_names += path.stat().st_size

    weights = machine.files["hf_weights"].stat().st_size
    configs = sum(machine.files[f"hf_config_{rev}"].stat().st_size for rev in ("rev1", "rev2"))
    assert total == weights + configs
    assert by_names == 2 * weights + configs  # what a walk that adds up names reports


def test_ollama_tags_sharing_a_layer_through_a_symlinked_root(machine: FakeMachine) -> None:
    assert machine.ollama_link.is_symlink()
    ledger = SizeLedger()
    charged: dict[str, int] = {}
    shared: dict[str, str | None] = {}
    for tag in ("1b", "1b-4k"):
        manifest = json.loads(machine.files[f"ollama_manifest_{tag}"].read_text())
        weights = next(
            layer for layer in manifest["layers"] if layer["mediaType"].endswith(".model")
        )
        # Reach the blob the way a scan would: through ~/.ollama/models.
        blob = machine.ollama_link / "blobs" / weights["digest"].replace(":", "-")
        charge = ledger.charge(blob, f"tiny:{tag}")
        charged[tag], shared[tag] = charge.apparent, charge.shared_with

    assert charged == {"1b": pytest.approx(5 * GB, rel=0.01), "1b-4k": 0}
    assert shared == {"1b": None, "1b-4k": "tiny:1b"}


def test_one_ledger_spans_providers(tmp_path: Path) -> None:
    hub = tmp_path / "hub" / "model.gguf"
    hub.parent.mkdir()
    hub.write_bytes(b"x" * 4096)
    other = tmp_path / "lmstudio" / "model.gguf"
    other.parent.mkdir()
    os.link(hub, other)
    ledger = SizeLedger()

    total = (
        ledger.charge(hub, "hf:model").apparent + ledger.charge(other, "lmstudio:model").apparent
    )

    assert total == 4096
