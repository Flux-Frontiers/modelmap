"""The fixture tree itself: fast, small on disk, and shaped as documented."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tests.fakemachine import GB, FakeMachine, build


def _tree_sizes(root: Path) -> tuple[int, int]:
    """Apparent and allocated bytes of every regular file, each inode once."""
    seen: set[tuple[int, int]] = set()
    apparent = disk = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            st = os.lstat(os.path.join(dirpath, name))
            if os.path.islink(os.path.join(dirpath, name)) or (st.st_dev, st.st_ino) in seen:
                continue
            seen.add((st.st_dev, st.st_ino))
            apparent += st.st_size
            disk += st.st_blocks * 512
    return apparent, disk


def test_builds_in_under_a_second(tmp_path: Path) -> None:
    start = time.perf_counter()
    m = build(tmp_path)
    elapsed = time.perf_counter() - start
    m.cleanup()
    assert elapsed < 1.0


def test_tree_looks_huge_and_costs_almost_nothing(machine: FakeMachine) -> None:
    apparent, disk = _tree_sizes(machine.root)
    if disk >= apparent:
        pytest.skip("this filesystem does not store sparse files sparsely")
    assert apparent > 80 * GB
    assert disk < 5_000_000


def test_ollama_store_is_outside_home_behind_a_symlink(machine: FakeMachine) -> None:
    assert machine.ollama_link.is_symlink()
    assert machine.home not in machine.ollama_store.parents
    assert os.path.realpath(machine.ollama_link) == os.path.realpath(machine.ollama_store)
    assert (machine.ollama_link / "blobs").is_dir()
    assert len(list((machine.ollama_store / "blobs").iterdir())) == 5


def test_hf_snapshots_are_links_into_blobs(machine: FakeMachine) -> None:
    weights = machine.files["hf_weights"]
    for rev in ("rev1", "rev2"):
        link = machine.files[f"hf_snapshot_{rev}"] / "model.safetensors"
        assert link.is_symlink()
        assert link.samefile(weights)
    broken = machine.files["hf_snapshot_rev2"] / "broken.bin"
    assert broken.is_symlink() and not broken.exists()
    assert (machine.hf_hub / "models--acme--tiny" / "refs" / "main").read_text() == "rev2"
    assert (machine.hf_hub / "datasets--acme--words").is_dir()


def test_named_files_exist(machine: FakeMachine) -> None:
    for name, path in machine.files.items():
        assert path.exists(), name


def test_hostile_links_and_files(machine: FakeMachine) -> None:
    hostile = machine.hostile
    assert (hostile / "loop").is_symlink()
    assert (hostile / "loop" / "loop" / "empty.gguf").exists()
    assert machine.files["empty"].stat().st_size == 0
    assert set(machine.skipped) <= {"denied", "non_utf8_name"}


def test_unreadable_directory(machine: FakeMachine) -> None:
    if "denied" in machine.skipped:
        pytest.skip(machine.skipped["denied"])
    with pytest.raises(PermissionError):
        os.listdir(machine.hostile / "denied")


def test_non_utf8_filename(machine: FakeMachine) -> None:
    if "non_utf8_name" in machine.skipped:
        pytest.skip(machine.skipped["non_utf8_name"])
    assert any(b"\xff" in os.fsencode(p.name) for p in machine.hostile.iterdir())


def test_cleanup_makes_the_tree_removable(tmp_path: Path) -> None:
    m = build(tmp_path)
    m.cleanup()
    assert os.access(m.hostile / "denied", os.R_OK | os.W_OK | os.X_OK)
