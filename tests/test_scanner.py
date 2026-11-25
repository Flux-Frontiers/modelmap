"""The scanner: which roots are scanned, once each, and what comes back."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from modelmap import headers
from modelmap.providers import PROVIDERS, omlx
from modelmap.providers.base import Context
from modelmap.records import ModelRecord, Scan
from modelmap.scanner import Options, redact, scan
from tests.fakemachine import GGML_F16, FakeMachine, write_gguf, write_safetensors


def _names(result: Scan) -> list[str]:
    return [f"{r.provider}:{r.name}" for r in result.records]


def _by_name(result: Scan) -> dict[str, ModelRecord]:
    names = _names(result)
    assert len(set(names)) == len(names), "a model is listed twice"
    return dict(zip(names, result.records))


def _unique_bytes(root: Path) -> int:
    """Bytes of every regular file under a directory, each inode once."""
    seen: set[tuple[int, int]] = set()
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            st = os.lstat(path)
            if not stat.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) in seen:
                continue
            seen.add((st.st_dev, st.st_ino))
            total += st.st_size
    return total


# ------------------------------------------------------------------ default scan


def test_default_scan_finds_stores_and_loose_models(machine: FakeMachine, ctx: Context) -> None:
    found = _by_name(scan(ctx))
    for expected in (
        "hf:acme/tiny",
        "hf:acme/big-GGUF",
        "hf:acme/shared",
        "hf:xet cache",
        "ollama:tiny:1b",
        "omlx:mlx-community/Tiny-4bit",
        "omlx:acme/Sharded",
        "lmstudio:pub/vision/vision-Q4_K_M",
        "llamacpp:acme_tiny-Q8_0",
        "generic:models/loose",
        "generic:.kgrag/small-Q8_0",
        "generic:ckpt/model",
    ):
        assert expected in found, expected


def test_scan_metadata(machine: FakeMachine, ctx: Context) -> None:
    result = scan(ctx)
    assert result.tool_version and result.hostname
    assert result.elapsed_s >= 0
    by_provider = {(r.provider, r.exists) for r in result.roots}
    assert ("ollama", True) in by_provider and ("sweep", True) in by_provider
    # Stores that are not installed are still listed, as absent.
    assert ("whisper", False) in by_provider and ("jan", False) in by_provider
    ollama_root = next(r for r in result.roots if r.provider == "ollama")
    assert ollama_root.path != ollama_root.resolved  # the symlink and where it leads


def test_ollama_store_behind_the_symlink_is_scanned(machine: FakeMachine, ctx: Context) -> None:
    record = _by_name(scan(ctx))["ollama:tiny:1b"]
    assert record.aliases == ["tiny:1b-4k"]
    assert record.size_bytes > 4_000_000_000


def test_total_is_every_model_file_once(machine: FakeMachine, ctx: Context) -> None:
    result = scan(ctx)
    stores = sum(
        _unique_bytes(Path(root.resolved))
        for provider in PROVIDERS
        for root in provider.roots(Context(machine.home, {}, "darwin"))
        if root.exists
    )
    in_stores = sum(r.size_bytes for r in result.records if r.provider != "generic")
    # Stores also hold a few small bookkeeping files that belong to no model.
    assert 0 <= stores - in_stores < 100_000
    assert result.size_bytes == sum(r.size_bytes for r in result.records)


def test_scan_is_deterministic_and_read_only(machine: FakeMachine) -> None:
    def snapshot() -> list[tuple[str, int, int]]:
        found = []
        for dirpath, dirnames, filenames in os.walk(machine.root):
            if os.path.basename(dirpath) == "denied":
                continue
            for name in dirnames + filenames:
                st = os.lstat(os.path.join(dirpath, name))
                found.append((os.path.join(dirpath, name), st.st_size, st.st_mtime_ns))
        return sorted(found)

    before = snapshot()
    first = scan(Context(machine.home, {}, "darwin"))
    second = scan(Context(machine.home, {}, "darwin"))
    assert snapshot() == before
    assert [r.__dict__ for r in first.records] == [r.__dict__ for r in second.records]


def test_hostile_files_never_abort_the_scan(machine: FakeMachine, ctx: Context) -> None:
    result = scan(ctx)
    found = _by_name(result)
    assert found["generic:hostile/truncated"].issues
    assert found["generic:hostile/lying"].issues
    if "denied" not in machine.skipped:
        assert str(machine.hostile / "denied") in [w.path for w in result.warnings]


def test_directory_that_lists_but_cannot_be_searched(tmp_path: Path) -> None:
    # Mode r--: the names can be read, the files cannot be examined. Seen on
    # a real machine, where it stopped an earlier version of the scan.
    locked = tmp_path / "data" / "locked"
    write_gguf(locked / "model.gguf", tensors=(("w", (4, 4), GGML_F16),))
    (locked / "weights.bin").write_bytes(b"\0" * 100)
    (locked / "1213").write_bytes(b"\0" * 100)
    (locked / "sub").mkdir()
    locked.chmod(0o444)
    try:
        if os.access(locked / "1213", os.F_OK):
            pytest.skip("running as a user that ignores directory permissions")
        result = scan(
            Context(tmp_path, {}, "linux"), Options(paths=[str(tmp_path)], defaults=False)
        )
    finally:
        locked.chmod(0o755)
    assert result.warnings, "an unreadable file must be reported"
    assert all(str(locked) in w.path for w in result.warnings)
    for record in result.records:
        assert record.size_bytes == 0 and record.issues


# ------------------------------------------------------------- roots scanned once


def test_home_reached_through_a_symlink(machine: FakeMachine, tmp_path: Path) -> None:
    link = tmp_path / "linked-home"
    os.symlink(machine.home, link)
    found = _by_name(scan(Context(home=link, env={}, platform="darwin")))
    assert "omlx:mlx-community/Tiny-4bit" in found
    # The sweep did not walk into the stores a second time.
    assert not any(n.startswith("generic:") and "Tiny-4bit" in n for n in found)


def test_two_providers_resolving_to_one_directory(machine: FakeMachine) -> None:
    env = {"LLAMA_CACHE": str(machine.hf_hub)}
    result = scan(Context(home=machine.home, env=env, platform="darwin"))
    found = _by_name(result)
    assert "hf:acme/big-GGUF" in found
    assert not any(n.startswith("llamacpp:") for n in found)


def test_user_path_inside_a_scanned_store_adds_nothing(machine: FakeMachine, ctx: Context) -> None:
    baseline = _names(scan(Context(machine.home, {}, "darwin"), Options(sweep=False)))
    inside = str(machine.omlx_models / "mlx-community")
    result = scan(ctx, Options(paths=[inside, str(machine.omlx_models)]))
    assert _names(result) == baseline


def test_same_user_path_twice(machine: FakeMachine, ctx: Context) -> None:
    path = str(machine.home / "models")
    result = scan(ctx, Options(paths=[path, path], defaults=False))
    assert _names(result) == ["generic:models/loose"]
    assert result.records[0].size_bytes > 0


# ----------------------------------------------------------------- user paths


def test_no_defaults_scans_only_the_given_path(machine: FakeMachine, ctx: Context) -> None:
    result = scan(ctx, Options(paths=[str(machine.home / "models")], defaults=False))
    assert _names(result) == ["generic:models/loose"]
    assert [r.provider for r in result.roots] == ["user"]


def test_path_that_is_an_ollama_store_gets_the_ollama_provider(
    machine: FakeMachine, ctx: Context
) -> None:
    result = scan(ctx, Options(paths=[str(machine.ollama_store)], defaults=False))
    found = _by_name(result)
    assert found["ollama:tiny:1b"].aliases == ["tiny:1b-4k"]
    assert "ollama:unreferenced blobs" in found


def test_stores_nested_under_a_path_are_recognized(machine: FakeMachine, ctx: Context) -> None:
    # An external drive holding a copy of someone's stores.
    result = scan(ctx, Options(paths=[str(machine.root)], defaults=False))
    found = _by_name(result)
    assert "ollama:tiny:1b" in found
    assert "hf:acme/big-GGUF" in found
    assert found["hf:acme/tiny"].params == 1_000_000_000
    layouts = sorted(r.provider for r in result.roots if r.source == "layout")
    assert layouts == ["hf", "ollama"]


def test_docker_model_store_is_not_taken_for_ollama(tmp_path: Path) -> None:
    (tmp_path / "models" / "manifests").mkdir(parents=True)
    (tmp_path / "models" / "blobs" / "sha256").mkdir(parents=True)
    result = scan(Context(tmp_path, {}, "linux"), Options(paths=[str(tmp_path)], defaults=False))
    assert result.records == [] and [r.source for r in result.roots] == ["argument"]


def test_lock_directory_is_not_taken_for_a_hub_cache(tmp_path: Path) -> None:
    (tmp_path / "locks").mkdir()
    (tmp_path / "locks" / "models--org--name.lock").write_text("")
    result = scan(Context(tmp_path, {}, "linux"), Options(paths=[str(tmp_path)], defaults=False))
    assert [r.source for r in result.roots] == ["argument"]


def test_path_can_be_one_model_directory(machine: FakeMachine, ctx: Context) -> None:
    path = machine.omlx_models / "mlx-community" / "Tiny-4bit"
    result = scan(ctx, Options(paths=[str(path)], defaults=False))
    (record,) = result.records
    assert (record.name, record.format, record.quant) == ("mlx-community/Tiny-4bit", "mlx", "4-bit")


def test_path_can_be_one_file(machine: FakeMachine, ctx: Context) -> None:
    result = scan(ctx, Options(paths=[str(machine.files["small"])], defaults=False))
    (record,) = result.records
    assert (record.name, record.quant) == (".kgrag/small-Q8_0", "Q8_0")
    assert result.warnings == []


def test_missing_path_is_a_warning(machine: FakeMachine, ctx: Context) -> None:
    result = scan(ctx, Options(paths=[str(machine.home / "nope")], defaults=False))
    assert result.records == []
    assert result.roots[0].exists is False
    assert result.warnings[0].message == "no such file or directory"


def test_path_with_tilde_and_extensionless_weights(machine: FakeMachine, ctx: Context) -> None:
    blob = machine.home / "blobs-copy" / "weights"
    write_gguf(
        blob, tensors=(("w", (10, 1000 * 1000), GGML_F16),), metadata={"general.file_type": 1}
    )
    result = scan(ctx, Options(paths=["~/blobs-copy"], defaults=False))
    (record,) = result.records
    assert (record.format, record.quant) == ("gguf", "F16")


# -------------------------------------------------------------------- options


def test_only_limits_the_providers(machine: FakeMachine, ctx: Context) -> None:
    result = scan(ctx, Options(only=["ollama"], sweep=False))
    assert {r.provider for r in result.records} == {"ollama"}


def test_excludes(machine: FakeMachine, ctx: Context) -> None:
    result = scan(ctx, Options(paths=[str(machine.home)], defaults=False, excludes=[".*", "host*"]))
    names = _names(result)
    assert "generic:models/loose" in names
    assert not any(".kgrag" in n or "hostile" in n or n.startswith(("hf:", "omlx:")) for n in names)

    by_path = scan(
        Context(machine.home, {}, "darwin"),
        Options(paths=[str(machine.home / "models")], defaults=False, excludes=["*/loose.gguf"]),
    )
    assert by_path.records == []


def test_max_depth_and_min_size(machine: FakeMachine) -> None:
    deep = machine.home / "a" / "b" / "deep.gguf"
    write_gguf(deep, tensors=(("w", (4,), GGML_F16),))

    def run(**options: object) -> list[str]:
        opts = Options(paths=[str(machine.home / "a")], defaults=False, **options)  # type: ignore[arg-type]
        return _names(scan(Context(machine.home, {}, "darwin"), opts))

    assert run() == ["generic:b/deep"]
    assert run(max_depth=0) == []
    assert run(max_depth=1) == ["generic:b/deep"]

    def ckpt(**options: object) -> list[str]:
        opts = Options(paths=[str(machine.home / "ckpt")], defaults=False, **options)  # type: ignore[arg-type]
        return _names(scan(Context(machine.home, {}, "darwin"), opts))

    assert ckpt() == ["generic:ckpt/model"]
    assert ckpt(min_size=500_000_000) == []


def test_sweep_skips_application_data(machine: FakeMachine) -> None:
    hidden = machine.home / "Library" / "Application Support" / "SomeApp" / "cache.gguf"
    write_gguf(hidden, tensors=(("w", (4,), GGML_F16),))
    mac = _names(scan(Context(machine.home, {}, "darwin")))
    linux = _names(scan(Context(machine.home, {}, "linux")))
    assert "generic:SomeApp/cache" not in mac
    assert "generic:SomeApp/cache" in linux


def test_sweep_can_be_turned_off(machine: FakeMachine, ctx: Context) -> None:
    result = scan(ctx, Options(sweep=False))
    assert not any(r.provider == "generic" for r in result.records)
    assert not any(r.provider == "sweep" for r in result.roots)


def test_interrupt_returns_what_was_found(
    machine: FakeMachine, ctx: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupted(*args: object) -> list[ModelRecord]:
        raise KeyboardInterrupt

    monkeypatch.setattr(omlx, "scan", interrupted)
    result = scan(ctx)
    providers = {r.provider for r in result.records}
    assert "hf" in providers and "ollama" in providers and "omlx" not in providers
    assert result.warnings[-1].message == "interrupted: the results are partial"


# --------------------------------------------------------------------- redact


def test_redact_hides_home_and_host(machine: FakeMachine, ctx: Context) -> None:
    result = redact(scan(ctx), machine.home)
    text = result.to_json()
    assert result.hostname is None
    assert str(machine.home) not in text
    assert os.path.realpath(machine.home) not in text
    assert any(r.path.startswith("~/") for r in result.records)
    # Paths outside the home directory are left alone.
    ollama_root = next(r for r in result.roots if r.provider == "ollama")
    assert ollama_root.path == "~/.ollama/models"
    assert ollama_root.resolved == os.path.realpath(machine.ollama_store)


def test_redact_hides_the_user_name_in_record_names(tmp_path: Path) -> None:
    home = tmp_path / "zzuser"
    # Large enough for the sharing note (files under 1 MB are not noted).
    write_gguf(home / "model.gguf", tensors=(("w", (2, 1000 * 1000), GGML_F16),))
    twin = home / "copies" / "model.gguf"
    twin.parent.mkdir()
    os.link(home / "model.gguf", twin)
    ctx = Context(home=home, env={}, platform="linux")
    result = scan(ctx)
    assert [r.name for r in result.records] == ["zzuser/model", "copies/model"]
    assert result.records[1].shared_with == ["generic:zzuser/model"]

    text = redact(result, home).to_json()

    assert "zzuser" not in text
    assert [r.name for r in result.records] == ["~/model", "copies/model"]
    assert result.records[1].shared_with == ["generic:~/model"]


def test_redact_keeps_names_that_only_look_like_the_user(tmp_path: Path) -> None:
    home = tmp_path / "library"
    ctx = Context(home=home, env={}, platform="linux")
    store = home / ".ollama" / "models"
    (store / "blobs").mkdir(parents=True)
    result = scan(ctx)
    result.records.append(
        ModelRecord(
            "library:latest", "ollama", "gguf", str(store), 1, 1, aliases=["hf.co/library/x:1"]
        )
    )
    redact(result, home)
    assert result.records[-1].name == "library:latest"
    assert result.records[-1].aliases == ["hf.co/library/x:1"]


def test_naming_providers_limits_the_whole_scan(machine: FakeMachine, ctx: Context) -> None:
    result = scan(ctx, Options(only=["ollama"]))
    assert {r.provider for r in result.records} == {"ollama"}
    assert not any(r.provider == "sweep" for r in result.roots)


# ----------------------------------------------------------- cloud placeholders


@pytest.mark.skipif(not hasattr(os, "chflags"), reason="file flags are a BSD and macOS feature")
def test_cloud_placeholder_is_never_opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The real flag (SF_DATALESS) can only be set by the system. Point the
    # check at a flag a user may set, and set it on a real file.
    monkeypatch.setattr(headers, "_SF_DATALESS", stat.UF_NODUMP)
    model = tmp_path / "icloud" / "model.gguf"
    write_gguf(model, tensors=(("w", (4, 4), GGML_F16),), metadata={"general.file_type": 15})
    weights = tmp_path / "icloud" / "w.safetensors"
    write_safetensors(weights, {"w": ("F16", (4,))})
    os.chflags(model, stat.UF_NODUMP)
    os.chflags(weights, stat.UF_NODUMP)

    assert headers.sniff(model) is None
    assert "not downloaded" in headers.read_gguf(model).issues[0]
    assert "not downloaded" in headers.read_safetensors(weights).issues[0]
    result = scan(
        Context(tmp_path, {}, sys.platform), Options(paths=[str(tmp_path)], defaults=False)
    )
    found = _by_name(result)
    # Still listed with its size, with the reason nothing more is known.
    assert found["generic:icloud/model"].size_bytes == model.stat().st_size
    assert "not downloaded" in found["generic:icloud/model"].issues[0]
