"""The Ollama provider: linked roots, tags sharing weights, manifest shapes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from modelmap.providers import ollama
from modelmap.providers.base import Context
from modelmap.records import ModelRecord
from tests.fakemachine import GB, GGML_F16, FakeMachine, write_gguf

HOME = Path("/home/u")
IMAGE = "application/vnd.ollama.image."


def test_default_root_is_resolved_through_the_symlink(machine: FakeMachine, ctx: Context) -> None:
    (root,) = ollama.roots(ctx)
    assert root.path == str(machine.ollama_link)
    assert root.resolved == os.path.realpath(machine.ollama_store)
    assert (root.source, root.exists) == ("default", True)


def test_root_rules() -> None:
    def paths(env: dict[str, str], platform: str) -> list[tuple[str, str]]:
        return [(r.path, r.source) for r in ollama.roots(Context(HOME, env, platform))]

    assert paths({}, "darwin") == [("/home/u/.ollama/models", "default")]
    assert paths({}, "win32") == [("/home/u/.ollama/models", "default")]
    assert paths({}, "linux") == [
        ("/home/u/.ollama/models", "default"),
        ("/usr/share/ollama/.ollama/models", "default (service)"),
    ]
    assert paths({"OLLAMA_MODELS": "/data/ollama"}, "linux") == [("/data/ollama", "OLLAMA_MODELS")]


@pytest.mark.parametrize(
    ("parts", "name"),
    [
        (("registry.ollama.ai", "library", "qwen3", "8b"), "qwen3:8b"),
        (
            ("registry.ollama.ai", "digitsflow", "bonsai-8b", "latest"),
            "digitsflow/bonsai-8b:latest",
        ),
        (("ollama.com", "library", "qwen3", "8b"), "qwen3:8b"),
        (("hf.co", "unsloth", "Qwen3-8B-GGUF", "Q8_0"), "hf.co/unsloth/Qwen3-8B-GGUF:Q8_0"),
        (("Registry.Ollama.AI", "library", "qwen3", "8b"), "qwen3:8b"),
        (("localhost%3A5000", "team", "model", "v1"), "localhost:5000/team/model:v1"),
    ],
)
def test_tag_names_match_ollama_list(parts: tuple[str, ...], name: str) -> None:
    assert ollama.tag_name(parts) == name


def _scan(ctx: Context) -> dict[str, ModelRecord]:
    (root,) = ollama.roots(ctx)
    return {r.name: r for r in ollama.scan(root, ctx)}


def test_tags_sharing_weights_are_one_record(machine: FakeMachine, ctx: Context) -> None:
    records = _scan(ctx)
    assert set(records) == {"tiny:1b", "hf.co/acme/gone-GGUF:Q4_K_M", "unreferenced blobs"}

    tiny = records["tiny:1b"]
    assert tiny.aliases == ["tiny:1b-4k"]
    assert (tiny.format, tiny.quant, tiny.params, tiny.family) == (
        "gguf",
        "Q4_K_M",
        5_000_000_000,
        "qwen",
    )
    # Weights, template, both params files and both config blobs, each once.
    assert tiny.files == 6
    assert tiny.size_bytes == pytest.approx(5 * GB, rel=0.01)
    assert tiny.issues == [] and tiny.shared_with == []
    assert tiny.path == str(Path(os.path.realpath(machine.files["ollama_weights"])))


def test_every_blob_is_accounted_for(machine: FakeMachine, ctx: Context) -> None:
    blobs = machine.ollama_store / "blobs"
    on_disk = sum(p.stat().st_size for p in blobs.iterdir())
    assert sum(r.size_bytes for r in _scan(ctx).values()) == on_disk


def test_missing_weights_and_orphans_are_reported(machine: FakeMachine, ctx: Context) -> None:
    records = _scan(ctx)
    gone = records["hf.co/acme/gone-GGUF:Q4_K_M"]
    assert gone.size_bytes == 0
    assert gone.format == "unknown"
    assert len(gone.issues) == 1 and "missing blob" in gone.issues[0]
    assert f"{3 * GB} bytes declared" in gone.issues[0]

    orphans = records["unreferenced blobs"]
    assert (orphans.kind, orphans.files) == ("unattributed", 1)
    assert orphans.size_bytes == machine.files["ollama_orphan"].stat().st_size


def test_dotfiles_are_ignored_and_bad_manifests_warn(machine: FakeMachine, ctx: Context) -> None:
    assert _scan(ctx) and ctx.warnings == []  # the .DS_Store in manifests/ is skipped

    bad = machine.ollama_store / "manifests" / "registry.ollama.ai" / "library" / "bad" / "latest"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json")
    fresh = Context(home=machine.home, env={}, platform="darwin")
    records = _scan(fresh)
    assert "tiny:1b" in records and "bad:latest" not in records
    assert [w.path for w in fresh.warnings] == [str(Path(os.path.realpath(bad)))]


# ------------------------------------------------- manifest shapes from source


class Store:
    """A hand-built store for manifest shapes no fixture machine has."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def blob(self, label: str, data: bytes | None = None, **gguf: Any) -> dict[str, Any]:
        digest = hashlib.sha256(label.encode()).hexdigest()
        path = self.root / "blobs" / f"sha256-{digest}"
        if data is None:
            write_gguf(path, **gguf)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return {"digest": f"sha256:{digest}", "size": path.stat().st_size}

    def manifest(self, tree: str, relpath: str, body: dict[str, Any]) -> None:
        path = self.root / tree / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body))

    def scan(self) -> dict[str, ModelRecord]:
        ctx = Context(home=self.root, env={"OLLAMA_MODELS": str(self.root)}, platform="darwin")
        (root,) = ollama.roots(ctx)
        return {r.name: r for r in ollama.scan(root, ctx)}


def test_split_model_projector_and_adapter_layers(tmp_path: Path) -> None:
    store = Store(tmp_path)
    shard = {"tensors": (("w", (1000, 1000), GGML_F16),), "metadata": {"general.file_type": 7}}
    layers = [
        {"mediaType": IMAGE + "model", **store.blob("shard 1", **shard)},
        {"mediaType": IMAGE + "model", **store.blob("shard 2", **shard)},
        {"mediaType": IMAGE + "projector", **store.blob("projector", **shard)},
        {"mediaType": IMAGE + "adapter", **store.blob("adapter", b"x" * 500)},
    ]
    store.manifest("manifests", "registry.ollama.ai/library/vision/7b", {"layers": layers})

    (record,) = store.scan().values()

    assert record.name == "vision:7b"
    assert (record.files, record.params, record.quant) == (4, 2_000_000, "Q8_0")


def test_safetensors_model_stored_as_tensor_layers(tmp_path: Path) -> None:
    store = Store(tmp_path)
    layers = [
        {"mediaType": IMAGE + "tensor", "name": f"t{i}", **store.blob(f"tensor {i}", b"\0" * 1000)}
        for i in range(3)
    ] + [{"mediaType": IMAGE + "json", "name": "config.json", **store.blob("cfg", b"{}")}]
    for tag in ("4bit", "latest"):
        store.manifest(
            "manifests", f"registry.ollama.ai/library/mlxmodel/{tag}", {"layers": layers}
        )

    (record,) = store.scan().values()

    assert (record.name, record.aliases) == ("mlxmodel:4bit", ["mlxmodel:latest"])
    assert (record.format, record.files, record.size_bytes, record.params) == (
        "safetensors",
        4,
        3002,
        None,
    )


def test_manifests_v2_is_read_first_and_lists_are_followed(tmp_path: Path) -> None:
    store = Store(tmp_path)
    gguf = {"tensors": (("w", (10, 10), GGML_F16),), "metadata": {"general.file_type": 15}}
    new = {"mediaType": IMAGE + "model", **store.blob("new weights", **gguf)}
    old = {"mediaType": IMAGE + "model", **store.blob("old weights", **gguf)}
    inline = {"runner": "llamacpp", "format": "gguf", "layers": [new]}
    child = {"layers": [{"mediaType": IMAGE + "license", **store.blob("license", b"MIT")}]}
    by_reference = store.blob("child manifest", json.dumps(child).encode())
    not_pulled = {"digest": "sha256:" + "0" * 64, "runner": "mlx"}
    store.manifest(
        "manifests-v2",
        "ollama.com/library/both/latest",
        {
            "mediaType": "application/vnd.ollama.manifest.list.v2+json",
            "manifests": [inline, by_reference, not_pulled],
        },
    )
    store.manifest("manifests", "registry.ollama.ai/library/both/latest", {"layers": [old]})

    records = store.scan()

    both = records["both:latest"]
    # The v2 entry wins: new weights, the referenced child manifest, its license.
    assert both.files == 3 and both.issues == []
    assert both.path.endswith(new["digest"].replace(":", "-"))
    # The legacy entry for the same tag is not read, so its weights are orphans.
    assert records["unreferenced blobs"].files == 1
