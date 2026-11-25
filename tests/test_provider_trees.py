"""oMLX, the plain-tree stores, and the generic scanner."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from modelmap.providers import PROVIDERS, hf, omlx
from modelmap.providers.base import Context
from modelmap.providers.describe import guess_family, shard_base
from modelmap.providers.generic import scan_tree
from modelmap.providers.stores import gpt4all, jan, llamacpp, lmstudio, whisper
from modelmap.records import ModelRecord
from tests.fakemachine import GB, GGML_F16, FakeMachine, build, write_gguf, write_safetensors

HOME = Path("/home/u")


def _paths(
    provider: object, env: dict[str, str], platform: str, home: Path = HOME
) -> list[tuple[str, str]]:
    ctx = Context(home=home, env=env, platform=platform)
    return [(r.path, r.source) for r in provider.roots(ctx)]  # type: ignore[attr-defined]


def _by_name(records: list[ModelRecord]) -> dict[str, ModelRecord]:
    assert len({r.name for r in records}) == len(records), "duplicate record names"
    return {r.name: r for r in records}


# ------------------------------------------------------------------------ oMLX


def test_omlx_roots_come_from_settings(machine: FakeMachine, ctx: Context) -> None:
    (root,) = omlx.roots(ctx)
    assert (root.path, root.source) == (str(machine.omlx_models), "settings.json")


def test_omlx_root_precedence(tmp_path: Path) -> None:
    home = tmp_path
    base = home / ".omlx"
    base.mkdir()
    assert _paths(omlx, {}, "darwin", home) == [(str(base / "models"), "default")]

    (base / "settings.json").write_text(json.dumps({"model": {"model_dir": "/legacy"}}))
    assert _paths(omlx, {}, "darwin", home) == [("/legacy", "settings.json")]

    (base / "settings.json").write_text(
        json.dumps({"model": {"model_dirs": ["/a", "~/b"], "model_dir": "/legacy"}})
    )
    assert _paths(omlx, {}, "darwin", home) == [
        ("/a", "settings.json"),
        (str(home / "b"), "settings.json"),
    ]

    assert _paths(omlx, {"OMLX_MODEL_DIR": "/x, /y"}, "darwin", home) == [
        ("/x", "OMLX_MODEL_DIR"),
        ("/y", "OMLX_MODEL_DIR"),
    ]


def test_omlx_base_can_be_moved(tmp_path: Path) -> None:
    home = tmp_path
    bootstrap = home / "Library" / "Application Support" / "oMLX" / "base-path"
    bootstrap.parent.mkdir(parents=True)
    bootstrap.write_text("/moved/omlx\n")
    assert _paths(omlx, {}, "darwin", home) == [("/moved/omlx/models", "base-path")]
    assert _paths(omlx, {"OMLX_BASE_PATH": "/env/omlx"}, "darwin", home) == [
        ("/env/omlx/models", "OMLX_BASE_PATH")
    ]


def test_omlx_unreadable_settings_warn_and_fall_back(tmp_path: Path) -> None:
    (tmp_path / ".omlx").mkdir()
    (tmp_path / ".omlx" / "settings.json").write_text("{broken")
    ctx = Context(home=tmp_path, env={}, platform="darwin")
    (root,) = omlx.roots(ctx)
    assert root.source == "default"
    assert len(ctx.warnings) == 1


def test_omlx_models(machine: FakeMachine, ctx: Context) -> None:
    (root,) = omlx.roots(ctx)
    records = _by_name(omlx.scan(root, ctx))
    assert set(records) == {"mlx-community/Tiny-4bit", "acme/Sharded"}

    packed = records["mlx-community/Tiny-4bit"]
    assert (packed.format, packed.quant, packed.params, packed.params_exact) == (
        "mlx",
        "4-bit",
        64 * 64 + 64 + 64,
        False,
    )
    assert packed.files == 2  # config.json and the weights

    sharded = records["acme/Sharded"]
    assert (sharded.format, sharded.quant, sharded.params, sharded.files) == (
        "safetensors",
        None,
        4_000_000_000,
        3,
    )
    assert sharded.size_bytes == pytest.approx(8 * GB, rel=0.01)


def test_model_directory_without_weights_is_reported(machine: FakeMachine, ctx: Context) -> None:
    broken = machine.omlx_models / "mlx-community" / "half-downloaded"
    broken.mkdir()
    (broken / "config.json").write_text("{}")
    (root,) = omlx.roots(ctx)
    record = _by_name(omlx.scan(root, ctx))["mlx-community/half-downloaded"]
    assert (record.format, record.issues) == ("unknown", ["no weight files"])


def test_mixed_precision_label_and_component_shards(tmp_path: Path) -> None:
    model = tmp_path / "models" / "org" / "Mixed-4bit"
    packed = {
        "l.weight": ("U32", (8, 8)),
        "l.scales": ("BF16", (8, 1)),
        "l.biases": ("BF16", (8, 1)),
    }
    write_safetensors(model / "model-00001-of-00001.safetensors", packed)
    write_safetensors(
        model / "model-vision-00001-of-00001.safetensors", {"v.weight": ("F16", (10,))}
    )
    (model / "config.json").write_text(json.dumps({"quantization": {"bits": 4, "l": {"bits": 8}}}))
    ctx = Context(home=tmp_path, env={}, platform="darwin")

    (record,) = scan_tree(str(tmp_path / "models"), ctx, provider="omlx", config_marks_model=True)

    assert record.quant == "4-bit mixed"
    # The language model at its own 8 bits, plus the vision tower.
    assert record.params == 8 * 8 * 32 // 8 + 10


def test_differently_named_weight_files_are_not_added_up(tmp_path: Path) -> None:
    model = tmp_path / "models" / "sd"
    for name in ("model.safetensors", "model.fp16.safetensors"):
        write_safetensors(model / name, {"w": ("F16", (10,))})
    (model / "config.json").write_text("{}")
    ctx = Context(home=tmp_path, env={}, platform="darwin")
    (record,) = scan_tree(str(tmp_path / "models"), ctx, provider="omlx")
    assert record.params is None
    assert record.files == 3


# ---------------------------------------------------------------- plain stores


def test_lmstudio_home_rules(tmp_path: Path) -> None:
    home = tmp_path
    assert _paths(lmstudio, {}, "darwin", home) == [(str(home / ".lmstudio" / "models"), "default")]

    (home / ".cache" / "lm-studio").mkdir(parents=True)
    assert _paths(lmstudio, {}, "darwin", home) == [
        (str(home / ".cache" / "lm-studio" / "models"), "default (legacy)")
    ]

    moved = home / "elsewhere"
    moved.mkdir()
    (home / ".lmstudio-home-pointer").write_text(str(moved))
    assert _paths(lmstudio, {}, "darwin", home) == [
        (str(moved / "models"), ".lmstudio-home-pointer")
    ]

    (moved / "settings.json").write_text(json.dumps({"downloadsFolder": "/big/disk/models"}))
    assert _paths(lmstudio, {}, "darwin", home) == [("/big/disk/models", "settings.json")]

    (moved / "settings.json").write_text(json.dumps({"downloadsFolder": 7}))
    assert _paths(lmstudio, {}, "darwin", home) == [
        (str(moved / "models"), ".lmstudio-home-pointer")
    ]


def test_lmstudio_empty_pointer_file_is_ignored(tmp_path: Path) -> None:
    (tmp_path / ".lmstudio-home-pointer").write_text("\n")
    assert _paths(lmstudio, {}, "darwin", tmp_path) == [
        (str(tmp_path / ".lmstudio" / "models"), "default")
    ]


def test_jan_data_folder_from_settings(tmp_path: Path) -> None:
    app = tmp_path / ".local" / "share" / "Jan"
    app.mkdir(parents=True)
    (app / "settings.json").write_text(json.dumps({"data_folder": "/big/jan"}))
    assert _paths(jan, {}, "linux", tmp_path) == [
        ("/big/jan/llamacpp/models", "settings.json"),
        ("/big/jan/mlx/models", "settings.json"),
    ]
    assert _paths(jan, {"JAN_DATA_FOLDER": "/env"}, "linux", tmp_path)[0] == (
        "/env/llamacpp/models",
        "JAN_DATA_FOLDER",
    )
    (app / "settings.json").write_text(json.dumps({"data_folder": ""}))
    assert _paths(jan, {}, "linux", tmp_path)[0] == (
        str(app / "data" / "llamacpp" / "models"),
        "default",
    )


def test_lmstudio_models_and_projector(machine: FakeMachine, ctx: Context) -> None:
    (root,) = lmstudio.roots(ctx)
    records = _by_name(lmstudio.scan(root, ctx))
    assert set(records) == {"pub/vision/vision-Q4_K_M", "pub/vision/mmproj-vision-f16"}
    model, projector = records["pub/vision/vision-Q4_K_M"], records["pub/vision/mmproj-vision-f16"]
    assert (model.kind, model.quant, model.provider) == ("model", "Q4_K_M", "lmstudio")
    assert (projector.kind, projector.quant) == ("auxiliary", "F16")


@pytest.mark.parametrize(
    ("env", "platform", "expected"),
    [
        ({}, "darwin", ("/home/u/Library/Caches/llama.cpp", "default")),
        ({}, "linux", ("/home/u/.cache/llama.cpp", "default")),
        ({"XDG_CACHE_HOME": "/x"}, "linux", ("/x/llama.cpp", "default")),
        ({"XDG_CACHE_HOME": "/x"}, "darwin", ("/home/u/Library/Caches/llama.cpp", "default")),
        (
            {"LOCALAPPDATA": "C:/Users/u/AppData/Local"},
            "win32",
            ("C:/Users/u/AppData/Local/llama.cpp", "default"),
        ),
        ({"LLAMA_CACHE": "/models"}, "linux", ("/models", "LLAMA_CACHE")),
    ],
)
def test_llamacpp_cache_rules(
    env: dict[str, str], platform: str, expected: tuple[str, str]
) -> None:
    assert _paths(llamacpp, env, platform) == [expected]


def test_llamacpp_flat_cache(machine: FakeMachine, ctx: Context) -> None:
    (root,) = llamacpp.roots(ctx)
    (record,) = llamacpp.scan(root, ctx)
    assert (record.name, record.quant, record.provider) == ("acme_tiny-Q8_0", "Q8_0", "llamacpp")


def test_llama_cache_holding_a_hub_layout(machine: FakeMachine) -> None:
    ctx = Context(home=machine.home, env={"LLAMA_CACHE": str(machine.hf_hub)}, platform="darwin")
    (root,) = llamacpp.roots(ctx)
    records = _by_name(llamacpp.scan(root, ctx))
    assert records["acme/big-GGUF"].provider == "llamacpp"
    assert records["acme/big-GGUF"].quant == "F16"


def test_other_store_locations() -> None:
    assert _paths(gpt4all, {}, "darwin") == [
        ("/home/u/Library/Application Support/nomic.ai/GPT4All", "default"),
        ("/home/u/.cache/gpt4all", "default (python)"),
    ]
    assert _paths(gpt4all, {}, "linux")[0][0] == "/home/u/.local/share/nomic.ai/GPT4All"
    assert _paths(gpt4all, {"LOCALAPPDATA": "C:/L"}, "win32")[0][0] == "C:/L/nomic.ai/GPT4All"

    assert [p for p, _ in _paths(jan, {}, "darwin")] == [
        "/home/u/Library/Application Support/Jan/data/llamacpp/models",
        "/home/u/Library/Application Support/Jan/data/mlx/models",
    ]
    assert _paths(jan, {}, "linux")[0][0] == "/home/u/.local/share/Jan/data/llamacpp/models"
    assert _paths(jan, {"APPDATA": "C:/R"}, "win32")[0][0] == "C:/R/Jan/data/llamacpp/models"
    assert _paths(jan, {"JAN_DATA_FOLDER": "/j"}, "linux") == [
        ("/j/llamacpp/models", "JAN_DATA_FOLDER"),
        ("/j/mlx/models", "JAN_DATA_FOLDER"),
    ]

    assert _paths(whisper, {}, "darwin") == [("/home/u/.cache/whisper", "default")]
    assert _paths(whisper, {"XDG_CACHE_HOME": "/x"}, "linux") == [("/x/whisper", "default")]


def test_whisper_checkpoints_are_listed_without_a_size_floor(tmp_path: Path) -> None:
    cache = tmp_path / ".cache" / "whisper"
    cache.mkdir(parents=True)
    (cache / "tiny.pt").write_bytes(b"PK" + b"\0" * 1000)
    ctx = Context(home=tmp_path, env={}, platform="linux")
    (root,) = whisper.roots(ctx)
    (record,) = whisper.scan(root, ctx)
    # Inside a tool's own store the location vouches for the file.
    assert (record.name, record.format, record.issues, record.family) == (
        "tiny",
        "pytorch",
        [],
        None,
    )


def test_registry_is_consistent() -> None:
    names = [p.NAME for p in PROVIDERS]
    assert names == ["hf", "ollama", "omlx", "lmstudio", "llamacpp", "gpt4all", "jan", "whisper"]
    assert all(callable(p.roots) and callable(p.scan) for p in PROVIDERS)


# --------------------------------------------------------------------- generic


def _claimed(ctx: Context) -> set[str]:
    return {root.resolved for provider in PROVIDERS for root in provider.roots(ctx)}


def _sweep(machine: FakeMachine, ctx: Context) -> dict[str, ModelRecord]:
    return _by_name(scan_tree(str(machine.home), ctx, max_depth=5, claimed=_claimed(ctx)))


def test_sweep_finds_loose_models_by_header(machine: FakeMachine, ctx: Context) -> None:
    records = _sweep(machine, ctx)
    assert records["models/loose"].format == "gguf"
    # 37 MB: below the floor for unverified files, listed because its header is real.
    small = records[".kgrag/small-Q8_0"]
    assert (small.quant, small.family) == ("Q8_0", "bert")
    assert small.size_bytes < 50_000_000
    # A tokenizer stub has no tensors and is not a model.
    assert not any("vocab" in name for name in records)
    assert {r.provider for r in records.values()} == {"generic"}


def test_sweep_lists_large_checkpoints_as_unverified(machine: FakeMachine, ctx: Context) -> None:
    record = _sweep(machine, ctx)["ckpt/model"]
    assert (record.format, record.path) == ("pytorch", str(machine.files["pickle"]))
    # It starts like a torch.save archive, so it is not flagged.
    assert record.issues == []

    fake = machine.home / "ckpt" / "firmware.bin"
    fake.write_bytes(b"\x7fELF" + b"\0" * 60_000_000)
    flagged = _by_name(scan_tree(str(machine.home / "ckpt"), Context(machine.home, {}, "darwin")))
    assert flagged["ckpt/firmware"].issues == ["unverified: identified by extension only"]

    floor = _by_name(scan_tree(str(machine.home / "ckpt"), ctx, min_size=200_000_000))
    assert floor == {}


def test_sweep_stays_out_of_claimed_stores(machine: FakeMachine, ctx: Context) -> None:
    records = _sweep(machine, ctx)
    assert not any("Tiny-4bit" in name or "vision" in name for name in records)
    # Without the claim list the same walk does enter them.
    unclaimed = scan_tree(str(machine.home), Context(machine.home, {}, "darwin"), max_depth=6)
    assert any(r.name == "mlx-community/Tiny-4bit" for r in unclaimed)


def test_hostile_files_become_issues_and_warnings(machine: FakeMachine, ctx: Context) -> None:
    records = _sweep(machine, ctx)
    for name in ("empty", "truncated", "lying-count", "lying", "huge"):
        assert records[f"hostile/{name}"].issues, f"{name} has no issue"
        assert records[f"hostile/{name}"].params is None
    # The loop link is not followed, so each file appears once (checked by _by_name).
    if "denied" not in machine.skipped:
        assert [w.path for w in ctx.warnings] == [str(machine.hostile / "denied")]


def test_depth_limit(machine: FakeMachine, ctx: Context) -> None:
    shallow = _by_name(scan_tree(str(machine.home), ctx, max_depth=1, claimed=_claimed(ctx)))
    assert "models/loose" in shallow and ".kgrag/small-Q8_0" in shallow
    assert "hostile/empty" in shallow
    deep_only = machine.home / "a" / "b" / "c"
    write_gguf(deep_only / "deep.gguf", tensors=(("w", (4,), GGML_F16),))
    fresh = Context(machine.home, {}, "darwin")
    assert "c/deep" not in _by_name(
        scan_tree(str(machine.home), fresh, max_depth=2, claimed=_claimed(fresh))
    )
    fresh = Context(machine.home, {}, "darwin")
    assert "c/deep" in _by_name(
        scan_tree(str(machine.home), fresh, max_depth=3, claimed=_claimed(fresh))
    )


def test_split_gguf_is_one_record(tmp_path: Path) -> None:
    for index in (1, 2, 3):
        write_gguf(
            tmp_path / f"big-Q4_K_M-0000{index}-of-00003.gguf",
            tensors=(("w", (100,), GGML_F16),),
            metadata={"general.file_type": 15} if index == 1 else {},
        )
    ctx = Context(home=tmp_path, env={}, platform="linux")
    (record,) = scan_tree(str(tmp_path), ctx)
    assert (record.name, record.files, record.params, record.quant) == (
        f"{tmp_path.name}/big-Q4_K_M",
        3,
        300,
        "Q4_K_M",
    )
    assert record.path == str(tmp_path)


def test_extensionless_files_are_sniffed_only_on_request(
    machine: FakeMachine, ctx: Context
) -> None:
    blobs = str(machine.ollama_store / "blobs")
    assert scan_tree(blobs, ctx) == []
    found = scan_tree(blobs, Context(machine.home, {}, "darwin"), sniff_all=True)
    # The 5 GB weights and the 1 GB orphan; the small blobs are never opened.
    assert sorted(r.format for r in found) == ["gguf", "gguf"]


def test_diffusers_pipeline_is_one_record(tmp_path: Path) -> None:
    pipeline = tmp_path / "sd"
    (pipeline / "unet").mkdir(parents=True)
    (pipeline / "model_index.json").write_text("{}")
    write_safetensors(
        pipeline / "unet" / "diffusion_pytorch_model.safetensors", {"w": ("F16", (10,))}
    )
    write_safetensors(
        pipeline / "vae" / "diffusion_pytorch_model.safetensors", {"w": ("F16", (10,))}
    )
    ctx = Context(home=tmp_path, env={}, platform="linux")
    (record,) = scan_tree(str(tmp_path), ctx)
    assert (record.name, record.format, record.files, record.params) == (
        f"{tmp_path.name}/sd",
        "safetensors",
        3,
        None,
    )


# ------------------------------------------------------------ across providers


def _scan_everything(machine: FakeMachine) -> list[ModelRecord]:
    ctx = Context(home=machine.home, env={}, platform="darwin")
    records: list[ModelRecord] = []
    for provider in PROVIDERS:
        for root in provider.roots(ctx):
            if root.exists:
                records += provider.scan(root, ctx)
    return records + scan_tree(str(machine.home), ctx, max_depth=5, claimed=_claimed(ctx))


def test_scan_is_deterministic(machine: FakeMachine) -> None:
    first = [r.__dict__ for r in _scan_everything(machine)]
    second = [r.__dict__ for r in _scan_everything(machine)]
    assert first == second


def test_scan_changes_nothing_on_disk(machine: FakeMachine) -> None:
    def snapshot() -> list[tuple[str, int, int]]:
        found = []
        for dirpath, dirnames, filenames in os.walk(machine.root):
            if os.path.basename(dirpath) == "denied":
                continue
            for name in sorted(dirnames + filenames):
                st = os.lstat(os.path.join(dirpath, name))
                found.append((os.path.join(dirpath, name), st.st_size, st.st_mtime_ns))
        return sorted(found)

    before = snapshot()
    _scan_everything(machine)
    assert snapshot() == before


def test_file_reachable_from_two_stores_is_charged_once(machine: FakeMachine) -> None:
    twin = machine.home / ".lmstudio" / "models" / "acme" / "big" / "big.F16.gguf"
    twin.parent.mkdir(parents=True)
    os.link(machine.files["hf_big_gguf"], twin)

    records = _by_name([r for r in _scan_everything(machine) if "big" in r.name])

    assert records["acme/big-GGUF"].size_bytes == pytest.approx(70 * GB, rel=0.01)
    copy = records["acme/big/big.F16"]
    assert (copy.size_bytes, copy.shared_with, copy.quant) == (0, ["hf:acme/big-GGUF"], "F16")


def test_user_path_given_twice_is_not_counted_twice(machine: FakeMachine, ctx: Context) -> None:
    first = scan_tree(str(machine.home / "models"), ctx)
    again = scan_tree(str(machine.home / "models"), ctx)
    assert first[0].size_bytes > 0 and again[0].size_bytes == 0


def test_building_twice_gives_the_same_tree(tmp_path: Path) -> None:
    a, b = build(tmp_path / "a"), build(tmp_path / "b")
    names_a = sorted(
        str(p.relative_to(a.root)) for p in a.root.rglob("*") if "denied" not in str(p)
    )
    names_b = sorted(
        str(p.relative_to(b.root)) for p in b.root.rglob("*") if "denied" not in str(p)
    )
    a.cleanup(), b.cleanup()
    assert names_a == names_b


# -------------------------------------------------------------------- describe


@pytest.mark.parametrize(
    ("name", "architecture", "family"),
    [
        ("mlx-community/Qwen3-8B-4bit", None, "qwen"),
        ("mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit", None, "qwen"),
        ("llama3.2:3b", "llama", "llama"),
        ("clef-flash:latest", "qwen35", "qwen"),
        ("mystery:latest", "gemma3", "gemma"),
        ("mystery:latest", "newarch", "newarch"),
        ("mystery", None, None),
    ],
)
def test_guess_family(name: str, architecture: str | None, family: str | None) -> None:
    assert guess_family(name, architecture) == family


def test_shard_base() -> None:
    assert shard_base("model-00001-of-00004.safetensors") == "model"
    assert shard_base("big-Q4_K_M-00002-of-00003.gguf") == "big-Q4_K_M"
    assert shard_base("model.safetensors") == "model"
    assert shard_base("qwen-of-many.gguf") == "qwen-of-many"


def test_hf_roots_are_claimed_separately(machine: FakeMachine, ctx: Context) -> None:
    hub, xet = hf.roots(ctx)
    assert {hub.resolved, xet.resolved} <= _claimed(ctx)
