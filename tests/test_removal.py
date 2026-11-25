"""Removal: what a plan lists is all that goes, and what it refuses."""

from __future__ import annotations

import io
import os
import shlex
from pathlib import Path

import pytest

from modelmap.cli import main
from modelmap.providers.base import Context
from modelmap.records import FileEntry, ModelRecord
from modelmap.removal import execute, find, plan, target_of
from modelmap.scanner import scan
from tests.fake_tools import Tools, install
from tests.fakemachine import GGML_F16, FakeMachine, write_gguf
from tests.test_provider_ollama import IMAGE, Store


@pytest.fixture
def tools(tmp_path: Path) -> Tools:
    return install(tmp_path / "tools", ["tiny:1b", "tiny:1b-4k", "other:latest"])


def _ctx(machine: FakeMachine, env: dict[str, str]) -> Context:
    return Context(home=machine.home, env=env, platform="darwin")


def _records(machine: FakeMachine, env: dict[str, str]) -> tuple[Context, dict[str, ModelRecord]]:
    ctx = _ctx(machine, env)
    result = scan(ctx)
    return ctx, {f"{r.provider}:{r.name}": r for r in result.records}


def _live(machine: FakeMachine) -> list[str]:
    return [os.path.realpath(machine.ollama_store)]


def rm(ctx: Context, *argv: str, answer: str | None = None) -> tuple[int, str]:
    out = io.StringIO()
    ask = None if answer is None else (lambda _prompt: answer)
    return main(["rm", *argv], ctx=ctx, out=out, ask=ask), out.getvalue()


# --------------------------------------------------------------- plans, static


def test_described_plans_without_looking_at_the_disk() -> None:
    def described(record: ModelRecord, stores: list[str] = ()) -> tuple[str, str]:  # type: ignore[assignment]
        step = plan(record, stores, check_disk=False)
        return step.action, step.note

    ollama = ModelRecord(
        "qwen3:8b", "ollama", "gguf", "/s/blobs/sha256-a", 5, 5, aliases=["qwen3:latest"]
    )
    assert described(ollama, ["/s"]) == (
        "tool",
        "Runs `ollama rm qwen3:8b qwen3:latest`: removes every tag of these weights from Ollama.",
    )
    action, note = described(ollama, ["/elsewhere"])
    assert action == "refuse" and "not the one the Ollama on this machine uses" in note

    repo = ModelRecord("acme/tiny", "hf", "safetensors", "/c/hub/models--acme--tiny", 5, 5)
    step = plan(repo, check_disk=False)
    assert step.argv == ["hf", "cache", "rm", "model/acme/tiny", "--cache-dir", "/c/hub", "--yes"]
    assert step.note.startswith("Runs `hf cache rm model/acme/tiny --cache-dir /c/hub`")

    blob = ModelRecord("shared blob 06ae", "hf", "gguf", "/c/hub/blobs/06/06aec939", 5, 5)
    action, note = described(blob)
    assert action == "refuse" and "`hf cache prune --cache-dir /c/hub`" in note

    for kind in ("cache", "unattributed"):
        assert (
            described(ModelRecord("x", "ollama", "files", "/s/blobs", 5, 5, kind=kind))[0]
            == "refuse"
        )

    shared = ModelRecord(
        "models/link",
        "generic",
        "gguf",
        "/h/models/link.gguf",
        0,
        0,
        file_list=[
            FileEntry("/c/hub/blobs/x", "link.gguf", 5, 5, charged=False, used_by=["hf:acme/big"])
        ],
    )
    action, note = described(shared)
    assert action == "refuse" and note.startswith("This is another name for files of hf:acme/big")

    loose = ModelRecord(
        "models/a",
        "generic",
        "gguf",
        "/h/models/a.gguf",
        5,
        5,
        file_list=[FileEntry("/h/models/a.gguf", "a.gguf", 5, 5)],
    )
    assert described(loose)[0] == "trash"


def test_names_in_commands_are_quoted_for_a_shell() -> None:
    evil = ModelRecord("a b; rm -rf ~", "ollama", "gguf", "/s/blobs/x", 5, 5, aliases=["$(id)"])
    step = plan(evil, ["/s"], check_disk=False)
    assert step.argv == ["ollama", "rm", "a b; rm -rf ~", "$(id)"]
    quoted = step.note.split("`")[1]
    assert shlex.split(quoted) == ["ollama", "rm", "a b; rm -rf ~", "$(id)"]


# --------------------------------------------------------- plans, on the disk


def test_no_tools_on_the_path_means_nothing_can_be_removed(
    machine: FakeMachine, tools: Tools
) -> None:
    # An empty environment must never fall back to the process's own PATH.
    ctx, records = _records(machine, {})
    for name in ("generic:models/loose", "ollama:tiny:1b", "hf:acme/tiny"):
        step = plan(records[name], _live(machine), env=ctx.env)
        assert step.action == "refuse", name
    status, text = rm(ctx, "generic:models/loose", "--yes")
    assert status == 1 and "No command to move files to the Trash" in text
    assert machine.files["loose"].exists()


def test_loose_file_moves_to_the_trash(machine: FakeMachine, tools: Tools) -> None:
    ctx, records = _records(machine, tools.env("trash"))
    step = plan(records["generic:models/loose"], env=ctx.env)
    assert (step.action, step.paths) == ("trash", [str(machine.files["loose"])])
    assert execute(step, ctx.env) == (True, "")
    assert not machine.files["loose"].exists()
    assert (tools.trash / "loose.gguf").exists()


def test_model_folder_moves_whole_only_when_every_entry_is_listed(
    machine: FakeMachine, tools: Tools
) -> None:
    ctx, records = _records(machine, tools.env("trash"))
    folder = machine.omlx_models / "acme" / "Sharded"
    step = plan(records["omlx:acme/Sharded"], env=ctx.env)
    assert step.paths == [str(folder)]
    assert step.note == "Moves the folder, with its 3 files, to the Trash."


def test_model_folder_holding_more_is_refused_and_untouched(
    machine: FakeMachine, tools: Tools
) -> None:
    # A model kept inside a code checkout: moving the model's files out would
    # leave a broken working tree, and moving the folder would take the code.
    folder = machine.omlx_models / "acme" / "Sharded"
    (folder / ".git").mkdir()
    (folder / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (folder / "src").mkdir()
    (folder / "src" / "main.py").write_text("print(1)\n")
    before = sorted(str(p.relative_to(folder)) for p in folder.rglob("*"))
    ctx, records = _records(machine, tools.env("trash"))
    step = plan(records["omlx:acme/Sharded"], env=ctx.env, home=str(machine.home))
    assert step.action == "refuse"
    assert f"The folder also holds {folder / '.git'}" in step.note
    status, _ = rm(ctx, "omlx:acme/Sharded", "--yes")
    assert status == 1
    assert sorted(str(p.relative_to(folder)) for p in folder.rglob("*")) == before


def test_a_folder_linking_out_is_refused(machine: FakeMachine, tools: Tools) -> None:
    folder = machine.omlx_models / "acme" / "Sharded"
    (folder / "elsewhere").symlink_to(machine.home)
    ctx, records = _records(machine, tools.env("trash"))
    step = plan(records["omlx:acme/Sharded"], env=ctx.env)
    assert step.action == "refuse" and str(folder / "elsewhere") in step.note


def test_loose_shards_move_one_by_one(machine: FakeMachine, tools: Tools) -> None:
    folder = machine.home / "downloads"
    for index in (1, 2):
        write_gguf(folder / f"big-0000{index}-of-00002.gguf", tensors=(("w", (4,), GGML_F16),))
    (folder / "receipt.pdf").write_bytes(b"%PDF")
    ctx, records = _records(machine, tools.env("trash"))
    step = plan(records["generic:downloads/big"], env=ctx.env)
    assert sorted(os.path.basename(p) for p in step.paths) == [
        "big-00001-of-00002.gguf",
        "big-00002-of-00002.gguf",
    ]
    assert execute(step, ctx.env) == (True, "")
    assert sorted(p.name for p in folder.iterdir()) == ["receipt.pdf"]


def test_hub_repository_with_a_link_from_outside_is_refused(
    machine: FakeMachine, tools: Tools
) -> None:
    link = machine.home / "models" / "link.gguf"
    link.symlink_to(machine.files["hf_big_gguf"])
    ctx, records = _records(machine, tools.env("hf"))
    step = plan(records["hf:acme/big-GGUF"], env=ctx.env)
    assert step.action == "refuse"
    assert step.note.startswith("generic:models/link uses files of this one through a link")


def test_ollama_tags_with_their_own_settings_are_named(machine: FakeMachine, tools: Tools) -> None:
    # In the fixture, tiny:1b and tiny:1b-4k share weights but not params:
    # the second was made with `ollama create` to change the context length.
    ctx, records = _records(machine, tools.env("ollama"))
    step = plan(records["ollama:tiny:1b"], _live(machine), env=ctx.env)
    assert step.variants == ["tiny:1b", "tiny:1b-4k"]
    assert "differ in their settings" in step.note and "cannot be pulled again" in step.note

    status, text = rm(ctx, "tiny:1b", "--yes")
    assert status == 1 and "name each of these tags to remove it: tiny:1b-4k" in text
    assert "tiny:1b-4k" in tools.tags.read_text()
    status, _ = rm(_ctx(machine, tools.env("ollama")), "tiny:1b", "tiny:1b-4k", "--yes")
    assert status == 0 and tools.tags.read_text() == "other:latest\n"


def test_ollama_tags_that_are_only_names_need_nothing_extra(tmp_path: Path, tools: Tools) -> None:
    store = Store(tmp_path / "store")
    weights = {"mediaType": IMAGE + "model", **store.blob("w", tensors=(("w", (4,), GGML_F16),))}
    template = {"mediaType": IMAGE + "template", **store.blob("t", b"{{ .Prompt }}")}
    for tag in ("8b", "latest"):
        store.manifest(
            "manifests", f"registry.ollama.ai/library/qwen3/{tag}", {"layers": [weights, template]}
        )
    tools.tags.write_text("qwen3:8b\nqwen3:latest\n")
    env = {**tools.env("ollama"), "OLLAMA_MODELS": str(store.root)}
    ctx = Context(home=tmp_path, env=env, platform="darwin")
    records = {r.name: r for r in scan(ctx).records}
    step = plan(records["qwen3:8b"], [str(store.root.resolve())], env=env)
    assert step.action == "tool" and step.variants == []
    assert "differ in their settings" not in step.note


def test_file_reached_through_a_symlink_is_refused(machine: FakeMachine, tools: Tools) -> None:
    # The loose name is a link to the Hugging Face blob. Moving the resolved
    # path would empty the other store; refusing is the only safe answer.
    link = machine.home / "models" / "link.gguf"
    link.symlink_to(machine.files["hf_big_gguf"])
    ctx, records = _records(machine, tools.env("trash"))
    step = plan(records["generic:models/link"], env=ctx.env)
    assert step.action == "refuse"
    assert step.note.startswith("This is another name for files of hf:acme/big-GGUF")
    assert records["hf:acme/big-GGUF"].file_list[0].used_by == ["generic:models/link"]
    status, _ = rm(ctx, "generic:models/link", "--yes")
    assert status == 1 and machine.files["hf_big_gguf"].exists() and link.is_symlink()


def test_ollama_needs_every_tag_listed_by_the_running_ollama(
    machine: FakeMachine, tools: Tools
) -> None:
    ctx, records = _records(machine, tools.env("ollama"))
    record = records["ollama:tiny:1b"]
    assert plan(record, _live(machine), env=ctx.env).action == "tool"

    tools.tags.write_text("tiny:1b\n")  # the server knows only one of the two tags
    step = plan(record, _live(machine), env=ctx.env)
    assert step.action == "refuse" and "does not list tiny:1b-4k" in step.note

    assert plan(record, _live(machine), env=tools.env()).action == "refuse"  # no ollama


def test_hub_repository_is_removed_with_hf(machine: FakeMachine, tools: Tools) -> None:
    ctx, records = _records(machine, tools.env("hf"))
    record = records["hf:acme/tiny"]
    step = plan(record, env=ctx.env)
    assert step.action == "tool"
    assert execute(step, ctx.env) == (True, "")
    assert not os.path.lexists(record.path)
    assert plan(records["hf:acme/big-GGUF"], env=tools.env()).action == "refuse"  # no hf


# ---------------------------------------------------------------- finding


def test_find_and_target_of() -> None:
    records = [
        ModelRecord("qwen3:8b", "ollama", "gguf", "/s/a", 5, 5, aliases=["qwen3:latest"]),
        ModelRecord("ckpt/last", "generic", "pytorch", "/a/ckpt/last.ckpt", 5, 5),
        ModelRecord("ckpt/last", "generic", "pytorch", "/b/ckpt/last.ckpt", 5, 5),
    ]
    assert find(records, "ollama:qwen3:latest") == [records[0]]
    assert find(records, "qwen3:latest") == [records[0]]
    assert find(records, "generic:ckpt/last") == records[1:]
    assert find(records, "nothing") == []
    assert target_of(records[0], records) == "ollama:qwen3:8b"
    assert target_of(records[1], records) == "/a/ckpt/last.ckpt"


# ---------------------------------------------------------------- the command


def test_dry_run_lists_and_keeps(machine: FakeMachine, tools: Tools) -> None:
    ctx = _ctx(machine, tools.env("trash", "ollama", "hf"))
    status, text = rm(ctx, "ollama:tiny:1b-4k", str(machine.files["loose"]), "--dry-run")
    assert status == 0
    assert text.startswith("Would remove 2 entries")
    assert "Runs `ollama rm tiny:1b tiny:1b-4k`" in text
    assert str(machine.files["loose"]) in text
    assert text.rstrip().endswith("Nothing was removed (--dry-run).")
    assert machine.files["loose"].exists() and "tiny:1b-4k" in tools.tags.read_text()


def test_asks_and_does_nothing_without_a_yes(machine: FakeMachine, tools: Tools) -> None:
    ctx = _ctx(machine, tools.env("trash"))
    assert rm(ctx, "generic:models/loose")[0] == 1  # no terminal, no --yes
    status, text = rm(ctx, "generic:models/loose", answer="n")
    assert status == 1 and text.rstrip().endswith("Nothing was removed.")
    assert machine.files["loose"].exists()


def test_removes_after_a_yes(machine: FakeMachine, tools: Tools) -> None:
    ctx = _ctx(machine, tools.env("trash", "ollama"))
    status, text = rm(ctx, "generic:models/loose", "tiny:1b", answer="y")
    assert status == 0, text
    assert "Removed models/loose." in text and "Removed tiny:1b." in text
    assert "freed when the Trash is emptied" in text
    assert not machine.files["loose"].exists()
    assert tools.tags.read_text() == "other:latest\n"  # both tags of the weights


def test_one_refusal_stops_everything(machine: FakeMachine, tools: Tools) -> None:
    ctx = _ctx(machine, tools.env("trash"))
    status, text = rm(ctx, "generic:models/loose", "hf:xet cache", "--yes")
    assert status == 1 and "cannot be removed" in text
    assert machine.files["loose"].exists()


def test_unknown_and_ambiguous_targets(machine: FakeMachine, tools: Tools) -> None:
    ctx = _ctx(machine, tools.env("trash"))
    status, text = rm(ctx, "no-such-model", "--yes")
    assert status == 1 and "No model matches no-such-model" in text
    for folder in ("a", "b"):
        write_gguf(machine.home / folder / "twin" / "same.gguf", tensors=(("w", (4,), GGML_F16),))
    status, text = rm(_ctx(machine, tools.env("trash")), "generic:twin/same", "--yes")
    assert status == 1 and "matches 2 entries; name one by its path" in text
    path = str(machine.home / "a" / "twin" / "same.gguf")
    status, _ = rm(_ctx(machine, tools.env("trash")), path, "--yes")
    assert status == 0 and not os.path.exists(path)
    assert (machine.home / "b" / "twin" / "same.gguf").exists()


def test_a_failing_tool_is_reported(machine: FakeMachine, tools: Tools) -> None:
    ctx = _ctx(machine, tools.env("ollama"))
    found = {f"{r.provider}:{r.name}": r for r in scan(ctx).records}
    step = plan(found["ollama:tiny:1b"], _live(machine), env=ctx.env)
    tools.tags.write_text("other:latest\n")  # gone between plan and run
    done, problem = execute(step, ctx.env)
    assert not done and "not found" in problem
