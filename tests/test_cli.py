"""The command line, run in-process against the fake machine."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from modelmap import __version__
from modelmap.cli import main
from modelmap.providers.base import Context
from modelmap.records import ModelRecord, Scan
from tests.fakemachine import FakeMachine


def run(ctx: Context, *argv: str) -> tuple[int, str]:
    out = io.StringIO()
    status = main(list(argv), ctx=ctx, out=out)
    return status, out.getvalue()


def test_scan_prints_a_summary(machine: FakeMachine, ctx: Context) -> None:
    status, text = run(ctx, "scan")
    assert status == 0
    lines = text.splitlines()
    assert lines[0].split() == ["PROVIDER", "SIZE", "ON", "DISK", "MODELS", "OTHER", "ISSUES"]
    providers = [
        line.split()[0] for line in lines[1:] if line and not line.startswith(("warning", " "))
    ]
    assert providers[0] == "hf"  # largest first
    assert {"hf", "omlx", "ollama", "lmstudio", "llamacpp", "generic", "total"} <= set(providers)
    assert "locations scanned in" in text


def test_scan_json_is_a_readable_inventory(machine: FakeMachine, ctx: Context) -> None:
    status, text = run(ctx, "scan", "--json")
    assert status == 0
    inventory = Scan.from_json(text)
    assert inventory.tool_version == __version__
    assert any(r.name == "tiny:1b" for r in inventory.records)


def test_inventory_from_another_machine_lists_the_same(
    machine: FakeMachine, ctx: Context, tmp_path: Path
) -> None:
    _, text = run(ctx, "scan", "--json")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(text)
    _, direct = run(Context(machine.home, {}, "darwin"), "list")
    _, from_file = run(Context(tmp_path, {}, "linux"), "list", "--from-json", str(inventory))
    assert from_file == direct


def test_list_shows_models_grouped_and_sorted(machine: FakeMachine, ctx: Context) -> None:
    status, text = run(ctx, "list", "--no-sweep")
    assert status == 0
    lines = text.splitlines()
    group_headers = [
        line for line in lines if line and "records" in line and "  " in line and "SIZE" not in line
    ]
    assert group_headers[0].startswith("hf  ")  # the largest provider first
    hf_rows = lines[lines.index(group_headers[0]) + 2 :]
    assert hf_rows[0].split()[:2] == ["70.0", "GB"] and "acme/big-GGUF" in hf_rows[0]
    assert "also tiny:1b-4k" in text
    assert "! broken link: broken.bin" in text
    assert "auxiliary" in text  # the projector
    assert "entries not shown; use --all" in text
    assert "acme/words" not in text


def test_list_all_sort_and_group(machine: FakeMachine, ctx: Context) -> None:
    _, text = run(ctx, "list", "--no-sweep", "--all", "--group", "none", "--sort", "name")
    header = text.splitlines()[0]
    start, end = header.index("NAME"), header.index("NOTES")
    names = [
        line[start:end].strip() for line in text.splitlines()[1:] if line[:1] != "" and line.strip()
    ]
    names = [n for n in names if n and "entries not shown" not in n]
    assert "acme/words" in text and "xet cache" in text
    assert text.count("SIZE") == 1
    assert names == sorted(names, key=str.lower)

    _, by_format = run(
        Context(machine.home, {}, "darwin"), "list", "--no-sweep", "--group", "format"
    )
    assert "\ngguf  " in by_format and "\nsafetensors  " in by_format

    _, by_params = run(
        Context(machine.home, {}, "darwin"),
        "list",
        "--no-sweep",
        "--group",
        "none",
        "--sort",
        "params",
    )
    assert "acme/big-GGUF" in by_params.splitlines()[1]  # 35B parameters


def test_zero_size_shared_record_says_shared(
    machine: FakeMachine, ctx: Context, tmp_path: Path
) -> None:
    result = Scan(
        records=[
            ModelRecord("copy", "lmstudio", "gguf", "/x", 0, 0, shared_with=["hf:org/original"]),
        ]
    )
    inventory = tmp_path / "i.json"
    inventory.write_text(result.to_json())
    _, text = run(ctx, "list", "--from-json", str(inventory))
    row = text.splitlines()[-1]
    assert row.split()[0] == "shared"
    assert "shared with hf:org/original" in row


def test_names_that_are_not_utf8_print_and_serialize(ctx: Context, tmp_path: Path) -> None:
    result = Scan(
        records=[ModelRecord("caf\udcff", "generic", "gguf", "/tmp/caf\udcff.gguf", 5, 5)]
    )
    inventory = tmp_path / "i.json"
    inventory.write_text(result.to_json())
    assert Scan.from_json(inventory.read_text()).records[0].name == "caf\udcff"
    _, text = run(ctx, "list", "--from-json", str(inventory))
    text.encode("utf-8")  # must not raise
    assert "caf\\udcff" in text


def test_providers_shows_where_each_store_is_looked_for(machine: FakeMachine, ctx: Context) -> None:
    status, text = run(ctx, "providers")
    assert status == 0
    rows = {line.split()[0]: line for line in text.splitlines()[1:]}
    assert f"{machine.ollama_link} -> " in rows["ollama"] and " yes " in rows["ollama"]
    assert " no " in rows["whisper"]
    assert "settings.json" in rows["omlx"]


def test_scan_options_reach_the_scanner(machine: FakeMachine, ctx: Context) -> None:
    _, text = run(
        ctx, "scan", "--json", "--no-defaults", "--max-depth", "0", str(machine.home / "models")
    )
    assert [r.name for r in Scan.from_json(text).records] == ["models/loose"]

    fresh = Context(machine.home, {}, "darwin")
    _, text = run(fresh, "scan", "--json", "--only", "ollama,omlx", "--no-sweep")
    assert {r.provider for r in Scan.from_json(text).records} == {"ollama", "omlx"}

    fresh = Context(machine.home, {}, "darwin")
    _, text = run(
        fresh, "scan", "--json", "--no-defaults", "--min-size", "1GB", str(machine.home / "ckpt")
    )
    assert Scan.from_json(text).records == []

    fresh = Context(machine.home, {}, "darwin")
    _, text = run(
        fresh,
        "scan",
        "--json",
        "--no-defaults",
        "--exclude",
        "loose*",
        str(machine.home / "models"),
    )
    assert Scan.from_json(text).records == []


def test_redact(machine: FakeMachine, ctx: Context) -> None:
    _, text = run(ctx, "scan", "--json", "--redact")
    assert str(machine.home) not in text
    assert json.loads(text)["hostname"] is None
    assert '"~/.omlx/models' in text


def test_strict_exit_status(machine: FakeMachine, ctx: Context) -> None:
    clean = str(machine.home / "models")
    assert run(ctx, "scan", "--no-defaults", "--strict", clean)[0] == 0
    fresh = Context(machine.home, {}, "darwin")
    assert run(fresh, "scan", "--no-defaults", "--strict", str(machine.hostile))[0] == 1
    fresh = Context(machine.home, {}, "darwin")
    assert run(fresh, "scan", "--no-defaults", str(machine.hostile))[0] == 0


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["frobnicate"],
        ["scan", "--min-size", "huge"],
        ["list", "--sort", "color"],
        ["list", "--from-json", "/nonexistent/inventory.json"],
        ["scan", "--no-defaults"],
        ["list", "--no-defaults"],
    ],
)
def test_usage_errors_exit_2(
    argv: list[str], ctx: Context, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(argv, ctx=ctx)
    assert raised.value.code == 2
    assert "modelmap" in capsys.readouterr().err


def test_bad_inventory_is_a_usage_error(
    ctx: Context, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "old.json"
    bad.write_text(json.dumps({"schema_version": 99}))
    with pytest.raises(SystemExit) as raised:
        main(["list", "--from-json", str(bad)], ctx=ctx)
    assert raised.value.code == 2
    assert "schema version" in capsys.readouterr().err


def test_version_and_module_entry_point() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "modelmap", "--version"], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == f"modelmap {__version__}"


# ------------------------------------------------------------------------- map


def test_map_writes_the_interactive_page(
    machine: FakeMachine, ctx: Context, tmp_path: Path
) -> None:
    target = tmp_path / "out" / "map.html"
    target.parent.mkdir()
    status, text = run(ctx, "map", "--no-sweep", "-o", str(target))
    assert status == 0 and text.strip() == f"Wrote {target}"
    html = target.read_text()
    assert html.startswith("<!doctype html>") and "modelmap rm ollama:tiny:1b" in html


def test_map_default_file_names(
    machine: FakeMachine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run(Context(machine.home, {}, "darwin"), "map", "--only", "ollama")
    run(Context(machine.home, {}, "darwin"), "map", "--only", "ollama", "--svg")
    assert (tmp_path / "modelmap.html").read_text().startswith("<!doctype html>")
    assert (tmp_path / "modelmap.svg").read_text().startswith("<svg ")


def test_map_from_an_inventory_made_elsewhere(
    machine: FakeMachine, ctx: Context, tmp_path: Path
) -> None:
    _, text = run(ctx, "scan", "--json", "--redact")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(text)
    target = tmp_path / "theirs.html"
    status, _ = run(
        Context(tmp_path, {}, "linux"), "map", "--from-json", str(inventory), "-o", str(target)
    )
    html = target.read_text()
    assert status == 0 and "tiny:1b" in html
    assert str(machine.home) not in html


def test_map_disk_changes_the_areas(tmp_path: Path, ctx: Context) -> None:
    scan_result = Scan(
        records=[
            ModelRecord("a", "hf", "gguf", "/a", 100, 10),
            ModelRecord("b", "hf", "gguf", "/b", 100, 90),
        ]
    )
    inventory = tmp_path / "i.json"
    inventory.write_text(scan_result.to_json())
    run(ctx, "map", "--from-json", str(inventory), "--svg", "-o", str(tmp_path / "size.svg"))
    run(
        ctx,
        "map",
        "--from-json",
        str(inventory),
        "--svg",
        "--disk",
        "-o",
        str(tmp_path / "disk.svg"),
    )
    assert 'data-size="100"' in (tmp_path / "size.svg").read_text()
    assert 'data-size="90"' in (tmp_path / "disk.svg").read_text()


def test_map_open_hands_the_file_to_the_browser(
    tmp_path: Path, ctx: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    # The browser is outside the program: the one thing replaced here.
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    inventory = tmp_path / "i.json"
    inventory.write_text(Scan().to_json())
    target = tmp_path / "m.html"
    run(ctx, "map", "--from-json", str(inventory), "-o", str(target))
    assert opened == []
    run(ctx, "map", "--from-json", str(inventory), "-o", str(target), "--open")
    assert opened == [f"file://{target}"]
    spaced = tmp_path / "my maps" / "m.html"
    spaced.parent.mkdir()
    run(ctx, "map", "--from-json", str(inventory), "-o", str(spaced), "--open")
    assert opened[-1].endswith("/my%20maps/m.html")


@pytest.mark.parametrize(
    "extra", [["--redact"], ["/some/path"], ["--only", "hf"], ["--no-sweep"], ["--max-depth", "2"]]
)
@pytest.mark.parametrize("command", ["list", "map"])
def test_inventory_is_read_as_it_is(
    command: str, extra: list[str], ctx: Context, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # An option that would be silently ignored is refused: a page made with
    # --from-json --redact would keep its real paths.
    inventory = tmp_path / "i.json"
    inventory.write_text(Scan().to_json())
    with pytest.raises(SystemExit) as raised:
        main([command, "--from-json", str(inventory), *extra], ctx=ctx)
    assert raised.value.code == 2
    assert "reads an inventory as it is" in capsys.readouterr().err
