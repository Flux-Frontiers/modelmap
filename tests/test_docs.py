"""The documentation names every command, option and store the code has.

A missing name is a feature the documents do not admit exists. The check is
deliberately weak: the name has to appear on the page, not be explained
well. Links and images must point at files that exist.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from modelmap.cli import _parser
from modelmap.providers import PROVIDERS

ROOT = Path(__file__).resolve().parent.parent
GUIDE = (ROOT / "docs" / "guide.md").read_text()
CLI = (ROOT / "docs" / "cli.md").read_text()
CHEATSHEET = (ROOT / "docs" / "cheatsheet.md").read_text()
README = (ROOT / "README.md").read_text()


def _subcommands() -> dict[str, argparse.ArgumentParser]:
    parser = _parser()
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return dict(action.choices)


def _options(parser: argparse.ArgumentParser) -> list[str]:
    return [o for a in parser._actions for o in a.option_strings if o not in ("-h", "--help")]


def test_there_is_something_to_check() -> None:
    assert len(_subcommands()) == 6
    assert sum(len(_options(p)) for p in _subcommands().values()) > 30


@pytest.mark.parametrize("command", sorted(_subcommands()))
def test_every_command_and_option_is_in_the_reference(command: str) -> None:
    assert f"modelmap {command}" in CLI
    for option in _options(_subcommands()[command]):
        # The option as a whole word: `--disk` must not be satisfied by `--diskX`.
        assert re.search(rf"(?<![\w-]){re.escape(option)}(?![\w-])", CLI), f"{command} {option}"


def test_every_store_is_in_the_guide_and_the_cheat_sheet() -> None:
    for provider in PROVIDERS:
        assert f"`{provider.NAME}`" in GUIDE, provider.NAME
        assert f"`{provider.NAME}`" in CHEATSHEET, provider.NAME
        assert f"`{provider.NAME}`" in CLI, provider.NAME


@pytest.mark.parametrize(
    "name", ["README.md", "docs/guide.md", "docs/cli.md", "docs/cheatsheet.md"]
)
def test_links_and_images_point_at_files_that_exist(name: str) -> None:
    page = ROOT / name
    text = page.read_text()
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text) + re.findall(
        r'src="([^"]+)"', text
    ):
        if target.startswith("https://raw.githubusercontent.com/Flux-Frontiers/modelmap/main/"):
            # Absolute so PyPI can show it; the file must exist in the repository.
            local = ROOT / target.split("/main/", 1)[1]
        elif target.startswith("http"):
            continue
        else:
            local = (page.parent / target).resolve()
        assert local.exists(), f"{name}: {target}"


def test_section_links_exist() -> None:
    assert "## Removing models" in GUIDE and "guide.md#removing-models" in CLI
