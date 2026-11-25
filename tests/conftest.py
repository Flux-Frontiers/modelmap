"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from modelmap.providers.base import Context
from tests.fakemachine import FakeMachine, build


@pytest.fixture
def machine(tmp_path: Path) -> Iterator[FakeMachine]:
    """A fake home directory full of model files, built fresh per test."""
    m = build(tmp_path)
    yield m
    m.cleanup()


@pytest.fixture
def ctx(machine: FakeMachine) -> Context:
    """A scan context for the fake machine: its home, an empty environment."""
    return Context(home=machine.home, env={}, platform="darwin")
