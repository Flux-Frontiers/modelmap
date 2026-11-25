"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.fakemachine import FakeMachine, build


@pytest.fixture
def machine(tmp_path: Path) -> Iterator[FakeMachine]:
    """A fake home directory full of model files, built fresh per test."""
    m = build(tmp_path)
    yield m
    m.cleanup()
