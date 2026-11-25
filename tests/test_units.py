"""Sizes and counts as text."""

from __future__ import annotations

import pytest

from modelmap.units import format_count, format_size, parse_size


@pytest.mark.parametrize(
    ("size", "text"),
    [
        (0, "0 B"),
        (999, "999 B"),
        (1000, "1.0 kB"),
        (36_685_200, "36.7 MB"),
        (999_400_000, "999 MB"),
        (999_600_000, "1.0 GB"),
        (27_200_000_000, "27.2 GB"),
        (109_045_000_000, "109 GB"),
        (1_800_000_000_000, "1.8 TB"),
        (5 * 10**18, "5000 PB"),
    ],
)
def test_format_size(size: int, text: str) -> None:
    assert format_size(size) == text


@pytest.mark.parametrize(
    ("text", "size"),
    [
        ("1000", 1000),
        ("50MB", 50_000_000),
        ("50 mb", 50_000_000),
        ("1.5GB", 1_500_000_000),
        ("4GiB", 4 * 1024**3),
        ("2k", 2000),
        ("1T", 10**12),
        ("7b", 7),
    ],
)
def test_parse_size(text: str, size: int) -> None:
    assert parse_size(text) == size


@pytest.mark.parametrize("text", ["", "big", "50XB", "-5MB", "1.2.3GB"])
def test_parse_size_rejects_nonsense(text: str) -> None:
    with pytest.raises(ValueError, match="not a size"):
        parse_size(text)


@pytest.mark.parametrize(
    ("count", "exact", "text"),
    [
        (None, True, ""),
        (512, True, "512"),
        (33_200_000, True, "33.2M"),
        (334_000_000, True, "334M"),
        (8_191_000_000, True, "8.19B"),
        (27_357_000_000, False, "~27.4B"),
        (1_500_000_000_000, True, "1.50T"),
        (4_160, False, "~4.16K"),
    ],
)
def test_format_count(count: int | None, exact: bool, text: str) -> None:
    assert format_count(count, exact) == text
