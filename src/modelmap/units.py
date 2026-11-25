"""Sizes and counts as people write them."""

from __future__ import annotations

import re

_UNITS = ("B", "kB", "MB", "GB", "TB", "PB")
_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgtp]?)(i?)b?\s*$", re.IGNORECASE)


def format_size(size: int) -> str:
    """A byte count in decimal units, as model hubs and macOS show sizes.

    :param size: Bytes.
    :return: Text such as ``27.2 GB``, with three significant figures at most.
    """
    value = float(size)
    for unit in _UNITS:
        if value < 999.5 or unit == _UNITS[-1]:
            break
        value /= 1000
    if unit == "B":
        return f"{int(value)} B"
    return f"{value:.1f} {unit}" if value < 99.95 else f"{value:.0f} {unit}"


def parse_size(text: str) -> int:
    """Read a size such as ``50MB``, ``1.5 GB``, ``4GiB`` or ``1000``.

    :param text: A number with an optional unit. ``k``, ``M``, ``G``, ``T``
        and ``P`` are powers of 1000; with ``i`` they are powers of 1024.
    :return: Bytes.
    :raises ValueError: If the text is not a size.
    """
    match = _SIZE.match(text)
    if not match:
        raise ValueError(f"not a size: {text!r} (examples: 50MB, 1.5GB, 4GiB)")
    number, prefix, binary = match.groups()
    exponent = "kmgtp".index(prefix.lower()) + 1 if prefix else 0
    return int(float(number) * (1024 if binary else 1000) ** exponent)


def format_count(count: int | None, exact: bool = True) -> str:
    """A parameter count such as ``8.19B`` or ``~335M``; empty when unknown.

    :param count: Number of parameters, or None.
    :param exact: False marks the count as an estimate with ``~``.
    """
    if count is None:
        return ""
    for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if count >= divisor:
            value = count / divisor
            text = (
                f"{value:.2f}"
                if value < 9.995
                else f"{value:.1f}"
                if value < 99.95
                else f"{value:.0f}"
            )
            return f"{'' if exact else '~'}{text}{suffix}"
    return f"{'' if exact else '~'}{count}"
