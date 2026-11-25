"""Squarified treemap layout: pure geometry, no drawing and no files.

The algorithm is the one in Bruls, Huizing and van Wijk, "Squarified
Treemaps" (2000): take the items in decreasing order and fill the
rectangle with rows laid along its shorter side, adding items to the
current row for as long as that does not worsen the row's worst aspect
ratio.

The area of every rectangle is its share of the total, exactly: nothing
here insets, pads or rounds. A drawing layer that wants gaps must draw
them on top of these rectangles, not shrink them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle; ``y`` grows downward, as in SVG."""

    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect(self) -> float:
        """Long side over short side; 1 for a square, infinite when flat."""
        short, long = sorted((self.w, self.h))
        return long / short if short > 0 else float("inf")


def worst(largest: float, smallest: float, total: float, side: float) -> float:
    """The highest aspect ratio in a row, as the paper defines it.

    :param largest: Largest area in the row.
    :param smallest: Smallest area in the row.
    :param total: Sum of the areas in the row.
    :param side: Length of the side the row is laid along.
    """
    return max(side * side * largest / (total * total), total * total / (side * side * smallest))


def squarify(sizes: Sequence[float], rect: Rect) -> list[Rect]:
    """Divide a rectangle among sizes, keeping the pieces close to square.

    :param sizes: Positive sizes, in any order.
    :param rect: The rectangle to fill.
    :return: One rectangle per size, in the order of ``sizes``. Their areas
        are in proportion to the sizes and together they tile ``rect``.
    :raises ValueError: If a size is not positive.
    """
    if any(not size > 0 for size in sizes):
        raise ValueError("sizes must be positive")
    if not sizes:
        return []
    order = sorted(range(len(sizes)), key=lambda i: (-sizes[i], i))
    scale = rect.area / sum(sizes)
    areas = [sizes[i] * scale for i in order]
    placed: list[Rect] = []
    x, y, w, h = rect.x, rect.y, rect.w, rect.h
    start = 0
    while start < len(areas):
        side = min(w, h)
        end = start + 1
        total = areas[start]
        ratio = worst(areas[start], areas[start], total, side)
        while end < len(areas):
            grown = total + areas[end]
            # The row is in decreasing order: the first is its largest.
            grown_ratio = worst(areas[start], areas[end], grown, side)
            if grown_ratio > ratio:
                break
            total, ratio, end = grown, grown_ratio, end + 1

        last_row = end == len(areas)
        if w >= h:
            # A column on the left; the last row takes all that is left,
            # so rounding never leaves a sliver uncovered.
            thick = w if last_row else total / h
            offset = y
            for k in range(start, end):
                length = (y + h - offset) if k == end - 1 else areas[k] / thick
                placed.append(Rect(x, offset, thick, length))
                offset += length
            x, w = x + thick, w - thick
        else:
            thick = h if last_row else total / w
            offset = x
            for k in range(start, end):
                length = (x + w - offset) if k == end - 1 else areas[k] / thick
                placed.append(Rect(offset, y, length, thick))
                offset += length
            y, h = y + thick, h - thick
        start = end

    result = [placed[0]] * len(sizes)
    for position, index in enumerate(order):
        result[index] = placed[position]
    return result


@dataclass(frozen=True)
class Item:
    """One thing to place on the map.

    :param key: Unique identity, for example a path.
    :param group: Name of the group the item belongs to.
    :param size: Size that sets the tile's area; must be positive.
    :param label: Text for sorting ties and for display.
    """

    key: str
    group: str
    size: float
    label: str = ""


@dataclass
class Tile:
    """A placed item, or several small items merged into one tile.

    :param key: The item's key; for a merged tile, a key made from the group.
    :param group: Group name.
    :param size: The item's size, or the sum for a merged tile.
    :param rect: Where the tile is.
    :param label: The item's label; empty for a merged tile.
    :param members: Keys of the items a merged tile stands for; empty for a
        tile that is one item.
    """

    key: str
    group: str
    size: float
    rect: Rect
    label: str = ""
    members: list[str] = field(default_factory=list)


@dataclass
class Group:
    """A group of tiles and the rectangle they fill."""

    name: str
    size: float
    rect: Rect
    tiles: list[Tile] = field(default_factory=list)


def layout(items: Sequence[Item], rect: Rect, *, min_fraction: float = 0.0) -> list[Group]:
    """Lay out items in two levels: groups, then the items of each group.

    :param items: Items with positive sizes and unique keys.
    :param rect: The canvas.
    :param min_fraction: Items smaller than this share of the whole are
        merged into one tile per group, when a group has two or more of
        them, so that every tile drawn is large enough to see.
    :return: Groups from largest to smallest, each with its tiles from
        largest to smallest. Equal sizes are ordered by label, then key,
        so the same items always give the same layout.
    """
    total = sum(item.size for item in items)
    by_group: dict[str, list[Item]] = {}
    for item in items:
        by_group.setdefault(item.group, []).append(item)
    names = sorted(by_group, key=lambda name: (-sum(i.size for i in by_group[name]), name))
    group_sizes = [sum(i.size for i in by_group[name]) for name in names]
    groups = []
    for name, size, group_rect in zip(names, group_sizes, squarify(group_sizes, rect)):
        members = sorted(by_group[name], key=lambda i: (-i.size, i.label, i.key))
        small = [i for i in members if i.size < min_fraction * total]
        if len(small) < 2:
            small = []
        shown = members[: len(members) - len(small)]
        tiles = [Tile(i.key, name, i.size, group_rect, i.label) for i in shown]
        if small:
            merged = Tile(
                key=f"{name}\x00merged",
                group=name,
                size=sum(i.size for i in small),
                rect=group_rect,
                members=[i.key for i in small],
            )
            tiles.append(merged)
        for tile, tile_rect in zip(tiles, squarify([t.size for t in tiles], group_rect)):
            tile.rect = tile_rect
        tiles.sort(key=lambda t: (-t.size, bool(t.members), t.label, t.key))
        groups.append(Group(name, size, group_rect, tiles))
    return groups
