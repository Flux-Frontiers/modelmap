"""Treemap layout: the paper's worked example, and properties of any input."""

from __future__ import annotations

from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelmap.treemap import Item, Rect, layout, squarify, worst

CANVAS = Rect(0, 0, 1000, 1000)
# Each tile's area may be off by this share of the canvas, no more. A
# per-tile relative bound is the wrong test: the last tile in a row takes
# what is left, so its error is that of the whole row.
TOLERANCE = 1e-9


def _overlap(a: Rect, b: Rect) -> float:
    w = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    h = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    return max(w, 0.0) * max(h, 0.0)


def _check_tiling(sizes: list[int], rect: Rect, rects: list[Rect]) -> None:
    """Areas in proportion, everything inside, nothing overlapping."""
    assert len(rects) == len(sizes)
    total = sum(sizes)
    slack = TOLERANCE * rect.area
    edge = TOLERANCE * max(rect.w, rect.h)
    for size, r in zip(sizes, rects):
        assert abs(r.area - size / total * rect.area) <= slack
        assert r.w >= 0 and r.h >= 0
        assert r.x >= rect.x - edge and r.y >= rect.y - edge
        assert r.x + r.w <= rect.x + rect.w + edge
        assert r.y + r.h <= rect.y + rect.h + edge
    assert abs(sum(r.area for r in rects) - rect.area) <= slack * max(len(rects), 1)


# ------------------------------------------------------- the paper's example


def test_worst_matches_the_ratios_printed_in_the_paper() -> None:
    # Bruls, Huizing, van Wijk (2000), section 3.1 and figure 4: a 6 by 4
    # rectangle and areas 6, 6, 4, 3, 2, 2, 1. Each step's ratio as printed.
    steps = [
        # (largest, smallest, row total, side, ratio)
        (6, 6, 6, 4, Fraction(8, 3)),  # step 1
        (6, 6, 12, 4, Fraction(3, 2)),  # step 2
        (6, 4, 16, 4, Fraction(4, 1)),  # step 3, rejected
        (4, 4, 4, 3, Fraction(9, 4)),  # step 4
        (4, 3, 7, 3, Fraction(49, 27)),  # step 5
        (4, 2, 9, 3, Fraction(9, 2)),  # step 6, rejected
        (2, 2, 2, Fraction(5, 3), Fraction(25, 18)),  # step 7
        (2, 2, 4, Fraction(5, 3), Fraction(144, 50)),  # step 8, rejected
        (1, 1, 1, Fraction(5, 3), Fraction(25, 9)),  # step 10
    ]
    for largest, smallest, total, side, ratio in steps:
        assert (
            worst(Fraction(largest), Fraction(smallest), Fraction(total), Fraction(side)) == ratio
        )


def test_layout_of_the_example_in_the_paper() -> None:
    rects = squarify([6, 6, 4, 3, 2, 2, 1], Rect(0, 0, 6, 4))
    shapes = [(pytest.approx(r.w), pytest.approx(r.h)) for r in rects]
    assert shapes == [
        (3, 2),  # the two sixes fill the left half
        (3, 2),
        (12 / 7, 7 / 3),  # 4 and 3 share a row across the right half
        (9 / 7, 7 / 3),
        (6 / 5, 5 / 3),  # 2, 2 and 1 stand side by side in what is left
        (6 / 5, 5 / 3),
        (3 / 5, 5 / 3),
    ]
    # Rows: sixes in one column, 4 and 3 level with each other, the rest level.
    assert rects[0].x == rects[1].x == 0
    assert rects[2].y == rects[3].y and rects[2].x == pytest.approx(3)
    assert rects[4].y == rects[5].y == rects[6].y
    assert max(r.aspect for r in rects) == pytest.approx(25 / 9)
    _check_tiling([6, 6, 4, 3, 2, 2, 1], Rect(0, 0, 6, 4), rects)


# ---------------------------------------------------------------- known shapes


@pytest.mark.parametrize("k", [1, 2, 3, 4, 7])
def test_square_number_of_equal_sizes_gives_squares(k: int) -> None:
    rects = squarify([5] * (k * k), CANVAS)
    for r in rects:
        assert r.w == pytest.approx(1000 / k) and r.h == pytest.approx(1000 / k)


def test_no_items_one_item_two_items() -> None:
    assert squarify([], CANVAS) == []
    assert squarify([42], CANVAS) == [CANVAS]
    left, right = squarify([3, 1], CANVAS)
    assert (left.x, left.w, left.h) == (0, pytest.approx(750), 1000)
    assert (right.x, right.w) == (pytest.approx(750), pytest.approx(250))


def test_result_follows_input_order_not_size_order() -> None:
    small, big = squarify([1, 9], CANVAS)
    assert small.area == pytest.approx(100_000) and big.area == pytest.approx(900_000)


@pytest.mark.parametrize("sizes", [[1, 0], [1, -2], [float("nan")], [0]])
def test_sizes_must_be_positive(sizes: list[float]) -> None:
    with pytest.raises(ValueError, match="positive"):
        squarify(sizes, CANVAS)


def test_canvas_need_not_be_square_or_at_the_origin() -> None:
    rect = Rect(10, 20, 300, 50)
    sizes = [5, 4, 3, 2, 1]
    _check_tiling(sizes, rect, squarify(sizes, rect))


def test_extreme_ratio() -> None:
    sizes = [10**9, 1]
    rects = squarify(sizes, CANVAS)
    _check_tiling(sizes, CANVAS, rects)
    assert rects[1].area == pytest.approx(1e6 / (10**9 + 1), rel=1e-6)


def test_ten_thousand_items() -> None:
    sizes = [(i * 7919) % 1000 + 1 for i in range(10_000)]
    _check_tiling(sizes, CANVAS, squarify(sizes, CANVAS))


# ------------------------------------------------------------------ properties

SIZES = st.lists(st.integers(min_value=1, max_value=10**9), min_size=1, max_size=40)


@given(SIZES)
@settings(max_examples=300, deadline=None)
def test_any_sizes_tile_the_canvas_in_proportion(sizes: list[int]) -> None:
    rects = squarify(sizes, CANVAS)
    _check_tiling(sizes, CANVAS, rects)
    slack = TOLERANCE * CANVAS.area
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            assert _overlap(a, b) <= slack


@given(SIZES)
@settings(max_examples=100, deadline=None)
def test_layout_is_deterministic(sizes: list[int]) -> None:
    assert squarify(sizes, CANVAS) == squarify(list(sizes), CANVAS)


@given(SIZES, st.floats(min_value=0.1, max_value=10))
@settings(max_examples=100, deadline=None)
def test_any_canvas_shape(sizes: list[int], stretch: float) -> None:
    rect = Rect(-5, 3, 1000 * stretch, 1000 / stretch)
    _check_tiling(sizes, rect, squarify(sizes, rect))


# ---------------------------------------------------------------------- groups


def _items(spec: dict[str, list[int]]) -> list[Item]:
    return [
        Item(key=f"/{group}/{n}", group=group, size=size, label=f"{group}-{n}")
        for group, sizes in spec.items()
        for n, size in enumerate(sizes)
    ]


def test_groups_and_tiles_are_in_proportion_across_groups() -> None:
    items = _items({"hf": [50, 30, 20], "ollama": [60, 5], "omlx": [35]})
    groups = layout(items, CANVAS)
    assert [g.name for g in groups] == ["hf", "ollama", "omlx"]
    total = sum(i.size for i in items)
    sizes = {i.key: i.size for i in items}
    for group in groups:
        assert group.rect.area == pytest.approx(group.size / total * CANVAS.area)
        assert [t.size for t in group.tiles] == sorted((t.size for t in group.tiles), reverse=True)
        for tile in group.tiles:
            # The same scale for every tile on the map, whatever its group.
            assert tile.rect.area == pytest.approx(sizes[tile.key] / total * CANVAS.area)
            assert _overlap(tile.rect, group.rect) == pytest.approx(tile.rect.area)


def test_small_items_are_merged_per_group() -> None:
    items = _items({"big": [1000, 2, 1, 1], "other": [500, 3]})
    groups = {g.name: g for g in layout(items, CANVAS, min_fraction=0.01)}

    big = groups["big"].tiles
    assert [t.size for t in big] == [1000, 4]
    assert big[1].members == ["/big/1", "/big/2", "/big/3"] and big[1].label == ""
    assert big[1].rect.area == pytest.approx(4 / 1507 * CANVAS.area)
    # One small item on its own is left alone: merging it would change nothing.
    assert [(t.size, t.members) for t in groups["other"].tiles] == [(500, []), (3, [])]


def test_merging_never_loses_or_invents_size() -> None:
    items = _items({"a": [100, 1, 1, 1, 1], "b": [1, 1], "c": [7]})
    groups = layout(items, CANVAS, min_fraction=0.05)
    assert sum(t.size for g in groups for t in g.tiles) == sum(i.size for i in items)
    keys = [k for g in groups for t in g.tiles for k in (t.members or [t.key])]
    assert sorted(keys) == sorted(i.key for i in items)
    assert sum(t.rect.area for g in groups for t in g.tiles) == pytest.approx(CANVAS.area)


def test_ties_are_broken_by_label_then_key() -> None:
    a = Item("/z", "g", 5, "alpha")
    b = Item("/a", "g", 5, "beta")
    c = Item("/b", "g", 5, "alpha")
    for order in ([a, b, c], [c, b, a], [b, a, c]):
        (group,) = layout(order, CANVAS)
        assert [t.key for t in group.tiles] == ["/b", "/z", "/a"]
    first = layout([a, b, c], CANVAS)[0].tiles
    again = layout([c, a, b], CANVAS)[0].tiles
    assert [(t.key, t.rect) for t in first] == [(t.key, t.rect) for t in again]


def test_empty_layout() -> None:
    assert layout([], CANVAS) == []
