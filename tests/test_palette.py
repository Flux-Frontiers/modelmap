"""Colors: every label readable, every provider colored, both modes."""

from __future__ import annotations

import pytest

from modelmap import palette
from modelmap.providers import PROVIDERS

GROUPS = [*palette.PROVIDER_COLORS, "generic", "something-new"]
AA = 4.5  # WCAG AA for normal text


def test_contrast_reference_values() -> None:
    assert palette.contrast("#ffffff", "#000000") == pytest.approx(21)
    assert palette.contrast("#777777", "#777777") == pytest.approx(1)
    assert palette.contrast("#000000", "#ffffff") == palette.contrast("#ffffff", "#000000")
    assert palette.luminance("#ffffff") == pytest.approx(1)


@pytest.mark.parametrize("mode", palette.MODES)
@pytest.mark.parametrize("group", GROUPS)
def test_label_is_readable_on_every_fill(group: str, mode: str) -> None:
    fill = palette.fill(group, mode)
    assert palette.contrast(palette.ink_on(fill), fill) >= AA


def test_better_ink_clears_aa_on_any_fill_at_all() -> None:
    # The claim in the module docstring, checked over a grid of colors.
    for r in range(0, 256, 15):
        for g in range(0, 256, 15):
            for b in range(0, 256, 15):
                color = f"#{r:02x}{g:02x}{b:02x}"
                assert palette.contrast(palette.ink_on(color), color) >= 4.58


@pytest.mark.parametrize("mode", palette.MODES)
def test_text_is_readable_on_the_surface(mode: str) -> None:
    surface = palette.SURFACE[mode]
    assert palette.contrast(palette.TEXT_PRIMARY[mode], surface) >= AA
    assert palette.contrast(palette.TEXT_SECONDARY[mode], surface) >= AA


def test_every_provider_has_its_own_hue() -> None:
    names = {p.NAME for p in PROVIDERS}
    assert names == set(palette.PROVIDER_COLORS)
    for index in (0, 1):
        fills = [pair[index] for pair in palette.PROVIDER_COLORS.values()]
        assert len(set(fills)) == len(fills)
    # Loose files and unknown groups take the neutral, never a ninth hue.
    assert palette.fill("generic", "light") == palette.fill("something-new", "light")
    assert palette.fill("generic", "light") not in [p[0] for p in palette.PROVIDER_COLORS.values()]


@pytest.mark.parametrize("mode", palette.MODES)
def test_neutral_is_far_from_every_hue_in_lightness(mode: str) -> None:
    # A mid gray is confusable with the aqua under red-green color
    # blindness, where hue is lost and lightness is what remains.
    neutral = palette.luminance(palette.fill("generic", mode))
    for group in palette.PROVIDER_COLORS:
        hue = palette.luminance(palette.fill(group, mode))
        assert max(neutral, hue) / min(neutral, hue) >= 1.9, group
    assert palette.contrast(palette.fill("generic", mode), palette.SURFACE[mode]) >= 3
