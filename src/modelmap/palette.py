"""Colors for the map, in a light and a dark mode.

Each provider has one hue, fixed by name, so a store keeps its color
whatever else is on the machine. The hues and their order come from a
palette validated for color-vision deficiency; only its first three slots
stay distinguishable in every pairing, and a treemap can put any two
groups side by side. Color is therefore never the only cue: every group
has a row in the legend, every tooltip names its provider, and a block
wide enough carries a name tag on the map.

Loose files take a neutral that is much darker than every hue in light
mode and much lighter in dark mode. A mid gray would not do: measured with
the palette's validator, it is hard to tell from the aqua under red-green
color blindness.

Text inside a tile is white or black, whichever contrasts more with the
fill. The two ratios multiply to 21 for any fill, so the better one is at
least 4.58 and always clears the WCAG AA ratio of 4.5. A softer near-black
ink does not: on the light-mode blue it reaches only 4.46.
"""

from __future__ import annotations

MODES = ("light", "dark")

SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
TEXT_PRIMARY = {"light": "#0b0b0b", "dark": "#f0efec"}
TEXT_SECONDARY = {"light": "#52514e", "dark": "#c3c2b7"}
INK_LIGHT = "#ffffff"
INK_DARK = "#000000"

# (light, dark) fills. The three stores most often found together take the
# three slots that are distinguishable in every pairing.
PROVIDER_COLORS = {
    "hf": ("#2a78d6", "#3987e5"),  # blue
    "ollama": ("#eb6834", "#d95926"),  # orange
    "omlx": ("#1baf7a", "#199e70"),  # aqua
    "lmstudio": ("#eda100", "#c98500"),  # yellow
    "llamacpp": ("#e87ba4", "#d55181"),  # magenta
    "gpt4all": ("#008300", "#0ca30c"),  # green
    "jan": ("#6250d6", "#9085e9"),  # violet
    "whisper": ("#e34948", "#e66767"),  # red
}
# Loose files and any provider without a hue of its own: a neutral, not a
# ninth hue. Against all eight hues these measure at least 13 (color-blind
# simulation) and 21 (normal vision) on the validator's scale, where the
# floors are 8 and 15.
OTHER_COLOR = ("#484842", "#c8c8c2")


def fill(group: str, mode: str) -> str:
    """The fill color of a group in a mode."""
    return PROVIDER_COLORS.get(group, OTHER_COLOR)[MODES.index(mode)]


def _channels(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def luminance(color: str) -> float:
    """WCAG relative luminance of an ``#rrggbb`` color."""

    def linear(channel: int) -> float:
        value = channel / 255
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(c) for c in _channels(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio of two colors, from 1 to 21."""
    lighter, darker = sorted((luminance(a), luminance(b)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def ink_on(background: str) -> str:
    """White or black, whichever reads better on a background."""
    return max((INK_LIGHT, INK_DARK), key=lambda ink: contrast(ink, background))
