# Release Notes: v0.3.1

> Released: 2026-11-26

A fix release: the map fits the window again. The banner added in 0.2.0
pushed the map down without shrinking it, so the bottom of the map sat
below the window's edge in `modelmap serve`, the Mac app and the written
page.

## What changed

**The map fits the window.** The page now measures where the map starts
and sizes the square to the space left, so it fits at any window size,
including narrow windows where the header wraps onto two lines. In the
one-column layout of a narrow window the map is no taller than the window
either, down to a 320 px minimum.

**Smaller fixes.** In the Mac app's Help, rows of the Menus table without
a keyboard shortcut no longer leave a gap. The README screenshots show the
banner, and `scripts/screenshots.py` retakes them from example data.

## Upgrading

`pipx upgrade modelmap` or `pip install -U modelmap`; nothing needs
migrating. The Mac app: download `modelmap-0.3.1.dmg` from this release
and replace the old copy in Applications.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
