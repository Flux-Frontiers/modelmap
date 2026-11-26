# Release Notes: v0.3.1

> Released: 2026-11-26

The map fits the window again, and the banner follows your light or dark
setting. The banner added in 0.2.0 pushed the map down without shrinking
it, so the bottom of the map sat below the window's edge in
`modelmap serve`, the Mac app and the written page.

## What changed

**The map fits the window.** The page now measures where the map starts
and sizes the square to the space left, so it fits at any window size,
including narrow windows where the header wraps onto two lines. In the
one-column layout of a narrow window the map is no taller than the window
either. It keeps a minimum size, so a very short window still shows it.

**A banner for light surroundings.** The Art Techno banner now has a paper
version: the same drawing on a light ground, with the name in iron and the
motif in deep gold and deep copper. The map page shows it in a light theme
and the dark original in a dark one, and the README and the documentation
site switch the same way.

**Smaller fixes.** In the Mac app's Help, rows of the Menus table without
a keyboard shortcut no longer leave a gap. The README screenshots show the
banner, and `scripts/screenshots.py` retakes them from example data.

## Upgrading

`pipx upgrade modelmap` or `pip install -U modelmap`; nothing needs
migrating. The Mac app: download `modelmap-0.3.1.dmg` from this release
and replace the old copy in Applications.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
