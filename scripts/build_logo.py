#!/usr/bin/env python3
"""Draw the modelmap logo in Art Techno and write every size the repo uses.

Follows the fleet's branding rules (``kgrag_priv/brands/README.md``) and the
layout of ``vault_kg/scripts/build_logo.py``. The palette, drawing helpers
and outlined lettering come from the Art Techno brand build, so the SVGs
render the same everywhere.

Writes ``assets/modelmap_logo``, ``_icon`` and ``_banner`` as SVG and PNG,
``assets/favicon.png``, and ``src/modelmap/icon.svg``, which the map page
embeds.

Usage::

    python scripts/build_logo.py --candidates DIR   # every motif, to choose from
    python scripts/build_logo.py                    # the chosen motif, into assets/

Needs ``fonttools`` and Pillow, Google Chrome for the PNG renders, and a clone
of https://github.com/Flux-Frontiers/art_techno beside this repo. Set ``CHROME``
to the browser binary if it is not at the default macOS path. For example::

    uv run --no-project --with fonttools --with pillow python scripts/build_logo.py
"""

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
AT_BUILD = ROOT.parent / "art_techno" / "docs" / "brand" / "build.py"
sys.path.insert(0, str(ROOT / "src"))

from modelmap.treemap import Rect, squarify  # noqa: E402 (after the path is set)

_spec = importlib.util.spec_from_file_location("at_build", AT_BUILD)
at = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(at)

f, line, pad, via, diamond, clipped_rect = (
    at.f,
    at.line,
    at.pad,
    at.via,
    at.diamond,
    at.clipped_rect,
)
OBSIDIAN, PANEL, COPPER, GOLD, GREEN, MINT, SILK = (
    at.OBSIDIAN,
    at.PANEL,
    at.COPPER,
    at.GOLD,
    at.GREEN,
    at.MINT,
    at.SILK,
)

NAME = "MODELMAP"
PART = "WHERE THE WEIGHTS LIVE"
#: The motif Eric chose; None until he has.
CHOSEN: str | None = "floorplan"

# Every motif is drawn on a 512 field, centered on (256, 212), the space
# above the name.
CX, CY = 256, 212


def rect(
    x: float, y: float, w: float, h: float, fill: str, stroke: str = "", width: float = 0
) -> str:
    s = f' stroke="{stroke}" stroke-width="{f(width)}"' if stroke else ""
    return f'<rect x="{f(x)}" y="{f(y)}" width="{f(w)}" height="{f(h)}" fill="{fill}"{s}/>'


def die(x: float, y: float, s: float, core: str = SILK) -> str:
    """A square green die in a gold line, centered on ``x, y``."""
    return rect(x - s / 2, y - s / 2, s, s, GREEN, GOLD, s / 9) + rect(
        x - s / 6, y - s / 6, s / 3, s / 3, core
    )


# A: the floorplan -------------------------------------------------------------
# The map itself: a squarified treemap laid out by modelmap's own code, drawn
# as a chip floorplan. The largest block holds the die; the rest are packages.
_SIZES = (34, 20, 13, 10, 8, 6, 5, 4)


def _floorplan(x: float, y: float, side: float, gap: float, stroke: float, small: bool) -> str:
    tiles = squarify(_SIZES, Rect(x, y, side, side))
    out = []
    for n, t in enumerate(tiles):
        inner = (t.x + gap / 2, t.y + gap / 2, t.w - gap, t.h - gap)
        color = GOLD if n < 3 else COPPER
        out.append(
            f'<path d="{clipped_rect(*inner, min(inner[2], inner[3]) * 0.12)}" fill="{PANEL}" '
            f'stroke="{color}" stroke-width="{f(stroke)}"/>'
        )
        if n == 0:
            out.append(die(t.x + t.w / 2, t.y + t.h / 2, min(t.w, t.h) * 0.42))
        elif not small and min(t.w, t.h) > 34:
            out.append(via(t.x + t.w / 2, t.y + t.h / 2, min(t.w, t.h) * 0.14, color))
    return "".join(out)


def floorplan() -> str:
    return _floorplan(CX - 150, CY - 150, 300, 14, 5, False)


def floorplan_small() -> str:
    return _floorplan(9, 9, 46, 3.5, 2.2, True)


# B: the compass -------------------------------------------------------------
# A compass rose of traces at 0, 45 and 90 degrees: the four cardinal arms in
# gold ending in pads, the diagonals in copper ending in vias, an octagonal
# bezel, and the die at the center.
def _octagon(cx: float, cy: float, r: float) -> str:
    c = r * 0.414
    return clipped_rect(cx - r, cy - r, 2 * r, 2 * r, r - c)


def _compass(cx: float, cy: float, s: float, small: bool) -> str:
    out = [
        f'<path d="{_octagon(cx, cy, 118 * s)}" fill="none" stroke="{GOLD}" '
        f'stroke-width="{f(5 * s if not small else 2.2)}" stroke-opacity=".55"/>'
    ]
    arm, diag = 150 * s, 92 * s
    w = 7 * s if not small else 3.2
    for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
        out.append(
            line([(cx + dx * 30 * s, cy + dy * 30 * s), (cx + dx * arm, cy + dy * arm)], GOLD, w)
        )
        out.append(pad(cx + dx * arm, cy + dy * arm, 14 * s, GOLD))
        out.append(pad(cx + dx * arm, cy + dy * arm, 6 * s, GREEN))
    for dx, dy in ((1, -1), (1, 1), (-1, 1), (-1, -1)):
        out.append(
            line(
                [(cx + dx * 26 * s, cy + dy * 26 * s), (cx + dx * diag, cy + dy * diag)],
                COPPER,
                w * 0.7,
            )
        )
        if not small:
            out.append(via(cx + dx * diag, cy + dy * diag, 9 * s, COPPER))
    # North: a needle, the one point that is not a trace.
    if not small:
        out.append(diamond(cx, cy - 74 * s, 16 * s, GOLD))
    out.append(die(cx, cy, 52 * s))
    return "".join(out)


def compass() -> str:
    return _compass(CX, CY, 1.0, False)


def compass_small() -> str:
    return _compass(32, 32, 0.19, True)


# C: the sweep ---------------------------------------------------------------
# The scan: a die in the corner sweeps three arcs across blocks of different
# sizes; the blocks inside the swept wedge are found, lit gold.
def _sweep(x0: float, y0: float, s: float, small: bool) -> str:
    out = []
    blocks = [  # x, y, side, in the swept wedge
        (120, 30, 60, True),
        (200, 110, 44, True),
        (60, 120, 52, True),
        (230, 30, 30, False),
        (140, 200, 34, False),
        (250, 190, 26, False),
        (30, 210, 22, False),
    ]
    if small:
        blocks = blocks[:3] + blocks[3:4]
    for bx, by, side, lit in blocks:
        color = GOLD if lit else COPPER
        out.append(
            f'<path d="{clipped_rect(x0 + bx * s, y0 + by * s, side * s, side * s, side * s * 0.18)}" '
            f'fill="{PANEL}" stroke="{color}" stroke-width="{f((4 if not small else 2.6) * s / (0.19 if small else 1) * (0.19 if small else 1))}"/>'
        )
    ox, oy = x0, y0 + 300 * s
    for n, r in enumerate((110, 200, 290)):
        r *= s
        out.append(
            f'<path d="M{f(ox)} {f(oy - r)}A{f(r)} {f(r)} 0 0 1 {f(ox + r)} {f(oy)}" fill="none" '
            f'stroke="{MINT}" stroke-width="{f((3 if not small else 1.6) * (1 if not small else 1))}" '
            f'stroke-opacity="{0.9 - n * 0.2}"/>'
        )
    out.append(line([(ox, oy), (ox + 210 * s, oy - 210 * s)], GOLD, 6 * s if not small else 3))
    if not small:
        out.append(
            pad(ox + 210 * s, oy - 210 * s, 12 * s, GOLD)
            + pad(ox + 210 * s, oy - 210 * s, 5 * s, GREEN)
        )
    out.append(die(ox + 22 * s, oy - 22 * s, 48 * s))
    return "".join(out)


def sweep() -> str:
    return _sweep(CX - 150, CY - 150, 1.0, False)


def sweep_small() -> str:
    return _sweep(8, 8, 0.16, True)


MOTIFS: dict[str, tuple[str, Callable[[], str], Callable[[], str]]] = {
    "floorplan": ("The floorplan: the treemap as a chip floorplan", floorplan, floorplan_small),
    "compass": ("The compass: a rose of traces round the die", compass, compass_small),
    "sweep": ("The sweep: a scan finding blocks of every size", sweep, sweep_small),
}


def frame(w: float, h: float, inset: float, c: float, heavy: float) -> str:
    """Double frame with 45-degree corners and a green diamond at each corner."""
    out = [
        f'<path d="{clipped_rect(inset, inset, w - 2 * inset, h - 2 * inset, c)}" fill="none" stroke="{GOLD}" stroke-width="{f(heavy)}"/>',
        f'<path d="{clipped_rect(inset * 1.9, inset * 1.9, w - 3.8 * inset, h - 3.8 * inset, c * 0.75)}" fill="none" stroke="{GOLD}" stroke-width="{f(heavy * 0.4)}" stroke-opacity=".6"/>',
    ]
    d = inset * 1.45 + c * 0.4375
    for cx, cy in ((d, d), (w - d, d), (d, h - d), (w - d, h - d)):
        out.append(diamond(cx, cy, inset * 0.5, GREEN))
    return "".join(out)


def svg(w: int, h: int, title: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f"<title>{title}</title>"
        f'<rect width="{w}" height="{h}" fill="{OBSIDIAN}"/>{body}</svg>\n'
    )


def mark(motif: Callable[[], str], display, mono) -> str:
    """512 square mark: the motif, the name and the part line beneath."""
    body = frame(512, 512, 10, 26, 5) + motif()
    body += display.path(NAME, 256, 430, 40, 0.2, "middle", GOLD)
    body += mono.path(PART, 256, 462, 13.5, 0.3, "middle", MINT)
    return svg(512, 512, "modelmap mark, Art Techno", body)


def icon(small: Callable[[], str]) -> str:
    """64 icon: the reduced motif in one heavy frame, no lettering."""
    body = f'<path d="{clipped_rect(2.5, 2.5, 59, 59, 7.5)}" fill="none" stroke="{GOLD}" stroke-width="3"/>'
    return svg(64, 64, "modelmap icon, Art Techno", body + small())


def banner(motif: Callable[[], str], display, mono) -> str:
    """1280 x 320 banner: the motif left, the name and part line right."""
    s = 0.78
    body = frame(1280, 320, 10, 18, 3)
    body += (
        f'<g transform="translate({f(230 - CX * s)} {f(160 - CY * s)}) scale({s})">{motif()}</g>'
    )
    cx, room, gap = 820, 760, 34
    size = min(120, 120 * room / display.width(NAME, 120, 0.2))
    part = min(26, 26 * room / mono.width(PART, 26, 0.3))
    cap = display.cap * size / display.upm
    pcap = mono.cap * part / mono.upm
    base = (320 - (cap + gap + pcap)) / 2 + cap
    body += display.path(NAME, cx, base, size, 0.2, "middle", GOLD)
    body += mono.path(PART, cx, base + gap + pcap, part, 0.3, "middle", MINT)
    return svg(1280, 320, "modelmap banner, Art Techno", body)


def render(src: Path, out: Path, w: int, h: int) -> None:
    """Render the SVG ``src`` to ``out`` at ``w`` x ``h`` px with headless Chrome."""
    svg_w = int(src.read_text().split('width="', 1)[1].split('"', 1)[0])
    subprocess.run(
        [
            at.CHROME,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={svg_w},{svg_w * h // w}",
            f"--force-device-scale-factor={w / svg_w}",
            f"--screenshot={out}",
            src.as_uri(),
        ],
        check=True,
        capture_output=True,
    )


def resized(master: Path, dest: Path, px: int) -> None:
    Image.open(master).convert("RGB").resize((px, px), Image.LANCZOS).save(dest, optimize=True)


def write_set(name: str, out: Path, display, mono) -> dict[str, Path]:
    """Write one motif's mark, icon and banner as SVG and PNG into ``out``."""
    _, motif, small = MOTIFS[name]
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "mark": out / f"modelmap_{name}_logo.svg",
        "icon": out / f"modelmap_{name}_icon.svg",
        "banner": out / f"modelmap_{name}_banner.svg",
    }
    files["mark"].write_text(mark(motif, display, mono))
    files["icon"].write_text(icon(small))
    files["banner"].write_text(banner(motif, display, mono))
    render(files["mark"], files["mark"].with_suffix(".png"), 1024, 1024)
    render(files["banner"], files["banner"].with_suffix(".png"), 2560, 640)
    render(files["icon"], files["icon"].with_suffix(".png"), 512, 512)
    return files


def sheet(out: Path) -> Path:
    """One picture of every candidate: banner, mark, and the icon at 64 and 32 px."""
    rows = []
    for name in MOTIFS:
        banner_png = (
            Image.open(out / f"modelmap_{name}_banner.png").convert("RGB").resize((1280, 320))
        )
        mark_png = Image.open(out / f"modelmap_{name}_logo.png").convert("RGB").resize((320, 320))
        icon_png = Image.open(out / f"modelmap_{name}_icon.png").convert("RGB")
        row = Image.new("RGB", (1280 + 320 + 160, 320), OBSIDIAN)
        row.paste(banner_png, (0, 0))
        row.paste(mark_png, (1280, 0))
        row.paste(icon_png.resize((64, 64), Image.LANCZOS), (1280 + 320 + 48, 80))
        row.paste(icon_png.resize((32, 32), Image.LANCZOS), (1280 + 320 + 64, 190))
        rows.append(row)
    board = Image.new("RGB", (rows[0].width, len(rows) * 340), PANEL)
    for n, row in enumerate(rows):
        board.paste(row, (0, n * 340))
    path = out / "candidates.png"
    board.save(path, optimize=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--candidates", metavar="DIR", help="write every motif here to choose from")
    args = parser.parse_args()
    at.fetch_fonts()
    display = at.Face(at.CACHE / "JosefinSans.ttf", weight=600)
    mono = at.Face(at.CACHE / "IBMPlexMono-Medium.ttf")

    if args.candidates:
        out = Path(args.candidates)
        for name in MOTIFS:
            write_set(name, out, display, mono)
        print(sheet(out))
        return
    if CHOSEN is None:
        parser.error("no motif chosen yet: run with --candidates and set CHOSEN")
    assets = ROOT / "assets"
    files = write_set(CHOSEN, assets, display, mono)
    for kind, path in files.items():
        for suffix in (".svg", ".png"):
            path.with_suffix(suffix).rename(
                assets / f"modelmap_{kind if kind != 'mark' else 'logo'}{suffix}"
            )
    with tempfile.TemporaryDirectory() as tmp:
        icon_png = Path(tmp) / "icon.png"
        render(assets / "modelmap_icon.svg", icon_png, 1024, 1024)
        resized(icon_png, assets / "favicon.png", 32)
    # The page embeds the icon, so it ships inside the package.
    shutil.copyfile(assets / "modelmap_icon.svg", ROOT / "src" / "modelmap" / "icon.svg")


if __name__ == "__main__":
    main()
