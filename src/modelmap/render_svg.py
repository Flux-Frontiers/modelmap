"""Draw a scan as one static SVG: a square treemap with a header and legend.

The square is divided among the records by size. Tiles of one provider
form one block. Each ``<rect>`` carries the exact layout geometry (to a
thousandth of a unit); the gaps between tiles are strokes in the surface
color drawn over the edges, so no tile is shrunk to make room.

The picture adapts to a light or dark setting through
``prefers-color-scheme``. Hovering a tile in a browser shows its details
(the SVG ``<title>``).
"""

from __future__ import annotations

from modelmap import palette
from modelmap.mapdata import MapData, build
from modelmap.records import ModelRecord, Scan
from modelmap.treemap import Group, Rect, Tile, layout
from modelmap.units import format_count, format_size

MARGIN = 24
HEADER = 84
LEGEND_ROW = 22
LEGEND_COLUMNS = 3
# Tiles under a quarter of a percent of the map are merged per provider.
MIN_FRACTION = 0.0025
# Width of an average character as a share of the font size, for deciding
# what fits. Labels are also clipped to their tile, so a wrong guess can
# cost a few letters but never overflow.
_CHAR = 0.58
_PAD = 7
_TAG_FONT = 10
_TAG_LINE = 15

DISPLAY_NAMES = {
    "hf": "Hugging Face",
    "ollama": "Ollama",
    "omlx": "oMLX",
    "lmstudio": "LM Studio",
    "llamacpp": "llama.cpp",
    "gpt4all": "GPT4All",
    "jan": "Jan",
    "whisper": "Whisper",
    "generic": "Loose files",
}


def _xml(text: object) -> str:
    """Text safe inside XML: escaped, with characters XML forbids replaced."""
    out = []
    for char in str(text):
        code = ord(char)
        legal = code in (9, 10, 13) or 0x20 <= code <= 0xD7FF or 0xE000 <= code <= 0xFFFD
        legal = legal or code >= 0x10000
        out.append(char if legal else "?")
    return (
        "".join(out)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _num(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _fit(text: str, width: float, font: float) -> str:
    """The text, shortened with ``...`` to fit a width; empty if it cannot."""
    room = int(width / (font * _CHAR))
    if len(text) <= room:
        return text
    return text[: room - 3] + "..." if room >= 6 else ""


def _short_name(record: ModelRecord) -> str:
    """The last part of a name: what tells models apart in a narrow tile."""
    return record.name.rsplit("/", 1)[-1]


def _tooltip(record: ModelRecord, size: int, total: int, notes: list[ModelRecord]) -> str:
    lines = [record.name, DISPLAY_NAMES.get(record.provider, record.provider)]
    facts = [record.format, record.quant, format_count(record.params, record.params_exact)]
    facts = [fact for fact in facts if fact]
    if record.params:
        facts[-1] += " parameters"
    if record.kind != "model":
        facts.insert(0, record.kind)
    lines.append(", ".join(facts))
    lines.append(f"{format_size(size)}, {100 * size / total:.1f}% of the map")
    if record.aliases:
        lines.append("Also: " + ", ".join(record.aliases))
    lines += [f"Same files as: {n.name} ({n.provider})" for n in notes]
    if record.shared_with:
        lines.append("Shares files with: " + ", ".join(record.shared_with))
    lines += [f"Problem: {issue}" for issue in record.issues]
    lines.append(record.path)
    return "\n".join(lines)


def _css(groups: list[Group]) -> list[str]:
    """Style rules: one set of color variables per mode."""

    def variables(mode: str) -> str:
        surface = palette.SURFACE[mode]
        parts = [
            f"--surface:{surface}",
            f"--text:{palette.TEXT_PRIMARY[mode]}",
            f"--text2:{palette.TEXT_SECONDARY[mode]}",
        ]
        for index, group in enumerate(groups):
            color = palette.fill(group.name, mode)
            parts += [f"--c{index}:{color}", f"--i{index}:{palette.ink_on(color)}"]
        return ";".join(parts)

    return [
        "<style>",
        f".modelmap{{{variables('light')}}}",
        f"@media (prefers-color-scheme:dark){{.modelmap{{{variables('dark')}}}}}",
        ".modelmap text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif}",
        ".modelmap .bg{fill:var(--surface)}",
        ".modelmap .tile{stroke:var(--surface);stroke-width:2}",
        ".modelmap .block{fill:none;stroke:var(--surface);stroke-width:6}",
        ".modelmap .hero{fill:var(--text);font-size:34px;font-weight:600}",
        ".modelmap .sub{fill:var(--text2);font-size:13px}",
        ".modelmap .key{fill:var(--text);font-size:12px}",
        ".modelmap .key2{fill:var(--text2);font-size:12px}",
        ".modelmap .tag{font-size:10px;font-weight:700;letter-spacing:.08em}",
        ".modelmap .name{font-weight:600}",
        "</style>",
    ]


def label_markup(
    r: Rect, clip_id: str, ink: str, name: str, detail: str, extra: str = "", tag: str = ""
) -> list[str]:
    """The label of a tile, clipped to it; nothing if no line fits.

    :param r: The tile.
    :param clip_id: Document-unique id for the clip path.
    :param ink: Attribute text that sets the label color, such as
        ``class="ink-hf"``.
    :param name: First line, in bold.
    :param detail: Second line.
    :param extra: Third line.
    :param tag: A small heading above the name, such as the group's name.
    """
    font = 16 if r.w >= 220 and r.h >= 96 else 13 if r.w >= 110 and r.h >= 44 else 11
    width, room = r.w - 2 * _PAD, r.h - 2 * _PAD + 2
    lines: list[tuple[str, str, float, float]] = []  # class, text, font size, line height
    if tag and r.w >= 96:
        fitted_tag = _fit(tag, width, _TAG_FONT * 1.15)
        if fitted_tag and _TAG_LINE <= room:
            lines.append(("tag", fitted_tag, _TAG_FONT, _TAG_LINE))
    used = sum(line[3] for line in lines)
    for css, text, size in (
        ("name", name, font),
        ("", detail, font * 0.88),
        ("", extra, font * 0.88),
    ):
        fitted = _fit(text, width, size) if text else ""
        if not fitted or used + size * 1.3 > room:
            break  # a detail never appears without the lines above it
        lines.append((css, fitted, size, size * 1.3))
        used += size * 1.3
    if not lines:
        return []

    out = [
        f'<clipPath id="{clip_id}"><rect x="{_num(r.x)}" y="{_num(r.y)}" '
        f'width="{_num(r.w)}" height="{_num(r.h)}"/></clipPath>',
        f'<g clip-path="url(#{clip_id})" {ink} pointer-events="none">',
    ]
    y = r.y + _PAD
    for css, text, size, line_height in lines:
        y += line_height
        attrs = f' class="{css}"' if css else ""
        out.append(
            f'<text{attrs} x="{_num(r.x + _PAD)}" y="{_num(y - line_height * 0.28)}" '
            f'font-size="{_num(size)}">{_xml(text)}</text>'
        )
    out.append("</g>")
    return out


def _tile(
    tile: Tile, index: int, group_index: int, data: MapData, first: bool, group: Group
) -> list[str]:
    """One tile: its rectangle, its tooltip and, if it fits, its label."""
    r = tile.rect
    merged = bool(tile.members)
    fill_var, ink_var = f"--c{group_index}", f"--i{group_index}"
    if merged:
        names = sorted(data.records[int(k)].name for k in tile.members)
        shown = names[:12] + ([f"and {len(names) - 12} more"] if len(names) > 12 else [])
        provider = DISPLAY_NAMES.get(group.name, group.name)
        title = (
            f"{len(names)} smaller entries, {format_size(int(tile.size))}\n{provider}\n"
            + "\n".join(shown)
        )
        name, detail, extra = f"{len(names)} smaller", format_size(int(tile.size)), ""
    else:
        record = data.records[int(tile.key)]
        notes = [data.records[i] for i in data.attached.get(int(tile.key), [])]
        title = _tooltip(record, int(tile.size), data.total, notes)
        name = _short_name(record)
        detail = "  ".join(part for part in (format_size(int(tile.size)), record.quant) if part)
        count = format_count(record.params, record.params_exact)
        extra = f"{count} parameters" if count else ""

    out = [
        f'<rect class="tile" x="{_num(r.x)}" y="{_num(r.y)}" width="{_num(r.w)}" '
        f'height="{_num(r.h)}" rx="3" style="fill:var({fill_var})" '
        f'data-size="{int(tile.size)}"><title>{_xml(title)}</title></rect>'
    ]

    tag = ""
    if first:
        label = DISPLAY_NAMES.get(group.name, group.name).upper()
        tag = f"{label}  {format_size(int(group.size))}"
    return out + label_markup(
        r, f"clip{index}", f'style="fill:var({ink_var})"', name, detail, extra, tag
    )


def render_svg(scan: Scan, *, size: int = 1000, use_disk: bool = False) -> str:
    """Draw a scan as an SVG document.

    :param scan: The scan to draw.
    :param size: Side of the square map, in SVG units.
    :param use_disk: Size tiles by bytes allocated on disk, not apparent size.
    :return: The SVG text, ending with a newline.
    """
    data = build(scan, use_disk=use_disk)
    canvas = Rect(MARGIN, HEADER, size, size)
    groups = layout(data.items, canvas, min_fraction=MIN_FRACTION)
    rows = -(-len(groups) // LEGEND_COLUMNS)
    notes = _notes(data, use_disk)
    width = size + 2 * MARGIN
    height = HEADER + size + 18 + rows * LEGEND_ROW + len(notes) * 18 + MARGIN

    models = sum(1 for r in data.records if r.kind in ("model", "auxiliary"))
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" class="modelmap" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="{_xml(_summary(data, models))}">',
        *_css(groups),
        f'<rect class="bg" width="{width}" height="{height}"/>',
        f'<text class="hero" x="{MARGIN}" y="46">{_xml(format_size(data.total))}</text>',
        f'<text class="sub" x="{MARGIN}" y="68">{_xml(_summary(data, models))}</text>',
    ]

    index = 0
    for group_index, group in enumerate(groups):
        out.append(f'<g data-group="{_xml(group.name)}">')
        for position, tile in enumerate(group.tiles):
            out += _tile(tile, index, group_index, data, position == 0, group)
            index += 1
        out.append("</g>")
    for group in groups:
        g = group.rect
        out.append(
            f'<rect class="block" x="{_num(g.x)}" y="{_num(g.y)}" width="{_num(g.w)}" '
            f'height="{_num(g.h)}"/>'
        )

    y = HEADER + size + 18
    column = size / LEGEND_COLUMNS
    for position, group in enumerate(groups):
        x = MARGIN + (position % LEGEND_COLUMNS) * column
        row_y = y + (position // LEGEND_COLUMNS) * LEGEND_ROW
        count = sum(len(t.members) or 1 for t in group.tiles)
        share = 100 * group.size / data.total
        label = DISPLAY_NAMES.get(group.name, group.name)
        out.append(
            f'<rect x="{_num(x)}" y="{_num(row_y)}" width="12" height="12" rx="2" '
            f'style="fill:var(--c{position})"/>'
        )
        out.append(
            f'<text class="key" x="{_num(x + 18)}" y="{_num(row_y + 10.5)}">{_xml(label)}'
            f'<tspan class="key2" dx="8">{_xml(format_size(int(group.size)))}, {share:.0f}%, '
            f"{count} {'entry' if count == 1 else 'entries'}</tspan></text>"
        )
    y += rows * LEGEND_ROW + 6
    for note in notes:
        out.append(f'<text class="sub" x="{MARGIN}" y="{_num(y + 10)}">{_xml(note)}</text>')
        y += 18
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _summary(data: MapData, models: int) -> str:
    drawn = len(data.items)
    other = drawn - sum(
        1 for item in data.items if data.records[int(item.key)].kind in ("model", "auxiliary")
    )
    text = f"{drawn} {'entry' if drawn == 1 else 'entries'} with a size"
    if other:
        text += f": {drawn - other} models, {other} other (datasets, caches and other files)"
    return text if drawn else "No models found"


def _notes(data: MapData, use_disk: bool) -> list[str]:
    notes = ["Area shows " + ("space allocated on disk." if use_disk else "file size.")]
    notes[0] += " A file reachable by several names is counted once."
    attached = sum(len(v) for v in data.attached.values())
    if attached == 1:
        notes.append(
            "1 more name for a file already on the map is listed on the tile that holds it."
        )
    elif attached:
        notes.append(
            f"{attached} more names for files already on the map are listed on the tiles "
            "that hold them."
        )
    problems = sum(1 for record in data.records if record.issues)
    if problems:
        notes.append(
            f"{problems} {'entry has' if problems == 1 else 'entries have'} a problem, "
            "named in the tooltip; `modelmap list` shows them all."
        )
    if data.not_drawn:
        count = len(data.not_drawn)
        notes.append(
            f"{count} {'entry has' if count == 1 else 'entries have'} no size and "
            f"{'is' if count == 1 else 'are'} not drawn."
        )
    return notes
