"""Draw a scan as one interactive page that works offline.

The page is a set of views, every one laid out here in Python and written
into the file as plain markup:

- a top view per grouping (by store, format, family or kind);
- a view per group, where the group's records fill the square;
- a view per record, where its individual files fill the square.

Clicking a tile opens the view below it, down to the files. The script in
the page only shows and hides views, moves a tooltip and copies text; it
creates no markup, so a model name can never become code. The page loads
nothing from the network.

The page deletes nothing itself. For each record it shows where the files
are, which other records use them, and the ``modelmap rm`` command that
removes the record, or why it cannot.
"""

from __future__ import annotations

import base64
import hashlib
import shlex
from importlib import resources

from modelmap import __version__, palette
from modelmap.mapdata import MapData, build
from modelmap.records import FileEntry, ModelRecord, Scan
from modelmap.removal import plan, target_of
from modelmap.render_svg import (
    DISPLAY_NAMES,
    MIN_FRACTION,
    _notes,
    _num,
    _summary,
    _tooltip,
    _xml,
    label_markup,
)
from modelmap.treemap import Item, Rect, layout
from modelmap.units import format_count, format_size

CANVAS = Rect(0, 0, 1000, 1000)
HOME_PAGE = "https://github.com/Flux-Frontiers/modelmap"
TAGLINE = "WHERE THE WEIGHTS LIVE"
#: The Art Techno icon that ``scripts/build_logo.py`` writes.
_ICON = resources.files("modelmap").joinpath("icon.svg").read_text(encoding="utf-8")
#: Its paper version, for the banner in light surroundings.
_ICON_PAPER = resources.files("modelmap").joinpath("icon_paper.svg").read_text(encoding="utf-8")

#: The banner's inks. Dark is the obsidian original; light is its paper
#: version (Art Techno STYLE.md, paper exception): Silkscreen ground, iron
#: name and text, deep copper frame, part line and link.
BRAND = {
    "light": {
        "ground": "#EDE6D3",
        "edge": "#8F4F24",
        "name": "#151B18",
        "tag": "#8F4F24",
        "text": "#151B18",
        "link": "#8F4F24",
        "focus": "#1F6652",
    },
    "dark": {
        "ground": "#0B100E",
        "edge": "#D4AF5A",
        "name": "#D4AF5A",
        "tag": "#7FBFA8",
        "text": "#EDE6D3",
        "link": "#D4AF5A",
        "focus": "#7FBFA8",
    },
}
GROUPINGS = {"provider": "Store", "format": "Format", "family": "Family", "kind": "Kind"}


def _key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8", "surrogateescape")).hexdigest()[:16]


def record_view(record: ModelRecord) -> str:
    """A record's view id: from its store and path, so it is the same in every scan.

    A view named by position would open another record once one is removed.
    """
    return f"rec-{_key(record.provider, record.path)}"


def group_view(by: str, name: str) -> str:
    """A group's view id, from the grouping and the group's name."""
    return f"grp-{by}-{_key(by, name)}"


def _group_of(record: ModelRecord, by: str) -> str:
    return str(getattr(record, by) or "unknown")


def _group_label(name: str, by: str) -> str:
    return DISPLAY_NAMES.get(name, name) if by == "provider" else name


def _slug(provider: str) -> str:
    """CSS class suffix for a provider's color."""
    return provider if provider in palette.PROVIDER_COLORS else "other"


def _short(name: str) -> str:
    return name.rsplit("/", 1)[-1]


class _Page:
    """Collects the views and their side panels."""

    def __init__(
        self, data: MapData, use_disk: bool, ollama_stores: list[str], removable: bool = False
    ) -> None:
        self.removable = removable
        self.data = data
        self.use_disk = use_disk
        self.ollama_stores = ollama_stores
        self.views: list[str] = []
        self.panels: list[str] = []
        self.clips = 0
        # A scan has one record per (store, path); an inventory written by
        # hand may not, and two views must not share an id.
        self.rids: list[str] = []
        for record in data.records:
            rid = record_view(record)
            self.rids.append(rid if rid not in self.rids else f"{rid}-{len(self.rids)}")

    def rid(self, index: int) -> str:
        return self.rids[index]

    def size_of(self, record: ModelRecord) -> int:
        return record.disk_bytes if self.use_disk else record.size_bytes

    def file_size(self, entry: FileEntry) -> int:
        return entry.disk_bytes if self.use_disk else entry.size_bytes

    def clip(self) -> str:
        self.clips += 1
        return f"clip{self.clips}"

    # ------------------------------------------------------------- tiles

    def tile(
        self,
        rect: Rect,
        provider: str,
        size: int,
        tip: str,
        name: str,
        lines: tuple[str, str, str, str],
        go: str = "",
    ) -> list[str]:
        """One tile: a focusable group holding the rectangle and its label.

        A tile that opens a record takes its tooltip from that record's
        panel, so the text is in the page once, not once per view.
        """
        slug = _slug(provider)
        attrs = f'class="t" data-name="{_xml(name.lower())}"'
        attrs += f' data-tip-of="p-{go}"' if go.startswith("rec-") else f' data-tip="{_xml(tip)}"'
        if go:
            label = _xml(f"{name}, {format_size(size)}")
            attrs += f' data-go="{go}" tabindex="0" role="button" aria-label="{label}"'
        short, detail, extra, tag = lines
        return [
            f"<g {attrs}>",
            f'<rect class="tile p-{slug}" x="{_num(rect.x)}" y="{_num(rect.y)}" '
            f'width="{_num(rect.w)}" height="{_num(rect.h)}" rx="3" data-size="{size}"/>',
            *label_markup(rect, self.clip(), f'class="i-{slug}"', short, detail, extra, tag),
            "</g>",
        ]

    def record_tile(self, index: int, rect: Rect, size: int, tag: str) -> list[str]:
        record = self.data.records[index]
        detail = "  ".join(part for part in (format_size(size), record.quant) if part)
        count = format_count(record.params, record.params_exact)
        lines = (_short(record.name), detail, f"{count} parameters" if count else "", tag)
        return self.tile(rect, record.provider, size, "", record.name, lines, self.rid(index))

    # ------------------------------------------------------------- views

    def view(self, view_id: str, title: str, parents: dict[str, str], body: list[str]) -> None:
        attrs = "".join(f' data-{key}="{_xml(value)}"' for key, value in parents.items())
        self.views += [
            # The id differs from the name used in links, so that following
            # a link never makes the browser scroll to the element.
            f'<g class="view" id="v-{view_id}" data-view="{view_id}" '
            f'data-title="{_xml(title)}"{attrs}>',
            *body,
            "</g>",
        ]

    def panel(self, view_id: str, body: list[str], tip: str = "") -> None:
        attr = f' data-tip="{_xml(tip)}"' if tip else ""
        self.panels += [
            f'<section class="panel" id="p-{view_id}"{attr} hidden>',
            *body,
            "</section>",
        ]

    def top(self, by: str) -> dict[str, str]:
        """The top view of one grouping; returns each group's view id."""
        data = self.data
        items = [
            Item(item.key, _group_of(data.records[int(item.key)], by), item.size, item.label)
            for item in data.items
        ]
        groups = layout(items, CANVAS, min_fraction=MIN_FRACTION)
        ids = {group.name: group_view(by, group.name) for group in groups}
        body: list[str] = []
        rows: list[str] = []
        for group in groups:
            label = _group_label(group.name, by)
            for position, tile in enumerate(group.tiles):
                tag = f"{label.upper()}  {format_size(int(group.size))}" if position == 0 else ""
                if tile.members:
                    count = len(tile.members)
                    names = sorted(data.records[int(k)].name for k in tile.members)
                    shown = names[:12] + ([f"and {count - 12} more"] if count > 12 else [])
                    tip = "\n".join(
                        [f"{count} smaller entries, {format_size(int(tile.size))}", label, *shown]
                    )
                    provider = (
                        data.records[int(tile.members[0])].provider
                        if by != "provider"
                        else group.name
                    )
                    lines = (f"{count} smaller", format_size(int(tile.size)), "", tag)
                    body += self.tile(
                        tile.rect,
                        provider,
                        int(tile.size),
                        tip + "\nClick to see them",
                        f"{count} smaller in {label}",
                        lines,
                        ids[group.name],
                    )
                else:
                    body += self.record_tile(int(tile.key), tile.rect, int(tile.size), tag)
            g = group.rect
            body.append(
                f'<rect class="block" x="{_num(g.x)}" y="{_num(g.y)}" width="{_num(g.w)}" '
                f'height="{_num(g.h)}"/>'
            )
            count = sum(len(t.members) or 1 for t in group.tiles)
            swatch = f'<span class="sw p-{_slug(group.name)}"></span>' if by == "provider" else ""
            rows.append(
                f'<tr data-name="{_xml(label.lower())}"><td>{swatch}'
                f'<button class="link" data-go="{ids[group.name]}">{_xml(label)}</button></td>'
                f'<td class="n">{format_size(int(group.size))}</td>'
                f'<td class="n">{100 * group.size / data.total:.0f}%</td><td class="n">{count}</td></tr>'
            )
            self.group(by, ids[group.name], group.name, [t for t in items if t.group == group.name])
        self.view(f"top-{by}", "All models", {"by": by}, body)

        panel = [
            f"<h2>By {GROUPINGS[by].lower()}</h2>",
            '<table><thead><tr><th scope="col">Group</th><th scope="col" class="n">Size</th>'
            '<th scope="col" class="n">Share</th><th scope="col" class="n">Entries</th></tr></thead>',
            "<tbody>",
            *rows,
            "</tbody></table>",
            *self.problems(),
        ]
        self.panel(f"top-{by}", panel)
        return ids

    def group(self, by: str, view_id: str, name: str, items: list[Item]) -> None:
        """One group's records filling the square, nothing merged."""
        label = _group_label(name, by)
        total = sum(item.size for item in items)
        body: list[str] = []
        rows: list[str] = []
        for group in layout([Item(i.key, "", i.size, i.label) for i in items], CANVAS):
            for tile in group.tiles:
                index = int(tile.key)
                body += self.record_tile(index, tile.rect, int(tile.size), "")
                rows.append(self.record_row(index, int(tile.size)))
        self.view(view_id, label, {"by": by, "parent": f"top-{by}"}, body)
        self.panel(
            view_id,
            [
                f"<h2>{_xml(label)}</h2>",
                f'<p class="sub">{format_size(int(total))} in {len(items)} '
                f"{'entry' if len(items) == 1 else 'entries'}. Click one to see its files.</p>",
                '<table><thead><tr><th scope="col">Name</th><th scope="col" class="n">Size</th>'
                '<th scope="col">Quantization</th><th scope="col" class="n">Parameters</th></tr></thead>',
                "<tbody>",
                *rows,
                "</tbody></table>",
            ],
        )

    def record_row(self, index: int, size: int) -> str:
        record = self.data.records[index]
        flag = ' <span class="flag">problem</span>' if record.issues else ""
        return (
            f'<tr data-name="{_xml(record.name.lower())}"><td>'
            f'<span class="sw p-{_slug(record.provider)}"></span>'
            f'<button class="link" data-go="{self.rid(index)}">{_xml(record.name)}</button>{flag}</td>'
            f'<td class="n">{format_size(size)}</td><td>{_xml(record.quant or "")}</td>'
            f'<td class="n">{format_count(record.params, record.params_exact)}</td></tr>'
        )

    def problems(self) -> list[str]:
        """Records with a problem, and those with nothing to draw, as lists.

        A problem must be findable without finding its tile: a broken
        download can be too small to see.
        """
        data = self.data
        out: list[str] = []
        flagged = [i for i, record in enumerate(data.records) if record.issues]
        if flagged:
            out.append(f'<h3>Problems ({len(flagged)})</h3><ul class="plain">')
            for index in flagged:
                record = data.records[index]
                out.append(
                    f'<li data-name="{_xml(record.name.lower())}">'
                    f'<button class="link" data-go="{self.rid(index)}">{_xml(record.name)}</button>'
                    f'<span class="sub"> {_xml("; ".join(record.issues))}</span></li>'
                )
            out.append("</ul>")
        if data.not_drawn:
            out.append(
                f'<h3>No size, not on the map ({len(data.not_drawn)})</h3><ul class="plain">'
            )
            for index in data.not_drawn:
                record = data.records[index]
                out.append(
                    f'<li data-name="{_xml(record.name.lower())}">'
                    f'<button class="link" data-go="{self.rid(index)}">{_xml(record.name)}</button></li>'
                )
            out.append("</ul>")
        return out

    def record(self, index: int, parents: dict[str, str]) -> None:
        """One record's files filling the square, and its details."""
        record = self.data.records[index]
        entries = record.file_list
        sized = [(n, self.file_size(e)) for n, e in enumerate(entries) if self.file_size(e) > 0]
        total = sum(size for _, size in sized)
        items = [Item(str(n), "", size, entries[n].label) for n, size in sized]
        body: list[str] = []
        for group in layout(items, CANVAS, min_fraction=MIN_FRACTION):
            for tile in group.tiles:
                if tile.members:
                    count = len(tile.members)
                    tip = f"{count} smaller files, {format_size(int(tile.size))}\nListed beside the map"
                    lines = (f"{count} smaller files", format_size(int(tile.size)), "", "")
                    body += self.tile(
                        tile.rect,
                        record.provider,
                        int(tile.size),
                        tip,
                        f"{count} smaller files",
                        lines,
                    )
                    continue
                entry = entries[int(tile.key)]
                extra = entry.note or ("also used elsewhere" if entry.used_by else "")
                lines = (_short(entry.label), format_size(int(tile.size)), extra, "")
                body += self.tile(
                    tile.rect,
                    record.provider,
                    int(tile.size),
                    _file_tip(entry, int(tile.size), total),
                    entry.label,
                    lines,
                )
        if not body:
            body.append(
                '<text class="empty" x="500" y="500" text-anchor="middle">No files to draw</text>'
            )
        self.view(self.rid(index), record.name, parents, body)
        notes = [self.data.records[i] for i in self.data.attached.get(index, [])]
        tip = _tooltip(record, self.size_of(record), max(self.data.total, 1), notes)
        self.panel(self.rid(index), self.record_panel(index), tip + "\nClick to see its files")

    def record_panel(self, index: int) -> list[str]:
        record = self.data.records[index]
        facts = [
            ("Store", DISPLAY_NAMES.get(record.provider, record.provider)),
            ("Kind", record.kind),
            ("Format", record.format),
            ("Quantization", record.quant or ""),
            ("Parameters", format_count(record.params, record.params_exact)),
            ("Size", format_size(record.size_bytes)),
            ("On disk", format_size(record.disk_bytes)),
            ("Also known as", ", ".join(record.aliases)),
        ]
        out = [f"<h2>{_xml(record.name)}</h2>", "<dl>"]
        out += [f"<dt>{label}</dt><dd>{_xml(value)}</dd>" for label, value in facts if value]
        out.append("</dl>")
        if record.issues:
            out.append('<ul class="issues">')
            out += [f"<li>{_xml(issue)}</li>" for issue in record.issues]
            out.append("</ul>")
        attached = [self.data.records[i] for i in self.data.attached.get(index, [])]
        if attached:
            names = ", ".join(
                f"{r.name} ({DISPLAY_NAMES.get(r.provider, r.provider)})" for r in attached
            )
            out.append(f'<p class="sub">The same files are also stored as: {_xml(names)}</p>')

        out += ["<h3>Location</h3>", _copyable(record.path, "Copy path")]
        step = plan(record, self.ollama_stores, check_disk=False)
        out.append("<h3>To remove</h3>")
        if step.action != "refuse":
            if self.removable:
                out.append(
                    f'<button type="button" class="remove" data-remove="{self.rid(index)}">'
                    "Remove</button>"
                )
            target = target_of(record, self.data.records)
            out.append(_copyable(f"modelmap rm {shlex.quote(target)}", "Copy command"))
            note = f"{step.note} It lists what will go and asks first."
        else:
            note = step.note
        out.append(f'<p class="sub">{_xml(note)}</p>')
        users = sorted({user for entry in record.file_list for user in entry.used_by})
        if users:
            out.append(
                '<p class="warn">Some of these files are also used by: '
                f"{_xml(', '.join(users))}. Removing them breaks those too.</p>"
            )
        how = (
            "The Remove button does the same from this page."
            if self.removable
            else "The page itself deletes nothing: run the command in a terminal."
        )
        out.append(
            f'<p class="sub">{how} Sharing is '
            "known only among what this scan covered: a file may also be used by a store that "
            "was not scanned.</p>"
            f"<h3>Files ({len(record.file_list)})</h3>"
        )
        out.append('<ul class="files">')
        order = sorted(range(len(record.file_list)), key=lambda n: -record.file_list[n].size_bytes)
        for n in order:
            entry = record.file_list[n]
            remarks = [entry.note] if entry.note else []
            if not entry.charged:
                remarks.append("counted under another record")
            if entry.used_by:
                remarks.append("also used by " + ", ".join(entry.used_by))
            remark = f'<span class="sub"> {_xml("; ".join(remarks))}</span>' if remarks else ""
            out.append(
                f'<li data-name="{_xml(entry.label.lower())}"><div class="row">'
                f"<strong>{_xml(entry.label)}</strong>"
                f'<span class="n">{format_size(self.file_size(entry))}</span></div>'
                f"{remark}{_copyable(entry.path, 'Copy path')}</li>"
            )
        out.append("</ul>")
        return out


def _copyable(text: str, label: str) -> str:
    """Text shown in full, selectable, with a button that copies it."""
    return (
        f'<div class="copy"><code>{_xml(text)}</code>'
        f'<button data-copy data-label="{label}">{label}</button></div>'
    )


def _file_tip(entry: FileEntry, size: int, total: int) -> str:
    lines = [entry.label, f"{format_size(size)}, {100 * size / total:.1f}% of this model"]
    if entry.note:
        lines.append(entry.note)
    if not entry.charged:
        lines.append("Counted under another record")
    if entry.used_by:
        lines.append("Also used by: " + ", ".join(entry.used_by))
    lines.append(entry.path)
    return "\n".join(lines)


def _colors() -> str:
    """CSS variables for both modes, and the rules that apply them."""

    def variables(mode: str) -> str:
        parts = [
            f"--surface:{palette.SURFACE[mode]}",
            f"--page:{'#f9f9f7' if mode == 'light' else '#0d0d0d'}",
            f"--text:{palette.TEXT_PRIMARY[mode]}",
            f"--text2:{palette.TEXT_SECONDARY[mode]}",
            f"--line:{'#e1e0d9' if mode == 'light' else '#2c2c2a'}",
            f"--warn:{'#a13a00' if mode == 'light' else '#f0a070'}",
            *(f"--brand-{role}:{ink}" for role, ink in BRAND[mode].items()),
        ]
        for name in [*palette.PROVIDER_COLORS, "other"]:
            color = palette.fill(name, mode)
            parts += [f"--c-{name}:{color}", f"--i-{name}:{palette.ink_on(color)}"]
        return ";".join(parts)

    rules = [
        f":root{{{variables('light')}}}",
        "@media (prefers-color-scheme:dark){:root:not([data-theme=light]){"
        f"{variables('dark')};color-scheme:dark}}}}",
        f":root[data-theme=dark]{{{variables('dark')};color-scheme:dark}}",
    ]
    for name in [*palette.PROVIDER_COLORS, "other"]:
        rules.append(f".p-{name}{{fill:var(--c-{name});background:var(--c-{name})}}")
        rules.append(f".i-{name}{{fill:var(--i-{name})}}")
    return "\n".join(rules)


_CSS = """
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
padding:20px max(16px,env(safe-area-inset-right,0px)) 32px max(16px,env(safe-area-inset-left,0px))}
header{display:flex;flex-wrap:wrap;gap:12px 24px;align-items:end;margin-bottom:12px}
h1{font-size:15px;font-weight:600;margin:0;color:var(--text2)}
.hero{font-size:34px;font-weight:600;line-height:1.1}
.sub{color:var(--text2);font-size:13px}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-left:auto}
.seg{display:flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{border:0;background:var(--surface);color:var(--text);padding:6px 12px;font:inherit;cursor:pointer}
.seg button[aria-pressed=true]{background:var(--text);color:var(--surface)}
input[type=search]{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:8px;
background:var(--surface);color:var(--text);min-width:0;width:14em}
#crumbs{display:flex;flex-wrap:wrap;gap:4px;align-items:center;min-height:28px;margin-bottom:8px}
#crumbs .sep{color:var(--text2)}
.layout{display:grid;grid-template-columns:minmax(0,min(100%,max(280px,calc(100vh - var(--map-top, 233px) - 32px)))) minmax(0,1fr);
gap:24px;align-items:start}
@media (max-width:900px){.layout{grid-template-columns:minmax(0,1fr)}.map{max-width:max(320px,calc(100vh - var(--map-top, 233px) - 32px))}}
.map{position:relative;background:var(--surface);border-radius:6px}
.map svg{display:block;width:100%;height:auto;aspect-ratio:1}
.view{display:none}.view.on{display:inline}
.tile{stroke:var(--surface);stroke-width:2}
.block{fill:none;stroke:var(--surface);stroke-width:6;pointer-events:none}
.t[data-go]{cursor:pointer}
.t:focus{outline:none}.t:focus-visible .tile{stroke:var(--text);stroke-width:4}
.t.dim{opacity:.18}
svg text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.tag{font-size:10px;font-weight:700;letter-spacing:.08em}.name{font-weight:600}
.empty{fill:var(--text2);font-size:28px}
#tip{position:fixed;z-index:5;max-width:min(460px,90vw);padding:8px 10px;border-radius:6px;
background:var(--text);color:var(--surface);font-size:12px;white-space:pre-line;
overflow-wrap:anywhere;pointer-events:none}
.panels{min-width:0}
.panel h2{font-size:20px;margin:0 0 6px;overflow-wrap:anywhere;text-wrap:balance}
.panel h3{font-size:13px;margin:18px 0 6px;color:var(--text2);text-transform:uppercase;letter-spacing:.06em}
table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:5px 8px 5px 0;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;font-size:12px;color:var(--text2)}
.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.dim,li.dim{opacity:.25}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
button.link{display:inline;border:0;background:none;color:inherit;font:inherit;padding:0;cursor:pointer;
text-align:left;text-decoration:underline;text-decoration-color:var(--line);
text-underline-offset:3px;overflow-wrap:anywhere}
button.link:hover{text-decoration-color:currentColor}
button:focus-visible,input:focus-visible{outline:2px solid var(--c-hf);outline-offset:2px}
dl{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:2px 16px;margin:8px 0}
dt{color:var(--text2)}dd{margin:0;overflow-wrap:anywhere}
.copy{display:flex;gap:8px;align-items:start;margin:4px 0}
.copy code{flex:1;min-width:0;font:12px/1.4 ui-monospace,Menlo,Consolas,monospace;
background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:5px 8px;
overflow-wrap:anywhere;user-select:all}
.copy button{flex:none;font:inherit;font-size:12px;padding:4px 8px;border:1px solid var(--line);
border-radius:6px;background:var(--surface);color:var(--text);cursor:pointer}
ul.plain,ul.files{list-style:none;margin:0;padding:0}
ul.files li{padding:8px 0;border-bottom:1px solid var(--line)}
.row{display:flex;gap:12px;justify-content:space-between}.row strong{overflow-wrap:anywhere;min-width:0}
.issues,.warn,.flag{color:var(--warn)}.flag{font-size:11px;font-weight:600;margin-left:6px}
.issues{padding-left:18px;margin:6px 0}
footer{margin-top:20px}
@media (prefers-reduced-motion:no-preference){.t,tr,li{transition:opacity .12s}}
"""

_JS = """
(function () {
  "use strict";
  var views = {}, current = null, by = "provider";
  var tip = document.getElementById("tip");
  var crumbs = document.getElementById("crumbs");
  var search = document.getElementById("search");
  Array.prototype.forEach.call(document.querySelectorAll(".view"), function (v) { views[v.dataset.view] = v; });

  function parentOf(view) {
    var key = "parent" + by.charAt(0).toUpperCase() + by.slice(1);
    return view.dataset[key] || view.dataset.parent || "";
  }
  function filter() {
    var q = search.value.trim().toLowerCase();
    var scope = [current, document.getElementById("p-" + current.dataset.view)];
    scope.forEach(function (root) {
      Array.prototype.forEach.call(root.querySelectorAll("[data-name]"), function (el) {
        el.classList.toggle("dim", q !== "" && el.dataset.name.indexOf(q) === -1);
      });
    });
  }
  function show(id) {
    var view = views[id] || views["top-" + by];
    if (view.dataset.by) { by = view.dataset.by; }
    if (current) {
      current.classList.remove("on");
      document.getElementById("p-" + current.dataset.view).hidden = true;
    }
    current = view;
    view.classList.add("on");
    document.getElementById("p-" + view.dataset.view).hidden = false;
    Array.prototype.forEach.call(document.querySelectorAll("[data-by]"), function (b) {
      if (b.tagName === "BUTTON") { b.setAttribute("aria-pressed", String(b.dataset.by === by)); }
    });
    var chain = [], at = view;
    while (at) { chain.unshift(at); at = views[parentOf(at)]; }
    crumbs.textContent = "";
    chain.forEach(function (v, n) {
      if (n) {
        var sep = document.createElement("span");
        sep.className = "sep"; sep.textContent = "/";
        crumbs.appendChild(sep);
      }
      var last = n === chain.length - 1;
      var el = document.createElement(last ? "strong" : "button");
      el.textContent = v.dataset.title;
      if (!last) { el.className = "link"; el.dataset.go = v.dataset.view; }
      crumbs.appendChild(el);
    });
    tip.hidden = true;
    filter();
  }
  function go(id) {
    if (location.hash.slice(1) === id) { show(id); } else { location.hash = id; }
  }
  function copy(button) {
    var done = function () {
      button.textContent = "Copied";
      setTimeout(function () { button.textContent = button.dataset.label; }, 1500);
    };
    var select = function () {
      var range = document.createRange();
      range.selectNodeContents(button.parentNode.querySelector("code"));
      var selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = "Selected";
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      var text = button.parentNode.querySelector("code").textContent;
      navigator.clipboard.writeText(text).then(done, select);
    } else { select(); }
  }

  document.addEventListener("click", function (event) {
    var target = event.target.closest("[data-copy],[data-go],button[data-by]");
    if (!target) { return; }
    if (target.dataset.copy !== undefined) { copy(target); }
    else if (target.dataset.go) { go(target.dataset.go); }
    else { by = target.dataset.by; go("top-" + by); }
  });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && current && parentOf(current)) { go(parentOf(current)); return; }
    var target = event.target.closest && event.target.closest("g[data-go]");
    if (target && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      go(target.dataset.go);
    }
  });
  document.addEventListener("pointermove", function (event) {
    var target = event.target.closest && event.target.closest("svg [data-tip],[data-tip-of]");
    if (!target) { tip.hidden = true; return; }
    var source = target.dataset.tipOf ? document.getElementById(target.dataset.tipOf) : target;
    tip.textContent = source.dataset.tip;
    tip.hidden = false;
    var box = tip.getBoundingClientRect();
    var x = Math.min(event.clientX + 14, window.innerWidth - box.width - 8);
    var y = event.clientY + 16;
    if (y + box.height > window.innerHeight - 8) { y = event.clientY - box.height - 12; }
    tip.style.left = Math.max(8, x) + "px";
    tip.style.top = Math.max(8, y) + "px";
  });
  // The square fits the window below whatever is above it: the header
  // wraps on a narrow window, so its height is measured, not assumed.
  var map = document.querySelector(".map");
  function fit() {
    var top = map.getBoundingClientRect().top + window.scrollY;
    document.documentElement.style.setProperty("--map-top", Math.round(top) + "px");
  }
  window.addEventListener("resize", fit);
  fit();
  search.addEventListener("input", filter);
  window.addEventListener("hashchange", function () { show(location.hash.slice(1)); });
  show(location.hash.slice(1));
})();
"""


# The banner keeps the brand's own colors in light and dark mode; the map
# below keeps the page's. Gold, mint and silkscreen on obsidian are each
# above 7:1.
_BRAND_CSS = """
.brand{display:flex;flex-wrap:wrap;align-items:center;gap:10px 14px;background:var(--brand-ground);
color:var(--brand-text);border:1px solid var(--brand-edge);border-radius:10px;padding:10px 14px;
margin-bottom:16px}
.brand .logo{width:44px;height:44px;flex:none}
.brand-name{font-weight:600;font-size:20px;line-height:1;letter-spacing:.24em;color:var(--brand-name)}
.brand-tag{font:500 10.5px/1 ui-monospace,Menlo,Consolas,monospace;letter-spacing:.28em;
color:var(--brand-tag);margin-top:6px}
.brand-meta{margin-left:auto;display:flex;gap:14px;align-items:center;
font:12px/1 ui-monospace,Menlo,Consolas,monospace}
.brand-meta a{color:var(--brand-link);text-underline-offset:3px}
.brand-meta a:focus-visible{outline:2px solid var(--brand-focus);outline-offset:2px}
.brand .logo-dark{display:none}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]) .brand .logo-dark{display:block}
:root:not([data-theme=light]) .brand .logo-paper{display:none}}
:root[data-theme=dark] .brand .logo-dark{display:block}
:root[data-theme=dark] .brand .logo-paper{display:none}
"""


def _inline_icon(svg: str, kind: str) -> str:
    head, _, rest = svg.partition("<title>")
    icon = head + rest.partition("</title>")[2]
    return icon.replace(
        "<svg ", f'<svg class="logo logo-{kind}" aria-hidden="true" focusable="false" ', 1
    ).strip()


def _banner() -> list[str]:
    """The Flux-Frontiers banner: icon, name, tagline, version and a link home.

    It follows the page's theme: the obsidian original in dark mode, its paper
    version in light mode, with the icon to match.
    """
    return [
        '<div class="brand">',
        _inline_icon(_ICON, "dark"),
        _inline_icon(_ICON_PAPER, "paper"),
        f'<div><div class="brand-name">MODELMAP</div><div class="brand-tag">{TAGLINE}</div></div>',
        f'<div class="brand-meta"><span>v{_xml(__version__)}</span>'
        f'<a href="{HOME_PAGE}" rel="noopener noreferrer">Flux-Frontiers</a></div>',
        "</div>",
    ]


_SERVED_CSS = (
    "#rescan,button.remove,.rm-buttons button{font:inherit;padding:6px 12px;"
    "border:1px solid var(--line);border-radius:8px;background:var(--surface);"
    "color:var(--text);cursor:pointer}"
    "button.remove,button.danger{color:var(--warn);border-color:var(--warn);font-weight:600}"
    "button.remove{margin:4px 0 8px}"
    "#rescan:disabled,.rm-buttons button:disabled,button.remove:disabled{opacity:.6;cursor:progress}"
    "dialog{width:min(36em,calc(100vw - 32px));border:1px solid var(--line);border-radius:12px;"
    "background:var(--surface);color:var(--text);padding:20px}"
    "dialog::backdrop{background:rgb(0 0 0 / .45)}"
    "dialog h2{margin:0 0 4px;font-size:17px;overflow-wrap:anywhere}"
    ".rm-list li{padding:3px 0}.rm-list code{font:12px/1.4 ui-monospace,Menlo,Consolas,monospace;"
    "overflow-wrap:anywhere}"
    ".rm-buttons{display:flex;gap:8px;justify-content:flex-end;margin-top:16px}"
)

# Filled by the script with textContent only. Cancel comes first and takes
# the focus, so Enter does not remove anything.
_DIALOG = (
    '<dialog id="rm-dialog" aria-labelledby="rm-title"><h2 id="rm-title"></h2>'
    '<p id="rm-what" class="sub"></p><p id="rm-note"></p><ul id="rm-list" class="plain rm-list"></ul>'
    '<p id="rm-variants" class="warn"></p><p id="rm-undo" class="warn"></p>'
    '<p id="rm-status" role="status"></p><div class="rm-buttons">'
    '<button type="button" id="rm-cancel">Cancel</button>'
    '<button type="button" id="rm-confirm" class="danger">Remove</button></div></dialog>'
)

# The served page reads the run's token from its own address, so the token
# is never written into the page.
_SERVED_JS = """
(function () {
  "use strict";
  var token = new URLSearchParams(location.search).get("t") || "";
  var button = document.getElementById("rescan");
  var status = document.getElementById("served-status");
  function call(path, body) {
    return fetch(path, {
      method: "POST",
      headers: {"Content-Type": "application/json", "X-Modelmap-Token": token},
      body: JSON.stringify(body || {})
    }).then(function (response) {
      return response.json().then(function (data) {
        if (!response.ok) {
          var error = new Error(data.error || response.statusText);
          error.data = data;
          throw error;
        }
        return data;
      });
    });
  }
  button.addEventListener("click", function () {
    button.disabled = true;
    status.textContent = "Scanning...";
    call("/api/rescan").then(function () { location.reload(); }, function (error) {
      status.textContent = "Rescan failed: " + error.message;
      button.disabled = false;
    });
  });

  var dialog = document.getElementById("rm-dialog");
  if (!dialog) { return; }
  var confirm = document.getElementById("rm-confirm");
  var cancel = document.getElementById("rm-cancel");
  var rmStatus = document.getElementById("rm-status");
  var shown = null, busy = false, stale = false;

  function put(id, value) {
    var el = document.getElementById(id);
    el.textContent = value || "";
    el.hidden = !value;
  }
  function fill(plan) {
    shown = plan;
    var refused = plan.action === "refuse";
    put("rm-title", (refused ? "Cannot remove " : "Remove ") + plan.name + (refused ? "" : "?"));
    put("rm-what", plan.store + ", " + plan.size);
    put("rm-note", plan.note);
    var list = document.getElementById("rm-list");
    list.textContent = "";
    var items = plan.command ? [plan.command] : (plan.paths.length ? plan.paths : [plan.path]);
    items.forEach(function (item) {
      var li = document.createElement("li"), code = document.createElement("code");
      code.textContent = item;
      li.appendChild(code);
      list.appendChild(li);
    });
    put("rm-variants", plan.variants.length ?
      "Also removed, with settings of their own: " + plan.variants.join(", ") : "");
    put("rm-undo", refused ? "" : (plan.undo ?
      "The files go to the Trash; they can be put back from there." : "This cannot be undone."));
    put("rm-status", "");
    confirm.hidden = refused;
  }
  function failed(message) {
    shown = null;
    put("rm-title", "Cannot remove");
    ["rm-what", "rm-variants", "rm-undo", "rm-status"].forEach(function (id) { put(id, ""); });
    put("rm-note", message);
    document.getElementById("rm-list").textContent = "";
    confirm.hidden = true;
  }
  function setBusy(on) {
    busy = on;
    confirm.disabled = on;
    cancel.disabled = on;
  }
  function parentOf(view) {
    var el = document.getElementById("v-" + view);
    if (!el) { return ""; }
    var pressed = document.querySelector('button[data-by][aria-pressed="true"]');
    var by = pressed ? pressed.dataset.by : "provider";
    return el.dataset["parent" + by.charAt(0).toUpperCase() + by.slice(1)] || el.dataset.parent || "";
  }
  function reloadAt(view) {
    history.replaceState(null, "", location.pathname + location.search + (view ? "#" + view : ""));
    location.reload();
  }

  document.addEventListener("click", function (event) {
    var target = event.target.closest("[data-remove]");
    if (!target) { return; }
    target.disabled = true;
    call("/api/plan", {view: target.dataset.remove}).then(fill, function (error) {
      failed(error.message);
    }).then(function () {
      target.disabled = false;
      setBusy(false);
      dialog.showModal();
      cancel.focus();
    });
  });
  confirm.addEventListener("click", function () {
    if (!shown || busy) { return; }
    var view = shown.view;
    setBusy(true);
    put("rm-status", "Removing...");
    call("/api/remove", {view: view, digest: shown.digest}).then(function () {
      reloadAt(parentOf(view));
    }, function (error) {
      setBusy(false);
      stale = true;  // the server scanned again before answering
      var data = error.data || {};
      if (data.gone) { reloadAt(parentOf(view)); return; }
      if (data.plan) { fill(data.plan); }
      put("rm-status", error.message);
    });
  });
  cancel.addEventListener("click", function () { if (!busy) { dialog.close(); } });
  dialog.addEventListener("cancel", function (event) { if (busy) { event.preventDefault(); } });
  dialog.addEventListener("keydown", function (event) {
    if (event.key === "Escape") { event.stopPropagation(); }
  });
  dialog.addEventListener("close", function () { if (stale) { location.reload(); } });
})();
"""

#: The page's inline scripts, for a server's Content-Security-Policy hashes.
SCRIPTS = (_JS.strip(), _SERVED_JS.strip())


def render_html(
    scan: Scan, *, use_disk: bool = False, served: bool = False, removable: bool = False
) -> str:
    """Draw a scan as a self-contained interactive page.

    :param scan: The scan to draw.
    :param use_disk: Size tiles by bytes allocated on disk, not apparent size.
    :param served: Add what ``modelmap serve`` needs: a Rescan button and
        the script that calls the server. Every addition is a line of its
        own, so the page is the file ``modelmap map`` writes plus those lines.
    :param removable: With ``served``, a Remove button in each record's
        panel and the dialog that confirms it. The server does the removing.
    :return: The HTML text, ending with a newline.
    """
    data = build(scan, use_disk=use_disk)
    # `ollama rm` acts on the running Ollama's store: its provider roots, not
    # an Ollama store that was only recognized by its layout somewhere else.
    live = [r.resolved for r in scan.roots if r.provider == "ollama" and r.source != "layout"]
    page = _Page(data, use_disk, live, removable=served and removable)
    parents: dict[int, dict[str, str]] = {i: {} for i in range(len(data.records))}
    drawn = {int(item.key) for item in data.items}
    for by in GROUPINGS:
        ids = page.top(by)
        for index in drawn:
            group = ids[_group_of(data.records[index], by)]
            parents[index][f"parent-{by}"] = group
    for index in range(len(data.records)):
        # A record with no tile is reached from the lists; it hangs off the top.
        links = parents[index] or {"parent": "top-provider"}
        page.record(index, links)

    models = sum(1 for r in data.records if r.kind in ("model", "auxiliary"))
    buttons = "".join(
        f'<button data-by="{by}" aria-pressed="false">{label}</button>'
        for by, label in GROUPINGS.items()
    )
    out = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
        "<title>Local models</title>",
        # The icon is inline, so a browser does not go looking for one.
        '<link rel="icon" href="data:image/svg+xml;base64,'
        + base64.b64encode(_ICON.encode("utf-8")).decode("ascii")
        + '">',
        "<style>",
        _colors(),
        _CSS.strip(),
        _BRAND_CSS.strip(),
        *([_SERVED_CSS] if served else []),
        "</style>",
        "</head>",
        "<body>",
        *_banner(),
        "<header>",
        f'<div><h1>Local models</h1><div class="hero">{_xml(format_size(data.total))}</div>'
        f'<div class="sub">{_xml(_summary(data, models))}</div></div>',
        '<div class="controls">',
        f'<div class="seg" role="group" aria-label="Group by">{buttons}</div>',
        '<input type="search" id="search" placeholder="Filter by name" aria-label="Filter by name">',
        *(
            [
                '<button id="rescan" type="button">Rescan</button>',
                '<span id="served-status" class="sub" role="status"></span>',
            ]
            if served
            else []
        ),
        "</div>",
        "</header>",
        '<nav id="crumbs" aria-label="Where you are"></nav>',
        '<div class="layout">',
        '<div class="map"><svg viewBox="0 0 1000 1000" role="group" aria-label="Map of models by size">',
        *page.views,
        "</svg></div>",
        '<div class="panels">',
        *page.panels,
        "</div>",
        "</div>",
        "<footer>",
        *[f'<p class="sub">{_xml(note)}</p>' for note in _notes(data, use_disk)],
        '<p class="sub">Click a tile to open it, down to its files. Press Escape to go back up.</p>',
        "</footer>",
        '<div id="tip" hidden></div>',
        *([_DIALOG] if served and removable else []),
        "<script>",
        _JS.strip(),
        "</script>",
        *(["<script>", _SERVED_JS.strip(), "</script>"] if served else []),
        "</body>",
        "</html>",
    ]
    return "\n".join(out) + "\n"
