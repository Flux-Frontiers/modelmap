"""The interactive page: its structure, its honesty and what it must not do."""

from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from modelmap import __version__
from modelmap.providers.base import Context
from modelmap.records import FileEntry, ModelRecord, Scan
from modelmap.render_html import GROUPINGS, HOME_PAGE, group_view, record_view, render_html
from modelmap.scanner import Options, redact, scan
from tests.fakemachine import FakeMachine
from tests.test_render_svg import small_scan

GB = 1000**3


class Page(HTMLParser):
    """Collects what the tests ask about, with the standard-library parser."""

    def __init__(self, text: str) -> None:
        super().__init__(convert_charrefs=True)
        self.views: dict[str, dict[str, str]] = {}
        self.tiles: dict[str, list[dict[str, str]]] = {}
        self.rects: dict[str, list[tuple[float, float, int]]] = {}
        self.panels: set[str] = set()
        self.targets: list[str] = []
        self.ids: list[str] = []
        self.elements: list[str] = []
        self._view = ""
        self.text = ""
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k: v or "" for k, v in attrs}
        self.elements.append(tag)
        if "id" in a:
            self.ids.append(a["id"])
        if a.get("class") == "view":
            self._view = a["data-view"]
            self.views[self._view] = a
            self.tiles[self._view] = []
            self.rects[self._view] = []
        elif a.get("class") == "t":
            self.tiles[self._view].append(a)
        elif tag == "rect" and "data-size" in a:
            self.rects[self._view].append(
                (float(a["width"]), float(a["height"]), int(a["data-size"]))
            )
        elif a.get("class") == "panel":
            self.panels.add(a["id"])
        if a.get("data-go"):
            self.targets.append(a["data-go"])

    def handle_data(self, data: str) -> None:
        self.text += data


@pytest.fixture(scope="module")
def page() -> Page:
    return Page(render_html(small_scan()))


def test_every_link_leads_to_a_view_and_every_view_has_a_panel(page: Page) -> None:
    assert set(page.targets) <= set(page.views)
    assert {f"p-{name}" for name in page.views} == page.panels
    assert len(page.ids) == len(set(page.ids)), "duplicate ids"
    # A link name is never an element id: following it must not scroll the page.
    assert not set(page.views) & set(page.ids)


def test_views_exist_for_every_grouping_group_and_record(page: Page) -> None:
    for by in GROUPINGS:
        assert f"top-{by}" in page.views
    assert sum(1 for v in page.views if v.startswith("grp-provider-")) == 4
    records = small_scan().records
    assert {record_view(r) for r in records} <= set(page.views)
    # Parents: a group hangs off its top view, a record off a group per grouping.
    assert page.views[group_view("format", "gguf")]["data-parent"] == "top-format"
    rec = page.views[record_view(records[0])]
    assert {f"data-parent-{by}" for by in GROUPINGS} <= set(rec)
    assert rec["data-parent-provider"].startswith("grp-provider-")
    # A record with no size has no tile, so it hangs off the top view.
    assert page.views[record_view(records[10])]["data-parent"] == "top-provider"


def test_areas_are_in_proportion_in_every_view(page: Page) -> None:
    checked = 0
    for name, rects in page.rects.items():
        total = sum(size for _, _, size in rects)
        if not rects:
            continue
        for w, h, size in rects:
            assert abs(w * h - size / total * 1_000_000) <= 4, name
        assert sum(w * h for w, h, _ in rects) == pytest.approx(1_000_000, rel=1e-5), name
        checked += 1
    assert checked >= 12  # the top view of each grouping and every group


def test_top_tiles_open_records_and_merged_tiles_open_their_group(page: Page) -> None:
    top = page.tiles["top-provider"]
    assert len(top) == 8
    assert sum(1 for t in top if t["data-go"].startswith("rec-")) == 7
    (merged,) = [t for t in top if t["data-go"].startswith("grp-")]
    assert merged["data-name"] == "2 smaller in ollama"
    # In its group nothing is merged: the two small models are there.
    group = page.tiles[merged["data-go"]]
    assert {"tiny-a", "tiny-b"} <= {t["data-name"] for t in group}
    assert all(t.get("tabindex") == "0" and t.get("role") == "button" for t in top)


def test_record_view_draws_its_files(machine: FakeMachine, ctx: Context) -> None:
    result = scan(ctx)
    built = Page(render_html(result))
    index = next(i for i, r in enumerate(result.records) if r.name == "acme/Sharded")
    files = built.tiles[record_view(result.records[index])]
    assert sorted(t["data-name"] for t in files) == [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert all("data-go" not in t for t in files)  # a file is the last level
    tiny = next(i for i, r in enumerate(result.records) if r.name == "acme/tiny")
    assert "not in the current revision" in built.text
    tiny_view = record_view(result.records[tiny])
    assert any(t["data-name"] == "model.safetensors" for t in built.tiles[tiny_view])


def test_panel_shows_location_files_and_removal(machine: FakeMachine, ctx: Context) -> None:
    result = scan(ctx)
    html = render_html(result)
    assert "modelmap rm ollama:tiny:1b" in html
    assert "Runs `ollama rm tiny:1b tiny:1b-4k`" in html
    assert "modelmap rm hf:acme/tiny" in html
    assert "Runs `hf cache rm model/acme/tiny --cache-dir " in html
    assert "The page itself deletes nothing" in html
    assert str(machine.files["ollama_weights"].resolve()) in html
    assert "Problems (" in html and "broken link: broken.bin" in html
    assert "No size, not on the map (" in html  # the model whose weights never arrived


def test_ollama_command_only_for_the_store_ollama_uses(machine: FakeMachine) -> None:
    # The same store, found only by its layout under a path the user named:
    # as on a backup drive. `ollama rm` would act on some other store.
    found = scan(
        Context(machine.home, {}, "darwin"),
        Options(paths=[str(machine.root / "shared")], defaults=False),
    )
    assert any(r.name == "tiny:1b" for r in found.records)
    html = render_html(found)
    assert "modelmap rm ollama:" not in html
    assert "not the one the Ollama on this machine uses" in html


def test_shared_files_are_called_out(machine: FakeMachine) -> None:
    twin = machine.home / ".lmstudio" / "models" / "acme" / "big" / "big.F16.gguf"
    twin.parent.mkdir(parents=True)
    os.link(machine.files["hf_big_gguf"], twin)
    html = render_html(scan(Context(machine.home, {}, "darwin")))
    assert "Some of these files are also used by: lmstudio:acme/big/big.F16." in html
    assert "counted under another record; also used by hf:acme/big-GGUF" in html


def test_page_loads_nothing_and_runs_nothing_from_names() -> None:
    evil = '</script><img src=x onerror="window.pwned=1">'
    hostile = Scan(
        records=[
            ModelRecord(
                evil,
                "generic",
                "gguf",
                "/tmp/" + evil,
                5 * GB,
                5 * GB,
                aliases=[evil],
                issues=[evil],
                file_list=[
                    FileEntry("/tmp/" + evil, evil, 5 * GB, 5 * GB, note=evil, used_by=[evil])
                ],
            ),
            ModelRecord("http://example.com/x", "hf", "gguf", "/c/models--a--b", 3 * GB, 3 * GB),
        ]
    )
    html = render_html(hostile)
    built = Page(html)
    assert built.elements.count("script") == 1 and "img" not in built.elements
    assert html.count("</script>") == 1
    assert evil.lower() in built.tiles["top-provider"][0]["data-name"]  # there, as text
    # Nothing is fetched: the only src/href are the inline icon and the link
    # home (followed only when clicked), and the script has no network calls.
    # A model name may itself contain "http".
    icon, home = re.findall(r'(?:src|href)="([^"]*)"', html)
    assert icon.startswith("data:image/svg+xml;base64,")
    assert home == HOME_PAGE
    script = html[html.index("<script>") : html.index("</script>")]
    for word in ("fetch(", "XMLHttpRequest", "import(", "innerHTML", "eval(", "document.write"):
        assert word not in script
    assert "url(" not in html.replace("url(#clip", "")


def test_page_is_stable_and_bounded() -> None:
    html = render_html(small_scan())
    assert html == render_html(small_scan())
    assert html.startswith("<!doctype html>") and html.endswith("</html>\n")
    assert "<title>Local models</title>" in html
    assert "prefers-color-scheme:dark" in html and "data-theme=dark" in html
    assert len(html) < 120_000  # 11 records; the page must not grow by the square


def test_empty_scan_and_disk_sizes() -> None:
    empty = Page(render_html(Scan()))
    assert "top-provider" in empty.views and "No models found" in empty.text
    sparse = Scan(
        records=[
            ModelRecord(
                "a",
                "hf",
                "gguf",
                "/c/models--x--a",
                100,
                10,
                file_list=[FileEntry("/a", "a", 100, 10)],
            ),
            ModelRecord(
                "b",
                "hf",
                "gguf",
                "/c/models--x--b",
                100,
                90,
                file_list=[FileEntry("/b", "b", 100, 90)],
            ),
        ]
    )
    by_size = Page(render_html(sparse))
    by_disk = Page(render_html(sparse, use_disk=True))
    assert sorted(s for _, _, s in by_size.rects["top-provider"]) == [100, 100]
    assert sorted(s for _, _, s in by_disk.rects["top-provider"]) == [10, 90]
    assert by_disk.rects[record_view(sparse.records[0])][0][2] == 10


def test_redacted_scan_makes_a_page_without_the_home_path(
    machine: FakeMachine, ctx: Context
) -> None:
    html = render_html(redact(scan(ctx), machine.home))
    assert str(machine.home) not in html
    assert str(Path(machine.home).resolve()) not in html
    assert "~/.omlx/models/acme/Sharded" in html


def test_view_ids_survive_a_removal() -> None:
    # After a removal the page reloads at its address; that must open the
    # same record or group, not whichever now sits at the same position.
    full = small_scan()
    fewer = Scan(records=full.records[1:])
    before, after = Page(render_html(full)), Page(render_html(fewer))
    gone = record_view(full.records[0])
    assert gone in before.views and gone not in after.views
    kept = set(before.views) - {gone}
    assert {v for v in kept if v.startswith("rec-")} <= set(after.views)
    for by in GROUPINGS:
        assert {v for v in kept if v.startswith(f"grp-{by}-")} >= {
            v for v in after.views if v.startswith(f"grp-{by}-")
        }


def test_records_with_one_path_get_two_views() -> None:
    (record,) = small_scan().records[:1]
    twice = Page(render_html(Scan(records=[record, record])))
    rec = [v for v in twice.views if v.startswith("rec-")]
    assert len(rec) == 2
    assert len(twice.ids) == len(set(twice.ids))


def test_the_banner_names_the_tool_its_version_and_home() -> None:
    html = render_html(small_scan())
    banner = html.split('<div class="brand">', 1)[1].split("<header>", 1)[0]
    assert "MODELMAP" in banner
    assert f"v{__version__}" in banner
    assert f'href="{HOME_PAGE}"' in banner and "Flux-Frontiers" in banner
    assert '<svg class="logo" aria-hidden="true"' in banner and "<title>" not in banner
    assert 'rel="icon" href="data:image/svg+xml;base64,' in html
    # Still nothing loaded from the network: the only external URL is a link.
    assert re.findall(r'(?:src|href)="(https?:[^"]+)"', html) == [HOME_PAGE]
