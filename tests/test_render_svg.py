"""The SVG: honest areas as drawn, legal XML, every record accounted for."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from modelmap.mapdata import build
from modelmap.providers.base import Context
from modelmap.records import ModelRecord, Scan
from modelmap.render_svg import HEADER, MARGIN, render_svg
from modelmap.scanner import scan
from tests.fakemachine import FakeMachine

GB = 1000**3
GOLDEN = Path(__file__).parent / "golden" / "small.svg"
NS = "{http://www.w3.org/2000/svg}"


def record(name: str, provider: str, size: int, **fields: object) -> ModelRecord:
    fields.setdefault("format", "gguf")
    fields.setdefault("path", f"/models/{provider}/{name}")
    return ModelRecord(name=name, provider=provider, size_bytes=size, disk_bytes=size, **fields)  # type: ignore[arg-type]


def small_scan() -> Scan:
    """A scan written by hand, so the picture never depends on a machine."""
    return Scan(
        records=[
            record(
                "org/Big-27B-8bit",
                "omlx",
                30 * GB,
                format="mlx",
                quant="8-bit",
                params=27_400_000_000,
                params_exact=False,
            ),
            record(
                "org/Mid-14B-4bit",
                "omlx",
                8 * GB,
                format="mlx",
                quant="4-bit",
                params=14_800_000_000,
                params_exact=False,
            ),
            record(
                "qwen3:8b",
                "ollama",
                5 * GB,
                quant="Q4_K_M",
                params=8_190_000_000,
                aliases=["qwen3:latest"],
            ),
            record("llama3.2:3b", "ollama", 2 * GB, quant="Q4_K_M", params=3_210_000_000),
            record("tiny-a", "ollama", 20_000_000, quant="F16"),
            record("tiny-b", "ollama", 10_000_000, quant="F16"),
            record("acme/sdxl", "hf", 14 * GB, format="safetensors"),
            record("acme/words", "hf", 1 * GB, format="files", kind="dataset"),
            record(
                "checkpoints/last",
                "generic",
                3 * GB,
                format="pytorch",
                issues=["unverified: identified by extension only"],
            ),
            record("acme/copy", "lmstudio", 0, shared_with=["hf:acme/sdxl"]),
            record("gone:latest", "ollama", 0, format="unknown", issues=["missing blob"]),
        ]
    )


def tiles(svg: str) -> list[tuple[float, float, float, float, int]]:
    root = ET.fromstring(svg)
    found = []
    for rect in root.iter(f"{NS}rect"):
        if rect.get("class") == "tile":
            box = [float(rect.get(k, "0")) for k in ("x", "y", "width", "height")]
            found.append((*box, int(rect.get("data-size", "0"))))  # type: ignore[arg-type]
    return found  # type: ignore[return-value]


# ------------------------------------------------------------ what is drawn


@pytest.mark.parametrize("size", [1000, 640])
def test_drawn_areas_are_in_proportion_across_groups(size: int) -> None:
    drawn = tiles(render_svg(small_scan(), size=size))
    total = sum(t[4] for t in drawn)
    canvas = size * size
    # Coordinates are written to a thousandth of a unit.
    slack = 4e-3 * size
    for x, y, w, h, tile_size in drawn:
        assert abs(w * h - tile_size / total * canvas) <= slack
        assert x >= MARGIN - 1e-3 and y >= HEADER - 1e-3
        assert x + w <= MARGIN + size + 1e-3 and y + h <= HEADER + size + 1e-3
    assert sum(w * h for _, _, w, h, _ in drawn) == pytest.approx(canvas, rel=1e-5)


def test_sizes_drawn_add_up_to_the_scan() -> None:
    scan_result = small_scan()
    drawn = tiles(render_svg(scan_result))
    assert sum(t[4] for t in drawn) == scan_result.size_bytes
    # The two tiny Ollama models are one merged tile.
    assert 30_000_000 in [t[4] for t in drawn]
    assert len(drawn) == 8


def test_every_record_is_a_tile_a_note_or_counted() -> None:
    scan_result = small_scan()
    data = build(scan_result)
    attached = sum(len(v) for v in data.attached.values())
    assert len(data.items) + attached + len(data.not_drawn) == len(scan_result.records)
    assert (len(data.items), attached, len(data.not_drawn)) == (9, 1, 1)

    svg = render_svg(scan_result)
    assert "Same files as: acme/copy (lmstudio)" in svg  # on the owner's tile
    assert "1 more name for a file already on the map is listed" in svg
    assert "1 entry has no size and is not drawn." in svg
    assert "9 entries with a size: 8 models, 1 other (datasets, caches and other files)" in svg


def test_note_needs_exactly_one_owner() -> None:
    twins = Scan(
        records=[
            record("last", "generic", 5, path="/a/last"),
            record("last", "generic", 7, path="/b/last"),
            record("copy", "generic", 0, shared_with=["generic:last"]),  # two answer
            record("lost", "generic", 0, shared_with=["hf:not/scanned"]),  # none answers
            record("self", "generic", 0, shared_with=["generic:self"]),  # only itself
            record("chain", "generic", 0, shared_with=["generic:lost"]),  # owner has no tile
        ]
    )
    data = build(twins)
    assert data.attached == {}
    assert data.not_drawn == [2, 3, 4, 5]
    assert len(data.items) == 2


def test_disk_size_can_set_the_areas() -> None:
    sparse = Scan(
        records=[
            ModelRecord("a", "hf", "gguf", "/a", 100, 10),
            ModelRecord("b", "hf", "gguf", "/b", 100, 90),
        ]
    )
    assert sorted(t[4] for t in tiles(render_svg(sparse))) == [100, 100]
    on_disk = render_svg(sparse, use_disk=True)
    assert sorted(t[4] for t in tiles(on_disk)) == [10, 90]
    assert "Area shows space allocated on disk." in on_disk


# ------------------------------------------------------------- labels and text


def test_labels_are_clipped_to_their_tiles() -> None:
    root = ET.fromstring(render_svg(small_scan()))
    clips = {
        clip.get("id"): tuple(clip[0].get(k) for k in ("x", "y", "width", "height"))
        for clip in root.iter(f"{NS}clipPath")
    }
    boxes = {
        tuple(rect.get(k) for k in ("x", "y", "width", "height"))
        for rect in root.iter(f"{NS}rect")
        if rect.get("class") == "tile"
    }
    labelled = [g for g in root.iter(f"{NS}g") if g.get("clip-path")]
    assert labelled, "at least the large tiles carry a label"
    for group in labelled:
        clip_id = group.get("clip-path", "")[5:-1]
        assert clips[clip_id] in boxes  # the clip is exactly one tile
        x, y, w, h = (float(v) for v in clips[clip_id])  # type: ignore[arg-type]
        for text in group.iter(f"{NS}text"):
            assert x <= float(text.get("x", "0")) <= x + w
            assert y <= float(text.get("y", "0")) <= y + h


def test_each_group_is_named_on_the_map_and_in_the_legend() -> None:
    svg = render_svg(small_scan())
    for tag in ("OMLX  38.0 GB", "HUGGING FACE  15.0 GB", "OLLAMA  7.0 GB"):
        assert tag in svg
    for legend in ("oMLX", "Hugging Face", "Ollama", "Loose files"):
        assert f">{legend}<tspan" in svg
    assert "38.0 GB, 60%, 2 entries" in svg
    # Tiles carry the short name; the tooltip has the full one.
    assert ">Big-27B-8bit</text>" in svg and "org/Big-27B-8bit\n" in svg
    assert "30.0 GB  8-bit" in svg and "~27.4B parameters" in svg


def test_block_too_narrow_for_a_tag_is_still_identified() -> None:
    lopsided = Scan(
        records=[
            record("huge", "hf", 1000 * GB),
            record("speck", "whisper", 1 * GB, format="pytorch"),
            record("dust-a", "jan", 100_000),
            record("dust-b", "jan", 100_000),
        ]
    )
    svg = render_svg(lopsided)
    assert "HUGGING FACE  " in svg
    assert "WHISPER  " not in svg and "JAN  " not in svg  # no room for a tag
    # The legend and every tooltip still name the provider.
    assert ">Whisper<tspan" in svg and ">Jan<tspan" in svg
    titles = [t.text or "" for t in ET.fromstring(svg).iter(f"{NS}title")]
    assert any(title.startswith("speck\nWhisper\n") for title in titles)
    assert any(title.startswith("2 smaller entries, 200 kB\nJan\n") for title in titles)


def test_problems_are_counted_in_a_note() -> None:
    svg = render_svg(small_scan())
    assert "2 entries have a problem, named in the tooltip" in svg
    assert "Problem: unverified: identified by extension only" in svg
    assert "a problem" not in render_svg(Scan(records=[record("fine", "hf", GB)]))


def test_label_text_is_drawn_at_full_strength() -> None:
    # Contrast is proved for the ink as it is; dimming it would undo that.
    svg = render_svg(small_scan())
    assert "opacity" not in svg


def test_names_are_escaped_and_illegal_characters_replaced() -> None:
    odd = Scan(
        records=[
            record('a<b&"c>', "generic", 5 * GB),
            record("caf\udcff", "generic", 4 * GB, path="/tmp/caf\udcff.gguf"),
            record("bell\x07", "generic", 3 * GB),
        ]
    )
    svg = render_svg(odd)
    root = ET.fromstring(svg)  # legal XML
    svg.encode("utf-8")  # no lone surrogate left
    titles = [t.text or "" for t in root.iter(f"{NS}title")]
    assert any(title.startswith('a<b&"c>') for title in titles)
    assert any(title.startswith("caf?") for title in titles)
    assert any(title.startswith("bell?") for title in titles)


def test_nothing_to_draw_still_renders() -> None:
    for empty in (Scan(), Scan(records=[record("gone", "ollama", 0)])):
        svg = render_svg(empty)
        ET.fromstring(svg)
        assert "No models found" in svg
        assert tiles(svg) == []


def test_one_record_fills_the_square() -> None:
    (only,) = tiles(render_svg(Scan(records=[record("solo", "hf", 7 * GB)])))
    assert only[:4] == (MARGIN, HEADER, 1000, 1000)


# ------------------------------------------------------------- the file itself


def test_output_is_stable_and_self_contained() -> None:
    svg = render_svg(small_scan())
    assert svg == render_svg(small_scan())
    assert svg.endswith("</svg>\n")
    assert not any(line != line.rstrip() for line in svg.splitlines())
    # No fonts, scripts, images or links fetched from anywhere.
    assert svg.count("http") == 1 and 'xmlns="http://www.w3.org/2000/svg"' in svg
    assert "<script" not in svg and "<image" not in svg and "href" not in svg
    assert "prefers-color-scheme:dark" in svg


def test_golden_picture() -> None:
    """The reference picture. To accept a deliberate change, run the tests
    with MODELMAP_UPDATE_GOLDEN=1 and look at the new file before committing.
    """
    svg = render_svg(small_scan())
    if os.environ.get("MODELMAP_UPDATE_GOLDEN"):
        GOLDEN.write_text(svg, encoding="utf-8")
    assert GOLDEN.read_text(encoding="utf-8") == svg


def test_a_real_scan_of_the_fake_machine_renders(machine: FakeMachine, ctx: Context) -> None:
    result = scan(ctx)
    svg = render_svg(result)
    ET.fromstring(svg)
    data = build(result)
    attached = sum(len(v) for v in data.attached.values())
    assert len(data.items) + attached + len(data.not_drawn) == len(result.records)
    assert sum(t[4] for t in tiles(svg)) == result.size_bytes
    assert str(machine.home) in svg  # paths are in the tooltips; redact before sharing
