#!/usr/bin/env python3
"""Take the README and docs screenshots of the map from example data.

The pictures never show a real machine: the scan is written here by hand,
the same models the render tests use (``tests/test_render_svg.small_scan``),
with a file list for each so a model can be opened down to its files.

Usage::

    poetry run python scripts/screenshots.py

Needs Google Chrome; set ``CHROME`` to the browser binary if it is not at
the default macOS path. Writes, in dark mode at 1400 x 980:

- ``assets/map.png``: the top view, grouped by store
- ``assets/map-files.png``: the largest model opened down to its files
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from modelmap.records import FileEntry, ModelRecord, Scan  # noqa: E402 (after the path is set)
from modelmap.render_html import record_view, render_html  # noqa: E402

CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
GB = 1000**3
SIZE = (1400, 980)


def record(name: str, provider: str, files: list[tuple[str, int]], **fields: object) -> ModelRecord:
    """A record under ``/models/<store>/<name>`` holding the given files."""
    path = f"/models/{provider}/{name}"
    entries = [FileEntry(f"{path}/{label}", label, size, size) for label, size in files]
    total = sum(size for _, size in files)
    fields.setdefault("format", "gguf")
    return ModelRecord(
        name=name,
        provider=provider,
        path=path,
        size_bytes=total,
        disk_bytes=total,
        file_list=entries,
        **fields,  # type: ignore[arg-type]
    )


def example() -> Scan:
    shards = [(f"model-0000{n}-of-00004.safetensors", 8 * GB) for n in (1, 2, 3)]
    return Scan(
        records=[
            record(
                "org/Big-27B-8bit",
                "omlx",
                [*shards, ("model-00004-of-00004.safetensors", 6 * GB), ("config.json", 4_000)],
                format="mlx",
                quant="8-bit",
                params=27_400_000_000,
                params_exact=False,
            ),
            record(
                "org/Mid-14B-4bit",
                "omlx",
                [("model.safetensors", 8 * GB), ("config.json", 3_000)],
                format="mlx",
                quant="4-bit",
                params=14_800_000_000,
                params_exact=False,
            ),
            record(
                "qwen3:8b",
                "ollama",
                [("model", 5 * GB)],
                quant="Q4_K_M",
                params=8_190_000_000,
                aliases=["qwen3:latest"],
            ),
            record(
                "llama3.2:3b", "ollama", [("model", 2 * GB)], quant="Q4_K_M", params=3_210_000_000
            ),
            record("tiny-a", "ollama", [("model", 20_000_000)], quant="F16"),
            record("tiny-b", "ollama", [("model", 10_000_000)], quant="F16"),
            record("acme/sdxl", "hf", [("unet.safetensors", 14 * GB)], format="safetensors"),
            record("acme/words", "hf", [("data.parquet", 1 * GB)], format="files", kind="dataset"),
            record(
                "checkpoints/last",
                "generic",
                [("last.ckpt", 3 * GB)],
                format="pytorch",
                issues=["unverified: identified by extension only"],
            ),
            record("gone:latest", "ollama", [], format="unknown", issues=["missing blob"]),
        ]
    )


def shoot(page: Path, fragment: str, out: Path) -> None:
    """Screenshot ``page`` at ``fragment`` in dark mode with headless Chrome."""
    subprocess.run(
        [
            CHROME,
            "--headless",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-dark-mode",
            "--blink-settings=preferredColorScheme=0",
            f"--window-size={SIZE[0]},{SIZE[1]}",
            "--force-device-scale-factor=1",
            f"--screenshot={out}",
            page.as_uri() + fragment,
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    scan = example()
    big = record_view(scan.records[0])
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "modelmap.html"
        page.write_text(render_html(scan), encoding="utf-8")
        shoot(page, "", ROOT / "assets" / "map.png")
        shoot(page, f"#{big}", ROOT / "assets" / "map-files.png")
    print("Wrote assets/map.png and assets/map-files.png")


if __name__ == "__main__":
    main()
