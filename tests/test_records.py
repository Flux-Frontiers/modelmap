"""Records: totals and the JSON inventory round trip."""

from __future__ import annotations

import json

import pytest

from modelmap.records import SCHEMA_VERSION, ModelRecord, Scan, ScanRoot, ScanWarning


def _scan() -> Scan:
    return Scan(
        records=[
            ModelRecord(
                name="acme/tiny",
                provider="hf",
                format="safetensors",
                path="/home/u/.cache/huggingface/hub/models--acme--tiny",
                size_bytes=2_000_000_000,
                disk_bytes=1_999_998_976,
                quant="F16",
                params=1_000_000_000,
                mtime=1_700_000_000.5,
                files=3,
                issues=["broken link: broken.bin"],
            ),
            ModelRecord(
                name="tiny:1b-4k",
                provider="ollama",
                format="gguf",
                path="/shared/ollama",
                size_bytes=0,
                disk_bytes=0,
                kind="model",
                params=5_000_000_000,
                params_exact=False,
                shared_with=["tiny:1b"],
            ),
        ],
        roots=[
            ScanRoot("ollama", "/home/u/.ollama/models", "/shared/ollama", "default"),
            ScanRoot(
                "lmstudio", "/home/u/.lmstudio/models", "/home/u/.lmstudio/models", "default", False
            ),
        ],
        warnings=[ScanWarning("/home/u/hostile/denied", "Permission denied")],
        elapsed_s=0.42,
        hostname="box",
        tool_version="0.1.0",
    )


def test_totals_sum_the_records() -> None:
    scan = _scan()
    assert scan.size_bytes == 2_000_000_000
    assert scan.disk_bytes == 1_999_998_976
    assert Scan().size_bytes == 0


def test_json_round_trip_is_lossless() -> None:
    scan = _scan()
    assert Scan.from_json(scan.to_json()) == scan
    assert Scan.from_json(scan.to_json(indent=None)) == scan


def test_json_is_plain_data() -> None:
    data = json.loads(_scan().to_json())
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["records"][0]["name"] == "acme/tiny"
    assert data["records"][1]["shared_with"] == ["tiny:1b"]
    assert data["roots"][0]["resolved"] == "/shared/ollama"
    assert data["warnings"][0]["message"] == "Permission denied"


@pytest.mark.parametrize("version", [None, 0, SCHEMA_VERSION + 1, "1"])
def test_other_schema_versions_are_refused(version: object) -> None:
    data = _scan().to_dict()
    data["schema_version"] = version
    with pytest.raises(ValueError, match="schema version"):
        Scan.from_dict(data)


def test_missing_schema_version_is_refused() -> None:
    with pytest.raises(ValueError, match="schema version"):
        Scan.from_json("{}")


def test_fields_added_by_a_newer_version_are_ignored() -> None:
    data = _scan().to_dict()
    data["added_later"] = True
    data["records"][0]["added_later"] = [1, 2]
    data["roots"][0]["added_later"] = "x"
    data["warnings"][0]["added_later"] = None

    assert Scan.from_dict(data) == _scan()
