"""Stand-ins for the programs `modelmap rm` runs: trash, ollama and hf.

They are real executables on a PATH of their own, so the code under test
runs them exactly as it would run the real ones. Only these programs are
replaced; everything they act on is real files in the test's directory.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

_TRASH = """#!/bin/sh
for item in "$@"; do
  /bin/mv "$item" "$FAKE_TRASH/" || exit 5
done
"""

_OLLAMA = """#!/bin/sh
case "$1" in
  list)
    echo "NAME ID SIZE MODIFIED"
    while read -r tag; do echo "$tag abc 1GB now"; done < "$FAKE_TAGS"
    ;;
  rm)
    shift
    for tag in "$@"; do
      /usr/bin/grep -qxF "$tag" "$FAKE_TAGS" || { echo "model '$tag' not found" >&2; exit 1; }
      /usr/bin/grep -vxF "$tag" "$FAKE_TAGS" > "$FAKE_TAGS.new"
      /bin/mv "$FAKE_TAGS.new" "$FAKE_TAGS"
    done
    ;;
esac
"""

_HF = """#!/bin/sh
# hf cache rm <type>/<org>/<name> --cache-dir DIR --yes
[ "$1 $2" = "cache rm" ] || exit 2
target="$3"; dir="$5"
kind="${target%%/*}"; repo="${target#*/}"
folder="$dir/${kind}s--$(echo "$repo" | /usr/bin/sed 's#/#--#g')"
[ -d "$folder" ] || { echo "not found: $target" >&2; exit 1; }
/bin/rm -rf "$folder"
"""


@dataclass
class Tools:
    """Where the stand-ins keep their state."""

    bin: Path
    trash: Path
    tags: Path

    def env(self, *names: str) -> dict[str, str]:
        """An environment whose PATH holds only the named programs."""
        chosen = self.bin.parent / ("bin-" + "-".join(sorted(names)) if names else "bin-none")
        chosen.mkdir(exist_ok=True)
        for name in names:
            link = chosen / name
            if not link.exists():
                link.symlink_to(self.bin / name)
        return {"PATH": str(chosen), "FAKE_TRASH": str(self.trash), "FAKE_TAGS": str(self.tags)}


def install(root: Path, tags: list[str]) -> Tools:
    """Write the stand-ins under ``root``."""
    tools = Tools(bin=root / "bin", trash=root / "Trash", tags=root / "ollama-tags")
    tools.bin.mkdir(parents=True)
    tools.trash.mkdir()
    tools.tags.write_text("".join(f"{tag}\n" for tag in tags))
    for name, body in (("trash", _TRASH), ("ollama", _OLLAMA), ("hf", _HF)):
        path = tools.bin / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    assert os.access(tools.bin / "trash", os.X_OK)
    return tools
