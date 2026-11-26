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

_OLLAMA = r"""#!/bin/sh
# The tags the daemon knows are lines of $FAKE_TAGS. Like Ollama, `list`
# shows each tag's ID, the first 12 hex digits of its manifest's sha256, and
# `rm` deletes the manifest, both in the store at $FAKE_OLLAMA_STORE.
manifest() {
  name="${1%:*}"; tag="${1##*:}"
  case "$name" in
    */*/*) repo="$name" ;;
    */*) repo="registry.ollama.ai/$name" ;;
    *) repo="registry.ollama.ai/library/$name" ;;
  esac
  for tree in manifests-v2 manifests; do
    file="$FAKE_OLLAMA_STORE/$tree/$repo/$tag"
    [ -n "$FAKE_OLLAMA_STORE" ] && [ -f "$file" ] && { echo "$file"; return; }
  done
}
digest() {
  if [ -x /usr/bin/sha256sum ]; then /usr/bin/sha256sum "$1"; else /usr/bin/shasum -a 256 "$1"; fi     | /usr/bin/cut -c1-12
}
case "$1" in
  list)
    echo "NAME ID SIZE MODIFIED"
    while read -r tag; do
      file="$(manifest "$tag")"
      if [ -n "$file" ]; then id="$(digest "$file")"; else id=000000000000; fi
      echo "$tag $id 1GB now"
    done < "$FAKE_TAGS"
    ;;
  rm)
    shift
    for tag in "$@"; do
      /usr/bin/grep -qxF "$tag" "$FAKE_TAGS" || { echo "model '$tag' not found" >&2; exit 1; }
      /usr/bin/grep -vxF "$tag" "$FAKE_TAGS" > "$FAKE_TAGS.new"
      /bin/mv "$FAKE_TAGS.new" "$FAKE_TAGS"
      file="$(manifest "$tag")"
      if [ -n "$file" ]; then /bin/rm -f "$file"; fi
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
    #: The Ollama store the fake daemon serves; None serves none.
    ollama_store: Path | None = None

    def env(self, *names: str) -> dict[str, str]:
        """An environment whose PATH holds only the named programs."""
        chosen = self.bin.parent / ("bin-" + "-".join(sorted(names)) if names else "bin-none")
        chosen.mkdir(exist_ok=True)
        for name in names:
            link = chosen / name
            if not link.exists():
                link.symlink_to(self.bin / name)
        env = {"PATH": str(chosen), "FAKE_TRASH": str(self.trash), "FAKE_TAGS": str(self.tags)}
        if self.ollama_store is not None:
            env["FAKE_OLLAMA_STORE"] = str(self.ollama_store)
        return env


def install(root: Path, tags: list[str], ollama_store: Path | None = None) -> Tools:
    """Write the stand-ins under ``root``.

    :param tags: The tags the fake Ollama daemon knows.
    :param ollama_store: The store it serves; its manifests give the IDs.
    """
    tools = Tools(
        bin=root / "bin", trash=root / "Trash", tags=root / "ollama-tags", ollama_store=ollama_store
    )
    tools.bin.mkdir(parents=True)
    tools.trash.mkdir()
    tools.tags.write_text("".join(f"{tag}\n" for tag in tags))
    for name, body in (("trash", _TRASH), ("ollama", _OLLAMA), ("hf", _HF)):
        path = tools.bin / name
        path.write_text(body)
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    assert os.access(tools.bin / "trash", os.X_OK)
    return tools
