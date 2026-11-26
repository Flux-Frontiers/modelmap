# Release Notes: v0.3.0

> Released: 2026-11-26

modelmap now comes as a Mac app. `modelmap-0.3.0.dmg`, attached to this
release, holds a signed and notarized `modelmap.app`: the map in its own
window, with the same Remove buttons as `modelmap serve`, and a Python
inside it so the Mac needs nothing else installed. This release also makes
Ollama removals safer.

## What changed

**The Mac app.** Open the disk image and drag modelmap to Applications. The
app scans, shows the map, and removes models through the same code the
command line runs. Its Map menu rescans, opens the page in a browser, and
chooses folders or drives to scan, with or without the usual model stores
and the home folder. A server log shows every removal and the command it
ran, and Help explains each part. The app reads your login shell's
settings, so stores moved by `HF_HOME` or `OLLAMA_MODELS` are found. It
needs macOS 15 or later on Apple silicon.

**Safer Ollama removal.** Two gaps are closed. A removal used to count as
done when Ollama could not be asked afterward; it now counts only when
Ollama answers without the tags and the scanned store no longer holds
them. And matching tag names was not proof that the running Ollama owns
the store that was scanned: each tag's ID must now match the scanned
manifest, and an `OLLAMA_HOST` that names another machine is refused. In
either case nothing is removed, and the dialog says why.

**`--sweep`.** Naming a folder or a store turns off the sweep of the home
folder. `--sweep` keeps it, so `modelmap map --sweep /Volumes/Drive` maps
an external drive together with everything the default scan finds.

## Upgrading

Upgrade the command with `pipx upgrade modelmap` or `pip install -U
modelmap`; nothing needs migrating. For the app, download the disk image
from this release.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
