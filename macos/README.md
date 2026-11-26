# modelmap for macOS

A Mac app around `modelmap serve`. It opens a window, starts the server
from a Python bundled inside the app, and shows the map with its Remove
buttons. Scanning, the map and removal are the Python code the command
line runs; the app adds a window, a Dock icon, a Map menu and a server log,
and stops the server when it quits.

## Build

Needs Xcode, [XcodeGen](https://github.com/yonaskolb/XcodeGen)
(`brew install xcodegen`), uv and Poetry.

```bash
cd macos
make run      # stage Python, generate the project, build, and open the app
make test     # the Swift unit tests
make dmg      # a Developer ID signed, notarized modelmap-<version>.dmg in build/dmg
make clean
```

`make python` (run by `build` and `test`) copies uv's standalone CPython
into `macos/Python`, installs the modelmap wheel built from this checkout,
and removes what modelmap does not use, about 41 MB. The app's version is
read from `../pyproject.toml`. The build is for Apple silicon and signed to
run on this Mac only.

## How it behaves

- The server gets the environment of your login shell, as a terminal
  would: an app opened from the Finder otherwise lacks Homebrew's PATH, so
  `ollama` and `hf` are not found, and settings such as `HF_HOME` or
  `OLLAMA_MODELS` that move a store.
- The window shows only the server's own page. Other links open in your
  browser.
- Map > Choose Folders to Scan (Cmd-O) scans folders or drives you pick,
  with or without the usual stores and the home-folder sweep; the choice
  is remembered, and Map > Scan Usual Locations goes back.
- Map > Rescan (Cmd-R) scans again; Map > Open in Browser (Cmd-B) opens the
  same page in your browser; Window > Server Log (Cmd-L) shows every
  request and every removal with what it ran, with the token hidden.
- Quitting the app stops the server.

The app is not sandboxed: modelmap reads model stores across your home
folder and other drives, runs `ollama`, `hf` and `trash`, and moves files
to the Trash, none of which a sandbox allows. That also keeps it off the
Mac App Store.
