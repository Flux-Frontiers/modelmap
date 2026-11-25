# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **The interactive map.** `modelmap map` writes one page that works
  offline. Clicking a tile opens the model and redraws the square as its
  individual files; clicking a store shows only that store. The page can
  group by store, format, family or kind, filter by name, and lists every
  entry with a problem. It follows the light or dark setting and can be
  used from the keyboard. `--svg` writes the static picture, `--disk`
  sizes by space on disk, `--open` opens the result.
- **Scanner.** A file named on the command line is no longer listed a
  second time by the home sweep.
- **Files of each record.** Every record now lists its files: real path,
  size, what the file is within the model, and every other record that
  uses it. Hugging Face blobs are named from the current snapshot, and a
  blob no longer in it is marked. Ollama blobs are named by layer.
- **`modelmap rm`.** Removes models, after listing every file or command
  and asking (`--dry-run` lists only; `--yes` skips the question). Ollama
  models go through `ollama rm`, with every tag of the same weights;
  Hugging Face cache repositories through `hf cache rm`; everything else
  moves to the Trash. Caches, unreferenced files, copies of an Ollama
  store and models whose files another model uses are refused. The map
  shows the `modelmap rm` command for each model.
- **Treemap layout.** `modelmap.treemap` divides a rectangle among sizes
  with the squarified algorithm of Bruls, Huizing and van Wijk. Areas are
  exact shares of the total; equal sizes are ordered by name, so a layout
  is repeatable. Items too small to see are merged into one tile per
  provider.
- **Static SVG.** `modelmap.render_svg` draws a scan as one self-contained
  SVG: a square map with one block per provider, a name tag on each block,
  labels clipped to their tiles, a tooltip per tile, a legend, and light
  and dark colors. Every record is a tile, a note on the tile that holds
  its files, or counted as not drawn.
- **The `modelmap` command.** `modelmap scan` scans the known stores and
  sweeps the home directory for loose models, then prints a summary by
  provider, or the whole inventory with `--json`. `modelmap list` prints
  every model, sorted and grouped, from a scan or from an inventory file
  made on another machine (`--from-json`). `modelmap providers` shows where
  each store is looked for and which setting chose the location.
- **Scanning any path.** A named path may be a directory, one model
  directory, or one file. A path that is an Ollama store or a Hugging Face
  cache, or contains one, is read by that provider. A directory is scanned
  once however many names lead to it.
- **`--redact`.** Replaces the home directory with `~` and drops the host
  name, for an inventory you send to someone else.
- **No downloads.** A model file that is a cloud placeholder (iCloud Drive
  and similar) is listed by size and not opened, since opening it would
  fetch it.
- **Providers.** `modelmap.providers` finds models in the Hugging Face hub
  cache (including the store shared between repositories and the Xet
  cache), Ollama, oMLX, LM Studio, llama.cpp, GPT4All, Jan and Whisper, and
  in any directory tree. Environment variables and settings files that
  move a store are honored, and a store reached through a symlink is
  followed. Ollama tags that point at the same weights are one record with
  aliases. A file reachable from two stores is charged to the first.
- **Header readers.** `modelmap.headers` reads GGUF and safetensors headers
  without reading weights: parameter count, quantization, architecture. A
  bad or truncated file gives an issue, never an exception.
- **Sizing.** `modelmap.sizing` reports apparent and on-disk size and
  charges a file shared by several names once.
- **Data model.** `modelmap.records` holds a scan and round-trips it
  through JSON.
