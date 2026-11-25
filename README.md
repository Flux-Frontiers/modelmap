# modelmap

Find the LLM model files on a machine and report them by size.

modelmap reads the stores of Hugging Face, Ollama, oMLX, LM Studio,
llama.cpp, GPT4All, Jan and Whisper, follows the settings and symlinks
that move them, and sweeps the home directory for loose GGUF, safetensors
and checkpoint files. It identifies files by their headers, reports
parameter counts and quantization, and counts a file once however many
names point at it.

Status: pre-alpha, not yet released.

## Use

```bash
modelmap map --open           # the map: one page, opened in your browser
modelmap scan                 # summary by provider
modelmap list                 # every model, largest first
modelmap providers            # where each store is looked for
modelmap rm ollama:qwen3:4b   # remove a model, after showing what goes
```

### The map

`modelmap map` writes `modelmap.html`, a single page that works offline. A
square is divided among everything found, by size, with one block per
store.

- Click a tile to open the model. The square is redrawn as that model's
  individual files, and each file's full path is listed beside it.
- Click a store's name in the list beside the map, or a "smaller" tile, to
  see only that store.
- Press Escape, or use the trail above the map, to go back up.
- Group by store, format, family or kind, and filter by name.

For each model the page shows where it is, which other models use the
same files, and the `modelmap rm` command that removes it, ready to copy.
The page itself deletes nothing.

### Removing models

```bash
modelmap rm --dry-run ollama:qwen3:4b    # show what would go
modelmap rm ollama:qwen3:4b              # show it, ask, then remove
modelmap rm ~/models/old.gguf --yes      # a path works too; --yes skips the question
```

A target is the `provider:name` that `modelmap list` shows, an Ollama tag,
or a path. When two entries share a name, `rm` lists their paths and
removes neither; name one by its path.

- Ollama models are removed with `ollama rm`, giving every tag that points
  at the same weights, since Ollama frees the space only when all of them
  are gone. When those tags differ in their settings (a longer context
  made with `ollama create`, for example), the plan names them, and with
  `--yes` each must be named on the command line. `ollama rm` acts on the
  Ollama that is running, so a copy of an Ollama store elsewhere is
  refused.
- Hugging Face cache repositories are removed with `hf cache rm`.
- Everything else moves to the Trash (`trash` on macOS; `trash` or `gio
  trash` on Linux). A model folder moves whole, and is refused when it
  also holds something that is not part of the model, such as a `.git`
  checkout. Loose files move one by one.
- Refused, and nothing is removed: caches, files nothing refers to (for
  Hugging Face, `hf cache prune` handles those), any model whose files
  another model also uses, and a repository that a file elsewhere links
  into.

Each run scans first, shows every file or command, and asks. When one
target is refused, none is removed. Space moved to the Trash is freed
when the Trash is emptied.

"Used by" is known only among what the scan covered. After `--only` or
`--no-defaults`, a file can show no other user and still have one in a
store that was not scanned. An inventory is read as it is: redact it when
you make it (`scan --json --redact`), not when you map it.

`modelmap map --svg` writes a static picture of the top level instead.

To scan somewhere else, name it. The path can be a directory, a single
model directory, or a single file:

```bash
modelmap list /Volumes/External
modelmap scan --no-defaults ~/Downloads/model.gguf
```

To look at a machine you cannot reach, ask its owner for an inventory:

```bash
modelmap scan --json --redact > inventory.json     # on their machine
modelmap map --from-json inventory.json --open     # on yours
```

`--redact` replaces the home directory with `~` and drops the host name.

### What the scan does and does not do

- It reads file headers only, never weights, and changes nothing on disk.
- It sends nothing anywhere.
- It does not open a model file that is a cloud placeholder (iCloud Drive
  and similar), because that would download it. Such a file is listed by
  size with a note.
- A file it cannot read becomes a warning. The scan continues.
- The first scan after a restart is slower, because the directory walk is
  not yet in the file cache.

### Options

| Option | Effect |
|--------|--------|
| `PATH ...` | Scan these files or directories as well |
| `--no-defaults` | Scan only the given paths |
| `--no-sweep` | Skip the sweep of the home directory |
| `--only NAMES` | Run only these providers, for example `hf,ollama` |
| `--exclude GLOB` | Skip matching files and directories; repeatable |
| `--max-depth N` | Levels to descend below each given path |
| `--min-size SIZE` | Smallest checkpoint to list when only its extension says it is a model (default `50MB`) |
| `--redact` | Hide the home directory and host name |
| `--strict` | Exit with status 1 if anything could not be read |
| `--json` (`scan`) | Print the inventory as JSON |
| `--from-json FILE` (`list`, `map`) | Read an inventory instead of scanning |
| `-o FILE`, `--svg`, `--disk`, `--open` (`map`) | Where to write; a static picture; size by space on disk; open in a browser |
| `--sort`, `--group`, `--all` (`list`) | Order, grouping, and whether datasets and caches are shown |

## Development

```bash
poetry install --with dev
.venv/bin/pytest
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
