# modelmap

Find the LLM model files on a machine and report them by size.

modelmap reads the stores of Hugging Face, Ollama, oMLX, LM Studio,
llama.cpp, GPT4All, Jan and Whisper, follows the settings and symlinks
that move them, and sweeps the home directory for loose GGUF, safetensors
and checkpoint files. It identifies files by their headers, reports
parameter counts and quantization, and counts a file once however many
names point at it.

Status: pre-alpha. The scanner and the command line work; the treemap is
not built yet.

## Use

```bash
modelmap scan                 # summary by provider
modelmap list                 # every model, largest first
modelmap providers            # where each store is looked for
```

To scan somewhere else, name it. The path can be a directory, a single
model directory, or a single file:

```bash
modelmap list /Volumes/External
modelmap scan --no-defaults ~/Downloads/model.gguf
```

To look at a machine you cannot reach, ask its owner for an inventory:

```bash
modelmap scan --json --redact > inventory.json     # on their machine
modelmap list --from-json inventory.json           # on yours
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
| `--from-json FILE` (`list`) | Read an inventory instead of scanning |
| `--sort`, `--group`, `--all` (`list`) | Order, grouping, and whether datasets and caches are shown |

## Development

```bash
poetry install --with dev
.venv/bin/pytest
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
