# Command reference

```
modelmap scan      [PATH ...] [--json] [scan options]
modelmap list      [PATH ...] [--from-json FILE] [--sort KEY] [--group KEY] [--all] [scan options]
modelmap map       [PATH ...] [--from-json FILE] [-o FILE] [--svg] [--disk] [--open] [scan options]
modelmap providers
modelmap rm        TARGET ... [--dry-run] [-y | --yes]
modelmap serve     [PATH ...] [--from-json FILE] [--disk] [--port N] [--no-open] [--read-only] [scan options]
modelmap --version
```

`python -m modelmap` runs the same commands.

## Scan options

`scan`, `list` and `map` accept these.

| Option | Effect |
|--------|--------|
| `PATH ...` | Also scan these files or folders. A path that is an Ollama store or a Hugging Face cache, or holds one, is read as that store. Naming a path turns off the home sweep |
| `--only NAMES` | Scan only these stores, comma-separated: `hf`, `ollama`, `omlx`, `lmstudio`, `llamacpp`, `gpt4all`, `jan`, `whisper`. Turns off the home sweep |
| `--no-defaults` | Scan only the given paths, not the stores. Needs at least one `PATH` |
| `--no-sweep` | Skip the sweep of the home folder |
| `--exclude GLOB` | Skip files and folders whose name or path matches. Repeat for several patterns |
| `--min-size SIZE` | Smallest checkpoint (`.pt`, `.bin` and similar) to list when only its extension says it is a model. Default `50MB`. Accepts `1.5GB`, `4GiB` or a byte count |
| `--max-depth N` | Levels to descend below each given path |
| `--redact` | Replace the home folder with `~` and drop the machine's name |
| `--strict` | Exit with status 1 if anything could not be read |

## `modelmap scan`

Prints a summary by store: size, space on disk, number of models, other
entries, and entries with problems.

| Option | Effect |
|--------|--------|
| `--json` | Print the whole inventory as JSON instead, for `--from-json` elsewhere |

## `modelmap list`

Prints every model, grouped by store and largest first.

| Option | Effect |
|--------|--------|
| `--from-json FILE` | Read an inventory instead of scanning. Takes no scan options |
| `--sort KEY` | `size` (default), `name`, `params` or `mtime` |
| `--group KEY` | `provider` (default), `format`, `family`, `kind` or `none` |
| `--all` | Include datasets, caches and other entries |

## `modelmap map`

Writes the map: `modelmap.html` in the current folder, or `modelmap.svg`
with `--svg`.

| Option | Effect |
|--------|--------|
| `--from-json FILE` | Draw an inventory instead of scanning. Takes no scan options |
| `-o FILE`, `--output FILE` | Write here instead |
| `--svg` | Write a static picture of the top level, not the interactive page |
| `--disk` | Size tiles by space allocated on disk, not by file size |
| `--open` | Open the result in a browser |

## `modelmap providers`

Shows, for each store, whether it was found, which setting chose its
place (`HF_HOME`, `settings.json`, `default` and so on), and its path,
with the target of any symlink.

## `modelmap rm`

Removes models after showing what will go and asking. A `TARGET` is the
`store:name` that `modelmap list` shows, an Ollama tag, or a path. See
[Removing models](guide.md#removing-models) for what is removed how, and
what is refused.

| Option | Effect |
|--------|--------|
| `--dry-run` | Show what would be removed; remove nothing |
| `-y`, `--yes` | Do not ask. Without it, `rm` asks on the terminal, and refuses when there is no terminal |

## `modelmap serve`

Scans, then serves the map from this machine at `127.0.0.1` and opens it
in a browser. The page is the one `map` writes, with a Rescan button and,
in each model's panel, a Remove button. Remove does what `modelmap rm`
does, through the same code: it shows every command or path in a dialog
and acts only when you confirm. Before acting, the server scans again and
removes nothing if what would go has changed since the dialog opened.
The address printed holds a token made for this run; without it the
server answers nothing. Ctrl-C stops the server.

| Option | Effect |
|--------|--------|
| `--from-json FILE` | Serve an inventory instead of scanning; Rescan reads the file again. Takes no scan options |
| `--disk` | Size tiles by space allocated on disk, not by file size |
| `--port N` | Listen on this port. The default, 0, takes any free port |
| `--no-open` | Print the address; do not open a browser |
| `--read-only` | Offer no Remove button. `--from-json` implies it: an inventory is not this disk as it is now |

`--redact` is refused: it is for sharing a map, and the served page is
for this machine's owner. The server listens on `127.0.0.1` only and
refuses requests that name another host, come from another page, or lack
the token.

## Exit status

| Status | Meaning |
|--------|---------|
| 0 | Done |
| 1 | `--strict` and something could not be read; or `rm` found no match for a target, refused, was declined, or failed; or `serve` could not listen on the port |
| 2 | The command line was wrong, or an inventory could not be read |
