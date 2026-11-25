# modelmap

Find the language-model files on a machine, see how much space each one
takes, and remove the ones you no longer need.

modelmap reads the stores of Hugging Face, Ollama, oMLX, LM Studio,
llama.cpp, GPT4All, Jan and Whisper, follows the settings and symlinks
that move them, and sweeps your home folder for loose GGUF, safetensors
and checkpoint files. It reads file headers for parameter counts and
quantization, counts each file once however many names point at it, and
draws everything as a map you can open down to the individual files.

<img src="https://raw.githubusercontent.com/Flux-Frontiers/modelmap/main/assets/map.png" alt="The map: one block per store, one tile per model, sized by space used" width="760">

Status: pre-alpha, not yet released. It runs on macOS and Linux with
Python 3.10 or later and needs nothing beyond the standard library.

## Install

From a clone:

```bash
pip install .
```

## Use

```bash
modelmap map --open             # the map, opened in your browser
modelmap scan                   # totals by store
modelmap list                   # every model, largest first
modelmap providers              # where each store is looked for
modelmap rm ollama:qwen3:4b     # remove a model, after showing what goes and asking
```

On the map, click a tile to open the model; the square is redrawn as its
files, with each file's path and the `modelmap rm` command beside it.
Escape goes back up.

`modelmap rm` removes Ollama models with `ollama rm`, Hugging Face cache
repositories with `hf cache rm`, and moves anything else to the Trash. It
shows every file or command first, asks, and refuses caches and files
that another model also uses.

To map a machine you cannot reach, ask its owner for an inventory:

```bash
modelmap scan --json --redact > inventory.json     # on their machine
modelmap map --from-json inventory.json --open     # on yours
```

## Documentation

- [Guide](docs/guide.md): what is scanned, how sizes are counted, the map,
  removing models, and the limits.
- [Command reference](docs/cli.md): every command and option.
- [Cheat sheet](docs/cheatsheet.md): one page.

## Development

```bash
poetry install --with dev
.venv/bin/pytest
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
