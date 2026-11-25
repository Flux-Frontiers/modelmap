# Release Notes: v0.1.0

> Released: 2026-11-24

First release. modelmap finds the language-model files on a machine, shows
how much space each one takes, draws them as a map you can open down to the
individual files, and removes the ones you choose.

## What it does

**Finds models wherever the tools keep them.** It reads the stores of
Hugging Face, Ollama, oMLX, LM Studio, llama.cpp, GPT4All, Jan and Whisper,
follows the settings, environment variables and symlinks that move them,
and sweeps your home folder for loose GGUF, safetensors and checkpoint
files. Any other folder, drive or single file can be named on the command
line.

**Counts honestly.** Model files are identified by their headers, which
give parameter counts and quantization without reading weights. A file
reached by several names, such as a Hugging Face blob shared by two
revisions or Ollama weights shared by several tags, is counted once, so
totals match what the disk holds.

**Draws a map.** `modelmap map` writes one page that works offline: a
square divided among everything found, one block per store. Clicking a
model redraws the square as its files, with each file's path, the other
models that share it, and the command that removes it.

**Removes models safely.** `modelmap rm` shows every file or command it
would act on and asks first. Ollama models are removed with `ollama rm`,
Hugging Face cache repositories with `hf cache rm`, and anything else moves
to the Trash. It refuses caches, files that another model also uses, and
model folders that hold anything else.

**Works across machines.** `modelmap scan --json --redact` makes an
inventory without the home folder or machine name in it, and
`modelmap map --from-json` draws it elsewhere.

## Installing

modelmap needs Python 3.10 or later and nothing beyond the standard
library. It is not on PyPI yet; install from a clone with `pip install .`,
or from the wheel attached to this release.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
