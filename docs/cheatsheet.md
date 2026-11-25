# Cheat sheet

| To | Run |
|----|-----|
| See the map in a browser, with Remove buttons | `modelmap serve` |
| Save the map as a file that works offline | `modelmap map --open` |
| See totals by store | `modelmap scan` |
| List every model, largest first | `modelmap list` |
| List by format, or include datasets and caches | `modelmap list --group format`, `modelmap list --all` |
| Scan an external drive too | `modelmap map /Volumes/Drive --open` |
| Scan only one drive | `modelmap list --no-defaults /Volumes/Drive` |
| Scan only Ollama and Hugging Face | `modelmap list --only ollama,hf` |
| Find out why a store was not found | `modelmap providers` |
| Size the map by space on disk | `modelmap map --disk --open` |
| Save a static picture | `modelmap map --svg -o models.svg` |
| See what removing a model would do | `modelmap rm --dry-run ollama:qwen3:4b` |
| Remove a model | `modelmap rm ollama:qwen3:4b` |
| Remove a file by path, without the question | `modelmap rm ~/models/old.gguf --yes` |
| Make an inventory to send to someone | `modelmap scan --json --redact > inventory.json` |
| Map an inventory you were sent | `modelmap map --from-json inventory.json --open` |

On the map: click a tile to open it down to its files; Escape goes back
up. The panel beside the map shows each model's paths and its
`modelmap rm` command; under `modelmap serve` it also has a Remove
button, which shows what will go and asks first.

Names: Hugging Face `hf`, Ollama `ollama`, oMLX `omlx`, LM Studio
`lmstudio`, llama.cpp `llamacpp`, GPT4All `gpt4all`, Jan `jan`, Whisper
`whisper`, everything else `generic`.
