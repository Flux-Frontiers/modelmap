# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
