"""Work out what a set of model files is: format, quantization, size class.

Used by every provider once it knows which files make up one model.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from modelmap.headers import read_gguf, read_safetensors

# Extensions that mark a weights file when no header can confirm it.
PICKLE_EXTENSIONS = frozenset({".bin", ".pt", ".pth", ".ckpt"})
_OTHER_FORMATS = {
    ".onnx": "onnx",
    ".mlmodel": "coreml",
    ".mlpackage": "coreml",
    ".tflite": "tflite",
}
# model-00001-of-00004.safetensors, name-00001-of-00003.gguf
_SHARD = re.compile(r"^(?P<base>.*?)-\d{5}-of-\d{5}$")

# First match wins, so more specific names come first.
_FAMILIES = (
    ("qwen", "qwen"),
    ("deepseek", "deepseek"),
    ("codellama", "llama"),
    ("llama", "llama"),
    ("mixtral", "mistral"),
    ("mistral", "mistral"),
    ("gemma", "gemma"),
    ("phi-", "phi"),
    ("phi3", "phi"),
    ("phi4", "phi"),
    ("falcon", "falcon"),
    ("granite", "granite"),
    ("smollm", "smollm"),
    ("lfm", "lfm"),
    ("whisper", "whisper"),
    ("parakeet", "parakeet"),
    ("stable-diffusion", "stable-diffusion"),
    ("sdxl", "stable-diffusion"),
    ("flux", "flux"),
    ("nomic", "nomic"),
    ("bge", "bge"),
    ("minilm", "minilm"),
    ("mxbai", "mxbai"),
    ("jina", "jina"),
    ("e5-", "e5"),
    ("t5", "t5"),
    ("clip", "clip"),
    ("bert", "bert"),
)


@dataclass
class Description:
    """What the weight files of one model say about it.

    :param format: ``gguf``, ``safetensors``, ``mlx``, ``pytorch`` and so on;
        ``unknown`` when no weight file was recognized.
    :param quant: Quantization label, or None.
    :param params: Parameter count, or None when unknown or ambiguous.
    :param params_exact: False when ``params`` is derived, not read.
    :param architecture: ``general.architecture`` of a GGUF file, if any.
    :param vocab_only: True for a GGUF tokenizer stub with no tensors.
    :param issues: Problems found in the headers.
    """

    format: str = "unknown"
    quant: str | None = None
    params: int | None = None
    params_exact: bool = True
    architecture: str | None = None
    vocab_only: bool = False
    issues: list[str] = field(default_factory=list)


def guess_family(name: str, architecture: str | None = None) -> str | None:
    """Guess a model family from its name, else from its GGUF architecture.

    :param name: Model name such as ``mlx-community/Qwen3-8B-4bit``.
    :param architecture: ``general.architecture`` from a GGUF header.
    :return: A short family name such as ``qwen``, or None.
    """
    for text in (name.lower(), (architecture or "").lower()):
        for needle, family in _FAMILIES:
            if needle in text:
                return family
    return architecture or None


def shard_base(filename: str) -> str:
    """File name without its extension and any ``-00001-of-00004`` suffix."""
    stem = os.path.splitext(filename)[0]
    match = _SHARD.match(stem)
    return match.group("base") if match else stem


def is_projector(filename: str) -> bool:
    """True for a multimodal projector file, which cannot run alone."""
    return "mmproj" in filename.lower()


def describe_gguf(paths: Iterable[str]) -> Description:
    """Describe one GGUF model stored as one file or as several shards.

    :param paths: The file, or every shard of one split model.
    """
    out = Description(format="gguf", vocab_only=True)
    params: int | None = 0
    for path in sorted(paths):
        info = read_gguf(path)
        out.vocab_only = out.vocab_only and info.is_vocab_only and not info.issues
        out.issues += [f"{os.path.basename(path)}: {issue}" for issue in info.issues]
        out.quant = out.quant or info.quant
        out.architecture = out.architecture or info.architecture
        params = None if params is None or info.param_count is None else params + info.param_count
    out.params = params
    return out


def read_quantization(directory: str) -> Mapping[str, Any] | None:
    """The ``quantization`` object of a model directory's ``config.json``."""
    try:
        with open(os.path.join(directory, "config.json"), encoding="utf-8") as fh:
            config = json.load(fh)
    except (OSError, ValueError):
        return None
    quantization = config.get("quantization") if isinstance(config, dict) else None
    return quantization if isinstance(quantization, dict) else None


def _quant_label(quantization: Mapping[str, Any] | None) -> str | None:
    if not quantization or not isinstance(quantization.get("bits"), int):
        return None
    default = quantization["bits"]
    layer_bits = {v.get("bits") for v in quantization.values() if isinstance(v, dict)}
    mixed = bool(layer_bits - {default, None})
    return f"{default}-bit mixed" if mixed else f"{default}-bit"


def describe_directory(directory: str, files: Iterable[str]) -> Description:
    """Describe a model stored as a directory of files.

    Parameters are counted only when the top-level weight files are plainly
    one model: a single file, or a set of numbered shards. A directory that
    holds several variants or components (a diffusers pipeline, a repo with
    five quantizations) gets no count, because a sum would be wrong.

    :param directory: The model directory, where ``config.json`` lives.
    :param files: Every file of the model, at any depth.
    """
    files = sorted(files)
    by_ext: dict[str, list[str]] = {}
    for path in files:
        by_ext.setdefault(os.path.splitext(path)[1].lower(), []).append(path)
    safetensors = by_ext.get(".safetensors", [])
    ggufs = [p for p in by_ext.get(".gguf", []) if not is_projector(os.path.basename(p))]

    if safetensors:
        return _describe_safetensors(directory, safetensors)
    if ggufs:
        if len({shard_base(os.path.basename(p)) for p in ggufs}) == 1:
            return describe_gguf(ggufs)
        each = [describe_gguf([p]) for p in ggufs]
        quants = sorted({d.quant for d in each if d.quant})
        counts = {d.params for d in each}
        return Description(
            format="gguf",
            quant=", ".join(quants) or None,
            params=counts.pop() if len(counts) == 1 else None,
            architecture=next((d.architecture for d in each if d.architecture), None),
            issues=[issue for d in each for issue in d.issues],
        )
    for ext, name in _OTHER_FORMATS.items():
        if ext in by_ext:
            return Description(format=name)
    if any(ext in by_ext for ext in PICKLE_EXTENSIONS):
        return Description(format="pytorch")
    return Description(issues=["no weight files"])


def _describe_safetensors(directory: str, safetensors: list[str]) -> Description:
    quantization = read_quantization(directory)
    out = Description(format="safetensors", quant=_quant_label(quantization))
    top = [p for p in safetensors if os.path.dirname(p) == directory]
    stems = [os.path.splitext(os.path.basename(p))[0] for p in top]
    # One file, one set of shards, or several sets of shards (a language
    # model plus its vision tower). Differently named whole files are
    # variants of one another, and adding them up would double count.
    one_model = len({shard_base(os.path.basename(p)) for p in top}) == 1 or (
        bool(stems) and all(_SHARD.match(s) for s in stems)
    )
    params: int | None = 0 if one_model else None
    for path in top if one_model else safetensors[:1]:
        info = read_safetensors(path, quantization=quantization)
        out.issues += [f"{os.path.basename(path)}: {issue}" for issue in info.issues]
        if info.packed or info.metadata.get("format") == "mlx":
            out.format = "mlx"
        out.params_exact = out.params_exact and info.params_exact
        params = None if params is None or info.param_count is None else params + info.param_count
    out.params = params
    return out
