"""One provider per tool that stores models, plus the generic tree scanner.

A provider has a ``NAME``, ``roots(ctx)`` and ``scan(root, ctx)``.
:data:`PROVIDERS` is the order they run in. The order matters: a file
reachable from two stores is charged to the first provider that claims it.
"""

from modelmap.providers import hf, ollama, omlx
from modelmap.providers.stores import gpt4all, jan, llamacpp, lmstudio, whisper

PROVIDERS = (hf, ollama, omlx, lmstudio, llamacpp, gpt4all, jan, whisper)
