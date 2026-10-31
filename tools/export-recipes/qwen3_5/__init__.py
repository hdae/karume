"""The export recipes of Qwen3.5 (repository only, never published — ADR 0065 decision 2).

What lives here is what the general exporter core cannot carry, for the Qwen3.5 family as a
whole (not for one consumer): the first consumer is MOSS-Image, whose text encoder is the text
tower of Qwen3.5-9B. The text tower's export is {@link qwen3_5.export} (the checkpoint it reads:
{@link qwen3_5.checkpoint}; the recipe-side rewrites: {@link qwen3_5.patch}).

The dependency direction is **recipe → core only** (`tools/exporter/tests/
test_architecture_boundary.py` checks it).

MUST: **do not re-export** (no `from qwen3_5 import X`): the scripts import transformers, and a
package-level import would pull it into every reader of a submodule. Callers import the
submodules by name.

Scripts start from the export-recipes root, with transformers pinned for the run (no dependency
group of its own, like `minicpm5` / `embeddinggemma`):

    uv run --with 'transformers==5.14.1' --inexact python -m qwen3_5.export --help
"""
