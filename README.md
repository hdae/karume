# Karume

A general-purpose NN inference stack running on WebGPU (Deno + browser, pure TypeScript + WGSL,
zero runtime dependencies). Composed of three JSR packages — `@karume/runtime` (IR execution) /
`@karume/hub` (model resolution, fetch, and caching) / `@karume/models` (pipelines and tokenizer)
— plus the PyPI package `karume` (exporter CLI) that lowers PyTorch models to IR.

Status: **WIP** (this description gets written during release preparation. The current design's
source of truth is [docs/decisions/](docs/decisions/))

License: MIT ([LICENSE](LICENSE))
