# Karume

A general-purpose NN inference stack running on WebGPU (Deno + browser, pure TypeScript + WGSL;
`@karume/runtime` itself has zero external dependencies, and the other packages depend only on
packages built from Web-standard APIs). Composed of three JSR packages — `@karume/runtime`
(IR execution) / `@karume/hub` (model resolution, fetch, and caching) / `@karume/models`
(pipelines and tokenizer) — plus the PyPI package `karume` that lowers PyTorch models to IR
(the installed CLI ships `dist` / `verify`; the `export*` commands run from the repository
work tree only).

Status: **WIP** (this description gets written during release preparation. The current design's
source of truth is [docs/decisions/](docs/decisions/))

License: MIT ([LICENSE](LICENSE))
