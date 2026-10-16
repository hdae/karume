# Karume

A general-purpose NN inference stack running on WebGPU (browsers and Node — Deno as a secondary
runtime — pure TypeScript + WGSL;
`@karume/runtime` itself has zero external dependencies, and the other packages depend only on
packages built from Web-standard APIs). Composed of three JSR packages — `@karume/runtime`
(IR execution) / `@karume/hub` (model resolution, fetch, and caching) / `@karume/models`
(pipelines and tokenizer) — plus the PyPI package `karume` that lowers PyTorch models to IR and
writes them as Karume containers (`krm`) (the installed CLI ships `dist` / `migrate` / `verify` —
see `karume --help`; per-model exports are recipes run from the repository work tree as
`python -m <family>.export`, see [tools/export-recipes/](tools/export-recipes/README.md)).

Status: **pre-1.0** (this page is a stub; the current design's source of truth is
[docs/decisions/](docs/decisions/))

License: MIT ([LICENSE](LICENSE))

**Carve-out**: [`tools/export-recipes/`](tools/export-recipes/README.md) contains upstream-derived
model code (patch layers, export scripts, reference pipelines). That directory is **not** covered by
the project's MIT license — each family there carries its own `THIRD_PARTY_NOTICES.md` recording
what the recipe derives from, and the upstream terms recorded there govern it.

Development runs on Node 24 with [Dawn](https://www.npmjs.com/package/webgpu) as the WebGPU
implementation (pnpm workspace, Vite+ toolchain; see [docs/decisions/0127](docs/decisions/0127-node-dawn-toolchain.md)).
A Node application passes its WebGPU implementation to `acquireGpu({ gpu })`; browsers use
`navigator.gpu`. The command-line examples of 0.14.0 (Gemma 4 QAT, MiniCPM5, Qwen3, Anima, …)
are being rebuilt as browser applications; until then they remain on the
[v0.14.0 tag](https://github.com/hdae/karume/tree/v0.14.0/examples).
