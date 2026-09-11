# Karume

A general-purpose NN inference stack running on WebGPU (Deno + browser, pure TypeScript + WGSL;
`@karume/runtime` itself has zero external dependencies, and the other packages depend only on
packages built from Web-standard APIs). Composed of three JSR packages — `@karume/runtime`
(IR execution) / `@karume/hub` (model resolution, fetch, and caching) / `@karume/models`
(pipelines and tokenizer) — plus the PyPI package `karume` that lowers PyTorch models to IR
(the installed CLI ships `dist` / `repack` / `verify` — see `karume --help`; the `export*`
commands run from the repository work tree only).

Status: **pre-1.0** (this page is a stub; the current design's source of truth is
[docs/decisions/](docs/decisions/))

License: MIT ([LICENSE](LICENSE))

**Carve-out**: [`tools/export-recipes/`](tools/export-recipes/README.md) contains upstream-derived
model code (patch layers, export scripts, reference pipelines). That directory is **not** covered by
the project's MIT license — each family there carries its own `THIRD_PARTY_NOTICES.md` recording
what the recipe derives from, and the upstream terms recorded there govern it.

Experimental fixed mobile QAT: [Gemma 4 QAT E2B / E4B](examples/gemma4-qat/README.md),
with a separate `gemma4-qat` pipeline and local chat CLI.

Experimental local LLM examples: [MiniCPM5-2B](examples/minicpm5/README.md) and
[Qwen3-0.6B](examples/qwen3/README.md). These CLIs use existing converted assets in `outputs/series/`;
see their READMEs for commands and the current context limit.
