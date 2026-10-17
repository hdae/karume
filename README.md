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

## Presets and the default

A distribution's `karume.json` lists named presets per model (the `quants` field). A preset fixes
the storage dtype of every weight component (`f32` / `f16` / `bf16` / `i8` / `i4`) and the
execution knobs, for example int8 activations (`a8`). `defaultQuant` names the preset used when
`quant` is omitted.

The default is chosen per model by the maintainer from measurements of speed, memory, download size
and output quality on the reference GPUs, and the choice is recorded in the decision record
([docs/decisions/0128](docs/decisions/0128-default-preset-original-weights.md)). It is a practical
preset and may quantize weights and activations. A preset holding the upstream checkpoint's own
values (`f32`, `bf16`, or `f16` when the upstream is `f16`; computed in `f32`) is shipped where
exporting it is practical, as best effort, and is listed in the model card like every other preset.
The model card's Quants table shows each preset's storage per component and its compute knobs, and
marks the default; pass `quant` to pin a preset.

Presets with int8 activations get their speed from the WGSL `packed_4x8_integer_dot_product`
feature (dp4a). On a GPU without it (Apple GPUs in Chrome, for example) the integer dot products
are emulated: same numbers, no speed-up. The runtime then warns once per device. Every such preset
has a sibling with the same weights and no activation quantization, so switching costs no extra
download.

Development runs on Node 24 with [Dawn](https://www.npmjs.com/package/webgpu) as the WebGPU
implementation (pnpm workspace, Vite+ toolchain; see [docs/decisions/0127](docs/decisions/0127-node-dawn-toolchain.md)).
A Node application passes its WebGPU implementation to `acquireGpu({ gpu })`; browsers use
`navigator.gpu`. The command-line examples of 0.14.0 (Gemma 4 QAT, MiniCPM5, Qwen3, Anima, …)
are being rebuilt as browser applications; until then they remain on the
[v0.14.0 tag](https://github.com/hdae/karume/tree/v0.14.0/examples).
