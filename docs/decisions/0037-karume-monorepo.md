# 0037: Karume monorepo — brand, package layout and distribution approach

- Status: accepted (2026-08-05); partly superseded by [0065](0065-exporter-core-recipe-split.md)
  (§2.5, the scope of PyPI `karume`, 2026-08-15), by [0041](0041-manifest-v2.md) §2 and
  [0108](0108-container-format.md) (§3, the distribution form, 2026-08-08 and 2026-09-24), and by
  [0127](0127-node-dawn-toolchain.md) decisions 1 and 3 (§1, the workspace, 2026-10-16)
- Date: 2026-08-05
- Related: ADR [0001](0001-scope-and-non-goals.md) (this ADR changes one point of its flat layout) /
  ADR [0003](0003-ir-v1.md) (IR container conventions) / ADR [0008](0008-public-api.md) (public API
  surface)

## Context

To publish the project, the runtime, the exporter and the pipeline wiring are organised as several
packages under one umbrella brand. This ADR decides the name, the package split and how model assets
are delivered.

## Decision

### 1. The umbrella brand is **Karume**

- JSR: `@karume/runtime` (IR execution) / `@karume/hub` (model resolve, download, cache and variant
  resolution) / `@karume/models` (pipelines and tokenizers).
- PyPI: `karume` (the exporter; its scope is narrowed by §2.5).
- Repository: the `hdae/karume` monorepo (Deno workspace). This changes only one point of ADR 0001
  ("flat layout, no workspace mechanism"). (Superseded by ADR 0127 decisions 1 and 3 on 2026-10-16:
  development runs on a pnpm workspace over `packages/*` and `tools/*` (`pnpm-workspace.yaml`); the
  Deno workspace in the root `deno.json` remains only for publishing the three packages to JSR.)

The name was chosen on 2026-08-05 because `karume` was free on JSR, PyPI and npm and carries no
technical claim. `Zarame` is reserved for a possible future project with TypeScript-native model
definitions (see §2).

### 2. The approach is **converted artefacts**

The Python exporter produces the assets, `@karume/hub` loads them and the `@karume/models` pipelines
run them. **Model definitions are managed on the Python side**; the TypeScript side only runs the
exported graphs.

- The purpose is to shrink the wiring: plain host wiring grew past 1,100 lines (measured). Reducing
  it to `AnimaPipeline.fromPretrained(...)` plus `generate(...)` on one screen is why hub and models
  exist.
- TypeScript-native model definitions that run the original Hugging Face safetensors directly are
  out of scope for this project (a possible separate project, `Zarame`).

### 2.5 Scope of PyPI `karume` — revised by ADR 0065

**Revised (2026-08-15, ADR [0065](0065-exporter-core-recipe-split.md)):** PyPI `karume` is a generic
exporter and IR tooling package. Model-specific recipes (patches, reference scripts, dist recipes,
model cards) live outside the wheel in `tools/export-recipes/`.

### 3. Distribution form: one Hugging Face repository per model, several safetensors

(Superseded by ADR [0041](0041-manifest-v2.md) §2 on 2026-08-08 — a family repository holds several
models, and `models` plus `defaultModel` in the manifest are the source of truth — and by ADR
[0108](0108-container-format.md) on 2026-09-24 — the distribution uses the dedicated container
`krm`, so "one graph = one safetensors" and "no `.krm` extension" no longer hold.) The part still in
force:

- The exporter output directory can be uploaded as a Hugging Face repository as is, and it carries a
  model card `README.md` (YAML front matter: `library_name` / `base_model` / `license` / `tags` /
  `pipeline_tag`), written today by `karume dist`.

### 4. `@karume/models` offers a barrel and subpath exports; zero side effects becomes an invariant

- The package provides both the `mod.ts` barrel and one subpath export per family (the `exports` map
  of `packages/models/package.json`, mirrored in `packages/models/deno.json` for JSR). A consumer can
  import one family or import from the barrel and rely on tree-shaking.
- As the condition for that, **every module MUST be free of side effects** (no top-level
  registration, no global mutable state, no execution at import). This is promoted to a cross-cutting
  invariant in `CLAUDE.md`. If it breaks, tree-shaking through the barrel fails silently.
- Not verified: whether the `package.json` that JSR's npm compatibility layer generates carries
  `sideEffects: false`. Without it, ESM side-effect analysis still shakes most code, but the flag is
  the reliable form. As of 2026-10-18 this is still not measured, and no `packages/*/package.json`
  declares `sideEffects`.

## Consequences

- Large assets under `models/` are not tracked by git (the exporter regenerates them).

## Addendum (2026-08-05) — bootstrap

- The packages were brought up in stages (hub, then the Anima pipeline, then the exporter CLI, then
  publishing); Anima came first and SBV2 later for model-licence reasons. Until the model end-to-end
  tests landed, test coverage was limited to the runtime core.

## Addendum (2026-09-22) — §3 replaced by ADR 0108

- ADR 0108 replaces the "one graph = one safetensors" and "no `.krm`" points of §3 with the dedicated
  `krm` / `krg` container; it took effect on 2026-09-24 when step 3d (`9e905d9f`) removed the reader
  of the old form. Only `karume migrate` still reads the old safetensors form.

## Addendum (2026-10-28) — §4: the barrel holds one namespace per family (ADR 0136)

- §4 still holds — a consumer imports one family through its subpath or imports from the barrel —
  but the barrel's shape changed in 0.15.0 (ADR [0136](0136-models-public-naming.md)): it exports
  the shared surface flat (option types, `ModelInputError`, `KARUME_SOURCES`, the PNG / WAV / image
  helpers) and each family as `export * as <namespace> from "./<subpath>.ts"`, and the subpaths
  export bare names (`sbv2.Pipeline` through the barrel, `Pipeline` from `@karume/models/sbv2`).
  Tree-shaking through a namespace was measured within 7 B of the flat barrel (ADR 0136). The
  `sideEffects: false` point above is still not verified; the cost it would remove (every family's
  source table in a barrel import) is filed in the backlog.

## Addendum (2026-11-04) — §4: no `sideEffects` flag through JSR; pure annotations instead

- The "not verified" point of §4 is settled: **JSR's npm compatibility layer does not carry
  `sideEffects`**. The `package.json` of the published 0.16.0 tarball
  (`https://npm.jsr.io/~/11/@jsr/karume__models/0.16.0.tgz`, `_jsr_revision` 11) holds exactly
  `name`, `version`, `homepage`, `type`, `dependencies`, `exports` and `_jsr_revision`; JSR's
  generator declares no other field (`NpmPackageJson` in `api/src/npm/types.rs` of `jsr-io/jsr`,
  built in `api/src/npm/tarball.rs`, read on 2026-11-04), and its documentation
  (`https://jsr.io/docs/npm-compatibility`) only says that `exports` is taken from `jsr.json` /
  `deno.json`. With no slot for the field in the generated file, the flag in
  `packages/models/package.json` would reach workspace consumers only.
- Instead, the invariant of §4 is made visible to a bundler in the source: a module-scope expression
  a bundler cannot prove free of side effects is a `/* @__PURE__ */` call — the frozen tables keep
  `Object.freeze` with the annotation, an array spread became `concat`, and the object spread of
  `KARUME_SOURCES` sits in an annotated arrow call. This works for every consumer (the npm layer,
  JSR imports through a bundler, the workspace) without a package flag. JSR's emit keeps comments
  (`remove_comments: false` in `api/src/npm/emit.rs`), and Deno's emit of the same files was seen to
  keep the annotations.
- Measured with Rolldown 1.2.13 (Vite+ 1.2.0), minified, `@karume/*` external: `encodeWav` from the
  barrel went from 3,587 B (1,769 B gzip) to 1,152 B (638 B), against 1,108 B (611 B) from its
  module — the 44 B left are the two bare imports of the external runtime and hub; a bare barrel
  import went from 2,469 B to those 44 B; a family through its namespace is within 6 B of its
  subpath. The check is `packages/models/tests/models_barrel_tree_shaking_test.ts` (a bare barrel
  import bundles to nothing; `encodeWav` costs the same through the barrel).
- Not covered: esbuild 0.25.12, whose tree-shaking is more conservative, still keeps about 3.9 KB of
  a bare barrel import (unused derived constants); and a bare import of the `@karume/runtime` barrel
  keeps about 160 KB under Rolldown, a separate matter of that package.
- `KARUME_SOURCES` now spells its type from the family tables: JSR's generated declaration of the
  bare spread was `{}` in 0.16.0.
