# 0008: Public API surface and SemVer

- Status: accepted (2026-08-01); partly superseded by [0009](0009-dtype-i32-bool.md) (2026-08-02:
  the public `Tensor` is a dtype-discriminated union — f32 = `Float32Array` / i32 = `Int32Array` /
  bool = `Uint32Array`) and by the registry decision of [0127](0127-node-dawn-toolchain.md) (the
  publish dry-run in CI was suspended, recorded 2026-10-18; restored 2026-10-28).
- Date: 2026-08-01
- Related: the prototype recon §9-1 (about 180 internal symbols re-exported, drift undetected) /
  [0129](0129-test-layout-and-lanes.md) decision 3 (the public-surface gate stays mandatory).

## Decision

- The public surface is **only the thin, deliberately designed surface in `packages/runtime/mod.ts`**.
  Internal modules MUST NOT be re-exported as they are. Exports are limited to the types and
  functions that appear in a user story (guideline: open a model / run it / release resources /
  query capabilities / get diagnostics).
- **Integration tests written only through the public surface are permanent**, so drift or breakage
  of the surface is detected mechanically (kept apart from tests that import internals —
  `packages/runtime/tests/e2e_public_api_test.ts`).
- A sub-surface, when needed, is added as an entry of the package's `exports` map (for example
  `./tune`); do not add them lightly. Today that map is each package's `package.json` `exports`
  (`packages/runtime/package.json:12`); the JSR configuration is each package's `deno.json` (ADR 0127
  addendum of 2026-10-28).
- Keep SemVer. A change to the public surface is made after its export diff is stated in review.
  `deno publish --dry-run` runs in CI so the packages stay publishable to JSR (superseded by the
  registry decision of ADR 0127, Open points, recorded 2026-10-18; restored 2026-10-28 as the CI job
  `publish-dry-run` — ADR 0127 addendum of 2026-10-28).
- Public APIs carry explicit return types (avoids JSR slow types; the global TypeScript convention).

## Consequences

- When porting from the prototype, "what to publish" is a selection criterion.
- Breaking changes are allowed during v0.x, but the CHANGELOG states them.

## Addendum (2026-08-05) — `parseSafetensors` on the public surface

Added `parseSafetensors` (and its types) so that models reads non-container payloads with the same
strict checks, removing a second safetensors implementation in models. Current state: 2026-09-25.

## Addendum (2026-09-03) — tools may import runtime internals; tests stay in their layer

- **Repository tools (`tools/`) MAY import `packages/runtime/src/` directly** (for example
  `tools/verify-diff/artifacts.ts`). The thin surface exists so that the published surface does not
  drift from the implementation; tools are neither published nor consumers. An internal API MUST NOT
  be exposed on the public surface for the sake of a tool: it grows the surface without a user story.
- **A test lives in the layer it verifies**: tests of a tool live under `tools/<tool>/`, and
  `packages/*/tests/` MUST NOT import from `tools/` (a reversed layer dependency).

## Addendum (2026-09-05) — tools may reach into runtime test helpers

The 2026-09-03 rule also covers `packages/runtime/tests/helpers/`: a tool may use a helper whose rule
depends on file-system APIs (so it cannot move into `src/`) instead of a third implementation. The
tools named then (`tools/_shared`) were removed by ADR 0127 decision 5.

## Addendum (2026-09-07) — `parseSafetensorsHeader` and the hub range reader

Added `parseSafetensorsHeader` / `safetensorsHeaderLength` / `SafetensorsHeader` for the PLE row
reads, and in hub `openAsset`, `AssetRangeReader` and `DirectoryAdapter.readFileRange?`. The three
runtime names were removed from the surface in 0.13.0 (2026-09-25).

## Addendum (2026-09-20) — a public-surface snapshot gate in every package

Each package has a permanent snapshot gate (`packages/<pkg>/tests/public_surface_test.ts`, fixture
`tests/fixtures/public-surface.json`) beside the named gate of
`packages/models/tests/models_barrel_surface_test.ts`: **the named gate declares intent**, **the
snapshot detects change**. Entries come from each `package.json` `exports`; names and kinds, type
exports included, are extracted with TypeScript 7's API (`packages/runtime/tests/helpers/public-surface.ts`,
ADR 0129 decision 3); it runs in the core lane. To update, run it with `KARUME_SURFACE=write` and
state the diff in the CHANGELOG.

## Addendum (2026-09-25) — the distribution form behind three addenda was retired

The container format (ADR 0108 / [0109](0109-manifest-v5-container.md)) retired the distribution form
behind the addenda of 2026-08-05, 09-05 (see above) and 09-07.

- 2026-08-05: **the decision stands; the reason changed.** `parseSafetensors` / `SafetensorsError`
  now read safetensors payloads delivered inside a container: the sbv2 style tables
  (`packages/models/src/sbv2/style.ts`), the vowel detector (`packages/models/src/vowel-detector/pipeline.ts`)
  and the rope base table (`packages/models/src/anima/rope-base.ts`). `tensorBytes` (only tests used
  it) was removed in 0.13.0 — breaking, as it had been released.
- 2026-09-07: **the reason to publish is gone** (PLE moved into the `model` container). The three
  unreleased names were removed in 0.13.0; `parseSafetensors` keeps the partial application inside.

## Addendum (2026-10-01) — runtime's second surface `./tune`

`@karume/runtime/tune` (ADR [0117](0117-app-geometry-tuning.md)) serves the user story "an
application sweeps geometries on its device, builds a table, saves it and injects it into the next
`acquireGpu`". It exports only the high-level entry points (sweep, generation, table JSON, matching,
case-set version); the sweep's codegen vocabulary stays internal (the 2026-09-03 rule). `mod.ts`
gains no values; the snapshot gate picks up `./tune` automatically.
