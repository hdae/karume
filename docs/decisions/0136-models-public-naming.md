# 0136: Public naming of `@karume/models` — bare names on the subpaths, one namespace per family on the barrel

- Status: accepted (2026-10-28, the maintainer's ruling 28 — option A, implemented as A-decl; the
  open points ruled the same day as recommended). Ships in 0.15.0.
- Date: 2026-10-28
- Scope: the exported names of the eleven subpath entries of `@karume/models` and the shape of its
  barrel (`mod.ts`); the declarations behind them in `packages/models/src`; the runtime `name` of
  the renamed error class and the family label in diagnostic messages.
- Related: [0037](0037-karume-monorepo.md) §4 (barrel + subpath exports; refined here) /
  [0079](0079-sbv2-two-layer-input.md) decision 2 (`toSbv2Utterance`, renamed here) /
  [0008](0008-public-api.md) (thin public surface) /
  [0092](0092-distribution-repos-and-sources.md) (per-family source tables and `KARUME_SOURCES`) /
  [0083](0083-generation-api-surface.md) (the generation vocabulary that stays unprefixed).

## Context

Up to 0.14.0 the barrel re-exported every family's names flat, so every family symbol carried a
family prefix: 118 of the 148 distinct names the subpaths export (`Sbv2Pipeline`, `ANIMA_SOURCES`,
`gemma4ChatPrompt`, `Umt5Encoder`, …). Without the prefix the families collide in one scope —
`Pipeline` names ten different classes, `Assets` ten types and `FromPretrainedOptions` eleven,
`SOURCES` nine tables. The prefix also misreads: SBV2's "2" reads as the x2y idiom, which already
forced `sbv2Utterance` → `toSbv2Utterance` (ADR 0079 decision 2) and left the backlog item "revisit the
barrel's family prefixes" open since 2026-08-25. `mod.ts` (416 lines) repeated every subpath's
export list and JSDoc; only tests held the two in step.

0.15.0 is a breaking release, so the maintainer ruled to settle the naming before it.

Facts measured before the ruling (2026-10-28, on a copy of the package):

- **Tree-shaking of a namespace barrel** (rolldown 1.2.13 — the bundler inside Vite+ 1.2.0 — ESM,
  `@karume/*` external, minified bytes): importing SBV2's pipeline, table and conversion costs 37,823
  B through the subpath, 39,400 B through the flat 0.14 barrel and 39,394 B through a namespace
  barrel (`sbv2.Pipeline`, or `models.sbv2.Pipeline` after `import * as models`); Anima 54,280 /
  55,667 / 55,660 B. No other family's pipeline class survives in any namespace case, including when
  the namespace object escapes (`return sbv2`, `sbv2[key]`, `export { sbv2 }`) — then only that
  family's own unused exports are kept (+234 B for SBV2, +6.5 KB for Gemma's larger surface). The
  ~1.5 KB that every barrel import pays over the subpath is independent of naming: all nine source
  tables (twelve `hdae/karume-*` entries) and a few top-level expressions the bundler cannot prove
  pure; with `treeshake.moduleSideEffects: false` the barrel is within 5 B of the subpath (the open point of ADR 0037 §4,
  "not verified").
- **TypeScript prints declaration names, not export aliases** (tsc 7.0.2): with
  `export { Sbv2Pipeline as Pipeline }`, a type error on `sbv2.Pipeline` reports `Sbv2Pipeline`.
- **deno doc** (2.9.6) shows the export name on a symbol's page but the declaration name inside
  signatures (an aliased `toUtterance(n): Sbv2Utterance` — a name no longer on the surface), and
  renders `export * as sbv2 from "./sbv2.ts"` as `namespace sbv2` with an empty member list; a JSDoc
  block on the `export * as` line is attached to the namespace. `deno publish --dry-run` accepts both
  forms.

## Decision

1. **Each subpath entry exports bare names.** The family prefix is dropped and the rest of the
   identifier kept, its first letter re-cased when a lowercase prefix led a camelCase name:
   `Sbv2GenerateOptions` → `GenerateOptions`, `gemma4ChatPrompt` → `chatPrompt`, `wanFrameToRgba`
   → `frameToRgba`, `ANIMA_SAMPLER_TYPES` → `SAMPLER_TYPES`, `WanTi2vPipeline` → `Ti2vPipeline`.
   Every family's source table is `SOURCES`.
2. **Exceptions (5)**, where dropping the prefix alone gives a poor or inconsistent name:
   `toSbv2Utterance` → `toUtterance`, `parseGemma4PipelineConfig` → `parsePipelineConfig` (the
   prefix sits inside the name), `animaLatents` → `latentStats` (not `latents`, a local-variable
   name used about 165 times in `src/`), and umT5's `Umt5EncoderAssets` / `Umt5EncoderFromPretrainedOptions`
   → `Assets` / `FromPretrainedOptions` (the names the other ten families use). Names that had no
   prefix keep their spelling (the generation vocabulary `Generation*` / `SamplerSpec` / …,
   `GeneratedImage`, `DepthMap`, `AlphaMatte`, `LabSegment`, …) but, being family names, leave the
   flat barrel like the others (`gemma.GenerationCapacityError`, `anima.GeneratedImage`);
   `VowelDetectorResult` → `Result` and `GemmaTokenizer` → `Tokenizer` follow the rule.
3. **The barrel exports the shared surface flat and one namespace per family.** The flat part is
   the twelve symbols that belong to no single family: `FromPretrainedHubOptions`,
   `FromPretrainedComponentOptions`, `ComponentSource`, `ModelInputError`, `KARUME_SOURCES`,
   `encodePng`, `normalizeToNchw`, `resizeRgb8`, `Rgb8Image`, `decodeWav`, `encodeWav`, `DecodedWav`.
   Each family is `export * as <namespace> from "./<subpath>.ts"`, the namespace being the subpath
   key in camelCase: `anima`, `birefnet`, `depthAnything`, `gemma`, `gemma4Qat`, `irodori`, `sbv2`,
   `siglip2`, `umt5`, `vowelDetector`, `wan`. The namespace *is* the subpath module, so the barrel
   cannot drift from a subpath; members are documented once, on the subpath. Each `export * as` line
   carries a JSDoc that names the subpath and links its JSR page, since JSR lists the namespace's
   members empty.
4. **The declarations are renamed (A-decl), not aliased at the boundary**, so TypeScript errors, JSR
   signatures and stack traces show the names users import. Where a family's public name would
   clash with Gemma's in one module, the declaration gets its own module: the QAT class is
   `Pipeline` in `src/gemma/qat-pipeline.ts` (extending the shared base class of
   `src/gemma/pipeline.ts`), and its option types and RoPE entry are in `src/gemma/qat.ts`, which
   refers to Gemma's types as `gemma.PipelineOptions` through a namespace import — JSR prints a type
   reference as written, so an import alias would reach the docs. Bare names that already existed
   inside `src/gemma` were renamed out of the way. Internal, unexported names keep their prefixes.
5. **Error names and diagnostics name the family.** A renamed error class keeps a qualified runtime
   `name`: `sbv2.InputError` (class `InputError`) has `name === "sbv2.InputError"`, so a log line or
   an `err.name` branch still says which family failed. `ModelInputError` and
   `GenerationCapacityError` keep their class and runtime names (the latter is reached through the
   barrel as `gemma.GenerationCapacityError`). Diagnostic messages spell the public name the way the
   barrel does (`"sbv2.Pipeline: quant …"`, `"gemma.ChatSession: …"`), and a missing-source hint reads
   `SOURCES["sbv2-jvnv"] (@karume/models/sbv2)`.
6. **Style.** Code that uses one family imports bare names from its subpath
   (`import { Pipeline, SOURCES } from "@karume/models/sbv2"`); code that touches several families
   uses namespaces (`import { anima, sbv2 } from "@karume/models"`, or
   `import * as sbv2 from "@karume/models/sbv2"` — the same module object). Prose that spans
   families writes `<namespace>.<Name>` (`sbv2.toUtterance`, `wan.Ti2vPipeline`).
7. **Gates.** The named barrel gate is an identity check: for every `exports` key the barrel's
   camelCase namespace is the very module object of that subpath, and the barrel's own values are
   exactly the shared values plus those namespaces. The public-surface snapshot's `.` entry holds 23
   symbols (12 shared + 11 `namespace`) instead of 158.

## Rejected

- **Bare names on the subpaths, prefixed names kept flat on the barrel**: every symbol would have two
  public names, `toSbv2Utterance` — the defect the backlog item exists for — would survive on the
  barrel, and `mod.ts` would keep duplicating the JSDoc.
- **A barrel with the shared surface only**: one import path per symbol, but it reverses ADR 0037
  §4 (import one family or the barrel); decision 3 keeps both at no bundle cost.
- **A-alias** (prefixed declarations, aliases at the subpath): a small diff, but by the facts above
  0.15.0 would ship docs and type errors that name types users cannot import.
- **Keeping the prefixes**: the flat barrel cannot hold `toUtterance`, so the misreading stays.

## Consequences

- Breaking for every consumer of a family name (CHANGELOG 0.15.0 **Breaking** holds the rule, the
  exceptions and the old → new table). Distributions, manifests and pins are unchanged.
- The usage snippets of the generated model cards import the bare names from the subpaths. The
  published cards say `jsr:@karume/models` without a version, so they are regenerated and
  re-uploaded at the 0.15.0 release as card-only commits (the `SOURCES` pins do not move).
- On JSR the barrel page lists 12 symbols and 11 namespaces whose members are on the subpath pages
  (assumed to render as `deno doc` 2.9.6 does; not checked on jsr.io before publication).
- A file that imports named members from two subpaths gets `Pipeline` twice; the namespace import of
  decision 6 (or a local `as`) resolves it.
- Tree-shaking is unchanged. The pre-existing cost of a barrel import (the source tables of every
  family) is filed in the backlog; the fix is the `"sideEffects": false` point of ADR 0037 §4 or
  `/* @__PURE__ */` on the retained expressions.
- ADRs and research notes written before 2026-10-28 keep the names of their time.

## Addendum (2026-11-04) — `gemma4Qat` is `gemma4QatMobile`

The subpath `./gemma4-qat` is `./gemma4-qat-mobile` and its namespace `gemma4QatMobile` (decision 3's rule — the
subpath key in camelCase), after the mobile QAT distribution was renamed `karume-gemma4-qat-mobile`
(ADR [0097](0097-gemma4-qat-integration.md) addendum 10). Its diagnostics read `gemma4QatMobile.Pipeline: …`
(decision 5). The subpath no longer exports `SOURCES` until the renamed repository is published, so decision 1's
"every family's source table is `SOURCES`" holds for the families that have one.
