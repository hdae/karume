# 0111: Composition of SessionOptions — one rule for every family: explicit > preset declaration > runtime default

- Status: accepted (2026-09-26 — phase 1 "contract and groundwork", A3; approved by the
  maintainer); decision 7 re-graded 2026-10-18 to SHOULD;
  decision 2 superseded and decision 6 amended by [0147](0147-numerics-and-kernel-adoption.md) decisions 7 and 8
  (2026-11-10): acceptance is derived from the graph; a missing `shader-f16` warns and switches to f32
- Date: 2026-09-26
- Related: [0058](0058-numerics-opt-in-contract.md) (addendum 2026-09-26) / [0104](0104-gemma-fast-quant.md)
  (gemma's rule, generalised here) / [0107](0107-model-input-error.md) / [0110](0110-practical-tier-numerics-contract.md)
  / [0038](0038-manifest-v1.md) §3 (`gpuFeatures`) / [0126](0126-numerics-profiles.md) (deferred
  2026-10-18; its withdrawal of decision 7 did not take effect). Code:
  `packages/models/src/session/options.ts`, `packages/runtime/src/runtime/session-options-check.ts`.

## Context

Only gemma composed a preset's `session` (a bundle of execution options) into `SessionOptions` as
"explicit > preset declaration > runtime default" (ADR 0104); the other families copied the
declaration with no explicit override. Combination rules (`fuseLinearStaticQuantize` needs
`linearGemvReduce: "parallel"`, …) lived in both Session construction and gemma's copy, so one could
be fixed without the other. Under ADR 0110 decision 1 (the preset is the only source of a bundle)
the rule must be one, and families may differ only in which keys they accept.

## Decision

1. **One composition rule for every family** (`resolveSessionOptions(policy, declared, overrides,
   where)`). Per key: an explicit value (anything but `undefined`) wins, else the declaration, else no
   field (= runtime default). An explicit `false` beats a declared `true`. An invalid explicit value
   such as `null` is rejected, not replaced by the declaration (no `??`).
2. **A family holds only an acceptance table** (`FamilySessionPolicy` = an exhaustive boolean mapping
   over `Required<SessionSpec>`), never value defaults (the runtime holds them in one place). A new
   `SessionSpec` key fails every table at type check. Tables on 2026-10-18: gemma =
   `linearGemvReduce`, `stateAttentionReduce` and the three fusions; anima and wan = `linearCompute` /
   `attentionCompute` / `attentionScoreStorage`; irodori = `linearCompute` (its DiT attention uses a
   runtime bool mask outside the fused-attention contract); sbv2 = `linearCompute`; birefnet,
   depth-anything, siglip2, vowel-detector and umT5 = none.
3. **Combination rules are one pure runtime function** (`sessionOptionsViolation`; Session
   construction rejects through it). `models` keeps no copy. Conditions needing GPU capabilities
   (shader-f16 / subgroups) stay in construction, which sees the device. Value checks (`null`,
   spellings outside the accepted set) live in `models`, because the runtime reads with `??`.
4. **Error type by origin** (ADR 0107 decision 2): a key the table rejects, declared by the manifest →
   plain `Error` (asset mismatch); given explicitly → `ModelInputError` (dropping it would mean "set
   but not applied"). A broken combination → `Error` if the declaration alone breaks it,
   `ModelInputError` if the explicit value is needed.
5. **Composition happens at admission** (before any byte of a weight part is fetched); later, a bad
   declaration or explicit value fails only after a GB-scale fetch.
6. **GPU feature requirements derive from the effective setting**: an explicit `"f16"` over a preset
   without shader-f16 would acquire a device without the feature and fail at construction after the
   upload. `sessionGpuFeatures` adds what the effective setting needs and fails by name at admission.
   (Extended to every family by the addendum.) (Amended 2026-11-10 by ADR
   [0147](0147-numerics-and-kernel-adoption.md) decision 8: the derivation stays; on a shared GPU without `shader-f16`
   admission warns and the Session runs f32; a self-acquired device requests only the available features and warns at
   acquisition. A manifest's own `gpuFeatures` is a request: a device without the feature warns and switches toward the
   reference, and on a shared GPU acquired without it the warning gives the re-acquire text.)
7. **The runtime SHOULD NOT carry a bulk axis** (`numerics: "practical"` and the like): the bundle
   differs per family and storage dtype, and only the manifest preset expresses it (ADR 0058 addendum
   2026-09-26). (Re-graded 2026-10-18 to SHOULD: ADR 0126 designed such an axis and was deferred with
   two re-entry conditions, so this is a judgement of current demand, not a principle.)

## Alternatives considered

- **Copy gemma's rule into each family** — rejected: copies miss rule changes (priority, error types)
  per family (the NOTE in `options.ts`).
- **Value defaults in `models`** — rejected: gemma's `models` default `stateAttentionReduce:
  "parallel"` took the `i4` preset off the reference path (ADR 0110 Context).

## Consequences

- Equivalence with the former `toSessionOptions(quant.session)` for every quant of the mirror
  (`packages/e2e/tests/e2e_session_options_mirror_test.ts`); no sha256 reference value moved.
- gemma's combination error messages became the runtime's (8 expected substrings followed; accepted
  sets and error types unchanged).
- The runtime's public surface gained `sessionOptionsViolation` (ADR 0008 — upper layers borrow the
  construction check before fetching).
- GPU-independent checks run first, GPU-capability checks last; only an input with several
  violations may see a different first message.

## Addendum (2026-09-26) — GPU feature requirements derived in every family

Decision 6 became the rule for every family (the maintainer's conditional "a, if routing it through
is cleaner" applied). At admission each family derives `sessionGpuFeatures(quant.gpuFeatures,
effective SessionOptions)` once; the shared-GPU check, the `acquireGpu` request and the post-acquire
check read that value (the MUST in `packages/models/src/session/gpu-features.ts`). Reason: written
per family, a table that starts accepting f16 compute would pass type check without its request.

- **The only behaviour change**: a manifest declaring f16 compute in `session` without
  `gpuFeatures.shaderF16` (then only sbv2 could). It failed at Session construction; now it fails by
  name at admission on a shared GPU and requests the feature on a self-acquired device. No mirror
  distribution has this shape (the mirror test checks "derived requirement = declaration"). (Amended
  2026-11-10 by ADR 0147 decision 8: on a shared GPU it warns and runs f32 instead of failing, and a
  self-acquired device requests the feature only when the adapter has it.)
- **gemma** never read `quant.gpuFeatures` (a silent degradation). `fromPretrained` now derives at
  admission and checks a shared GPU there; construction calls `acquireGpu({ ...required, subgroups })`
  and the post-acquire check. `fromAssets` has no manifest and requests only what the effective
  setting needs. The drafter Session shares the device and requests nothing.
