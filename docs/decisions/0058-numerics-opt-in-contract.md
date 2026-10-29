# 0058: General contract for numerics opt-ins (one named option each; the default is the reference path)

- Status: accepted (2026-08-14); partly superseded by [0110](0110-practical-tier-numerics-contract.md)
  (decision 4 ② and ③, 2026-09-26) and by the addendum of 2026-09-13 with 0110 decision 8
  (decisions 5 and 6, marked 2026-10-18); [0128](0128-default-preset-original-weights.md)
  (accepted 2026-10-29) takes over decision 6 and the same-weights bullet of the 2026-09-13
  addendum; decision 8 withdrawn 2026-10-18. Decision 2 stands: [0126](0126-numerics-profiles.md),
  which would amend it, is deferred (2026-10-18)
- Date: 2026-08-14
- Related: [0028](0028-f16-compute-variants.md) (decision 1, "no auto", generalised here) /
  [0074](0074-quant-seat-naming.md) (preset naming) / [0050](0050-irodori-quant-series.md) (first
  storage presets) / [0110](0110-practical-tier-numerics-contract.md) (numerics contract of the
  practical presets) / [0128](0128-default-preset-original-weights.md) (how a distribution chooses
  its default preset) / [perf-ledger](../perf-ledger.md) / [glossary](../glossary.md)

## Context

The runtime already had three execution knobs in `SessionOptions` (`linearCompute`,
`attentionCompute`, `attentionScoreStorage`). This ADR makes their pattern the contract for every
later optimisation that breaks the numeric contract. Presets of a distribution (storage dtype;
naming in ADR 0074) are another axis and out of scope, apart from how the axes cross (decision 7).
Background policy (the maintainer, 2026-08-13; [perf-ledger](../perf-ledger.md) header):
bit-identity checks indicate that the implementation is correct; they are not the goal.

## Decision

1. **One numerics-changing optimisation = one named option in `SessionOptions`.** The type is a
   string-literal union whose reference value (`"f32"` and the like) is the default. Names follow
   `<target><axis>` (as `linearCompute`). Booleans are not used: a second variant for the same
   target would force the axis to be redesigned (`attentionCompute` took `"a8"` after `"f16"`).
2. **The default is the reference path (MUST).** No numerics-changing path is chosen automatically
   from adapter capability, the storage dtype of the assets or the graph shape: a different machine
   or asset must not silently change the output. The sha256 regression checks (PNG / WAV / golden)
   of the reference path stay frozen as the machine check of correctness. (Since 2026-09-26 this
   covers the reference values recorded on a preset with an empty `session` — ADR 0110 decision 7
   and the addendum of 2026-09-26.)
3. **Silent fallback only for shape and limit eligibility** (as linear `k % 4` and attention
   `D % 4` / `N % 4`). When an opt-in is requested but the shape is not eligible, falling back to
   the reference path is allowed, and the fallback MUST appear in the diagnostic pipeline key. An
   unimplemented combination (for example `linearCompute: "f16"` with i8 resident weights) is not a
   fallback; it fails loudly. "The option was silently ignored" and "the shape is not eligible"
   are different things.
4. **Checks for each option (MUST).** ① The existing sha256 regression checks of the reference
   path pass unchanged (proof that the default is frozen). ② An A/B tolerance-band check of the
   opt-in path against the reference path, the band derived from measured drift times a margin,
   with the derivation table in the docstring of the test constant (precedent: the w8 latent band
   of 1e-2, `packages/e2e/tests/e2e_irodori_latent_test.ts`). ③ A census check that the variant
   actually ran when the option is set; the check is red when every dispatch fell back (the same
   defence against "only the speed silently drops" as the fusion-count checks,
   `packages/e2e/tests/e2e_*_fusion_counts_test.ts`). (② and ③ superseded by ADR 0110 decisions
   3–5 on 2026-09-26 and by the addendum of 2026-10-10 — see the addenda of both dates.)
5. **Division of quality control.** Correctness and non-regression are machine-checked by bit
   identity; the quality of opt-in options without bit identity is managed by human review
   (listening, viewing). Numeric metrics (LSD and the like) are advisory: a large numeric drop
   often leaves the output intact (precedent: w8a8 became the default because listening won over
   LSD 5.64 — ADR 0050). (Superseded by the addendum of 2026-09-13 and ADR 0110 decision 8, marked
   2026-10-18: adoption is decided from automatic evaluation and reading the generated output; a
   human quality check is required only when a preset becomes `defaultQuant`, a procedure ADR 0128
   decision 1 inherits.)
6. **Promotion to the default goes with a quality ruling.** Only when an opt-in option is promoted
   to the default is a human quality ruling made, and the sha256 reference digests and the default
   value change in the same commit (splitting them opens a window where the check and the default
   disagree; precedent: w8a8, ADR 0050). (Superseded by the addendum of 2026-09-13, ADR 0110
   decision 8 and ADR 0128 decision 1 (accepted 2026-10-29), marked 2026-10-18: the runtime default is not
   promoted; models choose their default through `defaultQuant`; sha256 reference values are kept
   per preset and per environment key (ADR 0106), so changing `defaultQuant` moves no value and the
   window does not arise.)
7. **Crossing with storage presets.** Execution options are orthogonal to the storage dtype; each
   option declares which combinations it accepts (`linearCompute: "a8"` with i8 resident weights is
   the example — renamed from `"i8a8"` in 0.5.0, ADR 0074 decision 3; an unsupported combination
   fails loudly per decision 3). Quality is ruled on when an execution option or a storage dtype is
   introduced or promoted, not for every combination (combinations multiply; band checks and census
   keep the combinations healthy by machine).
8. Withdrawn 2026-10-18: it named K-5 / K-2 as the first users, which was a plan. K-5 closed
   without an option (row-block execution is bit-identical — [ADR 0060](0060-row-block-attention.md)),
   and K-5b was rejected after a spike (perf-ledger K-5b). The first option under this contract is
   `stateAttentionReduce` (addendum of 2026-09-06).

## Addendum (2026-08-29) — exception to decision 2: automatic selection that claims bit identity needs a machine check

`i8a8Dot` (`dot4I8Packed` or its emulation) was chosen from the adapter's language features,
outside decision 2, on the claim that both variants are bit-identical. Apple M2 refuted the claim
for fused attention only ([known-issues](../known-issues.md); linear shares the `idot` code and
passes). Ruling:

1. **Separate knobs per family**: `linearI8a8Dot` / `attentionI8a8Dot` (`SessionState`,
   `packages/runtime/src/runtime/session-build.ts`). Linear keeps choosing by enumeration (no
   counter-example). The test-only `I8A8_DOT` still forces both.
2. **Attention decides by a run-time canary** (`packages/runtime/src/gpu/attention-dp4a-canary.ts`).
   It runs a fixed input (one full tile plus two K tiles, production geometry) through all six
   variants that can be generated, in one submit, and compares with the known answer of the TS
   reference (`packages/runtime/src/reference/i8a8.ts`). Decision rule v2 (revised on M2 on
   2026-08-29): ① dp4a matches exactly → dp4a; ② the emulation matches exactly → emulation;
   ③ otherwise a sanity band (relative 1e-5) decides: one arm inside → that arm; both inside and
   bit-identical to each other → dp4a (the variant choice does not affect the numbers, so decision
   2 is met by a machine check); both inside but different → the arm with the smaller error;
   ④ both outside → `GpuFeatureError`. A decision without an exact match prints one warning and
   continues. The fallback shows as `:dp4aEmu` in the pipeline key (decision 3). The canary runs
   once per device (memoised Promise), at the first Session built with `attentionCompute: "a8"`.
3. **General rule.** An automatic selection that claims "bit-identical, so outside decision 2" is
   allowed only when the identity is machine-checked on that device (for example a run-time canary).
   Enumeration or a specification claim is not enough.

History: on M2 the canary caught a 1-ULP epilogue difference against the TS reference, not a
variant mismatch (2026-08-29, rule v2). The fixed QK input moved to the f16 grid (2026-08-30,
RC1-1); M2 now decides by an exact dp4a match (2026-09-01; known-issues, Metal section).

## Addendum (2026-09-06) — the first option: `stateAttentionReduce`

`SessionOptions.stateAttentionReduce` (states-shaped attention; `"sequential"` is the reference
path; `"parallel"`, and `"parallel-fused"` since ADR
[0102](0102-state-attention-stats-pv-fusion.md)) carries K-12 (KV-parallel PV reduction) and K-14
(D-parallel QK reduction, M=1 plans only) — ADR [0067](0067-autoregressive-attention-vocabulary.md)
addendum. The option declares only that a numerics-changing path is selected; the implementation
decides by measurement which plans use it.
The code default of `Gemma4Pipeline` (2026-09-03) was removed on 2026-09-26; the preset's manifest
declaration (`session.stateAttentionReduce`) now decides, since a family default in models code
conflicts with ADR 0110 decision 1. `i4` (no declaration) and `fromAssets` run `"sequential"`.

## Addendum (2026-09-13) — the runtime's reference default versus the models' performance default

The maintainer confirmed both goals (a verified path that agrees numerically; a model default as
fast as possible) and delegated adoption decisions. Decisions 2, 5 and 6 apply as follows:

- Runtime and kernel defaults are the reference path. Numerics-changing speed-ups stay individual
  opt-ins.
- Models use the speed-ups in the selected preset's `session`; the caller's explicit options win.
  Models do not pick a parallel form from the model shape; the distribution names a preset with
  speed-ups as `defaultQuant`.
- Keep a preset with the same weights on the reference path (superseded on 2026-10-18 by ADR 0128
  decision 2 (C2) and decision 4, effective when 0128 is accepted: a preset that declares
  activation quantization MUST have a sibling with the same weights and no a8, checked by
  `packages/runtime/tests/distribution_gate_test.ts`).
- Bit or token differences alone are no reason to reject. Speed gain and the absence of large
  quality loss are judged from fixed evaluations, real output and evidence on the target devices;
  unmeasured ranges are stated.
- Where a human quality check is needed, the record says whether it was done. Within the delegated
  scope, decisions rest on automatic evaluation and on reading the output; a human evaluation that
  did not happen is never recorded as done.
- Decision 6's same-commit rule does not apply to model defaults. The frozen reference
  configuration and its digests are kept and named in the checks. A new model default is checked
  separately for: the variant actually applies, it matches the explicit options, and quality and
  speed. Invariants within one execution configuration (such as MTP agreement) still hold.

Applied to the parallel GEMV summation of Gemma E2B (ADR [0098](0098-linear-gemv-parallel.md))
and on 2026-09-15 to the RMS→add and linear→SRQ fusion declarations (ADR 0104).

## Addendum (2026-09-26) — reference path and practical presets; pointers to ADR 0110

The maintainer's ruling of 2026-09-25 separated the numeric paths into the reference path (runtime
defaults, strict oracle, sha256 regression checks) and the practical presets (opt-in paths bundled
by a preset's `session`). The contract lives in ADR 0110; this ADR reads as follows:

- **Decision 2.** "sha256 frozen on the reference path" applies to values recorded on a preset with
  an empty `session`. Values recorded on a preset with opt-in options stay as regression and
  determinism detectors of that preset (0110 decision 7; the class is derived from the preset's
  `session`, not kept in a separate table).
- **Decision 4 ② and ③.** Band derivation and the defence against vacuous checks are 0110 decisions
  3–5 (at E2E level: census, a difference floor against the reference path on the same machine, and
  a collapse ceiling). The census reads `SessionDiagnostics.lastRunPipelines` (derived from the
  recipe of the last run, independent of timing); the table of predicates (numerics-changing
  option × non-reference value → key predicate) lives in
  `packages/runtime/tests/helpers/pipeline-census.ts`.
- **Decision 6, extended.** In-device determinism MUST holds for practical presets too (0110
  decision 6). A host-side adaptive policy that picks paths by wall-clock time is allowed only when
  the identity of those paths is machine-checked on that device (an extension of rule 3 of the
  2026-08-29 addendum).
- **The bulk flag** (a request of 2026-08-13 to "use every opt-in speed-up"). ① Composition is one
  rule for every family: explicit option > preset declaration > runtime default. The runtime has no
  bulk axis such as `numerics: "practical"`; the bundle differs per family and storage dtype, and
  only a manifest preset can express it. ② A caller that pins (`*_SOURCES`) gets a stable meaning
  across versions; a caller that follows main (`fromPretrained` with a string ref) does not —
  whether an operating rule is needed (for example never changing a published preset's content) is
  not decided. ③ Band checks exist per knob (decision 4) and per bundle (0110 decision 5, only for
  distributed bundles), never for every combination (decision 7 stands).

## Addendum (2026-10-10) — decision 3: checking that every attention dispatch of a family took the declared variant

The key of decision 3 shows a fallback only to someone who reads it, and the census of decision 4 ③
("at least one variant") stays green on a partial fallback, where self-attention falls back and
cross-attention keeps the variant (Anima's default preset at resolutions whose sides are not
multiples of 32 px — [limitations](../limitations.md), the section on a8 / s16 fused attention).

- **One pure function decides.** The per-stage route of fused attention (QK / PV i8a8 and the
  storage of S) is decided by `attentionRoute` (`packages/runtime/src/runtime/attention-route.ts`);
  recipe derivation and the acceptance-table tests go through the same function. It is not public
  (ADR 0008).
- **Family checks test completeness as a ratio.** For a Session that declares a8 / s16, the census
  checks per fused-attention family that all dispatches equal the variant dispatches
  (`assertAttentionSeatsComplete`, `packages/runtime/tests/helpers/pipeline-census.ts`). Row-block
  counts multiply both sides alike, so the check does not depend on the device. It is kept apart
  from the "at least one" check (`assertSeatsApplied`), which runtime tests that deliberately hit
  fallback shapes call.
- **Host tests pin the acceptance table.** Every accepted shape of a family goes through
  `attentionRoute`; per self / cross attention the test asserts the declared route (Wan2.1 /
  Wan2.2: a new fallback shape turns it red) or the fallback subset fixed by a predicate (Anima: a
  changed acceptance set breaks the shape count).

Decision 3 itself does not change; handling the remainder on the declared route is a later item
of the backlog.
