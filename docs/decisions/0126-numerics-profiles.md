# 0126: Numerics profiles — practical by default, reference on request

- Status: deferred (2026-10-18 — not implemented in 0.15.0 by the maintainer's ruling at phase 2 step 2-8:
  the runtime default stays the reference path and the practical bundles reach users through the default
  presets of [0128](0128-default-preset-original-weights.md); reason and re-entry conditions in the addendum)
- Date: 2026-10-16
- Related:
  - [0058](0058-numerics-opt-in-contract.md) — the numerics opt-in contract; this ADR amends its decision 2
    (default = reference path, no automatic selection), decision 6 (promotion to the default) and the
    2026-09-26 addendum that rejected a runtime-wide numerics axis.
  - [0110](0110-practical-tier-numerics-contract.md) — the reference / practical tiers and the contract classes
    E / C / R / Q; this ADR amends decisions 1, 7 and 8 and keeps decisions 2–6 as the correctness contract.
  - [0028](0028-f16-compute-variants.md) — decision 1, the ban on automatic variant selection ("no auto");
    relaxed for the practical profile only.
  - [0111](0111-session-options-composition.md) — the composition rule "explicit > quant declaration > runtime
    default"; this ADR withdraws its decision 7 (no bulk axis in the runtime) and extends decision 1.
  - [0005](0005-verification.md) — verification tiers and the 2026-10-11 upstream-parity policy; this ADR states
    where upstream parity stops being a gate.
  - [0106](0106-device-keyed-references.md) — sha256 reference rows keyed by environment; rows are re-classified
    per profile.
  - [0022](0022-gemm-register-blocking.md) — the ban on run-time autotuning; unchanged, and it bounds what
    "automatic" may mean here.
  - [0008](0008-public-api.md) — the thin public surface; the profile is one new member of `SessionOptions`.
  - [0115](0115-geometry-profiles.md) — geometry profiles; relevant only because the word "profile" is taken.

## Context

### What the frozen default protects

Today every numerics-changing kernel variant reaches a user in exactly one way: as an explicit value of a named
`SessionOptions` knob (ADR 0058 decision 1), usually declared by a quant seat's `session` bundle in the manifest
(ADR 0110 decision 1). The runtime's omitted-value path is the reference path, and it is frozen:

- ADR 0028 decision 1 forbade choosing f16 compute from adapter capabilities ("changing the machine must not
  silently change the output"). ADR 0058 decision 2 generalised this: no numerics-changing path is selected
  from adapter capabilities, asset storage or graph shape, and the sha256 gates are frozen on the reference path.
- ADR 0110 decision 1 named the two tiers: reference = the runtime's omitted values; practical = the opt-in
  bundle a quant seat declares. Decision 8 states that the runtime's omitted values never change through the
  practical tier.
- ADR 0111 decision 7 (and the ADR 0058 addendum of 2026-09-26) rejected a runtime-wide axis such as
  `numerics: "practical"`, on the ground that only the manifest's quant seat can express a bundle.

The code still carries this contract on every knob: each numerics knob in
`packages/runtime/src/runtime/session-types.ts` documents "MUST: default is the reference value", and
`AcquireGpuOptions.shaderF16` in `packages/runtime/src/gpu/acquire.ts` documents that it is never decided
automatically because f16 compute changes numerics.

### Why it became the obstacle

The maintainer's reasoning on 2026-10-15 separated two things that the freeze had bundled together:

- **Upstream parity** — agreement with torch at the model exit (graph-level goldens, model-exit tolerance
  bands). It is evidence that an implementation is correct.
- **Determinism** — on the same machine the same input gives the same bytes (ADR 0106 rows, ADR 0110 decision
  6). It is evidence that nothing broke.

Only the first is to be relaxed. The question was never "upstream parity" as such; it was the freeze of the
runtime default (ADR 0058 decision 2 / ADR 0110 decision 1). The freeze has three costs:

1. **Parity is paid on every runtime change.** Any runtime change is checked against the model-exit goldens
   and bands of every family. The full verify takes 118–134 minutes (the Wan lane alone about an hour); the
   earlier 30 minutes was already judged too long. ADR 0110 Context already showed that model-exit bands
   become vacuous (four recorded cases), so this cost buys little correctness.
2. **Faster variants have no route to the default.** Reordering variants (class R: parallel GEMV, subgroup
   reductions, parallel state attention) whose f64 band equals the reference path's op band still run only
   when a quant seat declares them. Flash-style (online-softmax) attention — reordering in its exact form,
   approximation if bands are widened — has no route at all; the Wan2.2 attention investigation of
   2026-10-11 needs it. Class Q has zero occupants (ADR 0110 decision 9).
3. **A fixed fast default is not portable.** The alternative to automatic selection is to make one fast
   variant the default for everyone. The maintainer's premise (also behind the companion decision that the
   default weight preset is the original weights): Mac WebGPU has no hardware 4×8-bit dot product, so w8a8 is
   not faster there. A default chosen for speed on one adapter would make another adapter slower. The default
   MUST NOT be slower on Mac WebGPU than the reference path; selection from adapter capabilities gives that,
   a single fixed variant does not. *(Addendum 2026-10-18: this premise no longer holds — ADR 0128 decision 5
   keeps the activation-quantized default preset on a device without dp4a and warns once per device, and the
   companion decision "default = original weights" was withdrawn in the same ADR.)*

ADR 0028 was an early decision. The maintainer's judgement (2026-10-15): current GPUs commonly expose the
features involved (shader-f16, subgroups, the dp4a WGSL feature — all green under Node + Dawn in the
2026-10-15 spike), and a variant without quality loss may be selected from the adapter's capabilities.
ADR 0005's addendum of 2026-10-11 already lowered upstream parity to "numbers or their intent"; this ADR
moves the remaining piece, the default itself.

## Decision

1. **The runtime has two numerics profiles, and the practical profile is the default.** `SessionOptions`
   gains exactly one profile knob with the values "practical" and "reference"; when it is omitted the
   Session runs the practical profile.
   - The **reference profile** is today's frozen default path: every numerics knob at its reference value
     (`"f32"` compute and score storage, `"sequential"` / `"workgroup"` reductions, fusions off) and no
     automatic selection (decision 3).
   - The **practical profile** is the reference path plus the automatic selection of decision 2.
   - The individual knobs stay. A knob that is set explicitly overrides the profile for that knob only, in
     both profiles: `linearCompute: "f32"` under the practical profile pins linear to f32; `stateAttentionReduce:
     "parallel"` under the reference profile still runs the parallel reduction.
   - The profile is fixed at Session construction, like every `SessionOptions` member (ADR 0110 decision 1,
     last bullet, is kept).
   - The knob's name is not fixed (Open points).

   Rationale: one knob is the smallest surface that lets a user who needs the 0.14 numerics ask for them,
   while the default follows the adapter. The individual knobs remain the precise tool for A/B work and for
   the quant seats that declare them.

2. **Under the practical profile the runtime MAY select variants of classes E, C and R automatically, from
   the tensor shapes and the adapter's capabilities, on two conditions.**
   - **Observable** (MUST): what was selected is named by the pipeline key and therefore visible in
     `SessionDiagnostics.lastRunPipelines`, which is filled without timing and on the resident paths (ADR 0058
     decision 3 and its 2026-09-26 addendum). A selection that cannot be named in a key is not allowed.
   - **Overridable** (MUST): every variant that can be selected automatically corresponds to a value of an
     existing knob, and setting that knob explicitly forces the result in either direction (the reference
     value, or the variant). No automatic selection exists without a knob.
   - **Q is never automatic.** Approximation variants (wider band than the reference path's op band, for
     example flash-style attention with widened bands) run only when requested explicitly. Activation
     quantization (8-bit attention, 8-bit linear activations) is also explicit-only under this ADR, although
     ADR 0110 classifies its oracle as E: it changes the computation the oracle mirrors. "Explicit" means that
     the runtime never selects it automatically; a default preset may declare it through its `session` bundle
     (ADR 0128 decision 7 — wording adjusted 2026-10-18).
   - **Deterministic** (MUST): the selection is a pure function of the graph's shapes, the device's features
     and limits, and the Session options. It MUST NOT depend on wall-clock timing or run-time measurement —
     ADR 0022's ban on run-time autotuning and ADR 0110 decision 6 stay as they are.
   - A capability the device lacks removes the variant from automatic selection. An explicit request for a
     variant the device cannot run still fails loudly (ADR 0058 decision 3; e.g. `rmsNormReduce:
     "subgroup32"` without `acquireGpu({ subgroups: true })`).

   Rationale: this relaxes ADR 0028 decision 1 and ADR 0058 decision 2 only for the practical profile. The
   two conditions are the maintainer's terms for the relaxation (2026-10-15): the user can see what ran and can
   change it. Restricting automatic choice to E / C / R keeps the default inside the reference path's op band
   or exactly equal to an oracle.

3. **The reference profile has no automatic selection.** No numerics-changing variant is chosen from shapes,
   capabilities or assets under it.
   - Selections that are proven bit-identical on the device are not numerics selections and stay allowed in
     both profiles, under the existing rule (ADR 0058 addendum 2026-08-29, general rule 3): the dp4a /
     emulation choice settled by the attention canary, and the row-block count derived from
     `maxStorageBufferBindingSize` (bit-identical by ADR 0060 decision 3).

   Rationale: the reference profile is the stable target for parity work, for sha rows and for users who need
   the same bytes on every machine that runs the same path.

4. **The op-level oracle stays mandatory for both profiles; only upstream parity is relaxed.**
   - Every variant that either profile can run — automatically or explicitly — MUST keep its class gates as
     ADR 0110 decisions 2–4 define them: E and C at atol=0 against the TS mirror or the host-rounded
     composition, R and Q with the four-point kernel gate against the f64 reference.
   - What is relaxed is upstream parity: the torch golden at the model exit and the model-exit tolerance
     bands are not gates for the practical profile. ADR 0110 decision 5 already states that a model-exit band
     is not a numerics contract.

   Rationale: the op oracle is what shows a kernel is correct, independent of any upstream implementation;
   the model-exit comparison is the part that is slow, vacuous for quantized paths and blocking.

5. **Determinism stays.**
   - In-device determinism (ADR 0110 decision 6) is a MUST for both profiles, including automatic selection.
   - The per-environment sha256 rows of ADR 0106 continue. The class of a row (reference row / practical row)
     is derived from the profile together with the quant seat's `session`, instead of from the `session` alone
     (amends ADR 0110 decision 7). The re-classification happens when the default flips (phase 3-1).

   Rationale: the sha rows are the detector for "nothing broke" on a given machine. That check is the half of
   the old contract the maintainer kept.

6. **Composition order** (follows from decision 1 and ADR 0111): explicit caller option > quant seat
   `session` declaration > profile (automatic selection under the practical profile, the reference value under
   the reference profile). Quant seat declarations already reach the runtime as individual knob values through
   `resolveSessionOptions` (`packages/models/src/session/options.ts`), so they override the profile per key.

### Amendments (take effect in phase 3-1)

| ADR / decision       | Today                                                    | After phase 3-1                                               |
| -------------------- | -------------------------------------------------------- | ------------------------------------------------------------- |
| 0058 decision 2      | Default = reference path; no automatic selection         | Holds for the reference profile only                          |
| 0058 decision 6      | A default change needs a quality ruling and sha update   | Flip and row rewrite land together (3-1c); ruling: open       |
| 0058 add. 2026-09-26 | No runtime-wide `numerics` axis                          | Withdrawn: one profile knob                                   |
| 0110 decision 1      | Reference tier = runtime default                         | Reference profile = opt-in; practical profile = default       |
| 0110 decision 7      | Row class derived from the quant seat's `session`        | Derived from profile + `session`                              |
| 0110 decision 8      | The runtime default never changes via the practical tier | Superseded by decision 1                                      |
| 0028 decision 1      | No automatic f16 compute                                 | Holds for the reference profile; practical may select (dec 2) |
| 0111 decision 7      | No bulk axis in the runtime                              | Withdrawn                                                     |
| 0111 decision 1      | explicit > quant declaration > runtime default           | explicit > quant declaration > profile                        |

ADR 0110 decisions 2–6 (classes, four-point gate, band derivation, E2E A/B gate, determinism) are kept
unchanged and become the acceptance contract for automatic selection.

## Consequences

### For users

- The default numerics change in 0.15.0. Output of a Session that sets no numerics knob may differ bit-wise
  from 0.14.x on the same machine. The CHANGELOG for 0.15.0 lists this under **Breaking**, together with
  the knob that restores the 0.14 path (the reference profile).
- Two adapters may now run different variants for the same graph under the default. Cross-device bit
  identity was never guaranteed (docs/limitations.md, sha256 reference gate section); under the practical
  profile the differences can be larger, but each machine stays deterministic.
- Users can see which variants ran through `diagnostics().lastRunPipelines`. The examples and apps show the
  selected variants (from the pipeline keys) next to their output.
- Quantization is not selected automatically. Whether a quantized seat is used remains the caller's (or the
  manifest's) explicit choice; the default weight preset is decided in [0128](0128-default-preset-original-weights.md).

### For the repository

- `SessionOptions` gains one member (ADR 0008: one knob, no per-family surface). The in-code statements of
  the old contract — the "MUST: default is the reference value" lines on each numerics knob in
  `session-types.ts`, and the "never decided automatically" note on `AcquireGpuOptions.shaderF16` — are
  rewritten in phase 3-1, not before.
- The automatic selection lives in the runtime, next to the existing single selection point in
  `session-build.ts` (the dp4a / canary choice), so that recipe building and the census tests read one
  decision. Families keep their acceptance tables (ADR 0111 decision 2) and gain no value defaults.
- ADRs 0028, 0058, 0110 and 0111 receive a forward pointer to this ADR when they are reorganised and
  translated (phase 2-1); their text changes when the amendments take effect (phase 3-1).

### For verification

- Existing sha256 rows (Deno rows and the Node rows written during the port) and the existing model-exit
  goldens and bands were produced on today's default, which is the reference profile. They belong to the
  reference profile until they are re-measured. Until phase 3-1 rewrites them, the tests that hold them run
  with the reference profile selected explicitly; the graph-level golden test
  (`packages/runtime/tests/e2e_golden_test.ts`, `GOLDEN_TOLERANCE` atol 1e-6 / rtol 1e-5) is a reference-profile
  test.
- The same-machine A/B gate of ADR 0110 decision 5 already compares against the reference tier; under the
  profiles its control side is the reference profile on the same machine.
- The census (ADR 0110 decision 5 ①) gains a second source of variants: for each adapter that runs the
  verify, the automatically selected variants are asserted from `lastRunPipelines`, so a selection that
  silently stops happening turns a test red.
- Phases 1 and 2 of the migration (Node port, cleanup) MUST NOT change behaviour; they are checked against
  the frozen Deno baseline on the reference path. The profile flip is the first intended numerics change.

### Explicitly left open

See Open points. In particular this ADR does not choose the selection rule for any knob; it only states
which classes are eligible and on what conditions.

## Staging

The plan of record fixes only that the profile work is phase 3-1 ("numerics profile → re-classify and write
reference values"), after the Node port (phase 1) and the cleanup (phase 2). The order inside 3-1 below is a
proposal. *(2026-10-18: the plan of record moved this work to phase 2 step 2-8, where it was deferred — none of
3-1a–3-1d is scheduled; see the addendum.)*

1. **Now (phase 0-4):** this ADR is proposed. No code changes; the default stays the reference path.
2. **3-1a — knob without behaviour change:** add the profile knob with the reference profile as the effective
   default; pin the tests that hold sha rows and goldens to the reference profile explicitly. Check: every
   existing sha row and golden is green unchanged.
3. **3-1b — automatic selection:** implement the per-knob selection for E / C / R under the practical profile,
   with census tests per adapter and the ADR 0110 gates for every eligible variant. Check: under the
   reference profile no pipeline key changes; under the practical profile every selected variant is named in
   `lastRunPipelines` and can be forced off by its knob.
4. **3-1c — flip the default:** make the practical profile the omitted value; write practical rows per
   environment (ADR 0106 `write`) and keep the reference rows that remain under the reference profile. Check:
   only the intended rows differ.
5. **3-1d — documents:** amend ADRs 0028 / 0058 / 0110 / 0111 / 0106 as tabled above, rewrite the knob
   comments, add the CHANGELOG **Breaking** entry, and make the examples and apps show the selected variants.

## Open points

- **Knob name.** Candidates: `numericsProfile: "practical" | "reference"` (qualified; avoids confusion with
  `AcquireGpuOptions.geometryProfile`, ADR 0115), `numerics: "practical" | "reference"` (the name ADR 0111
  decision 7 rejected), `profile`. To be fixed at implementation.
- **Per-knob eligibility and selection rules.** Which shapes and capabilities select which variant, with
  measurements per adapter. Class C (f16 compute, f16 score storage) is eligible by class, but f16 compute
  measured 0.92–0.98× on the RTX 3080 Ti (ADR 0028), so eligibility alone does not make it a good default.
- **Activation quantization vs ADR 0110's table.** This ADR treats 8-bit activation variants as explicit-only
  although ADR 0110 classifies them E. Whether ADR 0110's table gains a separate "automatic-eligible" column
  is to be settled when it is translated.
- **Grace clause.** Whether R variants still missing points of the four-point gate (ADR 0110 decision 9:
  parallel GEMV ④, subgroup reductions ②④) may be selected automatically, or must complete their gates first.
- **Device features.** `acquireGpu` does not request shader-f16 or subgroups unless asked, and a Session
  profile cannot influence device creation. Whether the practical profile implies requesting optional
  features by default is undecided.
- **Quant seats and the profile.** Whether a quant seat's manifest `session` may declare the profile itself,
  and whether the family acceptance tables accept the knob.
- **Upstream-parity cadence.** Whether model-exit goldens for the reference profile run on every runtime
  change or only before release; tied to the test-layout ADR (runtime / models split).
- **Which sha cases move.** Which cases become practical rows at 3-1c and whether every family keeps at least
  one reference-profile row.
- **Quality ruling.** Whether the flip needs a human quality ruling per family (ADR 0058 decision 5, ADR 0110
  decision 8) for C and R selections, or whether the op gates suffice.
- **Release date.** 0.15.0 is the target release; no date is set.

## Addendum (2026-10-18) — deferred at phase 2 step 2-8

The maintainer's ruling at step 2-8: this ADR is not implemented in 0.15.0. The runtime's omitted-value path
stays the reference path (ADR 0058 decision 2 and ADR 0110 decisions 1, 7 and 8 are unchanged), the practical
bundles reach users through the default presets of ADR 0128, and the amendments tabled above do not take
effect. The knob, the automatic selection and the row re-classification are not built; 0.15.0 carries no
runtime numerics change and no **Breaking** entry on this account.

### Why

Each of the three costs in Context is handled by a later decision, and the automatic selection has no
occupant today that a preset does not already deliver.

| Cost in Context                                 | State on 2026-10-18 (`wip/0.15`)                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Parity is paid on every runtime change       | Upstream parity at the model exit is opt-in since step 2-1 (`KARUME_UPSTREAM_PARITY=1`, `packages/e2e/tests/helpers/upstream-parity.ts`) and the lanes split runtime from models (ADR 0129). The torch comparisons that still run by default are the runtime's small graph goldens (`packages/runtime/tests/e2e_golden_test.ts`) and EmbeddingGemma's — both cheap.              |
| 2. Faster variants have no route to the default | ADR 0128 keeps the per-model practical preset as `defaultQuant`, and the preset's `session` bundle carries every variant a family uses today (gemma4 `i4-fast` = parallel GEMV + fusions + parallel-fused state attention; anima / Wan / irodori = activation quantization). A new variant reaches the default by one knob in `SessionSpec` plus a row in the family's acceptance table — flash-style attention included (Q is explicit-only under decision 2 as well). |
| 3. A fixed fast default is not portable (Mac)   | ADR 0128 decision 5 keeps the activation-quantized default on a device without dp4a and warns once per device, so the MUST of Context item 3 no longer holds; activation quantization is outside this ADR's automatic selection by decision 2.                                                                                                                                       |

What the automatic selection could pick today, and why it cannot yet (measurements as recorded in the
repository; none of them was taken under Node + Dawn):

| Variant                                               | Measured                                                                                                                                        | Blocked by                                                                                                                                                                                                                                                                                                                      |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `rmsNormReduce: "subgroup32"` (gemma4-qat)            | +2–6 % on the RTX 3080 Ti under Chrome (ABBA +2.06 %, attribution runs +3.77 % / +6.16 % — research 2026-09-13), within noise on the M2 (ADR 0100) | not expressible in a manifest (`SessionSpec` has no such key, `GpuFeaturesSpec` only `shaderF16`); four-point gate items ② ④ missing (ADR 0110 §9); `subgroups` is requested from the adapter only on an explicit knob; not re-measured on top of `i4-fast`                                                                       |
| `linearCompute` / `attentionCompute: "f16"` (class C) | 0.92–0.98× on the RTX 3080 Ti (ADR 0028); FP16 2:1 devices unmeasured (perf-ledger K-68)                                                        | no measured win; siglip2 / depth-anything / birefnet reject `linearCompute` in their acceptance tables                                                                                                                                                                                                                          |
| `linearGemvReduce: "parallel-subgroup32"`             | QAT +4–5 % on the RTX 3080 Ti, slower on the M2 (research 2026-09-13, ADR 0101)                                                                 | not a candidate: the packed / fused variants of `i4-fast` require `"parallel"`, and a preset declaration ranks above the profile (decision 6)                                                                                                                                                                                  |

Variants that depend on an optional GPU feature, or whose win flips between adapters, are the one class a
preset cannot express; none of them wins today on the reference devices.

### Re-entry conditions

The ADR is reopened when either is measured:

1. an E / C / R variant that needs an optional GPU feature (`subgroups`, `shader-f16`) or whose win flips
   between adapters wins on a family's reference device — for example `-c16` on an FP16 2:1 device (K-68), or
   `subgroup32` reproducing its gain on top of `i4-fast`;
2. a family whose acceptance table rejects every knob (siglip2, depth-anything, birefnet, vowel-detector,
   umT5) gains an E / R variant that is not bit-identical to the reference path.

### Wording adjusted by this addendum

- Decision 2, last sentence of the "Q is never automatic" item: "explicit" means that the runtime never
  selects it automatically; a default preset may declare it (ADR 0128 decision 7).
- Context item 3: its MUST is superseded by ADR 0128 decision 5 (noted inline).
- Staging: the work moved to phase 2 step 2-8 and was deferred there; none of 3-1a–3-1d is scheduled.
- The glossary (`docs/GLOSSARY.ja.md`) marks the profile terms as deferred and restores "reference path"
  as the term for the runtime's omitted-value path.
