# 0147: Numerics and kernel adoption — levels, contract classes, who decides, evidence checks, option applicability and per-GPU evidence

- Status: accepted (2026-11-10 — rulings by the maintainer in conversation on 2026-11-09; moved to the decision record on
  2026-11-10)
- Date: 2026-11-10
- Supersedes: [0058](0058-numerics-opt-in-contract.md) (numerics opt-in contract) and
  [0110](0110-practical-tier-numerics-contract.md) (practical-tier numerics contract) as a whole. Their still-valid content
  is restated in this ADR, so a reader never needs their decisions or addenda; "Where the old clauses went" (after decision
  11) maps every old clause to this ADR or says why it is not restated. Each superseded ADR carries one status line, and each
  amended clause elsewhere a status line plus, where its rule changed, the new text in place with a dated pointer here.
- Amends in part: [0029](0029-sbv2-i8-series-and-quant-quality.md) decision 4 / [0040](0040-fusion-pass.md) addenda of
  2026-09-12 and 2026-10-31 / [0050](0050-irodori-quant-series.md) decision 13 / [0099](0099-rms-norm-add-fusion.md)
  decisions 1, 3 and 4 / [0100](0100-rms-subgroup-reduction.md) decision 2 (missing device features rejected) /
  [0101](0101-linear-gemv-subgroup.md) decision 2 / [0103](0103-linear-static-quantize-fusion.md)
  "Option and eligibility" / [0104](0104-gemma-fast-quant.md) table and "Stored vocabulary" /
  [0111](0111-session-options-composition.md) decisions 2 and 6 and its addendum / [0117](0117-app-geometry-tuning.md)
  decision 8 (a backend field) / [0120](0120-wan-dit-w8a8-seat.md) decision 7 / [0128](0128-default-preset-original-weights.md)
  decision 1, its Related line and the human-review sentences of its addenda of 2026-11-03 and 2026-11-04 /
  [0132](0132-online-int8-attention.md) decision 5 (the fallback warning) / [0138](0138-moss-image-dit-vae.md) decision
  13 and the addendum of 2026-11-01, B2 (`preset.ts` owns a preset's knobs) / [perf-ledger](../perf-ledger.md) policy
  preamble / [quantization](../quantization.md) Table 2-b / [release runbook](../release-runbook.md) (the `/kernel-check`
  step) / [glossary](../glossary.md) (the rows of the numerics terms). Status lines only, no change of rule:
  [0033](0033-vae-fixed-tile-decode.md), [0106](0106-device-keyed-references.md), [0118](0118-wan21-video-generation.md),
  [0121](0121-wan22-ti2v-5b.md), and the deferred [0126](0126-numerics-profiles.md) (its amendment targets in 0058, 0110 and
  0111 now live here).
- Evidence: [research 2026-11-10](../research/2026-11-10-session-key-applicability.md) (which `session` keys have a place in
  six families' graphs). Rules index: [kernel-rules](../kernel-rules.md).
- Related: [0005](0005-verification.md) (verification stages; stage 4; the upstream-parity policy of 2026-10-11) /
  [0006](0006-quantization.md) (fake-quant is the reference of quantized storage) / [0033](0033-vae-fixed-tile-decode.md),
  [0036](0036-freeform-resolution.md), [0038](0038-manifest-v1.md) §4, [0118](0118-wan21-video-generation.md),
  [0121](0121-wan22-ti2v-5b.md) (tiled VAE decode) / [0065](0065-exporter-core-recipe-split.md) (the wheel boundary) /
  [0098](0098-linear-gemv-parallel.md), [0100](0100-rms-subgroup-reduction.md), [0104](0104-gemma-fast-quant.md),
  [0105](0105-packed-static-quantize-activations.md) / [0106](0106-device-keyed-references.md) (sha256 rows per
  environment) / [0107](0107-model-input-error.md) (error type by origin) / [0115](0115-geometry-profiles.md),
  [0117](0117-app-geometry-tuning.md) (geometry profiles and sweep records) / [0129](0129-test-layout-and-lanes.md)
  (lanes; upstream parity opt-in since 2026-10-17) / [0137](0137-qwen3_5-text-encoder.md) decision 6 (IR CPU reference
  execution as an oracle) / [0144](0144-gemma4-12b-vision.md) decision 1 (a cross-implementation audit) /
  [0145](0145-gemma4-26b-moe-resident.md) decision 6 (routing compared first) / the sources of the classification in
  decision 2: [0025](0025-w8a8-linear-execution.md), [0028](0028-f16-compute-variants.md),
  [0030](0030-attention-a8-execution.md), [0031](0031-attention-score-f16-storage.md),
  [0102](0102-state-attention-stats-pv-fusion.md), [0133](0133-conv3d-w8a8.md), [0140](0140-integer-gemv.md),
  [0141](0141-byte-plane-gemv.md), [0143](0143-f16-full-state-storage.md) / [0069](0069-packed-w4-storage.md) addendum 9
  (listening for group-size rulings). Out of scope: the op vocabulary
  ([0059](0059-op-vocabulary-entry-doors.md) / [0064](0064-entry-door-generality-axes.md)) and the extension point for
  model-specific kernels (a separate ADR later — [0148](0148-external-fusion-rules.md), fusion rules from outside the runtime).

## Context

The rules for adding a kernel, a fusion or a path that changes numbers grew in two ADRs and many later clauses: ADR 0058
(2026-08-14) with five addenda, ADR 0110 (2026-09-26) with two, and pieces in ADR 0128, 0111, the ADR 0040 addenda, ADR 0132
and the family ADRs. A rules index of 2026-11-09 ([kernel-rules](../kernel-rules.md)) collected them and found
statements that no longer match the code or each other. The maintainer ruled on the open points the same day; this ADR
writes those rulings down and replaces 0058 and 0110.

What the review found:

- **Two axes were one.** The docs split presets by whether `session` is empty (ADR 0110 decisions 1 and 7). That split is
  right for sha256 rows, but it hides the purpose of an option: class C holds `attentionScoreStorage: "f16"` (s16 — it frees
  the S buffer) and `fullStateStorage: "f16"` (memory) next to f16 compute (c16), which ADR 0128 C1 keeps out of every
  default because it needs `shader-f16`. A purpose axis and an evidence axis are both needed.
- **Class Q is not empty.** ADR 0110's decision 2 table says "none (reserved)", as do the glossary's "Contract classes"
  row and the header of Table 2-b of `docs/quantization.md`. ADR 0132 decision 4 admitted
  `attentionScoreStorage: "none"` (online int8 attention) as Q, and it is in two default presets: Wan2.2
  `i8-a8-attn8-online-conv8` and MOSS-Image `f32+dit8-a8-attn8-online` (local mirrors `models/karume-wan2.2/karume.json`,
  `models/karume-moss-image/karume.json`, read 2026-11-09).
- **Human judgement sat in many places.** ADR 0058 decision 7 (quality ruled when an option is introduced), ADR 0029
  decision 4 (listening before adopting a conv1d path), ADR 0110 decision 8 and ADR 0128 decision 1 (human review before
  `defaultQuant`), the ADR 0058 addendum of 2026-09-13 ("reading the output" as part of adoption), ADR 0121's verification
  table and ADR 0069 addendum 9 (listening for group-size rulings). Some of these are kernel admission, some are preset
  choice; they were not told apart.
- **Torch parity cannot catch an upstream bug.** A model-exit golden runs the same upstream model code that the exporter
  captures. It does catch translation and kernel bugs, patch bugs where the reference is the unpatched upstream (Wan,
  Anima, MOSS), bugs in Karume's TS host code where the reference is the upstream pipeline, and generation-loop and state
  bugs. Since 2026-10-17 these cases run only with `KARUME_UPSTREAM_PARITY=1` (ADR 0129 addendum 2026-10-17 (2)). Karume's
  own oracle for a whole graph, the IR CPU reference execution (ADR 0137 decision 6), took 11 min 31 s single-threaded for
  the Qwen3.5 9B text encoder on two prompts (T = 10 and 63) over i8 weights
  (`outputs/diag/qwen3_5-export-2026-10-31/run_full_ir.log`, untracked; ADR 0137 rounds it to about 12 minutes); linear took
  441.7 s of the 518.4 s at T = 63 (`ir.full-i8.fridge-note.json`, same directory). The ruling's figure puts the
  interpreter at about 2.6 GFLOP/s. A count from ADR 0137's own facts (7.40 GiB
  uploaded in i8, less the 248,320 × 4,096 embedding that does not go through `linear`, so about 6.9 × 10⁹ linear weights)
  gives about 2.0 GFLOP/s at T = 63 — an estimate of the same order, not a measurement.
- **Bit-identical fusions hid behind boolean opt-ins.** Of the ten rules in `FUSION_RULES`
  (`packages/runtime/src/runtime/fusion.ts`), seven run without an option and three need one: `fuseRmsNormAdd`,
  `fuseLinearStaticQuantize`, `fuseGeluMulStaticQuantize` (each rule's first match condition, in `rms-norm-add.ts`,
  `linear-static-quantize.ts` and `gelu-mul-static-quantize.ts` under `packages/runtime/src/runtime/fusion-rules/`). All ten
  claim bit identity with the unfused plan, and that identity is measured behaviour, not a WGSL guarantee (ADR 0040
  decision 2; ADR 0099 decision 3 — WGSL allows reassociation and fma contraction). The ADR 0058 rule "an automatic
  selection that claims bit identity needs a machine check on that device" (addendum 2026-08-29, rule 3) was never
  reconciled with the default-on rules; their per-backend evidence is the twin-graph tests on the development machine, with
  other backends a SHOULD (ADR 0040 addendum 2026-09-12).
- **Option acceptance was hand-written per family.** Twelve `FamilySessionPolicy` tables (one per family under
  `packages/models/src/`, for example `GEMMA_SESSION_POLICY` and `WAN_SESSION_POLICY`; the type is in
  `packages/models/src/session/options.ts`) decide which manifest keys a family accepts. An investigation of 2026-11-09,
  recounted against the local mirrors on 2026-11-10 ([research 2026-11-10](../research/2026-11-10-session-key-applicability.md)),
  found that most `false` entries mean "no op in the graph could be affected", which the graph shows — in sbv2, birefnet,
  depth-anything, siglip2, vowel-detector and umT5, 66 of the 71 `false` entries are so — while others mean "not
  verified". Two wiring paths would drop an accepted key once acceptance is derived: gemma does not forward
  `linearCompute` to its Session (`packages/models/src/gemma/pipeline.ts`), and Wan's text stage builds the umT5 Session
  with `{}` (`encodeWanPrompts`, `packages/models/src/wan/text-stage.ts`); both tables reject those keys today
  (`GEMMA_SESSION_POLICY`, `UMT5_ENCODER_SESSION_POLICY`), so neither drops anything yet. The value × op combinations that
  cannot run throw only at the first run, after the weights are uploaded (`buildLinear` in
  `packages/runtime/src/runtime/recipe-builders/linear.ts`; `fusedAttentionViolation` and
  `stateAttentionVariantViolation` in `packages/runtime/src/runtime/recipe-builders/attention.ts`), while the hub's fetch
  is GB-scale.
- **An option the adapter cannot run is handled three ways: a throw, a silent fallback or a warning.** f16 compute without
  `shader-f16` throws at construction (`buildSessionState`, `packages/runtime/src/runtime/session-build.ts`) and, on a
  shared GPU, at admission (`sessionGpuFeatures` and `assertGpuFeaturesGranted`,
  `packages/models/src/session/gpu-features.ts`); `parallel-subgroup32` without subgroups throws (`buildSessionState`);
  online attention on a device that holds neither of its reduction forms (at D = 128, every 32 KiB device without
  subgroups) falls back silently to the three a8 dispatches (`attentionRoute`,
  `packages/runtime/src/runtime/attention-route.ts`); a8 without dp4a warns once (ADR 0128 decision 5); the attention dp4a
  canary throws `GpuFeatureError` when both arms miss its sanity band (`decideAttentionI8a8Dot`,
  `packages/runtime/src/gpu/attention-dp4a-canary.ts`).
- **Per-GPU evidence has no backend.** The node lanes run Dawn / Vulkan only; the `/kernel-check` page covers the subgroup
  reductions, the states-attention rewrites and a few kernels; of the fusions it covers only the subgroup32 form of
  rmsNormAdd ("fused vs unfused identity"), not the default-path fusion rules (`packages/kernel-checks/README.md`); the
  geometry sweep record keeps the four `GPUAdapterInfo` fields (`ReportAdapter`) and the user agent, not the backend
  (`packages/runtime/src/tune/report.ts`); WebGPU exposes no backend field.
- **The VAE tiled decode is an approximation in the only path** (Anima — ADR 0033 measurements and ADR 0038 §4; Wan2.1 —
  ADR 0118 ruling 1; Wan2.2 — ADR 0121 decision 6). No contract named it.

## Decision

### 1. Levels — the purpose and admission route of a path

A **level** says what a path is for and how it is admitted. It is a separate axis from the contract class of decision 2
(the form of the evidence); the two are not one-to-one.

| Level | What it is | Members today | Admitted by |
| --- | --- | --- | --- |
| Reference | The reference path (no numerics option — the runtime's omitted values) over whatever weights the preset stores | every preset with an empty `session` | the maintainer's intake evaluation of model correctness (decision 3 (4)); the checks of decision 4 are evidence read at intake, not a machine gate |
| Practical | What browser execution needs | storage quantization (f16 / bf16 / i8 / i4 / i2 …); a8 = int8 activations (`linearCompute` / `attentionCompute` / `convCompute: "a8"`); s16 (`attentionScoreStorage: "f16"`); f16 state (`fullStateStorage: "f16"`) | machine checks (decision 3 (1)) |
| Opt-in | Speed by a changed order of operations (class R) or by approximation (class Q), and c16 (f16 compute, class C) | for example `linearGemvReduce: "parallel"`, `stateAttentionReduce`, `attentionScoreStorage: "none"`, `linearCompute` / `attentionCompute: "f16"` | machine checks (decision 3 (1)) |
| Preset | A configuration the maintainer checked for speed and quality | the manifest's `quants.<name>` | the maintainer (decision 3 (2)) |

- **The reference level is not "original weights only".** For quantized storage the fake-quant model is the definition
  (ADR 0006: weights are rounded to representable values before export and golden generation). Defining the reference as
  the original weights would leave SBV2, Wan2.1 / 2.2, umT5 and MOSS-Image with no reference artifact: none of them ships an
  original-weights preset (ADR 0128 decision 4 table). Original weights stay a storage axis (ADR 0128 decision 4).
- **The reference is the upstream computation plus declared deviations** (check E of decision 4, decision 10). Read with
  ADR 0006's addendum "the reach of the core decision" (2026-09-24), which already calls the static SRQ of QAT the model's
  own semantics, the fixed SRQ of a quantization-aware model is part of the upstream computation and so belongs to the
  reference level (a reading of ADR 0006, not a separate ruling). A declared approximation in the only path is allowed
  ("approximate reference", decision 10).
- **dp4a is an implementation of a8, not a numerics change**: its output is bit-identical to the emulation (ADR 0025
  decision 5; the module doc of `packages/runtime/src/runtime/a8-emulation-warning.ts`). It is chosen by capability (decision 8 and
  decision 6) and is not a level member of its own.
- Levels and classes cross: class C spans the practical level (s16, f16 state) and the opt-in level (c16); a practical- or
  opt-in-level option may be in a preset, and options of both levels are in default presets today (Wan2.2 and MOSS-Image
  carry the Q option above; the mobile QAT default `i4-fast` carries the R option `linearGemvReduce: "parallel"`, local
  mirror `models/karume-gemma4-qat-mobile/karume.json`).
- The sha256 row classes keep their derivation from the preset's `session` (decision 11): an empty `session` is exactly the
  reference level. Storage quantization is a practical-level purpose, but a preset that stores quantized weights and has an
  empty `session` runs the reference path over them, so its rows are reference-level rows. The level names a purpose; a
  preset's sha256 row class follows its `session` alone.
- "Practical level" (this decision) and "practical preset" (a preset with a non-empty `session`, the side of a self A/B
  check and of the non-reference sha rows) are different terms: an R or Q option is opt-in level, yet a preset that declares
  it is a practical preset.

### 2. Contract classes E / C / R / Q — the form of the evidence

A numerics-changing variant declares the strongest class that can be built. The classification of 2026-11-09 is the table
under "Classification of existing options" below; `docs/quantization.md` Table 2-b is its living copy, kept current
with every new option.

| Class | Claim | Form of the check |
| --- | --- | --- |
| E (mirror) | bit-identical (atol = 0) to the TS mirror, or to the unfused plan for a plan transformation | exact comparison (u32 for a transformation) |
| C (composition) | identical (atol = 0) to "host-rounded input + the existing f32 kernel" | exact comparison after host rounding |
| R (addition order) | only the summation order differs; the f64 band **is** the reference op band | the four-point check (below) |
| Q (approximation) | a band wider than the reference op band, from an error model | the four-point check + the error model |

- **Q has a member**: `attentionScoreStorage: "none"` (ADR 0132 decision 4 — the P̃-lattice error model and its four points),
  in the Wan2.2 and MOSS-Image default presets. The "none (reserved)" of ADR 0110 is withdrawn.
- **MUST: an ADR that declares a weaker class states why a stronger one cannot be built** (the first defence against vacuous
  checks). IEEE-exact operations (add, subtract, multiply, explicit `fma`) can probably give E with a TS mirror; built-in
  transcendentals (`exp` …) cannot (driver differences), but a mirror with a statistical check is a precedent (qP of ③PV,
  ADR 0030).
- **C's atol = 0 relies on f32 → f16 narrowing that rounds to nearest even**, implementation-defined in WGSL; a C check holds
  after a per-device rounding check (ADR 0028 / 0031). s16 is the exception: D3D12 rounds toward zero, so its host rounding is
  the backend's own (ADR 0031 addendum 2026-11-07).
- **`packedStaticQuantize` is not strictly E**: finite values are u32-identical to the unfused SRQ, but NaN saturates to
  ±127 / −128 where the f32 path passes it through, and −0.0 becomes +0.0, which does not move the output (ADR 0105 decision
  3). Decision 5 keeps it opt-in.

**The four-point kernel check (MUST for every new R or Q option)**, from the state-attention parallel test
(the module doc of `packages/runtime/tests/gpu_state_attention_parallel_test.ts`):

1. the band against f64 (R: the reference op band; Q: the error model's upper bound);
2. the A/B band against the reference path, the same input through both;
3. exact checks of structural guarantees (special values, exact 0 on pad rows, the −inf bit pattern, NaN propagation,
   geometry invariance, two dispatches of the same input bit-identical);
4. a record of a fault injection that turns ① or ② red.

**Band rules.** R: the f64 band is the reference op band of the same class in
`packages/runtime/tests/helpers/op-tolerance.ts`; the A/B band is at least 4 × the worst measured and at most the theoretical
bound (2 × the f64 band). Q: the f64 band is the error model's bound, and a perturbation corpus shows at most a quarter of it
in use. A band is a declaration, not a per-environment value; the writing modes of `KARUME_REFERENCE` do not rewrite it, and
only a Q band shown to need a per-environment relaxation takes the form of the ADR 0106 addendum decision 1 (that machine
only). MUST NOT: derive an E2E band as "5–10 × the measured" (ADR 0025 decision 6, generalised).

**Grace clause.** The existing R options keep their admission with the missing points filed in the backlog: parallel GEMV
lacks ④ (and has no worst-case record); the subgroup32 reductions lack ② and ④. New R and Q options are admitted only with
all four points.

**Classification of existing options** (2026-11-09; 0110 decision 9 plus the options admitted since, from Table 2-b of
`docs/quantization.md`):

| Option (value) | Class | Four points | Source |
| --- | --- | --- | --- |
| `linearCompute: "a8"` | E (dp4a vs emulation also atol = 0) | — | ADR 0025 |
| `convCompute: "a8"` | E | — | ADR 0133 |
| `attentionCompute: "a8"` | E (statistical check for qP of ③PV) | — | ADR 0030 |
| `linearCompute` / `attentionCompute: "f16"` | C | — | ADR 0028 |
| `attentionScoreStorage: "f16"` | C | — | ADR 0031 |
| `fullStateStorage: "f16"` | C | — | ADR 0143 |
| `attentionScoreStorage: "none"` | Q (P̃-lattice error model) | ① ② ③ ④ | ADR 0132 |
| `stateAttentionReduce: "parallel"` | R | ① ② ③ ④ | `gpu_state_attention_parallel_test.ts`; ADR 0102 |
| `stateAttentionReduce: "parallel-fused"` | R, by u32 equality with `"parallel"` | inherits `"parallel"`; its own checks are the f64 band, NaN / Inf by class and zero padding (no ② ④ of its own) | `gpu_state_attention_parallel_test.ts`; ADR 0102 |
| `linearGemvReduce: "parallel"` | R | ① ② ③ (no ④, no worst-case record) | `gpu_linear_gemv_parallel_test.ts`, ADR 0098 |
| `linearGemvReduce: "parallel-subgroup32"` | R | ① ③ + u32 equality with `"parallel"` on the same device (no ② ④) | `packages/kernel-checks/src/gemv-subgroup.ts`; ADR 0101 |
| `rmsNormReduce: "subgroup32"` | R | ① ③ (no ② ④); its only u32 equality is the RMS→add fusion on against off, both under subgroup32 — none against the workgroup form | `packages/kernel-checks/src/rms-subgroup.ts`; ADR 0100 |
| `linearGemvPlanes: true` | R | ① ② ③ ④ | ADR 0141 |
| `integerGemv: true` | E (TS mirror) | — | ADR 0140 |
| `packedStaticQuantize: true` | E on finite values; NaN saturates (above) | — | ADR 0105 |
| the fusion rules (`fuseRmsNormAdd`, `fuseLinearStaticQuantize`, `fuseGeluMulStaticQuantize` and the seven without an option) | E (u32 equality with the unfused plan); default-on under decision 5 | — | ADR 0040, 0099, 0103 |

Point ④ is an executable mutant test for the online attention, the states-attention parallel reduction and the
byte-plane GEMV. The integer GEMV and the packed static quantize (class E, outside the four points) carry their fault
injection as a record in their ADRs.

**Census.** A selected variant must be seen to run: the census reads `SessionDiagnostics.lastRunPipelines` (derived from
the recipe of the last run, independent of timing) through the predicates of
`packages/runtime/tests/helpers/pipeline-census.ts`, and it is red when every dispatch fell back. For fused attention one
pure function decides the per-stage route (`attentionRoute`, `packages/runtime/src/runtime/attention-route.ts`); recipe
derivation and the acceptance tests go through it. A Session that declares a8 / s16 is checked per fused-attention family
for completeness as a ratio (`assertAttentionSeatsComplete`), kept apart from the "at least one" check
(`assertSeatsApplied`) that tests hitting fallback shapes on purpose call; host tests pin each family's accepted shapes
through `attentionRoute`.

**The E2E self A/B check of a preset** (not a numerics contract — a model-exit band goes vacuous: activation quantization
saturates 1e-5 upstream differences within a few layers, stacked options remove later detectors, and a torch control
carries its own error). **Scope**: a distributed preset with a non-empty `session` (0110 decision 5; per distributed
bundle, ADR 0058 addendum of 2026-09-26 ③). **The check is MUST when such a preset is distributed**, and it stays the
standing regression check of that preset afterwards. Gemma's distributed presets, which have no same-machine A/B check
today (0110 addendum of 2026-09-26), keep their standing under the grace clause above, with the missing check filed in the
backlog. The four families whose presets are f32-only are out of scope: they have no partner preset to compare
against. Against the reference path on the same machine (not torch goldens):

1. **Census MUST**: variant keys per option of the bundle; fully eligible ⇒ 0 reference-variant keys.
2. **Difference floor MUST**: the output differs by at least one bit from the preset with the same weights (same storage
   dtype) and an empty `session`, on the same machine and input (it detects a silent fallback of the whole bundle). For
   Anima's `f16+dit8-a8-attn8-s16` the control is `f16+dit8` (`f16` has other weights and would make the floor vacuous).
   The same-machine reference path is deterministic, so the floor can always be set; a fallback of a single option is
   item 1's. A torch golden or another preset's f32 is not a valid control: its implementation error shrinks the
   separation.
3. **Collapse ceiling MUST**: the relative RMS at the observation point is under a ceiling of about 2 × the measured value,
   with no non-finite values; for a quantized preset the docstring records agreement with the theoretical value.
4. **The observation point MUST sit right after the preset first takes effect** (DiT = the latent after step 1; encoder =
   layer 1 output; LLM = the first logits; sbv2 = the four front-graph outputs, since the waveform length may change).
   Final outputs (PNG / WAV / tokens) are recorded but carry no band.
5. Each check's docstring records at least one fault injection that turned it red.

Its implementation form stands as built: one result name per family, three sequential pipelines, no `sharedWeights` (the
distribution containers carry no `shared` declaration); the reference partner is derived from the manifest (same weights
mapping, empty `session`, GPU requirements a subset, exactly one); the floor has two levels ("bit-identical to the
reference → red", always checked, and `band.floor` above 0 only when measured — for Anima, removing `attentionCompute` gave
0.90 × the full bundle, so its floor stays 0 and single options are left to the census). **A check without a derived band
prints its measurement and stops red.** The census source is `packages/models/tests/helpers/census-table.ts`, and
`packages/models/tests/census_table_test.ts` is red when a non-reference bundle of a local mirror has no row; presets
without a row (prefill parallel GEMV / attention, parallel-fused, Anima's attention presets) keep the "at least one / zero"
checks.

### 3. Who decides what

1. **Admitting a kernel or numerics path as a runtime option: machine checks only (MUST).** No human judgement. The checks
   are those carried over from 0058 decision 4 and 0110: the checks of its class (decision 2, with the four-point check for
   R and Q), the census, and the reference path's sha256 rows passing unchanged (proof that the default did not move).
   Check C of decision 4 is the class check of a new op kernel and check D that of a plan transformation; they add no
   admission gate of their own. Bit-identity checks indicate that an implementation is correct; they are not the goal (the maintainer's
   background policy of 2026-08-13, ADR 0058 Context).
2. **Making a configuration a preset: the maintainer, after a human quality review.** Every preset is reviewed by a human
   before it is published, in every family: image, video, audio and text alike. Publishing a distribution is a manual step,
   so the review is part of it.
   - **The default preset's review is never skipped.**
   - **Any other preset's review MAY be skipped only when the maintainer approved the skip beforehand.** A skipped review
     is replaced by a machine evaluation: a measurement of quality that needs no human, or an LLM evaluation (for example
     the KL-divergence and top-1 measurement behind the removal of the regular Gemma 4 int4 presets — ADR 0128, second
     addendum of 2026-11-04). The record names the approval and the method used.
   - **The form of a human review** (carried from 0110 decision 8): a visual review uses at least 4 seeds × 2–3 prompts;
     audio is reviewed by listening; a text model is reviewed with a fixed evaluation plus reading the output.
   - **A dominated configuration is not made a preset**: the same speed with clearly worse quality, or slower with the same
     quality. Reference-level presets are excepted.
   - **Presets are not absolute**: a user may run other settings (decision 7).
   - Quality is judged on the shipped bytes, not on a simulation (ADR 0050 decision 13, restored for every preset: a
     measurement rig re-solves rounding and re-draws the output, so delicate properties such as prosody do not transfer).
   - A `fast` bundle's options are chosen by speed, measuring each option alone and all of them together on the same machine
     (ADR 0104 addendum of 2026-09-26).
   - **The delegation of 2026-09-13 covers evaluation and proposal** (narrowed by the ruling of 2026-11-09). The
     maintainer confirmed two goals — a verified path that agrees numerically, and a model default as fast as possible —
     and delegated adoption decisions. What stays delegated is measuring, evaluating and proposing candidates, resting on
     automatic evaluation and on reading the output. Making a configuration a preset, choosing the default and publishing
     stay with the maintainer. The delegation is not a standing approval to skip a human review: that approval is given
     per occasion. Publication is never an unattended step; it waits for the maintainer's explicit go-ahead, because the
     licence check and the reading of the published text are manual (item (5) below). The evaluation rules of the
     delegation stay. Bit or token differences alone are no reason to reject; speed and the absence of large quality loss are judged from fixed evaluations, real output and evidence on the
     target devices, and unmeasured ranges are stated; where a human check was needed, the record says whether it was done,
     and a human evaluation that did not happen is never recorded as done. Runtime and kernel defaults stay the reference
     path; models use speed-ups only through the selected preset's `session`, never by picking a form from the model shape,
     and the distribution's default is normally a preset with speed-ups (ADR 0128 decision 1). A new model default is
     machine-checked for two things — the variant actually applies, and it matches the same explicit options; its quality
     and speed are judged under this decision 3 (2) like any preset's — and invariants within one configuration (such as MTP
     agreement) still hold. The frozen reference configuration and its digests are kept and named in the checks. A preset
     that declares activation quantization has a sibling with the same weights and no a8 (ADR 0128 C2).
3. **The default preset: the maintainer picks one of the presets** and records the candidates, the measurements and the
   reason (the procedure of ADR 0128 decision 1). The manifest constraints C1–C3 of ADR 0128 decision 2 still hold and are
   machine-checked.
4. **Model correctness** — does the model produce sane results — is evaluated by the maintainer at intake, separately from
   kernel adoption.
5. **Non-numeric manual work stays**: the licence check and the real-browser `/kernel-check` run (both in §1 of the
   [release runbook](../release-runbook.md)) and ADR 0005 stage 4 (the real-browser acceptance lane).

Clauses elsewhere, as ruled:

| Clause | Change |
| --- | --- |
| ADR 0058 decision 7 (crossing with storage presets; quality ruled per option) | superseded: combinations by decision 7 of this ADR, quality by decision 3 (2) |
| ADR 0058 addendum 2026-09-13 (delegation) | content carried into decision 3 (2) above |
| ADR 0110 decision 8 (quality verdict only on entering `defaultQuant`) | rewritten as decision 3 (2): it covers every preset, not only `defaultQuant` |
| ADR 0029 decision 4 (listening before adopting a conv1d w8a8 / w8a16 path) | split: admitting the path as a runtime option is machine-only (3 (1)); making it a preset stays with the maintainer (3 (2)) |
| ADR 0069 addendum 9, lesson 2 (group-size rulings need a listening check) | kept: a group size is a preset decision of audio quality, and its listening check is the human review of decision 3 (2) |
| ADR 0050 decision 13 (judge on the shipped bytes) | restored to apply to every preset |
| The policy preamble of `docs/perf-ledger.md` | rewritten to point to this ADR |
| ADR 0121 decision 5 table and its Verification list | kept |
| perf-ledger K-80 / K-81 (umT5 `i8-a8`) | unchanged: the option may exist; it was not made a preset (K-81's reason: umT5 rows collapse under per-token int8) |
| The release runbook's `/kernel-check` run and licence check; ADR 0005 stage 4 | kept (the `/kernel-check` step's scope is decision 9 (1)) |

### 4. Correctness evidence beyond upstream parity

The upstream is declared as the specification, and correctness is split into checks that each have their own oracle (ruled
on 2026-11-09). Most of the checks below do not exist yet in this form; building them is filed in the backlog.

| Check | What it proves | Oracle | Where and how |
| --- | --- | --- | --- |
| A — export fidelity | the exported krm computes what the eager model computes | the eager upstream model on fake-quantized weights (the container's codecs, ADR 0006) | on the **published krm itself**, not a reduced model; at least 3 points per symbolic dimension, including window boundaries; compare routing first (precedent: ADR 0145 decision 6); lives in `tools/export-recipes` and calls Node, never in the PyPI wheel (ADR 0065) |
| P — patch fidelity | Karume's recipe patches do not change the model | the unpatched upstream eager model | patched vs unpatched, required for every family; recorded deviations (E) are excluded and carry their own checks |
| F — host glue | schedulers, tokenizers, pre- and post-processing match upstream | the upstream host code | bit-exact, no GPU (precedents: Wan2.2's host unpatchify bit-identical to upstream and the LANCZOS port matching Pillow in uint8 on 105 cases — ADR 0121 addenda of 2026-10-05, stage 5, and of 2026-10-09, stage 9a) |
| B — runtime fidelity | the GPU runs the IR as the CPU reference does | the IR CPU reference execution (ADR 0137 decision 6) | reference path only; small inputs only — the interpreter runs at about 2.6 GFLOP/s single-threaded (Context), and Chrome refuses one `ArrayBuffer` above 2,145,386,496 bytes ([limitations](../limitations.md), the section on Chromium's single-`ArrayBuffer` cap) |
| C — op level | each op kernel matches its contract | the TS oracle and the op tolerance band | keep the √K rule (maxAbs ≈ C · eps · √K · max\|expected\|, C fitted at 0.09–0.62 — research 2026-08-31 §3.1, the bands derived from it in §7, the module doc of `packages/runtime/tests/helpers/op-tolerance.ts`); add fault injection on the core ops (matmul, conv, …), filed in the backlog; worst-case forward-error bounds (Higham) are not pass lines — the ruling's figure is at least 60 × looser than measured; an estimate from the fitted rule at K = 4096 (n·u ≈ 2.4e-4 with u = 2⁻²⁴, against C · eps · √K ≈ 2.3e-6–4.7e-6 for C = 0.3–0.62, eps = 2⁻²³) gives about 50–110 ×, and the two normalise differently (Σ\|a\|\|b\| per element against max\|expected\|), so it is not a measurement |
| D — plan transformations | fusion, row blocks and geometry do not change a bit | the untransformed plan | bit identity |
| G — loop and state | chunk prefill and decode, and drafter acceptance, agree | the same krm run in one pass | see the pass criteria below |
| E — upstream bugs | Karume did not inherit an upstream defect it could see | other implementations | at intake, a cross-implementation audit (the official repository, transformers, vLLM, llama.cpp — LLM work, recorded; precedent: ADR 0144 decision 1); a deviation record of every place where Karume did not copy an upstream bug (ADR 0005 addendum 2026-10-11), each with its own check |

- What torch parity catches and what replaces it:

  | Bug class | Caught by model-exit torch parity today | Replacement |
  | --- | --- | --- |
  | upstream model definition | no — the reference runs the same code | E |
  | translation (aten → IR, normalize, convert) and kernels | yes | A, B, C, D |
  | recipe patches | only where the reference is unpatched (Wan, Anima, MOSS) | P |
  | TS host code and pipeline wiring | where the reference is the upstream pipeline | F |
  | generation loop and state | for the LLM loops | G |

- **These checks are evidence the maintainer reads at intake** (decision 3 (4)), not a machine gate of the reference
  level; the existing families keep their standing.
- **Checks C and D are class checks** (decision 2): check C is the class check of a new op kernel, check D that of a plan
  transformation. Neither adds an admission gate (decision 3 (1)).
- **Pass criteria of checks A and G**: a check passes exactly where both sides are exact by construction, otherwise within
  the band of the weaker contract class of the two sides. The detailed criteria (for G, the token sequence compared and
  whether KV slots are in scope; for A, which side of each window boundary is sampled and which graphs compare routing
  first) are fixed when each check is built.
- **The upstream-parity GPU lanes at model exit are retired file by file**, and only once each file's replacement check is
  recorded. A file whose bug class has no replacement keeps its opt-in lane.
- Check A on the published krm is what catches a fold of a symbol into a constant that agrees near the symbol maximum but not
  at small sizes: the exporter's own fold check evaluates only Tmax and Tmax − 1 and says it does not prove commutativity
  (`_check_prefix_commutes` and `_point_value` in `tools/exporter/src/karume/convert.py`, and the comment on the foldable
  ops there); a reduced-config export would fold other constants (the investigation's finding of 2026-11-09).
- In this ADR "check C" and "check E" name rows of this table; "class C" and "class E" name contract classes of decision 2.

### 5. Bit-identical fusions are default-on

- **A fusion that is bit-identical to the unfused plan runs without an option.** Bit identity is measured behaviour, not a
  WGSL guarantee (ADR 0040 decision 2; ADR 0099 decision 3). The rule that an automatic
  selection claiming bit identity needs a machine check on that device (decision 6) applies to default-on fusions too, and
  the Session canary of decision 9 meets it.
- `fuseRmsNormAdd` becomes default-on. `fuseLinearStaticQuantize` is applied automatically when its prerequisite path is
  enabled: `linearGemvReduce: "parallel"` with `linearCompute: "f32"` (today's combination rule in
  `sessionOptionsViolation`, `packages/runtime/src/runtime/session-options-check.ts`). `fuseGeluMulStaticQuantize` applies
  wherever its gelu → mul → SRQ chain matches, with no prerequisite option: the rule reads no option today besides its own
  flag (the doc of `SessionOptions.fuseGeluMulStaticQuantize`, `packages/runtime/src/runtime/session-types.ts`:
  "Independent of linearGemvReduce / linearCompute").
- **No public switch replaces the removed keys.** A test-only symbol key in the style of `I8A8_DOT`
  (`packages/runtime/src/runtime/session-types.ts`) disables the fusion rules. The Session canary of decision 9, the
  fused-vs-unfused checks (check D of decision 4, the twin-graph tests) and bisecting a fusion against the reference path
  all obtain the unfused plan through that one entry. It lands together with the key removal, because existing tests
  obtain the unfused plan by omitting the option. (Updated 2026-11-11: the switch landed ahead of the key removal —
  `DISABLED_FUSION_RULES` in `packages/runtime/src/runtime/session-types.ts` takes `"all"` or rule names, a name outside
  the rule list fails at Session construction, and the annotation folds are not rules and stay on. The fused-vs-unfused
  comparison is `checkFusionTwin` in `packages/kernel-checks`; the twin-graph tests keep the plain sides they had, and
  the tests that omit an option move to the switch with the key removal.)
- **The three keys are removed from the vocabulary**: `SessionOptions` and the manifest `SessionSpec` (`SESSION_KEYS` and
  `SessionSpec`, `packages/hub/src/manifest.ts`). The published manifests that declare them are re-uploaded. This is a
  breaking change: the hub rejects an unknown `session` key (`assertAllowedKeys`, called from `parseSession` in
  `packages/hub/src/manifest.ts`).
  In the local mirrors of 2026-11-09 they are declared by `karume-gemma4-qat-mobile` E2B / E4B `i4-fast` (all three, the
  default of both) and `karume-gemma4-qat` E2B / E4B / 12B `i4-fast` (`fuseRmsNormAdd`); no preset's `session` becomes empty
  by the removal.
- `packedStaticQuantize` stays opt-in for now (decision 2); making it pass NaN, and so become E, goes to the backlog.
- The other rules of ADR 0040 are unchanged (exact matching, counters in `lastRunFusions`, rounding barriers, twin-graph
  tests, observation points through the fusion-count tests).

### 6. API rules

- **Each numerics axis is a named option whose values are a string union**, named `<target><axis>` with the reference
  value as the default. A boolean MAY be used when the option is certainly on / off, that is, when a second variant is all
  but ruled out. This is an API-shape rule, not a safety rule: a second variant would force a boolean axis to be
  redesigned (`attentionCompute` gained `"a8"` after `"f16"`). `packedStaticQuantize`, `integerGemv` and
  `linearGemvPlanes` stay booleans in this release; any of them that does not meet that test becomes a string union at
  the next breaking change of this area or at the 1.0 release, whichever comes first.
- **The output does not change silently.** The runtime default MUST be the reference path, and no numerics-changing path is
  selected automatically from adapter capabilities, from the storage dtype of the assets or from the graph shape (MUST): a
  different machine or asset must not silently change the output. The distribution default is normally the practical
  preset (ADR 0128 decision 1); a8 reaches a Session only through a preset's `session` or an explicit option (ADR 0128
  decision 7). The runtime's omitted values do not change, and presets never change them. The reference path's sha256 rows
  (PNG / WAV / golden) are frozen as the regression check and are never turned into a tolerance.
- **An automatic selection that claims bit identity is allowed only when the identity is machine-checked on that device**
  (a run-time canary, for example); enumeration or a specification claim is not enough. The attention dp4a canary is the
  precedent: it runs one full tile plus two K tiles at production geometry through all six generable variants in one submit
  and compares with the TS known answer (`packages/runtime/src/reference/i8a8.ts`); rule v2 picks dp4a on an exact match,
  else the emulation on an exact match, else by a sanity band of relative 1e-5 (one arm inside → that arm; both inside and
  identical → dp4a; both inside and different → the closer arm), warns once on any inexact verdict and on the
  `emu-exact` verdict (decision 8, device-driven choices), shows the emulation as
  `:dp4aEmu` in the pipeline key, and runs once per device at the first Session with `attentionCompute: "a8"`. The dp4a
  choice is held per family (`SessionState.linearI8a8Dot` / `attentionI8a8Dot`,
  `packages/runtime/src/runtime/session-build.ts`); the test-only `I8A8_DOT` forces both. Linear and conv3d a8
  choose dp4a or the emulation by enumeration today (`buildSessionState` sets `SessionState.linearI8a8Dot` from the
  enumerated language feature), and
  online attention chooses `sg32` or `wg` by capability (ADR 0132 decision 5). Both choices get a run-time canary on
  devices where the path is requested (decision 9): enumeration alone is not evidence. They are revisited only if the
  start-up cost proves a problem. Decision 8 changes the verdict when both arms are outside the band.
- **A warned switch listed in decision 8's table is not an automatic selection of a numerics-changing path.** Each row
  lands on the reference path or on a stronger contract class, or keeps the numbers bit-identical
  (`parallel-subgroup32` → `parallel`, dp4a → the emulation).
- The path is fixed at Session construction and does not change within a Session. The composition rule is one for every
  family: an explicit value > the preset's declaration > the runtime default (ADR 0111 decision 1). The preset is the only
  source of a bundle; `models` keeps no table of option values. The runtime SHOULD NOT carry a bulk numerics axis (ADR 0111
  decision 7): the bundle differs per family and storage dtype, and only a manifest preset can express it.
- An option declares only that a numerics-changing path is selected; the implementation decides by measurement which plans
  use it (the precedent is `stateAttentionReduce`).

### 7. Session option acceptance and discovery (amends ADR 0111)

**A derived applicability check replaces the per-family acceptance tables** (`FamilySessionPolicy`). It runs at admission,
before any weight part is fetched, per receiving Session and per (key, value):

- **reject when no op could be affected at any binding**, decided from static facts only — op kinds, weight storage, n / k /
  group and static dimensions, read from part 0's graph and `planWeightResidency` (public, exported from
  `packages/runtime/mod.ts`);
- **reject structurally impossible value × op combinations**: every rejection decidable from op kind, graph structure,
  weight residency and static dimensions moves from the first run (after upload) to admission. Always decidable: f16
  compute × resident i8 / i4 weights and non-f32 compute × resident i2 weights (both in `buildLinear`,
  `packages/runtime/src/runtime/recipe-builders/linear.ts`), a8 × masked attention (`fusedAttentionViolation`,
  `packages/runtime/src/runtime/recipe-builders/attention.ts`), non-f32 attention compute or score storage × states-form or
  readonly attention (`stateAttentionVariantViolation`, same file), a8 × i8 weights with k above
  `LINEAR_I8A8_MAX_K` (131,072; `buildLinearI8a8`), and w4a8 with a group above `LINEAR_W4A8_MAX_GROUP`
  (`linearI8a8Params`, `packages/runtime/src/kernels/linear-i8a8.ts`). Decidable when the dimensions involved are static:
  a8 × GQA attention (same lines), a8 attention with a head depth D above `LINEAR_I8A8_MAX_K`
  (`fusedAttentionViolation`), and `fullStateStorage: "f16"` × a full slot with an odd D (`assertStateStorageDepth`,
  reached through `stateSlotBytes`);
- **use the same predicates as the recipe builders**: the linear rejections become one shared violation function used by
  admission, the recipe builder (`buildLinear`, `buildLinearI8a8`) and the memory estimate, as the attention ones already
  are (`fusedAttentionViolation` and `stateAttentionVariantViolation`, also called from
  `packages/runtime/src/runtime/estimate.ts`);
- **"not verified for this family" is no longer a reason to reject**; what was checked is shown by the presets (decision 3).
- An option with a place at some binding but not at the binding of a given run still falls back silently there (decision 8);
  "at any binding" keeps that rule intact.

Kept from ADR 0111: the exhaustiveness of `WRITERS` (`packages/models/src/session/options.ts` — adding a `SessionSpec`
key fails type check); the error type by origin (decision 4: a manifest-declared key → `Error`, an explicit one →
`ModelInputError`; a broken combination → `Error` when the declaration alone breaks it); composition at admission (decision
5); and the routing of options to component Sessions (Wan's VAE decoder receives only `convCompute`,
`wanVaeSessionOptions`, `packages/models/src/wan/tile-decode.ts`). An explicit override surface is added to a family only on request, in the
style of ADR 0120 decision 7; today gemma, anima and irodori have one, and the other families pass `{}`.

**MOSS-Image's per-preset knob table** (`packages/models/src/moss/preset.ts`, `mossPresetOf`) stops gating: a MOSS-Image
preset's knob values derive from the manifest, as in the other families. The routing of knobs to its component Sessions
stays. ADR 0138 decision 13 kept the gate so that a manifest could not run unchecked numerics under a checked name. For a
published distribution, three things now hold that instead of a run-time refusal (ruling of 2026-11-09): a package version
reads a pinned revision (ADR 0073); a published preset's `session` changes only through a new pinned revision, with the
E2E self A/B check redone when the numbers move (decision 11); and a test compares the `session` each preset declares in
the pinned distribution with the knob values its checks were run on, and fails on a difference. A manifest the caller
supplies is outside these three and runs the knobs it declares, as in the other families (Wan's form).

Known wiring gaps that would drop a key once the derived check admits it (latent today, since both tables reject those
keys): gemma does not forward `linearCompute` to its Session (`packages/models/src/gemma/pipeline.ts`); the umT5
options do not reach Wan's text stage (`encodeWanPrompts`, `packages/models/src/wan/text-stage.ts`, perf-ledger K-81).

Two rejections stay at the first run, because they need facts outside the static set: w4a8 with a group that is not a
multiple of the geometry profile's `tileK` (`assertI8a8GroupGeometry`, `packages/runtime/src/kernels/i8a8-geometry.ts`
— a profile fact) and the attention guard on the key length N above `LINEAR_I8A8_MAX_K` (`fusedAttentionViolation` — N
is usually bound at run time).

**Discovery.** The same applicability function is a public query API, callable with the manifest and part 0 only (before
any weight): per preset and per Session it returns key → the values that have a place, plus the device features each value
needs; given a device, it also reports which values would switch on that device (decision 8). The TypeScript type of the
overrides is fixed and covers every numerics option of `SessionOptions`, not narrowed per family: a Hugging Face revision
can change a graph. It replaces today's `SessionOverrides` (`packages/models/src/session/options.ts`,
`Partial<Pick<SessionOptions, keyof SessionSpec>>`, the manifest keys only), and gemma's two explicit options
`rmsNormReduce` / `fullStateStorage` fold into it. The manifest vocabulary does not grow. Model cards show the preset table plus one line on how to query; they do not list options (that MAY be
added on request).

### 8. An option that cannot be used — by who can fix it

| Cause | Who can fix it | Behaviour |
| --- | --- | --- |
| The model does not support it (no place, or structurally impossible) | the user, by choosing other options or pinning a revision; the distribution, by its graph | throw at admission, error type per ADR 0111 decision 4; the message advises pinning the Hugging Face revision |
| The adapter lacks a device feature | neither the user nor the distribution (the device decides) | warn once per device, switch to a defined path, show it in the diagnostic pipeline key |
| The shape at run time is not eligible | not a fault (the input decides) | silent fallback, shown in the pipeline key |

The switch table for a missing feature:

| Requested | Missing | Runs | Numbers |
| --- | --- | --- | --- |
| `linearCompute` / `attentionCompute: "f16"` | `shader-f16` | `"f32"` | change, toward the reference |
| `linearGemvReduce: "parallel-subgroup32"` | subgroups | `"parallel"` | u32-identical to `"parallel"` with the same storage, group and lanes on a device that has subgroups (`packages/kernel-checks/src/gemv-subgroup.ts`, ADR 0101) |
| `rmsNormReduce: "subgroup32"` | subgroups | `"workgroup"` (the reference value) | change, toward the reference |
| `attentionScoreStorage: "none"` (online) | the device holds neither reduction form (see the note below) | the three a8 dispatches with S in f32 | change (Q → E); silent today, gains the warning |
| any a8 knob | dp4a | the emulation | identical; already warns (ADR 0128 decision 5) |

- Explicit overrides behave as declared ones. On a shared GPU the warning comes at admission; on a device the family
  acquires itself, only the available features are requested and the warning comes at acquisition.
- **A manifest's `gpuFeatures` is a request**, not a requirement: on a device without the feature the Session warns once
  per device and switches toward the reference as in the table above. When a shared GPU was acquired without a feature the
  preset requests, the warning says how to re-acquire (the re-acquire text stays in
  `packages/models/src/session/gpu-features.ts`).
- **Shape fallback** (ADR 0058 decision 3, kept): a requested option that a shape or a shape-derived limit cannot take (as
  linear `k % 4` and attention `D % 4` / `N % 4`) falls back to the reference route silently, and the fallback MUST appear
  in the diagnostic pipeline key. An unimplemented combination is not a fallback; it fails loudly (now at admission).
  Extended: a device-capability switch is allowed when it warns and shows in the key. 0058 decision 3 also let a device
  limit fall back silently; that is replaced by the rules for device-driven choices below.
- **The MUST that throws on f16 without `shader-f16` (`buildSessionState`, `packages/runtime/src/runtime/session-build.ts`) is removed**, with a replacement
  safeguard: the warning, the pipeline key and the query API of decision 7. What the throw protected — that the difference
  between the requested and the executed path is visible — is kept.
- **The attention dp4a canary with both arms outside the sanity band warns and runs that attention in f32**, instead of
  throwing `GpuFeatureError` — the same treatment as the Session canary of decision 9.
- Strict callers (benchmarks, sha checks) decide through the query API and the diagnostics. No new option is added; an
  `onDeviceFallback: "warn" | "throw"` option MAY be added later on request.
- Note on the online row: without subgroups the kernel takes its portable `wg` form, which is bit-identical to `sg32` (ADR
  0132 decision 5), where the workgroup-storage limit holds it; the three-dispatch fallback happens when neither form fits
  (`attentionRoute`, `packages/runtime/src/runtime/attention-route.ts`). At D = 128 `sg32` declares 31,232 B and `wg`
  35,328 B (ADR 0132, the addendum of the v2 layout; `attentionOnlineI8a8WorkgroupStorageBytes`,
  `packages/runtime/src/kernels/attention-online-i8a8.ts`), so on 32 KiB devices such as D3D12 and Metal a device
  without subgroups always falls back at D = 128; on a device with a larger limit `wg` runs instead. The warning fires
  only on the three-dispatch fallback: `wg` replacing `sg32` is bit-identical and shows in the pipeline key only
  (`:sg32` / `:wg`).

**Device-driven choices.** A path, variant or geometry chosen from a device fact (a feature, a limit, a measured fact or an
adapter-keyed table) rather than from the model or the request follows one of three rules:

- (i) **the device lacks what a requested option needs**: warn once per device and switch toward the reference (the
  switch table above);
- (ii) **a choice between bit-identical forms**: silent, shown in the pipeline key, and backed by a check (decision 6);
- (iii) **a choice whose numbers differ**: warn once per device. The online attention's three-dispatch fallback is one;
  the attention dp4a canary's `emu-exact` verdict (dp4a misses the known answer, the emulation is exact) is another and
  warns once.

The device-driven sites that touch numerics or an option, as of 2026-11-09 (choices that move only memory layout or
residency, and throws on limits that no option relaxes, are left out; recording the full inventory of 2026-11-09 in
`docs/research/` is filed in the backlog):

| Site (symbol) | Device fact | Today | Under this ADR |
| --- | --- | --- | --- |
| `buildSessionState`, f16 check | no `shader-f16` | throw | (i): warn, run `"f32"` |
| `buildSessionState`, subgroup checks | no `subgroups` / `subgroup-size-control` / WGSL `subgroup_id` | throw for `rmsNormReduce: "subgroup32"` and `linearGemvReduce: "parallel-subgroup32"` | (i): warn, run `"workgroup"` / `"parallel"` |
| `assertGpuFeaturesGranted` (models) | a shared `GpuContext` without `shader-f16` | throw before any fetch | (i): warn at admission with the re-acquire text, switch |
| `a8EmulationWarning` | no `packed_4x8_integer_dot_product` | warn once, the emulation | (i): unchanged |
| `SessionState.linearI8a8Dot` (linear and conv3d a8) | dp4a in the enumeration | silent, by enumeration, no canary | (ii): key `:dp4a` / `:dp4aEmu`, plus a run-time canary |
| `decideAttentionI8a8Dot`, `emu-exact` | the canary's dp4a arm misses, the emulation is exact | silent, key only | (iii): warn once |
| `decideAttentionI8a8Dot`, inexact verdicts | the canary's arms are not exact | warn once | (iii): unchanged |
| `decideAttentionI8a8Dot`, both arms outside the band | the canary | throw `GpuFeatureError` | warn, run that attention in f32 (above) |
| `onlineReduce` | subgroups, `maxComputeWorkgroupStorageSize` | silent, `:sg32` / `:wg` in the key | (ii): plus a run-time canary |
| `attentionRoute`, online → three dispatches | neither online form fits `maxComputeWorkgroupStorageSize` | silent, keys only | (iii): warn once |
| `selectGeometryProfile` (via `sessionGeometryProfile`) | `adapterInfo`; the default table when nothing matches | silent, profile id in `SessionDiagnostics.geometryProfile` | (ii): the check is the sweep's `identicalToDefault` and `gpu_geometry_profile_test.ts` on the development GPU; an injected profile is to be verified |
| `rowBlockLimit` (fused, states-form and decomposed attention) | `maxStorageBufferBindingSize` | silent, the row-window bit in the key when there are two or more blocks | (ii): Uint32 tests of the enumerated variants |
| `gridStrideWorkgroups` | `maxComputeWorkgroupsPerDimension` | silent, not in the key | (ii): identity claimed, not checked (a `LIMIT_CAPS` Uint32 test is to be added) |

### 9. Per-GPU evidence — two layers

1. **Before a release.** The playground's `/kernel-check` page is extended to cover the fusions and the default kernels. It is
   run on each GPU at hand, and the results are recorded per (vendor, architecture, backend, browser). The backend is added to
   the tune report records (`packages/runtime/src/tune/report.ts` records `identicalToDefault` with the adapter fields and the
   user agent, no backend). MUST on the two everyday environments: Dawn / Vulkan on the RTX 3080 Ti and Chrome / D3D12 on the
   RTX 5070 Ti. The MUST reads the product verdict (`productVerdict` of the `/kernel-check` record,
   `apps/playground/src/models/kernel-check/verdict.ts`): a known FAIL outside the paths the product selects is recorded
   with its perf-ledger id and does not block (today the states-attention check on D3D12 in the e4b-full classes, K-95,
   as the release runbook's `/kernel-check` step states); a new FAIL blocks. Other GPUs (Apple M2, Intel, AMD, ARM Mali …) are run when at hand; a backend not run is recorded as not run
   and does not block the release. WebGPU exposes no backend field, and inferring it from the user agent is
   speculation-grade; how the backend field is filled (inference from the user agent, entry by the person running the page,
   or both) is an implementation-time item below. (Updated 2026-11-11: the page runs the built-in fusion twins — its entry
   "Fusion rules: fused vs unfused" runs `checkFusionTwins` of `packages/kernel-checks`, the twin cases the runtime's
   twin-graph tests run, from one implementation (372 cases in 13 parts, one per rule); a part the device cannot run reads
   as not run. So far these cases have run on Dawn / Vulkan only. The default kernels, the backend field and the evidence
   records are still to come.)
2. **At run time.** Every default-on E path is checked on every device by a canary at Session construction — the fusion
   rules that run without an option (seven of the ten in `FUSION_RULES` today) and the three of decision 5:
   the fused and the unfused form on the same input, compared as u32. On a mismatch the runtime warns and drops that fusion.
   The bit-identical choices made from device facts get a run-time canary on devices where the path is requested: the
   dp4a choice of linear and conv3d a8, and the online attention's `sg32` / `wg` choice (decision 6).
   Opt-in paths are usable on devices without evidence (the user chose them). A known-answer canary in the style of the dp4a
   one is used only where the known answer is cheap. There is no GPU-name allow-list.

Why two layers: the Session canary compares the fused with the unfused form on the same device, so it proves that the fusion
is safe, not that the base kernel is correct. Kernel correctness per backend comes only from the release-time runs.

Implementation-time items (not decided here): the canary's input size (the dp4a canary MUST fill at least one whole tile at
production geometry, the module doc of `packages/runtime/src/gpu/attention-dp4a-canary.ts`; the same likely applies —
speculation); the start-up cost (not
measured; if it is too slow the maintainer will consider countermeasures); where the evidence record lives; how the
backend field of the record is obtained.

### 10. Declared approximations in the only path ("approximate reference")

A path that approximates the upstream computation and has no alternative in the distribution is allowed at the reference
level when it is declared. **Each such approximation MUST record its oracle, its distance to the upstream computation, and
its reason.** Today there is one kind, the VAE tiled decode (an approximation of the untiled decode: the receptive field ends
at the tile). Diffusers itself offers tiled decode as a decode mode.

| Family | Oracle | Distance to the untiled upstream decode | Reason |
| --- | --- | --- | --- |
| Anima (ADR 0033, 0036, 0038 §4) | one tile Uint32-identical to the untiled path (0033 decision 4; the single-tile `assembleTiles` test of `packages/models/tests/anima_tiling_test.ts`); tile starts frozen in Python and TS (0033 decision 5 addendum); a torch implementation of the same geometry (`tools/export-recipes/anima/tiling.py`) | torch tiled vs untiled maxAbs 5.07e-2, three orders above the implementation error against torch (1.642e-5) — ADR 0033 measurements | memory: the 1024 px decode's intermediate peak 1,536.0 → 384.0 MiB at 2.39 × the time; above 2048 px the only way |
| Wan2.1 (ADR 0118 ruling 1) | tile plan frozen in Python and TS; one tile Uint32-exact; the same geometry implemented in the recipe as the reference (ADR 0118 decision 2), 832×480 × 33 tiled decode within the band 2e-5 (ratio 2.65e-6 / 3.15e-6, stage 5) | untiled difference 8.2e-2, an observation (ADR 0118, stage 5 row) | the full frame needs 3.52 GiB of cache input and as much output, expected not to fit the B570 (speculation recorded there) |
| Wan2.2 (ADR 0121 decision 6) | tile starts frozen in TS (`AXIS_STARTS_TI2V`) and Python (`MIRRORED_STARTS_TI2V`); one tile Uint32-identical; Karume's Python tile reference (`vae_tiling.py --model ti2v-5b`), band 5.0e-5 | **not recorded — to be added** | tile 32 would cost 5.2 GiB of cache alone; tile 16 / overlap 4 equals the diffusers 2.2 default |

- The tiled decode is kept as it is. An untiled reference path is considered only if a problem appears; the distribution can
  add an untiled decoder later (ADR 0038 §4).
- The 1.2e-1 in ADR 0121 (its addendum of 2026-10-04, stage 1 results) is a CPU f32 / f64 ratio with random text, not
  Wan2.2's untiled distance. Recording that distance is filed in the backlog.

### 11. Retained rules

Carried from ADR 0058 and ADR 0110 and still in force:

- **In-device determinism (MUST, practical and opt-in paths included).** The same environment, preset and input in two
  independent Sessions give byte-identical output. MUST NOT: floating-point atomics, merges in arrival order, or a reduction
  order that changes with a runtime parameter absent from the key (such as the subgroup width). A host-side adaptive policy
  (a path chosen by wall clock or load) is allowed only when the selectable paths are machine-verified on that device to be
  bit-identical under the preset; gemma's self-tuning gate meets this by an e2e check per device, and where it is red that
  machine selects `"always"` or sequential. Path equivalence (with speculation = without, and the like) is declared per
  preset.
- **sha256 reference rows per environment** (ADR 0106). The class is derived from whether the case's preset has an empty
  `session` (no separate table). Rows of presets with an empty `session` are frozen; a commit that changes a numerics option
  (a non-empty `session` value) MUST NOT touch them. Rows of a preset with a non-empty `session` detect regressions and
  nondeterminism of that preset; they are not evidence that the reference path did not move (decision 3 (1)). A commit
  that changes those numerics on purpose runs `rewrite` on them in the same commit and records the reason and the green
  checks of decision 2; such rows of environments not rewritten in that commit are deleted and rebuilt on the next run there
  (explicit SKIP + red reference guard → self A/B → `write`). Whether Anima's rows, recorded on its practical preset, move
  to its reference preset is not decided by this ADR.
- **Classification of existing options**: the table in decision 2 (as of 2026-11-09), with `docs/quantization.md` Table
  2-b as its living copy; the grace clause is in decision 2.
- **Composition of the bulk request** ("use every opt-in speed-up", 2026-08-13): one rule for every family (decision 6); a
  caller that pins its sources gets a stable meaning across versions, a caller that follows `main` does not; band checks
  exist per knob and per distributed bundle, never for every combination.
- **Changing a published preset**: a published preset's `session` changes only through a new distribution revision that
  a new package release pins (ADR 0073). Published package versions keep reading their old pin, so what a given version
  runs under a preset name never changes. A change that keeps the numbers (such as decision 5's key removal) needs a
  release note only. A change that moves the numbers under the same preset name is stated in the CHANGELOG, and the
  preset's E2E self A/B check (decision 2) is redone.

### Where the old clauses went

Pointers to ADR 0058 and ADR 0110 in other ADRs (for example 0040 decision 2, 0050, 0105, 0132 decision 5) stay as
history and resolve through this map. Every clause of the two ADRs is listed; "not restated" means the clause is history,
a record or withdrawn, and no rule of it remains.

| Old clause | Here |
| --- | --- |
| 0058 Context | decision 3 (1) (the background policy of 2026-08-13: bit-identity checks indicate a correct implementation, they are not the goal); the scope description is not restated (history) |
| 0058 decision 1 | decision 6 (string-union API rule; booleans only for a certain on / off) |
| 0058 decision 2, addendum 2026-09-26 (decision 2 reading) | decision 6; sha rows decision 11 |
| 0058 decision 3 | decision 8 (shape fallback, device-capability switch, device-driven choices) |
| 0058 decision 4 ①; ② ③ | decision 3 (1); decision 2 (four-point check, bands, census, E2E self A/B). The rule of ② that the band's derivation table sits in the docstring of the test constant is not restated (0110 did not restate it either; the practice continues) |
| 0058 decisions 5, 6, 7 | decision 3 (who decides: admission machine-only, presets and the default by the maintainer); decision 11 (rows per preset and environment); decision 7 (combinations derived from the graph) |
| 0058 decision 8 | not restated (withdrawn 2026-10-18) |
| 0058 addendum 2026-08-29 | decision 6 (the per-family dp4a choice and `I8A8_DOT`, the dp4a canary, automatic selection claiming bit identity); decision 8 (both arms outside); decisions 6 and 9 (a run-time canary for linear's and conv3d's dp4a choice). The canary's history on the M2 is not restated (also in `docs/known-issues.md` and the canary's module doc) |
| 0058 addendum 2026-09-06 | decision 6 (an option declares only that a changed path is selected); its Gemma pipeline history is not restated |
| 0058 addendum 2026-09-13 | decision 3 (2), narrowed to evaluation and proposal; its same-weights bullet is ADR 0128 C2 (cited there) |
| 0058 addendum 2026-09-26 (decision 2 reading, decision 4 ② ③, determinism, bulk flag) | decision 11 (sha row classes, determinism, the composition of the bulk request, changing a published preset); decision 2 (census); decision 6 (one composition rule, no bulk axis) |
| 0058 addendum 2026-10-10 | decision 2 (census, `attentionRoute`, completeness as a ratio); the remainder on the declared route is a backlog item |
| 0110 Context | not restated (history; the reason model-exit bands go vacuous is in decision 2) |
| 0110 decision 1 | decision 1 (levels); decision 6 (only source of a bundle, no value table in models, fixed at construction); its "an acceptance table is allowed" is replaced by decision 7 |
| 0110 decisions 2–5 | decision 2 (Q now has a member) |
| 0110 decisions 6, 7 | decision 11; the reading of storage quantization against the levels is decision 1 |
| 0110 decision 8 | decision 3 (2) (every preset, not only on entering `defaultQuant`; the form of a human review; `fast` bundles); "never change the omitted values" is decision 6 |
| 0110 decision 9 | decision 2 (classification, grace clause); its fusion row is decision 5, `packedStaticQuantize` decision 2 |
| 0110 alternatives 1, 4 | decision 2 (the E2E self A/B check is not a numerics contract; a torch golden or another preset's f32 is not a valid control) |
| 0110 alternative 2 | not restated (moot: this ADR is the consolidation) |
| 0110 alternative 3 | decision 11 (moving Anima's rows to its reference preset is not decided here; backlog) |
| 0110 Consequences | not restated (history) |
| 0110 addendum 2026-09-26 | decision 2 (implementation form of the E2E self A/B check; its scope, with gemma under the grace clause); the B570 derivation of the ceilings is a record (the table in `e2e_anima_ab_test.ts`) |
| 0110 addendum 2026-11-04 | not restated (a record of one human review, an instance of decision 3 (2)) |

## Alternatives considered

1. **Gate fusions and kernels on a per-GPU evidence table** (a generated table per adapter class, as geometry profiles) —
   rejected: per-class evidence is not a check on that device; WebGPU exposes no backend; enabling an R or Q path by table
   would select a numerics-changing path from adapter capabilities.
2. **Canary only, no release-time runs per backend** — rejected: the canary proves the fusion against the unfused form on the
   same device, not the base kernel (decision 9).
3. **Always throw when an option cannot be used** — rejected: when the adapter lacks a feature the user cannot fix it, and a
   switch toward the reference with a warning and a pipeline key keeps the difference visible.
4. **Narrow the overrides' TypeScript type per family** — rejected: a Hugging Face revision can change the graph, so a type
   fixed at build time can be wrong for the revision a user loads; the query API answers per revision.
5. **An untiled reference VAE path** — rejected for now: the memory figures of decision 10, and diffusers itself offers the
   tiled mode; reconsidered only if a problem appears.
6. **Keep the hand-written acceptance tables** — rejected: they mix "no place in the graph" with "not verified", they drift
   from the code (ADR 0111 decision 2's list is stale), and the structurally impossible combinations they do not cover throw
   only after the upload.
7. **Define the reference level as the original weights** — rejected: several families would have no reference artifact
   (decision 1).
8. **Keep model-exit torch parity as the proof of correctness** — rejected: it cannot catch an upstream definition bug, and
   it is retired only as replacement checks are recorded (decision 4).
9. **Use worst-case forward-error bounds as op pass lines** — rejected: at least 60 × looser than measured (check C of
   decision 4).
10. **Keep bit-identical fusions as boolean opt-ins** — rejected (decision 5); bisecting a fusion moves to the test-only
    switch of decision 5.
11. **Make `packedStaticQuantize` default-on now** — rejected: it saturates NaN (decision 2).
12. **No required quality review for a preset, or a human review only where an LLM cannot judge** — rejected (ruling of
    2026-11-09): decision 3 (2) requires a human review of every preset and allows a skip only on the maintainer's prior
    approval, never for the default preset. A review only on entering `defaultQuant` (ADR 0110 decision 8) is widened to
    every preset for the same reason.
13. **Refuse, in every family, a manifest whose `session` differs from a checked-in table** (ADR 0138 decision 13's
    form, generalised) — not adopted now (ruling of 2026-11-09). It is the strongest protection and the only one that
    covers a manifest the caller supplies, but a preset's knobs get two owners and every new preset needs a package
    release. It is reconsidered when a concrete problem appears.

## Consequences

- The reference path does not move by one bit: every reference-level sha256 row is the regression check of the default-on
  switch of decision 5. The fusion-count tests and the census rows that pin fused counts (for example Gemma 4 E2B's
  `fuseRmsNormAdd` counts in `CENSUS_TABLE`, `packages/models/tests/helpers/census-table.ts`) change with it. Presets that
  never declared the keys also take the fusions: the linear→SRQ rule applies to every preset on the parallel / f32 path (for
  example the mobile QAT `i4-gemvpar`, `session` `{ linearGemvReduce: "parallel", stateAttentionReduce: "parallel" }` in the
  local mirror; the gate is the rule's first match condition in
  `packages/runtime/src/runtime/fusion-rules/linear-static-quantize.ts`), and the
  rmsNormAdd rule to every matching graph. Their fusion-count and census rows change; their sha rows do not.
- Breaking changes: `SessionOptions` and the manifest `SessionSpec` lose three keys; a new reader rejects an old manifest
  that declares them, so the affected distributions are re-uploaded and the pins updated. An old reader that reads a
  re-uploaded manifest runs without those fusions (speed only).
- Session construction gains the Session canary and the canaries of the device-driven bit-identical choices (decision 9;
  cost not measured); admission gains the applicability check over part 0 (no
  weight is read).
- The runtime gains a public query API (decision 7).
- Human judgement is limited to presets, the default preset, model intake and the three manual steps of decision 3 (5).
- The amended clauses (the header's "Amends in part") carry a status line, and the ones whose rule changed carry the new
  text in place with a dated pointer to this ADR. The code changes, the new checks and the records this ADR asks for are
  filed in the [backlog](../backlog.md).

## Open questions

None: every point was ruled on 2026-11-09. Items left for implementation time are listed where they arise (decision 4, pass
criteria of checks A and G; decision 9, implementation-time items).
