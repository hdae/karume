# 0110: Numerics contract of the practical presets — reference path and practical presets, contract classes E / C / R / Q, in-device determinism, classes of sha256 reference values

- Status: accepted (2026-09-26 — the maintainer's ruling: split the numerics into a slow, exact
  reference path and fast practical presets; in-device determinism is MUST for the practical presets
  too; the plan of phase 1 "contract and groundwork" approved)
- Date: 2026-09-26
- Related: [0058](0058-numerics-opt-in-contract.md) (numerics opt-in contract, restated here for the
  two paths) / [0106](0106-device-keyed-references.md) (per-environment sha256 reference values) /
  [0074](0074-quant-seat-naming.md) (the `fast` token) / [0104](0104-gemma-fast-quant.md) (gemma
  `i4-fast`) / [0025](0025-w8a8-linear-execution.md), [0026](0026-w8a8-deberta-deployment.md),
  [0030](0030-attention-a8-execution.md), [0031](0031-attention-score-f16-storage.md) (E2E bands
  that went vacuous) / [0096](0096-speculative-decoding.md) (self-tuning gate, decision 6) /
  [0126](0126-numerics-profiles.md) (numerics profiles; it planned to amend decisions 1, 7 and 8 but
  was deferred on 2026-10-18, so they stand unchanged). Notes: `.claude/reviews/2026-09-25_perf-recon/`.

## Context

Karume's "bit-identical" is two claims. (a) Activation-quantized and composed variants (w8a8 / a8 /
c16 / s16 …) match the project's own TS reference implementation (the oracle) with atol = 0; the
reference path of the GEMM / conv family with i8 / f16 / i4 storage and f32 compute is checked by the
op band against f64 (`GEMM_TOLERANCE`), with bit-exact checks only for dequantization, "resident ≡ CPU
dequantization" and "GEMV ≡ GEMM". (b) The output sha256 for the same input on the same machine does
not move (ADR 0106). Torch golden comparisons always had a tolerance (`GOLDEN_TOLERANCE` in
`packages/runtime/tests/e2e_golden_test.ts`).

Numerics-changing speed-ups entered under ADR 0058 (one named option each, the reference path as
default, a three-part check). The review of 2026-09-25 found:

- **The two paths half exist.** Reference path = the runtime's omitted-value path + strict oracles +
  sha256 regression checks + WGSL snapshots. Practical presets = opt-in paths bundled by a preset's
  `session` (gemma `i4-fast`, anima `f16+dit8-a8-attn8-s16`; the `fast` token of ADR 0074).
- **No admission contract for practical presets.** ADR 0058 decision 4 ② derives an A/B band from
  measured drift × margin, but says nothing about bands **wider than the reference path's** and has
  no defence against vacuous checks. Three options were admitted on a band
  (`stateAttentionReduce: "parallel"`, `linearGemvReduce: "parallel"`, two subgroup reductions), all
  reordering additions; only `stateAttentionReduce` has all four points of decision 3.
- **Model-exit bands go vacuous** (any implementation passes): anima w8a8 (1.6 × measured, ADR 0025
  decision 6), DeBERTa (layer 1 10,058× / layer 24 1.46 vs 1.33, ADR 0026), anima attention a8
  (1.35×, ADR 0030), s16 (1.16×, ADR 0031). Activation quantization is discontinuous, so 1e-5
  upstream differences saturate within a few layers; stacked opt-ins remove later detectors; the
  control (a torch fake-quant reference or another f32 path) has its own implementation error.
- **10 of anima's 11 sha256 cases were recorded on a practical preset** (since 2026-08-05), against
  the letter of ADR 0058 decision 2, yet they were that preset's **only E2E detector**.
- **gemma's speculative decoding switches M=1 / M=4 by wall clock** (ADR 0096 decision 8); on the
  B570 (2026-09-26) both paths matched bit for bit (ADR 0096 addendum of 2026-09-26).

## Decision

### 1. The two paths

- **Reference path** = the runtime's omitted-value path. The omitted values do not change (ADR 0058
  decision 2 is kept).
- **Practical presets** = the opt-in paths bundled by a preset's `session`. The only source of a
  bundle is the manifest preset; `models` keeps no table of default **values** (an acceptance table
  is allowed — ADR 0111).
- The path is fixed at Session construction (`SessionOptions` are static construction-time options)
  and does not change within a Session.

### 2. Contract classes (a variant in a practical preset declares one, strongest first)

| Class                  | Claim                                                                      | Form of the check                      | Current members (decision 9)                           |
| ---------------------- | -------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------ |
| **E (mirror)**         | atol = 0 against the TS reference implementation                           | exact comparison                       | w8a8, attention a8 ① QK, fused vs unfused u32 equality |
| **C (composition)**    | atol = 0 against "host-rounded input + the existing f32 kernel"            | exact comparison                       | c16, s16                                               |
| **R (addition order)** | only the addition order differs; the f64 band **is the reference op band** | four points (decision 3)               | parallel GEMV, state attention parallel, subgroup      |
| **Q (approximation)**  | a band **wider** than the reference path's op band                         | four points + error model (decision 4) | none (reserved)                                        |

- C's atol = 0 relies on f32 → f16 narrowing rounding to nearest even (RTE), implementation-defined
  in WGSL; a C check holds after a per-device RTE check (ADR 0028 / 0031). s16 is the exception: on D3D12 it
  rounds toward zero, so its host rounding is the backend's own (ADR 0031 addendum 2026-11-07).
- MUST: before choosing a weaker class, the ADR states why a stronger class cannot be built (the first
  defence against vacuous checks). IEEE-exact operations (add, subtract, multiply, explicit `fma`)
  can probably give E with a TS mirror; built-in transcendentals (`exp` …) cannot (driver
  differences), but a mirror with a statistical check (qP, ADR 0030) is a precedent.

### 3. The four-point kernel check (MUST for R and Q)

Generalised from `packages/runtime/tests/gpu_state_attention_parallel_test.ts` (lines 1–18):

1. The band against f64 (R: the reference op band; Q: the error model's upper bound).
2. The A/B band against the reference path (the same input through both).
3. Exact checks of structural guarantees (special values, exact 0 on pad rows, the −inf bit pattern,
   NaN propagation, geometry invariance, two dispatches of the same input bit-identical).
4. A record of a fault injection that turns ① or ② red (the check is not vacuous).

The census (the variant runs when selected) is a separate check (ADR 0058 decision 4 ③).

### 4. Band derivation rules

- R: f64 band = the reference op band (same class in `packages/runtime/tests/helpers/op-tolerance.ts`);
  A/B band ≥ 4 × the worst measured and ≤ the theoretical bound (2 × the f64 band).
- Q: f64 band = the error model's bound; a perturbation corpus shows at most a quarter of it is used.
- A band is a **declaration**, not a per-environment value; the writing modes of `KARUME_REFERENCE`
  do not rewrite it (ADR 0106 alternative 4 stays rejected). Only a Q band shown to need a
  per-environment relaxation takes the form of the ADR 0106 addendum decision 1 (that machine only).
- MUST NOT: derive an E2E band as "5–10 × the measured" (ADR 0025 decision 6, generalised).

### 5. The E2E A/B check (not a numerics contract)

A model-exit band is not a numerics contract. E2E carries only these five items, against **the
reference path on the same machine** (not torch goldens):

1. **Census MUST**: variant keys per option of the bundle; fully eligible ⇒ 0 reference-variant keys.
2. **Difference floor MUST**: the output differs by at least one bit from the preset with **the same
   weights (same storage dtype) and an empty `session`** on the same machine and input (detects a
   silent fallback of the whole bundle). For anima's `f16+dit8-a8-attn8-s16` the control is
   `f16+dit8` (`f16` has other weights and would make the floor vacuous). The same-machine reference
   path is deterministic, so the floor can always be set; fallback of a single option is item ①.
3. **Collapse ceiling MUST**: relative RMS at the observation point under the ceiling, no non-finite
   values. Ceiling ≈ 2 × the measured value (precedents: 1.6× in ADR 0028, 2.0× for irodori); for a
   quantized preset the docstring records agreement with the theoretical value.
4. **The observation point MUST sit right after the preset first takes effect** (DiT = the latent
   after step 1; encoder = layer 1 output; LLM = the first logits). Final outputs (PNG / WAV / tokens)
   are recorded but carry no band.
5. Each check's docstring records at least one fault injection that turned it red.

The implementation form (result names, sequential pipelines, band constants) belongs to the ADR that
implements the check. Sharing weights between two Sessions through `sharedWeights` is not used,
because the distribution containers carry no `shared` declaration.

### 6. In-device determinism MUST (practical presets included)

The same environment, preset and input in two independent Sessions give byte-identical output.

- MUST NOT: floating-point atomics, merges in arrival order (last-arriver merge), or a reduction order
  that changes with a runtime parameter absent from the key (such as the subgroup width).
- **Host-side adaptive policies** (path chosen by wall clock or load) are allowed only when the
  selectable paths are **machine-verified on that device** to be bit-identical under the preset
  (extends rule 3 of the ADR 0058 addendum of 2026-08-29). gemma's self-tuning gate meets this by an
  e2e check per device; where it is red, that machine selects `"always"` or sequential.
- Path equivalence (with speculation = without, and the like) is declared per preset, separately.

### 7. Classes of sha256 reference values (an addition to ADR 0106)

- The class is **derived** from whether the case's preset has an empty `session` (no separate table).
- **Reference-path values** (empty `session`) are frozen; a commit that changes the practical presets
  MUST NOT touch them (the letter of ADR 0058 decision 2).
- **Practical-preset values** (non-empty `session`) detect regressions and nondeterminism of the
  practical presets; they are not evidence for ADR 0058 decision 4 ①.
- A commit that intentionally changes practical-preset numerics runs `rewrite` on those values **in
  the same commit** and records the reason and the green checks of decisions 3 and 5.
- Practical-preset values of environments not rewritten in that commit are **deleted**. The next run
  on that machine rebuilds them: explicit SKIP + red reference guard → self A/B → `write`.
- Inventory of 2026-09-26: ADR 0106 addendum (the third); anima's current 13 cases are in
  `packages/models/tests/fixtures/references/anima.json`.

### 8. Division of the quality verdict

The machine checks (decisions 2–7) show only "safe as an opt-in". A quality verdict is made only when
a preset enters the family's `defaultQuant` (procedure of the ADR 0058 addendum of 2026-09-13: visual
review with at least 4 seeds × 2–3 prompts, listening, for LLMs a fixed evaluation plus reading the
output; a human review not done is never recorded as done). Which options a `fast` bundle holds is
chosen by speed, measuring each option alone and all together on the same machine (ADR 0104
addendum). The practical presets never change the runtime's omitted values (ADR 0058 decision 6).

### 9. Classification of existing options

| Option                                                                 | Class                                | Four points                                         | Source                                       |
| ---------------------------------------------------------------------- | ------------------------------------ | --------------------------------------------------- | -------------------------------------------- |
| `linearCompute: "a8"`                                                  | E                                    | —                                                   | ADR 0025                                     |
| `attentionCompute: "a8"`                                               | E (statistical check for qP of ③ PV) | —                                                   | ADR 0030                                     |
| `linearCompute` / `attentionCompute: "f16"`                            | C                                    | —                                                   | ADR 0028                                     |
| `attentionScoreStorage: "f16"`                                         | C                                    | —                                                   | ADR 0031                                     |
| `stateAttentionReduce: "parallel"`                                     | R                                    | ① ② ③ ④                                             | `gpu_state_attention_parallel_test.ts`       |
| `linearGemvReduce: "parallel"`                                         | R                                    | ① ② ③ (no ④ fault injection, no worst-case record) | `gpu_linear_gemv_parallel_test.ts`, ADR 0098 |
| `rmsNormReduce` / `linearGemvReduce` subgroup32                        | R                                    | ① ③ + u32 equality with parallel (no ② ④)           | ADR 0100, 0101                               |
| `fuseRmsNormAdd` / `fuseLinearStaticQuantize` / `packedStaticQuantize` | E (the unfused path is the mirror)   | —                                                   | ADR 0099, 0103, 0105                         |

**Grace clause**: existing R options stay admitted with the missing points (④ of parallel GEMV; ② ④
of subgroup) filed in the backlog. New R / Q options are admitted only with all four points.

## Alternatives considered

1. **Admit options on an E2E band alone** — rejected: model-exit bands went vacuous (Context).
2. **Only add addenda to ADR 0058** — rejected: unreadable at its fifth addendum; 0058 / 0106 get
   one-line pointers.
3. **Move anima's sha256 values to the reference preset** — rejected (S1): it removes the practical
   preset's only E2E detector; to be reconsidered once the A/B check and the census exist (ADR 0106
   addendum, option (b)) — both exist today (`e2e_anima_ab_test.ts`, `census-table.ts`), so the
   question is open again.
4. **Use a torch golden or another preset's f32 as the floor's control** — rejected: the control's
   implementation error shrinks the separation ratio.

## Consequences

- The reference path does not move by one bit; WGSL of existing keys stays byte-identical.
- Approximate kernels (Q) have an entry, at the cost of error model + four points + fault injection.
- Each change to practical-preset numerics rewrites anima's practical-preset values per machine.
- The missing E2E census and A/B check of anima's practical preset surfaced as unmet decision 5; they
  now exist (`packages/e2e/tests/e2e_anima_ab_test.ts`, shared verdict in `e2e_ab_gate_test.ts`).
- gemma's self-tuning gate conforms to decision 6 (ADR 0096 addendum of 2026-09-26).

## Addendum (2026-09-26) — implementation form of decision 5 and first measurements

- One result name per family (`anima-ab`, `irodori-ab`, `sbv2-ab`); three sequential pipelines, no
  `sharedWeights`. The reference partner preset is derived from the manifest (same weights mapping,
  empty `session`, GPU feature requirements a subset, exactly one).
- The floor has two levels: "bit-identical to the reference → red" (decision 5 ②, always checked) and
  `band.floor` (above 0 only when measured). A check without a derived band prints its measurement
  and stops red. Ceilings = 2 × the first B570 measurements (2026-09-26; derivation table in
  `packages/e2e/tests/e2e_anima_ab_test.ts`); removing `attentionCompute` gave 0.90 × the full
  bundle, so the floor stays 0 and single options are left to the census.
- sbv2's observation points are the four front-graph outputs (the waveform length may change).
- The census source of decision 5 ① is `packages/models/tests/helpers/census-table.ts` (key =
  effective SessionOptions; value = expected counts per component × phase and the reference-path
  remainder); `packages/models/tests/census_table_test.ts` turns red when a non-reference bundle of
  the local mirror has no row. Presets without a row (prefill parallel GEMV / attention,
  parallel-fused, anima's attention presets) keep the "at least one / zero" checks.
- gemma (whose `i4-gemvpar` / `i4-fast` token ids are its practical-preset detectors) and the four
  f32-only families have no same-machine A/B check yet.

## Addendum (2026-11-04) — decision 8's human quality review of the q4_0 QAT Gemma 4 `i4` presets

The maintainer reviewed the `i4` presets of the q4_0 QAT line (`karume-gemma4-qat`: E2B, E4B and 12B
— Google's q4_0 quantization-aware-trained checkpoints on their own 4-bit grid, ADR
[0128](0128-default-preset-original-weights.md) addenda of 2026-11-03 and 2026-11-04) and passed them
on 2026-11-04. This is the quality verdict decision 8 asks for before a preset enters
`defaultQuant`: `i4` is the default of all three models. The same day the regular line
(`karume-gemma4`) dropped its own int4 presets and kept `bf16` alone (ADR 0128, second addendum of
2026-11-04), so the q4_0 `i4` presets are now the int4 option that `gemma.Pipeline` reads for E2B / E4B. Publication is
still to come; this addendum records the review only.
