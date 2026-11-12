# 0115: GEMM geometry selection becomes a static shape × adapter table (geometry profiles)

- Status: accepted (2026-09-27 — the maintainer's ruling after the M2 quick sweep showed the winning
  geometry reversing between GPUs); partly superseded: decision 2's selection order by addendum
  decision 7 (2026-09-30); decision 4's input check by [0117](0117-app-geometry-tuning.md) decision 3
  (2026-10-01), its three-bucket boundaries by [0116](0116-geometry-profile-row-buckets.md)
  decisions 1 / 3 (2026-10-01) and its disregard of drift by addendum decision 8 (2026-09-30);
  decision 8's "no entry point for users" by addendum decision 6 (2026-09-29) and 0117 decision 6
  (2026-10-01); the `tools/gpu-lab` tool of addendum 7 by [0127](0127-node-dawn-toolchain.md)
  decision 5 (2026-10-16); the Consequences' deferral of "drop a8 from the default quant on Metal" is
  taken up by [0128](0128-default-preset-original-weights.md) decision 5 (accepted 2026-10-29), which
  settles it. Re-graded 2026-10-18: decision 4's regeneration command and `--check`
  to SHOULD. Decisions 1 and 6's single `i8a8.linear` geometry for every row count by addendum
  decision 14 (2026-11-12).
- Date: 2026-09-27
- Related: [0022](0022-gemm-register-blocking.md) (GEMM skeleton; decision 3's MUST "the reduction
  order is a numerical contract"; the addendum's MUSTs "no run-time autotuning" and "a change of the
  default geometry goes together with re-running the sha256 checks", and its rule that per-backend
  static defaults are considered only after a clear measured regression) /
  [0023](0023-fused-attention.md) (fused attention ①QK / ③PV) / [0024](0024-conv2d-implicit-gemm.md)
  (conv implicit GEMM) / [0060](0060-row-block-attention.md) (decomposed attention in row blocks) /
  [0067](0067-autoregressive-attention-vocabulary.md) (states-shaped attention) /
  [0106](0106-device-keyed-references.md) (sha256 reference values per environment key) /
  [0058](0058-numerics-opt-in-contract.md) (the i8a8 attention variant is chosen by a run-time
  canary) / perf-ledger K-70, K-71, K-67, K-22. Research:
  [research 2026-09-27 K-70](../research/2026-09-27-k70-metal-per-op.md) §5–§10.
- Implementation: `packages/runtime/src/kernels/geometry-profile.ts` (types, default profile, pure
  selection functions) and `packages/runtime/src/kernels/geometry-profiles/` (generated profiles and
  the list `BUILTIN_GEOMETRY_PROFILES`); the sweep and the generator live in
  `packages/runtime/src/tune/` (ADR 0117).

## Context

On Apple M2 (Chrome 153 / Metal) the anima DiT ran 10 to 30 times slower than on Arc B570. Per-op
measurements (research §4–§6) showed the slowdown was not uniform:

| Group                                                      |  M2 / B570 | Excess over the nominal ratio (compute 3.2–4.0×, bandwidth 3.8×) |
| ---------------------------------------------------------- | ---------: | ---------------------------------------------------------------: |
| Bandwidth-bound small kernels (quantize_rows, rms_norm, …) |    3.6–4.7 |                                                          0.9–1.2 |
| f32 mid-tile linear (`reg64x32`)                           |        4.4 |                                                          1.1–1.4 |
| f32 large-tile linear (`reg128x128`)                       |       10.2 |                                                          2.6–3.2 |
| f32 attention ①QK / ③PV (large tile)                       | 13.5 / 9.8 |                                                          2.4–4.2 |
| f32 conv2d implicit GEMM (VAE)                             |    6.0–8.8 |                                                          1.5–2.8 |
| i8a8 linear / attention (dp4a)                             |  25.0–33.8 |                                                         6.3–10.6 |

Among f32 GEMMs the mid tile matched the nominal ratio and only the large tile (128×128, 64
accumulators per thread, 256 threads) fell off. A runtime difference (Dawn / Metal vs wgpu / Vulkan)
would hit every key alike, so **the main cause is that the large-tile geometry does not suit the M2**
(research §6). i8a8 has a separate cause (Tint expands `dot4I8Packed`) that geometry does not fix.

A quick sweep on the M2 (same shape, only the geometry varied, timed with GPU timestamps; research
§9, five op families, 33 cases, 177 rows):

| op                      | Best on M2                 | M2 vs default | Same geometry on B570 |
| ----------------------- | -------------------------- | ------------: | --------------------: |
| f32 linear (M ≥ 1024)   | `reg64x32r4x4w8`           |    ×1.58–1.60 |            ×0.72–0.98 |
| f32 attention ①QK       | `reg64x32r4x4w8`           |    ×1.55–1.60 |        (not measured) |
| f32 attention ③PV       | `reg64x64r8x4w16`          |    ×1.66–1.67 |        (not measured) |
| f32 conv2d              | `igemm64x64` (wg16x8 / 16) |    ×1.37–1.48 |        (not measured) |
| i8a8 linear / attention | `tile64x64r8x4w16x8k16`    |    ×1.06–1.13 |            ×0.64–0.84 |

- **The winning geometry reverses between GPUs.** On B570 the default was fastest for large-M f32
  and i8a8 linear.
- **Output sha256 matched the default on all 177 rows** (and on all 77 rows of the B570 quick
  sweep): geometry does not move bits for f32 or i8a8. For f32 this is an empirical claim, not a
  theorem (ADR 0022 addendum); the M2 joins its data points.
- The default re-measured at the end of each case stayed within 0.99–1.01, so machine drift did
  not affect the ratios.

ADR 0022 allowed per-backend static defaults only after a clear measured regression, with the ban
on run-time autotuning unchanged. The table meets that condition: the common default leaves 1.4 to
1.7× unused on the M2, and the M2 geometry regresses on B570. One default loses on one of them.

## Decision

### 1. A profile is a static table of op × bucket

Geometry selection widens from a pure function of shape to a pure function of shape × profile. A
profile is plain data, defined in `packages/runtime/src/kernels/geometry-profile.ts` (no side
effects). Current type, abridged (`match.description` came with addendum decision 7 and the
structured `provenance` with ADR 0117 decision 4 — geometry-profile.ts:56):

```ts
export type GemmRowsRule = { readonly maxRows: number; readonly geometry: GemmGeometry };
export type GeometryProfile = {
  readonly id: string;
  readonly match?: {
    readonly vendor?: string;
    readonly architecture?: string;
    readonly description?: string;
  };
  readonly gemmRows: readonly GemmRowsRule[];
  readonly attention: { readonly qk: GemmGeometry; readonly pv: GemmGeometry };
  readonly conv2d: { readonly rows64: GemmGeometry; readonly rows32: GemmGeometry };
  readonly i8a8: {
    readonly linear: readonly I8a8RowsRule[]; // addendum decision 14 (2026-11-12)
    readonly attentionQk: I8a8Geometry;
    readonly attentionPv: I8a8Geometry;
    readonly conv3d: I8a8Geometry; // addendum decision 11 (2026-10-26)
  };
  readonly provenance?: {/* sweep records, sha256, dates, adapter, kernels, caseSet, … */};
};
```

- `id` names the profile and matches the generated file name (default = `"default"`, M2 =
  `"apple-metal-3"`).
- `gemmRows` is the row-count bucket table: the first rule with `rows <= maxRows` applies; `maxRows`
  ascends and the last rule is `Number.POSITIVE_INFINITY`. It is read in one place,
  `gemmRowsGeometry(profile, rows)`.
- `conv2d.rows64` / `rows32` give the geometry per m-tile class of the implicit GEMM (64 or 32 rows,
  as returned by `conv2dIgemmMTile`). **The class is still decided by the predicate
  `conv2dIgemmMTile`**; the actual tile edges come from the geometry (`gemmTileM` / `gemmTileN` —
  the gemm-geometry.ts MUST that the geometry is the source of truth for tile edges). A 32-row class
  may therefore get a geometry whose tileM is not 32 (`apple-metal-3` gives the Cout = 96 class
  `igemm128x64:wg16x16`, tileM = 128).
- The storage dtype (f32 / f16 / i8 weights) is not distinguished, as in the default table.
- Only profiles generated from a sweep carry `provenance` (decision 4).

### 2. Selection: one profile per adapter, in a deterministic order

(Superseded by addendum decision 7 on 2026-09-30: matching also uses `description`, and a table
without `match` is injection-only.) `selectGeometryProfile(adapter, profiles =
BUILTIN_GEOMETRY_PROFILES)` picks once at plan time from the `vendor` / `architecture` of
`GpuContext.adapterInfo`, by exact string comparison (no normalisation, no prefix match):

1. the profile whose `match` vendor and architecture both match;
2. otherwise a profile with `match.architecture` unset and the vendor matching;
3. otherwise `DEFAULT_GEOMETRY_PROFILE`.

Two profiles at the same rank fail loudly with `CodegenError`; picking by list order would let an
edit of the list change the selection.

**No similarity matching** (nearest GPU name, core count, …). If the chosen profile varied from
machine to machine, the per-environment-key sha256 rows (ADR 0106) could no longer be tied to the
geometry the machine ran. With exact matching, the selection follows from the same `adapterInfo`
that builds the environment key.

### 3. Embedding: generated TS modules are committed

A profile lives in `packages/runtime/src/kernels/geometry-profiles/<id>.ts` as
`export const <ID_UPPER>: GeometryProfile` and is listed in `BUILTIN_GEOMETRY_PROFILES` of
`geometry-profiles/index.ts` (the generated profiles, without the default).

- **Nothing is read from outside at run time** (no JSON fetch, environment variable or file read).
  This keeps the Web-standard-only runtime dependency rule and barrel tree-shaking (no side effects).
- Generated modules do no registration at top level (constant exports only). The list is a single
  array in `index.ts`.

### 4. Generation: only geometries that are reliably faster become rules

The generator (now `deriveGeometryProfile` in `packages/runtime/src/tune/derive.ts`, ADR 0117) takes
one or more `karume-geometry-sweep/2` records and combines them into one profile.

- **Input check** (superseded by ADR 0117 decision 3 on 2026-10-01: each observation is judged by an
  upper bound on its rounding error, at most 1% of the ratio, instead of rejecting every record
  flagged `quantized`). Only GPU-timestamp sweeps are accepted (`gpuTiming.unit` `ns` or
  `deno-raw-tick` — `packages/runtime/src/tune/sweep.ts:138`); records without `gpuTiming` and
  wall-clock records are rejected (unchanged).
- **Combination**: all records come from the same adapter (otherwise generation stops), and the
  same record (same sha256) given twice also stops it. When several records measure the same
  (case, geometry), the ratio is the geometric mean of the observations; output match and failures
  are checked over all observations.
- **Default row check**: if a record's default row (`isDefault`) differs from the current runtime
  default (the per-field fallback of decision 5), generation stops. Ratios are relative to the
  default row, so a record on a shifted base cannot show "faster than the default" for the current
  runtime.
- **Candidates**: only geometries without `error` and with `identicalToDefault` true. A geometry with
  even one output mismatch or failure is not a candidate. A geometry whose output sha differs from
  the default is never a candidate, however fast (the f32 reduction order is a numerical contract —
  ADR 0022 decision 3).
- **Adoption**: a geometry is adopted for a profile field (one `gemmRows` bucket, `attention.qk`, …)
  only if `speedupVsDefault >= 1.05` on **every** sweep case of that field. Among those, the one with
  the largest geometric mean over the field's cases wins; ties go to the lexicographically first
  geometry name (deterministic output).
- **Otherwise the field keeps the default**, including when the field has no sweep case
  (unmeasured ranges are not interpolated — the same discipline as the ADR 0022 addendum MUST).
- Drift during the sweep (`defaultRepeat.driftRatio`) is not read by the generator (superseded by
  addendum decision 8 on 2026-09-30: cases whose re-measured default ratio is outside 0.9–1.1, or
  whose re-measurement failed or is missing, are dropped from that record's ratios).
- `gemmRows` bucket boundaries come from `GEMM_ROWS_BUCKETS` in `gemm-geometry.ts` (64 / 512 / ∞);
  generation stops on anything other than three buckets, and boundaries are never built from a
  sweep (superseded by ADR 0116 decisions 1 / 3 on 2026-10-01: a fixed set of seven boundaries,
  `PROFILE_GEMM_ROWS_BOUNDS`, that MUST refine the default boundaries; still not derived from the
  sweep input; the default table `GEMM_ROWS_BUCKETS` keeps three buckets).
- A generated profile records its `provenance` (sweep record paths, their sha256, sweep dates,
  adapter) **both in the field and in the file's header comment**, so the measurement behind a value
  can be traced from the source alone. For several records, the values are joined with `", "` in
  the order given; changing that order changes the output.
- The header comment SHOULD carry the regeneration command, and `--check` (regenerate and compare
  byte for byte) SHOULD pass (re-graded 2026-10-18 to SHOULD: the CLI that ran them,
  `tools/geometry-sweep`, was removed with the Deno tools by ADR 0127 decision 5 (`bb35fe2a`); they
  cannot be met until the tools are rebuilt, and restoring them belongs to that rebuild). The
  headers of `geometry-profiles/apple-metal-3.ts:12` and `nvidia-blackwell.ts:12` still name
  `deno run -A tools/geometry-sweep/main.ts profile`, which no longer exists in the tree; so do the
  comments at `packages/runtime/src/tune/derive.ts:5` and
  `packages/runtime/tests/tune_profile_test.ts:6`. (Updated 2026-11-11: `apple-metal-3.ts` keeps that command as the
  record of how it was generated and marks the tool removed; `nvidia-ampere.ts` and `nvidia-blackwell.ts` name the
  untracked Node scripts that derived them; `derive.ts` and `tune_profile_test.ts` name the removed tools as removed.
  No tracked command regenerates a table, so the SHOULD stays unmet.)

### 5. The default profile is byte-identical to the previous table

`DEFAULT_GEOMETRY_PROFILE` holds **the same values** as the selection before this ADR:

| Field       | Source                                                                                                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `gemmRows`  | `GEMM_ROWS_BUCKETS`, the three buckets of `gemmGeometryForRows` (≤ 64 → `M16N16 r1×4 wg4×16`, ≤ 512 → `M64N32 r4×4 wg8×16`, above → `defaultGemmGeometry`) |
| `attention` | `defaultGemmGeometry` for ①QK and ③PV (`M128N128 r8×8 wg16×16`)                                                                                              |
| `conv2d`    | `gemmMTileGeometry(64)` / `gemmMTileGeometry(32)`                                                                                                            |
| `i8a8`      | `defaultI8a8Geometry` (linear / ①QK = `M128N64`, ③PV = `M64N128`; conv3d = `M128N64`, added 2026-10-26 — the geometry conv3d w8a8 ran before)                |

`match` is empty. On a machine that gets the default profile, the generated WGSL and pipeline keys
are byte for byte what they were before this ADR; the unchanged codegen snapshots detect a drift.

### 6. Scope

| Path                                      | Profile field                    |
| ----------------------------------------- | -------------------------------- |
| linear / matmul / bmm (row-count buckets) | `gemmRows`                       |
| fused attention f32 ①QK / ③PV (ADR 0023)  | `attention.qk / pv`              |
| conv2d implicit GEMM (ADR 0024)           | `conv2d`                         |
| i8a8 linear / fused attention ①QK / ③PV   | `i8a8` (the linear by row-count buckets since addendum decision 14) |
| conv3d w8a8 implicit GEMM (ADR 0133)      | `i8a8.conv3d` (added 2026-10-26) |

Measured coverage:

- `gemmRows` is decided by linear / matmul / bmm cases (addendum decision 1; ADR 0116 decision 4);
  all three ops read the same table (only the B-side fill differs).
- The three `i8a8` fields of linear and attention are decided from M ≥ 1024 measurements only and
  apply to every row count (the default does not split by row count either). (Updated 2026-11-12:
  `i8a8.linear` is split by row count and swept on Wan2.2 and Irodori shapes too — addendum
  decision 14.)
- `i8a8.conv3d` is one geometry for every conv3d shape, decided from the Wan2.2 VAE decoder's
  heavy shapes at tile 16 and tile 32 plus two edge guards (addendum decision 11).

**Paths that keep the default for now** (they do not read the profile):

- the GEMM-skeleton tile path of states-shaped attention ①ₜ / ③ₜ (ADR 0067) and decomposed
  attention in row blocks (ADR 0060 — the irodori path). Both go through `gemmGeometryForRows` and
  are not in the sweep; they are widened only after measurement under a separate item.
- conv1d implicit GEMM (it shares `gemmMTileGeometry` with conv2d, but the sweep has no conv1d case).
- the M = 1 × i4 GEMV family (ADR 0082 — outside the GEMM skeleton).

### 7. The dp4a canary fires with the profile's i8a8 attention geometry

- The run-time canary that picks the i8a8 fused attention variant (dp4a / emu;
  `src/gpu/attention-dp4a-canary.ts`, ADR 0058 decision 2) uses `i8a8.attentionQk` /
  `i8a8.attentionPv` of the profile the Session selected.
- Reason: the canary's MUST is to cover at least one full tile with the production geometry. With
  the default geometry, on a machine whose profile changes the i8a8 attention geometry, the canary
  would check a different WGSL from the one that runs.
- The verdict stays cached per device. The profile is a pure function of the same `GpuContext`'s
  `adapterInfo`, so every Session on a device gets the same geometry.
- Fixed inputs, verdict rule and sanity band are unchanged. On B570 the `apple-metal-3` geometry
  still yields an exact dp4a match.

### 8. No run-time autotuning; no entry point for users to choose

(Superseded by addendum decision 6 on 2026-09-29 and ADR 0117 decision 6 on 2026-10-01: users can
inject a profile through `acquireGpu({ geometryProfile })`. The ban on run-time autotuning and the
absence from `SessionOptions` are unchanged.)

- Selection is a function of `adapterInfo` and the static tables only; nothing is measured during
  execution to re-select (the ADR 0022 addendum MUST). The same machine and shape always give the
  same geometry, key and WGSL.
- Geometry stays in the pipeline key (spelling unchanged). Different profiles give different keys,
  so the pipeline cache cannot mix them up.
- **Not exposed in `SessionOptions`.** The selected profile's `id` is shown in
  `SessionDiagnostics.geometryProfile`.
- The ADR 0022 addendum MUST "a change of the default geometry goes together with re-running the
  sha256 checks" applies unchanged to adding or changing a profile, because a profile is the
  default for its adapter. Reason: that geometry does not move bits is an empirical claim, so
  re-running the checks is what keeps correctness. The per-profile GPU test
  (`packages/runtime/tests/gpu_geometry_profile_test.ts:442`, `:471`) runs on any development
  machine; the self A/B on the target machine used `tools/gpu-lab`, removed by ADR 0127 decision 5,
  so until the tools are rebuilt an embedded table cannot be added or changed with this check run
  on its target machine.

### Alternatives not taken

- **Nearest-profile selection by similarity** (GPU name, generation, core count). An unknown machine
  could get a "close" profile, but the selection would change with naming noise and lose the tie
  between per-environment-key sha rows (ADR 0106) and geometry. Falling back to the default lets one
  state exactly which geometry ran.
- **Measuring at run time (autotuning)** — the ADR 0022 addendum MUST. Selection varying per run
  breaks "same key → byte-identical WGSL" and the meaning of the key; since f32 bit identity is
  empirical, an unmeasured combination could be chosen at run time outside every check; first-run
  measuring time is also needed.
- **Passing geometry or a profile in `SessionOptions`** (2026-09-29: exposed instead as a per-GPU
  entry point on `acquireGpu`, addendum decision 6). It would leak internal codegen vocabulary
  (`GemmGeometry`) into the thin public surface (ADR 0008) and let users bring geometries that never
  passed the sweep's checks (output sha match, 5% threshold).

## Consequences

- **Reference sha stays per environment key** (ADR 0106). A profile changes only geometry, and
  geometry does not move bits (empirical), so the sha on a machine is expected to match before and
  after a profile is added. Apple environment-key rows are added when available (explicit SKIP
  until then).
- **Key spelling is unchanged.** Geometry discriminators (`reg64x32r4x4w8`, …) were already in the
  key; no new vocabulary.
- **The DiT gain on the M2 was estimated at about 1.5×** (research §9); addendum decision 4 measured
  it.
- **`apple-metal-3` was generated and registered** on 2026-09-27 from two M2 sweeps (quick and
  full); it was regenerated several times since (addendum decisions 3 and 8, ADR 0116). The
  generated file and its header comment are the source of truth for the adopted and the rejected
  geometries.
- **Profiles also move dispatch limits.** A path with one workgroup per output tile handles n up to
  `maxComputeWorkgroupsPerDimension` (65,535) × tileN; beyond that it fails with
  `DispatchLimitError`, never a silent wrong value. `apple-metal-3`'s conv2d (tileN 64) handles
  Hout·Wout ≤ 4,194,240 (about 2047×2048), half of the default (tileN 128, 8,388,480). Linear N and
  attention ①QK N (key count) shrink to 2,097,120 with tileN 32, out of reach for practical shapes.
- **Adapter spellings (measured, 2026-09-27)**: Chrome on M2 = vendor `apple` / architecture
  `metal-3`; Chrome on RTX 5070 Ti (Windows) = `nvidia` / `blackwell`; Deno on B570 = vendor `32902`
  (PCI vendor ID in decimal) / empty architecture (ADR 0106 decision 2), so under Deno every machine
  then fell back to the default profile. The Deno spelling on macOS was not confirmed. Dawn on Node
  uses a different name source (ADR 0127).
- **Diagnostics field `SessionDiagnostics.geometryProfile`** (a field added to a public type) shows
  the selected profile's `id`. The name is a diagnostic label outside semver.
- The sweep's output-match check caught a real defect: the i8a8 fused attention ①QK generator
  produced wrong values for geometries with five or more K-side fill slots (shadowed WGSL variable
  names). It was fixed with this ADR and is guarded by
  `packages/runtime/tests/codegen_i8a8_shadowing_test.ts`.
- **No Intel / NVIDIA profile in this ADR**; `nvidia-blackwell` followed on 2026-09-29 (addendum
  decision 2).
- **"Drop a8 from the default quant on Metal" was deferred** to a separate decision (2026-09-27): on
  the M2, i8a8 takes as long as f32 even with a better geometry (research §9), which a geometry table
  cannot fix. The question was taken up by [0128](0128-default-preset-original-weights.md) (accepted
  2026-10-29; decision 5: a device without dp4a gets a warning, not a fallback), which settles it.
- Profiles for states-shaped and row-block attention are a separate item (decision 6).

## Acceptance (2026-09-27 to 2026-09-29)

Passed: unchanged codegen snapshots with the default profile, selection order and same-rank
failure (`geometry_profile_test`), full verify on B570 (2026-09-27, 3410 passed / 0 failed), M2 DiT
timing and PNG sha256 identity before and after (addendum decision 4), the selected `id` in the
diagnostics (seen on the RTX, addendum 5).

## Addendum (2026-09-29) — new sweep cases and the `nvidia-blackwell` profile (the maintainer's ruling)

### Addendum decision 1: `gemmRows` is decided by linear / matmul / bmm cases (updates decision 6)

- The sweep gained linear M = 16 / 32 (text_encoder shape `[1,M,1024] × [3072,1024]` with M
  replaced) and M = 128 / 256 (transformer cross-attention k / v projection `[1,M,1024] ×
  [2048,1024]`); three matmul cases (no family's op census has a rank-2 matmul, so each is the
  counterpart of one linear case per bucket — same M / N / K, B side `[K,N]`, marked by
  `censusCount: 0` and `mirrorOf`); five bmm cases from the anima census (text_encoder 2,
  text_conditioner 3). The case table is now `packages/runtime/src/tune/cases.ts`.
- Generator rule: every linear / matmul / bmm case counts toward its row bucket. Since the table
  serves all three ops, a geometry under ×1.05 on a matmul or bmm case is not adopted even if linear
  is faster.
- A class with new cases is re-run with **full**, not quick: the generator only considers
  geometries measured on every case of the field, so adding the cases with quick alone drops the
  adopted geometries that quick does not contain.
- Small bmm cases (15–40 µs per dispatch) did not reach the 80 ms pass target under the shared
  repetition cap of 1024 and drifted (re-measured ratio 0.705–1.552). By the maintainer's ruling the
  sweep got its own cap, `SWEEP_MAX_REPS` = 16384 (`packages/runtime/src/tune/harness.ts:145`;
  recorded as `settings.maxReps`).

### Addendum decision 2: generate and register `nvidia-blackwell` (Chrome on RTX 5070 Ti = `nvidia` / `blackwell`)

- Input: one full sweep on RTX 5070 Ti (Windows, Chrome 153, 2026-09-27T18-37; 33 cases, 1,599
  rows, no failure, no mismatch), rules of decision 4 (`--min-speedup 1.05`). At registration it
  adopted geometries for `gemmRows` ≤ 64 and 65–512 and the three i8a8 fields (×1.19–1.21); the two
  `gemmRows` fields returned to the default with addendum 8, and the table moved to seven buckets
  with ADR 0116. The generated file is the source of truth for its values.
- Acceptance: the per-profile GPU test passed; the before/after PNG sha256 match on the RTX was
  confirmed later with an injected `default` profile (addendum 9).

### Acceptance (addendum)

All items passed on 2026-09-29: new cases without failure or mismatch, sweep paths matching the
production recipe builders, the generator's all-cases rule over matmul / bmm, the per-profile GPU
test with `nvidia-blackwell`, the M2 full re-run, and the RTX before/after sha match (addendum 9).

### Addendum decision 3: regenerate `apple-metal-3` from three sweeps after the M2 full re-run (2026-09-29)

The third record (linear / matmul / bmm, full) sent `gemmRows` ≤ 64 and 65–512 back to the default
(no single geometry was ×1.05 on every case of the bucket); > 512, attention, conv2d and i8a8 were
unchanged. Details: commit `9f50530f`.

### Addendum decision 4: the M2 DiT re-measurement confirms `apple-metal-3` (2026-09-29, K-71 adopted)

The f16 DiT stage at 512² went from 80.5 s to 45.1 s (×1.78), above the kill criterion of 1.3×, with
identical PNG sha256; `apple-metal-3` is adopted (perf-ledger K-71). Details: research K-70 §11.

### Addendum 5: `nvidia-blackwell` is selected on a real RTX 5070 Ti (2026-09-29)

Confirmed in the diagnostics of all four anima stages; timings in research K-70 §12.

### Addendum decision 6: users can inject one static profile into `acquireGpu` (amends decision 8; the maintainer's ruling 2026-09-29)

- Decision 8's "no entry point for users" dated from when profiles came only from the repository; it
  did not weigh a user applying their own sweep on an unknown device, who otherwise stays on the
  default until a table ships in a release. One entry point is therefore exposed.
- Entry point: `acquireGpu({ geometryProfile })` (a field of `AcquireGpuOptions`; the types
  `GeometryProfile`, `GemmRowsRule`, `GemmGeometry` and `I8a8Geometry` become public — types only;
  the default table, the embedded list and the selection function are not exported as values; the
  embedded list was exported later by addendum decision 7). A profile is a property of the device,
  so it is given per GPU. When given, it is used without looking at the adapter, and `match` is not
  compared (for A/B runs with another machine's table). Every Session on that device and the dp4a
  canary (decision 7) use it. The table is copied at acquire time and does not change for the
  device's lifetime.
- Invariants kept: nothing is measured during execution to re-select (ADR 0022 addendum MUST);
  geometry changes only the work assignment, not the K reduction order; not in `SessionOptions` or
  the manifest (a device property does not belong on the model side, nor in the acceptance table of
  ADR 0111); selection happens in one place at Session construction (`session-build.ts`), and the
  diagnostics field and the canary read that result.
- Structural check: a broken table (empty id, malformed `match`, `gemmRows` not ascending or not
  ending in Infinity, a geometry that breaks divisibility) fails with the public `GpuFeatureError`
  before the device is created. Output identity is an empirical claim checked by the sweep; the
  runtime does not guarantee it for arbitrary user tables (generated tables pass, since the
  generator never adopts a mismatching geometry).
- An injected table's `id` may equal an embedded id. The diagnostics do not tell injection from
  embedding, so a recording tool keeps what it injected in a separate field.
- Acceptance (B570, real GPU, with fault injection): the injected table's id and geometry appear in
  the keys and match the default bit for bit; a broken table never reaches `requestDevice`; the
  canary compiles the injected geometries; snapshots without injection are unchanged.

### Addendum 7: the PoC tool `tools/gpu-lab` (2026-09-29, the maintainer's ruling)

(Superseded by ADR 0127 decision 5 on 2026-10-16: the Deno-era tools, `tools/gpu-lab` included, were
removed from the development tree in `bb35fe2a`; they remain on `main` at 0.14.0.) What remains: the
candidate set `quick+` (quick ∪ the adopted geometries of the registered profiles) is the default
of the sweep in `./tune` (`DEFAULT_CANDIDATE_SET` in `packages/runtime/src/tune/geometries.ts`),
because the fixed quick set lacked the M2 and RTX winners and trying other machines' winners costs
about as much as quick. Tables for registration are built from `full`.

### Addendum 8: full re-runs on M2 and RTX 5070 Ti (2026-09-29 evening)

`nvidia-blackwell` lost its single-case `gemmRows` rules (they lost badly at other M of the same
bucket) and keeps only the three i8a8 fields (until its re-derivation of 2026-10-30 — addendum decision 11); on the M2, thermal drift (re-measured ratio 1.160 on
matmul M 4096) would have sent > 512 back to the default. Settled by addendum decision 8 and ADR
0116; details in commit `b860e44b`.

### Addendum 9: RTX before/after match and M5 observation (2026-09-29 to 30)

On RTX 5070 Ti the PNG sha256 matched between an injected `default` and the profile. Chrome reports
Apple M5 as the same `apple` / `metal-3` as the M2, but the M2 winners lose on the M5 (up to 18% on
VAE conv2d), so (vendor, architecture) cannot separate them. Settled by addendum decision 7; the
bucket refinement approved here is ADR 0116. Details: research K-70 §13–§14.

### Addendum decision 7: `description` joins the match key; a table without `match` is injection-only; the embedded tables are exported (2026-09-30)

- Motivation (addendum 9): Chrome reports both Apple M2 and M5 as `apple` / `metal-3`. In Chrome
  Beta guest mode `description` was empty; with a developer flag it showed `"Apple M2"`.
- `match` gains `description` (exact match with `GPUAdapterInfo.description`; empty means absent).
  Order: (vendor, architecture, description) exact → (vendor, architecture) → vendor → default. Two
  tables at the same rank still fail loudly. A rule with `description` also requires vendor and
  architecture.
- A table without `match` is not auto-selected (injection-only, opt-in). It stays in the single
  list `BUILTIN_GEOMETRY_PROFILES` next to auto-selected tables. This is data, not a predicate or a
  flag, so tables stay plain data: JSON can carry them, the generator writes them from the sweep's
  adapter, tests iterate the list, and nothing drifts toward name-similarity matching.
- `BUILTIN_GEOMETRY_PROFILES` is exported as a value from `@karume/runtime`; an application picks a
  table by id and passes it to `acquireGpu({ geometryProfile })` (for the M2: only when
  `description` is available and contains `"Apple M2"`). The selection function and the default
  table are not exported.
- `apple-metal-3` matches `{ vendor: "apple", architecture: "metal-3", description: "Apple M2" }`.
  In Chrome without the flag every Apple device gets the default (the M5 loses nothing; the M2 gets
  ×1.75 by injection); with the flag only the M2 matches. `nvidia-blackwell` needs no `description`.
- Generator options: an explicit `description` and `optIn` (no `match`) —
  `packages/runtime/src/tune/derive.ts`.

### Addendum decision 8: the generator drops cases whose re-measured default is out of range (2026-09-30; amends decision 4)

- Motivation (addendum 8): on the M2 a thermally slowed candidate on matmul M 4096 (re-measured
  ratio 1.160) sent the whole > 512 field back to the default, although the record captured the
  drift in `defaultRepeat.driftRatio`.
- Rule: within one record, a case whose re-measured default ratio is outside 0.9–1.1, or whose
  re-measurement failed or is missing, is dropped from that record's ratios. If another record has
  the case, that one decides; otherwise the case counts as unmeasured and the field keeps the
  default (the safe side). Dropped cases and reasons are written in the generated header comment.
  There is no switch to disable dropping (one rule).
- Applied 2026-09-30: `apple-metal-3` regenerated from four records with `description` `"Apple M2"`
  (five cases dropped; > 512 kept), `nvidia-blackwell` regenerated with unchanged values. Details:
  commit `76812e94`.

### Addendum decision 9: sweep → generate → save → inject inside a user application (ADR 0117, the maintainer's ruling 2026-10-01)

- New user story: an application sweeps on its own device, builds and saves a table, and injects it
  into the next `acquireGpu`, so users of devices the developers do not own still get a tuned table.
  Entry point: the subpath `./tune` of `@karume/runtime` (ADR [0117](0117-app-geometry-tuning.md)
  decisions 1 / 2).
- Decision 4's input check is amended: instead of rejecting on the quantized flag
  (`gpuTiming.quantized`), each observation is judged by an upper bound on its rounding error (at
  most 1% of the ratio); observations above it are dropped from that record's ratios (the same style
  as addendum decision 8). Wall-clock-only records are still rejected (ADR 0117 decision 3).
- `provenance` is structured (the four adapter fields, a kernel fingerprint, a case-set version,
  the candidate set — ADR 0117 decision 4). Matching is done by the application with the pure
  function `geometryProfileMismatch`; the runtime does not match at injection (as in addendum
  decision 6). The injection entry point also takes a synchronous callback
  `(adapterInfo) => GeometryProfile | undefined` (ADR 0117 decision 6).
- Invariants kept: nothing is measured on the acquire / Session path (decision 8, ADR 0022 addendum
  MUST; the subject is restated from "the runtime" to "the acquire / Session path" because the
  `./tune` sweep is an entry point the user calls explicitly); a table does not change during the
  device's lifetime and takes effect from the next `acquireGpu`; sweeps run on a dedicated device,
  never alongside inference.
- Responsibilities are split in ADR 0117 decision 9. The runtime guarantees only that it does not
  change an injected table, does not measure on the acquire / Session path, and checks the
  structure at injection. For a table built inside an application, the only evidence of output
  identity is the sweep's check (over its case set); no end-to-end sha check applies. Recording
  which table ran is the application's responsibility.

### Addendum decision 10: generate and register `nvidia-ampere` (Node + Dawn on RTX 3080 Ti = `nvidia` / `ampere`; 2026-10-23)

- Input: one `quick+` sweep on the RTX 3080 Ti taken one case at a time with the GPU cooled to
  56 °C before each case (corrected 2026-10-27: `runGeometrySweep` ran once per case inside one
  Node process, not one process per case; a sweep of every case in one call throttles within a
  minute and loses most cases to the re-measurement band of addendum decision 8); 57 cases,
  521 rows, no failure, no mismatch, one case dropped from the ratio material. The merged
  record's `candidateKernels` is recomputed over the merged rows with the sweep's own function.
  Rules of decision 4 (×1.050).
- Adopted: `gemmRows` 129–256 `reg64x64r4x4w16` ×1.118, attention ③PV `reg64x64r4x4w16`
  ×1.095, i8a8 linear `tile64x64r8x4w16x8k16` ×1.160; every other class keeps the default. The
  generated file is the source of truth for its values.
- Acceptance: the per-profile GPU test and the fingerprint test pass with the table registered.
  On the Wan2.2 DiT (`f16+dit8-a8-attn8-online`, S = 27,280, cold) the table is ×1.03 on the i8a8
  linear and ×1.009 on the forward — small, because the DiT's rows lie past the sweep's largest
  case and the online attention kernel is not profile-selected; the two f32 adoptions apply to
  other families. Record and A/B: [research](../research/2026-10-23-ampere-geometry-sweep.md).

### Addendum decision 11: conv3d w8a8 reads a required field `i8a8.conv3d` (perf-ledger V-5; 2026-10-26 — the design is the maintainer's ruling 9 a of 2026-10-26, the adoption of the re-derived table its ruling 11 a of 2026-10-27)

- **What.** `GeometryProfile.i8a8` gains a required field `conv3d: I8a8Geometry`, the tile geometry
  of the w8a8 implicit GEMM of conv3d (`convCompute: "a8"`, ADR [0133](0133-conv3d-w8a8.md)). The
  recipe builder passes this one value to the key, the WGSL and the dispatch grid; the generator
  functions (`conv3dI8a8Key` / `conv3dI8a8Wgsl`) no longer have a default geometry. The field is
  validated, frozen, parsed and fingerprinted like the other i8a8 fields (both dp4a variants).
  `DEFAULT_GEOMETRY_PROFILE.i8a8.conv3d` is `defaultI8a8Geometry("conv3d")` =
  `M128N64 r8×8 wg8×16 K16`, the geometry conv3d ran before, so a machine with the default profile
  keeps its keys and WGSL byte for byte (the codegen snapshots are unchanged).
- **Breaking.** The public `GeometryProfile` type and `parseGeometryProfileJson` require the field
  (a stored table from before is rejected, naming `i8a8.conv3d`); `GeometrySweepOp` gains
  `"i8a8-conv3d"`; the case-set version moves, so every saved table mismatches once
  (`geometryProfileMismatch`) and needs one re-sweep.
- **Why a field of its own** (not `i8a8.linear`): on the RTX 5070 Ti (2026-10-25 recon,
  [research](../research/2026-10-25-speed-recon-5070ti.md) §9) two facts separate them. Kernel
  total: the Blackwell linear geometry is 12.7% (next chunk) / 18.2% (first chunk) short of the
  recon's best conv3d geometry. Per shape: the recon's best geometry was faster on 21 / 21 decoder
  shapes. And conv3d would otherwise follow linear's planned row-count buckets (K-85).
- **One geometry for every conv3d shape** (no buckets): in the 2026-10-26 sweep the adopted geometry
  is the fastest `quick+` geometry on 10 of 14 cases by the two-run mean and ties on two more; the
  two shapes a smaller tile wins carry 13.9% of the tile-16 cases' default time weighted by node
  count, and the best geometry per shape would save 1.3% of that time at the adopted geometry
  ([research](../research/2026-10-26-conv3d-geometry-sweep.md) §3).
- **Cases** (sweep op `i8a8-conv3d`, 14 cases from the Wan2.2 TI2V-5B decoder at 1280×704): the six
  heavy 3×3×3 shapes of the tile-16 decoder (98.2% of the next chunk's conv3d FLOPs), `conv_out`
  (Cout 12) and one 1×1×1 shortcut as edge guards, and the six heavy 3×3×3 shapes of the tile-32
  pair (ADR [0134](0134-wan22-vae-tile32-shared.md)). The rules of decision 4 apply unchanged
  (bit-identical and ×1.05 or more on every case of the class, the best geometric mean).
- **The tables' provenance.** A record that carries `caseSet` / `defaultKernels` is rejected once
  the case set moves, so `nvidia-ampere` (2026-10-23 record) needs a new sweep; the Chrome records
  of `nvidia-blackwell` and `apple-metal-3` carry neither and re-derive. Two re-sweeps of all 71 cases
  with the `quick+` set on the RTX 3080 Ti (one process per case, cooled to 56 °C) choose `tile64x64r8x4w16x8k16` for
  `i8a8.conv3d` (×1.339 by the two-run mean, ×1.139–2.472 per case, bit-identical on every row). On
  the default preset's VAE chunk graphs that table is ×1.342 (first) / ×1.204 (next) at tile 16 with
  equal output hashes, measured against the default profile (`DEFAULT_GEOMETRY_PROFILE`, not the
  2026-10-23 `nvidia-ampere`); the default profile with only `i8a8.conv3d` changed is ×1.342 (first)
  / ×1.205 (next). The re-derived table differs from the 2026-10-23 table in two fields,
  `i8a8.conv3d` and `i8a8.attentionQk`: `i8a8.attentionQk` takes the same geometry (×1.103 by the
  two-run mean over its four cases, ×1.110 / ×1.096 per run; the 2026-10-23 table kept the default
  because one of the four cases had been dropped from the ratio material). Every other field is
  unchanged. The 2026-10-26 ruling holds back a table that changes a field other than `i8a8.conv3d`
  until it is ruled on.
- **Decision (the maintainer's ruling 2026-10-27).** The re-derived `nvidia-ampere` is adopted as
  generated: both changed fields, `i8a8.conv3d` (×1.339) and `i8a8.attentionQk` (×1.103), take
  `tile64x64r8x4w16x8k16`, with the provenance of the two new records. `nvidia-blackwell` and
  `apple-metal-3` are re-derived from their records and written with every field unchanged except
  `i8a8.conv3d` (the default) and `provenance.kernels` / `provenance.caseSet`.
- **Caveat: an equal `caseSet` does not mean every class was measured.** `nvidia-blackwell` and
  `apple-metal-3` carry the new `caseSet` although their records contain no conv3d case; their
  `i8a8.conv3d` is the default by absence. Scoping the `caseSet` per op family (ADR
  [0117](0117-app-geometry-tuning.md) decisions 4 / 8) is filed in the [backlog](../backlog.md) (later).
  A conv3d-only sweep on the RTX 5070 Ti is prepared (perf-ledger V-15). (Updated 2026-11-11: V-15 closed on
  2026-10-30 with the maintainer's ruling 42 = c, which re-derived the whole `nvidia-blackwell` table from two full
  sweeps of the current kernels: `i8a8.conv3d` takes `tile64x64r8x4w16x8k16`, the i8a8 ③PV and conv2d rows32 change,
  and GEMMs of 129–512 rows and the i8a8 ①QK return to the default; the records are in the table's provenance.)
  (Updated 2026-11-12: the case-set version and the kernel fingerprint are now taken per op family — ADR
  [0117](0117-app-geometry-tuning.md) addendum 2026-11-12, "the matching keys per op family"; `apple-metal-3` has no
  `i8a8-conv3d` entry, so its `i8a8.conv3d` is the default by absence in the provenance too.)

### Addendum decision 12: the i8a8 geometries take an optional field `skeleton` (perf-ledger K-85; 2026-11-11)

- **What.** `I8a8Geometry` gains an optional field `skeleton?: "vec4"` (the skeleton form of the
  integer GEMM). Absent, a geometry builds the keys and WGSL it built before, byte for byte (the
  codegen snapshots are unchanged). Present, the shared inner-product loop is emitted unrolled (one
  block per pack, every pack offset a constant) in every i8a8 consumer, and the i8a8 linear with i8
  weights where `k % 16 == 0` fills its shared tiles one `vec4<u32>` of global memory per item
  (four whole-word stores with static components into the scalar `u32` tiles — the shared-tile
  MUST of gemm.ts holds). The key names the form: `s` after the K tile (`k32sv4`), `:fill16` for
  the vec4 fill. w4a8 and `k % 16 != 0` keep the word fill with the unrolled loop; ①QK / ③PV and
  conv3d take only the unrolled loop in this slice. The guard (`assertI8a8Geometry`) refuses a
  skeleton K tile that is not whole vec4s (`tileK % 16`) and any value but `"vec4"`.
- **Why a field of the geometry** (not a profile-wide flag or a Session option): the form changes
  no integer (exact i32 reduction — the "kernel variant or geometry" row of
  [kernel-rules](../kernel-rules.md) §3), so it is a speed choice like the tile shape, carried in
  the key and decided per field by the sweep of decision 4. The recorded gains split by field and
  by GPU (the spike: linear ×1.32–1.39 per Wan2.2 33f forward on the RTX 3080 Ti at K32, ×0.41 at
  K32 and about ×1.07 at K64 on the RTX 5070 Ti; ③PV ×1.50–1.58 on both), so one flag could not
  serve both tables.
- **Not breaking.** The field is optional: the default profile and the three embedded tables carry
  none (their keys, WGSL and `provenance.kernels` are unchanged), `acquireGpu({ geometryProfile })`
  and `parseGeometryProfileJson` accept it and reject any other value naming the field, and the
  sweep and the fingerprint build the same fill form the recipe builder takes
  (`linearI8a8UsesVecFill`).
- **The evaluation set `k85`.** The sweep gains a candidate set that measures the field on cases of
  its own instead of the case set of decision 6: the Wan2.2 TI2V-5B DiT linears at 33 frames
  (S = 7,920, from the distribution's transformer graph, with their node counts per forward — 303
  of its 310 linears) and an Irodori M = 185 guard, against the registered tables' `i8a8.linear`
  and a grid of 204 geometries (regM 8 / 12 / 16, regN 4 / 8, the existing workgroup shapes, K
  tile 16 / 32 / 64, with and without the field, within 16,384 B). Its record carries the case-set
  version and the default's fingerprint of its own cases, and the generator derives no table from
  it; `SWEEP_CASES`, the other sets and every saved table's matching keys are unchanged. Every sweep
  record now names the embedded table a Session on that adapter selects (`geometryProfile`), the
  "this adapter's table" a `k85` summary compares against (the Playground's K-85 preset).
  (Updated 2026-11-12: the set is retired — its cases joined `SWEEP_CASES` and its grid the i8a8
  linear's `full`, addendum decision 14.)
- **What stays for a table entry.** `i8a8.linear` is one geometry for every M (decision 6), and the
  recorded small-M losers (Irodori M = 185 / M = 1, ×0.90–0.93) read the same field, so an entry
  waits for the row-count buckets of `i8a8.linear` (the second slice of K-85; ADR
  [0116](0116-geometry-profile-row-buckets.md)). A K64 geometry in `i8a8.linear` would be refused
  at generation for w4a8 at group 32 (`groupSize % tileK`); no default preset runs w4a8 on the a8
  path, but that resolution belongs to the same slice. The vec4 fills of ①QK / ③PV and conv3d, and
  the table derivation on both GPUs (a Chrome sweep on the RTX 5070 Ti, a Node sweep on the RTX
  3080 Ti, then the sha256 re-checks of decision 8), are not done.
- **Measured on the development GPU (2026-11-11, an evaluation, not adoption evidence).** The k85 set
  on the RTX 3080 Ti (Node + Dawn / Vulkan, dp4a, one device per case with 60 s idle between):
  every one of the 1,242 rows was identical to the default; the census-weighted sum over one 33f
  forward's linears is 1,874.6 ms with the `nvidia-ampere` geometry, 1,532.7 ms with the per-case
  best skeleton and 1,578.0 ms with one skeleton geometry for every case. Every case but the guard
  drifted by ×1.14–1.21 during its run, outside the 0.9–1.1 range of addendum decision 8, so the
  derivation would not take these ratios; the numbers and their caveats are in perf-ledger K-85.

### Addendum decision 13: the f32 GEMM geometries take optional fields `form` and `decode` (perf-ledger K-88; 2026-11-11)

- **What.** `GemmGeometry` gains two optional fields. `form?: "tb-un" | "tas-tb-un"` is the form of
  the f32 GEMM skeleton: `"tb-un"` maps a thread's column quads a workgroup row apart
  (`lid.x + q · wgX`, in the accumulator update and in the store alike) and emits the 16-step K
  loop unrolled; `"tas-tb-un"` also stores the A tile k-major (`sa[k · tileM + row]`, still a
  scalar `array<f32>` with static stores — the shared-tile MUST of gemm.ts holds). `decode?:
  "counter"` is the im2col decode of conv2d's implicit GEMM: each B slot keeps an `(ic, kh, kw)`
  counter that advances by the K tile's mixed-radix step, the quad's column origin is computed
  once, and the edge path checks the input row once per quad (`xcol4`, a whole-vec4 constructor).
  Absent, a geometry builds the keys and WGSL it built before, byte for byte (no snapshot moved).
  Present, the key names them (`:tb-un`, `:tas-tb-un`, `:dec`). `form` takes effect in the vec4
  f32-compute kernels of linear / matmul / bmm / conv2d (`gemmFormGeometry`); every other kernel
  that receives such a geometry generates the original form. `decode` takes effect in conv2d's
  vec4 kernel; the generator refuses it on every other op.
- **Why a field of the geometry** (not a default and not a Session option). Both forms run the same
  multiply-adds in the same order per output element (`t` and `kk` ascending, the literal
  `acc + a * b`, bias first in conv) and read the same elements, so the output words do not move —
  the "kernel variant or geometry" row of [kernel-rules](../kernel-rules.md) §3; a speed choice
  like the tile shape, carried in the key and decided per field. This widens what a geometry
  decides from "which thread handles which output" (decision 1's invariant, reaffirmed by addendum
  decision 6) to that plus **a bit-identical code form that keeps the K reduction order and the
  literal multiply-add** — the same widening addendum decision 12 made for the i8a8 skeleton. The
  invariant that matters for the numeric contract (the K order, the MAC form) is unchanged. The
  recorded gains split by GPU and by bucket (×1.42–1.46 on the r8×8 kernels of the RTX 3080 Ti and
  ×1.05–1.06 on the RTX 5070 Ti, ×0.91 on its r1×4 shape; the conv2d decode ×1.07–1.12 and
  ×1.19–1.29), so one default for every device would lose somewhere; the profile route chooses it
  per bucket (`gemmRows` rules, `conv2d.rows64` / `rows32`).
- **Where a table may carry them.** The guard (`assertGeometryProfile`, also at
  `acquireGpu({ geometryProfile })`) and `parseGeometryProfileJson` accept `form` in the
  `gemmRows` rules and in `conv2d`, `decode` in `conv2d` only, and neither in `attention`: the fused
  attention kernels keep the original form whatever the geometry carries, and every op but conv2d
  refuses the decode at generation, so a table carrying them there would either not be what it says
  or fail only when that op is first built. Any other value is refused naming the field.
- **Not breaking.** Both fields are optional: the default profile and the three embedded tables
  carry neither (their keys, WGSL and `provenance.kernels` are unchanged). A candidate's name
  carries the fields as the key spells them, and a geometry without them keeps its name.
- **The evaluation set `k88`.** The sweep gains a second evaluation set (the pattern of addendum
  decision 12): a case table of its own read from the distributions' graphs — the gemma4 E2B
  prefill gate / up, down and q linears at the 768-row chunk (int4 g32, 140 of the chunk's 277
  linears), the umT5-XXL feed-forward wi (int8), the siglip2 so400m fc1 (f32), a small-M linear and
  the square 4,096³ linear and matmul, the three heaviest conv2d of a Wan2.2 VAE decoder next and
  first chunk (int8) and the anima VAE c192 / c384 conv2d (f16), each with its node count in a unit
  (`censusGroup`). Per case the candidates are the geometry the default profile and every
  registered table give it, each without the fields, with each form and, for conv2d, with the
  decode alone and with each form. `deriveGeometryProfile` derives no table from it; the other
  sets, their cases and every fingerprint are unchanged. The Playground's K-88 preset on
  `/geometry-sweep` runs it in a browser.
- **What stays before a table entry.** The intermittent `NVVM compilation failed: 3` seen on the
  RTX 3080 Ti in long multi-pipeline Node runs is unexplained (seven sightings on record, four of
  them without any K-88 code, one in a production-only gemma4 decode run, all at or after the end of
  the results — perf-ledger K-88); the exit-path experiment that would tell a
  teardown race from a kernel has not run. Until it has, no table carries either field and neither
  becomes a default. Then: a derivation on both GPUs (a Chrome sweep on the RTX 5070 Ti through the
  K-88 preset, a cooled Node sweep here) with the candidates in the derivation's own sets, and the
  sha256 re-checks of decision 8. Not covered: the scalar (non-v4) kernels, the f16 compute variant,
  the fused and states-form attention, conv1d / conv3d, and Metal / naga (only the k-major store
  form has passed the M2 probe).
- **Measured on the development GPU (2026-11-12, an evaluation, not adoption evidence).** The k88 set
  on the RTX 3080 Ti (Node + Dawn / Vulkan, `nvidia-ampere`, one device per case with 60 s idle
  between, 5 rounds): every one of the 159 rows was identical to the default, no row failed, and
  every case's drift was ×0.987–1.053 (inside the 0.9–1.1 range of addendum decision 8; the GPU
  peaked at 75 °C). The table geometry against the same geometry with its best form, decode, and
  decode + form, as census-weighted sums: one gemma4 E2B prefill chunk's 140 measured linears
  234.3 → 181.5 ms with the best form (×1.29); one Wan2.2 VAE next chunk's three heavy conv2d
  55.1 ms → 49.7 (form, ×1.11), 49.0 (decode, ×1.12), 39.9 (both, ×1.38); a first chunk 16.3 ms →
  14.4 / 14.8 / 12.0 (×1.13 / ×1.10 / ×1.35); the anima VAE's 11 conv2d 40.8 ms → 34.8 / 36.7 / 29.9
  (×1.17 / ×1.11 / ×1.36); the siglip2 so400m fc1 ×1.32; the umT5 feed-forward wi (r4×4) ×1.00 and
  the M = 64 guard (r1×4) ×1.02. The per-case ratios are in perf-ledger K-88.

### Addendum decision 14: `i8a8.linear` by row-count buckets 256 / 512 / ∞, swept on the Wan2.2 and Irodori shapes (perf-ledger K-85; 2026-11-12)

The maintainer decided this design on 2026-11-12 (K-85's second slice).

- **Why.** `i8a8.linear` was one geometry for every row count, decided from six anima shapes at M 1,024 / 4,096
  (decision 6), while Wan2.2's DiT runs the field at M 7,920 / 512 / 1 and Irodori's default preset `i8-a8` at
  M 185. In the K-85 evaluation the per-case best skeleton beat the table geometry most at the small-M shapes (M 1
  ×1.840 / ×2.400 and the Irodori guard ×1.664 / ×1.869 on the RTX 3080 Ti / RTX 5070 Ti — perf-ledger K-85, which
  marks them as what row-count buckets would take), and one field held one geometry for all of them.
- **The field** (amends decisions 1 and 6). `GeometryProfile.i8a8.linear` is `readonly I8a8RowsRule[]`
  (`{ maxRows, geometry: I8a8Geometry }`), the form and rule order of `gemmRows`: the first rule with
  `rows <= maxRows` applies, `maxRows` strictly ascending, the last `Infinity`. The runtime's guard accepts any
  rule count and checks the order with the very check `gemmRows` uses (one function for both fields). One selector
  reads it, `i8a8LinearGeometry(profile, rows)`. The default profile holds one rule, `defaultI8a8Geometry("linear")`
  up to Infinity, so a machine without a matching table builds the keys and WGSL it built before.
- **Breaking, no reader for the old shape.** `assertGeometryProfile` (and with it `acquireGpu({ geometryProfile })`),
  the limit check, the frozen copies of the embedded tables, `geometryProfileJson` and `parseGeometryProfileJson`
  take the rules array; a stored table with a single geometry there fails (`i8a8.linear is not an array`). The type
  `I8a8RowsRule` joins the public surface.
- **The builder and the w4a8 fallback.** The recipe builder selects the rule by the linear's M. With i4 resident
  weights whose group length is not a multiple of the selected rule's K tile, it uses `defaultI8a8Geometry("linear")`
  instead, whose K tile 16 divides every i4 group length (a power of two of at least 16): the w4a8 kernel refuses a K
  tile that crosses a group boundary (`assertI8a8GroupGeometry`), and a profile field applies to i8 and i4 weights
  alike, so without the fallback a K64 rule would make w4a8 at group 32 unbuildable. MUST: that one geometry, taken
  once, feeds the key, the WGSL, the dispatch grid and `linearI8a8UsesVecFill`; the sweep's case plan (its default
  row) and the fingerprint select by the case's M through the same selector. The vec4 fill stays i8-only, so no i4
  key carries `:fill16`. The sweep's i8a8 linear cases are i8; the fallback never applies there.
- **The bounds.** `I8A8_LINEAR_ROWS_BOUNDS = [256, 512, ∞]` (`src/tune/cases.ts`) is the `i8a8-linear` family's
  bounds: it enters that family's case-set version and no other (`familyBounds` — ADR
  [0117](0117-app-geometry-tuning.md) addendum 2026-11-12, per op family). 512 keeps the Wan2.2 cross-attention k / v
  at M 512 apart from the M 7,920 shapes, and 256 the small-M shapes (M 1, Irodori's M 185) apart from M 512. The
  generator writes exactly these three rules for every table; a bucket with no case in the records keeps the default
  geometry (the existing rule for a class without a case). Its slots are named by range like the `gemmRows` ones
  (`i8a8.linear ≤ 256`, `i8a8.linear 257〜512`, `i8a8.linear > 512`) and checked by the same bucket checker. Every
  bound but the last is the M of a sweep case (ADR [0116](0116-geometry-profile-row-buckets.md) decision 1's rule that
  a bucket's upper end is measured — `tune_cases_test.ts`).
- **The cases.** The five Wan2.2 TI2V-5B DiT shapes at 33 frames (M 7,920 × 3, M 512 and M 1, with their node counts
  per forward 180 / 30 / 30 / 61 / 2 — 303 of its 310 linears), the Irodori v4-small guard (M 185, N 1,280, K 1,280)
  and the same shape at M 256 (the upper end of the bucket ≤ 256) join the six anima cases: 13 `i8a8-linear` cases
  with the ordinary ids (`i8a8-linear-m7920-n3072-k3072`, …); the two Irodori shapes count 0. The evaluation set
  `k85` of decision 12 is retired with everything that existed only for it (its case table, its candidate list as a
  set, the Playground's K-85 preset and summary, the opt-in Node runner and their tests): its cases would otherwise
  live twice under two ids. `k88` stays.
- **The candidates.** `full` gives `i8a8-linear` a grid of its own: the 48-geometry i8a8 grid, then the skeleton grid
  of the retired `k85` (regM 8 / 12 / 16, regN 4 / 8, wgX 8 / 16, wgY 4 / 8 / 16, K tile 16 / 32 / 64, each without
  and with the skeleton), deduplicated and within the 16,384 B of workgroup storage every device provides (the
  automatic selection path does not run the limit check) — 230 geometries, 103 with the skeleton and 62 with K tile
  64. `i8a8-attention` and `i8a8-conv3d` keep the 48; `quick` and `quick+` keep their logic (`quick+` adds every
  rule of each registered table's `i8a8.linear`). The generator reads `skeleton: "vec4"` in a row's
  `geometryParams` (any other value fails the record) and counts it in a geometry's identity, so a skeleton row is a
  candidate like any other.
- **The embedded tables, in the interim.** The family's cases and bounds changed, so each table's `i8a8-linear` entry
  was regenerated with `deriveGeometryProfile` (no `base`) from the table's own source records, read through a script
  that lifted them in memory to the current record format (the current runtime's `caseSets` / `defaultKernels` for
  the families of their rows; the rows untouched). The script checked before writing that every field and provenance
  entry of the other five families is byte-identical and that `i8a8.linear` is three rules: the default geometry for
  ≤ 256 and 257〜512 (the records hold i8a8 linear cases at M 1,024 / 4,096 only) and the geometry the field held
  for > 512. That rests on the rows having been measured on the same kernels: the `candidateKernels` of the
  `nvidia-ampere` and `nvidia-blackwell` records re-derive equal with the current runtime; the `apple-metal-3`
  records predate the field, so for them it rests on the table's own fingerprint of the family re-deriving equal
  before the cases changed. **Until a sweep of the family is registered, the small-M buckets of all three tables run
  the default geometry.** On the two NVIDIA GPUs that is a measured slowdown for the small-M i8a8 linears (Irodori's
  M = 185, for one): in the K-85 evaluation records (untracked) the geometry the table held ran faster than the
  default per dispatch (each row's `speedupVsDefault`), with each case's default re-measurement drift
  (`cases[].defaultRepeat.driftRatio`) beside it:

  | GPU (table, the geometry it held) | Record | M 185 | M 1 | M 512 |
  | --- | --- | --- | --- | --- |
  | RTX 5070 Ti (`nvidia-blackwell`, `tile128x64r8x4w16x16k16`) | `outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_k85_2026-11-12T02-04-54.json` | ×1.338 (drift ×1.000) | ×1.117 (×1.013) | ×1.122 (×1.011) |
  | RTX 3080 Ti (`nvidia-ampere`, `tile64x64r8x4w16x8k16`) | `outputs/diag/k85-linear-2026-11-11/geometry-sweep_k85_*.json` (one per case) | ×2.013 (drift ×1.089) | ×1.991 (×1.194) | ×1.412 (×1.171) |

  The RTX 3080 Ti drifts at M 1 and M 512 are outside the 0.9–1.1 range of addendum decision 8 (five of that run's
  six cases drifted outside it, all but the M 185 guard), so those figures are an evaluation, not adoption evidence. No such record exists
  for the Apple M2 (`apple-metal-3`): the effect there is not measured. Outputs do not change (the i8a8 geometries
  are bit-identical by contract — decision 1, addendum decision 12).
- **Not in this decision.** A `full` sweep of `i8a8-linear` alone on each GPU, derived with the adapter's table as
  `base`, and the sha256 re-checks of decision 8 on the families whose default presets reach the i8a8 linear. On a
  GPU that heats within one case, the sweep is taken in chunks (ADR [0117](0117-app-geometry-tuning.md) addendum
  2026-11-12, "the `geometries` sweep option").
- **Checks.** `geometry_profile_test.ts` (the selector at M = bound and bound + 1, the guard naming `i8a8.linear`,
  the builder's geometry with the fallback and never `:fill16` for i4); `gpu_geometry_profile_test.ts` (a Session at
  M = 256 / 257 runs each rule's key; w4a8 at group 32 under a K64 rule runs the default geometry and at group 64 the
  K64 rule without `:fill16`, each with the default profile's words; one i8a8 linear case per rule of every embedded
  table); `tune_cases_test.ts` (the bounds measured, the 13 cases); `tune_profile_test.ts` (a bucket's rows decide that
  bucket alone; a skeleton row is read and adopted); `tune_fingerprint_test.ts` (the bounds enter the i8a8-linear
  family's case-set version only); `tune_geometries_test.ts` (the grid); `tune_mismatch_test.ts` (every embedded table
  against its own adapter).
