> This document is a point-in-time snapshot (2026-10-26, corrected 2026-10-27) of perf-ledger V-5 on the RTX 3080 Ti
> (Node + Dawn on Vulkan): the geometry sweep that adds conv3d w8a8 cases and re-derives `nvidia-ampere`, and the
> chunk-graph A/B of the derived geometry. It is not a design source of truth (the rules are ADR 0115 / 0116 / 0117 and
> the addendum decision 11 of ADR 0115; the values are the generated files). Raw material is under
> `outputs/diag/wave-b-2026-10-26/v5/` (not tracked).

# conv3d w8a8 geometry: the sweep, the re-derivation of `nvidia-ampere`, and the ①QK ruling

Question (the maintainer's ruling 2026-10-26): conv3d w8a8 (`convCompute: "a8"`, ADR 0133) ran linear's default tile on
every adapter. With a profile field `i8a8.conv3d` and sweep cases of the Wan2.2 VAE decoder's shapes, what does the
sweep choose on this machine, is the choice bit-identical and faster on the real chunk graphs, and does the full
re-sweep that the new case set forces leave every other field of `nvidia-ampere` as it was?

This record overturns three statements of the [2026-10-23 Ampere sweep note](2026-10-23-ampere-geometry-sweep.md): its
①QK class was not "decided by its other three cases, all of which keep the default" (§2); the nearest miss of its i8a8
①QK class was not "nothing above ×0.95" (§3) — in that record `tile64x64r8x4w16x8k16` read ×1.095 / ×1.102 / ×1.088
and `tile128x64r8x4w16x16k16` ×1.052 / ×1.066 / ×1.091 on the three measured ①QK cases (for ③PV the statement holds:
its nearest miss is ×0.944); and its sweep did not run one process per case (§2).

## 1. Summary

- Both runs choose `tile64x64r8x4w16x8k16` for `i8a8.conv3d` (the geometry `nvidia-ampere` already uses for the i8a8
  linear). The derivation's figure is ×1.339: the geometric mean over the 14 conv3d cases of each case's two-run
  geometric mean; the per-run geometric means are ×1.338 (run 1) and ×1.340 (run 2). Per case the two-run mean is
  ×1.139–2.472, and the lowest single-run value is ×1.134 (run 1) / ×1.144 (run 2). The output is bit-identical to the
  default on all 126 non-default conv3d rows of each run.
- On the chunk graphs of the default preset's VAE decoder (tile 16), the derived table is ×1.342 on the first chunk and
  ×1.204 on the next chunk (median GPU time) against `DEFAULT_GEOMETRY_PROFILE`, with the same output hash. The conv3d
  GEMM alone is ×1.579 / ×1.374. On the tile-32 next chunk the conv3d GEMM is ×1.243 (median); the whole-chunk ratio
  there is not reliable (§5).
- The re-derived table also changes `i8a8.attentionQk`, from the default `tile128x64r8x8w8x16k16` to
  `tile64x64r8x4w16x8k16`: ×1.103 by the two-run mean (per run ×1.110 / ×1.096; the lowest single-run case is ×1.067,
  run 2 on `i8a8-attention-qk-cross-m1024-n512`). The 2026-10-26 ruling holds back a table that changes a field other
  than `i8a8.conv3d` until it is ruled on; the maintainer accepted this change on 2026-10-27, and the re-derived tables
  are the ones in the repository. No other field moves.

## 2. Method

The sweep is `runGeometrySweep` of `@karume/runtime/tune` with the `quick+` candidate set and 3 rounds per geometry,
the protocol of the 2026-10-23 sweep (§2 of the [Ampere sweep note](2026-10-23-ampere-geometry-sweep.md)), made
stricter in one point:

- One process per case. A small zsh driver (`sweep-run.zsh`) starts a new Node process (`sweep-one-case.mjs`) for each
  of the 71 cases of `SWEEP_CASES`. The 2026-10-23 script (`.claude/reviews/2026-10-11_wan-attention-recon/
  sweep-ampere-percase.mjs`, not tracked — git-ignored) called `runGeometrySweep` once per case inside one Node process.
- Start rule: before each case the driver waits until `nvidia-smi` reads 56 °C or less (at most 900 s). Run 1's first
  case started cold at 43 °C with no wait; every other case of both runs started at 55–56 °C.
- Temperature at the end of a case, as the logs show it (`run1.log` / `run2.log`): run 1 62–76 °C (62 °C is the cold
  first case; 66–76 °C for the rest), run 2 66–77 °C.
- Lock: the driver takes no lock itself. Its header says to run the whole script under `flock /tmp/karume-gpu.lock`, so
  that the lock is also held through the cooling waits; the logs do not record whether it was.
- The 71 per-case records of a run are merged into one record (`merge-derive.mjs`). The merge checks that every
  per-case record carries the `caseSet` and `defaultKernels` of the code it runs against, and recomputes
  `candidateKernels` over the merged rows with the sweep's own function (a per-case record fingerprints only its own
  rows).
- The table is derived with `deriveGeometryProfile` from both merged records at once. Where both runs measured the same
  (case, geometry), the ratio is the geometric mean of the two observations.
- No clock or power log was kept; only the temperatures above.

**Heating inside a case and the fixed candidate order.** A case heats the GPU while it runs: the conv3d cases rise
15–21 °C (run 1) / 16–22 °C (run 2) in 11–19 s. Each case measures the default first and once more at the end (the
default re-measurement); the ratio of the second to the first reading has a median of ×1.014 (run 1) / ×1.013 (run 2)
and a maximum of ×1.044 / ×1.054 over the 71 cases. The candidates run in one fixed order in every case, so the later
ones are measured on a hotter GPU: in the conv3d class the adopted geometry is the 4th of 10 rows and
`tile128x64r8x4w16x16k16` (the `nvidia-blackwell` linear geometry) is the 10th and last. Where two geometries are within
this bias of each other the ranking between them is not resolved: on the 1024→512 shapes and the 1×1×1 shortcut the two
are within 0.4% (§3). The class verdict is outside it: ×1.339 against ×1.211.

The conv3d plan of the harness launches only the integer dot-product GEMM of the w8a8 path (the dispatch that has a
geometry), from activation words, scales and tap-major weights written directly in the formats that the production
quantiser and the Session's reorder produce. Its binding table is pinned to the builder's, and on the GPU its default
geometry's output is bit-identical to a CPU reference over the same inputs (tests/tune_harness_test.ts). The harness
and the recipe builder take the store variant, the key and the WGSL from one function (`conv3dI8a8Kernel` in
`src/kernels/conv3d-i8a8.ts`), and a CPU test checks for every conv3d case that the harness takes the store variant of
an independent rule (vec4 exactly when the output pixel count is a multiple of 4) and the key `conv3dI8a8Key` gives.

The 14 conv3d cases (`i8a8-conv3d-*`; shapes copied from the containers' static shapes, the next chunk's where the first
and next chunks differ):

| Group | Cases (Cin → Cout @ T×H×W input) |
| --- | --- |
| tile 16, the six heavy 3×3×3 shapes (98.2% of the next chunk's conv3d FLOPs, `shapes/conv3d-shapes.json`) | 256→256 @6×128², 512→512 @6×64², 1024→1024 @4×32², 1024→1024 @3×16², 1024→512 @6×64², 512→256 @6×128² |
| tile 16, edge guards | `conv_out` 256→12 @6×128² (the smallest Cout); the 1×1×1 shortcut 1024→512 @4×64² |
| tile 32 (ADR 0134), the same six heavy 3×3×3 shapes | 256→256 @6×256², 512→512 @6×128², 1024→1024 @4×64², 1024→1024 @3×32², 1024→512 @6×128², 512→256 @6×256² |

| Record (under `outputs/diag/wave-b-2026-10-26/v5/`) | Cases | Rows | Errors / output mismatches | Default re-measurement (min / median / max) | Dropped |
| --- | --: | --: | --: | --- | --: |
| `geometry-sweep-node-percase-run1-2026-10-26T19-31-37.851Z.json` (sha256 `aa978a90…`) | 71 | 661 | 0 / 0 | ×0.982 / ×1.014 / ×1.044 | 0 |
| `geometry-sweep-node-percase-run2-2026-10-26T20-20-58.449Z.json` (sha256 `d265e379…`) | 71 | 661 | 0 / 0 | ×0.982 / ×1.013 / ×1.054 | 0 |

**Source files written during run 1.** A CPU fault-injection pass (`mutants/mutants.mjs`: 16 mutants, each edits one
line of a `packages/runtime/src` file, runs CPU tests and restores the file) ran in the same checkout as the sweep.
Run 1 ran 18:44:11–19:31:37 UTC; its 10th case ended at 18:49:37. These surviving file times (UTC, 2026-10-26) fall
inside run 1:

| File | Last written |
| --- | --- |
| `mutants/mutants.mjs` (the pass's script) | 18:49:35 |
| `src/kernels/geometry-profiles/index.ts`, `src/tune/profile-json.ts` | 18:49:37 |
| `src/tune/derive.ts` | 18:49:42 |
| `src/tune/fingerprint.ts` | 18:49:52 |
| `mutants/cpu-mutants.json` (the pass's results, all 16 mutants) | 18:49:54 |

The mutant list also edits `src/tune/cases.ts` (M15, after `fingerprint.ts`'s M10–M12 in the list),
`src/tune/harness.ts`, `src/tune/geometries.ts`, `src/kernels/geometry-profile.ts`, `src/kernels/i8a8-geometry.ts` and
`src/runtime/recipe-builders/conv.ts`; their times from this pass are overwritten by later writes (`geometry-profile.ts`
and `i8a8-geometry.ts` at 20:34 on 2026-10-26, the other four on 2026-10-27), so these times do not show them.
`harness.ts` and `conv.ts` were also edited and restored after run 2 by the GPU fault-injection pass
(`mutants/gpu-mutants.json`, 20:23:41). Which mutation, if any, was in place when a given per-case process loaded its
modules cannot be told from the file times.

Why the data still holds, whichever op a case runs: every per-case process computes `caseSet` and `defaultKernels` over
all of `SWEEP_CASES` when it writes its record, so an edit to the case set, the fingerprint or a default kernel shows in
every case's record. The merge rejects any per-case record whose `caseSet` / `defaultKernels` differ from the code it
merges against; both merged records carry `caseSet` `0c9e4c7b7202eb2a` and `defaultKernels` `82f8e6c9ad8129ad`, which
the final code computes as well. And run 2, inside which no surviving file time falls, derives on its own the same table
as run 1 (every field equal) and as both runs together.

## 3. The conv3d class

Ratios to the default geometry (`tile128x64r8x8w8x16k16`) per case; each run's ratio is its minimum over 3 rounds, and
"mean" is the geometric mean of the two runs, which is what the derivation compares:

| case | default per dispatch (run 1 / run 2) | `tile64x64r8x4w16x8k16` (run 1 / run 2 / mean) | `tile64x64r4x4w16x16k16` (mean) | `tile128x64r8x4w16x16k16` (mean) | default re-measurement (run 1 / run 2) |
| --- | --: | --: | --: | --: | --: |
| `cin256-cout256-k333-in6x128x128` | 7.85 / 7.88 ms | ×1.246 / ×1.245 / ×1.246 | ×1.126 | ×1.185 | ×1.011 / ×1.007 |
| `cin512-cout512-k333-in6x64x64` | 7.65 / 7.66 ms | ×1.204 / ×1.219 / ×1.211 | ×1.118 | ×1.186 | ×1.017 / ×1.012 |
| `cin1024-cout1024-k333-in4x32x32` | 4.88 / 4.88 ms | ×1.487 / ×1.487 / ×1.487 | ×1.361 | ×1.256 | ×1.012 / ×1.021 |
| `cin1024-cout1024-k333-in3x16x16` | 1.88 / 1.87 ms | ×1.860 / ×1.864 / ×1.862 | ×1.752 | ×1.355 | ×1.012 / ×1.010 |
| `cin1024-cout512-k333-in6x64x64` | 15.62 / 15.69 ms | ×1.183 / ×1.188 / ×1.186 | ×1.128 | ×1.186 | ×1.023 / ×1.017 |
| `cin512-cout256-k333-in6x128x128` | 15.23 / 15.30 ms | ×1.201 / ×1.213 / ×1.207 | ×1.097 | ×1.184 | ×1.021 / ×1.019 |
| `cin256-cout12-k333-in6x128x128` | 3.59 / 3.59 ms | ×2.468 / ×2.477 / ×2.472 | ×2.204 | ×1.236 | ×1.010 / ×1.009 |
| `cin1024-cout512-k111-in4x64x64` | 0.60 / 0.60 ms | ×1.167 / ×1.167 / ×1.167 | ×1.107 | ×1.166 | ×1.015 / ×1.013 |
| `cin256-cout256-k333-in6x256x256` | 30.54 / 30.65 ms | ×1.229 / ×1.222 / ×1.225 | ×1.105 | ×1.162 | ×1.026 / ×1.028 |
| `cin512-cout512-k333-in6x128x128` | 30.41 / 30.23 ms | ×1.204 / ×1.180 / ×1.192 | ×1.102 | ×1.183 | ×1.018 / ×1.020 |
| `cin1024-cout1024-k333-in4x64x64` | 15.74 / 15.67 ms | ×1.214 / ×1.206 / ×1.210 | ×1.140 | ×1.184 | ×1.018 / ×1.023 |
| `cin1024-cout1024-k333-in3x32x32` | 2.69 / 2.69 ms | ×1.451 / ×1.465 / ×1.458 | ×1.339 | ×1.384 | ×1.019 / ×1.015 |
| `cin1024-cout512-k333-in6x128x128` | 61.17 / 61.51 ms | ×1.134 / ×1.144 / ×1.139 | ×1.084 | ×1.143 | ×1.036 / ×1.031 |
| `cin512-cout256-k333-in6x256x256` | 60.19 / 60.09 ms | ×1.177 / ×1.182 / ×1.180 | ×1.088 | ×1.166 | ×1.023 / ×1.017 |

- Adopted: `tile64x64r8x4w16x8k16` ×1.339 (two-run mean; ×1.139–2.472 per case). Runner-ups, also at ×1.05 or more on
  every case: `tile64x64r4x4w16x16k16` ×1.238 and `tile128x64r8x4w16x16k16` (`nvidia-blackwell`'s linear geometry)
  ×1.211.
- Every other `quick+` geometry falls below ×1.05 on tile-16 cases as well as on tile-32 cases (two-run mean; the same
  tile-16 counts in each run):

  | geometry | tile-16 cases below ×1.05 (of 8) | tile-32 cases below ×1.05 (of 6) | lowest tile-16 / tile-32 case |
  | --- | --: | --: | --- |
  | `tile64x128r8x8w16x8k16` | 7 | 6 | ×0.971 / ×0.951 |
  | `tile64x32r4x4w8x16k16` | 5 | 5 | ×0.853 / ×0.821 |
  | `tile32x32r4x4w8x8k16` | 5 | 5 | ×0.782 / ×0.730 |
  | `tile64x64r4x8w8x16k16` | 3 | 4 | ×1.025 / ×1.002 |
  | `tile32x64r4x8w8x8k16` | 5 | 5 | ×0.638 / ×0.408 |
  | `tile16x128r4x8w16x4k16` | 6 | 6 | ×0.224 / ×0.149 |

- One geometry for every shape costs little here. By the two-run mean the adopted geometry is the fastest of the
  `quick+` set on 10 of the 14 cases. On the two 1024→512 shapes `tile128x64r8x4w16x16k16` is ahead by 0.02% and 0.39%
  (×1.186 / ×1.186, ×1.143 / ×1.139), and on the 1×1×1 shortcut it is 0.06% behind; these are inside the
  candidate-order bias of §2. Smaller tiles win the two small shapes: on 1024→1024 @3×16² `tile32x32r4x4w8x8k16` ×2.063
  and `tile64x32r4x4w8x16k16` ×1.982 beat the adopted ×1.862; on `conv_out` `tile16x128r4x8w16x4k16` ×4.019,
  `tile32x32r4x4w8x8k16` ×3.568 and `tile32x64r4x8w8x8k16` ×2.971 beat the adopted ×2.472. Those two shapes take 13.9%
  of the time the eight tile-16 cases take at the default geometry, weighted by their node counts in the next chunk
  (default per-dispatch time × node count, either run). Weighted the same way, with each case's time taken as its
  two-run mean default per-dispatch time divided by its two-run mean ratio, the adopted geometry gives 120.74 ms and
  the best geometry per shape 119.20 ms (−1.3%), against more keys and test surface. The tile-32 shapes prefer the same
  geometry as the tile-16 shapes.

## 4. Every field of the re-derived `nvidia-ampere`, the stop rule, and the ruling

The maintainer's rule of 2026-10-26: if the re-derived table differs from the current one in any field other than
`i8a8.conv3d` (and the provenance), it is not written into the repository until ruled on. Field by field, the re-derived
table against the 2026-10-23 `nvidia-ampere` (which had no `i8a8.conv3d` field):

| field | 2026-10-23 table | re-derived |
| --- | --- | --- |
| `i8a8.attentionQk` | `{ regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 }` (default) | `{ regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 }` |
| `i8a8.conv3d` | no such field — conv3d ran the default `{ regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 }` | `{ regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 }` |
| every `gemmRows` bucket, `attention.qk` / `.pv`, `conv2d.rows64` / `.rows32`, `i8a8.linear`, `i8a8.attentionPv` | — | unchanged |
| `provenance.sweep` / `sha256` / `date` / `candidateSet` / `kernels` / `caseSet` | the 2026-10-23 record | the two new records; `kernels` `c1b6cb7ea2ede6d2`, `caseSet` `0c9e4c7b7202eb2a` |

Under that rule `i8a8.attentionQk` needed a ruling. **The maintainer accepted the change on 2026-10-27**, and the
re-derived table is the one in the repository (`src/kernels/geometry-profiles/nvidia-ampere.ts`, generated; equal to
`nvidia-ampere-combined.profile.json` in every field but the provenance). `nvidia-blackwell` and `apple-metal-3` are
re-derived from their own Chrome records and written too: their check passed, as the only changes are `i8a8.conv3d`
(at the default — none of their records has a conv3d case) and `provenance.kernels` / `provenance.caseSet`.

The adoptions the 2026-10-23 table made are re-adopted with the same geometry (the 2026-10-23 figures from
`baseline/nvidia-ampere.verdicts.txt`): `gemmRows` 129–256 ×1.114 (was ×1.118), attention ③PV ×1.113 (was ×1.095), i8a8
linear ×1.170 (was ×1.160).

**Why `i8a8.attentionQk` moves.** In the 2026-10-23 record the case `i8a8-attention-qk-self-m1024-n1024` was dropped
from the ratio material (its default re-measurement read ×1.115). A class adopts a geometry only with a ratio on every
one of its cases, so with that case unmeasured for every candidate the class kept the default, although the other three
cases read ×1.095 / ×1.102 / ×1.088 for this geometry in that record. In the new runs no case is dropped:

| case | run 1 (default per dispatch, ratio) | run 2 | default re-measurement (run 1 / run 2) |
| --- | --- | --- | --- |
| `i8a8-attention-qk-self-m1024-n1024` | 0.1402 ms, ×1.110 | 0.1394 ms, ×1.101 | ×1.007 / ×1.027 |
| `i8a8-attention-qk-self-m4096-n4096` | 2.0649 ms, ×1.090 | 2.0682 ms, ×1.092 | ×1.015 / ×1.012 |
| `i8a8-attention-qk-cross-m1024-n512` | 0.0710 ms, ×1.125 | 0.0715 ms, ×1.067 | ×1.044 / ×1.054 |
| `i8a8-attention-qk-cross-m4096-n512` | 0.2756 ms, ×1.116 | 0.2812 ms, ×1.123 | ×1.013 / ×0.982 |

The class figure is ×1.103 by the two-run mean (×1.091–1.120 per case); the per-run geometric means are ×1.110 (run 1)
and ×1.096 (run 2), and the lowest single-run case is ×1.067 (run 2, `cross-m1024-n512`, whose default re-measurement
read ×1.054). Output bit-identical in every row. Where the field acts: i8a8 fused attention's ①QK on this adapter (the
materialised `attentionScoreStorage` routes, not the online route the Wan2.2 presets use) and the dp4a canary, which
compiles the profile's ①QK geometry (ADR 0115 decision 7).

## 5. The chunk-graph A/B

One Node process, two timing devices: device A with `DEFAULT_GEOMETRY_PROFILE` injected (the baseline is the default
profile, not the previously embedded `nvidia-ampere`), device B with the derived table
(`nvidia-ampere-combined.profile.json`) injected. Both run the default preset's VAE decoder chunk graphs (int8 weights,
`convCompute: "a8"`) with the same random inputs; the order flips every round (A B / B A); one warm-up run per graph is
not counted. GPU time per run is the Session's per-dispatch timing. A second run with B = the default profile with only
`i8a8.conv3d` changed (`ab-tile16-default+conv3d.json`) gives the same ratios (first ×1.342, next ×1.205).

| chunk graph | rounds | A: median / min | B: median / min | ratio (median / min) | conv3d GEMM, A → B (median) | the rest, A → B (median) | output hash |
| --- | --: | --: | --: | --: | --: | --: | --- |
| `vae_decoder_first` (tile 16) | 15 | 100.17 / 99.29 ms | 74.66 / 74.06 ms | ×1.342 / ×1.341 | 70.67 → 44.75 ms (×1.579) | 29.53 → 29.92 ms | equal (one hash each side) |
| `vae_decoder_next` (tile 16) | 15 | 256.15 / 253.58 ms | 212.79 / 210.98 ms | ×1.204 / ×1.202 | 170.51 → 124.08 ms (×1.374) | 85.29 → 88.77 ms | equal |
| `vae_decoder_next_32` (tile 32) | 9 | 1164.40 / 927.28 ms | 1028.08 / 936.04 ms | not reliable (see below) | 783.95 → 630.80 ms (×1.243) | 381.96 → 395.39 ms | equal |

- The rest of the chunk is slower on device B by 0.39 ms (first) and 3.47 ms (next), mostly the f32 conv2d kernel
  (55.95 → 58.82 ms median on the next chunk, +2.88 ms), whose geometry is the same on both devices. The default +
  conv3d-only run shows the same shift (conv2d 56.81 → 59.57 ms, +2.76 ms), so it does not come from another field of
  the table. An A/A control (`ab-chunks-aa.ts`: B = a `structuredClone` of `DEFAULT_GEOMETRY_PROFILE` with the id
  `default-aa-copy`, tile 16, next chunk, 9 rounds; record `ab-aa-tile16-next.json`, 2026-10-27) read conv2d 53.81 ms
  (A) and 53.78 ms (B) (−0.03 ms), a total ratio of ×1.004 (median; ×1.005 by minimum) and equal output hashes, so being
  device B does not by itself move conv2d. The shift stays unexplained (the V-6 A/B saw the same kernel move with a
  conv3d change).
- Tile 32 is not stationary (`ab-tile32-next-nvidia-ampere-combined.json`). Round 1 (A first): A 927.28 ms (conv3d
  599.11, rest 328.17), B 936.04 ms (conv3d 540.65, rest 395.39) — B's total is slower although its conv3d is 58.46 ms
  faster, because its non-conv3d part is 67.22 ms slower. From round 2 on both devices sit in a slower state, by
  different amounts: A 1148.28–1192.71 ms per run (×1.24–1.29 of its round 1), B 1006.54–1079.77 ms (×1.08–1.15). The
  median ratio ×1.133 therefore compares two unequal slow states, and the minimum ratio ×0.991 is round 1 alone; neither
  is a reliable whole-chunk number. The conv3d GEMM is faster on B in every round (540.65–670.84 ms against
  599.11–808.12 ms). `vae_decoder_first_32` was not measured; the two-device A/B of the tile-32 pair does not fit in
  12 GB per the script (`ab-chunks.ts`).
- Estimate (speculation, tile-16 median deltas × chunk counts): one tile-16 tile of a 33-frame stage is one first and
  eight next chunks, so about −0.37 s of GPU time per tile (25.51 + 8 × 43.36 ms) and about −10 s per 1280×704×33 VAE
  stage (28 tiles) on this machine. The chunk and tile counts are the stage's, not taken from this record.

## 6. What this does not settle

- Shapes without a case of their own. The conv3d scalar store (output pixel count P % 4 ≠ 0) has no sweep case (every
  case and every decoder shape in `shapes/conv3d-shapes.json` takes the vec4 store). The 3×1×1 time convolution
  (Cout 2048 × Cin 1024, `k311`, two nodes per next chunk) has none either: 28.99 GF of the 4,162.6 GF of conv3d work
  per next chunk at tile 16 (25.77 + 3.22 GF; `shapes/conv3d-shapes.json`). These run the adopted geometry unmeasured;
  their output is bit-identical by construction (ADR 0133 decision 1), not by a sweep row. Filed as perf-ledger V-14.
- Devices with a small storage binding limit. Two tile-32 cases bind a 256 MiB output buffer (256→256 @6×256² and
  512→256 @6×256²; two more bind 128 MiB). On a device whose `maxStorageBufferBindingSize` is below that, every
  candidate fails on those cases, so the whole conv3d class stays at the default there. No such device was measured.
  Filed as perf-ledger V-14.
- `nvidia-blackwell` and `apple-metal-3` carry the new `caseSet` although their records contain no conv3d case: an
  equal `caseSet` does not mean every class was measured, and their `i8a8.conv3d` is the default by absence. Scoping the
  `caseSet` per op family (ADR 0117 decisions 4 / 8) is filed in the [backlog](../backlog.md) (later). A conv3d-only
  sweep on the RTX 5070 Ti is prepared (perf-ledger V-15; `outputs/diag/wave-b-2026-10-26/v5/runbook-5070ti.md`).
- The new ①QK geometry is checked by the sweep's `identicalToDefault` on every row and by the per-profile GPU test
  (tests/gpu_geometry_profile_test.ts — the embedded table against the default, Uint32-equal, on one i8a8 attention
  case). No end-to-end run of a materialised-attention preset (anima, or the Wan2.1 practical preset
  `f16+dit8-a8-attn8-s16`) was measured with it.
- D3D12: the adopted geometry's bit identity on Chrome / D3D12 rests on the 2026-10-25 recon (0 of 310 / 330 chunk
  hashes differed) and on the sweep's own check where a sweep runs; the `nvidia-ampere` table also applies to Chrome on
  Ampere (match is vendor / architecture only), which is unmeasured.
- Tile 24 (V-9) and other tile sizes have no case.
- The Wan2.2 conv8 sha rows (tile 16 and tile 32, Node / RTX 3080 Ti) were run on 2026-10-26 while the embedded table
  still carried the default `i8a8.conv3d`, so they ran the default geometry; they check the adopted geometry only when
  they run again with the regenerated table. Until then its end-to-end evidence is the equal chunk hashes of §5.

## 7. Reproduction

The scripts are in `outputs/diag/wave-b-2026-10-26/v5/` (not tracked): `sweep-run.zsh` / `sweep-one-case.mjs` (the
per-process sweep), `merge-derive.mjs` (merge and per-run derivation), `combine-stoprule.mjs` (the derivation from both
runs and the field-by-field check), `tables-md.mjs` (the tables of §3–§4), `ab-chunks.ts` (§5), `gen-profile-ts.mjs`
(the generator of the embedded file, with the conv3d line), `rederive-builtin.mjs` / `patch-builtin.mjs` (the
re-derivation of the two Chrome tables), `mutants/` (the fault-injection passes of §2).
