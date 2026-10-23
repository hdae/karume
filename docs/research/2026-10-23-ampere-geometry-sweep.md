> This document is a point-in-time snapshot (2026-10-23) of the geometry sweep on the RTX 3080 Ti (Node + Dawn on
> Vulkan) that produced the embedded profile `nvidia-ampere`, and of the one measurement that checked it on the Wan2.2
> DiT; it is not a design source of truth (the rules are ADR 0115 / 0116 / 0117, the values are the generated file).

# `nvidia-ampere` — the geometry sweep on the RTX 3080 Ti and what it bought

Question (a follow-up of the K-82 work, the maintainer's ruling 2026-10-23): the machine that now runs every Node lane
is `nvidia` / `ampere` (RTX 3080 Ti) and had no embedded table, so every GEMM-shaped kernel ran the default geometries.
Does a sweep on it adopt anything, and does the adopted table move the Wan2.2 DiT?

## 1. Summary

- Three classes adopt a geometry (the rules of `deriveGeometryProfile`, ×1.050 or more on every case of the class,
  bit-identical output on every row): `gemmRows` 129–256 `reg64x64r4x4w16` ×1.118, the fused f32 attention ③PV
  `reg64x64r4x4w16` ×1.095, and the i8a8 linear `tile64x64r8x4w16x8k16` ×1.160 (×1.073–1.247 across six cases; the
  gain is largest at M = 1,024 and shrinks to ×1.08 at M = 4,096 / N = 8,192). Every other class keeps the default.
- On the Wan2.2 DiT (`f16+dit8-a8-attn8-online`, S = 27,280, measured mode, cold GPU) the table is worth ×1.03 on the
  i8a8 linear (6.84 → 6.64 s per forward) and ×1.009 on the forward (20.22 → 20.04 s GPU): the DiT's rows (M = 27,280)
  sit where the sweep's gain has already flattened, and the online attention kernel (58% of the forward) is not a
  profile-selected kernel. The two f32 adoptions do not run in this preset.
- The table is registered (ADR 0115 addendum decision 10): the gain is small on the Wan2.2 DiT but real, the output is
  bit-identical by construction and by the sweep's check, and the f32 adoptions apply to every family that runs f32
  attention or 129–256-row GEMMs on this machine.

## 2. Method — one process per case, cooled between cases

The first attempt was the whole `quick+` set in one process. The GPU went from 50 °C to 90 °C in about 53 s and then
throttled, so the default re-measurement of most cases fell outside the 0.9–1.1 band and the derivation dropped 495
rows from the ratio material — a table cannot be built from a record like that (ADR 0115 addendum decision 8).

The sweep that produced the table runs **one case per process** (`runGeometrySweep` of `@karume/runtime/tune` with
`cases: [id]`), waits for the GPU to cool to 56 °C or below before each case (the idle floor of this machine is 55–60 °C
with the display attached), and merges the 57 per-case records into one record afterwards. Merging has to recompute the
record's `candidateKernels` over the merged rows with the same function the sweep uses (`sweepCandidateKernelsId` in
`packages/runtime/src/tune/fingerprint.ts`) — a per-case record fingerprints only its own rows, so the per-case values
are not the value of the whole set and `deriveGeometryProfile` rejects them.

| Record | Cases | Rows | Default re-measurement drift (min / median / max) | Dropped |
| --- | --: | --: | --- | --- |
| `outputs/bench/geometry-sweep-node-percase-2026-10-23T14-19-39.869Z.json` (sha256 `f0ee26d4…`) | 57 | 521 | ×0.922 / ×1.017 / ×1.115 | 1 case (`i8a8-attention-qk-self-m1024-n1024`, ×1.115) |

No row failed and every row's output is bit-identical to the default's (`identicalToDefault` on all 521 rows). The
dropped case leaves the i8a8 ①QK class decided by its other three cases, all of which keep the default.

## 3. Adoptions

| Class | Default | Adopted | Ratio (geometric mean, range) |
| --- | --- | --- | --- |
| `gemmRows` 129–256 (linear / matmul / bmm) | `reg64x32r4x4w8` | `reg64x64r4x4w16` | ×1.118 (×1.109–1.125, 4 cases) |
| attention f32 ③PV | `reg128x128r8x8w16` | `reg64x64r4x4w16` | ×1.095 (×1.072–1.112, 4 cases) |
| i8a8 linear | `tile128x64r8x8w8x16k16` | `tile64x64r8x4w16x8k16` | ×1.160 (×1.073–1.247, 6 cases) |

Per case for the i8a8 linear (per-dispatch medians, the adopted geometry against the default): M1024 N2048 K2048 ×1.24,
M1024 N2048 K8192 ×1.24, M1024 N8192 K2048 ×1.09, M4096 N2048 K2048 ×1.18, M4096 N2048 K8192 ×1.15, M4096 N8192
K2048 ×1.08. The runner-up `tile128x64r8x4w16x16k16` (the geometry the `nvidia-blackwell` table adopts for all three
i8a8 fields) is ×1.141 on the geometric mean and faster than the adopted one on the three largest cases.

Classes that keep the default and the nearest miss: `gemmRows` 33–64 (`reg32x32r2x4w8` ×1.040), 257–512
(`reg64x64r4x4w16` ×1.029), conv2d rows64 (`igemm64x64:wg16x16` ×1.009), conv2d rows32 (`igemm128x128:wg16x16`
×1.033), i8a8 ①QK / ③PV (nothing above ×0.95). The generated file lists every rejection with the case that rejected it.

## 4. Effect on the Wan2.2 DiT (measured mode, S = 27,280, 1280×704×121 frames)

Each run is a fresh process after the GPU cooled to 45 °C, the online preset (`f16+dit8-a8-attn8-online`) with the
subgroup form of the attention kernel, two forwards per run; the second forward's per-kernel GPU time is read. The
first pair of runs (the default first, the second run started at 58 °C) was discarded: the online attention kernel,
which no table touches, moved 11.37 → 12.48 s between the two runs, so the pair measured heat, not the table.

| Kernel | Without a table (default geometries) | With `nvidia-ampere` | Ratio |
| --- | --: | --: | --: |
| i8a8 linear (310 dispatches) | 6.84 s (`tile128x64r8x8w8x16k16`) | 6.64 s (`tile64x64r8x4w16x8k16`) | ×1.030 |
| online attention (180 dispatches) | 11.53 s | 11.53 s | ×1.000 |
| forward, GPU | 20.22 s | 20.04 s | ×1.009 |
| forward, wall | 20.48 s | 20.39 s | ×1.004 |

The i8a8 linear of this DiT runs at M = 27,280 with N ≤ 14,336, past the largest sweep case (M = 4,096, ×1.08 for that
case), so ×1.03 is in line with the sweep rather than below it. The f32 adoptions (③PV, 129–256 rows) run in no Wan2.2
preset.

## 5. What this does not settle

- No other family was measured with the table. The f32 ③PV adoption applies wherever fused f32 attention runs on this
  machine; the per-profile GPU test checks its output identity on the linear cases only, the sweep on its own case set.
- The i8a8 linear class was decided on six cases up to M = 4,096; a case at the DiT's M (tens of thousands of rows)
  would tell whether the runner-up is the better geometry for long rows.
- The sweep was taken on Node + Dawn (Vulkan). Chrome on the same GPU may rank the geometries differently (the
  `nvidia-blackwell` table, taken in Chrome, adopts a different i8a8 geometry); the `match` rule does not distinguish the
  browser.

## 6. Reproduction

The short Node scripts that ran the per-case sweep, merged the records, derived the table, generated the file and ran
the DiT A/B are kept untracked in `.claude/reviews/2026-10-11_wan-attention-recon/` (`sweep-ampere-percase.mjs`,
`derive-ampere.mjs`, `gen-profile-ts.mjs`, `run-timing4.zsh` / `run-timing5.zsh`); the A/B's per-kernel records are
`2026-10-23-timing4-*.json` / `2026-10-23-timing5-*.json` in the same folder.
