/**
 * Geometry profile `apple-metal-3` (**generated — do not edit by hand**).
 *
 * A static table of tile geometries for adapter `apple / metal-3 / Apple M2`, written from the
 * sweep records by the `profile` command of tools/geometry-sweep (removed in 0.15) (perf-ledger
 * K-71). The runtime only selects this table by the adapter's `match` (exact match on vendor /
 * architecture / description).
 * Nothing is measured at run time (no autotuning — ADR 0022 decision 3). To change a value, take the
 * sweep again and regenerate the table.
 *
 * Regeneration: the generator (tools/geometry-sweep) was removed in 0.15 and is to be rebuilt as a
 * browser application (ADR 0127 decision 5). Until then this file is maintained by hand, and the data
 * must not be edited without a new sweep. The command that generated it (from the repository root;
 * adding `--check` only checked that regeneration is byte-identical):
 *
 *   deno run -A tools/geometry-sweep/main.ts profile \
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-09-27T16-28-00.787Z.json \
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json \
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-09-29T13-08-11.329Z.json \
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json \
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-10-01T12-28-47.052Z.json \
 *     --id apple-metal-3 --vendor apple --architecture metal-3 --description 'Apple M2' \
 *     --out packages/runtime/src/kernels/geometry-profiles/apple-metal-3.ts --min-speedup 1.05
 *
 * Sweeps (adapter apple / metal-3 / 0x0000 / Apple M2):
 *
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-27T16-28-00.787Z.json (sha256 921e89ba70b34ef68ec21c153c2a2c313c80bbbc98d53da720942c2919ead034, 2026-09-27T16:28:00.787Z)
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json (sha256 f464ca82eb61f4c8fd60efa06365815eb25cec5e1f8ed905cccd242cadacfc2a, 2026-09-27T18:31:46.471Z)
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-29T13-08-11.329Z.json (sha256 38d97dac5f04a7e80fb6591ab437da7dd2b3cee66b0d03e40cd1475c0f398550, 2026-09-29T13:08:11.329Z)
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json (sha256 d9e5e7814370c741562819474f5e0837012d05f233869688d28bb91902ac8621, 2026-09-29T21:04:14.586Z)
 * - outputs/bench-browser/geometry-sweep-browser-2026-10-01T12-28-47.052Z.json (sha256 11fed6a888c91d5e5217a8cd1205dbd46b5ffa2c639dcf2f2af9d17be962eefd, 2026-10-01T12:28:47.052Z)
 *
 * Adoption criteria: among the geometries whose output matches the default on every case of the class
 * and whose ratio to the default is ×1.050 or more, the one with the largest geometric mean across the
 * cases. If there is none, the default (the geometry of the sweep's default row). When several sweeps
 * measured the same case, the ratio is the geometric mean of those observations. gemmRows is decided
 * from the sweep's linear / matmul / bmm cases, and the same table applies to all three paths.
 * Input check: per sweep, a case whose default re-measurement ratio
 * (cases[].defaultRepeat.driftRatio) is outside 0.9–1.1, or whose re-measurement failed / is missing,
 * is dropped from that sweep's ratio material (output match and failures are still checked — a
 * geometry that mismatches / fails in the dropped sweep is not adopted. The ratio is judged from
 * another sweep if one measured the same case; if none remains, the case counts as unmeasured).
 * Dropped cases are shown per adoption entry by a "sweep …" line.
 * Rounding check: for a sweep with rounded timestamps (Chrome's 100 µs without the flag), each
 * observation (one row within one sweep) gets an upper bound on the ratio's rounding error,
 * E = e(row) + e(default row) (e = tick ÷ the smallest round), and an observation with E above 1% is
 * dropped from that sweep's ratio material (output match and failures are still checked). For a case
 * whose default row's e exceeds the bound (or cannot be computed), every observation is dropped.
 * Dropped observations are shown by a "sweep …: <geometry> … rounding-error upper bound E …" line.
 *
 * Adoption:
 *
 * - gemmRows ≤ 16 (linear / matmul / bmm with row count ≤ 16, 4 cases): default reg16x16r1x4w4 kept (no geometry matches the output and reaches ×1.050 on every case)
 *   - reg128x128r8x8w16: ×0.077 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg128x16r8x4w4: ×0.159 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg128x32r8x4w8: ×0.206 < ×1.050 on matmul-m16-n3072-k1024
 *   - reg128x32r8x8w4: ×0.168 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg128x64r8x4w16: ×0.186 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg128x64r8x8w8: ×0.143 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg16x128r4x8w16: ×0.336 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg16x16r2x4w4: ×0.796 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg16x16r4x4w4: ×0.463 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg16x32r1x8w4: ×0.730 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg16x32r2x4w8: ×0.895 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg16x32r2x8w4: ×0.542 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg16x32r4x4w8: ×0.552 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg16x32r4x8w4: ×0.350 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg16x64r2x8w8: ×0.582 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg16x64r4x4w16: ×0.588 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg16x64r4x8w8: ×0.362 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg32x128r4x8w16: ×0.304 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg32x128r8x8w16: ×0.198 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg32x16r2x4w4: ×0.685 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg32x16r4x4w4: ×0.494 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg32x16r8x4w4: ×0.297 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg32x32r2x4w8: ×0.763 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg32x32r2x8w4: ×0.542 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg32x32r4x4w8: ×0.585 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg32x32r4x8w4: ×0.408 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg32x32r8x4w8: ×0.337 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg32x32r8x8w4: ×0.198 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg32x64r2x8w8: ×0.481 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg32x64r4x4w16: ×0.504 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg32x64r4x8w8: ×0.391 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg32x64r8x4w16: ×0.362 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg32x64r8x8w8: ×0.204 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg4x16r1x4w4: ×0.201 < ×1.050 on linear-m16-n3072-k1024
 *   - reg4x32r1x8w4: ×0.123 < ×1.050 on linear-m16-n3072-k1024
 *   - reg64x128r4x8w16: ×0.154 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg64x128r8x8w16: ×0.141 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg64x16r4x4w4: ×0.365 < ×1.050 on matmul-m16-n3072-k1024
 *   - reg64x16r8x4w4: ×0.261 < ×1.050 on matmul-m16-n3072-k1024
 *   - reg64x32r4x4w8: ×0.424 < ×1.050 on matmul-m16-n3072-k1024
 *   - reg64x32r4x8w4: ×0.340 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg64x32r8x4w8: ×0.314 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg64x32r8x8w4: ×0.199 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg64x64r4x4w16: ×0.312 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg64x64r4x8w8: ×0.294 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg64x64r8x4w16: ×0.268 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg64x64r8x8w8: ×0.200 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg8x16r1x4w4: ×0.594 < ×1.050 on linear-m16-n3072-k1024
 *   - reg8x16r2x4w4: ×0.316 < ×1.050 on linear-m16-n3072-k1024
 *   - reg8x32r1x8w4: ×0.429 < ×1.050 on linear-m16-n3072-k1024
 *   - reg8x32r2x4w8: ×0.450 < ×1.050 on linear-m16-n3072-k1024
 *   - reg8x32r2x8w4: ×0.233 < ×1.050 on linear-m16-n3072-k1024
 *   - reg8x64r2x8w8: ×0.237 < ×1.050 on linear-m16-n3072-k1024
 * - gemmRows 17–32 (linear / matmul / bmm with row count 17–32, 4 cases): adopted reg32x32r2x4w8 ×1.308 (×1.153–×1.633)
 *   - reg128x128r8x8w16: ×0.118 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg128x16r8x4w4: ×0.249 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg128x32r8x4w8: ×0.340 < ×1.050 on matmul-m32-n3072-k1024
 *   - reg128x32r8x8w4: ×0.253 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg128x64r8x4w16: ×0.280 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg128x64r8x8w8: ×0.220 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg16x128r4x8w16: ×0.458 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg16x16r2x4w4: ×0.983 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg16x16r4x4w4: ×0.493 < ×1.050 on linear-m32-n3072-k1024
 *   - reg16x32r1x8w4: ×0.872 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg16x32r2x4w8: geometric mean ×1.098 (short of the adopted ×1.308)
 *   - reg16x32r2x8w4: ×0.695 < ×1.050 on linear-m32-n3072-k1024
 *   - reg16x32r4x4w8: ×0.708 < ×1.050 on linear-m32-n3072-k1024
 *   - reg16x32r4x8w4: ×0.380 < ×1.050 on linear-m32-n3072-k1024
 *   - reg16x64r2x8w8: ×0.704 < ×1.050 on linear-m32-n3072-k1024
 *   - reg16x64r4x4w16: ×0.719 < ×1.050 on linear-m32-n3072-k1024
 *   - reg16x64r4x8w8: ×0.396 < ×1.050 on linear-m32-n3072-k1024
 *   - reg32x128r4x8w16: ×0.462 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x128r8x8w16: ×0.310 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x16r2x4w4: ×1.020 < ×1.050 on bmm-b16-m32-n64-k64
 *   - reg32x16r4x4w4: ×0.717 < ×1.050 on bmm-b16-m32-n64-k64
 *   - reg32x16r8x4w4: ×0.413 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x32r2x8w4: ×0.808 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x32r4x4w8: ×0.840 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x32r4x8w4: ×0.601 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x32r8x4w8: ×0.526 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x32r8x8w4: ×0.286 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x64r2x8w8: ×0.725 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x64r4x4w16: ×0.774 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x64r4x8w8: ×0.616 < ×1.050 on bmm-b16-m32-n64-k64
 *   - reg32x64r8x4w16: ×0.545 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x64r8x8w8: ×0.318 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg4x16r1x4w4: ×0.185 < ×1.050 on linear-m32-n3072-k1024
 *   - reg4x32r1x8w4: ×0.124 < ×1.050 on linear-m32-n3072-k1024
 *   - reg64x128r4x8w16: ×0.235 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg64x128r8x8w16: ×0.217 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg64x16r4x4w4: ×0.583 < ×1.050 on matmul-m32-n3072-k1024
 *   - reg64x16r8x4w4: ×0.385 < ×1.050 on matmul-m32-n3072-k1024
 *   - reg64x32r4x4w8: ×0.636 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg64x32r4x8w4: ×0.538 < ×1.050 on bmm-b16-m32-n64-k64
 *   - reg64x32r8x4w8: ×0.489 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg64x32r8x8w4: ×0.301 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg64x64r4x4w16: ×0.478 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg64x64r4x8w8: ×0.448 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg64x64r8x4w16: ×0.410 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg64x64r8x8w8: ×0.305 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg8x16r1x4w4: ×0.559 < ×1.050 on linear-m32-n3072-k1024
 *   - reg8x16r2x4w4: ×0.317 < ×1.050 on linear-m32-n3072-k1024
 *   - reg8x32r1x8w4: ×0.386 < ×1.050 on linear-m32-n3072-k1024
 *   - reg8x32r2x4w8: ×0.406 < ×1.050 on linear-m32-n3072-k1024
 *   - reg8x32r2x8w4: ×0.209 < ×1.050 on linear-m32-n3072-k1024
 *   - reg8x64r2x8w8: ×0.281 < ×1.050 on linear-m32-n3072-k1024
 * - gemmRows 33–64 (linear / matmul / bmm with row count 33–64, 4 cases): adopted reg64x32r4x4w8 ×1.412 (×1.142–×2.236)
 *   - sweep outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json: bmm-b16-m64-n128-k64 excluded from the ratio material because the default's re-measurement ratio ×0.800 is out of range (output match and failures are still checked)
 *   - reg128x128r8x8w16: ×0.198 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg128x16r8x4w4: ×0.455 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg128x32r8x4w8: ×0.677 < ×1.050 on matmul-m64-n3072-k1024
 *   - reg128x32r8x8w4: ×0.468 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg128x64r8x4w16: ×0.490 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg128x64r8x8w8: ×0.410 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg16x128r4x8w16: ×0.505 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg16x16r2x4w4: ×0.814 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg16x16r4x4w4: ×0.529 < ×1.050 on linear-m64-n3072-k1024
 *   - reg16x32r1x8w4: ×0.880 < ×1.050 on bmm-b16-m64-n128-k64
 *   - reg16x32r2x4w8: geometric mean ×1.144 (short of the adopted ×1.412)
 *   - reg16x32r2x8w4: ×0.686 < ×1.050 on linear-m64-n3072-k1024
 *   - reg16x32r4x4w8: ×0.701 < ×1.050 on linear-m64-n3072-k1024
 *   - reg16x32r4x8w4: ×0.377 < ×1.050 on linear-m64-n3072-k1024
 *   - reg16x64r2x8w8: ×0.879 < ×1.050 on linear-m64-n3072-k1024
 *   - reg16x64r4x4w16: ×0.903 < ×1.050 on linear-m64-n3072-k1024
 *   - reg16x64r4x8w8: ×0.516 < ×1.050 on linear-m64-n3072-k1024
 *   - reg32x128r4x8w16: ×0.653 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x128r8x8w16: ×0.494 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x16r2x4w4: geometric mean ×1.201 (short of the adopted ×1.412)
 *   - reg32x16r4x4w4: ×0.834 < ×1.050 on bmm-b16-m64-n128-k64
 *   - reg32x16r8x4w4: ×0.477 < ×1.050 on matmul-m64-n3072-k1024
 *   - reg32x32r2x4w8: geometric mean ×1.360 (short of the adopted ×1.412)
 *   - reg32x32r2x8w4: geometric mean ×1.228 (short of the adopted ×1.412)
 *   - reg32x32r4x4w8: geometric mean ×1.282 (short of the adopted ×1.412)
 *   - reg32x32r4x8w4: ×0.905 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x32r8x4w8: ×0.819 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x32r8x8w4: ×0.395 < ×1.050 on linear-m64-n3072-k1024
 *   - reg32x64r2x8w8: geometric mean ×1.305 (short of the adopted ×1.412)
 *   - reg32x64r4x4w16: geometric mean ×1.350 (short of the adopted ×1.412)
 *   - reg32x64r4x8w8: ×0.948 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x64r8x4w16: ×0.899 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x64r8x8w8: ×0.568 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg4x16r1x4w4: ×0.189 < ×1.050 on linear-m64-n3072-k1024
 *   - reg4x32r1x8w4: ×0.130 < ×1.050 on linear-m64-n3072-k1024
 *   - reg64x128r4x8w16: ×0.453 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x128r8x8w16: ×0.411 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x16r4x4w4: ×1.023 < ×1.050 on bmm-b16-m64-n128-k64
 *   - reg64x16r8x4w4: ×0.624 < ×1.050 on matmul-m64-n3072-k1024
 *   - reg64x32r4x8w4: ×0.929 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x32r8x4w8: ×0.882 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x32r8x8w4: ×0.546 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x64r4x4w16: ×0.906 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x64r4x8w8: ×0.824 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x64r8x4w16: ×0.781 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x64r8x8w8: ×0.572 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg8x16r1x4w4: ×0.601 < ×1.050 on linear-m64-n3072-k1024
 *   - reg8x16r2x4w4: ×0.327 < ×1.050 on linear-m64-n3072-k1024
 *   - reg8x32r1x8w4: ×0.428 < ×1.050 on linear-m64-n3072-k1024
 *   - reg8x32r2x4w8: ×0.453 < ×1.050 on linear-m64-n3072-k1024
 *   - reg8x32r2x8w4: ×0.233 < ×1.050 on linear-m64-n3072-k1024
 *   - reg8x64r2x8w8: ×0.312 < ×1.050 on linear-m64-n3072-k1024
 * - gemmRows 65–128 (linear / matmul / bmm with row count 65–128, 4 cases): default reg64x32r4x4w8 kept (no geometry matches the output and reaches ×1.050 on every case)
 *   - reg128x128r8x8w16: ×0.322 < ×1.050 on bmm-b16-m128-n64-k128
 *   - reg128x16r8x4w4: ×0.596 < ×1.050 on bmm-b16-m128-n64-k64
 *   - reg128x32r8x4w8: ×0.910 < ×1.050 on matmul-m128-n2048-k1024
 *   - reg128x32r8x8w4: ×0.627 < ×1.050 on matmul-m128-n2048-k1024
 *   - reg128x64r8x4w16: ×0.759 < ×1.050 on bmm-b16-m128-n64-k64
 *   - reg128x64r8x8w8: ×0.609 < ×1.050 on bmm-b16-m128-n64-k128
 *   - reg16x128r4x8w16: ×0.299 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x16r1x4w4: ×0.442 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x16r2x4w4: ×0.453 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x16r4x4w4: ×0.230 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x32r1x8w4: ×0.435 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x32r2x4w8: ×0.484 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x32r2x8w4: ×0.320 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x32r4x4w8: ×0.330 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x32r4x8w4: ×0.177 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x64r2x8w8: ×0.392 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x64r4x4w16: ×0.405 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x64r4x8w8: ×0.230 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x128r4x8w16: ×0.553 < ×1.050 on bmm-b16-m128-n64-k128
 *   - reg32x128r8x8w16: ×0.376 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x16r2x4w4: ×0.652 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x16r4x4w4: ×0.479 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x16r8x4w4: ×0.267 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x32r2x4w8: ×0.743 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x32r2x8w4: ×0.680 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x32r4x4w8: ×0.729 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x32r4x8w4: ×0.470 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x32r8x4w8: ×0.464 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x32r8x8w4: ×0.225 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x64r2x8w8: ×0.739 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x64r4x4w16: ×0.783 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x64r4x8w8: ×0.560 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x64r8x4w16: ×0.660 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x64r8x8w8: ×0.362 < ×1.050 on linear-m128-n2048-k1024
 *   - reg4x16r1x4w4: ×0.084 < ×1.050 on linear-m128-n2048-k1024
 *   - reg4x32r1x8w4: ×0.059 < ×1.050 on linear-m128-n2048-k1024
 *   - reg64x128r4x8w16: ×0.507 < ×1.050 on bmm-b16-m128-n64-k64
 *   - reg64x128r8x8w16: ×0.486 < ×1.050 on bmm-b16-m128-n64-k128
 *   - reg64x16r4x4w4: ×0.743 < ×1.050 on linear-m128-n2048-k1024
 *   - reg64x16r8x4w4: ×0.459 < ×1.050 on linear-m128-n2048-k1024
 *   - reg64x32r4x8w4: ×0.889 < ×1.050 on matmul-m128-n2048-k1024
 *   - reg64x32r8x4w8: ×0.896 < ×1.050 on bmm-b16-m128-n64-k64
 *   - reg64x32r8x8w4: ×0.507 < ×1.050 on linear-m128-n2048-k1024
 *   - reg64x64r4x4w16: ×1.004 < ×1.050 on matmul-m128-n2048-k1024
 *   - reg64x64r4x8w8: ×0.930 < ×1.050 on linear-m128-n2048-k1024
 *   - reg64x64r8x4w16: ×0.938 < ×1.050 on linear-m128-n2048-k1024
 *   - reg64x64r8x8w8: ×0.555 < ×1.050 on linear-m128-n2048-k1024
 *   - reg8x16r1x4w4: ×0.271 < ×1.050 on linear-m128-n2048-k1024
 *   - reg8x16r2x4w4: ×0.149 < ×1.050 on linear-m128-n2048-k1024
 *   - reg8x32r1x8w4: ×0.196 < ×1.050 on linear-m128-n2048-k1024
 *   - reg8x32r2x4w8: ×0.208 < ×1.050 on linear-m128-n2048-k1024
 *   - reg8x32r2x8w4: ×0.107 < ×1.050 on linear-m128-n2048-k1024
 *   - reg8x64r2x8w8: ×0.141 < ×1.050 on linear-m128-n2048-k1024
 * - gemmRows 129–256 (linear / matmul / bmm with row count 129–256, 4 cases): default reg64x32r4x4w8 kept (no geometry matches the output and reaches ×1.050 on every case)
 *   - reg128x128r8x8w16: ×0.468 < ×1.050 on bmm-b16-m256-n64-k64
 *   - reg128x16r8x4w4: ×0.588 < ×1.050 on matmul-m256-n2048-k1024
 *   - reg128x32r8x4w8: ×0.911 < ×1.050 on matmul-m256-n2048-k1024
 *   - reg128x32r8x8w4: ×0.644 < ×1.050 on bmm-b16-m256-n64-k64
 *   - reg128x64r8x4w16: ×0.987 < ×1.050 on bmm-b16-m256-n64-k64
 *   - reg128x64r8x8w8: ×0.702 < ×1.050 on bmm-b16-m256-n64-k64
 *   - reg16x128r4x8w16: ×0.288 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x16r1x4w4: ×0.425 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x16r2x4w4: ×0.450 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x16r4x4w4: ×0.235 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x32r1x8w4: ×0.443 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x32r2x4w8: ×0.480 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x32r2x8w4: ×0.335 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x32r4x4w8: ×0.346 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x32r4x8w4: ×0.175 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x64r2x8w8: ×0.414 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x64r4x4w16: ×0.431 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x64r4x8w8: ×0.249 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x128r4x8w16: ×0.543 < ×1.050 on bmm-b16-m256-n64-k128
 *   - reg32x128r8x8w16: ×0.424 < ×1.050 on bmm-b16-m256-n64-k128
 *   - reg32x16r2x4w4: ×0.651 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x16r4x4w4: ×0.498 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x16r8x4w4: ×0.278 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x32r2x4w8: ×0.735 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x32r2x8w4: ×0.706 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x32r4x4w8: ×0.762 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x32r4x8w4: ×0.524 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x32r8x4w8: ×0.514 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x32r8x8w4: ×0.218 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x64r2x8w8: ×0.716 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x64r4x4w16: ×0.799 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x64r4x8w8: ×0.587 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x64r8x4w16: ×0.638 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x64r8x8w8: ×0.348 < ×1.050 on linear-m256-n2048-k1024
 *   - reg4x16r1x4w4: ×0.081 < ×1.050 on linear-m256-n2048-k1024
 *   - reg4x32r1x8w4: ×0.057 < ×1.050 on linear-m256-n2048-k1024
 *   - reg64x128r4x8w16: ×0.485 < ×1.050 on bmm-b16-m256-n64-k128
 *   - reg64x128r8x8w16: ×0.469 < ×1.050 on bmm-b16-m256-n64-k128
 *   - reg64x16r4x4w4: ×0.769 < ×1.050 on linear-m256-n2048-k1024
 *   - reg64x16r8x4w4: ×0.493 < ×1.050 on linear-m256-n2048-k1024
 *   - reg64x32r4x8w4: ×0.852 < ×1.050 on linear-m256-n2048-k1024
 *   - reg64x32r8x4w8: ×0.862 < ×1.050 on linear-m256-n2048-k1024
 *   - reg64x32r8x8w4: ×0.489 < ×1.050 on linear-m256-n2048-k1024
 *   - reg64x64r4x4w16: ×0.984 < ×1.050 on linear-m256-n2048-k1024
 *   - reg64x64r4x8w8: ×0.897 < ×1.050 on linear-m256-n2048-k1024
 *   - reg64x64r8x4w16: ×0.907 < ×1.050 on linear-m256-n2048-k1024
 *   - reg64x64r8x8w8: ×0.669 < ×1.050 on linear-m256-n2048-k1024
 *   - reg8x16r1x4w4: ×0.262 < ×1.050 on linear-m256-n2048-k1024
 *   - reg8x16r2x4w4: ×0.142 < ×1.050 on linear-m256-n2048-k1024
 *   - reg8x32r1x8w4: ×0.196 < ×1.050 on linear-m256-n2048-k1024
 *   - reg8x32r2x4w8: ×0.208 < ×1.050 on linear-m256-n2048-k1024
 *   - reg8x32r2x8w4: ×0.107 < ×1.050 on linear-m256-n2048-k1024
 *   - reg8x64r2x8w8: ×0.141 < ×1.050 on linear-m256-n2048-k1024
 * - gemmRows 257–512 (linear / matmul / bmm with row count 257–512, 5 cases): default reg64x32r4x4w8 kept (no geometry matches the output and reaches ×1.050 on every case)
 *   - sweep outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json: bmm-b16-m512-n512-k64 excluded from the ratio material because the default's re-measurement ratio ×1.176 is out of range (output match and failures are still checked)
 *   - reg128x128r8x8w16: ×0.383 < ×1.050 on bmm-b16-m512-n64-k512
 *   - reg128x16r8x4w4: ×0.661 < ×1.050 on bmm-b16-m512-n64-k512
 *   - reg128x32r8x4w8: ×0.974 < ×1.050 on bmm-b16-m512-n64-k512
 *   - reg128x32r8x8w4: ×0.748 < ×1.050 on bmm-b16-m512-n64-k64
 *   - reg128x64r8x4w16: ×0.987 < ×1.050 on matmul-m512-n2048-k1024
 *   - reg128x64r8x8w8: ×0.799 < ×1.050 on bmm-b16-m512-n512-k64
 *   - reg16x128r4x8w16: ×0.307 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x16r1x4w4: ×0.426 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x16r2x4w4: ×0.447 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x16r4x4w4: ×0.231 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x32r1x8w4: ×0.452 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x32r2x4w8: ×0.480 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x32r2x8w4: ×0.347 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x32r4x4w8: ×0.361 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x32r4x8w4: ×0.180 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x64r2x8w8: ×0.429 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x64r4x4w16: ×0.447 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x64r4x8w8: ×0.264 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x128r4x8w16: ×0.504 < ×1.050 on bmm-b16-m512-n64-k512
 *   - reg32x128r8x8w16: ×0.421 < ×1.050 on bmm-b16-m512-n64-k512
 *   - reg32x16r2x4w4: ×0.653 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x16r4x4w4: ×0.518 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x16r8x4w4: ×0.290 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x32r2x4w8: ×0.738 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x32r2x8w4: ×0.704 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x32r4x4w8: ×0.759 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x32r4x8w4: ×0.565 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x32r8x4w8: ×0.565 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x32r8x8w4: ×0.229 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x64r2x8w8: ×0.737 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x64r4x4w16: ×0.795 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x64r4x8w8: ×0.613 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x64r8x4w16: ×0.656 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x64r8x8w8: ×0.387 < ×1.050 on linear-m512-n2048-k1024
 *   - reg4x16r1x4w4: ×0.080 < ×1.050 on linear-m512-n2048-k1024
 *   - reg4x32r1x8w4: ×0.057 < ×1.050 on linear-m512-n2048-k1024
 *   - reg64x128r4x8w16: ×0.476 < ×1.050 on bmm-b16-m512-n64-k512
 *   - reg64x128r8x8w16: ×0.438 < ×1.050 on bmm-b16-m512-n64-k512
 *   - reg64x16r4x4w4: ×0.724 < ×1.050 on bmm-b16-m512-n64-k512
 *   - reg64x16r8x4w4: ×0.508 < ×1.050 on linear-m512-n2048-k1024
 *   - reg64x32r4x8w4: ×0.868 < ×1.050 on linear-m512-n2048-k1024
 *   - reg64x32r8x4w8: ×0.883 < ×1.050 on linear-m512-n2048-k1024
 *   - reg64x32r8x8w4: ×0.540 < ×1.050 on linear-m512-n2048-k1024
 *   - reg64x64r4x4w16: ×0.929 < ×1.050 on bmm-b16-m512-n64-k512
 *   - reg64x64r4x8w8: ×0.908 < ×1.050 on linear-m512-n2048-k1024
 *   - reg64x64r8x4w16: ×0.952 < ×1.050 on bmm-b16-m512-n64-k512
 *   - reg64x64r8x8w8: ×0.694 < ×1.050 on linear-m512-n2048-k1024
 *   - reg8x16r1x4w4: ×0.264 < ×1.050 on linear-m512-n2048-k1024
 *   - reg8x16r2x4w4: ×0.141 < ×1.050 on linear-m512-n2048-k1024
 *   - reg8x32r1x8w4: ×0.198 < ×1.050 on linear-m512-n2048-k1024
 *   - reg8x32r2x4w8: ×0.210 < ×1.050 on linear-m512-n2048-k1024
 *   - reg8x32r2x8w4: ×0.108 < ×1.050 on linear-m512-n2048-k1024
 *   - reg8x64r2x8w8: ×0.142 < ×1.050 on linear-m512-n2048-k1024
 * - gemmRows > 512 (linear / matmul / bmm with row count > 512, 7 cases): adopted reg128x32r8x4w8 ×1.513 (×1.145–×1.600)
 *   - sweep outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json: matmul-m4096-n2048-k2048 excluded from the ratio material because the default's re-measurement ratio ×1.160 is out of range (output match and failures are still checked)
 *   - reg128x16r8x4w4: ×0.704 < ×1.050 on matmul-m4096-n2048-k2048
 *   - reg128x32r8x8w4: ×0.917 < ×1.050 on matmul-m4096-n2048-k2048
 *   - reg128x64r8x4w16: geometric mean ×1.389 (short of the adopted ×1.513)
 *   - reg128x64r8x8w8: ×1.002 < ×1.050 on matmul-m4096-n2048-k2048
 *   - reg16x128r4x8w16: ×0.435 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg16x16r1x4w4: ×0.607 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg16x16r2x4w4: ×0.635 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg16x16r4x4w4: ×0.319 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg16x32r1x8w4: ×0.649 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg16x32r2x4w8: ×0.682 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg16x32r2x8w4: ×0.498 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg16x32r4x4w8: ×0.512 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg16x32r4x8w4: ×0.254 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg16x64r2x8w8: ×0.614 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg16x64r4x4w16: ×0.624 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg16x64r4x8w8: ×0.373 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg32x128r4x8w16: ×0.913 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg32x128r8x8w16: ×0.702 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg32x16r2x4w4: ×0.923 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg32x16r4x4w4: ×0.723 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg32x16r8x4w4: ×0.400 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg32x32r2x4w8: geometric mean ×1.073 (short of the adopted ×1.513)
 *   - reg32x32r2x8w4: ×1.002 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg32x32r4x4w8: geometric mean ×1.114 (short of the adopted ×1.513)
 *   - reg32x32r4x8w4: ×0.827 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg32x32r8x4w8: ×0.824 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg32x32r8x8w4: ×0.320 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg32x64r2x8w8: ×1.007 < ×1.050 on matmul-m4096-n2048-k2048
 *   - reg32x64r4x4w16: geometric mean ×1.153 (short of the adopted ×1.513)
 *   - reg32x64r4x8w8: ×0.898 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg32x64r8x4w16: ×0.963 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg32x64r8x8w8: ×0.584 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg4x16r1x4w4: ×0.113 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg4x32r1x8w4: ×0.080 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg64x128r4x8w16: geometric mean ×1.180 (short of the adopted ×1.513)
 *   - reg64x128r8x8w16: ×0.995 < ×1.050 on matmul-m4096-n2048-k2048
 *   - reg64x16r4x4w4: ×0.986 < ×1.050 on matmul-m4096-n2048-k2048
 *   - reg64x16r8x4w4: ×0.621 < ×1.050 on matmul-m4096-n2048-k2048
 *   - reg64x32r4x4w8: geometric mean ×1.426 (short of the adopted ×1.513)
 *   - reg64x32r4x8w4: geometric mean ×1.261 (short of the adopted ×1.513)
 *   - reg64x32r8x4w8: geometric mean ×1.293 (short of the adopted ×1.513)
 *   - reg64x32r8x8w4: ×0.793 < ×1.050 on matmul-m4096-n2048-k2048
 *   - reg64x64r4x4w16: geometric mean ×1.399 (short of the adopted ×1.513)
 *   - reg64x64r4x8w8: geometric mean ×1.282 (short of the adopted ×1.513)
 *   - reg64x64r8x4w16: geometric mean ×1.410 (short of the adopted ×1.513)
 *   - reg64x64r8x8w8: ×1.008 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg8x16r1x4w4: ×0.367 < ×1.050 on linear-m1024-n2048-k2048
 *   - reg8x16r2x4w4: ×0.196 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg8x32r1x8w4: ×0.280 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg8x32r2x4w8: ×0.300 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg8x32r2x8w4: ×0.153 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg8x64r2x8w8: ×0.197 < ×1.050 on linear-m1024-n2048-k8192
 * - attention.qk (fused attention f32 ①QK, 4 cases): adopted reg128x32r8x4w8 ×1.527 (×1.470–×1.564)
 *   - reg128x16r8x4w4: ×0.999 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg128x32r8x8w4: geometric mean ×1.145 (short of the adopted ×1.527)
 *   - reg128x64r8x4w16: geometric mean ×1.394 (short of the adopted ×1.527)
 *   - reg128x64r8x8w8: geometric mean ×1.207 (short of the adopted ×1.527)
 *   - reg16x128r4x8w16: ×0.434 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x16r1x4w4: ×0.574 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x16r2x4w4: ×0.621 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg16x16r4x4w4: ×0.340 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x32r1x8w4: ×0.639 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg16x32r2x4w8: ×0.679 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg16x32r2x8w4: ×0.507 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x32r4x4w8: ×0.530 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x32r4x8w4: ×0.255 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x64r2x8w8: ×0.617 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x64r4x4w16: ×0.623 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg16x64r4x8w8: ×0.370 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x128r4x8w16: ×0.933 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x128r8x8w16: ×0.691 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x16r2x4w4: ×0.887 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg32x16r4x4w4: ×0.730 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x16r8x4w4: ×0.409 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x32r2x4w8: ×0.997 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg32x32r2x8w4: ×0.975 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg32x32r4x4w8: geometric mean ×1.091 (short of the adopted ×1.527)
 *   - reg32x32r4x8w4: ×0.833 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg32x32r8x4w8: ×0.852 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg32x32r8x8w4: ×0.329 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x64r2x8w8: ×1.008 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg32x64r4x4w16: geometric mean ×1.108 (short of the adopted ×1.527)
 *   - reg32x64r4x8w8: ×0.948 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x64r8x4w16: ×0.952 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg32x64r8x8w8: ×0.595 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg4x16r1x4w4: ×0.115 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg4x32r1x8w4: ×0.081 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg64x128r4x8w16: geometric mean ×1.137 (short of the adopted ×1.527)
 *   - reg64x128r8x8w16: ×1.005 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg64x16r4x4w4: ×1.042 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg64x16r8x4w4: ×0.723 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg64x32r4x4w8: geometric mean ×1.420 (short of the adopted ×1.527)
 *   - reg64x32r4x8w4: geometric mean ×1.233 (short of the adopted ×1.527)
 *   - reg64x32r8x4w8: geometric mean ×1.273 (short of the adopted ×1.527)
 *   - reg64x32r8x8w4: ×0.812 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg64x64r4x4w16: geometric mean ×1.387 (short of the adopted ×1.527)
 *   - reg64x64r4x8w8: geometric mean ×1.271 (short of the adopted ×1.527)
 *   - reg64x64r8x4w16: geometric mean ×1.408 (short of the adopted ×1.527)
 *   - reg64x64r8x8w8: geometric mean ×1.112 (short of the adopted ×1.527)
 *   - reg8x16r1x4w4: ×0.363 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg8x16r2x4w4: ×0.199 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg8x32r1x8w4: ×0.289 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg8x32r2x4w8: ×0.304 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg8x32r2x8w4: ×0.156 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg8x64r2x8w8: ×0.202 < ×1.050 on attention-qk-self-m4096-n4096
 * - attention.pv (fused attention f32 ③PV, 4 cases): adopted reg64x64r4x8w8 ×1.661 (×1.542–×1.712)
 *   - sweep outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json: attention-pv-cross-m1024-n512 excluded from the ratio material because the default's re-measurement ratio ×1.126 is out of range (output match and failures are still checked)
 *   - sweep outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json: attention-pv-self-m1024-n1024 excluded from the ratio material because the default's re-measurement ratio ×1.228 is out of range (output match and failures are still checked)
 *   - reg128x16r8x4w4: ×0.959 < ×1.050 on attention-pv-cross-m4096-n512
 *   - reg128x32r8x4w8: geometric mean ×1.504 (short of the adopted ×1.661)
 *   - reg128x32r8x8w4: geometric mean ×1.165 (short of the adopted ×1.661)
 *   - reg128x64r8x4w16: geometric mean ×1.523 (short of the adopted ×1.661)
 *   - reg128x64r8x8w8: geometric mean ×1.290 (short of the adopted ×1.661)
 *   - reg16x128r4x8w16: geometric mean ×1.196 (short of the adopted ×1.661)
 *   - reg16x16r1x4w4: ×1.050 < ×1.050 on attention-pv-cross-m4096-n512
 *   - reg16x16r2x4w4: geometric mean ×1.324 (short of the adopted ×1.661)
 *   - reg16x16r4x4w4: ×0.673 < ×1.050 on attention-pv-cross-m4096-n512
 *   - reg16x32r1x8w4: geometric mean ×1.161 (short of the adopted ×1.661)
 *   - reg16x32r2x4w8: geometric mean ×1.495 (short of the adopted ×1.661)
 *   - reg16x32r2x8w4: geometric mean ×1.153 (short of the adopted ×1.661)
 *   - reg16x32r4x4w8: geometric mean ×1.199 (short of the adopted ×1.661)
 *   - reg16x32r4x8w4: ×0.640 < ×1.050 on attention-pv-self-m4096-n4096
 *   - reg16x64r2x8w8: geometric mean ×1.352 (short of the adopted ×1.661)
 *   - reg16x64r4x4w16: geometric mean ×1.379 (short of the adopted ×1.661)
 *   - reg16x64r4x8w8: ×0.948 < ×1.050 on attention-pv-self-m4096-n4096
 *   - reg32x128r4x8w16: geometric mean ×1.612 (short of the adopted ×1.661)
 *   - reg32x128r8x8w16: geometric mean ×1.300 (short of the adopted ×1.661)
 *   - reg32x16r2x4w4: geometric mean ×1.373 (short of the adopted ×1.661)
 *   - reg32x16r4x4w4: geometric mean ×1.135 (short of the adopted ×1.661)
 *   - reg32x16r8x4w4: ×0.591 < ×1.050 on attention-pv-self-m4096-n4096
 *   - reg32x32r2x4w8: geometric mean ×1.547 (short of the adopted ×1.661)
 *   - reg32x32r2x8w4: geometric mean ×1.452 (short of the adopted ×1.661)
 *   - reg32x32r4x4w8: geometric mean ×1.524 (short of the adopted ×1.661)
 *   - reg32x32r4x8w4: geometric mean ×1.466 (short of the adopted ×1.661)
 *   - reg32x32r8x4w8: geometric mean ×1.412 (short of the adopted ×1.661)
 *   - reg32x32r8x8w4: ×0.542 < ×1.050 on attention-pv-self-m4096-n4096
 *   - reg32x64r2x8w8: geometric mean ×1.490 (short of the adopted ×1.661)
 *   - reg32x64r4x4w16: geometric mean ×1.583 (short of the adopted ×1.661)
 *   - reg32x64r4x8w8: geometric mean ×1.633 (short of the adopted ×1.661)
 *   - reg32x64r8x4w16: geometric mean ×1.582 (short of the adopted ×1.661)
 *   - reg32x64r8x8w8: geometric mean ×1.153 (short of the adopted ×1.661)
 *   - reg4x16r1x4w4: ×0.473 < ×1.050 on attention-pv-self-m4096-n4096
 *   - reg4x32r1x8w4: ×0.302 < ×1.050 on attention-pv-self-m4096-n4096
 *   - reg64x128r4x8w16: geometric mean ×1.549 (short of the adopted ×1.661)
 *   - reg64x128r8x8w16: geometric mean ×1.309 (short of the adopted ×1.661)
 *   - reg64x16r4x4w4: geometric mean ×1.270 (short of the adopted ×1.661)
 *   - reg64x16r8x4w4: ×0.856 < ×1.050 on attention-pv-self-m4096-n4096
 *   - reg64x32r4x4w8: geometric mean ×1.519 (short of the adopted ×1.661)
 *   - reg64x32r4x8w4: geometric mean ×1.597 (short of the adopted ×1.661)
 *   - reg64x32r8x4w8: geometric mean ×1.471 (short of the adopted ×1.661)
 *   - reg64x32r8x8w4: ×0.994 < ×1.050 on attention-pv-cross-m4096-n512
 *   - reg64x64r4x4w16: geometric mean ×1.556 (short of the adopted ×1.661)
 *   - reg64x64r8x4w16: geometric mean ×1.621 (short of the adopted ×1.661)
 *   - reg64x64r8x8w8: geometric mean ×1.337 (short of the adopted ×1.661)
 *   - reg8x16r1x4w4: ×0.947 < ×1.050 on attention-pv-cross-m4096-n512
 *   - reg8x16r2x4w4: ×0.644 < ×1.050 on attention-pv-cross-m4096-n512
 *   - reg8x32r1x8w4: ×0.730 < ×1.050 on attention-pv-self-m4096-n4096
 *   - reg8x32r2x4w8: ×0.853 < ×1.050 on attention-pv-self-m4096-n4096
 *   - reg8x32r2x8w4: ×0.457 < ×1.050 on attention-pv-self-m4096-n4096
 *   - reg8x64r2x8w8: ×0.612 < ×1.050 on attention-pv-self-m4096-n4096
 * - conv2d.rows64 (conv2d implicit GEMM, m tile of 64 rows, 2 cases): adopted igemm128x64:wg16x16 ×1.398 (×1.172–×1.666)
 *   - igemm128x128:wg16x16: ×0.796 < ×1.050 on conv2d-c192-256x256
 *   - igemm128x64:wg8x16: ×1.015 < ×1.050 on conv2d-c192-256x256
 *   - igemm16x128:wg16x4: ×0.413 < ×1.050 on conv2d-c384-128x128
 *   - igemm16x64:wg16x4: ×0.552 < ×1.050 on conv2d-c384-128x128
 *   - igemm16x64:wg8x4: ×0.368 < ×1.050 on conv2d-c384-128x128
 *   - igemm16x64:wg8x8: ×0.538 < ×1.050 on conv2d-c384-128x128
 *   - igemm32x128:wg16x4: ×0.648 < ×1.050 on conv2d-c384-128x128
 *   - igemm32x128:wg16x8: ×0.808 < ×1.050 on conv2d-c384-128x128
 *   - igemm32x64:wg16x4: ×0.861 < ×1.050 on conv2d-c384-128x128
 *   - igemm32x64:wg16x8: ×0.961 < ×1.050 on conv2d-c384-128x128
 *   - igemm32x64:wg8x16: ×0.965 < ×1.050 on conv2d-c384-128x128
 *   - igemm32x64:wg8x4: ×0.591 < ×1.050 on conv2d-c384-128x128
 *   - igemm32x64:wg8x8: ×0.869 < ×1.050 on conv2d-c384-128x128
 *   - igemm64x128:wg16x16: geometric mean ×1.243 (short of the adopted ×1.398)
 *   - igemm64x64:wg16x16: geometric mean ×1.372 (short of the adopted ×1.398)
 *   - igemm64x64:wg16x8: geometric mean ×1.292 (short of the adopted ×1.398)
 *   - igemm64x64:wg8x16: geometric mean ×1.295 (short of the adopted ×1.398)
 *   - igemm64x64:wg8x8: ×1.048 < ×1.050 on conv2d-c384-128x128
 *   - igemm8x64:wg8x4: ×0.195 < ×1.050 on conv2d-c384-128x128
 * - conv2d.rows32 (conv2d implicit GEMM, m tile of 32 rows, 1 case): adopted igemm128x64:wg16x16 ×1.600 (×1.600–×1.600)
 *   - igemm128x128:wg16x16: geometric mean ×1.139 (short of the adopted ×1.600)
 *   - igemm128x64:wg8x16: geometric mean ×1.427 (short of the adopted ×1.600)
 *   - igemm16x128:wg16x4: ×0.655 < ×1.050 on conv2d-c96-512x512
 *   - igemm16x64:wg16x4: ×0.885 < ×1.050 on conv2d-c96-512x512
 *   - igemm16x64:wg8x4: ×0.579 < ×1.050 on conv2d-c96-512x512
 *   - igemm16x64:wg8x8: ×0.872 < ×1.050 on conv2d-c96-512x512
 *   - igemm32x128:wg16x8: geometric mean ×1.291 (short of the adopted ×1.600)
 *   - igemm32x64:wg16x4: geometric mean ×1.369 (short of the adopted ×1.600)
 *   - igemm32x64:wg16x8: geometric mean ×1.420 (short of the adopted ×1.600)
 *   - igemm32x64:wg8x16: geometric mean ×1.401 (short of the adopted ×1.600)
 *   - igemm32x64:wg8x4: ×0.908 < ×1.050 on conv2d-c96-512x512
 *   - igemm32x64:wg8x8: geometric mean ×1.395 (short of the adopted ×1.600)
 *   - igemm64x128:wg16x16: geometric mean ×1.355 (short of the adopted ×1.600)
 *   - igemm64x128:wg16x8: geometric mean ×1.136 (short of the adopted ×1.600)
 *   - igemm64x64:wg16x16: geometric mean ×1.382 (short of the adopted ×1.600)
 *   - igemm64x64:wg16x8: geometric mean ×1.417 (short of the adopted ×1.600)
 *   - igemm64x64:wg8x16: geometric mean ×1.395 (short of the adopted ×1.600)
 *   - igemm64x64:wg8x8: geometric mean ×1.186 (short of the adopted ×1.600)
 *   - igemm8x64:wg8x4: ×0.316 < ×1.050 on conv2d-c96-512x512
 * - i8a8.linear (i8a8 linear, 6 cases): adopted tile64x64r4x8w8x16k16 ×1.116 (×1.104–×1.134)
 *   - tile128x128r8x8w16x16k16: ×0.769 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile128x128r8x8w16x16k32: ×0.813 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile128x32r8x4w8x16k16: geometric mean ×1.110 (short of the adopted ×1.116)
 *   - tile128x32r8x4w8x16k32: geometric mean ×1.091 (short of the adopted ×1.116)
 *   - tile128x64r8x4w16x16k16: geometric mean ×1.086 (short of the adopted ×1.116)
 *   - tile128x64r8x4w16x16k32: ×1.001 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile128x64r8x8w8x16k32: ×1.029 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile16x128r4x8w16x4k16: geometric mean ×1.106 (short of the adopted ×1.116)
 *   - tile16x128r4x8w16x4k32: geometric mean ×1.091 (short of the adopted ×1.116)
 *   - tile16x32r4x4w8x4k16: ×0.988 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile16x32r4x4w8x4k32: ×0.987 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile16x64r4x4w16x4k16: ×0.988 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile16x64r4x4w16x4k32: ×0.988 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile16x64r4x8w8x4k16: geometric mean ×1.105 (short of the adopted ×1.116)
 *   - tile16x64r4x8w8x4k32: geometric mean ×1.089 (short of the adopted ×1.116)
 *   - tile32x128r4x8w16x8k16: geometric mean ×1.110 (short of the adopted ×1.116)
 *   - tile32x128r4x8w16x8k32: geometric mean ×1.088 (short of the adopted ×1.116)
 *   - tile32x128r8x8w16x4k16: ×0.964 < ×1.050 on i8a8-linear-m1024-n2048-k8192
 *   - tile32x128r8x8w16x4k32: ×0.997 < ×1.050 on i8a8-linear-m1024-n2048-k8192
 *   - tile32x32r4x4w8x8k16: ×0.993 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile32x32r4x4w8x8k32: ×0.988 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile32x32r8x4w8x4k16: geometric mean ×1.111 (short of the adopted ×1.116)
 *   - tile32x32r8x4w8x4k32: geometric mean ×1.103 (short of the adopted ×1.116)
 *   - tile32x64r4x4w16x8k16: ×0.984 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile32x64r4x4w16x8k32: ×0.989 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile32x64r4x8w8x8k16: geometric mean ×1.114 (short of the adopted ×1.116)
 *   - tile32x64r4x8w8x8k32: geometric mean ×1.111 (short of the adopted ×1.116)
 *   - tile32x64r8x4w16x4k16: geometric mean ×1.113 (short of the adopted ×1.116)
 *   - tile32x64r8x4w16x4k32: geometric mean ×1.109 (short of the adopted ×1.116)
 *   - tile32x64r8x8w8x4k16: ×0.958 < ×1.050 on i8a8-linear-m1024-n2048-k8192
 *   - tile32x64r8x8w8x4k32: ×0.922 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile64x128r4x8w16x16k16: geometric mean ×1.086 (short of the adopted ×1.116)
 *   - tile64x128r4x8w16x16k32: ×0.997 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile64x128r8x8w16x8k16: ×0.982 < ×1.050 on i8a8-linear-m1024-n2048-k8192
 *   - tile64x128r8x8w16x8k32: ×1.017 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile64x32r4x4w8x16k16: ×0.990 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile64x32r4x4w8x16k32: ×0.990 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile64x32r8x4w8x8k16: geometric mean ×1.113 (short of the adopted ×1.116)
 *   - tile64x32r8x4w8x8k32: geometric mean ×1.110 (short of the adopted ×1.116)
 *   - tile64x64r4x4w16x16k16: ×0.958 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile64x64r4x4w16x16k32: ×0.967 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile64x64r4x8w8x16k32: geometric mean ×1.110 (short of the adopted ×1.116)
 *   - tile64x64r8x4w16x8k16: geometric mean ×1.113 (short of the adopted ×1.116)
 *   - tile64x64r8x4w16x8k32: geometric mean ×1.107 (short of the adopted ×1.116)
 *   - tile64x64r8x8w8x8k16: ×1.013 < ×1.050 on i8a8-linear-m1024-n2048-k8192
 *   - tile64x64r8x8w8x8k32: ×1.025 < ×1.050 on i8a8-linear-m1024-n2048-k8192
 * - i8a8.attentionQk (i8a8 attention ①QK, 4 cases): adopted tile32x64r4x8w8x8k16 ×1.157 (×1.156–×1.158)
 *   - tile128x128r8x8w16x16k16: ×0.757 < ×1.050 on i8a8-attention-qk-cross-m1024-n512
 *   - tile128x128r8x8w16x16k32: ×0.819 < ×1.050 on i8a8-attention-qk-self-m1024-n1024
 *   - tile128x32r8x4w8x16k16: geometric mean ×1.132 (short of the adopted ×1.157)
 *   - tile128x32r8x4w8x16k32: geometric mean ×1.120 (short of the adopted ×1.157)
 *   - tile128x64r8x4w16x16k16: geometric mean ×1.119 (short of the adopted ×1.157)
 *   - tile128x64r8x4w16x16k32: ×1.029 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile128x64r8x8w8x16k32: ×1.009 < ×1.050 on i8a8-attention-qk-cross-m1024-n512
 *   - tile16x128r4x8w16x4k16: geometric mean ×1.147 (short of the adopted ×1.157)
 *   - tile16x128r4x8w16x4k32: geometric mean ×1.122 (short of the adopted ×1.157)
 *   - tile16x32r4x4w8x4k16: ×1.027 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile16x32r4x4w8x4k32: ×1.014 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile16x64r4x4w16x4k16: ×1.033 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile16x64r4x4w16x4k32: ×1.026 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile16x64r4x8w8x4k16: geometric mean ×1.140 (short of the adopted ×1.157)
 *   - tile16x64r4x8w8x4k32: geometric mean ×1.113 (short of the adopted ×1.157)
 *   - tile32x128r4x8w16x8k16: geometric mean ×1.132 (short of the adopted ×1.157)
 *   - tile32x128r4x8w16x8k32: geometric mean ×1.121 (short of the adopted ×1.157)
 *   - tile32x128r8x8w16x4k16: ×0.967 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile32x128r8x8w16x4k32: ×0.953 < ×1.050 on i8a8-attention-qk-self-m4096-n4096
 *   - tile32x32r4x4w8x8k16: ×1.033 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile32x32r4x4w8x8k32: ×1.031 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile32x32r8x4w8x4k16: geometric mean ×1.153 (short of the adopted ×1.157)
 *   - tile32x32r8x4w8x4k32: geometric mean ×1.141 (short of the adopted ×1.157)
 *   - tile32x64r4x4w16x8k16: ×1.024 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile32x64r4x4w16x8k32: ×1.012 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile32x64r4x8w8x8k32: geometric mean ×1.150 (short of the adopted ×1.157)
 *   - tile32x64r8x4w16x4k16: geometric mean ×1.155 (short of the adopted ×1.157)
 *   - tile32x64r8x4w16x4k32: geometric mean ×1.147 (short of the adopted ×1.157)
 *   - tile32x64r8x8w8x4k16: ×0.953 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile32x64r8x8w8x4k32: ×0.929 < ×1.050 on i8a8-attention-qk-self-m4096-n4096
 *   - tile64x128r4x8w16x16k16: geometric mean ×1.118 (short of the adopted ×1.157)
 *   - tile64x128r4x8w16x16k32: ×1.039 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile64x128r8x8w16x8k16: ×1.004 < ×1.050 on i8a8-attention-qk-cross-m1024-n512
 *   - tile64x128r8x8w16x8k32: ×1.009 < ×1.050 on i8a8-attention-qk-cross-m1024-n512
 *   - tile64x32r4x4w8x16k16: ×1.025 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile64x32r4x4w8x16k32: ×1.012 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile64x32r8x4w8x8k16: geometric mean ×1.155 (short of the adopted ×1.157)
 *   - tile64x32r8x4w8x8k32: geometric mean ×1.147 (short of the adopted ×1.157)
 *   - tile64x64r4x4w16x16k16: ×0.989 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile64x64r4x4w16x16k32: ×0.992 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile64x64r4x8w8x16k16: geometric mean ×1.142 (short of the adopted ×1.157)
 *   - tile64x64r4x8w8x16k32: geometric mean ×1.148 (short of the adopted ×1.157)
 *   - tile64x64r8x4w16x8k16: geometric mean ×1.136 (short of the adopted ×1.157)
 *   - tile64x64r8x4w16x8k32: geometric mean ×1.147 (short of the adopted ×1.157)
 *   - tile64x64r8x8w8x8k16: ×1.020 < ×1.050 on i8a8-attention-qk-self-m4096-n4096
 *   - tile64x64r8x8w8x8k32: geometric mean ×1.072 (short of the adopted ×1.157)
 * - i8a8.attentionPv (i8a8 attention ③PV, 4 cases): adopted tile16x128r4x8w16x4k16 ×1.131 (×1.119–×1.142)
 *   - tile128x128r8x8w16x16k16: ×0.769 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 *   - tile128x128r8x8w16x16k32: ×0.814 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 *   - tile128x32r8x4w8x16k16: ×1.038 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile128x32r8x4w8x16k32: ×1.026 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile128x64r8x4w16x16k16: geometric mean ×1.084 (short of the adopted ×1.131)
 *   - tile128x64r8x4w16x16k32: ×0.985 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 *   - tile128x64r8x8w8x16k16: ×0.983 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 *   - tile128x64r8x8w8x16k32: ×1.025 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 *   - tile16x128r4x8w16x4k32: geometric mean ×1.116 (short of the adopted ×1.131)
 *   - tile16x32r4x4w8x4k16: ×0.949 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile16x32r4x4w8x4k32: ×0.943 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile16x64r4x4w16x4k16: ×0.992 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile16x64r4x4w16x4k32: ×0.991 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile16x64r4x8w8x4k16: geometric mean ×1.101 (short of the adopted ×1.131)
 *   - tile16x64r4x8w8x4k32: geometric mean ×1.083 (short of the adopted ×1.131)
 *   - tile32x128r4x8w16x8k16: geometric mean ×1.124 (short of the adopted ×1.131)
 *   - tile32x128r4x8w16x8k32: geometric mean ×1.112 (short of the adopted ×1.131)
 *   - tile32x128r8x8w16x4k16: ×0.990 < ×1.050 on i8a8-attention-pv-cross-m1024-n512
 *   - tile32x128r8x8w16x4k32: ×1.003 < ×1.050 on i8a8-attention-pv-cross-m1024-n512
 *   - tile32x32r4x4w8x8k16: ×0.955 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x32r4x4w8x8k32: ×0.951 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x32r8x4w8x4k16: ×1.046 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x32r8x4w8x4k32: ×1.036 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x64r4x4w16x8k16: ×0.986 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x64r4x4w16x8k32: ×0.979 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x64r4x8w8x8k16: geometric mean ×1.116 (short of the adopted ×1.131)
 *   - tile32x64r4x8w8x8k32: geometric mean ×1.111 (short of the adopted ×1.131)
 *   - tile32x64r8x4w16x4k16: geometric mean ×1.114 (short of the adopted ×1.131)
 *   - tile32x64r8x4w16x4k32: geometric mean ×1.108 (short of the adopted ×1.131)
 *   - tile32x64r8x8w8x4k16: ×0.965 < ×1.050 on i8a8-attention-pv-cross-m1024-n512
 *   - tile32x64r8x8w8x4k32: ×0.939 < ×1.050 on i8a8-attention-pv-cross-m1024-n512
 *   - tile64x128r4x8w16x16k16: ×1.015 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile64x128r4x8w16x16k32: ×1.019 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile64x128r8x8w16x8k32: ×1.043 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 *   - tile64x32r4x4w8x16k16: ×0.950 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile64x32r4x4w8x16k32: ×0.939 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile64x32r8x4w8x8k16: geometric mean ×1.065 (short of the adopted ×1.131)
 *   - tile64x32r8x4w8x8k32: ×1.043 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile64x64r4x4w16x16k16: ×0.961 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile64x64r4x4w16x16k32: ×0.966 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile64x64r4x8w8x16k16: geometric mean ×1.106 (short of the adopted ×1.131)
 *   - tile64x64r4x8w8x16k32: geometric mean ×1.097 (short of the adopted ×1.131)
 *   - tile64x64r8x4w16x8k16: geometric mean ×1.102 (short of the adopted ×1.131)
 *   - tile64x64r8x4w16x8k32: geometric mean ×1.094 (short of the adopted ×1.131)
 *   - tile64x64r8x8w8x8k16: ×1.003 < ×1.050 on i8a8-attention-pv-self-m1024-n1024
 *   - tile64x64r8x8w8x8k32: ×1.020 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 * - i8a8.conv3d (i8a8 conv3d (w8a8 implicit GEMM), 0 cases): keeps the default tile128x64r8x8w8x16k16 (the sweeps have no case of this class)
 *
 * Re-derived on 2026-10-26 from the same records with `deriveGeometryProfile` when the profile gained
 * `i8a8.conv3d` (ADR 0115 addendum 2026-10-26; the scripts are kept, untracked, in
 * `outputs/diag/wave-b-2026-10-26/v5/`). Every other field is unchanged; `i8a8.conv3d` keeps the default
 * because no record has a conv3d case, and `provenance.kernels` / `caseSet` are the current runtime's.
 */
import type { GeometryProfile } from "../geometry-profile.ts";

export const APPLE_METAL_3: GeometryProfile = {
  id: "apple-metal-3",
  match: { vendor: "apple", architecture: "metal-3", description: "Apple M2" },
  gemmRows: [
    { maxRows: 16, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 32, geometry: { regM: 2, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: 64, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: 128, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: 512, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: Number.POSITIVE_INFINITY, geometry: { regM: 8, regN: 4, wgX: 8, wgY: 16 } },
  ],
  attention: {
    qk: { regM: 8, regN: 4, wgX: 8, wgY: 16 },
    pv: { regM: 4, regN: 8, wgX: 8, wgY: 16 },
  },
  conv2d: {
    rows64: { regM: 8, regN: 4, wgX: 16, wgY: 16 },
    rows32: { regM: 8, regN: 4, wgX: 16, wgY: 16 },
  },
  i8a8: {
    linear: { regM: 4, regN: 8, wgX: 8, wgY: 16, tileK: 16 },
    attentionQk: { regM: 4, regN: 8, wgX: 8, wgY: 8, tileK: 16 },
    attentionPv: { regM: 4, regN: 8, wgX: 16, wgY: 4, tileK: 16 },
    conv3d: { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 },
  },
  provenance: {
    sweep:
      "outputs/bench-browser/geometry-sweep-browser-2026-09-27T16-28-00.787Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-29T13-08-11.329Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json, outputs/bench-browser/geometry-sweep-browser-2026-10-01T12-28-47.052Z.json",
    sha256:
      "921e89ba70b34ef68ec21c153c2a2c313c80bbbc98d53da720942c2919ead034, f464ca82eb61f4c8fd60efa06365815eb25cec5e1f8ed905cccd242cadacfc2a, 38d97dac5f04a7e80fb6591ab437da7dd2b3cee66b0d03e40cd1475c0f398550, d9e5e7814370c741562819474f5e0837012d05f233869688d28bb91902ac8621, 11fed6a888c91d5e5217a8cd1205dbd46b5ffa2c639dcf2f2af9d17be962eefd",
    date: "2026-09-27T16:28:00.787Z, 2026-09-27T18:31:46.471Z, 2026-09-29T13:08:11.329Z, 2026-09-29T21:04:14.586Z, 2026-10-01T12:28:47.052Z",
    candidateSet: "quick, full, full, full, full",
    userAgent: [
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36",
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
    ],
    adapter: {
      vendor: "apple",
      architecture: "metal-3",
      device: "0x0000",
      description: "Apple M2",
    },
    kernels: "da4542673a039990",
    caseSet: "0c9e4c7b7202eb2a",
  },
};
