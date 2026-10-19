/**
 * Geometry profile `nvidia-blackwell` (**generated — do not edit by hand**).
 *
 * A static table of tile geometries for adapter `nvidia / blackwell`, written from the
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
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-37-27.914Z.json \
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-09-29T20-42-08.545Z.json \
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-10-01T16-46-50.379Z.json \
 *     --id nvidia-blackwell --vendor nvidia --architecture blackwell \
 *     --out packages/runtime/src/kernels/geometry-profiles/nvidia-blackwell.ts --min-speedup 1.05
 *
 * Sweeps (adapter nvidia / blackwell / 0x2c05 / NVIDIA GeForce RTX 5070 Ti):
 *
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-37-27.914Z.json (sha256 084cd0efd7ea7780dfcdab5b53ee4be8177638c7d8984920096a4851b767aa90, 2026-09-27T18:37:27.914Z)
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-29T20-42-08.545Z.json (sha256 1ae3a8f6d9a1270d8e3b3ad41aa7fe13757c976a323de7d2b0b99139800cb2e5, 2026-09-29T20:42:08.545Z)
 * - outputs/bench-browser/geometry-sweep-browser-2026-10-01T16-46-50.379Z.json (sha256 0e57d4efcb14efe0e01767066fce05e1784ca43660a4f8a78c5a1a2c48d1ae87, 2026-10-01T16:46:50.379Z)
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
 *   - reg128x128r8x8w16: ×0.166 < ×1.050 on matmul-m16-n3072-k1024
 *   - reg128x16r8x4w4: ×0.168 < ×1.050 on matmul-m16-n3072-k1024
 *   - reg128x32r8x4w8: ×0.233 < ×1.050 on matmul-m16-n3072-k1024
 *   - reg128x32r8x8w4: ×0.206 < ×1.050 on matmul-m16-n3072-k1024
 *   - reg128x64r8x4w16: ×0.268 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg128x64r8x8w8: ×0.264 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg16x128r4x8w16: ×0.209 < ×1.050 on linear-m16-n3072-k1024
 *   - reg16x16r2x4w4: ×0.703 < ×1.050 on linear-m16-n3072-k1024
 *   - reg16x16r4x4w4: ×0.373 < ×1.050 on linear-m16-n3072-k1024
 *   - reg16x32r1x8w4: ×0.698 < ×1.050 on linear-m16-n3072-k1024
 *   - reg16x32r2x4w8: ×0.773 < ×1.050 on linear-m16-n3072-k1024
 *   - reg16x32r2x8w4: ×0.402 < ×1.050 on linear-m16-n3072-k1024
 *   - reg16x32r4x4w8: ×0.414 < ×1.050 on linear-m16-n3072-k1024
 *   - reg16x32r4x8w4: ×0.212 < ×1.050 on linear-m16-n3072-k1024
 *   - reg16x64r2x8w8: ×0.418 < ×1.050 on linear-m16-n3072-k1024
 *   - reg16x64r4x4w16: ×0.432 < ×1.050 on linear-m16-n3072-k1024
 *   - reg16x64r4x8w8: ×0.216 < ×1.050 on linear-m16-n3072-k1024
 *   - reg32x128r4x8w16: ×0.319 < ×1.050 on linear-m16-n3072-k1024
 *   - reg32x128r8x8w16: ×0.166 < ×1.050 on linear-m16-n3072-k1024
 *   - reg32x16r2x4w4: ×0.638 < ×1.050 on matmul-m16-n3072-k1024
 *   - reg32x16r4x4w4: ×0.465 < ×1.050 on linear-m16-n3072-k1024
 *   - reg32x16r8x4w4: ×0.280 < ×1.050 on linear-m16-n3072-k1024
 *   - reg32x32r2x4w8: ×0.644 < ×1.050 on matmul-m16-n3072-k1024
 *   - reg32x32r2x8w4: ×0.555 < ×1.050 on linear-m16-n3072-k1024
 *   - reg32x32r4x4w8: ×0.585 < ×1.050 on linear-m16-n3072-k1024
 *   - reg32x32r4x8w4: ×0.298 < ×1.050 on linear-m16-n3072-k1024
 *   - reg32x32r8x4w8: ×0.305 < ×1.050 on linear-m16-n3072-k1024
 *   - reg32x32r8x8w4: ×0.162 < ×1.050 on linear-m16-n3072-k1024
 *   - reg32x64r2x8w8: ×0.547 < ×1.050 on linear-m16-n3072-k1024
 *   - reg32x64r4x4w16: ×0.524 < ×1.050 on linear-m16-n3072-k1024
 *   - reg32x64r4x8w8: ×0.336 < ×1.050 on linear-m16-n3072-k1024
 *   - reg32x64r8x4w16: ×0.349 < ×1.050 on linear-m16-n3072-k1024
 *   - reg32x64r8x8w8: ×0.160 < ×1.050 on linear-m16-n3072-k1024
 *   - reg4x16r1x4w4: ×0.358 < ×1.050 on linear-m16-n3072-k1024
 *   - reg4x32r1x8w4: ×0.239 < ×1.050 on linear-m16-n3072-k1024
 *   - reg64x128r4x8w16: ×0.274 < ×1.050 on matmul-m16-n3072-k1024
 *   - reg64x128r8x8w16: ×0.237 < ×1.050 on linear-m16-n3072-k1024
 *   - reg64x16r4x4w4: ×0.347 < ×1.050 on matmul-m16-n3072-k1024
 *   - reg64x16r8x4w4: ×0.263 < ×1.050 on matmul-m16-n3072-k1024
 *   - reg64x32r4x4w8: ×0.430 < ×1.050 on matmul-m16-n3072-k1024
 *   - reg64x32r4x8w4: ×0.356 < ×1.050 on linear-m16-n3072-k1024
 *   - reg64x32r8x4w8: ×0.368 < ×1.050 on linear-m16-n3072-k1024
 *   - reg64x32r8x8w4: ×0.207 < ×1.050 on linear-m16-n3072-k1024
 *   - reg64x64r4x4w16: ×0.455 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg64x64r4x8w8: ×0.399 < ×1.050 on bmm-b16-m16-n64-k128
 *   - reg64x64r8x4w16: ×0.349 < ×1.050 on linear-m16-n3072-k1024
 *   - reg64x64r8x8w8: ×0.226 < ×1.050 on linear-m16-n3072-k1024
 *   - reg8x16r1x4w4: ×0.778 < ×1.050 on linear-m16-n3072-k1024
 *   - reg8x16r2x4w4: ×0.445 < ×1.050 on linear-m16-n3072-k1024
 *   - reg8x32r1x8w4: ×0.464 < ×1.050 on linear-m16-n3072-k1024
 *   - reg8x32r2x4w8: ×0.504 < ×1.050 on linear-m16-n3072-k1024
 *   - reg8x32r2x8w4: ×0.259 < ×1.050 on linear-m16-n3072-k1024
 *   - reg8x64r2x8w8: ×0.233 < ×1.050 on linear-m16-n3072-k1024
 * - gemmRows 17–32 (linear / matmul / bmm with row count 17–32, 4 cases): default reg16x16r1x4w4 kept (no geometry matches the output and reaches ×1.050 on every case)
 *   - reg128x128r8x8w16: ×0.194 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg128x16r8x4w4: ×0.230 < ×1.050 on matmul-m32-n3072-k1024
 *   - reg128x32r8x4w8: ×0.318 < ×1.050 on matmul-m32-n3072-k1024
 *   - reg128x32r8x8w4: ×0.272 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg128x64r8x4w16: ×0.299 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg128x64r8x8w8: ×0.294 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg16x128r4x8w16: ×0.330 < ×1.050 on linear-m32-n3072-k1024
 *   - reg16x16r2x4w4: ×0.767 < ×1.050 on matmul-m32-n3072-k1024
 *   - reg16x16r4x4w4: ×0.495 < ×1.050 on linear-m32-n3072-k1024
 *   - reg16x32r1x8w4: ×0.945 < ×1.050 on linear-m32-n3072-k1024
 *   - reg16x32r2x4w8: ×0.978 < ×1.050 on linear-m32-n3072-k1024
 *   - reg16x32r2x8w4: ×0.588 < ×1.050 on linear-m32-n3072-k1024
 *   - reg16x32r4x4w8: ×0.610 < ×1.050 on linear-m32-n3072-k1024
 *   - reg16x32r4x8w4: ×0.335 < ×1.050 on linear-m32-n3072-k1024
 *   - reg16x64r2x8w8: ×0.607 < ×1.050 on linear-m32-n3072-k1024
 *   - reg16x64r4x4w16: ×0.656 < ×1.050 on linear-m32-n3072-k1024
 *   - reg16x64r4x8w8: ×0.316 < ×1.050 on linear-m32-n3072-k1024
 *   - reg32x128r4x8w16: ×0.472 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x128r8x8w16: ×0.260 < ×1.050 on linear-m32-n3072-k1024
 *   - reg32x16r2x4w4: ×0.859 < ×1.050 on matmul-m32-n3072-k1024
 *   - reg32x16r4x4w4: ×0.658 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x16r8x4w4: ×0.399 < ×1.050 on linear-m32-n3072-k1024
 *   - reg32x32r2x4w8: ×0.750 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x32r2x8w4: ×0.724 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x32r4x4w8: ×0.740 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x32r4x8w4: ×0.456 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x32r8x4w8: ×0.464 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x32r8x8w4: ×0.242 < ×1.050 on linear-m32-n3072-k1024
 *   - reg32x64r2x8w8: ×0.628 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x64r4x4w16: ×0.671 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x64r4x8w8: ×0.484 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x64r8x4w16: ×0.503 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg32x64r8x8w8: ×0.253 < ×1.050 on linear-m32-n3072-k1024
 *   - reg4x16r1x4w4: ×0.280 < ×1.050 on linear-m32-n3072-k1024
 *   - reg4x32r1x8w4: ×0.295 < ×1.050 on linear-m32-n3072-k1024
 *   - reg64x128r4x8w16: ×0.316 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg64x128r8x8w16: ×0.307 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg64x16r4x4w4: ×0.465 < ×1.050 on matmul-m32-n3072-k1024
 *   - reg64x16r8x4w4: ×0.353 < ×1.050 on matmul-m32-n3072-k1024
 *   - reg64x32r4x4w8: ×0.577 < ×1.050 on matmul-m32-n3072-k1024
 *   - reg64x32r4x8w4: ×0.483 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg64x32r8x4w8: ×0.489 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg64x32r8x8w4: ×0.263 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg64x64r4x4w16: ×0.498 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg64x64r4x8w8: ×0.430 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg64x64r8x4w16: ×0.409 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg64x64r8x8w8: ×0.294 < ×1.050 on bmm-b16-m32-n64-k128
 *   - reg8x16r1x4w4: ×0.749 < ×1.050 on linear-m32-n3072-k1024
 *   - reg8x16r2x4w4: ×0.456 < ×1.050 on linear-m32-n3072-k1024
 *   - reg8x32r1x8w4: ×0.623 < ×1.050 on linear-m32-n3072-k1024
 *   - reg8x32r2x4w8: ×0.655 < ×1.050 on linear-m32-n3072-k1024
 *   - reg8x32r2x8w4: ×0.343 < ×1.050 on linear-m32-n3072-k1024
 *   - reg8x64r2x8w8: ×0.330 < ×1.050 on linear-m32-n3072-k1024
 * - gemmRows 33–64 (linear / matmul / bmm with row count 33–64, 4 cases): default reg16x16r1x4w4 kept (no geometry matches the output and reaches ×1.050 on every case)
 *   - reg128x128r8x8w16: ×0.224 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg128x16r8x4w4: ×0.420 < ×1.050 on matmul-m64-n3072-k1024
 *   - reg128x32r8x4w8: ×0.458 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg128x32r8x8w4: ×0.323 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg128x64r8x4w16: ×0.345 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg128x64r8x8w8: ×0.336 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg16x128r4x8w16: ×0.545 < ×1.050 on linear-m64-n3072-k1024
 *   - reg16x16r2x4w4: ×0.773 < ×1.050 on bmm-b16-m64-n128-k64
 *   - reg16x16r4x4w4: ×0.621 < ×1.050 on linear-m64-n3072-k1024
 *   - reg16x32r1x8w4: ×1.022 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg16x32r2x4w8: ×1.030 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg16x32r2x8w4: ×0.817 < ×1.050 on linear-m64-n3072-k1024
 *   - reg16x32r4x4w8: ×0.828 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg16x32r4x8w4: ×0.500 < ×1.050 on linear-m64-n3072-k1024
 *   - reg16x64r2x8w8: ×0.851 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg16x64r4x4w16: ×0.874 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg16x64r4x8w8: ×0.535 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x128r4x8w16: ×0.563 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x128r8x8w16: ×0.354 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x16r2x4w4: ×0.835 < ×1.050 on matmul-m64-n3072-k1024
 *   - reg32x16r4x4w4: ×0.719 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x16r8x4w4: ×0.499 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x32r2x4w8: ×0.886 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x32r2x8w4: ×0.858 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x32r4x4w8: ×0.875 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x32r4x8w4: ×0.545 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x32r8x4w8: ×0.554 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x32r8x8w4: ×0.308 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x64r2x8w8: ×0.749 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x64r4x4w16: ×0.800 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x64r4x8w8: ×0.578 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x64r8x4w16: ×0.601 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg32x64r8x8w8: ×0.326 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg4x16r1x4w4: ×0.278 < ×1.050 on linear-m64-n3072-k1024
 *   - reg4x32r1x8w4: ×0.276 < ×1.050 on linear-m64-n3072-k1024
 *   - reg64x128r4x8w16: ×0.360 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x128r8x8w16: ×0.349 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x16r4x4w4: ×0.735 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x16r8x4w4: ×0.430 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x32r4x4w8: ×0.733 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x32r4x8w4: ×0.542 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x32r8x4w8: ×0.557 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x32r8x8w4: ×0.311 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x64r4x4w16: ×0.556 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x64r4x8w8: ×0.481 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x64r8x4w16: ×0.465 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg64x64r8x8w8: ×0.343 < ×1.050 on bmm-b16-m64-n64-k128
 *   - reg8x16r1x4w4: ×0.689 < ×1.050 on linear-m64-n3072-k1024
 *   - reg8x16r2x4w4: ×0.423 < ×1.050 on linear-m64-n3072-k1024
 *   - reg8x32r1x8w4: ×0.801 < ×1.050 on linear-m64-n3072-k1024
 *   - reg8x32r2x4w8: ×0.745 < ×1.050 on linear-m64-n3072-k1024
 *   - reg8x32r2x8w4: ×0.481 < ×1.050 on linear-m64-n3072-k1024
 *   - reg8x64r2x8w8: ×0.529 < ×1.050 on linear-m64-n3072-k1024
 * - gemmRows 65–128 (linear / matmul / bmm with row count 65–128, 4 cases): default reg64x32r4x4w8 kept (no geometry matches the output and reaches ×1.050 on every case)
 *   - reg128x128r8x8w16: ×0.289 < ×1.050 on bmm-b16-m128-n64-k128
 *   - reg128x16r8x4w4: ×0.551 < ×1.050 on linear-m128-n2048-k1024
 *   - reg128x32r8x4w8: ×0.582 < ×1.050 on bmm-b16-m128-n64-k128
 *   - reg128x32r8x8w4: ×0.430 < ×1.050 on bmm-b16-m128-n64-k128
 *   - reg128x64r8x4w16: ×0.446 < ×1.050 on bmm-b16-m128-n64-k128
 *   - reg128x64r8x8w8: ×0.424 < ×1.050 on bmm-b16-m128-n64-k128
 *   - reg16x128r4x8w16: ×0.453 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x16r1x4w4: ×0.641 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x16r2x4w4: ×0.530 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x16r4x4w4: ×0.381 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x32r1x8w4: ×0.730 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x32r2x4w8: ×0.675 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x32r2x8w4: ×0.634 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x32r4x4w8: ×0.645 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x32r4x8w4: ×0.387 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x64r2x8w8: ×0.709 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x64r4x4w16: ×0.750 < ×1.050 on linear-m128-n2048-k1024
 *   - reg16x64r4x8w8: ×0.441 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x128r4x8w16: ×0.747 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x128r8x8w16: ×0.397 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x16r2x4w4: ×0.623 < ×1.050 on matmul-m128-n2048-k1024
 *   - reg32x16r4x4w4: ×0.580 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x16r8x4w4: ×0.437 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x32r2x4w8: ×0.888 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x32r2x8w4: ×0.864 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x32r4x4w8: ×0.883 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x32r4x8w4: ×0.581 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x32r8x4w8: ×0.602 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x32r8x8w4: ×0.361 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x64r2x8w8: ×0.870 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x64r4x4w16: ×0.898 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x64r4x8w8: ×0.741 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x64r8x4w16: ×0.790 < ×1.050 on linear-m128-n2048-k1024
 *   - reg32x64r8x8w8: ×0.357 < ×1.050 on linear-m128-n2048-k1024
 *   - reg4x16r1x4w4: ×0.174 < ×1.050 on linear-m128-n2048-k1024
 *   - reg4x32r1x8w4: ×0.171 < ×1.050 on linear-m128-n2048-k1024
 *   - reg64x128r4x8w16: ×0.494 < ×1.050 on bmm-b16-m128-n64-k128
 *   - reg64x128r8x8w16: ×0.480 < ×1.050 on bmm-b16-m128-n64-k128
 *   - reg64x16r4x4w4: ×0.627 < ×1.050 on matmul-m128-n2048-k1024
 *   - reg64x16r8x4w4: ×0.492 < ×1.050 on linear-m128-n2048-k1024
 *   - reg64x32r4x8w4: ×0.733 < ×1.050 on bmm-b16-m128-n64-k64
 *   - reg64x32r8x4w8: ×0.761 < ×1.050 on bmm-b16-m128-n64-k128
 *   - reg64x32r8x8w4: ×0.427 < ×1.050 on linear-m128-n2048-k1024
 *   - reg64x64r4x4w16: ×0.763 < ×1.050 on bmm-b16-m128-n64-k128
 *   - reg64x64r4x8w8: ×0.659 < ×1.050 on bmm-b16-m128-n64-k128
 *   - reg64x64r8x4w16: ×0.642 < ×1.050 on bmm-b16-m128-n64-k128
 *   - reg64x64r8x8w8: ×0.474 < ×1.050 on bmm-b16-m128-n64-k128
 *   - reg8x16r1x4w4: ×0.427 < ×1.050 on linear-m128-n2048-k1024
 *   - reg8x16r2x4w4: ×0.258 < ×1.050 on linear-m128-n2048-k1024
 *   - reg8x32r1x8w4: ×0.499 < ×1.050 on linear-m128-n2048-k1024
 *   - reg8x32r2x4w8: ×0.471 < ×1.050 on linear-m128-n2048-k1024
 *   - reg8x32r2x8w4: ×0.307 < ×1.050 on linear-m128-n2048-k1024
 *   - reg8x64r2x8w8: ×0.426 < ×1.050 on linear-m128-n2048-k1024
 * - gemmRows 129–256 (linear / matmul / bmm with row count 129–256, 4 cases): adopted reg64x64r4x4w16 ×1.203 (×1.149–×1.291)
 *   - reg128x128r8x8w16: ×0.450 < ×1.050 on bmm-b16-m256-n64-k128
 *   - reg128x16r8x4w4: ×0.566 < ×1.050 on matmul-m256-n2048-k1024
 *   - reg128x32r8x4w8: ×0.898 < ×1.050 on bmm-b16-m256-n64-k128
 *   - reg128x32r8x8w4: ×0.655 < ×1.050 on bmm-b16-m256-n64-k64
 *   - reg128x64r8x4w16: ×0.694 < ×1.050 on bmm-b16-m256-n64-k128
 *   - reg128x64r8x8w8: ×0.645 < ×1.050 on bmm-b16-m256-n64-k64
 *   - reg16x128r4x8w16: ×0.693 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x16r1x4w4: ×0.628 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x16r2x4w4: ×0.476 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x16r4x4w4: ×0.353 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x32r1x8w4: ×0.719 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x32r2x4w8: ×0.668 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x32r2x8w4: ×0.664 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x32r4x4w8: ×0.658 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x32r4x8w4: ×0.462 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x64r2x8w8: ×0.709 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x64r4x4w16: ×0.791 < ×1.050 on linear-m256-n2048-k1024
 *   - reg16x64r4x8w8: ×0.678 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x128r4x8w16: ×0.752 < ×1.050 on bmm-b16-m256-n64-k64
 *   - reg32x128r8x8w16: ×0.684 < ×1.050 on bmm-b16-m256-n64-k128
 *   - reg32x16r2x4w4: ×0.619 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x16r4x4w4: ×0.584 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x16r8x4w4: ×0.464 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x32r2x4w8: ×0.837 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x32r2x8w4: ×0.867 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x32r4x4w8: ×0.855 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x32r4x8w4: ×0.806 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x32r8x4w8: ×0.813 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x32r8x8w4: ×0.520 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x64r2x8w8: ×0.960 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x64r4x4w16: ×1.035 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x64r4x8w8: ×0.969 < ×1.050 on bmm-b16-m256-n64-k64
 *   - reg32x64r8x4w16: ×0.996 < ×1.050 on linear-m256-n2048-k1024
 *   - reg32x64r8x8w8: ×0.657 < ×1.050 on linear-m256-n2048-k1024
 *   - reg4x16r1x4w4: ×0.166 < ×1.050 on linear-m256-n2048-k1024
 *   - reg4x32r1x8w4: ×0.175 < ×1.050 on linear-m256-n2048-k1024
 *   - reg64x128r4x8w16: ×0.726 < ×1.050 on bmm-b16-m256-n64-k64
 *   - reg64x128r8x8w16: ×0.713 < ×1.050 on bmm-b16-m256-n64-k64
 *   - reg64x16r4x4w4: ×0.586 < ×1.050 on matmul-m256-n2048-k1024
 *   - reg64x16r8x4w4: ×0.549 < ×1.050 on matmul-m256-n2048-k1024
 *   - reg64x32r4x8w4: ×0.879 < ×1.050 on bmm-b16-m256-n64-k64
 *   - reg64x32r8x4w8: ×0.939 < ×1.050 on bmm-b16-m256-n64-k64
 *   - reg64x32r8x8w4: ×0.639 < ×1.050 on bmm-b16-m256-n64-k64
 *   - reg64x64r4x8w8: ×0.945 < ×1.050 on bmm-b16-m256-n64-k64
 *   - reg64x64r8x4w16: ×0.981 < ×1.050 on bmm-b16-m256-n64-k64
 *   - reg64x64r8x8w8: ×0.705 < ×1.050 on bmm-b16-m256-n64-k64
 *   - reg8x16r1x4w4: ×0.428 < ×1.050 on linear-m256-n2048-k1024
 *   - reg8x16r2x4w4: ×0.259 < ×1.050 on linear-m256-n2048-k1024
 *   - reg8x32r1x8w4: ×0.464 < ×1.050 on linear-m256-n2048-k1024
 *   - reg8x32r2x4w8: ×0.430 < ×1.050 on linear-m256-n2048-k1024
 *   - reg8x32r2x8w4: ×0.279 < ×1.050 on linear-m256-n2048-k1024
 *   - reg8x64r2x8w8: ×0.462 < ×1.050 on linear-m256-n2048-k1024
 * - gemmRows 257–512 (linear / matmul / bmm with row count 257–512, 5 cases): adopted reg64x64r8x4w16 ×1.330 (×1.184–×1.451)
 *   - reg128x128r8x8w16: ×0.709 < ×1.050 on bmm-b16-m512-n64-k64
 *   - reg128x16r8x4w4: ×0.552 < ×1.050 on matmul-m512-n2048-k1024
 *   - reg128x32r8x4w8: ×0.966 < ×1.050 on bmm-b16-m512-n64-k64
 *   - reg128x32r8x8w4: ×0.832 < ×1.050 on bmm-b16-m512-n64-k64
 *   - reg128x64r8x4w16: geometric mean ×1.273 (short of the adopted ×1.330)
 *   - reg128x64r8x8w8: ×0.993 < ×1.050 on bmm-b16-m512-n64-k64
 *   - reg16x128r4x8w16: ×0.726 < ×1.050 on bmm-b16-m512-n64-k512
 *   - reg16x16r1x4w4: ×0.623 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x16r2x4w4: ×0.511 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x16r4x4w4: ×0.376 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x32r1x8w4: ×0.751 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x32r2x4w8: ×0.691 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x32r2x8w4: ×0.650 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x32r4x4w8: ×0.634 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x32r4x8w4: ×0.442 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x64r2x8w8: ×0.718 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x64r4x4w16: ×0.794 < ×1.050 on linear-m512-n2048-k1024
 *   - reg16x64r4x8w8: ×0.767 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x128r4x8w16: ×0.785 < ×1.050 on bmm-b16-m512-n64-k64
 *   - reg32x128r8x8w16: ×0.710 < ×1.050 on bmm-b16-m512-n64-k64
 *   - reg32x16r2x4w4: ×0.594 < ×1.050 on bmm-b16-m512-n512-k64
 *   - reg32x16r4x4w4: ×0.540 < ×1.050 on matmul-m512-n2048-k1024
 *   - reg32x16r8x4w4: ×0.441 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x32r2x4w8: ×0.908 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x32r2x8w4: ×0.875 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x32r4x4w8: ×0.852 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x32r4x8w4: ×0.859 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x32r8x4w8: ×0.849 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x32r8x8w4: ×0.489 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x64r2x8w8: ×0.970 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x64r4x4w16: geometric mean ×1.209 (short of the adopted ×1.330)
 *   - reg32x64r4x8w8: ×1.048 < ×1.050 on linear-m512-n2048-k1024
 *   - reg32x64r8x4w16: geometric mean ×1.236 (short of the adopted ×1.330)
 *   - reg32x64r8x8w8: ×0.930 < ×1.050 on bmm-b16-m512-n64-k64
 *   - reg4x16r1x4w4: ×0.167 < ×1.050 on linear-m512-n2048-k1024
 *   - reg4x32r1x8w4: ×0.174 < ×1.050 on linear-m512-n2048-k1024
 *   - reg64x128r4x8w16: ×0.759 < ×1.050 on bmm-b16-m512-n64-k64
 *   - reg64x128r8x8w16: ×0.732 < ×1.050 on bmm-b16-m512-n64-k64
 *   - reg64x16r4x4w4: ×0.557 < ×1.050 on matmul-m512-n2048-k1024
 *   - reg64x16r8x4w4: ×0.557 < ×1.050 on bmm-b16-m512-n64-k512
 *   - reg64x32r4x8w4: ×0.953 < ×1.050 on bmm-b16-m512-n64-k64
 *   - reg64x32r8x4w8: ×0.977 < ×1.050 on linear-m512-n2048-k1024
 *   - reg64x32r8x8w4: ×0.859 < ×1.050 on bmm-b16-m512-n64-k64
 *   - reg64x64r4x4w16: geometric mean ×1.278 (short of the adopted ×1.330)
 *   - reg64x64r4x8w8: ×1.017 < ×1.050 on bmm-b16-m512-n64-k64
 *   - reg64x64r8x8w8: ×0.986 < ×1.050 on bmm-b16-m512-n64-k64
 *   - reg8x16r1x4w4: ×0.428 < ×1.050 on linear-m512-n2048-k1024
 *   - reg8x16r2x4w4: ×0.261 < ×1.050 on linear-m512-n2048-k1024
 *   - reg8x32r1x8w4: ×0.490 < ×1.050 on linear-m512-n2048-k1024
 *   - reg8x32r2x4w8: ×0.461 < ×1.050 on linear-m512-n2048-k1024
 *   - reg8x32r2x8w4: ×0.301 < ×1.050 on linear-m512-n2048-k1024
 *   - reg8x64r2x8w8: ×0.454 < ×1.050 on linear-m512-n2048-k1024
 * - gemmRows > 512 (linear / matmul / bmm with row count > 512, 7 cases): default reg128x128r8x8w16 kept (no geometry matches the output and reaches ×1.050 on every case)
 *   - reg128x16r8x4w4: ×0.353 < ×1.050 on linear-m1024-n2048-k2048
 *   - reg128x32r8x4w8: ×0.625 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg128x32r8x8w4: ×0.572 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg128x64r8x4w16: ×0.836 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg128x64r8x8w8: ×0.905 < ×1.050 on linear-m1024-n2048-k2048
 *   - reg16x128r4x8w16: ×0.485 < ×1.050 on linear-m1024-n2048-k2048
 *   - reg16x16r1x4w4: ×0.340 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg16x16r2x4w4: ×0.277 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg16x16r4x4w4: ×0.206 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg16x32r1x8w4: ×0.412 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg16x32r2x4w8: ×0.376 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg16x32r2x8w4: ×0.381 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg16x32r4x4w8: ×0.371 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg16x32r4x8w4: ×0.257 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg16x64r2x8w8: ×0.418 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg16x64r4x4w16: ×0.466 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg16x64r4x8w8: ×0.430 < ×1.050 on linear-m1024-n2048-k2048
 *   - reg32x128r4x8w16: ×0.706 < ×1.050 on linear-m1024-n2048-k2048
 *   - reg32x128r8x8w16: ×0.575 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg32x16r2x4w4: ×0.342 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg32x16r4x4w4: ×0.318 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg32x16r8x4w4: ×0.256 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg32x32r2x4w8: ×0.501 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg32x32r2x8w4: ×0.513 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg32x32r4x4w8: ×0.493 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg32x32r4x8w4: ×0.478 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg32x32r8x4w8: ×0.473 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg32x32r8x8w4: ×0.330 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg32x64r2x8w8: ×0.578 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg32x64r4x4w16: ×0.637 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg32x64r4x8w8: ×0.615 < ×1.050 on linear-m1024-n2048-k2048
 *   - reg32x64r8x4w16: ×0.651 < ×1.050 on linear-m1024-n2048-k2048
 *   - reg32x64r8x8w8: ×0.512 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg4x16r1x4w4: ×0.089 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg4x32r1x8w4: ×0.095 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg64x128r4x8w16: ×0.832 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg64x128r8x8w16: ×0.930 < ×1.050 on linear-m1024-n2048-k2048
 *   - reg64x16r4x4w4: ×0.357 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg64x16r8x4w4: ×0.332 < ×1.050 on linear-m1024-n2048-k2048
 *   - reg64x32r4x4w8: ×0.589 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg64x32r4x8w4: ×0.571 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg64x32r8x4w8: ×0.566 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg64x32r8x8w4: ×0.502 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg64x64r4x4w16: ×0.744 < ×1.050 on linear-m1024-n2048-k2048
 *   - reg64x64r4x8w8: ×0.762 < ×1.050 on linear-m1024-n2048-k2048
 *   - reg64x64r8x4w16: ×0.790 < ×1.050 on linear-m1024-n2048-k2048
 *   - reg64x64r8x8w8: ×0.687 < ×1.050 on linear-m1024-n2048-k8192
 *   - reg8x16r1x4w4: ×0.233 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg8x16r2x4w4: ×0.142 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg8x32r1x8w4: ×0.269 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg8x32r2x4w8: ×0.252 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg8x32r2x8w4: ×0.166 < ×1.050 on linear-m4096-n8192-k2048
 *   - reg8x64r2x8w8: ×0.268 < ×1.050 on linear-m4096-n8192-k2048
 * - attention.qk (fused attention f32 ①QK, 4 cases): default reg128x128r8x8w16 kept (no geometry matches the output and reaches ×1.050 on every case)
 *   - reg128x16r8x4w4: ×0.368 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg128x32r8x4w8: ×0.647 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg128x32r8x8w4: ×0.627 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg128x64r8x4w16: ×0.857 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg128x64r8x8w8: ×0.910 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg16x128r4x8w16: ×0.463 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x16r1x4w4: ×0.322 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x16r2x4w4: ×0.269 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x16r4x4w4: ×0.203 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x32r1x8w4: ×0.386 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x32r2x4w8: ×0.355 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x32r2x8w4: ×0.360 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x32r4x4w8: ×0.353 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x32r4x8w4: ×0.254 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x64r2x8w8: ×0.393 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x64r4x4w16: ×0.432 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg16x64r4x8w8: ×0.412 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x128r4x8w16: ×0.689 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x128r8x8w16: ×0.686 < ×1.050 on attention-qk-self-m1024-n1024
 *   - reg32x16r2x4w4: ×0.341 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x16r4x4w4: ×0.318 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x16r8x4w4: ×0.261 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x32r2x4w8: ×0.487 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x32r2x8w4: ×0.500 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x32r4x4w8: ×0.482 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x32r4x8w4: ×0.475 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x32r8x4w8: ×0.471 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x32r8x8w4: ×0.349 < ×1.050 on attention-qk-self-m1024-n1024
 *   - reg32x64r2x8w8: ×0.556 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x64r4x4w16: ×0.619 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x64r4x8w8: ×0.604 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x64r8x4w16: ×0.643 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg32x64r8x8w8: ×0.614 < ×1.050 on attention-qk-self-m1024-n1024
 *   - reg4x16r1x4w4: ×0.085 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg4x32r1x8w4: ×0.090 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg64x128r4x8w16: ×0.851 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg64x128r8x8w16: ×0.923 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg64x16r4x4w4: ×0.364 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg64x16r8x4w4: ×0.344 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg64x32r4x4w8: ×0.590 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg64x32r4x8w4: ×0.584 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg64x32r8x4w8: ×0.576 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg64x32r8x8w4: ×0.561 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg64x64r4x4w16: ×0.733 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg64x64r4x8w8: ×0.765 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg64x64r8x4w16: ×0.796 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg64x64r8x8w8: ×0.775 < ×1.050 on attention-qk-cross-m4096-n512
 *   - reg8x16r1x4w4: ×0.217 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg8x16r2x4w4: ×0.138 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg8x32r1x8w4: ×0.247 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg8x32r2x4w8: ×0.232 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg8x32r2x8w4: ×0.160 < ×1.050 on attention-qk-self-m4096-n4096
 *   - reg8x64r2x8w8: ×0.246 < ×1.050 on attention-qk-self-m4096-n4096
 * - attention.pv (fused attention f32 ③PV, 4 cases): default reg128x128r8x8w16 kept (no geometry matches the output and reaches ×1.050 on every case)
 *   - reg128x16r8x4w4: ×0.327 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg128x32r8x4w8: ×0.590 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg128x32r8x8w4: ×0.452 < ×1.050 on attention-pv-self-m1024-n1024
 *   - reg128x64r8x4w16: ×0.800 < ×1.050 on attention-pv-self-m1024-n1024
 *   - reg128x64r8x8w8: ×0.881 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg16x128r4x8w16: ×0.746 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg16x16r1x4w4: ×0.436 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg16x16r2x4w4: ×0.323 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg16x16r4x4w4: ×0.299 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg16x32r1x8w4: ×0.612 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg16x32r2x4w8: ×0.521 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg16x32r2x8w4: ×0.568 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg16x32r4x4w8: ×0.529 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg16x32r4x8w4: ×0.437 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg16x64r2x8w8: ×0.629 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg16x64r4x4w16: ×0.712 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg16x64r4x8w8: ×0.649 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg32x128r4x8w16: ×0.875 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg32x128r8x8w16: ×0.677 < ×1.050 on attention-pv-self-m1024-n1024
 *   - reg32x16r2x4w4: ×0.370 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg32x16r4x4w4: ×0.343 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg32x16r8x4w4: ×0.299 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg32x32r2x4w8: ×0.615 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg32x32r2x8w4: ×0.641 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg32x32r4x4w8: ×0.601 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg32x32r4x8w4: ×0.560 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg32x32r8x4w8: ×0.539 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg32x32r8x8w4: ×0.358 < ×1.050 on attention-pv-self-m1024-n1024
 *   - reg32x64r2x8w8: ×0.717 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg32x64r4x4w16: ×0.819 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg32x64r4x8w8: ×0.753 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg32x64r8x4w16: ×0.788 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg32x64r8x8w8: ×0.614 < ×1.050 on attention-pv-self-m1024-n1024
 *   - reg4x16r1x4w4: ×0.200 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg4x32r1x8w4: ×0.283 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg64x128r4x8w16: ×0.910 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg64x128r8x8w16: ×1.014 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg64x16r4x4w4: ×0.364 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg64x16r8x4w4: ×0.319 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg64x32r4x4w8: ×0.636 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg64x32r4x8w4: ×0.597 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg64x32r8x4w8: ×0.577 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg64x32r8x8w4: ×0.454 < ×1.050 on attention-pv-self-m1024-n1024
 *   - reg64x64r4x4w16: ×0.762 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg64x64r4x8w8: ×0.815 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg64x64r8x4w16: ×0.824 < ×1.050 on attention-pv-self-m1024-n1024
 *   - reg64x64r8x8w8: ×0.601 < ×1.050 on attention-pv-self-m1024-n1024
 *   - reg8x16r1x4w4: ×0.331 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg8x16r2x4w4: ×0.263 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg8x32r1x8w4: ×0.508 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg8x32r2x4w8: ×0.431 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg8x32r2x8w4: ×0.386 < ×1.050 on attention-pv-cross-m1024-n512
 *   - reg8x64r2x8w8: ×0.495 < ×1.050 on attention-pv-cross-m1024-n512
 * - conv2d.rows64 (conv2d implicit GEMM, m tile of 64 rows, 2 cases): default igemm64x128:wg16x8 kept (no geometry matches the output and reaches ×1.050 on every case)
 *   - igemm128x128:wg16x16: ×0.848 < ×1.050 on conv2d-c192-256x256
 *   - igemm128x64:wg16x16: ×0.783 < ×1.050 on conv2d-c192-256x256
 *   - igemm128x64:wg8x16: ×0.805 < ×1.050 on conv2d-c192-256x256
 *   - igemm16x128:wg16x4: ×0.602 < ×1.050 on conv2d-c384-128x128
 *   - igemm16x64:wg16x4: ×0.601 < ×1.050 on conv2d-c384-128x128
 *   - igemm16x64:wg8x4: ×0.581 < ×1.050 on conv2d-c384-128x128
 *   - igemm16x64:wg8x8: ×0.587 < ×1.050 on conv2d-c384-128x128
 *   - igemm32x128:wg16x4: ×0.800 < ×1.050 on conv2d-c384-128x128
 *   - igemm32x128:wg16x8: ×0.835 < ×1.050 on conv2d-c384-128x128
 *   - igemm32x64:wg16x4: ×0.747 < ×1.050 on conv2d-c192-256x256
 *   - igemm32x64:wg16x8: ×0.793 < ×1.050 on conv2d-c384-128x128
 *   - igemm32x64:wg8x16: ×0.773 < ×1.050 on conv2d-c384-128x128
 *   - igemm32x64:wg8x4: ×0.739 < ×1.050 on conv2d-c192-256x256
 *   - igemm32x64:wg8x8: ×0.806 < ×1.050 on conv2d-c192-256x256
 *   - igemm64x128:wg16x16: ×1.000 < ×1.050 on conv2d-c192-256x256
 *   - igemm64x64:wg16x16: ×0.949 < ×1.050 on conv2d-c192-256x256
 *   - igemm64x64:wg16x8: ×0.934 < ×1.050 on conv2d-c384-128x128
 *   - igemm64x64:wg8x16: ×0.982 < ×1.050 on conv2d-c192-256x256
 *   - igemm64x64:wg8x8: ×0.927 < ×1.050 on conv2d-c192-256x256
 *   - igemm8x64:wg8x4: ×0.363 < ×1.050 on conv2d-c384-128x128
 * - conv2d.rows32 (conv2d implicit GEMM, m tile of 32 rows, 1 case): default igemm32x128:wg16x4 kept (no geometry matches the output and reaches ×1.050 on every case)
 *   - igemm128x128:wg16x16: ×1.032 < ×1.050 on conv2d-c96-512x512
 *   - igemm128x64:wg16x16: ×0.972 < ×1.050 on conv2d-c96-512x512
 *   - igemm128x64:wg8x16: ×0.995 < ×1.050 on conv2d-c96-512x512
 *   - igemm16x128:wg16x4: ×0.774 < ×1.050 on conv2d-c96-512x512
 *   - igemm16x64:wg16x4: ×0.758 < ×1.050 on conv2d-c96-512x512
 *   - igemm16x64:wg8x4: ×0.730 < ×1.050 on conv2d-c96-512x512
 *   - igemm16x64:wg8x8: ×0.744 < ×1.050 on conv2d-c96-512x512
 *   - igemm32x128:wg16x8: ×1.039 < ×1.050 on conv2d-c96-512x512
 *   - igemm32x64:wg16x4: ×0.922 < ×1.050 on conv2d-c96-512x512
 *   - igemm32x64:wg16x8: ×1.003 < ×1.050 on conv2d-c96-512x512
 *   - igemm32x64:wg8x16: ×0.957 < ×1.050 on conv2d-c96-512x512
 *   - igemm32x64:wg8x4: ×0.936 < ×1.050 on conv2d-c96-512x512
 *   - igemm32x64:wg8x8: ×0.993 < ×1.050 on conv2d-c96-512x512
 *   - igemm64x128:wg16x16: ×0.940 < ×1.050 on conv2d-c96-512x512
 *   - igemm64x128:wg16x8: ×0.930 < ×1.050 on conv2d-c96-512x512
 *   - igemm64x64:wg16x16: ×0.904 < ×1.050 on conv2d-c96-512x512
 *   - igemm64x64:wg16x8: ×0.898 < ×1.050 on conv2d-c96-512x512
 *   - igemm64x64:wg8x16: ×0.939 < ×1.050 on conv2d-c96-512x512
 *   - igemm64x64:wg8x8: ×0.891 < ×1.050 on conv2d-c96-512x512
 *   - igemm8x64:wg8x4: ×0.478 < ×1.050 on conv2d-c96-512x512
 * - i8a8.linear (i8a8 linear, 6 cases): adopted tile128x64r8x4w16x16k16 ×1.194 (×1.164–×1.245)
 *   - tile128x128r8x8w16x16k16: ×0.825 < ×1.050 on i8a8-linear-m4096-n2048-k2048
 *   - tile128x128r8x8w16x16k32: ×0.870 < ×1.050 on i8a8-linear-m4096-n2048-k2048
 *   - tile128x32r8x4w8x16k16: ×0.897 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile128x32r8x4w8x16k32: ×1.036 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile128x64r8x4w16x16k32: geometric mean ×1.150 (short of the adopted ×1.194)
 *   - tile128x64r8x8w8x16k32: geometric mean ×1.161 (short of the adopted ×1.194)
 *   - tile16x128r4x8w16x4k16: ×0.568 < ×1.050 on i8a8-linear-m1024-n2048-k2048
 *   - tile16x128r4x8w16x4k32: ×0.624 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile16x32r4x4w8x4k16: ×0.545 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile16x32r4x4w8x4k32: ×0.672 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile16x64r4x4w16x4k16: ×0.544 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile16x64r4x4w16x4k32: ×0.677 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile16x64r4x8w8x4k16: ×0.505 < ×1.050 on i8a8-linear-m1024-n2048-k2048
 *   - tile16x64r4x8w8x4k32: ×0.740 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile32x128r4x8w16x8k16: ×0.751 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile32x128r4x8w16x8k32: ×0.763 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile32x128r8x8w16x4k16: ×0.842 < ×1.050 on i8a8-linear-m4096-n2048-k2048
 *   - tile32x128r8x8w16x4k32: ×0.826 < ×1.050 on i8a8-linear-m4096-n2048-k2048
 *   - tile32x32r4x4w8x8k16: ×0.714 < ×1.050 on i8a8-linear-m1024-n2048-k8192
 *   - tile32x32r4x4w8x8k32: ×0.767 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile32x32r8x4w8x4k16: ×0.607 < ×1.050 on i8a8-linear-m1024-n2048-k2048
 *   - tile32x32r8x4w8x4k32: ×0.883 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile32x64r4x4w16x8k16: ×0.813 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile32x64r4x4w16x8k32: ×0.802 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile32x64r4x8w8x8k16: ×0.796 < ×1.050 on i8a8-linear-m1024-n2048-k8192
 *   - tile32x64r4x8w8x8k32: ×0.926 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile32x64r8x4w16x4k16: ×0.875 < ×1.050 on i8a8-linear-m4096-n2048-k2048
 *   - tile32x64r8x4w16x4k32: ×0.959 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile32x64r8x8w8x4k16: ×0.889 < ×1.050 on i8a8-linear-m4096-n2048-k2048
 *   - tile32x64r8x8w8x4k32: ×0.856 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile64x128r4x8w16x16k16: ×0.838 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile64x128r4x8w16x16k32: ×0.811 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile64x128r8x8w16x8k16: ×0.908 < ×1.050 on i8a8-linear-m1024-n2048-k2048
 *   - tile64x128r8x8w16x8k32: ×0.895 < ×1.050 on i8a8-linear-m4096-n2048-k2048
 *   - tile64x32r4x4w8x16k16: ×0.855 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile64x32r4x4w8x16k32: ×0.816 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile64x32r8x4w8x8k16: ×0.855 < ×1.050 on i8a8-linear-m1024-n2048-k2048
 *   - tile64x32r8x4w8x8k32: ×0.998 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile64x64r4x4w16x16k16: ×0.882 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile64x64r4x4w16x16k32: ×0.895 < ×1.050 on i8a8-linear-m4096-n8192-k2048
 *   - tile64x64r4x8w8x16k16: geometric mean ×1.082 (short of the adopted ×1.194)
 *   - tile64x64r4x8w8x16k32: ×1.037 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile64x64r8x4w16x8k16: geometric mean ×1.124 (short of the adopted ×1.194)
 *   - tile64x64r8x4w16x8k32: geometric mean ×1.111 (short of the adopted ×1.194)
 *   - tile64x64r8x8w8x8k16: ×0.955 < ×1.050 on i8a8-linear-m4096-n2048-k8192
 *   - tile64x64r8x8w8x8k32: geometric mean ×1.104 (short of the adopted ×1.194)
 * - i8a8.attentionQk (i8a8 attention ①QK, 4 cases): adopted tile128x64r8x4w16x16k16 ×1.218 (×1.200–×1.238)
 *   - tile128x128r8x8w16x16k16: ×0.951 < ×1.050 on i8a8-attention-qk-cross-m1024-n512
 *   - tile128x128r8x8w16x16k32: ×0.894 < ×1.050 on i8a8-attention-qk-cross-m1024-n512
 *   - tile128x32r8x4w8x16k16: ×0.976 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile128x32r8x4w8x16k32: ×1.038 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile128x64r8x4w16x16k32: geometric mean ×1.196 (short of the adopted ×1.218)
 *   - tile128x64r8x8w8x16k32: ×1.015 < ×1.050 on i8a8-attention-qk-cross-m1024-n512
 *   - tile16x128r4x8w16x4k16: output differs from the default on i8a8-attention-qk-cross-m1024-n512
 *   - tile16x128r4x8w16x4k32: output differs from the default on i8a8-attention-qk-cross-m1024-n512
 *   - tile16x32r4x4w8x4k16: ×0.715 < ×1.050 on i8a8-attention-qk-self-m4096-n4096
 *   - tile16x32r4x4w8x4k32: output differs from the default on i8a8-attention-qk-cross-m1024-n512
 *   - tile16x64r4x4w16x4k16: ×0.719 < ×1.050 on i8a8-attention-qk-self-m4096-n4096
 *   - tile16x64r4x4w16x4k32: output differs from the default on i8a8-attention-qk-cross-m1024-n512
 *   - tile16x64r4x8w8x4k16: output differs from the default on i8a8-attention-qk-cross-m1024-n512
 *   - tile16x64r4x8w8x4k32: output differs from the default on i8a8-attention-qk-cross-m1024-n512
 *   - tile32x128r4x8w16x8k16: ×0.813 < ×1.050 on i8a8-attention-qk-self-m4096-n4096
 *   - tile32x128r4x8w16x8k32: output differs from the default on i8a8-attention-qk-cross-m1024-n512
 *   - tile32x128r8x8w16x4k16: output differs from the default on i8a8-attention-qk-cross-m1024-n512
 *   - tile32x128r8x8w16x4k32: output differs from the default on i8a8-attention-qk-cross-m1024-n512
 *   - tile32x32r4x4w8x8k16: ×0.825 < ×1.050 on i8a8-attention-qk-self-m4096-n4096
 *   - tile32x32r4x4w8x8k32: ×0.802 < ×1.050 on i8a8-attention-qk-self-m4096-n4096
 *   - tile32x32r8x4w8x4k16: ×0.903 < ×1.050 on i8a8-attention-qk-cross-m1024-n512
 *   - tile32x32r8x4w8x4k32: output differs from the default on i8a8-attention-qk-cross-m1024-n512
 *   - tile32x64r4x4w16x8k16: ×0.885 < ×1.050 on i8a8-attention-qk-self-m4096-n4096
 *   - tile32x64r4x4w16x8k32: ×0.875 < ×1.050 on i8a8-attention-qk-self-m4096-n4096
 *   - tile32x64r4x8w8x8k16: ×0.939 < ×1.050 on i8a8-attention-qk-self-m4096-n4096
 *   - tile32x64r4x8w8x8k32: output differs from the default on i8a8-attention-qk-cross-m1024-n512
 *   - tile32x64r8x4w16x4k16: ×1.000 < ×1.050 on i8a8-attention-qk-cross-m1024-n512
 *   - tile32x64r8x4w16x4k32: output differs from the default on i8a8-attention-qk-cross-m1024-n512
 *   - tile32x64r8x8w8x4k16: output differs from the default on i8a8-attention-qk-cross-m1024-n512
 *   - tile32x64r8x8w8x4k32: output differs from the default on i8a8-attention-qk-cross-m1024-n512
 *   - tile64x128r4x8w16x16k16: ×0.878 < ×1.050 on i8a8-attention-qk-cross-m1024-n512
 *   - tile64x128r4x8w16x16k32: ×0.885 < ×1.050 on i8a8-attention-qk-cross-m1024-n512
 *   - tile64x128r8x8w16x8k16: ×0.993 < ×1.050 on i8a8-attention-qk-self-m4096-n4096
 *   - tile64x128r8x8w16x8k32: output differs from the default on i8a8-attention-qk-cross-m1024-n512
 *   - tile64x32r4x4w8x16k16: ×0.894 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile64x32r4x4w8x16k32: ×0.880 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile64x32r8x4w8x8k16: ×0.990 < ×1.050 on i8a8-attention-qk-cross-m1024-n512
 *   - tile64x32r8x4w8x8k32: ×0.974 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile64x64r4x4w16x16k16: ×0.958 < ×1.050 on i8a8-attention-qk-self-m4096-n4096
 *   - tile64x64r4x4w16x16k32: ×0.963 < ×1.050 on i8a8-attention-qk-self-m4096-n4096
 *   - tile64x64r4x8w8x16k16: geometric mean ×1.087 (short of the adopted ×1.218)
 *   - tile64x64r4x8w8x16k32: ×1.008 < ×1.050 on i8a8-attention-qk-self-m4096-n4096
 *   - tile64x64r8x4w16x8k16: geometric mean ×1.168 (short of the adopted ×1.218)
 *   - tile64x64r8x4w16x8k32: geometric mean ×1.169 (short of the adopted ×1.218)
 *   - tile64x64r8x8w8x8k16: ×0.971 < ×1.050 on i8a8-attention-qk-cross-m4096-n512
 *   - tile64x64r8x8w8x8k32: output differs from the default on i8a8-attention-qk-cross-m1024-n512
 * - i8a8.attentionPv (i8a8 attention ③PV, 4 cases): adopted tile128x64r8x4w16x16k16 ×1.210 (×1.135–×1.300)
 *   - tile128x128r8x8w16x16k16: ×0.878 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 *   - tile128x128r8x8w16x16k32: ×0.883 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 *   - tile128x32r8x4w8x16k16: ×0.814 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile128x32r8x4w8x16k32: ×0.752 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile128x64r8x4w16x16k32: geometric mean ×1.175 (short of the adopted ×1.210)
 *   - tile128x64r8x8w8x16k16: ×0.966 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 *   - tile128x64r8x8w8x16k32: ×0.837 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 *   - tile16x128r4x8w16x4k16: ×0.728 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile16x128r4x8w16x4k32: ×0.743 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile16x32r4x4w8x4k16: ×0.649 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile16x32r4x4w8x4k32: ×0.643 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile16x64r4x4w16x4k16: ×0.679 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile16x64r4x4w16x4k32: ×0.768 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile16x64r4x8w8x4k16: ×0.760 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile16x64r4x8w8x4k32: ×0.764 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x128r4x8w16x8k16: ×0.879 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x128r4x8w16x8k32: ×0.894 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x128r8x8w16x4k16: ×0.933 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x128r8x8w16x4k32: ×0.896 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x32r4x4w8x8k16: ×0.764 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x32r4x4w8x8k32: ×0.789 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x32r8x4w8x4k16: ×0.750 < ×1.050 on i8a8-attention-pv-self-m1024-n1024
 *   - tile32x32r8x4w8x4k32: ×0.716 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 *   - tile32x64r4x4w16x8k16: ×0.897 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x64r4x4w16x8k32: ×0.910 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x64r4x8w8x8k16: ×0.933 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x64r4x8w8x8k32: ×0.934 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x64r8x4w16x4k16: ×0.957 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x64r8x4w16x4k32: ×0.950 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x64r8x8w8x4k16: ×0.890 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile32x64r8x8w8x4k32: ×0.776 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 *   - tile64x128r4x8w16x16k16: ×0.944 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile64x128r4x8w16x16k32: ×0.941 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile64x128r8x8w16x8k32: ×0.975 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 *   - tile64x32r4x4w8x16k16: ×0.840 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile64x32r4x4w8x16k32: ×0.843 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile64x32r8x4w8x8k16: ×0.814 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile64x32r8x4w8x8k32: ×0.748 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile64x64r4x4w16x16k16: ×0.964 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile64x64r4x4w16x16k32: ×0.983 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile64x64r4x8w8x16k16: ×1.048 < ×1.050 on i8a8-attention-pv-self-m4096-n4096
 *   - tile64x64r4x8w8x16k32: ×1.013 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 *   - tile64x64r8x4w16x8k16: geometric mean ×1.173 (short of the adopted ×1.210)
 *   - tile64x64r8x4w16x8k32: geometric mean ×1.104 (short of the adopted ×1.210)
 *   - tile64x64r8x8w8x8k16: ×0.939 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 *   - tile64x64r8x8w8x8k32: ×0.836 < ×1.050 on i8a8-attention-pv-cross-m4096-n512
 */
import type { GeometryProfile } from "../geometry-profile.ts";

export const NVIDIA_BLACKWELL: GeometryProfile = {
  id: "nvidia-blackwell",
  match: { vendor: "nvidia", architecture: "blackwell" },
  gemmRows: [
    { maxRows: 16, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 32, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 64, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 128, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 16 } },
    { maxRows: 512, geometry: { regM: 8, regN: 4, wgX: 16, wgY: 8 } },
    { maxRows: Number.POSITIVE_INFINITY, geometry: { regM: 8, regN: 8, wgX: 16, wgY: 16 } },
  ],
  attention: {
    qk: { regM: 8, regN: 8, wgX: 16, wgY: 16 },
    pv: { regM: 8, regN: 8, wgX: 16, wgY: 16 },
  },
  conv2d: {
    rows64: { regM: 8, regN: 8, wgX: 16, wgY: 8 },
    rows32: { regM: 8, regN: 8, wgX: 16, wgY: 4 },
  },
  i8a8: {
    linear: { regM: 8, regN: 4, wgX: 16, wgY: 16, tileK: 16 },
    attentionQk: { regM: 8, regN: 4, wgX: 16, wgY: 16, tileK: 16 },
    attentionPv: { regM: 8, regN: 4, wgX: 16, wgY: 16, tileK: 16 },
  },
  provenance: {
    sweep:
      "outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-37-27.914Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-29T20-42-08.545Z.json, outputs/bench-browser/geometry-sweep-browser-2026-10-01T16-46-50.379Z.json",
    sha256:
      "084cd0efd7ea7780dfcdab5b53ee4be8177638c7d8984920096a4851b767aa90, 1ae3a8f6d9a1270d8e3b3ad41aa7fe13757c976a323de7d2b0b99139800cb2e5, 0e57d4efcb14efe0e01767066fce05e1784ca43660a4f8a78c5a1a2c48d1ae87",
    date: "2026-09-27T18:37:27.914Z, 2026-09-29T20:42:08.545Z, 2026-10-01T16:46:50.379Z",
    candidateSet: "full, full, full",
    userAgent: [
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36",
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
    ],
    adapter: {
      vendor: "nvidia",
      architecture: "blackwell",
      device: "0x2c05",
      description: "NVIDIA GeForce RTX 5070 Ti",
    },
    kernels: "27067d014c2f3b30",
    caseSet: "c1060d8829f3b9de",
  },
};
