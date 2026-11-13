/**
 * Geometry profile `nvidia-blackwell` (**generated — do not edit by hand**).
 *
 * A static table of tile geometries for adapter `nvidia / blackwell`, derived from the
 * sweep records below with `deriveGeometryProfile` of the public surface `@karume/runtime/tune` (ADR 0117
 * decision 2). The records were made in Chrome by
 * the geometry-sweep page of apps/playground
 * (full candidate set); the short Node scripts that derived the table and wrote this file are kept, untracked,
 * in `outputs/diag/blackwell-rederive-2026-10-29/`. The runtime only selects this table by the adapter's
 * `match` (exact match on vendor / architecture). Nothing is measured at run time (no autotuning — the
 * MUST NOT of the ADR 0022 addendum of 2026-08-10 (later), reaffirmed by ADR 0115 decision 8 and ADR 0117 decision 9).
 * To change a value, take the sweep again and regenerate the table.
 *
 * Sweeps (adapter nvidia / blackwell / 0x2c05 / NVIDIA GeForce RTX 5070 Ti; Chrome 156, Windows;
 * WebGPU backend not recorded; timestamps unquantised in all 2 records — every family except `i8a8-linear`,
 * whose records are listed at the end of this comment):
 *
 * - outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-10-28T20-44-29.json (sha256 39d203a76278ddafd386e451445ff739acc765837f5429cc9c33b8157df020e4, 2026-10-28T20:44:29.316Z)
 *   Chrome 156, Windows; backend not recorded; timestamps unquantised; page the geometry-sweep page of apps/playground
 * - outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-10-30T20-12-41.json (sha256 5778eb62ad937e6850f62a4cae93c7d1ab8dcea59f0e9a42198fbd2fd06b257e, 2026-10-30T20:12:41.839Z)
 *   Chrome 156, Windows; backend not recorded; timestamps unquantised; page the geometry-sweep page of apps/playground
 *
 * Adoption criteria (the rules of `deriveProfile` — ADR 0115 decision 4, ADR 0116, ADR 0117 decision 3): among
 * the geometries whose output matches the default on every case of the class and whose ratio to the default is
 * ×1.050 or more, the one with the largest geometric mean across the cases; if there is none, the default. When
 * several sweeps measured the same case, the ratio is the geometric mean of those observations. A case
 * whose default re-measurement ratio is outside 0.9–1.1, or whose re-measurement failed, is dropped from that
 * sweep's ratio material (output match and failures are still checked).
 * An observation whose ratio has a rounding-error bound above 1% (Chrome's quantised timestamps) is dropped
 * likewise.
 *
 * Adoption:
 *
 * - gemmRows ≤ 16 (linear / matmul / bmm row count ≤ 16, 4 cases): keeps the default reg16x16r1x4w4 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: matmul-m16-n3072-k1024: ×0.131 < ×1.050
 *   - reg128x16r8x4w4: matmul-m16-n3072-k1024: ×0.240 < ×1.050
 *   - reg128x32r8x4w8: matmul-m16-n3072-k1024: ×0.272 < ×1.050
 *   - reg128x32r8x8w4: matmul-m16-n3072-k1024: ×0.206 < ×1.050
 *   - reg128x64r8x4w16: bmm-b16-m16-n64-k128: ×0.277 < ×1.050
 *   - reg128x64r8x8w8: bmm-b16-m16-n64-k128: ×0.238 < ×1.050
 *   - reg16x128r4x8w16: linear-m16-n3072-k1024: ×0.205 < ×1.050
 *   - reg16x16r2x4w4: bmm-b16-m16-n64-k128: ×0.590 < ×1.050
 *   - reg16x16r4x4w4: linear-m16-n3072-k1024: ×0.300 < ×1.050
 *   - reg16x32r1x8w4: linear-m16-n3072-k1024: ×0.705 < ×1.050
 *   - reg16x32r2x4w8: linear-m16-n3072-k1024: ×0.764 < ×1.050
 *   - reg16x32r2x8w4: linear-m16-n3072-k1024: ×0.357 < ×1.050
 *   - reg16x32r4x4w8: linear-m16-n3072-k1024: ×0.362 < ×1.050
 *   - reg16x32r4x8w4: linear-m16-n3072-k1024: ×0.185 < ×1.050
 *   - reg16x64r2x8w8: linear-m16-n3072-k1024: ×0.414 < ×1.050
 *   - reg16x64r4x4w16: linear-m16-n3072-k1024: ×0.413 < ×1.050
 *   - reg16x64r4x8w8: linear-m16-n3072-k1024: ×0.198 < ×1.050
 *   - reg32x128r4x8w16: linear-m16-n3072-k1024: ×0.326 < ×1.050
 *   - reg32x128r8x8w16: linear-m16-n3072-k1024: ×0.159 < ×1.050
 *   - reg32x16r2x4w4: matmul-m16-n3072-k1024: ×0.600 < ×1.050
 *   - reg32x16r4x4w4: bmm-b16-m16-n64-k128: ×0.419 < ×1.050
 *   - reg32x16r8x4w4: linear-m16-n3072-k1024: ×0.234 < ×1.050
 *   - reg32x32r2x4w8: bmm-b16-m16-n64-k128: ×0.640 < ×1.050
 *   - reg32x32r2x8w4: bmm-b16-m16-n64-k128: ×0.528 < ×1.050
 *   - reg32x32r4x4w8: bmm-b16-m16-n64-k128: ×0.535 < ×1.050
 *   - reg32x32r4x8w4: linear-m16-n3072-k1024: ×0.265 < ×1.050
 *   - reg32x32r8x4w8: linear-m16-n3072-k1024: ×0.278 < ×1.050
 *   - reg32x32r8x8w4: linear-m16-n3072-k1024: ×0.147 < ×1.050
 *   - reg32x64r2x8w8: bmm-b16-m16-n64-k128: ×0.577 < ×1.050
 *   - reg32x64r4x4w16: bmm-b16-m16-n64-k128: ×0.588 < ×1.050
 *   - reg32x64r4x8w8: linear-m16-n3072-k1024: ×0.310 < ×1.050
 *   - reg32x64r8x4w16: linear-m16-n3072-k1024: ×0.307 < ×1.050
 *   - reg32x64r8x8w8: linear-m16-n3072-k1024: ×0.155 < ×1.050
 *   - reg4x16r1x4w4: linear-m16-n3072-k1024: ×0.348 < ×1.050
 *   - reg4x32r1x8w4: linear-m16-n3072-k1024: ×0.239 < ×1.050
 *   - reg64x128r4x8w16: matmul-m16-n3072-k1024: ×0.250 < ×1.050
 *   - reg64x128r8x8w16: linear-m16-n3072-k1024: ×0.223 < ×1.050
 *   - reg64x16r4x4w4: matmul-m16-n3072-k1024: ×0.435 < ×1.050
 *   - reg64x16r8x4w4: bmm-b16-m16-n64-k128: ×0.284 < ×1.050
 *   - reg64x32r4x4w8: matmul-m16-n3072-k1024: ×0.428 < ×1.050
 *   - reg64x32r4x8w4: bmm-b16-m16-n64-k128: ×0.330 < ×1.050
 *   - reg64x32r8x4w8: bmm-b16-m16-n64-k128: ×0.376 < ×1.050
 *   - reg64x32r8x8w4: linear-m16-n3072-k1024: ×0.195 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m16-n64-k128: ×0.400 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m16-n64-k128: ×0.361 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m16-n64-k128: ×0.390 < ×1.050
 *   - reg64x64r8x8w8: linear-m16-n3072-k1024: ×0.226 < ×1.050
 *   - reg8x16r1x4w4: linear-m16-n3072-k1024: ×0.780 < ×1.050
 *   - reg8x16r2x4w4: linear-m16-n3072-k1024: ×0.389 < ×1.050
 *   - reg8x32r1x8w4: linear-m16-n3072-k1024: ×0.455 < ×1.050
 *   - reg8x32r2x4w8: linear-m16-n3072-k1024: ×0.485 < ×1.050
 *   - reg8x32r2x8w4: linear-m16-n3072-k1024: ×0.234 < ×1.050
 *   - reg8x64r2x8w8: linear-m16-n3072-k1024: ×0.227 < ×1.050
 * - gemmRows 17〜32 (linear / matmul / bmm row count 17〜32, 4 cases): keeps the default reg16x16r1x4w4 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: bmm-b16-m32-n64-k128: ×0.163 < ×1.050
 *   - reg128x16r8x4w4: matmul-m32-n3072-k1024: ×0.305 < ×1.050
 *   - reg128x32r8x4w8: matmul-m32-n3072-k1024: ×0.368 < ×1.050
 *   - reg128x32r8x8w4: bmm-b16-m32-n64-k128: ×0.235 < ×1.050
 *   - reg128x64r8x4w16: bmm-b16-m32-n64-k128: ×0.302 < ×1.050
 *   - reg128x64r8x8w8: bmm-b16-m32-n64-k128: ×0.256 < ×1.050
 *   - reg16x128r4x8w16: linear-m32-n3072-k1024: ×0.318 < ×1.050
 *   - reg16x16r2x4w4: bmm-b16-m32-n64-k128: ×0.632 < ×1.050
 *   - reg16x16r4x4w4: bmm-b16-m32-n64-k128: ×0.414 < ×1.050
 *   - reg16x32r1x8w4: bmm-b16-m32-n64-k128: ×0.893 < ×1.050
 *   - reg16x32r2x4w8: bmm-b16-m32-n64-k64: ×0.901 < ×1.050
 *   - reg16x32r2x8w4: linear-m32-n3072-k1024: ×0.534 < ×1.050
 *   - reg16x32r4x4w8: linear-m32-n3072-k1024: ×0.542 < ×1.050
 *   - reg16x32r4x8w4: linear-m32-n3072-k1024: ×0.286 < ×1.050
 *   - reg16x64r2x8w8: linear-m32-n3072-k1024: ×0.616 < ×1.050
 *   - reg16x64r4x4w16: linear-m32-n3072-k1024: ×0.620 < ×1.050
 *   - reg16x64r4x8w8: linear-m32-n3072-k1024: ×0.286 < ×1.050
 *   - reg32x128r4x8w16: bmm-b16-m32-n64-k128: ×0.439 < ×1.050
 *   - reg32x128r8x8w16: linear-m32-n3072-k1024: ×0.236 < ×1.050
 *   - reg32x16r2x4w4: bmm-b16-m32-n64-k128: ×0.672 < ×1.050
 *   - reg32x16r4x4w4: bmm-b16-m32-n64-k128: ×0.407 < ×1.050
 *   - reg32x16r8x4w4: bmm-b16-m32-n64-k128: ×0.232 < ×1.050
 *   - reg32x32r2x4w8: bmm-b16-m32-n64-k128: ×0.685 < ×1.050
 *   - reg32x32r2x8w4: bmm-b16-m32-n64-k128: ×0.569 < ×1.050
 *   - reg32x32r4x4w8: bmm-b16-m32-n64-k128: ×0.582 < ×1.050
 *   - reg32x32r4x8w4: bmm-b16-m32-n64-k128: ×0.298 < ×1.050
 *   - reg32x32r8x4w8: bmm-b16-m32-n64-k128: ×0.330 < ×1.050
 *   - reg32x32r8x8w4: bmm-b16-m32-n64-k128: ×0.182 < ×1.050
 *   - reg32x64r2x8w8: bmm-b16-m32-n64-k128: ×0.596 < ×1.050
 *   - reg32x64r4x4w16: bmm-b16-m32-n64-k128: ×0.624 < ×1.050
 *   - reg32x64r4x8w8: bmm-b16-m32-n64-k128: ×0.386 < ×1.050
 *   - reg32x64r8x4w16: bmm-b16-m32-n64-k128: ×0.418 < ×1.050
 *   - reg32x64r8x8w8: bmm-b16-m32-n64-k128: ×0.222 < ×1.050
 *   - reg4x16r1x4w4: linear-m32-n3072-k1024: ×0.273 < ×1.050
 *   - reg4x32r1x8w4: linear-m32-n3072-k1024: ×0.283 < ×1.050
 *   - reg64x128r4x8w16: bmm-b16-m32-n64-k128: ×0.281 < ×1.050
 *   - reg64x128r8x8w16: bmm-b16-m32-n64-k128: ×0.264 < ×1.050
 *   - reg64x16r4x4w4: bmm-b16-m32-n64-k128: ×0.467 < ×1.050
 *   - reg64x16r8x4w4: bmm-b16-m32-n64-k128: ×0.266 < ×1.050
 *   - reg64x32r4x4w8: bmm-b16-m32-n64-k128: ×0.538 < ×1.050
 *   - reg64x32r4x8w4: bmm-b16-m32-n64-k128: ×0.330 < ×1.050
 *   - reg64x32r8x4w8: bmm-b16-m32-n64-k128: ×0.368 < ×1.050
 *   - reg64x32r8x8w4: bmm-b16-m32-n64-k128: ×0.204 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m32-n64-k128: ×0.431 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m32-n64-k128: ×0.380 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m32-n64-k128: ×0.413 < ×1.050
 *   - reg64x64r8x8w8: bmm-b16-m32-n64-k128: ×0.246 < ×1.050
 *   - reg8x16r1x4w4: linear-m32-n3072-k1024: ×0.738 < ×1.050
 *   - reg8x16r2x4w4: linear-m32-n3072-k1024: ×0.463 < ×1.050
 *   - reg8x32r1x8w4: linear-m32-n3072-k1024: ×0.599 < ×1.050
 *   - reg8x32r2x4w8: linear-m32-n3072-k1024: ×0.633 < ×1.050
 *   - reg8x32r2x8w4: linear-m32-n3072-k1024: ×0.318 < ×1.050
 *   - reg8x64r2x8w8: linear-m32-n3072-k1024: ×0.326 < ×1.050
 * - gemmRows 33〜64 (linear / matmul / bmm row count 33〜64, 4 cases): keeps the default reg16x16r1x4w4 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: bmm-b16-m64-n64-k128: ×0.200 < ×1.050
 *   - reg128x16r8x4w4: bmm-b16-m64-n64-k128: ×0.334 < ×1.050
 *   - reg128x32r8x4w8: bmm-b16-m64-n64-k128: ×0.426 < ×1.050
 *   - reg128x32r8x8w4: bmm-b16-m64-n64-k128: ×0.256 < ×1.050
 *   - reg128x64r8x4w16: bmm-b16-m64-n64-k128: ×0.360 < ×1.050
 *   - reg128x64r8x8w8: bmm-b16-m64-n64-k128: ×0.291 < ×1.050
 *   - reg16x128r4x8w16: linear-m64-n3072-k1024: ×0.522 < ×1.050
 *   - reg16x16r2x4w4: bmm-b16-m64-n64-k128: ×0.663 < ×1.050
 *   - reg16x16r4x4w4: bmm-b16-m64-n64-k128: ×0.506 < ×1.050
 *   - reg16x32r1x8w4: bmm-b16-m64-n64-k128: ×1.006 < ×1.050
 *   - reg16x32r2x4w8: bmm-b16-m64-n64-k128: ×1.013 < ×1.050
 *   - reg16x32r2x8w4: bmm-b16-m64-n64-k128: ×0.683 < ×1.050
 *   - reg16x32r4x4w8: bmm-b16-m64-n64-k128: ×0.692 < ×1.050
 *   - reg16x32r4x8w4: bmm-b16-m64-n64-k128: ×0.394 < ×1.050
 *   - reg16x64r2x8w8: bmm-b16-m64-n64-k128: ×0.838 < ×1.050
 *   - reg16x64r4x4w16: bmm-b16-m64-n64-k128: ×0.852 < ×1.050
 *   - reg16x64r4x8w8: linear-m64-n3072-k1024: ×0.489 < ×1.050
 *   - reg32x128r4x8w16: bmm-b16-m64-n64-k128: ×0.552 < ×1.050
 *   - reg32x128r8x8w16: bmm-b16-m64-n64-k128: ×0.336 < ×1.050
 *   - reg32x16r2x4w4: bmm-b16-m64-n64-k128: ×0.756 < ×1.050
 *   - reg32x16r4x4w4: bmm-b16-m64-n64-k128: ×0.503 < ×1.050
 *   - reg32x16r8x4w4: bmm-b16-m64-n64-k128: ×0.290 < ×1.050
 *   - reg32x32r2x4w8: bmm-b16-m64-n64-k128: ×0.857 < ×1.050
 *   - reg32x32r2x8w4: bmm-b16-m64-n64-k128: ×0.713 < ×1.050
 *   - reg32x32r4x4w8: bmm-b16-m64-n64-k128: ×0.728 < ×1.050
 *   - reg32x32r4x8w4: bmm-b16-m64-n64-k128: ×0.372 < ×1.050
 *   - reg32x32r8x4w8: bmm-b16-m64-n64-k128: ×0.411 < ×1.050
 *   - reg32x32r8x8w4: bmm-b16-m64-n64-k128: ×0.228 < ×1.050
 *   - reg32x64r2x8w8: bmm-b16-m64-n64-k128: ×0.749 < ×1.050
 *   - reg32x64r4x4w16: bmm-b16-m64-n64-k128: ×0.782 < ×1.050
 *   - reg32x64r4x8w8: bmm-b16-m64-n64-k128: ×0.485 < ×1.050
 *   - reg32x64r8x4w16: bmm-b16-m64-n64-k128: ×0.526 < ×1.050
 *   - reg32x64r8x8w8: bmm-b16-m64-n64-k128: ×0.278 < ×1.050
 *   - reg4x16r1x4w4: linear-m64-n3072-k1024: ×0.263 < ×1.050
 *   - reg4x32r1x8w4: linear-m64-n3072-k1024: ×0.257 < ×1.050
 *   - reg64x128r4x8w16: bmm-b16-m64-n64-k128: ×0.333 < ×1.050
 *   - reg64x128r8x8w16: bmm-b16-m64-n64-k128: ×0.319 < ×1.050
 *   - reg64x16r4x4w4: bmm-b16-m64-n64-k128: ×0.506 < ×1.050
 *   - reg64x16r8x4w4: bmm-b16-m64-n64-k128: ×0.269 < ×1.050
 *   - reg64x32r4x4w8: bmm-b16-m64-n64-k128: ×0.651 < ×1.050
 *   - reg64x32r4x8w4: bmm-b16-m64-n64-k128: ×0.371 < ×1.050
 *   - reg64x32r8x4w8: bmm-b16-m64-n64-k128: ×0.411 < ×1.050
 *   - reg64x32r8x8w4: bmm-b16-m64-n64-k128: ×0.216 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m64-n64-k128: ×0.508 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m64-n64-k128: ×0.449 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m64-n64-k128: ×0.496 < ×1.050
 *   - reg64x64r8x8w8: bmm-b16-m64-n64-k128: ×0.281 < ×1.050
 *   - reg8x16r1x4w4: linear-m64-n3072-k1024: ×0.679 < ×1.050
 *   - reg8x16r2x4w4: linear-m64-n3072-k1024: ×0.429 < ×1.050
 *   - reg8x32r1x8w4: linear-m64-n3072-k1024: ×0.764 < ×1.050
 *   - reg8x32r2x4w8: linear-m64-n3072-k1024: ×0.784 < ×1.050
 *   - reg8x32r2x8w4: linear-m64-n3072-k1024: ×0.452 < ×1.050
 *   - reg8x64r2x8w8: linear-m64-n3072-k1024: ×0.516 < ×1.050
 * - gemmRows 65〜128 (linear / matmul / bmm row count 65〜128, 4 cases): keeps the default reg64x32r4x4w8 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: bmm-b16-m128-n64-k128: ×0.284 < ×1.050
 *   - reg128x16r8x4w4: bmm-b16-m128-n64-k128: ×0.410 < ×1.050
 *   - reg128x32r8x4w8: bmm-b16-m128-n64-k128: ×0.585 < ×1.050
 *   - reg128x32r8x8w4: bmm-b16-m128-n64-k128: ×0.328 < ×1.050
 *   - reg128x64r8x4w16: bmm-b16-m128-n64-k128: ×0.524 < ×1.050
 *   - reg128x64r8x8w8: bmm-b16-m128-n64-k64: ×0.398 < ×1.050
 *   - reg16x128r4x8w16: linear-m128-n2048-k1024: ×0.458 < ×1.050
 *   - reg16x16r1x4w4: linear-m128-n2048-k1024: ×0.660 < ×1.050
 *   - reg16x16r2x4w4: linear-m128-n2048-k1024: ×0.652 < ×1.050
 *   - reg16x16r4x4w4: linear-m128-n2048-k1024: ×0.452 < ×1.050
 *   - reg16x32r1x8w4: linear-m128-n2048-k1024: ×0.732 < ×1.050
 *   - reg16x32r2x4w8: linear-m128-n2048-k1024: ×0.766 < ×1.050
 *   - reg16x32r2x8w4: linear-m128-n2048-k1024: ×0.622 < ×1.050
 *   - reg16x32r4x4w8: linear-m128-n2048-k1024: ×0.648 < ×1.050
 *   - reg16x32r4x8w4: linear-m128-n2048-k1024: ×0.374 < ×1.050
 *   - reg16x64r2x8w8: linear-m128-n2048-k1024: ×0.720 < ×1.050
 *   - reg16x64r4x4w16: linear-m128-n2048-k1024: ×0.732 < ×1.050
 *   - reg16x64r4x8w8: linear-m128-n2048-k1024: ×0.424 < ×1.050
 *   - reg32x128r4x8w16: linear-m128-n2048-k1024: ×0.782 < ×1.050
 *   - reg32x128r8x8w16: linear-m128-n2048-k1024: ×0.375 < ×1.050
 *   - reg32x16r2x4w4: linear-m128-n2048-k1024: ×0.821 < ×1.050
 *   - reg32x16r4x4w4: linear-m128-n2048-k1024: ×0.719 < ×1.050
 *   - reg32x16r8x4w4: linear-m128-n2048-k1024: ×0.405 < ×1.050
 *   - reg32x32r2x4w8: linear-m128-n2048-k1024: ×1.022 < ×1.050
 *   - reg32x32r2x8w4: linear-m128-n2048-k1024: ×0.908 < ×1.050
 *   - reg32x32r4x4w8: linear-m128-n2048-k1024: ×0.924 < ×1.050
 *   - reg32x32r4x8w4: linear-m128-n2048-k1024: ×0.536 < ×1.050
 *   - reg32x32r8x4w8: linear-m128-n2048-k1024: ×0.562 < ×1.050
 *   - reg32x32r8x8w4: linear-m128-n2048-k1024: ×0.315 < ×1.050
 *   - reg32x64r2x8w8: linear-m128-n2048-k1024: ×0.957 < ×1.050
 *   - reg32x64r4x4w16: linear-m128-n2048-k1024: ×0.992 < ×1.050
 *   - reg32x64r4x8w8: linear-m128-n2048-k1024: ×0.711 < ×1.050
 *   - reg32x64r8x4w16: linear-m128-n2048-k1024: ×0.679 < ×1.050
 *   - reg32x64r8x8w8: linear-m128-n2048-k1024: ×0.334 < ×1.050
 *   - reg4x16r1x4w4: linear-m128-n2048-k1024: ×0.171 < ×1.050
 *   - reg4x32r1x8w4: linear-m128-n2048-k1024: ×0.166 < ×1.050
 *   - reg64x128r4x8w16: bmm-b16-m128-n64-k128: ×0.512 < ×1.050
 *   - reg64x128r8x8w16: bmm-b16-m128-n64-k128: ×0.491 < ×1.050
 *   - reg64x16r4x4w4: bmm-b16-m128-n64-k128: ×0.729 < ×1.050
 *   - reg64x16r8x4w4: bmm-b16-m128-n64-k128: ×0.410 < ×1.050
 *   - reg64x32r4x8w4: bmm-b16-m128-n64-k128: ×0.569 < ×1.050
 *   - reg64x32r8x4w8: bmm-b16-m128-n64-k128: ×0.630 < ×1.050
 *   - reg64x32r8x8w4: bmm-b16-m128-n64-k128: ×0.332 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m128-n64-k128: ×0.779 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m128-n64-k64: ×0.681 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m128-n64-k128: ×0.763 < ×1.050
 *   - reg64x64r8x8w8: bmm-b16-m128-n64-k128: ×0.432 < ×1.050
 *   - reg8x16r1x4w4: linear-m128-n2048-k1024: ×0.431 < ×1.050
 *   - reg8x16r2x4w4: linear-m128-n2048-k1024: ×0.277 < ×1.050
 *   - reg8x32r1x8w4: linear-m128-n2048-k1024: ×0.495 < ×1.050
 *   - reg8x32r2x4w8: linear-m128-n2048-k1024: ×0.508 < ×1.050
 *   - reg8x32r2x8w4: linear-m128-n2048-k1024: ×0.312 < ×1.050
 *   - reg8x64r2x8w8: linear-m128-n2048-k1024: ×0.425 < ×1.050
 * - gemmRows 129〜256 (linear / matmul / bmm row count 129〜256, 4 cases): keeps the default reg64x32r4x4w8 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: bmm-b16-m256-n64-k128: ×0.385 < ×1.050
 *   - reg128x16r8x4w4: bmm-b16-m256-n64-k128: ×0.531 < ×1.050
 *   - reg128x32r8x4w8: bmm-b16-m256-n64-k128: ×0.783 < ×1.050
 *   - reg128x32r8x8w4: bmm-b16-m256-n64-k128: ×0.443 < ×1.050
 *   - reg128x64r8x4w16: bmm-b16-m256-n64-k128: ×0.710 < ×1.050
 *   - reg128x64r8x8w8: bmm-b16-m256-n64-k64: ×0.539 < ×1.050
 *   - reg16x128r4x8w16: linear-m256-n2048-k1024: ×0.536 < ×1.050
 *   - reg16x16r1x4w4: linear-m256-n2048-k1024: ×0.510 < ×1.050
 *   - reg16x16r2x4w4: linear-m256-n2048-k1024: ×0.477 < ×1.050
 *   - reg16x16r4x4w4: linear-m256-n2048-k1024: ×0.332 < ×1.050
 *   - reg16x32r1x8w4: linear-m256-n2048-k1024: ×0.583 < ×1.050
 *   - reg16x32r2x4w8: linear-m256-n2048-k1024: ×0.603 < ×1.050
 *   - reg16x32r2x8w4: linear-m256-n2048-k1024: ×0.598 < ×1.050
 *   - reg16x32r4x4w8: linear-m256-n2048-k1024: ×0.619 < ×1.050
 *   - reg16x32r4x8w4: linear-m256-n2048-k1024: ×0.374 < ×1.050
 *   - reg16x64r2x8w8: linear-m256-n2048-k1024: ×0.596 < ×1.050
 *   - reg16x64r4x4w16: linear-m256-n2048-k1024: ×0.642 < ×1.050
 *   - reg16x64r4x8w8: linear-m256-n2048-k1024: ×0.516 < ×1.050
 *   - reg32x128r4x8w16: bmm-b16-m256-n64-k64: ×0.703 < ×1.050
 *   - reg32x128r8x8w16: linear-m256-n2048-k1024: ×0.528 < ×1.050
 *   - reg32x16r2x4w4: linear-m256-n2048-k1024: ×0.651 < ×1.050
 *   - reg32x16r4x4w4: linear-m256-n2048-k1024: ×0.680 < ×1.050
 *   - reg32x16r8x4w4: linear-m256-n2048-k1024: ×0.428 < ×1.050
 *   - reg32x32r2x4w8: linear-m256-n2048-k1024: ×0.800 < ×1.050
 *   - reg32x32r2x8w4: linear-m256-n2048-k1024: ×0.812 < ×1.050
 *   - reg32x32r4x4w8: linear-m256-n2048-k1024: ×0.848 < ×1.050
 *   - reg32x32r4x8w4: linear-m256-n2048-k1024: ×0.641 < ×1.050
 *   - reg32x32r8x4w8: linear-m256-n2048-k1024: ×0.656 < ×1.050
 *   - reg32x32r8x8w4: linear-m256-n2048-k1024: ×0.376 < ×1.050
 *   - reg32x64r2x8w8: linear-m256-n2048-k1024: ×0.832 < ×1.050
 *   - reg32x64r4x4w16: linear-m256-n2048-k1024: ×0.899 < ×1.050
 *   - reg32x64r4x8w8: linear-m256-n2048-k1024: ×0.772 < ×1.050
 *   - reg32x64r8x4w16: linear-m256-n2048-k1024: ×0.752 < ×1.050
 *   - reg32x64r8x8w8: linear-m256-n2048-k1024: ×0.481 < ×1.050
 *   - reg4x16r1x4w4: linear-m256-n2048-k1024: ×0.129 < ×1.050
 *   - reg4x32r1x8w4: linear-m256-n2048-k1024: ×0.134 < ×1.050
 *   - reg64x128r4x8w16: bmm-b16-m256-n64-k64: ×0.669 < ×1.050
 *   - reg64x128r8x8w16: bmm-b16-m256-n64-k128: ×0.649 < ×1.050
 *   - reg64x16r4x4w4: linear-m256-n2048-k1024: ×0.773 < ×1.050
 *   - reg64x16r8x4w4: bmm-b16-m256-n64-k128: ×0.529 < ×1.050
 *   - reg64x32r4x8w4: bmm-b16-m256-n64-k64: ×0.720 < ×1.050
 *   - reg64x32r8x4w8: bmm-b16-m256-n64-k128: ×0.808 < ×1.050
 *   - reg64x32r8x8w4: bmm-b16-m256-n64-k128: ×0.440 < ×1.050
 *   - reg64x64r4x4w16: linear-m256-n2048-k1024: ×1.018 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m256-n64-k64: ×0.874 < ×1.050
 *   - reg64x64r8x4w16: linear-m256-n2048-k1024: ×0.965 < ×1.050
 *   - reg64x64r8x8w8: bmm-b16-m256-n64-k128: ×0.580 < ×1.050
 *   - reg8x16r1x4w4: linear-m256-n2048-k1024: ×0.344 < ×1.050
 *   - reg8x16r2x4w4: linear-m256-n2048-k1024: ×0.224 < ×1.050
 *   - reg8x32r1x8w4: linear-m256-n2048-k1024: ×0.361 < ×1.050
 *   - reg8x32r2x4w8: linear-m256-n2048-k1024: ×0.371 < ×1.050
 *   - reg8x32r2x8w4: linear-m256-n2048-k1024: ×0.228 < ×1.050
 *   - reg8x64r2x8w8: linear-m256-n2048-k1024: ×0.376 < ×1.050
 * - gemmRows 257〜512 (linear / matmul / bmm row count 257〜512, 5 cases): keeps the default reg64x32r4x4w8 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: bmm-b16-m512-n64-k512: ×0.522 < ×1.050
 *   - reg128x16r8x4w4: bmm-b16-m512-n64-k512: ×0.538 < ×1.050
 *   - reg128x32r8x4w8: bmm-b16-m512-n64-k512: ×0.803 < ×1.050
 *   - reg128x32r8x8w4: bmm-b16-m512-n64-k512: ×0.532 < ×1.050
 *   - reg128x64r8x4w16: bmm-b16-m512-n64-k64: ×0.964 < ×1.050
 *   - reg128x64r8x8w8: bmm-b16-m512-n64-k64: ×0.706 < ×1.050
 *   - reg16x128r4x8w16: bmm-b16-m512-n64-k512: ×0.594 < ×1.050
 *   - reg16x16r1x4w4: linear-m512-n2048-k1024: ×0.488 < ×1.050
 *   - reg16x16r2x4w4: bmm-b16-m512-n512-k64: ×0.456 < ×1.050
 *   - reg16x16r4x4w4: linear-m512-n2048-k1024: ×0.345 < ×1.050
 *   - reg16x32r1x8w4: linear-m512-n2048-k1024: ×0.573 < ×1.050
 *   - reg16x32r2x4w8: linear-m512-n2048-k1024: ×0.596 < ×1.050
 *   - reg16x32r2x8w4: linear-m512-n2048-k1024: ×0.557 < ×1.050
 *   - reg16x32r4x4w8: linear-m512-n2048-k1024: ×0.576 < ×1.050
 *   - reg16x32r4x8w4: linear-m512-n2048-k1024: ×0.342 < ×1.050
 *   - reg16x64r2x8w8: linear-m512-n2048-k1024: ×0.575 < ×1.050
 *   - reg16x64r4x4w16: linear-m512-n2048-k1024: ×0.625 < ×1.050
 *   - reg16x64r4x8w8: linear-m512-n2048-k1024: ×0.610 < ×1.050
 *   - reg32x128r4x8w16: bmm-b16-m512-n64-k512: ×0.635 < ×1.050
 *   - reg32x128r8x8w16: bmm-b16-m512-n64-k64: ×0.579 < ×1.050
 *   - reg32x16r2x4w4: bmm-b16-m512-n512-k64: ×0.584 < ×1.050
 *   - reg32x16r4x4w4: matmul-m512-n2048-k1024: ×0.597 < ×1.050
 *   - reg32x16r8x4w4: linear-m512-n2048-k1024: ×0.401 < ×1.050
 *   - reg32x32r2x4w8: linear-m512-n2048-k1024: ×0.832 < ×1.050
 *   - reg32x32r2x8w4: linear-m512-n2048-k1024: ×0.783 < ×1.050
 *   - reg32x32r4x4w8: linear-m512-n2048-k1024: ×0.822 < ×1.050
 *   - reg32x32r4x8w4: bmm-b16-m512-n64-k64: ×0.726 < ×1.050
 *   - reg32x32r8x4w8: linear-m512-n2048-k1024: ×0.765 < ×1.050
 *   - reg32x32r8x8w4: linear-m512-n2048-k1024: ×0.340 < ×1.050
 *   - reg32x64r2x8w8: linear-m512-n2048-k1024: ×0.807 < ×1.050
 *   - reg32x64r4x4w16: linear-m512-n2048-k1024: ×0.895 < ×1.050
 *   - reg32x64r4x8w8: bmm-b16-m512-n64-k64: ×0.822 < ×1.050
 *   - reg32x64r8x4w16: linear-m512-n2048-k1024: ×0.885 < ×1.050
 *   - reg32x64r8x8w8: bmm-b16-m512-n64-k64: ×0.683 < ×1.050
 *   - reg4x16r1x4w4: linear-m512-n2048-k1024: ×0.124 < ×1.050
 *   - reg4x32r1x8w4: linear-m512-n2048-k1024: ×0.128 < ×1.050
 *   - reg64x128r4x8w16: bmm-b16-m512-n64-k64: ×0.602 < ×1.050
 *   - reg64x128r8x8w16: bmm-b16-m512-n64-k64: ×0.592 < ×1.050
 *   - reg64x16r4x4w4: matmul-m512-n2048-k1024: ×0.665 < ×1.050
 *   - reg64x16r8x4w4: bmm-b16-m512-n64-k512: ×0.545 < ×1.050
 *   - reg64x32r4x8w4: bmm-b16-m512-n64-k64: ×0.723 < ×1.050
 *   - reg64x32r8x4w8: bmm-b16-m512-n64-k512: ×0.802 < ×1.050
 *   - reg64x32r8x8w4: bmm-b16-m512-n64-k512: ×0.533 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m512-n64-k512: ×1.028 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m512-n64-k64: ×0.816 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m512-n64-k64: ×0.965 < ×1.050
 *   - reg64x64r8x8w8: bmm-b16-m512-n64-k64: ×0.712 < ×1.050
 *   - reg8x16r1x4w4: linear-m512-n2048-k1024: ×0.328 < ×1.050
 *   - reg8x16r2x4w4: linear-m512-n2048-k1024: ×0.215 < ×1.050
 *   - reg8x32r1x8w4: linear-m512-n2048-k1024: ×0.368 < ×1.050
 *   - reg8x32r2x4w8: linear-m512-n2048-k1024: ×0.378 < ×1.050
 *   - reg8x32r2x8w4: linear-m512-n2048-k1024: ×0.233 < ×1.050
 *   - reg8x64r2x8w8: linear-m512-n2048-k1024: ×0.351 < ×1.050
 * - gemmRows > 512 (linear / matmul / bmm row count > 512, 7 cases): keeps the default reg128x128r8x8w16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x16r8x4w4: linear-m4096-n8192-k2048: ×0.492 < ×1.050
 *   - reg128x32r8x4w8: linear-m1024-n2048-k2048: ×0.789 < ×1.050
 *   - reg128x32r8x8w4: linear-m1024-n2048-k2048: ×0.579 < ×1.050
 *   - reg128x64r8x4w16: linear-m1024-n2048-k8192: ×0.941 < ×1.050
 *   - reg128x64r8x8w8: linear-m1024-n2048-k8192: ×0.852 < ×1.050
 *   - reg16x128r4x8w16: linear-m1024-n2048-k2048: ×0.483 < ×1.050
 *   - reg16x16r1x4w4: linear-m4096-n8192-k2048: ×0.343 < ×1.050
 *   - reg16x16r2x4w4: linear-m4096-n8192-k2048: ×0.341 < ×1.050
 *   - reg16x16r4x4w4: linear-m4096-n8192-k2048: ×0.247 < ×1.050
 *   - reg16x32r1x8w4: linear-m4096-n8192-k2048: ×0.405 < ×1.050
 *   - reg16x32r2x4w8: linear-m4096-n8192-k2048: ×0.421 < ×1.050
 *   - reg16x32r2x8w4: linear-m4096-n8192-k2048: ×0.425 < ×1.050
 *   - reg16x32r4x4w8: linear-m4096-n8192-k2048: ×0.444 < ×1.050
 *   - reg16x32r4x8w4: linear-m4096-n8192-k2048: ×0.261 < ×1.050
 *   - reg16x64r2x8w8: linear-m4096-n8192-k2048: ×0.432 < ×1.050
 *   - reg16x64r4x4w16: linear-m4096-n8192-k2048: ×0.481 < ×1.050
 *   - reg16x64r4x8w8: linear-m1024-n2048-k2048: ×0.444 < ×1.050
 *   - reg32x128r4x8w16: linear-m1024-n2048-k2048: ×0.725 < ×1.050
 *   - reg32x128r8x8w16: linear-m1024-n2048-k2048: ×0.709 < ×1.050
 *   - reg32x16r2x4w4: linear-m4096-n8192-k2048: ×0.450 < ×1.050
 *   - reg32x16r4x4w4: linear-m4096-n8192-k2048: ×0.474 < ×1.050
 *   - reg32x16r8x4w4: linear-m1024-n2048-k8192: ×0.288 < ×1.050
 *   - reg32x32r2x4w8: linear-m4096-n8192-k2048: ×0.598 < ×1.050
 *   - reg32x32r2x8w4: linear-m4096-n8192-k2048: ×0.609 < ×1.050
 *   - reg32x32r4x4w8: linear-m4096-n8192-k2048: ×0.640 < ×1.050
 *   - reg32x32r4x8w4: linear-m1024-n2048-k2048: ×0.556 < ×1.050
 *   - reg32x32r8x4w8: linear-m1024-n2048-k2048: ×0.564 < ×1.050
 *   - reg32x32r8x8w4: linear-m1024-n2048-k8192: ×0.286 < ×1.050
 *   - reg32x64r2x8w8: linear-m4096-n8192-k2048: ×0.621 < ×1.050
 *   - reg32x64r4x4w16: linear-m4096-n8192-k2048: ×0.700 < ×1.050
 *   - reg32x64r4x8w8: linear-m1024-n2048-k2048: ×0.653 < ×1.050
 *   - reg32x64r8x4w16: linear-m1024-n2048-k2048: ×0.691 < ×1.050
 *   - reg32x64r8x8w8: linear-m4096-n8192-k2048: ×0.613 < ×1.050
 *   - reg4x16r1x4w4: linear-m4096-n8192-k2048: ×0.085 < ×1.050
 *   - reg4x32r1x8w4: linear-m4096-n8192-k2048: ×0.089 < ×1.050
 *   - reg64x128r4x8w16: linear-m1024-n2048-k2048: ×0.868 < ×1.050
 *   - reg64x128r8x8w16: linear-m1024-n2048-k2048: ×0.920 < ×1.050
 *   - reg64x16r4x4w4: linear-m4096-n8192-k2048: ×0.574 < ×1.050
 *   - reg64x16r8x4w4: linear-m1024-n2048-k2048: ×0.451 < ×1.050
 *   - reg64x32r4x4w8: linear-m4096-n8192-k2048: ×0.779 < ×1.050
 *   - reg64x32r4x8w4: linear-m1024-n2048-k2048: ×0.687 < ×1.050
 *   - reg64x32r8x4w8: linear-m1024-n2048-k2048: ×0.703 < ×1.050
 *   - reg64x32r8x8w4: linear-m4096-n8192-k2048: ×0.527 < ×1.050
 *   - reg64x64r4x4w16: linear-m1024-n2048-k2048: ×0.812 < ×1.050
 *   - reg64x64r4x8w8: linear-m1024-n2048-k2048: ×0.820 < ×1.050
 *   - reg64x64r8x4w16: linear-m1024-n2048-k2048: ×0.884 < ×1.050
 *   - reg64x64r8x8w8: linear-m1024-n2048-k2048: ×0.770 < ×1.050
 *   - reg8x16r1x4w4: linear-m4096-n8192-k2048: ×0.229 < ×1.050
 *   - reg8x16r2x4w4: linear-m4096-n8192-k2048: ×0.151 < ×1.050
 *   - reg8x32r1x8w4: linear-m4096-n8192-k2048: ×0.259 < ×1.050
 *   - reg8x32r2x4w8: linear-m4096-n8192-k2048: ×0.266 < ×1.050
 *   - reg8x32r2x8w4: linear-m4096-n8192-k2048: ×0.166 < ×1.050
 *   - reg8x64r2x8w8: linear-m4096-n8192-k2048: ×0.267 < ×1.050
 * - attention.qk (fused attention f32 ①QK, 4 cases): keeps the default reg128x128r8x8w16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x16r8x4w4: attention-qk-self-m4096-n4096: ×0.508 < ×1.050
 *   - reg128x32r8x4w8: attention-qk-self-m4096-n4096: ×0.812 < ×1.050
 *   - reg128x32r8x8w4: attention-qk-self-m4096-n4096: ×0.601 < ×1.050
 *   - reg128x64r8x4w16: attention-qk-self-m4096-n4096: ×0.981 < ×1.050
 *   - reg128x64r8x8w8: attention-qk-cross-m4096-n512: ×0.888 < ×1.050
 *   - reg16x128r4x8w16: attention-qk-self-m4096-n4096: ×0.455 < ×1.050
 *   - reg16x16r1x4w4: attention-qk-self-m4096-n4096: ×0.327 < ×1.050
 *   - reg16x16r2x4w4: attention-qk-self-m4096-n4096: ×0.328 < ×1.050
 *   - reg16x16r4x4w4: attention-qk-self-m4096-n4096: ×0.241 < ×1.050
 *   - reg16x32r1x8w4: attention-qk-self-m4096-n4096: ×0.380 < ×1.050
 *   - reg16x32r2x4w8: attention-qk-self-m4096-n4096: ×0.396 < ×1.050
 *   - reg16x32r2x8w4: attention-qk-self-m4096-n4096: ×0.398 < ×1.050
 *   - reg16x32r4x4w8: attention-qk-self-m4096-n4096: ×0.413 < ×1.050
 *   - reg16x32r4x8w4: attention-qk-self-m4096-n4096: ×0.253 < ×1.050
 *   - reg16x64r2x8w8: attention-qk-self-m4096-n4096: ×0.406 < ×1.050
 *   - reg16x64r4x4w16: attention-qk-self-m4096-n4096: ×0.445 < ×1.050
 *   - reg16x64r4x8w8: attention-qk-self-m4096-n4096: ×0.421 < ×1.050
 *   - reg32x128r4x8w16: attention-qk-self-m4096-n4096: ×0.693 < ×1.050
 *   - reg32x128r8x8w16: attention-qk-self-m1024-n1024: ×0.648 < ×1.050
 *   - reg32x16r2x4w4: attention-qk-self-m4096-n4096: ×0.444 < ×1.050
 *   - reg32x16r4x4w4: attention-qk-self-m4096-n4096: ×0.467 < ×1.050
 *   - reg32x16r8x4w4: attention-qk-cross-m4096-n512: ×0.281 < ×1.050
 *   - reg32x32r2x4w8: attention-qk-self-m4096-n4096: ×0.580 < ×1.050
 *   - reg32x32r2x8w4: attention-qk-self-m4096-n4096: ×0.583 < ×1.050
 *   - reg32x32r4x4w8: attention-qk-self-m4096-n4096: ×0.616 < ×1.050
 *   - reg32x32r4x8w4: attention-qk-self-m4096-n4096: ×0.537 < ×1.050
 *   - reg32x32r8x4w8: attention-qk-self-m4096-n4096: ×0.558 < ×1.050
 *   - reg32x32r8x8w4: attention-qk-self-m1024-n1024: ×0.304 < ×1.050
 *   - reg32x64r2x8w8: attention-qk-self-m4096-n4096: ×0.595 < ×1.050
 *   - reg32x64r4x4w16: attention-qk-self-m4096-n4096: ×0.670 < ×1.050
 *   - reg32x64r4x8w8: attention-qk-self-m4096-n4096: ×0.631 < ×1.050
 *   - reg32x64r8x4w16: attention-qk-self-m4096-n4096: ×0.680 < ×1.050
 *   - reg32x64r8x8w8: attention-qk-self-m1024-n1024: ×0.563 < ×1.050
 *   - reg4x16r1x4w4: attention-qk-self-m4096-n4096: ×0.082 < ×1.050
 *   - reg4x32r1x8w4: attention-qk-self-m4096-n4096: ×0.086 < ×1.050
 *   - reg64x128r4x8w16: attention-qk-self-m4096-n4096: ×0.882 < ×1.050
 *   - reg64x128r8x8w16: attention-qk-self-m4096-n4096: ×0.906 < ×1.050
 *   - reg64x16r4x4w4: attention-qk-self-m4096-n4096: ×0.576 < ×1.050
 *   - reg64x16r8x4w4: attention-qk-self-m4096-n4096: ×0.462 < ×1.050
 *   - reg64x32r4x4w8: attention-qk-self-m4096-n4096: ×0.776 < ×1.050
 *   - reg64x32r4x8w4: attention-qk-self-m4096-n4096: ×0.686 < ×1.050
 *   - reg64x32r8x4w8: attention-qk-self-m4096-n4096: ×0.716 < ×1.050
 *   - reg64x32r8x8w4: attention-qk-cross-m4096-n512: ×0.524 < ×1.050
 *   - reg64x64r4x4w16: attention-qk-self-m4096-n4096: ×0.811 < ×1.050
 *   - reg64x64r4x8w8: attention-qk-self-m4096-n4096: ×0.813 < ×1.050
 *   - reg64x64r8x4w16: attention-qk-self-m4096-n4096: ×0.884 < ×1.050
 *   - reg64x64r8x8w8: attention-qk-self-m4096-n4096: ×0.779 < ×1.050
 *   - reg8x16r1x4w4: attention-qk-self-m4096-n4096: ×0.214 < ×1.050
 *   - reg8x16r2x4w4: attention-qk-self-m4096-n4096: ×0.147 < ×1.050
 *   - reg8x32r1x8w4: attention-qk-self-m4096-n4096: ×0.240 < ×1.050
 *   - reg8x32r2x4w8: attention-qk-self-m4096-n4096: ×0.245 < ×1.050
 *   - reg8x32r2x8w4: attention-qk-self-m4096-n4096: ×0.158 < ×1.050
 *   - reg8x64r2x8w8: attention-qk-self-m4096-n4096: ×0.247 < ×1.050
 * - attention.pv (fused attention f32 ③PV, 4 cases): keeps the default reg128x128r8x8w16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x16r8x4w4: attention-pv-cross-m1024-n512: ×0.484 < ×1.050
 *   - reg128x32r8x4w8: attention-pv-self-m1024-n1024: ×0.784 < ×1.050
 *   - reg128x32r8x8w4: attention-pv-self-m1024-n1024: ×0.469 < ×1.050
 *   - reg128x64r8x4w16: attention-pv-self-m1024-n1024: ×0.913 < ×1.050
 *   - reg128x64r8x8w8: attention-pv-cross-m1024-n512: ×0.897 < ×1.050
 *   - reg16x128r4x8w16: attention-pv-self-m1024-n1024: ×0.824 < ×1.050
 *   - reg16x16r1x4w4: attention-pv-cross-m1024-n512: ×0.474 < ×1.050
 *   - reg16x16r2x4w4: attention-pv-cross-m1024-n512: ×0.406 < ×1.050
 *   - reg16x16r4x4w4: attention-pv-self-m1024-n1024: ×0.397 < ×1.050
 *   - reg16x32r1x8w4: attention-pv-cross-m1024-n512: ×0.657 < ×1.050
 *   - reg16x32r2x4w8: attention-pv-cross-m1024-n512: ×0.664 < ×1.050
 *   - reg16x32r2x8w4: attention-pv-cross-m1024-n512: ×0.718 < ×1.050
 *   - reg16x32r4x4w8: attention-pv-self-m1024-n1024: ×0.735 < ×1.050
 *   - reg16x32r4x8w4: attention-pv-self-m1024-n1024: ×0.446 < ×1.050
 *   - reg16x64r2x8w8: attention-pv-cross-m1024-n512: ×0.728 < ×1.050
 *   - reg16x64r4x4w16: attention-pv-self-m1024-n1024: ×0.833 < ×1.050
 *   - reg16x64r4x8w8: attention-pv-cross-m1024-n512: ×0.755 < ×1.050
 *   - reg32x128r4x8w16: attention-pv-cross-m1024-n512: ×0.934 < ×1.050
 *   - reg32x128r8x8w16: attention-pv-self-m1024-n1024: ×0.710 < ×1.050
 *   - reg32x16r2x4w4: attention-pv-cross-m1024-n512: ×0.529 < ×1.050
 *   - reg32x16r4x4w4: attention-pv-cross-m1024-n512: ×0.591 < ×1.050
 *   - reg32x16r8x4w4: attention-pv-self-m1024-n1024: ×0.326 < ×1.050
 *   - reg32x32r2x4w8: attention-pv-self-m1024-n1024: ×0.790 < ×1.050
 *   - reg32x32r2x8w4: attention-pv-self-m1024-n1024: ×0.798 < ×1.050
 *   - reg32x32r4x4w8: attention-pv-self-m1024-n1024: ×0.837 < ×1.050
 *   - reg32x32r4x8w4: attention-pv-cross-m1024-n512: ×0.718 < ×1.050
 *   - reg32x32r8x4w8: attention-pv-cross-m1024-n512: ×0.729 < ×1.050
 *   - reg32x32r8x8w4: attention-pv-self-m1024-n1024: ×0.370 < ×1.050
 *   - reg32x64r2x8w8: attention-pv-cross-m1024-n512: ×0.854 < ×1.050
 *   - reg32x64r4x4w16: attention-pv-cross-m1024-n512: ×0.955 < ×1.050
 *   - reg32x64r4x8w8: attention-pv-cross-m1024-n512: ×0.850 < ×1.050
 *   - reg32x64r8x4w16: attention-pv-self-m1024-n1024: ×0.898 < ×1.050
 *   - reg32x64r8x8w8: attention-pv-self-m1024-n1024: ×0.638 < ×1.050
 *   - reg4x16r1x4w4: attention-pv-cross-m1024-n512: ×0.221 < ×1.050
 *   - reg4x32r1x8w4: attention-pv-self-m4096-n4096: ×0.295 < ×1.050
 *   - reg64x128r4x8w16: attention-pv-self-m1024-n1024: ×0.970 < ×1.050
 *   - reg64x128r8x8w16: attention-pv-cross-m1024-n512: ×1.029 < ×1.050
 *   - reg64x16r4x4w4: attention-pv-cross-m4096-n512: ×0.619 < ×1.050
 *   - reg64x16r8x4w4: attention-pv-cross-m1024-n512: ×0.468 < ×1.050
 *   - reg64x32r4x4w8: attention-pv-self-m1024-n1024: ×0.880 < ×1.050
 *   - reg64x32r4x8w4: attention-pv-self-m1024-n1024: ×0.754 < ×1.050
 *   - reg64x32r8x4w8: attention-pv-self-m1024-n1024: ×0.779 < ×1.050
 *   - reg64x32r8x8w4: attention-pv-self-m1024-n1024: ×0.471 < ×1.050
 *   - reg64x64r4x4w16: attention-pv-self-m1024-n1024: ×0.917 < ×1.050
 *   - reg64x64r4x8w8: attention-pv-self-m1024-n1024: ×0.866 < ×1.050
 *   - reg64x64r8x4w16: attention-pv-self-m1024-n1024: ×0.932 < ×1.050
 *   - reg64x64r8x8w8: attention-pv-cross-m1024-n512: ×0.895 < ×1.050
 *   - reg8x16r1x4w4: attention-pv-cross-m1024-n512: ×0.361 < ×1.050
 *   - reg8x16r2x4w4: attention-pv-cross-m1024-n512: ×0.287 < ×1.050
 *   - reg8x32r1x8w4: attention-pv-cross-m1024-n512: ×0.540 < ×1.050
 *   - reg8x32r2x4w8: attention-pv-cross-m1024-n512: ×0.534 < ×1.050
 *   - reg8x32r2x8w4: attention-pv-cross-m1024-n512: ×0.419 < ×1.050
 *   - reg8x64r2x8w8: attention-pv-cross-m1024-n512: ×0.556 < ×1.050
 * - conv2d.rows64 (conv2d implicit GEMM m tile of 64 rows, 2 cases): keeps the default igemm64x128:wg16x8 (no geometry matches the output in every case at ×1.050 or faster)
 *   - igemm128x128:wg16x16: conv2d-c192-256x256: ×0.850 < ×1.050
 *   - igemm128x64:wg16x16: conv2d-c192-256x256: ×0.858 < ×1.050
 *   - igemm128x64:wg8x16: conv2d-c192-256x256: ×0.797 < ×1.050
 *   - igemm16x128:wg16x4: conv2d-c384-128x128: ×0.553 < ×1.050
 *   - igemm16x64:wg16x4: conv2d-c384-128x128: ×0.584 < ×1.050
 *   - igemm16x64:wg8x4: conv2d-c384-128x128: ×0.535 < ×1.050
 *   - igemm16x64:wg8x8: conv2d-c384-128x128: ×0.607 < ×1.050
 *   - igemm32x128:wg16x4: conv2d-c384-128x128: ×0.767 < ×1.050
 *   - igemm32x128:wg16x8: conv2d-c384-128x128: ×0.805 < ×1.050
 *   - igemm32x64:wg16x4: conv2d-c192-256x256: ×0.783 < ×1.050
 *   - igemm32x64:wg16x8: conv2d-c384-128x128: ×0.838 < ×1.050
 *   - igemm32x64:wg8x16: conv2d-c384-128x128: ×0.800 < ×1.050
 *   - igemm32x64:wg8x4: conv2d-c192-256x256: ×0.711 < ×1.050
 *   - igemm32x64:wg8x8: conv2d-c192-256x256: ×0.786 < ×1.050
 *   - igemm64x128:wg16x16: conv2d-c192-256x256: ×1.048 < ×1.050
 *   - igemm64x64:wg16x16: conv2d-c192-256x256: ×0.952 < ×1.050
 *   - igemm64x64:wg16x8: conv2d-c192-256x256: ×1.038 < ×1.050
 *   - igemm64x64:wg8x16: conv2d-c192-256x256: ×0.980 < ×1.050
 *   - igemm64x64:wg8x8: conv2d-c192-256x256: ×0.918 < ×1.050
 *   - igemm8x64:wg8x4: conv2d-c384-128x128: ×0.348 < ×1.050
 * - conv2d.rows32 (conv2d implicit GEMM m tile of 32 rows, 1 cases): adopted igemm32x64:wg16x8 ×1.064 (×1.064–×1.064)
 *   - igemm128x128:wg16x16: conv2d-c96-512x512: ×1.039 < ×1.050
 *   - igemm128x64:wg16x16: geometric mean ×1.055 (below the adopted ×1.064)
 *   - igemm128x64:wg8x16: conv2d-c96-512x512: ×0.984 < ×1.050
 *   - igemm16x128:wg16x4: conv2d-c96-512x512: ×0.743 < ×1.050
 *   - igemm16x64:wg16x4: conv2d-c96-512x512: ×0.750 < ×1.050
 *   - igemm16x64:wg8x4: conv2d-c96-512x512: ×0.700 < ×1.050
 *   - igemm16x64:wg8x8: conv2d-c96-512x512: ×0.773 < ×1.050
 *   - igemm32x128:wg16x8: conv2d-c96-512x512: ×1.040 < ×1.050
 *   - igemm32x64:wg16x4: conv2d-c96-512x512: ×0.992 < ×1.050
 *   - igemm32x64:wg8x16: conv2d-c96-512x512: ×0.996 < ×1.050
 *   - igemm32x64:wg8x4: conv2d-c96-512x512: ×0.918 < ×1.050
 *   - igemm32x64:wg8x8: conv2d-c96-512x512: ×0.991 < ×1.050
 *   - igemm64x128:wg16x16: conv2d-c96-512x512: ×0.973 < ×1.050
 *   - igemm64x128:wg16x8: conv2d-c96-512x512: ×0.926 < ×1.050
 *   - igemm64x64:wg16x16: conv2d-c96-512x512: ×0.907 < ×1.050
 *   - igemm64x64:wg16x8: conv2d-c96-512x512: ×0.990 < ×1.050
 *   - igemm64x64:wg8x16: conv2d-c96-512x512: ×0.938 < ×1.050
 *   - igemm64x64:wg8x8: conv2d-c96-512x512: ×0.904 < ×1.050
 *   - igemm8x64:wg8x4: conv2d-c96-512x512: ×0.474 < ×1.050
 * - i8a8.linear ≤ 256 (i8a8 linear row count ≤ 256, 3 cases): adopted tile32x64r4x4w16x8k32 ×2.326 (×1.913–×2.794)
 *   - tile128x128r16x8w16x8k16: i8a8-linear-m185-n1280-k1280: ×0.678 < ×1.050
 *   - tile128x128r16x8w16x8k16s: i8a8-linear-m185-n1280-k1280: ×0.599 < ×1.050
 *   - tile128x128r16x8w16x8k32: i8a8-linear-m185-n1280-k1280: ×0.747 < ×1.050
 *   - tile128x128r16x8w16x8k32s: i8a8-linear-m256-n1280-k1280: ×0.440 < ×1.050
 *   - tile128x128r16x8w16x8k64: i8a8-linear-m185-n1280-k1280: ×0.710 < ×1.050
 *   - tile128x128r16x8w16x8k64s: i8a8-linear-m185-n1280-k1280: ×0.525 < ×1.050
 *   - tile128x128r8x8w16x16k16: i8a8-linear-m185-n1280-k1280: ×0.662 < ×1.050
 *   - tile128x128r8x8w16x16k16s: i8a8-linear-m185-n1280-k1280: ×0.600 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-linear-m1-n18432-k3072: ×0.733 < ×1.050
 *   - tile128x128r8x8w16x16k32s: i8a8-linear-m185-n1280-k1280: ×0.814 < ×1.050
 *   - tile128x128r8x8w16x16k64: i8a8-linear-m185-n1280-k1280: ×0.737 < ×1.050
 *   - tile128x128r8x8w16x16k64s: i8a8-linear-m185-n1280-k1280: ×0.893 < ×1.050
 *   - tile128x32r16x4w8x8k16: i8a8-linear-m1-n18432-k3072: ×0.936 < ×1.050
 *   - tile128x32r16x4w8x8k16s: i8a8-linear-m256-n1280-k1280: ×0.602 < ×1.050
 *   - tile128x32r16x4w8x8k32: i8a8-linear-m1-n18432-k3072: ×0.862 < ×1.050
 *   - tile128x32r16x4w8x8k32s: i8a8-linear-m1-n18432-k3072: ×0.451 < ×1.050
 *   - tile128x32r16x4w8x8k64: i8a8-linear-m1-n18432-k3072: ×0.692 < ×1.050
 *   - tile128x32r16x4w8x8k64s: i8a8-linear-m1-n18432-k3072: ×0.949 < ×1.050
 *   - tile128x32r8x4w8x16k16: geometric mean ×1.259 (below the adopted ×2.326)
 *   - tile128x32r8x4w8x16k16s: i8a8-linear-m256-n1280-k1280: ×0.769 < ×1.050
 *   - tile128x32r8x4w8x16k32: geometric mean ×1.300 (below the adopted ×2.326)
 *   - tile128x32r8x4w8x16k32s: geometric mean ×1.154 (below the adopted ×2.326)
 *   - tile128x32r8x4w8x16k64: i8a8-linear-m1-n18432-k3072: ×0.774 < ×1.050
 *   - tile128x32r8x4w8x16k64s: geometric mean ×1.632 (below the adopted ×2.326)
 *   - tile128x64r16x4w16x8k16: i8a8-linear-m1-n18432-k3072: ×1.000 < ×1.050
 *   - tile128x64r16x4w16x8k16s: i8a8-linear-m185-n1280-k1280: ×0.700 < ×1.050
 *   - tile128x64r16x4w16x8k32: i8a8-linear-m185-n1280-k1280: ×0.847 < ×1.050
 *   - tile128x64r16x4w16x8k32s: i8a8-linear-m1-n18432-k3072: ×0.558 < ×1.050
 *   - tile128x64r16x4w16x8k64: i8a8-linear-m1-n18432-k3072: ×0.815 < ×1.050
 *   - tile128x64r16x4w16x8k64s: geometric mean ×1.417 (below the adopted ×2.326)
 *   - tile128x64r16x8w8x8k16: i8a8-linear-m185-n1280-k1280: ×0.741 < ×1.050
 *   - tile128x64r16x8w8x8k16s: i8a8-linear-m185-n1280-k1280: ×0.570 < ×1.050
 *   - tile128x64r16x8w8x8k32: i8a8-linear-m185-n1280-k1280: ×0.786 < ×1.050
 *   - tile128x64r16x8w8x8k32s: i8a8-linear-m185-n1280-k1280: ×0.631 < ×1.050
 *   - tile128x64r16x8w8x8k64: i8a8-linear-m1-n18432-k3072: ×0.612 < ×1.050
 *   - tile128x64r16x8w8x8k64s: i8a8-linear-m185-n1280-k1280: ×0.968 < ×1.050
 *   - tile128x64r8x4w16x16k16: geometric mean ×1.260 (below the adopted ×2.326)
 *   - tile128x64r8x4w16x16k16s: i8a8-linear-m185-n1280-k1280: ×0.842 < ×1.050
 *   - tile128x64r8x4w16x16k32: i8a8-linear-m1-n18432-k3072: ×0.966 < ×1.050
 *   - tile128x64r8x4w16x16k32s: i8a8-linear-m1-n18432-k3072: ×1.022 < ×1.050
 *   - tile128x64r8x4w16x16k64: i8a8-linear-m1-n18432-k3072: ×0.893 < ×1.050
 *   - tile128x64r8x4w16x16k64s: geometric mean ×1.405 (below the adopted ×2.326)
 *   - tile128x64r8x8w8x16k16s: i8a8-linear-m185-n1280-k1280: ×0.848 < ×1.050
 *   - tile128x64r8x8w8x16k32: i8a8-linear-m1-n18432-k3072: ×0.899 < ×1.050
 *   - tile128x64r8x8w8x16k32s: geometric mean ×1.139 (below the adopted ×2.326)
 *   - tile128x64r8x8w8x16k64: i8a8-linear-m1-n18432-k3072: ×0.777 < ×1.050
 *   - tile128x64r8x8w8x16k64s: geometric mean ×1.378 (below the adopted ×2.326)
 *   - tile16x128r4x8w16x4k16: geometric mean ×1.696 (below the adopted ×2.326)
 *   - tile16x128r4x8w16x4k32: geometric mean ×1.727 (below the adopted ×2.326)
 *   - tile16x32r4x4w8x4k16: geometric mean ×2.170 (below the adopted ×2.326)
 *   - tile16x32r4x4w8x4k32: geometric mean ×2.169 (below the adopted ×2.326)
 *   - tile16x64r4x4w16x4k16: geometric mean ×1.765 (below the adopted ×2.326)
 *   - tile16x64r4x4w16x4k32: geometric mean ×1.842 (below the adopted ×2.326)
 *   - tile16x64r4x8w8x4k16: geometric mean ×1.806 (below the adopted ×2.326)
 *   - tile16x64r4x8w8x4k32: geometric mean ×1.735 (below the adopted ×2.326)
 *   - tile192x128r12x8w16x16k16: i8a8-linear-m256-n1280-k1280: ×0.525 < ×1.050
 *   - tile192x128r12x8w16x16k16s: i8a8-linear-m185-n1280-k1280: ×0.463 < ×1.050
 *   - tile192x128r12x8w16x16k32: i8a8-linear-m185-n1280-k1280: ×0.572 < ×1.050
 *   - tile192x128r12x8w16x16k32s: i8a8-linear-m1-n18432-k3072: ×0.543 < ×1.050
 *   - tile192x32r12x4w8x16k16: i8a8-linear-m1-n18432-k3072: ×0.796 < ×1.050
 *   - tile192x32r12x4w8x16k16s: i8a8-linear-m256-n1280-k1280: ×0.756 < ×1.050
 *   - tile192x32r12x4w8x16k32: i8a8-linear-m1-n18432-k3072: ×0.732 < ×1.050
 *   - tile192x32r12x4w8x16k32s: i8a8-linear-m1-n18432-k3072: ×0.350 < ×1.050
 *   - tile192x32r12x4w8x16k64: i8a8-linear-m1-n18432-k3072: ×0.519 < ×1.050
 *   - tile192x32r12x4w8x16k64s: i8a8-linear-m1-n18432-k3072: ×0.820 < ×1.050
 *   - tile192x64r12x4w16x16k16: i8a8-linear-m1-n18432-k3072: ×0.704 < ×1.050
 *   - tile192x64r12x4w16x16k16s: i8a8-linear-m256-n1280-k1280: ×0.648 < ×1.050
 *   - tile192x64r12x4w16x16k32: i8a8-linear-m1-n18432-k3072: ×0.680 < ×1.050
 *   - tile192x64r12x4w16x16k32s: i8a8-linear-m1-n18432-k3072: ×0.389 < ×1.050
 *   - tile192x64r12x4w16x16k64: i8a8-linear-m1-n18432-k3072: ×0.591 < ×1.050
 *   - tile192x64r12x4w16x16k64s: i8a8-linear-m1-n18432-k3072: ×0.240 < ×1.050
 *   - tile192x64r12x8w8x16k16: i8a8-linear-m1-n18432-k3072: ×0.664 < ×1.050
 *   - tile192x64r12x8w8x16k16s: i8a8-linear-m185-n1280-k1280: ×0.607 < ×1.050
 *   - tile192x64r12x8w8x16k32: i8a8-linear-m1-n18432-k3072: ×0.646 < ×1.050
 *   - tile192x64r12x8w8x16k32s: i8a8-linear-m1-n18432-k3072: ×0.772 < ×1.050
 *   - tile192x64r12x8w8x16k64: i8a8-linear-m1-n18432-k3072: ×0.485 < ×1.050
 *   - tile192x64r12x8w8x16k64s: i8a8-linear-m1-n18432-k3072: ×0.789 < ×1.050
 *   - tile256x128r16x8w16x16k16: i8a8-linear-m1-n18432-k3072: ×0.389 < ×1.050
 *   - tile256x128r16x8w16x16k16s: i8a8-linear-m256-n1280-k1280: ×0.376 < ×1.050
 *   - tile256x128r16x8w16x16k32: i8a8-linear-m1-n18432-k3072: ×0.399 < ×1.050
 *   - tile256x128r16x8w16x16k32s: i8a8-linear-m1-n18432-k3072: ×0.445 < ×1.050
 *   - tile256x32r16x4w8x16k16: i8a8-linear-m1-n18432-k3072: ×0.535 < ×1.050
 *   - tile256x32r16x4w8x16k16s: i8a8-linear-m256-n1280-k1280: ×0.577 < ×1.050
 *   - tile256x32r16x4w8x16k32: i8a8-linear-m1-n18432-k3072: ×0.504 < ×1.050
 *   - tile256x32r16x4w8x16k32s: i8a8-linear-m1-n18432-k3072: ×0.643 < ×1.050
 *   - tile256x64r16x4w16x16k16: i8a8-linear-m1-n18432-k3072: ×0.482 < ×1.050
 *   - tile256x64r16x4w16x16k16s: i8a8-linear-m1-n18432-k3072: ×0.530 < ×1.050
 *   - tile256x64r16x4w16x16k32: i8a8-linear-m1-n18432-k3072: ×0.473 < ×1.050
 *   - tile256x64r16x4w16x16k32s: i8a8-linear-m1-n18432-k3072: ×0.575 < ×1.050
 *   - tile256x64r16x8w8x16k16: i8a8-linear-m1-n18432-k3072: ×0.503 < ×1.050
 *   - tile256x64r16x8w8x16k16s: i8a8-linear-m256-n1280-k1280: ×0.469 < ×1.050
 *   - tile256x64r16x8w8x16k32: i8a8-linear-m1-n18432-k3072: ×0.505 < ×1.050
 *   - tile256x64r16x8w8x16k32s: i8a8-linear-m1-n18432-k3072: ×0.295 < ×1.050
 *   - tile32x128r4x8w16x8k16: geometric mean ×1.755 (below the adopted ×2.326)
 *   - tile32x128r4x8w16x8k32: geometric mean ×1.824 (below the adopted ×2.326)
 *   - tile32x128r8x8w16x4k16: geometric mean ×1.293 (below the adopted ×2.326)
 *   - tile32x128r8x8w16x4k16s: i8a8-linear-m256-n1280-k1280: ×0.744 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-linear-m256-n1280-k1280: ×0.500 < ×1.050
 *   - tile32x128r8x8w16x4k32s: i8a8-linear-m256-n1280-k1280: ×0.838 < ×1.050
 *   - tile32x128r8x8w16x4k64: i8a8-linear-m256-n1280-k1280: ×0.961 < ×1.050
 *   - tile32x128r8x8w16x4k64s: geometric mean ×1.560 (below the adopted ×2.326)
 *   - tile32x32r4x4w8x8k16: geometric mean ×1.955 (below the adopted ×2.326)
 *   - tile32x32r4x4w8x8k32: geometric mean ×1.403 (below the adopted ×2.326)
 *   - tile32x32r8x4w8x4k16: geometric mean ×1.967 (below the adopted ×2.326)
 *   - tile32x32r8x4w8x4k16s: i8a8-linear-m256-n1280-k1280: ×0.852 < ×1.050
 *   - tile32x32r8x4w8x4k32: geometric mean ×1.801 (below the adopted ×2.326)
 *   - tile32x32r8x4w8x4k32s: geometric mean ×1.819 (below the adopted ×2.326)
 *   - tile32x32r8x4w8x4k64: geometric mean ×1.421 (below the adopted ×2.326)
 *   - tile32x32r8x4w8x4k64s: geometric mean ×1.942 (below the adopted ×2.326)
 *   - tile32x64r4x4w16x8k16: geometric mean ×2.045 (below the adopted ×2.326)
 *   - tile32x64r4x8w8x8k16: geometric mean ×1.870 (below the adopted ×2.326)
 *   - tile32x64r4x8w8x8k32: geometric mean ×1.907 (below the adopted ×2.326)
 *   - tile32x64r8x4w16x4k16: geometric mean ×1.855 (below the adopted ×2.326)
 *   - tile32x64r8x4w16x4k16s: i8a8-linear-m256-n1280-k1280: ×0.906 < ×1.050
 *   - tile32x64r8x4w16x4k32: geometric mean ×1.929 (below the adopted ×2.326)
 *   - tile32x64r8x4w16x4k32s: geometric mean ×1.715 (below the adopted ×2.326)
 *   - tile32x64r8x4w16x4k64: geometric mean ×1.613 (below the adopted ×2.326)
 *   - tile32x64r8x4w16x4k64s: geometric mean ×2.304 (below the adopted ×2.326)
 *   - tile32x64r8x8w8x4k16: geometric mean ×1.309 (below the adopted ×2.326)
 *   - tile32x64r8x8w8x4k16s: i8a8-linear-m256-n1280-k1280: ×0.793 < ×1.050
 *   - tile32x64r8x8w8x4k32: geometric mean ×1.295 (below the adopted ×2.326)
 *   - tile32x64r8x8w8x4k32s: geometric mean ×1.469 (below the adopted ×2.326)
 *   - tile32x64r8x8w8x4k64: i8a8-linear-m256-n1280-k1280: ×0.965 < ×1.050
 *   - tile32x64r8x8w8x4k64s: geometric mean ×1.633 (below the adopted ×2.326)
 *   - tile48x128r12x8w16x4k16: i8a8-linear-m185-n1280-k1280: ×0.885 < ×1.050
 *   - tile48x128r12x8w16x4k16s: i8a8-linear-m185-n1280-k1280: ×0.701 < ×1.050
 *   - tile48x128r12x8w16x4k32: i8a8-linear-m185-n1280-k1280: ×0.866 < ×1.050
 *   - tile48x128r12x8w16x4k32s: i8a8-linear-m185-n1280-k1280: ×0.762 < ×1.050
 *   - tile48x128r12x8w16x4k64: i8a8-linear-m185-n1280-k1280: ×0.948 < ×1.050
 *   - tile48x128r12x8w16x4k64s: geometric mean ×1.301 (below the adopted ×2.326)
 *   - tile48x32r12x4w8x4k16: geometric mean ×1.332 (below the adopted ×2.326)
 *   - tile48x32r12x4w8x4k16s: i8a8-linear-m256-n1280-k1280: ×0.808 < ×1.050
 *   - tile48x32r12x4w8x4k32: geometric mean ×1.372 (below the adopted ×2.326)
 *   - tile48x32r12x4w8x4k32s: geometric mean ×1.398 (below the adopted ×2.326)
 *   - tile48x32r12x4w8x4k64: geometric mean ×1.490 (below the adopted ×2.326)
 *   - tile48x32r12x4w8x4k64s: geometric mean ×1.752 (below the adopted ×2.326)
 *   - tile48x64r12x4w16x4k16: geometric mean ×1.637 (below the adopted ×2.326)
 *   - tile48x64r12x4w16x4k16s: i8a8-linear-m185-n1280-k1280: ×0.902 < ×1.050
 *   - tile48x64r12x4w16x4k32: geometric mean ×1.703 (below the adopted ×2.326)
 *   - tile48x64r12x4w16x4k32s: geometric mean ×1.455 (below the adopted ×2.326)
 *   - tile48x64r12x4w16x4k64: geometric mean ×1.569 (below the adopted ×2.326)
 *   - tile48x64r12x4w16x4k64s: geometric mean ×1.932 (below the adopted ×2.326)
 *   - tile48x64r12x8w8x4k16: i8a8-linear-m185-n1280-k1280: ×0.881 < ×1.050
 *   - tile48x64r12x8w8x4k16s: i8a8-linear-m185-n1280-k1280: ×0.639 < ×1.050
 *   - tile48x64r12x8w8x4k32: i8a8-linear-m185-n1280-k1280: ×0.732 < ×1.050
 *   - tile48x64r12x8w8x4k32s: i8a8-linear-m185-n1280-k1280: ×0.680 < ×1.050
 *   - tile48x64r12x8w8x4k64: i8a8-linear-m185-n1280-k1280: ×0.855 < ×1.050
 *   - tile48x64r12x8w8x4k64s: geometric mean ×1.304 (below the adopted ×2.326)
 *   - tile64x128r16x8w16x4k16: i8a8-linear-m185-n1280-k1280: ×0.731 < ×1.050
 *   - tile64x128r16x8w16x4k16s: i8a8-linear-m185-n1280-k1280: ×0.571 < ×1.050
 *   - tile64x128r16x8w16x4k32: i8a8-linear-m185-n1280-k1280: ×0.763 < ×1.050
 *   - tile64x128r16x8w16x4k32s: i8a8-linear-m185-n1280-k1280: ×0.605 < ×1.050
 *   - tile64x128r16x8w16x4k64: i8a8-linear-m185-n1280-k1280: ×0.779 < ×1.050
 *   - tile64x128r16x8w16x4k64s: i8a8-linear-m185-n1280-k1280: ×0.942 < ×1.050
 *   - tile64x128r4x8w16x16k16: geometric mean ×1.407 (below the adopted ×2.326)
 *   - tile64x128r4x8w16x16k32: geometric mean ×1.399 (below the adopted ×2.326)
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m256-n1280-k1280: ×0.966 < ×1.050
 *   - tile64x128r8x8w16x8k16s: i8a8-linear-m185-n1280-k1280: ×0.904 < ×1.050
 *   - tile64x128r8x8w16x8k32: geometric mean ×1.256 (below the adopted ×2.326)
 *   - tile64x128r8x8w16x8k32s: geometric mean ×1.257 (below the adopted ×2.326)
 *   - tile64x128r8x8w16x8k64: geometric mean ×1.168 (below the adopted ×2.326)
 *   - tile64x128r8x8w16x8k64s: geometric mean ×1.593 (below the adopted ×2.326)
 *   - tile64x32r16x4w8x4k16: i8a8-linear-m256-n1280-k1280: ×0.712 < ×1.050
 *   - tile64x32r16x4w8x4k16s: i8a8-linear-m256-n1280-k1280: ×0.683 < ×1.050
 *   - tile64x32r16x4w8x4k32: geometric mean ×1.240 (below the adopted ×2.326)
 *   - tile64x32r16x4w8x4k32s: geometric mean ×1.349 (below the adopted ×2.326)
 *   - tile64x32r16x4w8x4k64: i8a8-linear-m1-n18432-k3072: ×0.734 < ×1.050
 *   - tile64x32r16x4w8x4k64s: geometric mean ×1.491 (below the adopted ×2.326)
 *   - tile64x32r4x4w8x16k16: geometric mean ×1.863 (below the adopted ×2.326)
 *   - tile64x32r4x4w8x16k32: geometric mean ×1.553 (below the adopted ×2.326)
 *   - tile64x32r8x4w8x8k16: geometric mean ×1.664 (below the adopted ×2.326)
 *   - tile64x32r8x4w8x8k16s: i8a8-linear-m256-n1280-k1280: ×0.886 < ×1.050
 *   - tile64x32r8x4w8x8k32: geometric mean ×1.752 (below the adopted ×2.326)
 *   - tile64x32r8x4w8x8k32s: geometric mean ×1.621 (below the adopted ×2.326)
 *   - tile64x32r8x4w8x8k64: geometric mean ×1.364 (below the adopted ×2.326)
 *   - tile64x32r8x4w8x8k64s: geometric mean ×1.916 (below the adopted ×2.326)
 *   - tile64x64r16x4w16x4k16: geometric mean ×1.226 (below the adopted ×2.326)
 *   - tile64x64r16x4w16x4k16s: i8a8-linear-m256-n1280-k1280: ×0.760 < ×1.050
 *   - tile64x64r16x4w16x4k32: i8a8-linear-m256-n1280-k1280: ×0.706 < ×1.050
 *   - tile64x64r16x4w16x4k32s: i8a8-linear-m256-n1280-k1280: ×0.617 < ×1.050
 *   - tile64x64r16x4w16x4k64: geometric mean ×1.240 (below the adopted ×2.326)
 *   - tile64x64r16x4w16x4k64s: geometric mean ×1.405 (below the adopted ×2.326)
 *   - tile64x64r16x8w8x4k16: i8a8-linear-m256-n1280-k1280: ×0.706 < ×1.050
 *   - tile64x64r16x8w8x4k16s: i8a8-linear-m256-n1280-k1280: ×0.517 < ×1.050
 *   - tile64x64r16x8w8x4k32: i8a8-linear-m256-n1280-k1280: ×0.657 < ×1.050
 *   - tile64x64r16x8w8x4k32s: i8a8-linear-m256-n1280-k1280: ×0.523 < ×1.050
 *   - tile64x64r16x8w8x4k64: i8a8-linear-m256-n1280-k1280: ×0.729 < ×1.050
 *   - tile64x64r16x8w8x4k64s: i8a8-linear-m256-n1280-k1280: ×0.926 < ×1.050
 *   - tile64x64r4x4w16x16k16: geometric mean ×1.700 (below the adopted ×2.326)
 *   - tile64x64r4x4w16x16k32: geometric mean ×1.505 (below the adopted ×2.326)
 *   - tile64x64r4x8w8x16k16: geometric mean ×1.736 (below the adopted ×2.326)
 *   - tile64x64r4x8w8x16k32: geometric mean ×1.715 (below the adopted ×2.326)
 *   - tile64x64r8x4w16x8k16: geometric mean ×1.789 (below the adopted ×2.326)
 *   - tile64x64r8x4w16x8k16s: i8a8-linear-m256-n1280-k1280: ×0.882 < ×1.050
 *   - tile64x64r8x4w16x8k32: geometric mean ×1.784 (below the adopted ×2.326)
 *   - tile64x64r8x4w16x8k32s: geometric mean ×1.370 (below the adopted ×2.326)
 *   - tile64x64r8x4w16x8k64: geometric mean ×1.554 (below the adopted ×2.326)
 *   - tile64x64r8x4w16x8k64s: geometric mean ×2.070 (below the adopted ×2.326)
 *   - tile64x64r8x8w8x8k16: i8a8-linear-m256-n1280-k1280: ×1.018 < ×1.050
 *   - tile64x64r8x8w8x8k16s: i8a8-linear-m256-n1280-k1280: ×0.879 < ×1.050
 *   - tile64x64r8x8w8x8k32: geometric mean ×1.328 (below the adopted ×2.326)
 *   - tile64x64r8x8w8x8k32s: geometric mean ×1.240 (below the adopted ×2.326)
 *   - tile64x64r8x8w8x8k64: i8a8-linear-m256-n1280-k1280: ×1.044 < ×1.050
 *   - tile64x64r8x8w8x8k64s: geometric mean ×1.645 (below the adopted ×2.326)
 *   - tile96x128r12x8w16x8k16: i8a8-linear-m256-n1280-k1280: ×0.834 < ×1.050
 *   - tile96x128r12x8w16x8k16s: i8a8-linear-m185-n1280-k1280: ×0.725 < ×1.050
 *   - tile96x128r12x8w16x8k32: i8a8-linear-m185-n1280-k1280: ×0.926 < ×1.050
 *   - tile96x128r12x8w16x8k32s: i8a8-linear-m185-n1280-k1280: ×0.841 < ×1.050
 *   - tile96x128r12x8w16x8k64: i8a8-linear-m185-n1280-k1280: ×0.896 < ×1.050
 *   - tile96x128r12x8w16x8k64s: geometric mean ×1.164 (below the adopted ×2.326)
 *   - tile96x32r12x4w8x8k16: geometric mean ×1.407 (below the adopted ×2.326)
 *   - tile96x32r12x4w8x8k16s: i8a8-linear-m185-n1280-k1280: ×0.982 < ×1.050
 *   - tile96x32r12x4w8x8k32: geometric mean ×1.195 (below the adopted ×2.326)
 *   - tile96x32r12x4w8x8k32s: i8a8-linear-m1-n18432-k3072: ×0.623 < ×1.050
 *   - tile96x32r12x4w8x8k64: i8a8-linear-m1-n18432-k3072: ×0.881 < ×1.050
 *   - tile96x32r12x4w8x8k64s: geometric mean ×1.533 (below the adopted ×2.326)
 *   - tile96x64r12x4w16x8k16: geometric mean ×1.457 (below the adopted ×2.326)
 *   - tile96x64r12x4w16x8k16s: i8a8-linear-m185-n1280-k1280: ×0.852 < ×1.050
 *   - tile96x64r12x4w16x8k32: geometric mean ×1.562 (below the adopted ×2.326)
 *   - tile96x64r12x4w16x8k32s: geometric mean ×1.422 (below the adopted ×2.326)
 *   - tile96x64r12x4w16x8k64: geometric mean ×1.391 (below the adopted ×2.326)
 *   - tile96x64r12x4w16x8k64s: geometric mean ×1.789 (below the adopted ×2.326)
 *   - tile96x64r12x8w8x8k16: i8a8-linear-m185-n1280-k1280: ×0.924 < ×1.050
 *   - tile96x64r12x8w8x8k16s: i8a8-linear-m185-n1280-k1280: ×0.707 < ×1.050
 *   - tile96x64r12x8w8x8k32: i8a8-linear-m185-n1280-k1280: ×0.856 < ×1.050
 *   - tile96x64r12x8w8x8k32s: i8a8-linear-m185-n1280-k1280: ×0.816 < ×1.050
 *   - tile96x64r12x8w8x8k64: i8a8-linear-m1-n18432-k3072: ×0.735 < ×1.050
 *   - tile96x64r12x8w8x8k64s: geometric mean ×1.163 (below the adopted ×2.326)
 * - i8a8.linear 257〜512 (i8a8 linear row count 257〜512, 1 cases): adopted tile128x64r16x4w16x8k64s ×1.190 (×1.190–×1.190)
 *   - tile128x128r16x8w16x8k16: i8a8-linear-m512-n3072-k3072: ×0.755 < ×1.050
 *   - tile128x128r16x8w16x8k16s: i8a8-linear-m512-n3072-k3072: ×0.695 < ×1.050
 *   - tile128x128r16x8w16x8k32: i8a8-linear-m512-n3072-k3072: ×0.741 < ×1.050
 *   - tile128x128r16x8w16x8k32s: i8a8-linear-m512-n3072-k3072: ×0.473 < ×1.050
 *   - tile128x128r16x8w16x8k64: i8a8-linear-m512-n3072-k3072: ×0.717 < ×1.050
 *   - tile128x128r16x8w16x8k64s: i8a8-linear-m512-n3072-k3072: ×0.553 < ×1.050
 *   - tile128x128r8x8w16x16k16: i8a8-linear-m512-n3072-k3072: ×0.755 < ×1.050
 *   - tile128x128r8x8w16x16k16s: i8a8-linear-m512-n3072-k3072: ×0.650 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-linear-m512-n3072-k3072: ×0.784 < ×1.050
 *   - tile128x128r8x8w16x16k32s: i8a8-linear-m512-n3072-k3072: ×0.835 < ×1.050
 *   - tile128x128r8x8w16x16k64: i8a8-linear-m512-n3072-k3072: ×0.755 < ×1.050
 *   - tile128x128r8x8w16x16k64s: i8a8-linear-m512-n3072-k3072: ×0.901 < ×1.050
 *   - tile128x32r16x4w8x8k16: i8a8-linear-m512-n3072-k3072: ×0.819 < ×1.050
 *   - tile128x32r16x4w8x8k16s: i8a8-linear-m512-n3072-k3072: ×0.586 < ×1.050
 *   - tile128x32r16x4w8x8k32: i8a8-linear-m512-n3072-k3072: ×0.921 < ×1.050
 *   - tile128x32r16x4w8x8k32s: i8a8-linear-m512-n3072-k3072: ×0.391 < ×1.050
 *   - tile128x32r16x4w8x8k64: i8a8-linear-m512-n3072-k3072: ×0.759 < ×1.050
 *   - tile128x32r16x4w8x8k64s: i8a8-linear-m512-n3072-k3072: ×0.872 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-linear-m512-n3072-k3072: ×0.891 < ×1.050
 *   - tile128x32r8x4w8x16k16s: i8a8-linear-m512-n3072-k3072: ×0.573 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-linear-m512-n3072-k3072: ×1.013 < ×1.050
 *   - tile128x32r8x4w8x16k32s: i8a8-linear-m512-n3072-k3072: ×0.829 < ×1.050
 *   - tile128x32r8x4w8x16k64: i8a8-linear-m512-n3072-k3072: ×0.751 < ×1.050
 *   - tile128x32r8x4w8x16k64s: geometric mean ×1.127 (below the adopted ×1.190)
 *   - tile128x64r16x4w16x8k16: i8a8-linear-m512-n3072-k3072: ×0.975 < ×1.050
 *   - tile128x64r16x4w16x8k16s: i8a8-linear-m512-n3072-k3072: ×0.716 < ×1.050
 *   - tile128x64r16x4w16x8k32: i8a8-linear-m512-n3072-k3072: ×0.981 < ×1.050
 *   - tile128x64r16x4w16x8k32s: i8a8-linear-m512-n3072-k3072: ×0.501 < ×1.050
 *   - tile128x64r16x4w16x8k64: i8a8-linear-m512-n3072-k3072: ×0.903 < ×1.050
 *   - tile128x64r16x8w8x8k16: i8a8-linear-m512-n3072-k3072: ×0.750 < ×1.050
 *   - tile128x64r16x8w8x8k16s: i8a8-linear-m512-n3072-k3072: ×0.714 < ×1.050
 *   - tile128x64r16x8w8x8k32: i8a8-linear-m512-n3072-k3072: ×0.784 < ×1.050
 *   - tile128x64r16x8w8x8k32s: i8a8-linear-m512-n3072-k3072: ×0.736 < ×1.050
 *   - tile128x64r16x8w8x8k64: i8a8-linear-m512-n3072-k3072: ×0.734 < ×1.050
 *   - tile128x64r16x8w8x8k64s: i8a8-linear-m512-n3072-k3072: ×0.946 < ×1.050
 *   - tile128x64r8x4w16x16k16: geometric mean ×1.115 (below the adopted ×1.190)
 *   - tile128x64r8x4w16x16k16s: i8a8-linear-m512-n3072-k3072: ×0.733 < ×1.050
 *   - tile128x64r8x4w16x16k32: geometric mean ×1.054 (below the adopted ×1.190)
 *   - tile128x64r8x4w16x16k32s: i8a8-linear-m512-n3072-k3072: ×1.009 < ×1.050
 *   - tile128x64r8x4w16x16k64: i8a8-linear-m512-n3072-k3072: ×0.939 < ×1.050
 *   - tile128x64r8x4w16x16k64s: geometric mean ×1.173 (below the adopted ×1.190)
 *   - tile128x64r8x8w8x16k16s: i8a8-linear-m512-n3072-k3072: ×0.777 < ×1.050
 *   - tile128x64r8x8w8x16k32: i8a8-linear-m512-n3072-k3072: ×0.960 < ×1.050
 *   - tile128x64r8x8w8x16k32s: geometric mean ×1.062 (below the adopted ×1.190)
 *   - tile128x64r8x8w8x16k64: i8a8-linear-m512-n3072-k3072: ×0.870 < ×1.050
 *   - tile128x64r8x8w8x16k64s: geometric mean ×1.189 (below the adopted ×1.190)
 *   - tile16x128r4x8w16x4k16: i8a8-linear-m512-n3072-k3072: ×0.477 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-linear-m512-n3072-k3072: ×0.722 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-linear-m512-n3072-k3072: ×0.472 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.688 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-linear-m512-n3072-k3072: ×0.362 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-linear-m512-n3072-k3072: ×0.458 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-linear-m512-n3072-k3072: ×0.431 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.684 < ×1.050
 *   - tile192x128r12x8w16x16k16: i8a8-linear-m512-n3072-k3072: ×0.589 < ×1.050
 *   - tile192x128r12x8w16x16k16s: i8a8-linear-m512-n3072-k3072: ×0.439 < ×1.050
 *   - tile192x128r12x8w16x16k32: i8a8-linear-m512-n3072-k3072: ×0.612 < ×1.050
 *   - tile192x128r12x8w16x16k32s: i8a8-linear-m512-n3072-k3072: ×0.526 < ×1.050
 *   - tile192x32r12x4w8x16k16: i8a8-linear-m512-n3072-k3072: ×0.861 < ×1.050
 *   - tile192x32r12x4w8x16k16s: i8a8-linear-m512-n3072-k3072: ×0.566 < ×1.050
 *   - tile192x32r12x4w8x16k32: i8a8-linear-m512-n3072-k3072: ×0.822 < ×1.050
 *   - tile192x32r12x4w8x16k32s: i8a8-linear-m512-n3072-k3072: ×0.382 < ×1.050
 *   - tile192x32r12x4w8x16k64: i8a8-linear-m512-n3072-k3072: ×0.614 < ×1.050
 *   - tile192x32r12x4w8x16k64s: i8a8-linear-m512-n3072-k3072: ×0.864 < ×1.050
 *   - tile192x64r12x4w16x16k16: i8a8-linear-m512-n3072-k3072: ×0.746 < ×1.050
 *   - tile192x64r12x4w16x16k16s: i8a8-linear-m512-n3072-k3072: ×0.556 < ×1.050
 *   - tile192x64r12x4w16x16k32: i8a8-linear-m512-n3072-k3072: ×0.750 < ×1.050
 *   - tile192x64r12x4w16x16k32s: i8a8-linear-m512-n3072-k3072: ×0.451 < ×1.050
 *   - tile192x64r12x4w16x16k64: i8a8-linear-m512-n3072-k3072: ×0.637 < ×1.050
 *   - tile192x64r12x4w16x16k64s: i8a8-linear-m512-n3072-k3072: ×0.276 < ×1.050
 *   - tile192x64r12x8w8x16k16: i8a8-linear-m512-n3072-k3072: ×0.703 < ×1.050
 *   - tile192x64r12x8w8x16k16s: i8a8-linear-m512-n3072-k3072: ×0.621 < ×1.050
 *   - tile192x64r12x8w8x16k32: i8a8-linear-m512-n3072-k3072: ×0.646 < ×1.050
 *   - tile192x64r12x8w8x16k32s: i8a8-linear-m512-n3072-k3072: ×0.786 < ×1.050
 *   - tile192x64r12x8w8x16k64: i8a8-linear-m512-n3072-k3072: ×0.638 < ×1.050
 *   - tile192x64r12x8w8x16k64s: i8a8-linear-m512-n3072-k3072: ×0.824 < ×1.050
 *   - tile256x128r16x8w16x16k16: i8a8-linear-m512-n3072-k3072: ×0.726 < ×1.050
 *   - tile256x128r16x8w16x16k16s: i8a8-linear-m512-n3072-k3072: ×0.646 < ×1.050
 *   - tile256x128r16x8w16x16k32: i8a8-linear-m512-n3072-k3072: ×0.782 < ×1.050
 *   - tile256x128r16x8w16x16k32s: i8a8-linear-m512-n3072-k3072: ×0.774 < ×1.050
 *   - tile256x32r16x4w8x16k16: i8a8-linear-m512-n3072-k3072: ×0.853 < ×1.050
 *   - tile256x32r16x4w8x16k16s: i8a8-linear-m512-n3072-k3072: ×0.536 < ×1.050
 *   - tile256x32r16x4w8x16k32: i8a8-linear-m512-n3072-k3072: ×0.908 < ×1.050
 *   - tile256x32r16x4w8x16k32s: i8a8-linear-m512-n3072-k3072: ×0.844 < ×1.050
 *   - tile256x64r16x4w16x16k16: i8a8-linear-m512-n3072-k3072: ×0.767 < ×1.050
 *   - tile256x64r16x4w16x16k16s: i8a8-linear-m512-n3072-k3072: ×0.595 < ×1.050
 *   - tile256x64r16x4w16x16k32: i8a8-linear-m512-n3072-k3072: ×0.790 < ×1.050
 *   - tile256x64r16x4w16x16k32s: i8a8-linear-m512-n3072-k3072: ×0.747 < ×1.050
 *   - tile256x64r16x8w8x16k16: i8a8-linear-m512-n3072-k3072: ×0.728 < ×1.050
 *   - tile256x64r16x8w8x16k16s: i8a8-linear-m512-n3072-k3072: ×0.588 < ×1.050
 *   - tile256x64r16x8w8x16k32: i8a8-linear-m512-n3072-k3072: ×0.699 < ×1.050
 *   - tile256x64r16x8w8x16k32s: i8a8-linear-m512-n3072-k3072: ×0.418 < ×1.050
 *   - tile32x128r4x8w16x8k16: i8a8-linear-m512-n3072-k3072: ×0.847 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-linear-m512-n3072-k3072: ×0.957 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-linear-m512-n3072-k3072: ×0.849 < ×1.050
 *   - tile32x128r8x8w16x4k16s: i8a8-linear-m512-n3072-k3072: ×0.624 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-linear-m512-n3072-k3072: ×0.652 < ×1.050
 *   - tile32x128r8x8w16x4k32s: i8a8-linear-m512-n3072-k3072: ×0.836 < ×1.050
 *   - tile32x128r8x8w16x4k64: i8a8-linear-m512-n3072-k3072: ×0.765 < ×1.050
 *   - tile32x128r8x8w16x4k64s: i8a8-linear-m512-n3072-k3072: ×0.920 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-linear-m512-n3072-k3072: ×0.507 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-linear-m512-n3072-k3072: ×0.415 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-linear-m512-n3072-k3072: ×0.573 < ×1.050
 *   - tile32x32r8x4w8x4k16s: i8a8-linear-m512-n3072-k3072: ×0.443 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.866 < ×1.050
 *   - tile32x32r8x4w8x4k32s: i8a8-linear-m512-n3072-k3072: ×0.724 < ×1.050
 *   - tile32x32r8x4w8x4k64: i8a8-linear-m512-n3072-k3072: ×0.580 < ×1.050
 *   - tile32x32r8x4w8x4k64s: i8a8-linear-m512-n3072-k3072: ×0.837 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-linear-m512-n3072-k3072: ×0.668 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-linear-m512-n3072-k3072: ×0.996 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-linear-m512-n3072-k3072: ×0.669 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-linear-m512-n3072-k3072: ×0.950 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-linear-m512-n3072-k3072: ×0.661 < ×1.050
 *   - tile32x64r8x4w16x4k16s: i8a8-linear-m512-n3072-k3072: ×0.578 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-linear-m512-n3072-k3072: ×0.976 < ×1.050
 *   - tile32x64r8x4w16x4k32s: i8a8-linear-m512-n3072-k3072: ×0.841 < ×1.050
 *   - tile32x64r8x4w16x4k64: i8a8-linear-m512-n3072-k3072: ×0.725 < ×1.050
 *   - tile32x64r8x4w16x4k64s: geometric mean ×1.080 (below the adopted ×1.190)
 *   - tile32x64r8x8w8x4k16: i8a8-linear-m512-n3072-k3072: ×0.675 < ×1.050
 *   - tile32x64r8x8w8x4k16s: i8a8-linear-m512-n3072-k3072: ×0.587 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.861 < ×1.050
 *   - tile32x64r8x8w8x4k32s: i8a8-linear-m512-n3072-k3072: ×0.894 < ×1.050
 *   - tile32x64r8x8w8x4k64: i8a8-linear-m512-n3072-k3072: ×0.550 < ×1.050
 *   - tile32x64r8x8w8x4k64s: i8a8-linear-m512-n3072-k3072: ×0.919 < ×1.050
 *   - tile48x128r12x8w16x4k16: i8a8-linear-m512-n3072-k3072: ×0.831 < ×1.050
 *   - tile48x128r12x8w16x4k16s: i8a8-linear-m512-n3072-k3072: ×0.719 < ×1.050
 *   - tile48x128r12x8w16x4k32: i8a8-linear-m512-n3072-k3072: ×0.687 < ×1.050
 *   - tile48x128r12x8w16x4k32s: i8a8-linear-m512-n3072-k3072: ×0.858 < ×1.050
 *   - tile48x128r12x8w16x4k64: i8a8-linear-m512-n3072-k3072: ×0.807 < ×1.050
 *   - tile48x128r12x8w16x4k64s: geometric mean ×1.076 (below the adopted ×1.190)
 *   - tile48x32r12x4w8x4k16: i8a8-linear-m512-n3072-k3072: ×0.631 < ×1.050
 *   - tile48x32r12x4w8x4k16s: i8a8-linear-m512-n3072-k3072: ×0.501 < ×1.050
 *   - tile48x32r12x4w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.958 < ×1.050
 *   - tile48x32r12x4w8x4k32s: i8a8-linear-m512-n3072-k3072: ×0.625 < ×1.050
 *   - tile48x32r12x4w8x4k64: i8a8-linear-m512-n3072-k3072: ×0.753 < ×1.050
 *   - tile48x32r12x4w8x4k64s: i8a8-linear-m512-n3072-k3072: ×0.922 < ×1.050
 *   - tile48x64r12x4w16x4k16: i8a8-linear-m512-n3072-k3072: ×0.930 < ×1.050
 *   - tile48x64r12x4w16x4k16s: i8a8-linear-m512-n3072-k3072: ×0.665 < ×1.050
 *   - tile48x64r12x4w16x4k32: i8a8-linear-m512-n3072-k3072: ×1.027 < ×1.050
 *   - tile48x64r12x4w16x4k32s: i8a8-linear-m512-n3072-k3072: ×0.848 < ×1.050
 *   - tile48x64r12x4w16x4k64: i8a8-linear-m512-n3072-k3072: ×0.860 < ×1.050
 *   - tile48x64r12x4w16x4k64s: geometric mean ×1.054 (below the adopted ×1.190)
 *   - tile48x64r12x8w8x4k16: i8a8-linear-m512-n3072-k3072: ×0.765 < ×1.050
 *   - tile48x64r12x8w8x4k16s: i8a8-linear-m512-n3072-k3072: ×0.636 < ×1.050
 *   - tile48x64r12x8w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.615 < ×1.050
 *   - tile48x64r12x8w8x4k32s: i8a8-linear-m512-n3072-k3072: ×0.766 < ×1.050
 *   - tile48x64r12x8w8x4k64: i8a8-linear-m512-n3072-k3072: ×0.782 < ×1.050
 *   - tile48x64r12x8w8x4k64s: geometric mean ×1.061 (below the adopted ×1.190)
 *   - tile64x128r16x8w16x4k16: i8a8-linear-m512-n3072-k3072: ×0.741 < ×1.050
 *   - tile64x128r16x8w16x4k16s: i8a8-linear-m512-n3072-k3072: ×0.709 < ×1.050
 *   - tile64x128r16x8w16x4k32: i8a8-linear-m512-n3072-k3072: ×0.772 < ×1.050
 *   - tile64x128r16x8w16x4k32s: i8a8-linear-m512-n3072-k3072: ×0.729 < ×1.050
 *   - tile64x128r16x8w16x4k64: i8a8-linear-m512-n3072-k3072: ×0.740 < ×1.050
 *   - tile64x128r16x8w16x4k64s: i8a8-linear-m512-n3072-k3072: ×0.952 < ×1.050
 *   - tile64x128r4x8w16x16k16: geometric mean ×1.111 (below the adopted ×1.190)
 *   - tile64x128r4x8w16x16k32: i8a8-linear-m512-n3072-k3072: ×1.027 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m512-n3072-k3072: ×0.971 < ×1.050
 *   - tile64x128r8x8w16x8k16s: i8a8-linear-m512-n3072-k3072: ×0.789 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-linear-m512-n3072-k3072: ×0.967 < ×1.050
 *   - tile64x128r8x8w16x8k32s: i8a8-linear-m512-n3072-k3072: ×1.045 < ×1.050
 *   - tile64x128r8x8w16x8k64: i8a8-linear-m512-n3072-k3072: ×0.871 < ×1.050
 *   - tile64x128r8x8w16x8k64s: geometric mean ×1.178 (below the adopted ×1.190)
 *   - tile64x32r16x4w8x4k16: i8a8-linear-m512-n3072-k3072: ×0.741 < ×1.050
 *   - tile64x32r16x4w8x4k16s: i8a8-linear-m512-n3072-k3072: ×0.582 < ×1.050
 *   - tile64x32r16x4w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.907 < ×1.050
 *   - tile64x32r16x4w8x4k32s: i8a8-linear-m512-n3072-k3072: ×0.848 < ×1.050
 *   - tile64x32r16x4w8x4k64: i8a8-linear-m512-n3072-k3072: ×0.528 < ×1.050
 *   - tile64x32r16x4w8x4k64s: i8a8-linear-m512-n3072-k3072: ×0.929 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-linear-m512-n3072-k3072: ×0.723 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-linear-m512-n3072-k3072: ×0.666 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-linear-m512-n3072-k3072: ×0.693 < ×1.050
 *   - tile64x32r8x4w8x8k16s: i8a8-linear-m512-n3072-k3072: ×0.576 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-linear-m512-n3072-k3072: ×1.002 < ×1.050
 *   - tile64x32r8x4w8x8k32s: i8a8-linear-m512-n3072-k3072: ×0.874 < ×1.050
 *   - tile64x32r8x4w8x8k64: i8a8-linear-m512-n3072-k3072: ×0.726 < ×1.050
 *   - tile64x32r8x4w8x8k64s: i8a8-linear-m512-n3072-k3072: ×0.991 < ×1.050
 *   - tile64x64r16x4w16x4k16: i8a8-linear-m512-n3072-k3072: ×0.898 < ×1.050
 *   - tile64x64r16x4w16x4k16s: i8a8-linear-m512-n3072-k3072: ×0.750 < ×1.050
 *   - tile64x64r16x4w16x4k32: i8a8-linear-m512-n3072-k3072: ×0.835 < ×1.050
 *   - tile64x64r16x4w16x4k32s: i8a8-linear-m512-n3072-k3072: ×0.419 < ×1.050
 *   - tile64x64r16x4w16x4k64: i8a8-linear-m512-n3072-k3072: ×0.853 < ×1.050
 *   - tile64x64r16x4w16x4k64s: i8a8-linear-m512-n3072-k3072: ×0.963 < ×1.050
 *   - tile64x64r16x8w8x4k16: i8a8-linear-m512-n3072-k3072: ×0.710 < ×1.050
 *   - tile64x64r16x8w8x4k16s: i8a8-linear-m512-n3072-k3072: ×0.632 < ×1.050
 *   - tile64x64r16x8w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.734 < ×1.050
 *   - tile64x64r16x8w8x4k32s: i8a8-linear-m512-n3072-k3072: ×0.651 < ×1.050
 *   - tile64x64r16x8w8x4k64: i8a8-linear-m512-n3072-k3072: ×0.654 < ×1.050
 *   - tile64x64r16x8w8x4k64s: i8a8-linear-m512-n3072-k3072: ×0.877 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-linear-m512-n3072-k3072: ×1.027 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-linear-m512-n3072-k3072: ×0.787 < ×1.050
 *   - tile64x64r4x8w8x16k16: geometric mean ×1.090 (below the adopted ×1.190)
 *   - tile64x64r4x8w8x16k32: i8a8-linear-m512-n3072-k3072: ×1.039 < ×1.050
 *   - tile64x64r8x4w16x8k16: i8a8-linear-m512-n3072-k3072: ×0.931 < ×1.050
 *   - tile64x64r8x4w16x8k16s: i8a8-linear-m512-n3072-k3072: ×0.705 < ×1.050
 *   - tile64x64r8x4w16x8k32: geometric mean ×1.066 (below the adopted ×1.190)
 *   - tile64x64r8x4w16x8k32s: i8a8-linear-m512-n3072-k3072: ×0.756 < ×1.050
 *   - tile64x64r8x4w16x8k64: i8a8-linear-m512-n3072-k3072: ×0.890 < ×1.050
 *   - tile64x64r8x4w16x8k64s: geometric mean ×1.150 (below the adopted ×1.190)
 *   - tile64x64r8x8w8x8k16: i8a8-linear-m512-n3072-k3072: ×0.927 < ×1.050
 *   - tile64x64r8x8w8x8k16s: i8a8-linear-m512-n3072-k3072: ×0.714 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-linear-m512-n3072-k3072: ×0.966 < ×1.050
 *   - tile64x64r8x8w8x8k32s: i8a8-linear-m512-n3072-k3072: ×1.013 < ×1.050
 *   - tile64x64r8x8w8x8k64: i8a8-linear-m512-n3072-k3072: ×0.844 < ×1.050
 *   - tile64x64r8x8w8x8k64s: geometric mean ×1.113 (below the adopted ×1.190)
 *   - tile96x128r12x8w16x8k16: i8a8-linear-m512-n3072-k3072: ×0.714 < ×1.050
 *   - tile96x128r12x8w16x8k16s: i8a8-linear-m512-n3072-k3072: ×0.660 < ×1.050
 *   - tile96x128r12x8w16x8k32: i8a8-linear-m512-n3072-k3072: ×0.690 < ×1.050
 *   - tile96x128r12x8w16x8k32s: i8a8-linear-m512-n3072-k3072: ×0.802 < ×1.050
 *   - tile96x128r12x8w16x8k64: i8a8-linear-m512-n3072-k3072: ×0.647 < ×1.050
 *   - tile96x128r12x8w16x8k64s: i8a8-linear-m512-n3072-k3072: ×0.846 < ×1.050
 *   - tile96x32r12x4w8x8k16: i8a8-linear-m512-n3072-k3072: ×0.860 < ×1.050
 *   - tile96x32r12x4w8x8k16s: i8a8-linear-m512-n3072-k3072: ×0.565 < ×1.050
 *   - tile96x32r12x4w8x8k32: i8a8-linear-m512-n3072-k3072: ×0.826 < ×1.050
 *   - tile96x32r12x4w8x8k32s: i8a8-linear-m512-n3072-k3072: ×0.359 < ×1.050
 *   - tile96x32r12x4w8x8k64: i8a8-linear-m512-n3072-k3072: ×0.610 < ×1.050
 *   - tile96x32r12x4w8x8k64s: i8a8-linear-m512-n3072-k3072: ×0.838 < ×1.050
 *   - tile96x64r12x4w16x8k16: i8a8-linear-m512-n3072-k3072: ×0.952 < ×1.050
 *   - tile96x64r12x4w16x8k16s: i8a8-linear-m512-n3072-k3072: ×0.712 < ×1.050
 *   - tile96x64r12x4w16x8k32: i8a8-linear-m512-n3072-k3072: ×0.928 < ×1.050
 *   - tile96x64r12x4w16x8k32s: i8a8-linear-m512-n3072-k3072: ×0.874 < ×1.050
 *   - tile96x64r12x4w16x8k64: i8a8-linear-m512-n3072-k3072: ×0.728 < ×1.050
 *   - tile96x64r12x4w16x8k64s: i8a8-linear-m512-n3072-k3072: ×1.010 < ×1.050
 *   - tile96x64r12x8w8x8k16: i8a8-linear-m512-n3072-k3072: ×0.707 < ×1.050
 *   - tile96x64r12x8w8x8k16s: i8a8-linear-m512-n3072-k3072: ×0.671 < ×1.050
 *   - tile96x64r12x8w8x8k32: i8a8-linear-m512-n3072-k3072: ×0.662 < ×1.050
 *   - tile96x64r12x8w8x8k32s: i8a8-linear-m512-n3072-k3072: ×0.796 < ×1.050
 *   - tile96x64r12x8w8x8k64: i8a8-linear-m512-n3072-k3072: ×0.582 < ×1.050
 *   - tile96x64r12x8w8x8k64s: i8a8-linear-m512-n3072-k3072: ×0.770 < ×1.050
 * - i8a8.linear > 512 (i8a8 linear row count > 512, 9 cases): adopted tile128x64r8x8w8x16k64s ×1.195 (×1.167–×1.220)
 *   - tile128x128r16x8w16x8k16: i8a8-linear-m1024-n2048-k2048: ×0.924 < ×1.050
 *   - tile128x128r16x8w16x8k16s: i8a8-linear-m1024-n2048-k2048: ×0.847 < ×1.050
 *   - tile128x128r16x8w16x8k32: i8a8-linear-m1024-n2048-k2048: ×0.931 < ×1.050
 *   - tile128x128r16x8w16x8k32s: i8a8-linear-m1024-n8192-k2048: ×0.541 < ×1.050
 *   - tile128x128r16x8w16x8k64: i8a8-linear-m1024-n2048-k2048: ×0.909 < ×1.050
 *   - tile128x128r16x8w16x8k64s: i8a8-linear-m4096-n2048-k2048: ×0.604 < ×1.050
 *   - tile128x128r8x8w16x16k16: i8a8-linear-m4096-n2048-k2048: ×0.914 < ×1.050
 *   - tile128x128r8x8w16x16k16s: i8a8-linear-m1024-n8192-k2048: ×0.791 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-linear-m1024-n8192-k2048: ×0.932 < ×1.050
 *   - tile128x128r8x8w16x16k32s: i8a8-linear-m1024-n8192-k2048: ×1.029 < ×1.050
 *   - tile128x128r8x8w16x16k64: i8a8-linear-m4096-n2048-k2048: ×0.903 < ×1.050
 *   - tile128x128r8x8w16x16k64s: geometric mean ×1.127 (below the adopted ×1.195)
 *   - tile128x32r16x4w8x8k16: i8a8-linear-m1024-n2048-k2048: ×0.690 < ×1.050
 *   - tile128x32r16x4w8x8k16s: i8a8-linear-m7920-n3072-k3072: ×0.583 < ×1.050
 *   - tile128x32r16x4w8x8k32: i8a8-linear-m1024-n2048-k8192: ×0.910 < ×1.050
 *   - tile128x32r16x4w8x8k32s: i8a8-linear-m7920-n3072-k3072: ×0.355 < ×1.050
 *   - tile128x32r16x4w8x8k64: i8a8-linear-m1024-n2048-k8192: ×0.697 < ×1.050
 *   - tile128x32r16x4w8x8k64s: i8a8-linear-m1024-n8192-k2048: ×0.863 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-linear-m7920-n14336-k3072: ×0.665 < ×1.050
 *   - tile128x32r8x4w8x16k16s: i8a8-linear-m4096-n2048-k2048: ×0.574 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-linear-m1024-n2048-k8192: ×0.958 < ×1.050
 *   - tile128x32r8x4w8x16k32s: i8a8-linear-m1024-n2048-k8192: ×0.766 < ×1.050
 *   - tile128x32r8x4w8x16k64: i8a8-linear-m1024-n2048-k8192: ×0.736 < ×1.050
 *   - tile128x32r8x4w8x16k64s: i8a8-linear-m4096-n2048-k2048: ×1.040 < ×1.050
 *   - tile128x64r16x4w16x8k16: i8a8-linear-m7920-n14336-k3072: ×0.956 < ×1.050
 *   - tile128x64r16x4w16x8k16s: i8a8-linear-m1024-n2048-k2048: ×0.749 < ×1.050
 *   - tile128x64r16x4w16x8k32: i8a8-linear-m1024-n2048-k8192: ×0.998 < ×1.050
 *   - tile128x64r16x4w16x8k32s: i8a8-linear-m7920-n3072-k14336: ×0.468 < ×1.050
 *   - tile128x64r16x4w16x8k64: i8a8-linear-m1024-n2048-k8192: ×0.904 < ×1.050
 *   - tile128x64r16x4w16x8k64s: geometric mean ×1.193 (below the adopted ×1.195)
 *   - tile128x64r16x8w8x8k16: i8a8-linear-m1024-n2048-k2048: ×0.881 < ×1.050
 *   - tile128x64r16x8w8x8k16s: i8a8-linear-m4096-n2048-k2048: ×0.815 < ×1.050
 *   - tile128x64r16x8w8x8k32: i8a8-linear-m1024-n2048-k2048: ×0.958 < ×1.050
 *   - tile128x64r16x8w8x8k32s: i8a8-linear-m4096-n2048-k8192: ×0.840 < ×1.050
 *   - tile128x64r16x8w8x8k64: i8a8-linear-m4096-n2048-k2048: ×0.824 < ×1.050
 *   - tile128x64r16x8w8x8k64s: geometric mean ×1.119 (below the adopted ×1.195)
 *   - tile128x64r8x4w16x16k16: geometric mean ×1.137 (below the adopted ×1.195)
 *   - tile128x64r8x4w16x16k16s: i8a8-linear-m1024-n2048-k8192: ×0.735 < ×1.050
 *   - tile128x64r8x4w16x16k32: i8a8-linear-m1024-n2048-k8192: ×0.991 < ×1.050
 *   - tile128x64r8x4w16x16k32s: i8a8-linear-m1024-n2048-k8192: ×0.956 < ×1.050
 *   - tile128x64r8x4w16x16k64: i8a8-linear-m1024-n2048-k8192: ×0.925 < ×1.050
 *   - tile128x64r8x4w16x16k64s: geometric mean ×1.125 (below the adopted ×1.195)
 *   - tile128x64r8x8w8x16k16s: i8a8-linear-m1024-n2048-k8192: ×0.791 < ×1.050
 *   - tile128x64r8x8w8x16k32: i8a8-linear-m7920-n14336-k3072: ×0.987 < ×1.050
 *   - tile128x64r8x8w8x16k32s: geometric mean ×1.084 (below the adopted ×1.195)
 *   - tile128x64r8x8w8x16k64: i8a8-linear-m1024-n2048-k2048: ×0.883 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-linear-m1024-n2048-k8192: ×0.416 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-linear-m4096-n2048-k8192: ×0.699 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-linear-m7920-n3072-k14336: ×0.368 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-linear-m7920-n3072-k14336: ×0.629 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-linear-m7920-n3072-k3072: ×0.326 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-linear-m7920-n3072-k14336: ×0.474 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-linear-m1024-n2048-k8192: ×0.400 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-linear-m1024-n2048-k8192: ×0.674 < ×1.050
 *   - tile192x128r12x8w16x16k16: i8a8-linear-m1024-n2048-k2048: ×0.730 < ×1.050
 *   - tile192x128r12x8w16x16k16s: i8a8-linear-m1024-n2048-k2048: ×0.520 < ×1.050
 *   - tile192x128r12x8w16x16k32: i8a8-linear-m1024-n2048-k2048: ×0.761 < ×1.050
 *   - tile192x128r12x8w16x16k32s: i8a8-linear-m1024-n2048-k2048: ×0.625 < ×1.050
 *   - tile192x32r12x4w8x16k16: i8a8-linear-m1024-n2048-k8192: ×0.638 < ×1.050
 *   - tile192x32r12x4w8x16k16s: i8a8-linear-m1024-n2048-k8192: ×0.514 < ×1.050
 *   - tile192x32r12x4w8x16k32: i8a8-linear-m1024-n2048-k8192: ×0.853 < ×1.050
 *   - tile192x32r12x4w8x16k32s: i8a8-linear-m1024-n8192-k2048: ×0.380 < ×1.050
 *   - tile192x32r12x4w8x16k64: i8a8-linear-m1024-n2048-k8192: ×0.654 < ×1.050
 *   - tile192x32r12x4w8x16k64s: i8a8-linear-m1024-n2048-k2048: ×0.923 < ×1.050
 *   - tile192x64r12x4w16x16k16: i8a8-linear-m1024-n2048-k8192: ×0.938 < ×1.050
 *   - tile192x64r12x4w16x16k16s: i8a8-linear-m1024-n2048-k8192: ×0.655 < ×1.050
 *   - tile192x64r12x4w16x16k32: i8a8-linear-m1024-n2048-k8192: ×0.923 < ×1.050
 *   - tile192x64r12x4w16x16k32s: i8a8-linear-m1024-n8192-k2048: ×0.464 < ×1.050
 *   - tile192x64r12x4w16x16k64: i8a8-linear-m1024-n2048-k8192: ×0.803 < ×1.050
 *   - tile192x64r12x4w16x16k64s: i8a8-linear-m1024-n8192-k2048: ×0.312 < ×1.050
 *   - tile192x64r12x8w8x16k16: i8a8-linear-m1024-n2048-k8192: ×0.866 < ×1.050
 *   - tile192x64r12x8w8x16k16s: i8a8-linear-m1024-n2048-k8192: ×0.711 < ×1.050
 *   - tile192x64r12x8w8x16k32: i8a8-linear-m1024-n2048-k2048: ×0.799 < ×1.050
 *   - tile192x64r12x8w8x16k32s: i8a8-linear-m1024-n2048-k2048: ×0.923 < ×1.050
 *   - tile192x64r12x8w8x16k64: i8a8-linear-m1024-n2048-k2048: ×0.802 < ×1.050
 *   - tile192x64r12x8w8x16k64s: i8a8-linear-m1024-n2048-k2048: ×0.997 < ×1.050
 *   - tile256x128r16x8w16x16k16: i8a8-linear-m1024-n8192-k2048: ×0.866 < ×1.050
 *   - tile256x128r16x8w16x16k16s: i8a8-linear-m1024-n8192-k2048: ×0.746 < ×1.050
 *   - tile256x128r16x8w16x16k32: i8a8-linear-m4096-n2048-k8192: ×0.913 < ×1.050
 *   - tile256x128r16x8w16x16k32s: i8a8-linear-m7920-n3072-k14336: ×0.863 < ×1.050
 *   - tile256x32r16x4w8x16k16: i8a8-linear-m7920-n3072-k3072: ×0.710 < ×1.050
 *   - tile256x32r16x4w8x16k16s: i8a8-linear-m1024-n2048-k2048: ×0.559 < ×1.050
 *   - tile256x32r16x4w8x16k32: i8a8-linear-m1024-n2048-k8192: ×0.894 < ×1.050
 *   - tile256x32r16x4w8x16k32s: i8a8-linear-m4096-n2048-k8192: ×0.849 < ×1.050
 *   - tile256x64r16x4w16x16k16: i8a8-linear-m4096-n2048-k2048: ×0.919 < ×1.050
 *   - tile256x64r16x4w16x16k16s: i8a8-linear-m1024-n8192-k2048: ×0.723 < ×1.050
 *   - tile256x64r16x4w16x16k32: i8a8-linear-m4096-n2048-k2048: ×0.924 < ×1.050
 *   - tile256x64r16x4w16x16k32s: i8a8-linear-m4096-n2048-k8192: ×0.906 < ×1.050
 *   - tile256x64r16x8w8x16k16: i8a8-linear-m1024-n2048-k2048: ×0.821 < ×1.050
 *   - tile256x64r16x8w8x16k16s: i8a8-linear-m1024-n2048-k2048: ×0.735 < ×1.050
 *   - tile256x64r16x8w8x16k32: i8a8-linear-m1024-n2048-k2048: ×0.882 < ×1.050
 *   - tile256x64r16x8w8x16k32s: i8a8-linear-m4096-n2048-k2048: ×0.466 < ×1.050
 *   - tile32x128r4x8w16x8k16: i8a8-linear-m1024-n2048-k8192: ×0.689 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-linear-m1024-n2048-k8192: ×0.919 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-linear-m7920-n14336-k3072: ×0.719 < ×1.050
 *   - tile32x128r8x8w16x4k16s: i8a8-linear-m4096-n2048-k2048: ×0.580 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-linear-m1024-n2048-k8192: ×0.769 < ×1.050
 *   - tile32x128r8x8w16x4k32s: i8a8-linear-m4096-n2048-k2048: ×0.833 < ×1.050
 *   - tile32x128r8x8w16x4k64: i8a8-linear-m1024-n2048-k2048: ×0.702 < ×1.050
 *   - tile32x128r8x8w16x4k64s: i8a8-linear-m7920-n14336-k3072: ×0.881 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-linear-m7920-n14336-k3072: ×0.446 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-linear-m1024-n2048-k8192: ×0.344 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-linear-m1024-n2048-k8192: ×0.461 < ×1.050
 *   - tile32x32r8x4w8x4k16s: i8a8-linear-m1024-n2048-k8192: ×0.387 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-linear-m1024-n2048-k8192: ×0.804 < ×1.050
 *   - tile32x32r8x4w8x4k32s: i8a8-linear-m7920-n14336-k3072: ×0.646 < ×1.050
 *   - tile32x32r8x4w8x4k64: i8a8-linear-m1024-n2048-k8192: ×0.565 < ×1.050
 *   - tile32x32r8x4w8x4k64s: i8a8-linear-m7920-n14336-k3072: ×0.784 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-linear-m7920-n14336-k3072: ×0.580 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-linear-m7920-n3072-k3072: ×0.909 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-linear-m1024-n2048-k8192: ×0.614 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-linear-m1024-n2048-k2048: ×0.881 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-linear-m4096-n8192-k2048: ×0.611 < ×1.050
 *   - tile32x64r8x4w16x4k16s: i8a8-linear-m4096-n2048-k8192: ×0.494 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-linear-m1024-n2048-k8192: ×0.916 < ×1.050
 *   - tile32x64r8x4w16x4k32s: i8a8-linear-m1024-n2048-k8192: ×0.739 < ×1.050
 *   - tile32x64r8x4w16x4k64: i8a8-linear-m1024-n2048-k2048: ×0.716 < ×1.050
 *   - tile32x64r8x4w16x4k64s: i8a8-linear-m4096-n2048-k2048: ×0.992 < ×1.050
 *   - tile32x64r8x8w8x4k16: i8a8-linear-m7920-n3072-k14336: ×0.641 < ×1.050
 *   - tile32x64r8x8w8x4k16s: i8a8-linear-m4096-n2048-k2048: ×0.497 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-linear-m7920-n3072-k3072: ×0.840 < ×1.050
 *   - tile32x64r8x8w8x4k32s: i8a8-linear-m7920-n3072-k3072: ×0.792 < ×1.050
 *   - tile32x64r8x8w8x4k64: i8a8-linear-m1024-n8192-k2048: ×0.580 < ×1.050
 *   - tile32x64r8x8w8x4k64s: i8a8-linear-m7920-n3072-k14336: ×0.864 < ×1.050
 *   - tile48x128r12x8w16x4k16: i8a8-linear-m1024-n2048-k8192: ×0.837 < ×1.050
 *   - tile48x128r12x8w16x4k16s: i8a8-linear-m1024-n8192-k2048: ×0.721 < ×1.050
 *   - tile48x128r12x8w16x4k32: i8a8-linear-m1024-n2048-k8192: ×0.682 < ×1.050
 *   - tile48x128r12x8w16x4k32s: i8a8-linear-m1024-n2048-k8192: ×0.871 < ×1.050
 *   - tile48x128r12x8w16x4k64: i8a8-linear-m1024-n2048-k8192: ×0.573 < ×1.050
 *   - tile48x128r12x8w16x4k64s: i8a8-linear-m1024-n2048-k2048: ×0.897 < ×1.050
 *   - tile48x32r12x4w8x4k16: i8a8-linear-m1024-n2048-k8192: ×0.547 < ×1.050
 *   - tile48x32r12x4w8x4k16s: i8a8-linear-m1024-n8192-k2048: ×0.464 < ×1.050
 *   - tile48x32r12x4w8x4k32: i8a8-linear-m1024-n2048-k8192: ×0.794 < ×1.050
 *   - tile48x32r12x4w8x4k32s: i8a8-linear-m1024-n2048-k8192: ×0.579 < ×1.050
 *   - tile48x32r12x4w8x4k64: i8a8-linear-m1024-n2048-k8192: ×0.701 < ×1.050
 *   - tile48x32r12x4w8x4k64s: i8a8-linear-m1024-n2048-k8192: ×0.849 < ×1.050
 *   - tile48x64r12x4w16x4k16: i8a8-linear-m1024-n2048-k8192: ×0.767 < ×1.050
 *   - tile48x64r12x4w16x4k16s: i8a8-linear-m4096-n2048-k8192: ×0.596 < ×1.050
 *   - tile48x64r12x4w16x4k32: i8a8-linear-m1024-n2048-k8192: ×0.933 < ×1.050
 *   - tile48x64r12x4w16x4k32s: i8a8-linear-m1024-n2048-k8192: ×0.743 < ×1.050
 *   - tile48x64r12x4w16x4k64: i8a8-linear-m1024-n2048-k8192: ×0.781 < ×1.050
 *   - tile48x64r12x4w16x4k64s: i8a8-linear-m1024-n2048-k8192: ×0.964 < ×1.050
 *   - tile48x64r12x8w8x4k16: i8a8-linear-m4096-n2048-k8192: ×0.715 < ×1.050
 *   - tile48x64r12x8w8x4k16s: i8a8-linear-m4096-n2048-k2048: ×0.609 < ×1.050
 *   - tile48x64r12x8w8x4k32: i8a8-linear-m1024-n2048-k2048: ×0.684 < ×1.050
 *   - tile48x64r12x8w8x4k32s: i8a8-linear-m1024-n2048-k8192: ×0.814 < ×1.050
 *   - tile48x64r12x8w8x4k64: i8a8-linear-m1024-n2048-k8192: ×0.619 < ×1.050
 *   - tile48x64r12x8w8x4k64s: i8a8-linear-m1024-n2048-k2048: ×0.866 < ×1.050
 *   - tile64x128r16x8w16x4k16: i8a8-linear-m1024-n2048-k8192: ×0.883 < ×1.050
 *   - tile64x128r16x8w16x4k16s: i8a8-linear-m1024-n2048-k2048: ×0.788 < ×1.050
 *   - tile64x128r16x8w16x4k32: i8a8-linear-m1024-n2048-k2048: ×0.930 < ×1.050
 *   - tile64x128r16x8w16x4k32s: i8a8-linear-m7920-n3072-k14336: ×0.821 < ×1.050
 *   - tile64x128r16x8w16x4k64: i8a8-linear-m1024-n2048-k2048: ×0.815 < ×1.050
 *   - tile64x128r16x8w16x4k64s: geometric mean ×1.134 (below the adopted ×1.195)
 *   - tile64x128r4x8w16x16k16: geometric mean ×1.108 (below the adopted ×1.195)
 *   - tile64x128r4x8w16x16k32: i8a8-linear-m1024-n2048-k2048: ×1.009 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m1024-n8192-k2048: ×0.954 < ×1.050
 *   - tile64x128r8x8w16x8k16s: i8a8-linear-m1024-n2048-k8192: ×0.791 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-linear-m1024-n8192-k2048: ×0.986 < ×1.050
 *   - tile64x128r8x8w16x8k32s: i8a8-linear-m1024-n2048-k8192: ×1.028 < ×1.050
 *   - tile64x128r8x8w16x8k64: i8a8-linear-m7920-n3072-k3072: ×0.884 < ×1.050
 *   - tile64x128r8x8w16x8k64s: geometric mean ×1.181 (below the adopted ×1.195)
 *   - tile64x32r16x4w8x4k16: i8a8-linear-m1024-n2048-k2048: ×0.604 < ×1.050
 *   - tile64x32r16x4w8x4k16s: i8a8-linear-m7920-n3072-k3072: ×0.511 < ×1.050
 *   - tile64x32r16x4w8x4k32: i8a8-linear-m1024-n2048-k8192: ×0.888 < ×1.050
 *   - tile64x32r16x4w8x4k32s: i8a8-linear-m7920-n3072-k14336: ×0.768 < ×1.050
 *   - tile64x32r16x4w8x4k64: i8a8-linear-m1024-n8192-k2048: ×0.579 < ×1.050
 *   - tile64x32r16x4w8x4k64s: i8a8-linear-m1024-n2048-k2048: ×0.912 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-linear-m7920-n3072-k3072: ×0.574 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-linear-m7920-n3072-k3072: ×0.592 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-linear-m7920-n14336-k3072: ×0.612 < ×1.050
 *   - tile64x32r8x4w8x8k16s: i8a8-linear-m4096-n2048-k2048: ×0.497 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-linear-m1024-n2048-k8192: ×0.922 < ×1.050
 *   - tile64x32r8x4w8x8k32s: i8a8-linear-m1024-n2048-k8192: ×0.759 < ×1.050
 *   - tile64x32r8x4w8x8k64: i8a8-linear-m1024-n2048-k2048: ×0.719 < ×1.050
 *   - tile64x32r8x4w8x8k64s: i8a8-linear-m1024-n2048-k8192: ×0.952 < ×1.050
 *   - tile64x64r16x4w16x4k16: i8a8-linear-m1024-n2048-k2048: ×0.845 < ×1.050
 *   - tile64x64r16x4w16x4k16s: i8a8-linear-m7920-n3072-k3072: ×0.702 < ×1.050
 *   - tile64x64r16x4w16x4k32: i8a8-linear-m1024-n2048-k2048: ×0.923 < ×1.050
 *   - tile64x64r16x4w16x4k32s: i8a8-linear-m7920-n3072-k3072: ×0.391 < ×1.050
 *   - tile64x64r16x4w16x4k64: i8a8-linear-m1024-n2048-k2048: ×0.781 < ×1.050
 *   - tile64x64r16x4w16x4k64s: i8a8-linear-m1024-n2048-k8192: ×0.900 < ×1.050
 *   - tile64x64r16x8w8x4k16: i8a8-linear-m4096-n2048-k2048: ×0.800 < ×1.050
 *   - tile64x64r16x8w8x4k16s: i8a8-linear-m4096-n2048-k2048: ×0.697 < ×1.050
 *   - tile64x64r16x8w8x4k32: i8a8-linear-m1024-n2048-k2048: ×0.879 < ×1.050
 *   - tile64x64r16x8w8x4k32s: i8a8-linear-m7920-n3072-k14336: ×0.690 < ×1.050
 *   - tile64x64r16x8w8x4k64: i8a8-linear-m4096-n2048-k2048: ×0.743 < ×1.050
 *   - tile64x64r16x8w8x4k64s: i8a8-linear-m1024-n8192-k2048: ×1.009 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-linear-m1024-n2048-k8192: ×0.999 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-linear-m7920-n14336-k3072: ×0.742 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-linear-m1024-n2048-k2048: ×1.042 < ×1.050
 *   - tile64x64r4x8w8x16k32: i8a8-linear-m1024-n2048-k8192: ×1.009 < ×1.050
 *   - tile64x64r8x4w16x8k16: i8a8-linear-m7920-n14336-k3072: ×0.812 < ×1.050
 *   - tile64x64r8x4w16x8k16s: i8a8-linear-m4096-n2048-k2048: ×0.667 < ×1.050
 *   - tile64x64r8x4w16x8k32: i8a8-linear-m1024-n2048-k8192: ×1.026 < ×1.050
 *   - tile64x64r8x4w16x8k32s: i8a8-linear-m7920-n14336-k3072: ×0.698 < ×1.050
 *   - tile64x64r8x4w16x8k64: i8a8-linear-m1024-n2048-k8192: ×0.861 < ×1.050
 *   - tile64x64r8x4w16x8k64s: geometric mean ×1.113 (below the adopted ×1.195)
 *   - tile64x64r8x8w8x8k16: i8a8-linear-m4096-n2048-k2048: ×0.877 < ×1.050
 *   - tile64x64r8x8w8x8k16s: i8a8-linear-m1024-n8192-k2048: ×0.701 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-linear-m7920-n3072-k3072: ×0.975 < ×1.050
 *   - tile64x64r8x8w8x8k32s: i8a8-linear-m4096-n2048-k2048: ×1.000 < ×1.050
 *   - tile64x64r8x8w8x8k64: i8a8-linear-m1024-n2048-k2048: ×0.761 < ×1.050
 *   - tile64x64r8x8w8x8k64s: geometric mean ×1.089 (below the adopted ×1.195)
 *   - tile96x128r12x8w16x8k16: i8a8-linear-m1024-n2048-k2048: ×0.905 < ×1.050
 *   - tile96x128r12x8w16x8k16s: i8a8-linear-m1024-n2048-k8192: ×0.789 < ×1.050
 *   - tile96x128r12x8w16x8k32: i8a8-linear-m1024-n2048-k2048: ×0.819 < ×1.050
 *   - tile96x128r12x8w16x8k32s: i8a8-linear-m1024-n2048-k2048: ×0.954 < ×1.050
 *   - tile96x128r12x8w16x8k64: i8a8-linear-m1024-n2048-k2048: ×0.811 < ×1.050
 *   - tile96x128r12x8w16x8k64s: geometric mean ×1.148 (below the adopted ×1.195)
 *   - tile96x32r12x4w8x8k16: i8a8-linear-m1024-n2048-k2048: ×0.686 < ×1.050
 *   - tile96x32r12x4w8x8k16s: i8a8-linear-m1024-n2048-k8192: ×0.519 < ×1.050
 *   - tile96x32r12x4w8x8k32: i8a8-linear-m1024-n2048-k2048: ×0.871 < ×1.050
 *   - tile96x32r12x4w8x8k32s: i8a8-linear-m4096-n2048-k2048: ×0.362 < ×1.050
 *   - tile96x32r12x4w8x8k64: i8a8-linear-m1024-n2048-k8192: ×0.664 < ×1.050
 *   - tile96x32r12x4w8x8k64s: i8a8-linear-m1024-n2048-k8192: ×0.886 < ×1.050
 *   - tile96x64r12x4w16x8k16: i8a8-linear-m1024-n2048-k8192: ×0.994 < ×1.050
 *   - tile96x64r12x4w16x8k16s: i8a8-linear-m4096-n2048-k2048: ×0.732 < ×1.050
 *   - tile96x64r12x4w16x8k32: i8a8-linear-m1024-n2048-k8192: ×0.966 < ×1.050
 *   - tile96x64r12x4w16x8k32s: i8a8-linear-m1024-n2048-k8192: ×0.851 < ×1.050
 *   - tile96x64r12x4w16x8k64: i8a8-linear-m1024-n2048-k8192: ×0.786 < ×1.050
 *   - tile96x64r12x4w16x8k64s: i8a8-linear-m1024-n2048-k8192: ×1.049 < ×1.050
 *   - tile96x64r12x8w8x8k16: i8a8-linear-m1024-n2048-k8192: ×0.862 < ×1.050
 *   - tile96x64r12x8w8x8k16s: i8a8-linear-m1024-n2048-k8192: ×0.732 < ×1.050
 *   - tile96x64r12x8w8x8k32: i8a8-linear-m1024-n2048-k2048: ×0.788 < ×1.050
 *   - tile96x64r12x8w8x8k32s: i8a8-linear-m1024-n2048-k2048: ×0.901 < ×1.050
 *   - tile96x64r12x8w8x8k64: i8a8-linear-m1024-n2048-k2048: ×0.598 < ×1.050
 *   - tile96x64r12x8w8x8k64s: i8a8-linear-m1024-n2048-k2048: ×0.904 < ×1.050
 * - i8a8.attentionQk (i8a8 attention ①QK, 4 cases): keeps the default tile128x64r8x8w8x16k16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - tile128x128r8x8w16x16k16: i8a8-attention-qk-cross-m1024-n512: ×0.822 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-attention-qk-cross-m1024-n512: ×0.847 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-attention-qk-cross-m4096-n512: ×0.857 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-attention-qk-cross-m4096-n512: ×0.910 < ×1.050
 *   - tile128x64r8x4w16x16k16: i8a8-attention-qk-cross-m4096-n512: ×1.030 < ×1.050
 *   - tile128x64r8x4w16x16k32: i8a8-attention-qk-cross-m4096-n512: ×1.009 < ×1.050
 *   - tile128x64r8x8w8x16k32: i8a8-attention-qk-cross-m4096-n512: ×0.871 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-attention-qk-self-m4096-n4096: ×0.564 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-attention-qk-self-m4096-n4096: ×0.675 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-attention-qk-self-m4096-n4096: ×0.617 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-attention-qk-self-m4096-n4096: ×0.613 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-attention-qk-self-m4096-n4096: ×0.554 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-attention-qk-cross-m1024-n512: ×0.567 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-attention-qk-self-m4096-n4096: ×0.525 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-attention-qk-self-m4096-n4096: ×0.664 < ×1.050
 *   - tile32x128r4x8w16x8k16: i8a8-attention-qk-cross-m1024-n512: ×0.859 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-attention-qk-cross-m4096-n512: ×0.872 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-attention-qk-cross-m1024-n512: ×0.776 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-attention-qk-cross-m4096-n512: ×0.790 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-attention-qk-cross-m4096-n512: ×0.598 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-attention-qk-self-m4096-n4096: ×0.441 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-attention-qk-cross-m1024-n512: ×0.753 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-attention-qk-self-m4096-n4096: ×0.806 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-attention-qk-self-m4096-n4096: ×0.740 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-attention-qk-self-m4096-n4096: ×0.510 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-attention-qk-cross-m1024-n512: ×0.833 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-attention-qk-self-m4096-n4096: ×0.849 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-attention-qk-cross-m1024-n512: ×0.860 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-attention-qk-cross-m4096-n512: ×0.906 < ×1.050
 *   - tile32x64r8x8w8x4k16: i8a8-attention-qk-cross-m1024-n512: ×0.756 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-attention-qk-cross-m4096-n512: ×0.782 < ×1.050
 *   - tile64x128r4x8w16x16k16: i8a8-attention-qk-cross-m4096-n512: ×0.976 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-attention-qk-cross-m4096-n512: ×0.955 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-attention-qk-self-m4096-n4096: ×0.910 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-attention-qk-cross-m4096-n512: ×0.877 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-attention-qk-cross-m1024-n512: ×0.734 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-attention-qk-cross-m4096-n512: ×0.864 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-attention-qk-cross-m1024-n512: ×0.866 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-attention-qk-cross-m4096-n512: ×0.898 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-attention-qk-cross-m4096-n512: ×0.950 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-attention-qk-self-m4096-n4096: ×0.713 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-attention-qk-cross-m4096-n512: ×1.006 < ×1.050
 *   - tile64x64r4x8w8x16k32: i8a8-attention-qk-cross-m4096-n512: ×0.943 < ×1.050
 *   - tile64x64r8x4w16x8k16: i8a8-attention-qk-cross-m4096-n512: ×1.028 < ×1.050
 *   - tile64x64r8x4w16x8k32: i8a8-attention-qk-self-m4096-n4096: ×1.002 < ×1.050
 *   - tile64x64r8x8w8x8k16: i8a8-attention-qk-cross-m4096-n512: ×0.855 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-attention-qk-cross-m4096-n512: ×0.861 < ×1.050
 * - i8a8.attentionPv (i8a8 attention ③PV, 4 cases): adopted tile64x128r4x8w16x16k32 ×1.142 (×1.126–×1.161)
 *   - tile128x128r8x8w16x16k16: i8a8-attention-pv-cross-m4096-n512: ×0.924 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-attention-pv-cross-m4096-n512: ×0.877 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-attention-pv-self-m1024-n1024: ×0.703 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-attention-pv-self-m1024-n1024: ×0.652 < ×1.050
 *   - tile128x64r8x4w16x16k16: i8a8-attention-pv-self-m1024-n1024: ×0.961 < ×1.050
 *   - tile128x64r8x4w16x16k32: i8a8-attention-pv-self-m1024-n1024: ×0.902 < ×1.050
 *   - tile128x64r8x8w8x16k16: i8a8-attention-pv-cross-m4096-n512: ×0.820 < ×1.050
 *   - tile128x64r8x8w8x16k32: i8a8-attention-pv-self-m1024-n1024: ×0.696 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-attention-pv-cross-m1024-n512: ×0.605 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-attention-pv-self-m1024-n1024: ×0.804 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-attention-pv-self-m4096-n4096: ×0.614 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-attention-pv-self-m4096-n4096: ×0.607 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-attention-pv-self-m1024-n1024: ×0.390 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-attention-pv-self-m1024-n1024: ×0.566 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-attention-pv-self-m1024-n1024: ×0.681 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-attention-pv-self-m1024-n1024: ×0.701 < ×1.050
 *   - tile32x128r4x8w16x8k16: i8a8-attention-pv-self-m1024-n1024: ×0.884 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-attention-pv-self-m1024-n1024: ×0.944 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-attention-pv-self-m1024-n1024: ×0.857 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-attention-pv-self-m1024-n1024: ×0.807 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-attention-pv-self-m1024-n1024: ×0.434 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-attention-pv-self-m1024-n1024: ×0.636 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-attention-pv-self-m1024-n1024: ×0.536 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-attention-pv-self-m1024-n1024: ×0.567 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-attention-pv-self-m1024-n1024: ×0.595 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-attention-pv-self-m4096-n4096: ×0.984 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-attention-pv-self-m1024-n1024: ×0.784 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-attention-pv-self-m1024-n1024: ×0.782 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-attention-pv-self-m1024-n1024: ×0.733 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-attention-pv-self-m1024-n1024: ×0.813 < ×1.050
 *   - tile32x64r8x8w8x4k16: i8a8-attention-pv-self-m1024-n1024: ×0.692 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-attention-pv-cross-m4096-n512: ×0.691 < ×1.050
 *   - tile64x128r4x8w16x16k16: i8a8-attention-pv-cross-m1024-n512: ×1.049 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-attention-pv-self-m1024-n1024: ×0.952 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-attention-pv-self-m1024-n1024: ×0.504 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-attention-pv-self-m1024-n1024: ×0.742 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-attention-pv-self-m1024-n1024: ×0.621 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-attention-pv-cross-m4096-n512: ×0.584 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-attention-pv-self-m1024-n1024: ×1.043 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-attention-pv-self-m1024-n1024: ×0.771 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-attention-pv-self-m1024-n1024: ×0.915 < ×1.050
 *   - tile64x64r4x8w8x16k32: i8a8-attention-pv-self-m1024-n1024: ×0.856 < ×1.050
 *   - tile64x64r8x4w16x8k16: i8a8-attention-pv-self-m1024-n1024: ×0.953 < ×1.050
 *   - tile64x64r8x4w16x8k32: i8a8-attention-pv-self-m1024-n1024: ×0.882 < ×1.050
 *   - tile64x64r8x8w8x8k16: i8a8-attention-pv-self-m4096-n4096: ×0.782 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-attention-pv-self-m1024-n1024: ×0.627 < ×1.050
 * - i8a8.conv3d (i8a8 conv3d (w8a8 implicit GEMM), 14 cases): adopted tile64x64r8x4w16x8k16 ×1.346 (×1.179–×2.351)
 *   - tile128x128r8x8w16x16k16: i8a8-conv3d-cin1024-cout1024-k333-in3x16x16: ×0.689 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-conv3d-cin1024-cout1024-k333-in3x16x16: ×0.748 < ×1.050
 *   - tile128x32r8x4w8x16k16: geometric mean ×1.198 (below the adopted ×1.346)
 *   - tile128x32r8x4w8x16k32: i8a8-conv3d-cin256-cout12-k333-in6x128x128: ×0.993 < ×1.050
 *   - tile128x64r8x4w16x16k16: geometric mean ×1.178 (below the adopted ×1.346)
 *   - tile128x64r8x4w16x16k32: i8a8-conv3d-cin256-cout12-k333-in6x128x128: ×1.021 < ×1.050
 *   - tile128x64r8x8w8x16k32: i8a8-conv3d-cin256-cout12-k333-in6x128x128: ×0.944 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.493 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-conv3d-cin512-cout256-k333-in6x256x256: ×0.703 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.405 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-conv3d-cin512-cout256-k333-in6x256x256: ×0.682 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.532 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.719 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.444 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.677 < ×1.050
 *   - tile32x128r4x8w16x8k16: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×1.007 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.919 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-conv3d-cin1024-cout1024-k333-in4x64x64: ×0.756 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.885 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.875 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-conv3d-cin512-cout256-k333-in6x256x256: ×0.874 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-conv3d-cin1024-cout1024-k333-in4x32x32: ×0.792 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-conv3d-cin512-cout256-k333-in6x256x256: ×0.839 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-conv3d-cin512-cout256-k333-in6x256x256: ×0.973 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.971 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.985 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.902 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×1.021 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-conv3d-cin512-cout256-k333-in6x256x256: ×0.958 < ×1.050
 *   - tile32x64r8x8w8x4k16: i8a8-conv3d-cin512-cout512-k333-in6x64x64: ×0.757 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-conv3d-cin512-cout256-k333-in6x256x256: ×0.880 < ×1.050
 *   - tile64x128r4x8w16x16k16: geometric mean ×1.197 (below the adopted ×1.346)
 *   - tile64x128r4x8w16x16k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×1.018 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-conv3d-cin512-cout256-k333-in6x256x256: ×0.963 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-conv3d-cin512-cout512-k333-in6x64x64: ×0.798 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.983 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.993 < ×1.050
 *   - tile64x32r8x4w8x8k16: geometric mean ×1.227 (below the adopted ×1.346)
 *   - tile64x32r8x4w8x8k32: i8a8-conv3d-cin512-cout256-k333-in6x256x256: ×0.980 < ×1.050
 *   - tile64x64r4x4w16x16k16: geometric mean ×1.218 (below the adopted ×1.346)
 *   - tile64x64r4x4w16x16k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×1.045 < ×1.050
 *   - tile64x64r4x8w8x16k16: geometric mean ×1.265 (below the adopted ×1.346)
 *   - tile64x64r4x8w8x16k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×1.048 < ×1.050
 *   - tile64x64r8x4w16x8k32: geometric mean ×1.235 (below the adopted ×1.346)
 *   - tile64x64r8x8w8x8k16: i8a8-conv3d-cin512-cout256-k333-in6x256x256: ×0.977 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.974 < ×1.050
 *
 * Provenance converted on 2026-11-12 to one entry per op family (ADR 0117 addendum 2026-11-12), no field
 * value changed: both records have rows of every family, so every family's entry names both, and its
 * `kernels` / `caseSet` are the current runtime's keys over the family's cases.
 *
 * `i8a8.linear` (the row-count rules of ADR 0115 addendum decision 14) and the `i8a8-linear` entry come from
 * two `full` sweeps of that family alone (its 13 cases), derived on 2026-11-13 with `deriveGeometryProfile`
 * and `base` = this table: every other family's fields, verdicts and provenance entries are this table's
 * as they were (the records above). The script that ran the derivation and wrote these parts is kept,
 * untracked, in `outputs/diag/k85-buckets-2026-11-12/` (`blackwell.mjs`, `register.mjs`).
 *
 * - outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-11-12T18-35-29.json (sha256 7e31ec7fb0f233b1daf4fb267b231e5943e920a291e42b74e97b36d556ca7c8d, 2026-11-12T18:35:29.077Z)
 *   Chrome 157, Windows; timestamps unquantised; page the geometry-sweep page of apps/playground
 * - outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-11-13T02-32-10.json (sha256 a4e4b0d1adad271deb6f046749f8ae41a5d66b59229dab7a6e4f25e83d62a24d, 2026-11-13T02:32:10.083Z)
 *   Chrome 157, Windows; timestamps unquantised; page the geometry-sweep page of apps/playground
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
    { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: 512, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: Number.POSITIVE_INFINITY, geometry: { regM: 8, regN: 8, wgX: 16, wgY: 16 } },
  ],
  attention: {
    qk: { regM: 8, regN: 8, wgX: 16, wgY: 16 },
    pv: { regM: 8, regN: 8, wgX: 16, wgY: 16 },
  },
  conv2d: {
    rows64: { regM: 8, regN: 8, wgX: 16, wgY: 8 },
    rows32: { regM: 4, regN: 4, wgX: 16, wgY: 8 },
  },
  i8a8: {
    linear: [
      { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 8, tileK: 32 } },
      {
        maxRows: 512,
        geometry: { regM: 16, regN: 4, wgX: 16, wgY: 8, tileK: 64, skeleton: "vec4" },
      },
      {
        maxRows: Number.POSITIVE_INFINITY,
        geometry: { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 64, skeleton: "vec4" },
      },
    ],
    attentionQk: { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 },
    attentionPv: { regM: 4, regN: 8, wgX: 16, wgY: 16, tileK: 32 },
    conv3d: { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 },
  },
  provenance: {
    adapter: {
      vendor: "nvidia",
      architecture: "blackwell",
      device: "0x2c05",
      description: "NVIDIA GeForce RTX 5070 Ti",
    },
    families: {
      gemm: {
        sweep:
          "outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-10-28T20-44-29.json, outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-10-30T20-12-41.json",
        sha256:
          "39d203a76278ddafd386e451445ff739acc765837f5429cc9c33b8157df020e4, 5778eb62ad937e6850f62a4cae93c7d1ab8dcea59f0e9a42198fbd2fd06b257e",
        date: "2026-10-28T20:44:29.316Z, 2026-10-30T20:12:41.839Z",
        candidateSet: "full, full",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36",
        ],
        kernels: "6ac8d4833a01b2c5",
        caseSet: "e6ddd45226ab2155",
      },
      attention: {
        sweep:
          "outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-10-28T20-44-29.json, outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-10-30T20-12-41.json",
        sha256:
          "39d203a76278ddafd386e451445ff739acc765837f5429cc9c33b8157df020e4, 5778eb62ad937e6850f62a4cae93c7d1ab8dcea59f0e9a42198fbd2fd06b257e",
        date: "2026-10-28T20:44:29.316Z, 2026-10-30T20:12:41.839Z",
        candidateSet: "full, full",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36",
        ],
        kernels: "c3f4a946ccbf1df9",
        caseSet: "9ee211cf69674928",
      },
      conv2d: {
        sweep:
          "outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-10-28T20-44-29.json, outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-10-30T20-12-41.json",
        sha256:
          "39d203a76278ddafd386e451445ff739acc765837f5429cc9c33b8157df020e4, 5778eb62ad937e6850f62a4cae93c7d1ab8dcea59f0e9a42198fbd2fd06b257e",
        date: "2026-10-28T20:44:29.316Z, 2026-10-30T20:12:41.839Z",
        candidateSet: "full, full",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36",
        ],
        kernels: "f7e574d8925c82ea",
        caseSet: "02186e45558e4151",
      },
      "i8a8-linear": {
        sweep:
          "outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-11-12T18-35-29.json, outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-11-13T02-32-10.json",
        sha256:
          "7e31ec7fb0f233b1daf4fb267b231e5943e920a291e42b74e97b36d556ca7c8d, a4e4b0d1adad271deb6f046749f8ae41a5d66b59229dab7a6e4f25e83d62a24d",
        date: "2026-11-12T18:35:29.077Z, 2026-11-13T02:32:10.083Z",
        candidateSet: "full, full",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36",
        ],
        kernels: "0bd149287a89377f",
        caseSet: "3d32dc6f10559af2",
      },
      "i8a8-attention": {
        sweep:
          "outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-10-28T20-44-29.json, outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-10-30T20-12-41.json",
        sha256:
          "39d203a76278ddafd386e451445ff739acc765837f5429cc9c33b8157df020e4, 5778eb62ad937e6850f62a4cae93c7d1ab8dcea59f0e9a42198fbd2fd06b257e",
        date: "2026-10-28T20:44:29.316Z, 2026-10-30T20:12:41.839Z",
        candidateSet: "full, full",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36",
        ],
        kernels: "0d3e6b59734db2c9",
        caseSet: "8ab1a86b13087640",
      },
      "i8a8-conv3d": {
        sweep:
          "outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-10-28T20-44-29.json, outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-10-30T20-12-41.json",
        sha256:
          "39d203a76278ddafd386e451445ff739acc765837f5429cc9c33b8157df020e4, 5778eb62ad937e6850f62a4cae93c7d1ab8dcea59f0e9a42198fbd2fd06b257e",
        date: "2026-10-28T20:44:29.316Z, 2026-10-30T20:12:41.839Z",
        candidateSet: "full, full",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36",
        ],
        kernels: "c91b3368411e71d9",
        caseSet: "27ce9da45b359b14",
      },
    },
  },
};
