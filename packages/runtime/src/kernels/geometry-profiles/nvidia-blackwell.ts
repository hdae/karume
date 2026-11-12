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
 * WebGPU backend not recorded; timestamps unquantised in all 2 records):
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
 * - i8a8.linear ≤ 256 (i8a8 linear row count ≤ 256, 0 cases): keeps the default tile128x64r8x8w8x16k16 (the sweeps have no case of this class)
 * - i8a8.linear 257〜512 (i8a8 linear row count 257〜512, 0 cases): keeps the default tile128x64r8x8w8x16k16 (the sweeps have no case of this class)
 * - i8a8.linear > 512 (i8a8 linear row count > 512, 6 cases): adopted tile128x64r8x4w16x16k16 ×1.148 (×1.107–×1.185)
 *   - tile128x128r8x8w16x16k16: i8a8-linear-m1024-n8192-k2048: ×0.921 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-linear-m1024-n8192-k2048: ×0.939 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-linear-m1024-n2048-k2048: ×0.691 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-linear-m1024-n2048-k8192: ×0.961 < ×1.050
 *   - tile128x64r8x4w16x16k32: i8a8-linear-m1024-n2048-k8192: ×0.994 < ×1.050
 *   - tile128x64r8x8w8x16k32: i8a8-linear-m4096-n2048-k2048: ×1.003 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-linear-m1024-n2048-k8192: ×0.413 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-linear-m1024-n2048-k8192: ×0.700 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-linear-m1024-n2048-k8192: ×0.398 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-linear-m4096-n2048-k8192: ×0.652 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-linear-m4096-n8192-k2048: ×0.339 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-linear-m1024-n2048-k8192: ×0.497 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-linear-m1024-n2048-k8192: ×0.397 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-linear-m1024-n2048-k2048: ×0.671 < ×1.050
 *   - tile32x128r4x8w16x8k16: i8a8-linear-m1024-n2048-k2048: ×0.688 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-linear-m1024-n2048-k8192: ×0.921 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-linear-m4096-n2048-k8192: ×0.744 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-linear-m1024-n2048-k2048: ×0.768 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-linear-m1024-n2048-k8192: ×0.467 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-linear-m1024-n2048-k8192: ×0.341 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-linear-m1024-n2048-k8192: ×0.463 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-linear-m1024-n2048-k2048: ×0.790 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-linear-m1024-n2048-k8192: ×0.586 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-linear-m4096-n8192-k2048: ×0.923 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-linear-m1024-n2048-k2048: ×0.611 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-linear-m1024-n2048-k2048: ×0.883 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-linear-m1024-n2048-k8192: ×0.613 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-linear-m1024-n2048-k8192: ×0.916 < ×1.050
 *   - tile32x64r8x8w8x4k16: i8a8-linear-m1024-n2048-k2048: ×0.626 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-linear-m4096-n2048-k2048: ×0.854 < ×1.050
 *   - tile64x128r4x8w16x16k16: geometric mean ×1.112 (below the adopted ×1.148)
 *   - tile64x128r4x8w16x16k32: i8a8-linear-m1024-n2048-k2048: ×1.011 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m4096-n2048-k2048: ×0.964 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-linear-m1024-n8192-k2048: ×0.995 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-linear-m1024-n2048-k2048: ×0.589 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-linear-m4096-n8192-k2048: ×0.618 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-linear-m1024-n2048-k2048: ×0.605 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-linear-m1024-n2048-k8192: ×0.922 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-linear-m1024-n2048-k2048: ×0.989 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-linear-m1024-n2048-k8192: ×0.762 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-linear-m1024-n2048-k2048: ×1.042 < ×1.050
 *   - tile64x64r4x8w8x16k32: i8a8-linear-m1024-n2048-k2048: ×1.014 < ×1.050
 *   - tile64x64r8x4w16x8k16: i8a8-linear-m1024-n2048-k2048: ×0.823 < ×1.050
 *   - tile64x64r8x4w16x8k32: i8a8-linear-m1024-n2048-k8192: ×1.031 < ×1.050
 *   - tile64x64r8x8w8x8k16: i8a8-linear-m1024-n2048-k2048: ×0.885 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-linear-m4096-n2048-k2048: ×0.987 < ×1.050
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
 * `i8a8.linear` re-derived on 2026-11-12 when the field became row-count rules and the `i8a8-linear`
 * family got new cases and the bounds 256 / 512 / ∞ (ADR 0115 addendum decision 14): `deriveGeometryProfile`
 * (no base) over the records above, read through a script that lifts them in memory to the current record
 * format (`karume-geometry-sweep/3` with the current runtime's `caseSets` / `defaultKernels` for the
 * families of their rows; the rows untouched).
 * That rests on the rows having been measured on the same kernels: each record's `candidateKernels`
 * re-derives equal with the current runtime.
 * The records hold i8a8 linear cases at M = 1024 / 4096 only, so the buckets ≤ 256 and 257〜512 keep the
 * default geometry — slower for the small-M i8a8 linears (Irodori's M = 185, for one; per the K-85 evaluation
 * records named in ADR 0115 addendum decision 14) until a sweep
 * of the family is registered; outputs do not change (the geometries are bit-identical by contract). The
 * bucket > 512 keeps the geometry the field held. Every other field and provenance entry is unchanged;
 * the `i8a8-linear` entry has the current `kernels` / `caseSet`.
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
      { maxRows: 256, geometry: { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 } },
      { maxRows: 512, geometry: { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 } },
      {
        maxRows: Number.POSITIVE_INFINITY,
        geometry: { regM: 8, regN: 4, wgX: 16, wgY: 16, tileK: 16 },
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
          "outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-10-28T20-44-29.json, outputs/bench-browser/geometry-sweep_nvidia-geforce-rtx-5070-ti_full_2026-10-30T20-12-41.json",
        sha256:
          "39d203a76278ddafd386e451445ff739acc765837f5429cc9c33b8157df020e4, 5778eb62ad937e6850f62a4cae93c7d1ab8dcea59f0e9a42198fbd2fd06b257e",
        date: "2026-10-28T20:44:29.316Z, 2026-10-30T20:12:41.839Z",
        candidateSet: "full, full",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36",
        ],
        kernels: "0a459a91e7c502c1",
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
