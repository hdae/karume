/**
 * Geometry profile `apple-m5` (**generated — do not edit by hand**).
 *
 * A static table of tile geometries for the adapter match: vendor apple, architecture metal-3, description Apple M5.
 * The runtime only selects the table by the adapter's match; nothing is measured at run time.
 * To change a value, take a sweep with the workbench's `geometry-sweep` entry and regenerate the
 * table with `node tools/geometry-profiles/generate.ts apple-m5 --records=<path>[,<path>…]`.
 *
 * Derived in this generation: gemm, attention, conv2d, i8a8-linear, i8a8-attention, i8a8-conv3d.
 *
 * - outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json (sha256 afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad, 2026-11-23T15:06:51.706Z)
 *   workbench; Chrome 158 on macOS; backend metal (declared); timestamps not quantised; checkout 8c9364d8481f83e45f18cc29d813ffd3350d1841
 *
 * Adoption: a geometry replaces the default of a field only when, on every case of the field,
 * its output is identical to the default geometry's and it is at least ×1.050 faster;
 * among those, the one with the largest geometric mean of the ratios is taken, else the default
 * stays.
 * A case whose default re-measurement ratio lies outside 0.9–1.1 (or whose re-measurement failed
 * or is missing) is dropped from that record's ratios, as is an observation whose timestamp
 * rounding could move its ratio by more than 1%; output mismatches and failures still count.
 *
 * - gemmRows ≤ 16 (linear / matmul / bmm row count ≤ 16, 4 cases): keeps the default reg16x16r1x4w4 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x16r8x4w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x16r8x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x32r8x8w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x32r8x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x64r8x4w16: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x64r8x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x64r8x4w16:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x64r8x8w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x64r8x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x64r8x8w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x128r4x8w16: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x16r1x4w4:tas-tb-un: linear-m16-n3072-k1024: ×0.981 < ×1.050
 *   - reg16x16r1x4w4:tb-un: linear-m16-n3072-k1024: ×1.019 < ×1.050
 *   - reg16x16r2x4w4: bmm-b16-m16-n64-k128: ×0.826 < ×1.050
 *   - reg16x16r2x4w4:tas-tb-un: bmm-b16-m16-n64-k64: ×0.889 < ×1.050
 *   - reg16x16r2x4w4:tb-un: bmm-b16-m16-n64-k64: ×0.879 < ×1.050
 *   - reg16x16r4x4w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x16r4x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x16r4x4w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x32r1x8w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x32r1x8w4:tas-tb-un: bmm-b16-m16-n64-k64: ×0.964 < ×1.050
 *   - reg16x32r1x8w4:tb-un: bmm-b16-m16-n64-k64: ×0.955 < ×1.050
 *   - reg16x32r2x4w8: bmm-b16-m16-n64-k128: ×0.886 < ×1.050
 *   - reg16x32r2x4w8:tas-tb-un: bmm-b16-m16-n64-k64: ×1.007 < ×1.050
 *   - reg16x32r2x4w8:tb-un: bmm-b16-m16-n64-k64: ×0.977 < ×1.050
 *   - reg16x32r2x8w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x32r2x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x32r2x8w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x32r4x4w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x32r4x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x32r4x4w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x32r4x8w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x32r4x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x32r4x8w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x64r2x8w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x64r2x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x64r2x8w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x64r4x4w16: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x64r4x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x64r4x4w16:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x64r4x8w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x64r4x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x64r4x8w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x128r4x8w16: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x128r8x8w16: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x16r2x4w4: screened out by the quick look in linear-m16-n3072-k1024 (no timing to compare)
 *   - reg32x16r2x4w4:tas-tb-un: linear-m16-n3072-k1024: ×0.851 < ×1.050
 *   - reg32x16r2x4w4:tb-un: linear-m16-n3072-k1024: ×0.819 < ×1.050
 *   - reg32x16r4x4w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x16r4x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x16r4x4w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x16r8x4w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x16r8x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r2x4w8: bmm-b16-m16-n64-k128: ×0.810 < ×1.050
 *   - reg32x32r2x4w8:tas-tb-un: bmm-b16-m16-n64-k128: ×0.860 < ×1.050
 *   - reg32x32r2x4w8:tb-un: bmm-b16-m16-n64-k128: ×0.843 < ×1.050
 *   - reg32x32r2x8w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r2x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r2x8w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r4x4w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r4x4w8:tas-tb-un: bmm-b16-m16-n64-k64: ×0.816 < ×1.050
 *   - reg32x32r4x4w8:tb-un: bmm-b16-m16-n64-k128: ×0.803 < ×1.050
 *   - reg32x32r4x8w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r4x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r4x8w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r8x4w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r8x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r8x4w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r8x8w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r8x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x64r2x8w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x64r2x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x64r2x8w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x64r4x4w16: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x64r4x4w16:tas-tb-un: bmm-b16-m16-n64-k128: ×0.829 < ×1.050
 *   - reg32x64r4x4w16:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x64r4x8w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x64r4x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x64r4x8w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x64r8x4w16: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x64r8x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x64r8x4w16:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x64r8x8w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x64r8x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x64r8x8w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg4x16r1x4w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg4x16r1x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg4x16r1x4w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg4x32r1x8w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg4x32r1x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg4x32r1x8w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x128r4x8w16: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x128r8x8w16: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x16r4x4w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x16r4x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x16r4x4w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x16r8x4w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x16r8x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x32r4x8w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x32r4x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x32r4x8w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x32r8x4w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x32r8x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x32r8x4w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x32r8x8w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x32r8x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r4x8w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r4x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r4x8w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r8x4w16: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r8x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r8x4w16:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r8x8w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r8x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r8x8w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg8x16r1x4w4: screened out by the quick look in linear-m16-n3072-k1024 (no timing to compare)
 *   - reg8x16r1x4w4:tas-tb-un: screened out by the quick look in linear-m16-n3072-k1024 (no timing to compare)
 *   - reg8x16r1x4w4:tb-un: screened out by the quick look in linear-m16-n3072-k1024 (no timing to compare)
 *   - reg8x16r2x4w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg8x16r2x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg8x16r2x4w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg8x32r1x8w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg8x32r1x8w4:tas-tb-un: screened out by the quick look in linear-m16-n3072-k1024 (no timing to compare)
 *   - reg8x32r1x8w4:tb-un: screened out by the quick look in linear-m16-n3072-k1024 (no timing to compare)
 *   - reg8x32r2x4w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg8x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m16-n3072-k1024 (no timing to compare)
 *   - reg8x32r2x4w8:tb-un: screened out by the quick look in linear-m16-n3072-k1024 (no timing to compare)
 *   - reg8x32r2x8w4: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg8x32r2x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg8x32r2x8w4:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg8x64r2x8w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg8x64r2x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg8x64r2x8w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 * - gemmRows 17〜32 (linear / matmul / bmm row count 17〜32, 4 cases): adopted reg32x64r4x4w16:tas-tb-un ×1.317 (×1.067–×1.656)
 *   - reg128x128r8x8w16: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x16r8x4w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x16r8x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x32r8x8w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x32r8x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x64r8x4w16: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x64r8x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x64r8x4w16:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x64r8x8w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x64r8x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x64r8x8w8:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x128r4x8w16: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x16r1x4w4:tas-tb-un: bmm-b16-m32-n64-k128: ×0.966 < ×1.050
 *   - reg16x16r1x4w4:tb-un: bmm-b16-m32-n64-k128: ×0.990 < ×1.050
 *   - reg16x16r2x4w4: bmm-b16-m32-n64-k128: ×0.984 < ×1.050
 *   - reg16x16r2x4w4:tas-tb-un: linear-m32-n3072-k1024: ×1.039 < ×1.050
 *   - reg16x16r2x4w4:tb-un: bmm-b16-m32-n64-k128: ×1.014 < ×1.050
 *   - reg16x16r4x4w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x16r4x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x16r4x4w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x32r1x8w4: bmm-b16-m32-n64-k128: ×0.857 < ×1.050
 *   - reg16x32r1x8w4:tas-tb-un: linear-m32-n3072-k1024: ×0.949 < ×1.050
 *   - reg16x32r1x8w4:tb-un: linear-m32-n3072-k1024: ×0.972 < ×1.050
 *   - reg16x32r2x4w8: bmm-b16-m32-n64-k128: ×1.044 < ×1.050
 *   - reg16x32r2x4w8:tas-tb-un: geometric mean ×1.174 (below the adopted ×1.317)
 *   - reg16x32r2x4w8:tb-un: geometric mean ×1.179 (below the adopted ×1.317)
 *   - reg16x32r2x8w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x32r2x8w4:tas-tb-un: bmm-b16-m32-n64-k128: ×0.843 < ×1.050
 *   - reg16x32r2x8w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x32r4x4w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x32r4x4w8:tas-tb-un: bmm-b16-m32-n64-k128: ×0.896 < ×1.050
 *   - reg16x32r4x4w8:tb-un: bmm-b16-m32-n64-k128: ×0.840 < ×1.050
 *   - reg16x32r4x8w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x32r4x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x32r4x8w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x64r2x8w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x64r2x8w8:tas-tb-un: bmm-b16-m32-n64-k128: ×0.895 < ×1.050
 *   - reg16x64r2x8w8:tb-un: bmm-b16-m32-n64-k128: ×0.920 < ×1.050
 *   - reg16x64r4x4w16: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x64r4x4w16:tas-tb-un: bmm-b16-m32-n64-k64: ×0.934 < ×1.050
 *   - reg16x64r4x4w16:tb-un: bmm-b16-m32-n64-k64: ×0.940 < ×1.050
 *   - reg16x64r4x8w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x64r4x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x64r4x8w8:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x128r4x8w16: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x128r8x8w16: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x16r2x4w4: bmm-b16-m32-n64-k64: ×0.993 < ×1.050
 *   - reg32x16r2x4w4:tas-tb-un: geometric mean ×1.252 (below the adopted ×1.317)
 *   - reg32x16r2x4w4:tb-un: geometric mean ×1.237 (below the adopted ×1.317)
 *   - reg32x16r4x4w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x16r4x4w4:tas-tb-un: bmm-b16-m32-n64-k128: ×0.868 < ×1.050
 *   - reg32x16r4x4w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x16r8x4w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x16r8x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x32r2x4w8: geometric mean ×1.260 (below the adopted ×1.317)
 *   - reg32x32r2x4w8:tas-tb-un: geometric mean ×1.297 (below the adopted ×1.317)
 *   - reg32x32r2x4w8:tb-un: geometric mean ×1.270 (below the adopted ×1.317)
 *   - reg32x32r2x8w4: bmm-b16-m32-n64-k128: ×0.806 < ×1.050
 *   - reg32x32r2x8w4:tas-tb-un: bmm-b16-m32-n64-k128: ×0.970 < ×1.050
 *   - reg32x32r2x8w4:tb-un: bmm-b16-m32-n64-k128: ×0.972 < ×1.050
 *   - reg32x32r4x4w8: bmm-b16-m32-n64-k128: ×0.804 < ×1.050
 *   - reg32x32r4x4w8:tas-tb-un: bmm-b16-m32-n64-k128: ×1.025 < ×1.050
 *   - reg32x32r4x4w8:tb-un: bmm-b16-m32-n64-k128: ×1.015 < ×1.050
 *   - reg32x32r4x8w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x32r4x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x32r4x8w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x32r8x4w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x32r8x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x32r8x4w8:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x32r8x8w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x32r8x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x64r2x8w8: bmm-b16-m32-n64-k128: ×0.845 < ×1.050
 *   - reg32x64r2x8w8:tas-tb-un: bmm-b16-m32-n64-k128: ×1.040 < ×1.050
 *   - reg32x64r2x8w8:tb-un: bmm-b16-m32-n64-k128: ×1.022 < ×1.050
 *   - reg32x64r4x4w16: bmm-b16-m32-n64-k128: ×0.853 < ×1.050
 *   - reg32x64r4x4w16:tb-un: geometric mean ×1.278 (below the adopted ×1.317)
 *   - reg32x64r4x8w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x64r4x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x64r4x8w8:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x64r8x4w16: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x64r8x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x64r8x4w16:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x64r8x8w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x64r8x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x64r8x8w8:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg4x16r1x4w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg4x16r1x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg4x16r1x4w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg4x32r1x8w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg4x32r1x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg4x32r1x8w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x128r4x8w16: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x128r8x8w16: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x16r4x4w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x16r4x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k64 (no timing to compare)
 *   - reg64x16r4x4w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x16r8x4w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x16r8x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x32r4x8w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x32r4x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x32r4x8w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x32r8x4w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x32r8x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x32r8x4w8:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x32r8x8w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x32r8x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r4x8w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r4x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r4x8w8:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r8x4w16: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r8x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r8x4w16:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r8x8w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r8x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r8x8w8:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg8x16r1x4w4: screened out by the quick look in linear-m32-n3072-k1024 (no timing to compare)
 *   - reg8x16r1x4w4:tas-tb-un: screened out by the quick look in linear-m32-n3072-k1024 (no timing to compare)
 *   - reg8x16r1x4w4:tb-un: screened out by the quick look in linear-m32-n3072-k1024 (no timing to compare)
 *   - reg8x16r2x4w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg8x16r2x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg8x16r2x4w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg8x32r1x8w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg8x32r1x8w4:tas-tb-un: screened out by the quick look in linear-m32-n3072-k1024 (no timing to compare)
 *   - reg8x32r1x8w4:tb-un: screened out by the quick look in linear-m32-n3072-k1024 (no timing to compare)
 *   - reg8x32r2x4w8: screened out by the quick look in linear-m32-n3072-k1024 (no timing to compare)
 *   - reg8x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m32-n3072-k1024 (no timing to compare)
 *   - reg8x32r2x4w8:tb-un: screened out by the quick look in linear-m32-n3072-k1024 (no timing to compare)
 *   - reg8x32r2x8w4: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg8x32r2x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg8x32r2x8w4:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg8x64r2x8w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg8x64r2x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg8x64r2x8w8:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 * - gemmRows 33〜64 (linear / matmul / bmm row count 33〜64, 4 cases): adopted reg64x32r4x4w8:tas-tb-un ×1.688 (×1.442–×2.104)
 *   - reg128x128r8x8w16: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x16r8x4w4: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x16r8x4w4:tas-tb-un: bmm-b16-m64-n64-k128: ×0.815 < ×1.050
 *   - reg128x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x32r8x4w8: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8:tas-tb-un: bmm-b16-m64-n64-k128: ×0.824 < ×1.050
 *   - reg128x32r8x4w8:tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg128x32r8x8w4: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x32r8x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x64r8x4w16: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x64r8x4w16:tas-tb-un: bmm-b16-m64-n64-k128: ×0.836 < ×1.050
 *   - reg128x64r8x4w16:tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x64r8x8w8: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x64r8x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x64r8x8w8:tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg16x128r4x8w16: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg16x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg16x128r4x8w16:tb-un: bmm-b16-m64-n64-k128: ×0.802 < ×1.050
 *   - reg16x16r1x4w4:tas-tb-un: linear-m64-n3072-k1024: ×0.983 < ×1.050
 *   - reg16x16r1x4w4:tb-un: linear-m64-n3072-k1024: ×0.998 < ×1.050
 *   - reg16x16r2x4w4: geometric mean ×1.188 (below the adopted ×1.688)
 *   - reg16x16r2x4w4:tas-tb-un: geometric mean ×1.219 (below the adopted ×1.688)
 *   - reg16x16r2x4w4:tb-un: geometric mean ×1.189 (below the adopted ×1.688)
 *   - reg16x16r4x4w4: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg16x16r4x4w4:tas-tb-un: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare)
 *   - reg16x16r4x4w4:tb-un: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare)
 *   - reg16x32r1x8w4: linear-m64-n3072-k1024: ×1.002 < ×1.050
 *   - reg16x32r1x8w4:tas-tb-un: linear-m64-n3072-k1024: ×1.011 < ×1.050
 *   - reg16x32r1x8w4:tb-un: linear-m64-n3072-k1024: ×1.029 < ×1.050
 *   - reg16x32r2x4w8: geometric mean ×1.277 (below the adopted ×1.688)
 *   - reg16x32r2x4w8:tas-tb-un: geometric mean ×1.277 (below the adopted ×1.688)
 *   - reg16x32r2x4w8:tb-un: geometric mean ×1.290 (below the adopted ×1.688)
 *   - reg16x32r2x8w4: linear-m64-n3072-k1024: ×0.902 < ×1.050
 *   - reg16x32r2x8w4:tas-tb-un: linear-m64-n3072-k1024: ×0.935 < ×1.050
 *   - reg16x32r2x8w4:tb-un: linear-m64-n3072-k1024: ×0.926 < ×1.050
 *   - reg16x32r4x4w8: linear-m64-n3072-k1024: ×0.919 < ×1.050
 *   - reg16x32r4x4w8:tas-tb-un: linear-m64-n3072-k1024: ×0.951 < ×1.050
 *   - reg16x32r4x4w8:tb-un: linear-m64-n3072-k1024: ×0.923 < ×1.050
 *   - reg16x32r4x8w4: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg16x32r4x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg16x32r4x8w4:tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg16x64r2x8w8: linear-m64-n3072-k1024: ×0.906 < ×1.050
 *   - reg16x64r2x8w8:tas-tb-un: linear-m64-n3072-k1024: ×0.909 < ×1.050
 *   - reg16x64r2x8w8:tb-un: linear-m64-n3072-k1024: ×0.916 < ×1.050
 *   - reg16x64r4x4w16: linear-m64-n3072-k1024: ×0.913 < ×1.050
 *   - reg16x64r4x4w16:tas-tb-un: linear-m64-n3072-k1024: ×0.934 < ×1.050
 *   - reg16x64r4x4w16:tb-un: linear-m64-n3072-k1024: ×0.935 < ×1.050
 *   - reg16x64r4x8w8: bmm-b16-m64-n64-k128: ×0.908 < ×1.050
 *   - reg16x64r4x8w8:tas-tb-un: bmm-b16-m64-n64-k128: ×0.959 < ×1.050
 *   - reg16x64r4x8w8:tb-un: bmm-b16-m64-n64-k128: ×0.922 < ×1.050
 *   - reg32x128r4x8w16: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg32x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg32x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg32x128r8x8w16: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg32x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg32x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg32x16r2x4w4: geometric mean ×1.338 (below the adopted ×1.688)
 *   - reg32x16r2x4w4:tas-tb-un: geometric mean ×1.408 (below the adopted ×1.688)
 *   - reg32x16r2x4w4:tb-un: geometric mean ×1.358 (below the adopted ×1.688)
 *   - reg32x16r4x4w4: geometric mean ×1.243 (below the adopted ×1.688)
 *   - reg32x16r4x4w4:tas-tb-un: geometric mean ×1.418 (below the adopted ×1.688)
 *   - reg32x16r4x4w4:tb-un: geometric mean ×1.290 (below the adopted ×1.688)
 *   - reg32x16r8x4w4: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg32x16r8x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg32x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg32x32r2x4w8: geometric mean ×1.434 (below the adopted ×1.688)
 *   - reg32x32r2x4w8:tas-tb-un: geometric mean ×1.458 (below the adopted ×1.688)
 *   - reg32x32r2x4w8:tb-un: geometric mean ×1.444 (below the adopted ×1.688)
 *   - reg32x32r2x8w4: geometric mean ×1.402 (below the adopted ×1.688)
 *   - reg32x32r2x8w4:tas-tb-un: geometric mean ×1.505 (below the adopted ×1.688)
 *   - reg32x32r2x8w4:tb-un: geometric mean ×1.505 (below the adopted ×1.688)
 *   - reg32x32r4x4w8: geometric mean ×1.439 (below the adopted ×1.688)
 *   - reg32x32r4x4w8:tas-tb-un: geometric mean ×1.587 (below the adopted ×1.688)
 *   - reg32x32r4x4w8:tb-un: geometric mean ×1.569 (below the adopted ×1.688)
 *   - reg32x32r4x8w4: bmm-b16-m64-n64-k128: ×0.976 < ×1.050
 *   - reg32x32r4x8w4:tas-tb-un: bmm-b16-m64-n64-k128: ×1.046 < ×1.050
 *   - reg32x32r4x8w4:tb-un: bmm-b16-m64-n64-k128: ×0.960 < ×1.050
 *   - reg32x32r8x4w8: bmm-b16-m64-n64-k128: ×1.003 < ×1.050
 *   - reg32x32r8x4w8:tas-tb-un: bmm-b16-m64-n64-k128: ×1.022 < ×1.050
 *   - reg32x32r8x4w8:tb-un: bmm-b16-m64-n64-k128: ×0.930 < ×1.050
 *   - reg32x32r8x8w4: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg32x32r8x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg32x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg32x64r2x8w8: geometric mean ×1.383 (below the adopted ×1.688)
 *   - reg32x64r2x8w8:tas-tb-un: geometric mean ×1.445 (below the adopted ×1.688)
 *   - reg32x64r2x8w8:tb-un: geometric mean ×1.450 (below the adopted ×1.688)
 *   - reg32x64r4x4w16: geometric mean ×1.437 (below the adopted ×1.688)
 *   - reg32x64r4x4w16:tas-tb-un: geometric mean ×1.589 (below the adopted ×1.688)
 *   - reg32x64r4x4w16:tb-un: geometric mean ×1.541 (below the adopted ×1.688)
 *   - reg32x64r4x8w8: geometric mean ×1.381 (below the adopted ×1.688)
 *   - reg32x64r4x8w8:tas-tb-un: geometric mean ×1.488 (below the adopted ×1.688)
 *   - reg32x64r4x8w8:tb-un: geometric mean ×1.462 (below the adopted ×1.688)
 *   - reg32x64r8x4w16: geometric mean ×1.371 (below the adopted ×1.688)
 *   - reg32x64r8x4w16:tas-tb-un: geometric mean ×1.519 (below the adopted ×1.688)
 *   - reg32x64r8x4w16:tb-un: bmm-b16-m64-n64-k128: ×1.027 < ×1.050
 *   - reg32x64r8x8w8: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg32x64r8x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg32x64r8x8w8:tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg4x16r1x4w4: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg4x16r1x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg4x16r1x4w4:tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg4x32r1x8w4: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg4x32r1x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg4x32r1x8w4:tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg64x128r4x8w16: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x128r8x8w16: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x16r4x4w4: geometric mean ×1.385 (below the adopted ×1.688)
 *   - reg64x16r4x4w4:tas-tb-un: geometric mean ×1.546 (below the adopted ×1.688)
 *   - reg64x16r4x4w4:tb-un: geometric mean ×1.429 (below the adopted ×1.688)
 *   - reg64x16r8x4w4: bmm-b16-m64-n64-k128: ×0.871 < ×1.050
 *   - reg64x16r8x4w4:tas-tb-un: bmm-b16-m64-n64-k128: ×0.910 < ×1.050
 *   - reg64x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8: geometric mean ×1.532 (below the adopted ×1.688)
 *   - reg64x32r4x4w8:tb-un: geometric mean ×1.603 (below the adopted ×1.688)
 *   - reg64x32r4x8w4: bmm-b16-m64-n64-k128: ×1.024 < ×1.050
 *   - reg64x32r4x8w4:tas-tb-un: geometric mean ×1.549 (below the adopted ×1.688)
 *   - reg64x32r4x8w4:tb-un: geometric mean ×1.512 (below the adopted ×1.688)
 *   - reg64x32r8x4w8: geometric mean ×1.426 (below the adopted ×1.688)
 *   - reg64x32r8x4w8:tas-tb-un: geometric mean ×1.557 (below the adopted ×1.688)
 *   - reg64x32r8x4w8:tb-un: bmm-b16-m64-n64-k128: ×0.998 < ×1.050
 *   - reg64x32r8x8w4: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x32r8x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16: geometric mean ×1.519 (below the adopted ×1.688)
 *   - reg64x64r4x4w16:tas-tb-un: geometric mean ×1.677 (below the adopted ×1.688)
 *   - reg64x64r4x4w16:tb-un: geometric mean ×1.607 (below the adopted ×1.688)
 *   - reg64x64r4x8w8: geometric mean ×1.498 (below the adopted ×1.688)
 *   - reg64x64r4x8w8:tas-tb-un: geometric mean ×1.615 (below the adopted ×1.688)
 *   - reg64x64r4x8w8:tb-un: geometric mean ×1.556 (below the adopted ×1.688)
 *   - reg64x64r8x4w16: geometric mean ×1.457 (below the adopted ×1.688)
 *   - reg64x64r8x4w16:tas-tb-un: geometric mean ×1.638 (below the adopted ×1.688)
 *   - reg64x64r8x4w16:tb-un: geometric mean ×1.494 (below the adopted ×1.688)
 *   - reg64x64r8x8w8: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x64r8x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x64r8x8w8:tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg8x16r1x4w4: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare)
 *   - reg8x16r1x4w4:tas-tb-un: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare)
 *   - reg8x16r1x4w4:tb-un: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare)
 *   - reg8x16r2x4w4: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg8x16r2x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg8x16r2x4w4:tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg8x32r1x8w4: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare)
 *   - reg8x32r1x8w4:tas-tb-un: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare)
 *   - reg8x32r1x8w4:tb-un: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare)
 *   - reg8x32r2x4w8: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare)
 *   - reg8x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare)
 *   - reg8x32r2x4w8:tb-un: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare)
 *   - reg8x32r2x8w4: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg8x32r2x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg8x32r2x8w4:tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg8x64r2x8w8: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare)
 *   - reg8x64r2x8w8:tas-tb-un: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare)
 *   - reg8x64r2x8w8:tb-un: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare)
 * - gemmRows 65〜128 (linear / matmul / bmm row count 65〜128, 4 cases): adopted reg64x32r4x4w8:tas-tb-un ×1.110 (×1.077–×1.138)
 *   - reg128x128r8x8w16: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg128x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg128x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg128x16r8x4w4: screened out by the quick look in matmul-m128-n2048-k1024 (no timing to compare)
 *   - reg128x16r8x4w4:tas-tb-un: matmul-m128-n2048-k1024: ×0.893 < ×1.050
 *   - reg128x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8: bmm-b16-m128-n64-k128: ×0.860 < ×1.050
 *   - reg128x32r8x4w8:tas-tb-un: bmm-b16-m128-n64-k64: ×0.961 < ×1.050
 *   - reg128x32r8x4w8:tb-un: bmm-b16-m128-n64-k128: ×0.883 < ×1.050
 *   - reg128x32r8x8w4: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg128x32r8x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg128x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg128x64r8x4w16: bmm-b16-m128-n64-k128: ×0.872 < ×1.050
 *   - reg128x64r8x4w16:tas-tb-un: bmm-b16-m128-n64-k64: ×0.960 < ×1.050
 *   - reg128x64r8x4w16:tb-un: matmul-m128-n2048-k1024: ×0.869 < ×1.050
 *   - reg128x64r8x8w8: bmm-b16-m128-n64-k64: ×0.853 < ×1.050
 *   - reg128x64r8x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg128x64r8x8w8:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg16x128r4x8w16: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg16x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg16x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg16x16r1x4w4: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg16x16r1x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg16x16r1x4w4:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg16x16r2x4w4: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x16r2x4w4:tas-tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x16r2x4w4:tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x16r4x4w4: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg16x16r4x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg16x16r4x4w4:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg16x32r1x8w4: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg16x32r1x8w4:tas-tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x32r1x8w4:tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x32r2x4w8: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x32r2x4w8:tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x32r2x8w4: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x32r2x8w4:tas-tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x32r2x8w4:tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x32r4x4w8: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x32r4x4w8:tas-tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x32r4x4w8:tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x32r4x8w4: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg16x32r4x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg16x32r4x8w4:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg16x64r2x8w8: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x64r2x8w8:tas-tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x64r2x8w8:tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x64r4x4w16: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x64r4x4w16:tas-tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x64r4x4w16:tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x64r4x8w8: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x64r4x8w8:tas-tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x64r4x8w8:tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x128r4x8w16: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg32x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg32x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg32x128r8x8w16: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg32x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg32x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg32x16r2x4w4: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x16r2x4w4:tas-tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x16r2x4w4:tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x16r4x4w4: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x16r4x4w4:tas-tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x16r4x4w4:tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x16r8x4w4: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg32x16r8x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg32x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg32x32r2x4w8: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x32r2x4w8:tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x32r2x8w4: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x32r2x8w4:tas-tb-un: linear-m128-n2048-k1024: ×0.822 < ×1.050
 *   - reg32x32r2x8w4:tb-un: linear-m128-n2048-k1024: ×0.806 < ×1.050
 *   - reg32x32r4x4w8: linear-m128-n2048-k1024: ×0.788 < ×1.050
 *   - reg32x32r4x4w8:tas-tb-un: linear-m128-n2048-k1024: ×0.860 < ×1.050
 *   - reg32x32r4x4w8:tb-un: linear-m128-n2048-k1024: ×0.835 < ×1.050
 *   - reg32x32r4x8w4: linear-m128-n2048-k1024: ×0.824 < ×1.050
 *   - reg32x32r4x8w4:tas-tb-un: linear-m128-n2048-k1024: ×0.852 < ×1.050
 *   - reg32x32r4x8w4:tb-un: linear-m128-n2048-k1024: ×0.814 < ×1.050
 *   - reg32x32r8x4w8: linear-m128-n2048-k1024: ×0.808 < ×1.050
 *   - reg32x32r8x4w8:tas-tb-un: linear-m128-n2048-k1024: ×0.838 < ×1.050
 *   - reg32x32r8x4w8:tb-un: linear-m128-n2048-k1024: ×0.856 < ×1.050
 *   - reg32x32r8x8w4: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg32x32r8x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg32x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg32x64r2x8w8: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x64r2x8w8:tas-tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x64r2x8w8:tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x64r4x4w16: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x64r4x4w16:tas-tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x64r4x4w16:tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x64r4x8w8: linear-m128-n2048-k1024: ×0.814 < ×1.050
 *   - reg32x64r4x8w8:tas-tb-un: linear-m128-n2048-k1024: ×0.843 < ×1.050
 *   - reg32x64r4x8w8:tb-un: linear-m128-n2048-k1024: ×0.824 < ×1.050
 *   - reg32x64r8x4w16: linear-m128-n2048-k1024: ×0.814 < ×1.050
 *   - reg32x64r8x4w16:tas-tb-un: linear-m128-n2048-k1024: ×0.849 < ×1.050
 *   - reg32x64r8x4w16:tb-un: linear-m128-n2048-k1024: ×0.856 < ×1.050
 *   - reg32x64r8x8w8: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg32x64r8x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg32x64r8x8w8:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg4x16r1x4w4: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg4x16r1x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg4x16r1x4w4:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg4x32r1x8w4: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg4x32r1x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg4x32r1x8w4:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x128r4x8w16: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x128r8x8w16: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x16r4x4w4: matmul-m128-n2048-k1024: ×0.908 < ×1.050
 *   - reg64x16r4x4w4:tas-tb-un: bmm-b16-m128-n64-k64: ×1.031 < ×1.050
 *   - reg64x16r4x4w4:tb-un: linear-m128-n2048-k1024: ×0.929 < ×1.050
 *   - reg64x16r8x4w4: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg64x16r8x4w4:tas-tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg64x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m128-n64-k64 (no timing to compare)
 *   - reg64x32r4x4w8:tb-un: bmm-b16-m128-n64-k128: ×1.038 < ×1.050
 *   - reg64x32r4x8w4: bmm-b16-m128-n64-k128: ×0.942 < ×1.050
 *   - reg64x32r4x8w4:tas-tb-un: bmm-b16-m128-n64-k64: ×0.982 < ×1.050
 *   - reg64x32r4x8w4:tb-un: bmm-b16-m128-n64-k128: ×0.996 < ×1.050
 *   - reg64x32r8x4w8: matmul-m128-n2048-k1024: ×0.885 < ×1.050
 *   - reg64x32r8x4w8:tas-tb-un: matmul-m128-n2048-k1024: ×0.928 < ×1.050
 *   - reg64x32r8x4w8:tb-un: bmm-b16-m128-n64-k128: ×0.917 < ×1.050
 *   - reg64x32r8x8w4: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x32r8x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16: bmm-b16-m128-n64-k128: ×0.897 < ×1.050
 *   - reg64x64r4x4w16:tas-tb-un: linear-m128-n2048-k1024: ×0.989 < ×1.050
 *   - reg64x64r4x4w16:tb-un: bmm-b16-m128-n64-k128: ×0.951 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m128-n64-k128: ×0.915 < ×1.050
 *   - reg64x64r4x8w8:tas-tb-un: bmm-b16-m128-n64-k128: ×0.992 < ×1.050
 *   - reg64x64r4x8w8:tb-un: bmm-b16-m128-n64-k64: ×0.950 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m128-n64-k128: ×0.896 < ×1.050
 *   - reg64x64r8x4w16:tas-tb-un: bmm-b16-m128-n64-k64: ×1.011 < ×1.050
 *   - reg64x64r8x4w16:tb-un: bmm-b16-m128-n64-k128: ×0.923 < ×1.050
 *   - reg64x64r8x8w8: bmm-b16-m128-n64-k64: ×0.833 < ×1.050
 *   - reg64x64r8x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x64r8x8w8:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg8x16r1x4w4: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg8x16r1x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg8x16r1x4w4:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg8x16r2x4w4: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg8x16r2x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg8x16r2x4w4:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg8x32r1x8w4: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg8x32r1x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg8x32r1x8w4:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg8x32r2x4w8: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg8x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg8x32r2x4w8:tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg8x32r2x8w4: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg8x32r2x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg8x32r2x8w4:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg8x64r2x8w8: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg8x64r2x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg8x64r2x8w8:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 * - gemmRows 129〜256 (linear / matmul / bmm row count 129〜256, 4 cases): adopted reg64x32r4x4w8:tas-tb-un ×1.138 (×1.100–×1.194)
 *   - reg128x128r8x8w16: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg128x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg128x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg128x16r8x4w4: bmm-b16-m256-n64-k64: ×0.791 < ×1.050
 *   - reg128x16r8x4w4:tas-tb-un: bmm-b16-m256-n64-k64: ×0.895 < ×1.050
 *   - reg128x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8: bmm-b16-m256-n64-k64: ×0.910 < ×1.050
 *   - reg128x32r8x4w8:tas-tb-un: bmm-b16-m256-n64-k64: ×1.022 < ×1.050
 *   - reg128x32r8x4w8:tb-un: bmm-b16-m256-n64-k64: ×0.899 < ×1.050
 *   - reg128x32r8x8w4: bmm-b16-m256-n64-k64: ×0.818 < ×1.050
 *   - reg128x32r8x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare)
 *   - reg128x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare)
 *   - reg128x64r8x4w16: bmm-b16-m256-n64-k64: ×0.764 < ×1.050
 *   - reg128x64r8x4w16:tas-tb-un: bmm-b16-m256-n64-k64: ×0.925 < ×1.050
 *   - reg128x64r8x4w16:tb-un: bmm-b16-m256-n64-k64: ×0.819 < ×1.050
 *   - reg128x64r8x8w8: bmm-b16-m256-n64-k64: ×0.821 < ×1.050
 *   - reg128x64r8x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare)
 *   - reg128x64r8x8w8:tb-un: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare)
 *   - reg16x128r4x8w16: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x16r1x4w4: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x16r1x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x16r1x4w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x16r2x4w4: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x16r2x4w4:tas-tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x16r2x4w4:tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x16r4x4w4: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x16r4x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x16r4x4w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x32r1x8w4: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x32r1x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x32r1x8w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x32r2x4w8: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x32r2x4w8:tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x32r2x8w4: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x32r2x8w4:tas-tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x32r2x8w4:tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x32r4x4w8: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x32r4x4w8:tas-tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x32r4x4w8:tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x32r4x8w4: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x32r4x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x32r4x8w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x64r2x8w8: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x64r2x8w8:tas-tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x64r2x8w8:tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x64r4x4w16: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x64r4x4w16:tas-tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x64r4x4w16:tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x64r4x8w8: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare)
 *   - reg16x64r4x8w8:tas-tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg16x64r4x8w8:tb-un: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare)
 *   - reg32x128r4x8w16: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg32x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg32x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg32x128r8x8w16: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg32x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg32x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg32x16r2x4w4: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x16r2x4w4:tas-tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x16r2x4w4:tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x16r4x4w4: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x16r4x4w4:tas-tb-un: linear-m256-n2048-k1024: ×0.819 < ×1.050
 *   - reg32x16r4x4w4:tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x16r8x4w4: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg32x16r8x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg32x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg32x32r2x4w8: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x32r2x4w8:tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x32r2x8w4: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x32r2x8w4:tas-tb-un: linear-m256-n2048-k1024: ×0.834 < ×1.050
 *   - reg32x32r2x8w4:tb-un: linear-m256-n2048-k1024: ×0.819 < ×1.050
 *   - reg32x32r4x4w8: linear-m256-n2048-k1024: ×0.808 < ×1.050
 *   - reg32x32r4x4w8:tas-tb-un: linear-m256-n2048-k1024: ×0.885 < ×1.050
 *   - reg32x32r4x4w8:tb-un: linear-m256-n2048-k1024: ×0.852 < ×1.050
 *   - reg32x32r4x8w4: linear-m256-n2048-k1024: ×0.813 < ×1.050
 *   - reg32x32r4x8w4:tas-tb-un: linear-m256-n2048-k1024: ×0.835 < ×1.050
 *   - reg32x32r4x8w4:tb-un: linear-m256-n2048-k1024: ×0.804 < ×1.050
 *   - reg32x32r8x4w8: linear-m256-n2048-k1024: ×0.806 < ×1.050
 *   - reg32x32r8x4w8:tas-tb-un: linear-m256-n2048-k1024: ×0.827 < ×1.050
 *   - reg32x32r8x4w8:tb-un: linear-m256-n2048-k1024: ×0.842 < ×1.050
 *   - reg32x32r8x8w4: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg32x32r8x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg32x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg32x64r2x8w8: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x64r2x8w8:tas-tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x64r2x8w8:tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x64r4x4w16: linear-m256-n2048-k1024: ×0.787 < ×1.050
 *   - reg32x64r4x4w16:tas-tb-un: linear-m256-n2048-k1024: ×0.815 < ×1.050
 *   - reg32x64r4x4w16:tb-un: linear-m256-n2048-k1024: ×0.813 < ×1.050
 *   - reg32x64r4x8w8: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x64r4x8w8:tas-tb-un: linear-m256-n2048-k1024: ×0.821 < ×1.050
 *   - reg32x64r4x8w8:tb-un: linear-m256-n2048-k1024: ×0.803 < ×1.050
 *   - reg32x64r8x4w16: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x64r8x4w16:tas-tb-un: linear-m256-n2048-k1024: ×0.814 < ×1.050
 *   - reg32x64r8x4w16:tb-un: linear-m256-n2048-k1024: ×0.809 < ×1.050
 *   - reg32x64r8x8w8: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x64r8x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare)
 *   - reg32x64r8x8w8:tb-un: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare)
 *   - reg4x16r1x4w4: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg4x16r1x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg4x16r1x4w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg4x32r1x8w4: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg4x32r1x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg4x32r1x8w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg64x128r4x8w16: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg64x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg64x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg64x128r8x8w16: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg64x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg64x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg64x16r4x4w4: matmul-m256-n2048-k1024: ×0.898 < ×1.050
 *   - reg64x16r4x4w4:tas-tb-un: matmul-m256-n2048-k1024: ×1.043 < ×1.050
 *   - reg64x16r4x4w4:tb-un: matmul-m256-n2048-k1024: ×0.922 < ×1.050
 *   - reg64x16r8x4w4: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg64x16r8x4w4:tas-tb-un: bmm-b16-m256-n64-k128: ×0.783 < ×1.050
 *   - reg64x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8:tb-un: geometric mean ×1.070 (below the adopted ×1.138)
 *   - reg64x32r4x8w4: bmm-b16-m256-n64-k64: ×0.956 < ×1.050
 *   - reg64x32r4x8w4:tas-tb-un: bmm-b16-m256-n64-k64: ×1.025 < ×1.050
 *   - reg64x32r4x8w4:tb-un: linear-m256-n2048-k1024: ×0.996 < ×1.050
 *   - reg64x32r8x4w8: matmul-m256-n2048-k1024: ×0.921 < ×1.050
 *   - reg64x32r8x4w8:tas-tb-un: geometric mean ×1.107 (below the adopted ×1.138)
 *   - reg64x32r8x4w8:tb-un: bmm-b16-m256-n64-k64: ×0.972 < ×1.050
 *   - reg64x32r8x8w4: bmm-b16-m256-n64-k64: ×0.804 < ×1.050
 *   - reg64x32r8x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare)
 *   - reg64x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare)
 *   - reg64x64r4x4w16: bmm-b16-m256-n64-k64: ×0.936 < ×1.050
 *   - reg64x64r4x4w16:tas-tb-un: bmm-b16-m256-n64-k64: ×1.047 < ×1.050
 *   - reg64x64r4x4w16:tb-un: bmm-b16-m256-n64-k64: ×1.020 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m256-n64-k64: ×0.936 < ×1.050
 *   - reg64x64r4x8w8:tas-tb-un: bmm-b16-m256-n64-k64: ×0.979 < ×1.050
 *   - reg64x64r4x8w8:tb-un: bmm-b16-m256-n64-k64: ×0.950 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m256-n64-k64: ×0.919 < ×1.050
 *   - reg64x64r8x4w16:tas-tb-un: bmm-b16-m256-n64-k64: ×1.048 < ×1.050
 *   - reg64x64r8x4w16:tb-un: bmm-b16-m256-n64-k64: ×0.964 < ×1.050
 *   - reg64x64r8x8w8: bmm-b16-m256-n64-k64: ×0.867 < ×1.050
 *   - reg64x64r8x8w8:tas-tb-un: bmm-b16-m256-n64-k64: ×0.867 < ×1.050
 *   - reg64x64r8x8w8:tb-un: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare)
 *   - reg8x16r1x4w4: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg8x16r1x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg8x16r1x4w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg8x16r2x4w4: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg8x16r2x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg8x16r2x4w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg8x32r1x8w4: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg8x32r1x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg8x32r1x8w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg8x32r2x4w8: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg8x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg8x32r2x4w8:tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg8x32r2x8w4: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg8x32r2x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg8x32r2x8w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg8x64r2x8w8: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg8x64r2x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg8x64r2x8w8:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 * - gemmRows 257〜512 (linear / matmul / bmm row count 257〜512, 5 cases): adopted reg128x32r8x4w8:tas-tb-un ×1.135 (×1.057–×1.262)
 *   - reg128x128r8x8w16: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg128x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg128x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg128x16r8x4w4: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg128x16r8x4w4:tas-tb-un: bmm-b16-m512-n64-k512: ×0.903 < ×1.050
 *   - reg128x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg128x32r8x4w8: bmm-b16-m512-n64-k64: ×0.921 < ×1.050
 *   - reg128x32r8x4w8:tb-un: bmm-b16-m512-n64-k64: ×0.832 < ×1.050
 *   - reg128x32r8x8w4: bmm-b16-m512-n64-k512: ×0.924 < ×1.050
 *   - reg128x32r8x8w4:tas-tb-un: bmm-b16-m512-n64-k512: ×0.877 < ×1.050
 *   - reg128x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg128x64r8x4w16: bmm-b16-m512-n512-k64: ×0.861 < ×1.050
 *   - reg128x64r8x4w16:tas-tb-un: bmm-b16-m512-n64-k64: ×0.875 < ×1.050
 *   - reg128x64r8x4w16:tb-un: screened out by the quick look in bmm-b16-m512-n64-k64 (no timing to compare)
 *   - reg128x64r8x8w8: bmm-b16-m512-n64-k64: ×0.946 < ×1.050
 *   - reg128x64r8x8w8:tas-tb-un: bmm-b16-m512-n64-k64: ×0.934 < ×1.050
 *   - reg128x64r8x8w8:tb-un: bmm-b16-m512-n64-k64: ×0.875 < ×1.050
 *   - reg16x128r4x8w16: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg16x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg16x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg16x16r1x4w4: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg16x16r1x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg16x16r1x4w4:tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg16x16r2x4w4: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg16x16r2x4w4:tas-tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg16x16r2x4w4:tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg16x16r4x4w4: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg16x16r4x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg16x16r4x4w4:tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg16x32r1x8w4: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg16x32r1x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg16x32r1x8w4:tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg16x32r2x4w8: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg16x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg16x32r2x4w8:tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg16x32r2x8w4: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg16x32r2x8w4:tas-tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg16x32r2x8w4:tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg16x32r4x4w8: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg16x32r4x4w8:tas-tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg16x32r4x4w8:tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg16x32r4x8w4: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg16x32r4x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg16x32r4x8w4:tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg16x64r2x8w8: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg16x64r2x8w8:tas-tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg16x64r2x8w8:tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg16x64r4x4w16: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg16x64r4x4w16:tas-tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg16x64r4x4w16:tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg16x64r4x8w8: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg16x64r4x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg16x64r4x8w8:tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg32x128r4x8w16: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg32x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg32x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg32x128r8x8w16: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg32x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg32x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg32x16r2x4w4: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg32x16r2x4w4:tas-tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x16r2x4w4:tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg32x16r4x4w4: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x16r4x4w4:tas-tb-un: linear-m512-n2048-k1024: ×0.811 < ×1.050
 *   - reg32x16r4x4w4:tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x16r8x4w4: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg32x16r8x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg32x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg32x32r2x4w8: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x32r2x4w8:tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x32r2x8w4: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x32r2x8w4:tas-tb-un: linear-m512-n2048-k1024: ×0.830 < ×1.050
 *   - reg32x32r2x8w4:tb-un: linear-m512-n2048-k1024: ×0.807 < ×1.050
 *   - reg32x32r4x4w8: linear-m512-n2048-k1024: ×0.805 < ×1.050
 *   - reg32x32r4x4w8:tas-tb-un: linear-m512-n2048-k1024: ×0.882 < ×1.050
 *   - reg32x32r4x4w8:tb-un: linear-m512-n2048-k1024: ×0.842 < ×1.050
 *   - reg32x32r4x8w4: matmul-m512-n2048-k1024: ×0.785 < ×1.050
 *   - reg32x32r4x8w4:tas-tb-un: linear-m512-n2048-k1024: ×0.816 < ×1.050
 *   - reg32x32r4x8w4:tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x32r8x4w8: linear-m512-n2048-k1024: ×0.785 < ×1.050
 *   - reg32x32r8x4w8:tas-tb-un: linear-m512-n2048-k1024: ×0.822 < ×1.050
 *   - reg32x32r8x4w8:tb-un: linear-m512-n2048-k1024: ×0.827 < ×1.050
 *   - reg32x32r8x8w4: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg32x32r8x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg32x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg32x64r2x8w8: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x64r2x8w8:tas-tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x64r2x8w8:tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x64r4x4w16: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x64r4x4w16:tas-tb-un: linear-m512-n2048-k1024: ×0.797 < ×1.050
 *   - reg32x64r4x4w16:tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x64r4x8w8: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x64r4x8w8:tas-tb-un: linear-m512-n2048-k1024: ×0.830 < ×1.050
 *   - reg32x64r4x8w8:tb-un: linear-m512-n2048-k1024: ×0.797 < ×1.050
 *   - reg32x64r8x4w16: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x64r8x4w16:tas-tb-un: linear-m512-n2048-k1024: ×0.859 < ×1.050
 *   - reg32x64r8x4w16:tb-un: linear-m512-n2048-k1024: ×0.850 < ×1.050
 *   - reg32x64r8x8w8: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x64r8x8w8:tas-tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x64r8x8w8:tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg4x16r1x4w4: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg4x16r1x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg4x16r1x4w4:tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg4x32r1x8w4: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg4x32r1x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg4x32r1x8w4:tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg64x128r4x8w16: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg64x128r4x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg64x128r4x8w16:tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg64x128r8x8w16: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg64x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg64x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg64x16r4x4w4: bmm-b16-m512-n512-k64: ×0.833 < ×1.050
 *   - reg64x16r4x4w4:tas-tb-un: bmm-b16-m512-n512-k64: ×0.976 < ×1.050
 *   - reg64x16r4x4w4:tb-un: bmm-b16-m512-n64-k64: ×0.913 < ×1.050
 *   - reg64x16r8x4w4: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg64x16r8x4w4:tas-tb-un: bmm-b16-m512-n64-k512: ×0.722 < ×1.050
 *   - reg64x16r8x4w4:tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg64x32r4x4w8:tas-tb-un: bmm-b16-m512-n64-k512: ×1.049 < ×1.050
 *   - reg64x32r4x4w8:tb-un: bmm-b16-m512-n64-k512: ×1.034 < ×1.050
 *   - reg64x32r4x8w4: matmul-m512-n2048-k1024: ×0.974 < ×1.050
 *   - reg64x32r4x8w4:tas-tb-un: bmm-b16-m512-n64-k512: ×0.920 < ×1.050
 *   - reg64x32r4x8w4:tb-un: linear-m512-n2048-k1024: ×1.009 < ×1.050
 *   - reg64x32r8x4w8: bmm-b16-m512-n64-k512: ×0.905 < ×1.050
 *   - reg64x32r8x4w8:tas-tb-un: matmul-m512-n2048-k1024: ×1.046 < ×1.050
 *   - reg64x32r8x4w8:tb-un: bmm-b16-m512-n64-k512: ×0.812 < ×1.050
 *   - reg64x32r8x8w4: bmm-b16-m512-n64-k512: ×0.878 < ×1.050
 *   - reg64x32r8x8w4:tas-tb-un: bmm-b16-m512-n64-k512: ×0.854 < ×1.050
 *   - reg64x32r8x8w4:tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg64x64r4x4w16: bmm-b16-m512-n64-k64: ×0.969 < ×1.050
 *   - reg64x64r4x4w16:tas-tb-un: bmm-b16-m512-n64-k64: ×0.994 < ×1.050
 *   - reg64x64r4x4w16:tb-un: bmm-b16-m512-n64-k64: ×1.016 < ×1.050
 *   - reg64x64r4x8w8: matmul-m512-n2048-k1024: ×0.989 < ×1.050
 *   - reg64x64r4x8w8:tas-tb-un: bmm-b16-m512-n64-k64: ×1.008 < ×1.050
 *   - reg64x64r4x8w8:tb-un: bmm-b16-m512-n64-k64: ×0.914 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m512-n64-k512: ×0.955 < ×1.050
 *   - reg64x64r8x4w16:tas-tb-un: bmm-b16-m512-n64-k64: ×1.015 < ×1.050
 *   - reg64x64r8x4w16:tb-un: bmm-b16-m512-n64-k64: ×0.825 < ×1.050
 *   - reg64x64r8x8w8: bmm-b16-m512-n512-k64: ×0.881 < ×1.050
 *   - reg64x64r8x8w8:tas-tb-un: bmm-b16-m512-n64-k64: ×0.980 < ×1.050
 *   - reg64x64r8x8w8:tb-un: bmm-b16-m512-n64-k512: ×0.863 < ×1.050
 *   - reg8x16r1x4w4: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg8x16r1x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg8x16r1x4w4:tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg8x16r2x4w4: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg8x16r2x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg8x16r2x4w4:tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg8x32r1x8w4: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg8x32r1x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg8x32r1x8w4:tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg8x32r2x4w8: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg8x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg8x32r2x4w8:tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg8x32r2x8w4: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg8x32r2x8w4:tas-tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg8x32r2x8w4:tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg8x64r2x8w8: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg8x64r2x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg8x64r2x8w8:tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 * - gemmRows > 512 (linear / matmul / bmm row count > 512, 7 cases): keeps the default reg128x128r8x8w16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - Sweep outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json: linear-m4096-n2048-k2048 dropped from the ratio material due to default re-measurement ratio ×1.108 out of range (output matches and failures are still checked)
 *   - reg128x128r8x8w16:tas-tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg128x128r8x8w16:tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg128x16r8x4w4: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg128x16r8x4w4:tas-tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg128x16r8x4w4:tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg128x32r8x4w8: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg128x32r8x4w8:tas-tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg128x32r8x4w8:tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg128x32r8x8w4: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg128x32r8x8w4:tas-tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg128x32r8x8w4:tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg128x64r8x4w16: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg128x64r8x4w16:tas-tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg128x64r8x4w16:tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg128x64r8x8w8: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg128x64r8x8w8:tas-tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg128x64r8x8w8:tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg16x128r4x8w16: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x128r4x8w16:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x128r4x8w16:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x16r1x4w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x16r1x4w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x16r1x4w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x16r2x4w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x16r2x4w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x16r2x4w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x16r4x4w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x16r4x4w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x16r4x4w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x32r1x8w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x32r1x8w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x32r1x8w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x32r2x4w8: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x32r2x4w8:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x32r2x8w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x32r2x8w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x32r2x8w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x32r4x4w8: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x32r4x4w8:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x32r4x4w8:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x32r4x8w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x32r4x8w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x32r4x8w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x64r2x8w8: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x64r2x8w8:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x64r2x8w8:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x64r4x4w16: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x64r4x4w16:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x64r4x4w16:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x64r4x8w8: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x64r4x8w8:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x64r4x8w8:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x128r4x8w16: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x128r4x8w16:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x128r4x8w16:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x128r8x8w16: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x128r8x8w16:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x128r8x8w16:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x16r2x4w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x16r2x4w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x16r2x4w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x16r4x4w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x16r4x4w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x16r4x4w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x16r8x4w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x16r8x4w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x16r8x4w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r2x4w8: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r2x4w8:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r2x8w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r2x8w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r2x8w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r4x4w8: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r4x4w8:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r4x4w8:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r4x8w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r4x8w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r4x8w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r8x4w8: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r8x4w8:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r8x4w8:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r8x8w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r8x8w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r8x8w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x64r2x8w8: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x64r2x8w8:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x64r2x8w8:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x64r4x4w16: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x64r4x4w16:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x64r4x4w16:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x64r4x8w8: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x64r4x8w8:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x64r4x8w8:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x64r8x4w16: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x64r8x4w16:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x64r8x4w16:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x64r8x8w8: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x64r8x8w8:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x64r8x8w8:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg4x16r1x4w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg4x16r1x4w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg4x16r1x4w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg4x32r1x8w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg4x32r1x8w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg4x32r1x8w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg64x128r4x8w16: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x128r4x8w16:tas-tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x128r4x8w16:tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x128r8x8w16: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x128r8x8w16:tas-tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x128r8x8w16:tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x16r4x4w4: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x16r4x4w4:tas-tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x16r4x4w4:tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x16r8x4w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg64x16r8x4w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg64x16r8x4w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg64x32r4x4w8: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x32r4x4w8:tas-tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x32r4x4w8:tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x32r4x8w4: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x32r4x8w4:tas-tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x32r4x8w4:tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x32r8x4w8: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x32r8x4w8:tas-tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x32r8x4w8:tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x32r8x8w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg64x32r8x8w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg64x32r8x8w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg64x64r4x4w16: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x64r4x4w16:tas-tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x64r4x4w16:tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x64r4x8w8: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x64r4x8w8:tas-tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x64r4x8w8:tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x64r8x4w16: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x64r8x4w16:tas-tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x64r8x4w16:tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x64r8x8w8: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x64r8x8w8:tas-tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg64x64r8x8w8:tb-un: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep)
 *   - reg8x16r1x4w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x16r1x4w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x16r1x4w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x16r2x4w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x16r2x4w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x16r2x4w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x32r1x8w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x32r1x8w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x32r1x8w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x32r2x4w8: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x32r2x4w8:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x32r2x8w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x32r2x8w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x32r2x8w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x64r2x8w8: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x64r2x8w8:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg8x64r2x8w8:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 * - attention.qk (fused attention f32 ①QK, 4 cases): keeps the default reg128x128r8x8w16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x16r8x4w4: attention-qk-cross-m4096-n512: ×0.819 < ×1.050
 *   - reg128x32r8x4w8: attention-qk-cross-m4096-n512: ×0.949 < ×1.050
 *   - reg128x32r8x8w4: attention-qk-cross-m4096-n512: ×0.938 < ×1.050
 *   - reg128x64r8x4w16: attention-qk-cross-m4096-n512: ×0.956 < ×1.050
 *   - reg128x64r8x8w8: attention-qk-cross-m4096-n512: ×1.016 < ×1.050
 *   - reg16x128r4x8w16: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg16x16r1x4w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg16x16r2x4w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg16x16r4x4w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg16x32r1x8w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg16x32r2x4w8: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg16x32r2x8w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg16x32r4x4w8: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg16x32r4x8w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg16x64r2x8w8: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg16x64r4x4w16: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg16x64r4x8w8: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x128r4x8w16: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x128r8x8w16: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x16r2x4w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x16r4x4w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x16r8x4w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x32r2x4w8: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x32r2x8w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x32r4x4w8: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x32r4x8w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x32r8x4w8: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x32r8x8w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x64r2x8w8: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x64r4x4w16: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x64r4x8w8: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x64r8x4w16: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x64r8x8w8: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg4x16r1x4w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg4x32r1x8w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg64x128r4x8w16: attention-qk-cross-m4096-n512: ×0.884 < ×1.050
 *   - reg64x128r8x8w16: attention-qk-cross-m4096-n512: ×0.882 < ×1.050
 *   - reg64x16r4x4w4: screened out by the quick look in attention-qk-self-m4096-n4096 (no timing to compare)
 *   - reg64x16r8x4w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg64x32r4x4w8: attention-qk-cross-m4096-n512: ×0.861 < ×1.050
 *   - reg64x32r4x8w4: attention-qk-cross-m4096-n512: ×0.878 < ×1.050
 *   - reg64x32r8x4w8: attention-qk-self-m1024-n1024: ×0.865 < ×1.050
 *   - reg64x32r8x8w4: screened out by the quick look in attention-qk-cross-m4096-n512 (no timing to compare)
 *   - reg64x64r4x4w16: attention-qk-cross-m4096-n512: ×0.867 < ×1.050
 *   - reg64x64r4x8w8: attention-qk-cross-m4096-n512: ×0.907 < ×1.050
 *   - reg64x64r8x4w16: attention-qk-cross-m4096-n512: ×0.879 < ×1.050
 *   - reg64x64r8x8w8: attention-qk-self-m1024-n1024: ×0.878 < ×1.050
 *   - reg8x16r1x4w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg8x16r2x4w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg8x32r1x8w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg8x32r2x4w8: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg8x32r2x8w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg8x64r2x8w8: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 * - attention.pv (fused attention f32 ③PV, 4 cases): keeps the default reg128x128r8x8w16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x16r8x4w4: screened out by the quick look in attention-pv-self-m1024-n1024 (no timing to compare)
 *   - reg128x32r8x4w8: attention-pv-self-m4096-n4096: ×0.860 < ×1.050
 *   - reg128x32r8x8w4: attention-pv-self-m1024-n1024: ×0.862 < ×1.050
 *   - reg128x64r8x4w16: attention-pv-self-m4096-n4096: ×0.890 < ×1.050
 *   - reg128x64r8x8w8: attention-pv-self-m1024-n1024: ×0.960 < ×1.050
 *   - reg16x128r4x8w16: attention-pv-self-m4096-n4096: ×0.855 < ×1.050
 *   - reg16x16r1x4w4: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg16x16r2x4w4: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg16x16r4x4w4: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg16x32r1x8w4: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg16x32r2x4w8: screened out by the quick look in attention-pv-self-m1024-n1024 (no timing to compare)
 *   - reg16x32r2x8w4: screened out by the quick look in attention-pv-self-m1024-n1024 (no timing to compare)
 *   - reg16x32r4x4w8: attention-pv-self-m4096-n4096: ×0.790 < ×1.050
 *   - reg16x32r4x8w4: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg16x64r2x8w8: attention-pv-self-m4096-n4096: ×0.819 < ×1.050
 *   - reg16x64r4x4w16: attention-pv-self-m4096-n4096: ×0.829 < ×1.050
 *   - reg16x64r4x8w8: attention-pv-self-m4096-n4096: ×0.816 < ×1.050
 *   - reg32x128r4x8w16: attention-pv-self-m4096-n4096: ×0.912 < ×1.050
 *   - reg32x128r8x8w16: attention-pv-cross-m4096-n512: ×0.934 < ×1.050
 *   - reg32x16r2x4w4: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg32x16r4x4w4: screened out by the quick look in attention-pv-self-m1024-n1024 (no timing to compare)
 *   - reg32x16r8x4w4: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg32x32r2x4w8: attention-pv-self-m4096-n4096: ×0.834 < ×1.050
 *   - reg32x32r2x8w4: attention-pv-self-m4096-n4096: ×0.832 < ×1.050
 *   - reg32x32r4x4w8: attention-pv-self-m4096-n4096: ×0.840 < ×1.050
 *   - reg32x32r4x8w4: attention-pv-self-m4096-n4096: ×0.853 < ×1.050
 *   - reg32x32r8x4w8: attention-pv-self-m4096-n4096: ×0.829 < ×1.050
 *   - reg32x32r8x8w4: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg32x64r2x8w8: attention-pv-self-m4096-n4096: ×0.866 < ×1.050
 *   - reg32x64r4x4w16: attention-pv-self-m4096-n4096: ×0.877 < ×1.050
 *   - reg32x64r4x8w8: attention-pv-self-m4096-n4096: ×0.891 < ×1.050
 *   - reg32x64r8x4w16: attention-pv-cross-m4096-n512: ×0.859 < ×1.050
 *   - reg32x64r8x8w8: attention-pv-self-m1024-n1024: ×0.886 < ×1.050
 *   - reg4x16r1x4w4: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg4x32r1x8w4: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg64x128r4x8w16: attention-pv-self-m4096-n4096: ×0.926 < ×1.050
 *   - reg64x128r8x8w16: attention-pv-self-m1024-n1024: ×0.918 < ×1.050
 *   - reg64x16r4x4w4: screened out by the quick look in attention-pv-self-m1024-n1024 (no timing to compare)
 *   - reg64x16r8x4w4: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg64x32r4x4w8: attention-pv-self-m4096-n4096: ×0.850 < ×1.050
 *   - reg64x32r4x8w4: attention-pv-self-m4096-n4096: ×0.876 < ×1.050
 *   - reg64x32r8x4w8: attention-pv-self-m4096-n4096: ×0.845 < ×1.050
 *   - reg64x32r8x8w4: attention-pv-self-m4096-n4096: ×0.811 < ×1.050
 *   - reg64x64r4x4w16: attention-pv-self-m4096-n4096: ×0.890 < ×1.050
 *   - reg64x64r4x8w8: attention-pv-self-m4096-n4096: ×0.920 < ×1.050
 *   - reg64x64r8x4w16: attention-pv-self-m1024-n1024: ×0.904 < ×1.050
 *   - reg64x64r8x8w8: attention-pv-cross-m1024-n512: ×0.811 < ×1.050
 *   - reg8x16r1x4w4: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg8x16r2x4w4: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg8x32r1x8w4: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg8x32r2x4w8: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg8x32r2x8w4: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg8x64r2x8w8: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 * - conv2d.rows64 (conv2d implicit GEMM m tile of 64 rows, 2 cases): adopted igemm64x64:wg16x8:tas-tb-un:dec ×1.368 (×1.311–×1.427)
 *   - igemm128x128:wg16x16: conv2d-c192-256x256: ×0.861 < ×1.050
 *   - igemm128x128:wg16x16:dec: conv2d-c192-256x256: ×0.961 < ×1.050
 *   - igemm128x128:wg16x16:tas-tb-un: conv2d-c192-256x256: ×0.866 < ×1.050
 *   - igemm128x128:wg16x16:tas-tb-un:dec: conv2d-c192-256x256: ×0.922 < ×1.050
 *   - igemm128x128:wg16x16:tb-un: conv2d-c192-256x256: ×0.815 < ×1.050
 *   - igemm128x128:wg16x16:tb-un:dec: conv2d-c192-256x256: ×0.882 < ×1.050
 *   - igemm128x64:wg16x16: conv2d-c192-256x256: ×0.821 < ×1.050
 *   - igemm128x64:wg16x16:dec: conv2d-c192-256x256: ×0.904 < ×1.050
 *   - igemm128x64:wg16x16:tas-tb-un: conv2d-c192-256x256: ×0.945 < ×1.050
 *   - igemm128x64:wg16x16:tas-tb-un:dec: conv2d-c192-256x256: ×0.980 < ×1.050
 *   - igemm128x64:wg16x16:tb-un: conv2d-c192-256x256: ×0.860 < ×1.050
 *   - igemm128x64:wg16x16:tb-un:dec: conv2d-c192-256x256: ×0.947 < ×1.050
 *   - igemm128x64:wg8x16: conv2d-c192-256x256: ×0.900 < ×1.050
 *   - igemm128x64:wg8x16:dec: conv2d-c192-256x256: ×0.967 < ×1.050
 *   - igemm128x64:wg8x16:tas-tb-un: conv2d-c192-256x256: ×0.892 < ×1.050
 *   - igemm128x64:wg8x16:tas-tb-un:dec: conv2d-c192-256x256: ×0.969 < ×1.050
 *   - igemm128x64:wg8x16:tb-un: conv2d-c192-256x256: ×0.813 < ×1.050
 *   - igemm128x64:wg8x16:tb-un:dec: conv2d-c192-256x256: ×0.852 < ×1.050
 *   - igemm16x128:wg16x4: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm16x128:wg16x4:dec: conv2d-c192-256x256: ×0.861 < ×1.050
 *   - igemm16x128:wg16x4:tas-tb-un: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm16x128:wg16x4:tas-tb-un:dec: conv2d-c192-256x256: ×0.921 < ×1.050
 *   - igemm16x128:wg16x4:tb-un: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm16x128:wg16x4:tb-un:dec: conv2d-c192-256x256: ×0.877 < ×1.050
 *   - igemm16x64:wg16x4: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm16x64:wg16x4:dec: conv2d-c192-256x256: ×0.841 < ×1.050
 *   - igemm16x64:wg16x4:tas-tb-un: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm16x64:wg16x4:tas-tb-un:dec: conv2d-c192-256x256: ×0.985 < ×1.050
 *   - igemm16x64:wg16x4:tb-un: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm16x64:wg16x4:tb-un:dec: conv2d-c192-256x256: ×0.961 < ×1.050
 *   - igemm16x64:wg8x4: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm16x64:wg8x4:dec: conv2d-c192-256x256: ×0.886 < ×1.050
 *   - igemm16x64:wg8x4:tas-tb-un: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm16x64:wg8x4:tas-tb-un:dec: conv2d-c192-256x256: ×0.894 < ×1.050
 *   - igemm16x64:wg8x4:tb-un: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm16x64:wg8x4:tb-un:dec: conv2d-c192-256x256: ×0.859 < ×1.050
 *   - igemm16x64:wg8x8: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm16x64:wg8x8:dec: conv2d-c192-256x256: ×0.849 < ×1.050
 *   - igemm16x64:wg8x8:tas-tb-un: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm16x64:wg8x8:tas-tb-un:dec: conv2d-c192-256x256: ×0.921 < ×1.050
 *   - igemm16x64:wg8x8:tb-un: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm16x64:wg8x8:tb-un:dec: conv2d-c192-256x256: ×0.928 < ×1.050
 *   - igemm32x128:wg16x4: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm32x128:wg16x4:dec: geometric mean ×1.142 (below the adopted ×1.368)
 *   - igemm32x128:wg16x4:tas-tb-un: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm32x128:wg16x4:tas-tb-un:dec: conv2d-c192-256x256: ×0.951 < ×1.050
 *   - igemm32x128:wg16x4:tb-un: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm32x128:wg16x4:tb-un:dec: conv2d-c192-256x256: ×0.847 < ×1.050
 *   - igemm32x128:wg16x8: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm32x128:wg16x8:dec: geometric mean ×1.116 (below the adopted ×1.368)
 *   - igemm32x128:wg16x8:tas-tb-un: screened out by the quick look in conv2d-c384-128x128 (no timing to compare)
 *   - igemm32x128:wg16x8:tas-tb-un:dec: geometric mean ×1.184 (below the adopted ×1.368)
 *   - igemm32x128:wg16x8:tb-un: screened out by the quick look in conv2d-c384-128x128 (no timing to compare)
 *   - igemm32x128:wg16x8:tb-un:dec: geometric mean ×1.240 (below the adopted ×1.368)
 *   - igemm32x64:wg16x4: conv2d-c384-128x128: ×0.765 < ×1.050
 *   - igemm32x64:wg16x4:dec: geometric mean ×1.151 (below the adopted ×1.368)
 *   - igemm32x64:wg16x4:tas-tb-un: screened out by the quick look in conv2d-c384-128x128 (no timing to compare)
 *   - igemm32x64:wg16x4:tas-tb-un:dec: geometric mean ×1.217 (below the adopted ×1.368)
 *   - igemm32x64:wg16x4:tb-un: screened out by the quick look in conv2d-c384-128x128 (no timing to compare)
 *   - igemm32x64:wg16x4:tb-un:dec: geometric mean ×1.230 (below the adopted ×1.368)
 *   - igemm32x64:wg16x8: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm32x64:wg16x8:dec: conv2d-c192-256x256: ×1.000 < ×1.050
 *   - igemm32x64:wg16x8:tas-tb-un: conv2d-c384-128x128: ×0.886 < ×1.050
 *   - igemm32x64:wg16x8:tas-tb-un:dec: geometric mean ×1.202 (below the adopted ×1.368)
 *   - igemm32x64:wg16x8:tb-un: conv2d-c384-128x128: ×0.873 < ×1.050
 *   - igemm32x64:wg16x8:tb-un:dec: geometric mean ×1.177 (below the adopted ×1.368)
 *   - igemm32x64:wg8x16: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm32x64:wg8x16:dec: conv2d-c192-256x256: ×0.986 < ×1.050
 *   - igemm32x64:wg8x16:tas-tb-un: conv2d-c192-256x256: ×0.843 < ×1.050
 *   - igemm32x64:wg8x16:tas-tb-un:dec: geometric mean ×1.097 (below the adopted ×1.368)
 *   - igemm32x64:wg8x16:tb-un: conv2d-c192-256x256: ×0.854 < ×1.050
 *   - igemm32x64:wg8x16:tb-un:dec: geometric mean ×1.105 (below the adopted ×1.368)
 *   - igemm32x64:wg8x4: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm32x64:wg8x4:dec: geometric mean ×1.144 (below the adopted ×1.368)
 *   - igemm32x64:wg8x4:tas-tb-un: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm32x64:wg8x4:tas-tb-un:dec: conv2d-c192-256x256: ×0.865 < ×1.050
 *   - igemm32x64:wg8x4:tb-un: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm32x64:wg8x4:tb-un:dec: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm32x64:wg8x8: screened out by the quick look in conv2d-c384-128x128 (no timing to compare)
 *   - igemm32x64:wg8x8:dec: geometric mean ×1.101 (below the adopted ×1.368)
 *   - igemm32x64:wg8x8:tas-tb-un: screened out by the quick look in conv2d-c384-128x128 (no timing to compare)
 *   - igemm32x64:wg8x8:tas-tb-un:dec: geometric mean ×1.214 (below the adopted ×1.368)
 *   - igemm32x64:wg8x8:tb-un: screened out by the quick look in conv2d-c384-128x128 (no timing to compare)
 *   - igemm32x64:wg8x8:tb-un:dec: geometric mean ×1.215 (below the adopted ×1.368)
 *   - igemm64x128:wg16x16: conv2d-c384-128x128: ×0.987 < ×1.050
 *   - igemm64x128:wg16x16:dec: geometric mean ×1.204 (below the adopted ×1.368)
 *   - igemm64x128:wg16x16:tas-tb-un: geometric mean ×1.084 (below the adopted ×1.368)
 *   - igemm64x128:wg16x16:tas-tb-un:dec: geometric mean ×1.249 (below the adopted ×1.368)
 *   - igemm64x128:wg16x16:tb-un: geometric mean ×1.131 (below the adopted ×1.368)
 *   - igemm64x128:wg16x16:tb-un:dec: geometric mean ×1.271 (below the adopted ×1.368)
 *   - igemm64x128:wg16x8:dec: geometric mean ×1.247 (below the adopted ×1.368)
 *   - igemm64x128:wg16x8:tas-tb-un: conv2d-c192-256x256: ×0.984 < ×1.050
 *   - igemm64x128:wg16x8:tas-tb-un:dec: geometric mean ×1.335 (below the adopted ×1.368)
 *   - igemm64x128:wg16x8:tb-un: conv2d-c384-128x128: ×0.851 < ×1.050
 *   - igemm64x128:wg16x8:tb-un:dec: geometric mean ×1.131 (below the adopted ×1.368)
 *   - igemm64x64:wg16x16: conv2d-c192-256x256: ×0.937 < ×1.050
 *   - igemm64x64:wg16x16:dec: geometric mean ×1.126 (below the adopted ×1.368)
 *   - igemm64x64:wg16x16:tas-tb-un: geometric mean ×1.073 (below the adopted ×1.368)
 *   - igemm64x64:wg16x16:tas-tb-un:dec: geometric mean ×1.283 (below the adopted ×1.368)
 *   - igemm64x64:wg16x16:tb-un: conv2d-c192-256x256: ×1.016 < ×1.050
 *   - igemm64x64:wg16x16:tb-un:dec: geometric mean ×1.225 (below the adopted ×1.368)
 *   - igemm64x64:wg16x8: conv2d-c192-256x256: ×0.967 < ×1.050
 *   - igemm64x64:wg16x8:dec: geometric mean ×1.226 (below the adopted ×1.368)
 *   - igemm64x64:wg16x8:tas-tb-un: geometric mean ×1.157 (below the adopted ×1.368)
 *   - igemm64x64:wg16x8:tb-un: geometric mean ×1.075 (below the adopted ×1.368)
 *   - igemm64x64:wg16x8:tb-un:dec: geometric mean ×1.312 (below the adopted ×1.368)
 *   - igemm64x64:wg8x16: conv2d-c192-256x256: ×1.010 < ×1.050
 *   - igemm64x64:wg8x16:dec: geometric mean ×1.228 (below the adopted ×1.368)
 *   - igemm64x64:wg8x16:tas-tb-un: conv2d-c192-256x256: ×1.021 < ×1.050
 *   - igemm64x64:wg8x16:tas-tb-un:dec: geometric mean ×1.322 (below the adopted ×1.368)
 *   - igemm64x64:wg8x16:tb-un: geometric mean ×1.119 (below the adopted ×1.368)
 *   - igemm64x64:wg8x16:tb-un:dec: geometric mean ×1.337 (below the adopted ×1.368)
 *   - igemm64x64:wg8x8: conv2d-c384-128x128: ×1.027 < ×1.050
 *   - igemm64x64:wg8x8:dec: geometric mean ×1.243 (below the adopted ×1.368)
 *   - igemm64x64:wg8x8:tas-tb-un: conv2d-c192-256x256: ×0.969 < ×1.050
 *   - igemm64x64:wg8x8:tas-tb-un:dec: geometric mean ×1.285 (below the adopted ×1.368)
 *   - igemm64x64:wg8x8:tb-un: conv2d-c384-128x128: ×0.842 < ×1.050
 *   - igemm64x64:wg8x8:tb-un:dec: conv2d-c192-256x256: ×0.997 < ×1.050
 *   - igemm8x64:wg8x4: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm8x64:wg8x4:dec: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm8x64:wg8x4:tas-tb-un: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm8x64:wg8x4:tas-tb-un:dec: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm8x64:wg8x4:tb-un: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm8x64:wg8x4:tb-un:dec: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 * - conv2d.rows32 (conv2d implicit GEMM m tile of 32 rows, 1 cases): adopted igemm32x64:wg16x8:tas-tb-un:dec ×1.488 (×1.488–×1.488)
 *   - igemm128x128:wg16x16: geometric mean ×1.137 (below the adopted ×1.488)
 *   - igemm128x128:wg16x16:dec: geometric mean ×1.159 (below the adopted ×1.488)
 *   - igemm128x128:wg16x16:tas-tb-un: geometric mean ×1.121 (below the adopted ×1.488)
 *   - igemm128x128:wg16x16:tas-tb-un:dec: geometric mean ×1.209 (below the adopted ×1.488)
 *   - igemm128x128:wg16x16:tb-un: geometric mean ×1.105 (below the adopted ×1.488)
 *   - igemm128x128:wg16x16:tb-un:dec: geometric mean ×1.133 (below the adopted ×1.488)
 *   - igemm128x64:wg16x16: conv2d-c96-512x512: ×1.042 < ×1.050
 *   - igemm128x64:wg16x16:dec: geometric mean ×1.110 (below the adopted ×1.488)
 *   - igemm128x64:wg16x16:tas-tb-un: geometric mean ×1.156 (below the adopted ×1.488)
 *   - igemm128x64:wg16x16:tas-tb-un:dec: geometric mean ×1.253 (below the adopted ×1.488)
 *   - igemm128x64:wg16x16:tb-un: geometric mean ×1.138 (below the adopted ×1.488)
 *   - igemm128x64:wg16x16:tb-un:dec: geometric mean ×1.192 (below the adopted ×1.488)
 *   - igemm128x64:wg8x16: geometric mean ×1.175 (below the adopted ×1.488)
 *   - igemm128x64:wg8x16:dec: geometric mean ×1.144 (below the adopted ×1.488)
 *   - igemm128x64:wg8x16:tas-tb-un: geometric mean ×1.152 (below the adopted ×1.488)
 *   - igemm128x64:wg8x16:tas-tb-un:dec: geometric mean ×1.249 (below the adopted ×1.488)
 *   - igemm128x64:wg8x16:tb-un: geometric mean ×1.122 (below the adopted ×1.488)
 *   - igemm128x64:wg8x16:tb-un:dec: geometric mean ×1.089 (below the adopted ×1.488)
 *   - igemm16x128:wg16x4: screened out by the quick look in conv2d-c96-512x512 (no timing to compare)
 *   - igemm16x128:wg16x4:dec: geometric mean ×1.211 (below the adopted ×1.488)
 *   - igemm16x128:wg16x4:tas-tb-un: screened out by the quick look in conv2d-c96-512x512 (no timing to compare)
 *   - igemm16x128:wg16x4:tas-tb-un:dec: geometric mean ×1.193 (below the adopted ×1.488)
 *   - igemm16x128:wg16x4:tb-un: screened out by the quick look in conv2d-c96-512x512 (no timing to compare)
 *   - igemm16x128:wg16x4:tb-un:dec: geometric mean ×1.134 (below the adopted ×1.488)
 *   - igemm16x64:wg16x4: screened out by the quick look in conv2d-c96-512x512 (no timing to compare)
 *   - igemm16x64:wg16x4:dec: geometric mean ×1.094 (below the adopted ×1.488)
 *   - igemm16x64:wg16x4:tas-tb-un: conv2d-c96-512x512: ×0.828 < ×1.050
 *   - igemm16x64:wg16x4:tas-tb-un:dec: geometric mean ×1.278 (below the adopted ×1.488)
 *   - igemm16x64:wg16x4:tb-un: screened out by the quick look in conv2d-c96-512x512 (no timing to compare)
 *   - igemm16x64:wg16x4:tb-un:dec: geometric mean ×1.244 (below the adopted ×1.488)
 *   - igemm16x64:wg8x4: screened out by the quick look in conv2d-c96-512x512 (no timing to compare)
 *   - igemm16x64:wg8x4:dec: geometric mean ×1.145 (below the adopted ×1.488)
 *   - igemm16x64:wg8x4:tas-tb-un: screened out by the quick look in conv2d-c96-512x512 (no timing to compare)
 *   - igemm16x64:wg8x4:tas-tb-un:dec: geometric mean ×1.152 (below the adopted ×1.488)
 *   - igemm16x64:wg8x4:tb-un: screened out by the quick look in conv2d-c96-512x512 (no timing to compare)
 *   - igemm16x64:wg8x4:tb-un:dec: geometric mean ×1.085 (below the adopted ×1.488)
 *   - igemm16x64:wg8x8: screened out by the quick look in conv2d-c96-512x512 (no timing to compare)
 *   - igemm16x64:wg8x8:dec: geometric mean ×1.098 (below the adopted ×1.488)
 *   - igemm16x64:wg8x8:tas-tb-un: screened out by the quick look in conv2d-c96-512x512 (no timing to compare)
 *   - igemm16x64:wg8x8:tas-tb-un:dec: geometric mean ×1.194 (below the adopted ×1.488)
 *   - igemm16x64:wg8x8:tb-un: screened out by the quick look in conv2d-c96-512x512 (no timing to compare)
 *   - igemm16x64:wg8x8:tb-un:dec: geometric mean ×1.200 (below the adopted ×1.488)
 *   - igemm32x128:wg16x4:dec: geometric mean ×1.384 (below the adopted ×1.488)
 *   - igemm32x128:wg16x4:tas-tb-un: conv2d-c96-512x512: ×1.009 < ×1.050
 *   - igemm32x128:wg16x4:tas-tb-un:dec: geometric mean ×1.213 (below the adopted ×1.488)
 *   - igemm32x128:wg16x4:tb-un: conv2d-c96-512x512: ×0.812 < ×1.050
 *   - igemm32x128:wg16x4:tb-un:dec: geometric mean ×1.081 (below the adopted ×1.488)
 *   - igemm32x128:wg16x8: geometric mean ×1.125 (below the adopted ×1.488)
 *   - igemm32x128:wg16x8:dec: geometric mean ×1.355 (below the adopted ×1.488)
 *   - igemm32x128:wg16x8:tas-tb-un: geometric mean ×1.121 (below the adopted ×1.488)
 *   - igemm32x128:wg16x8:tas-tb-un:dec: geometric mean ×1.472 (below the adopted ×1.488)
 *   - igemm32x128:wg16x8:tb-un: geometric mean ×1.159 (below the adopted ×1.488)
 *   - igemm32x128:wg16x8:tb-un:dec: geometric mean ×1.363 (below the adopted ×1.488)
 *   - igemm32x64:wg16x4: geometric mean ×1.137 (below the adopted ×1.488)
 *   - igemm32x64:wg16x4:dec: geometric mean ×1.434 (below the adopted ×1.488)
 *   - igemm32x64:wg16x4:tas-tb-un: geometric mean ×1.152 (below the adopted ×1.488)
 *   - igemm32x64:wg16x4:tas-tb-un:dec: geometric mean ×1.460 (below the adopted ×1.488)
 *   - igemm32x64:wg16x4:tb-un: conv2d-c96-512x512: ×1.021 < ×1.050
 *   - igemm32x64:wg16x4:tb-un:dec: geometric mean ×1.471 (below the adopted ×1.488)
 *   - igemm32x64:wg16x8: geometric mean ×1.050 (below the adopted ×1.488)
 *   - igemm32x64:wg16x8:dec: geometric mean ×1.286 (below the adopted ×1.488)
 *   - igemm32x64:wg16x8:tas-tb-un: geometric mean ×1.186 (below the adopted ×1.488)
 *   - igemm32x64:wg16x8:tb-un: geometric mean ×1.166 (below the adopted ×1.488)
 *   - igemm32x64:wg16x8:tb-un:dec: geometric mean ×1.448 (below the adopted ×1.488)
 *   - igemm32x64:wg8x16: geometric mean ×1.052 (below the adopted ×1.488)
 *   - igemm32x64:wg8x16:dec: geometric mean ×1.263 (below the adopted ×1.488)
 *   - igemm32x64:wg8x16:tas-tb-un: geometric mean ×1.125 (below the adopted ×1.488)
 *   - igemm32x64:wg8x16:tas-tb-un:dec: geometric mean ×1.355 (below the adopted ×1.488)
 *   - igemm32x64:wg8x16:tb-un: geometric mean ×1.141 (below the adopted ×1.488)
 *   - igemm32x64:wg8x16:tb-un:dec: geometric mean ×1.363 (below the adopted ×1.488)
 *   - igemm32x64:wg8x4: conv2d-c96-512x512: ×0.986 < ×1.050
 *   - igemm32x64:wg8x4:dec: geometric mean ×1.375 (below the adopted ×1.488)
 *   - igemm32x64:wg8x4:tas-tb-un: conv2d-c96-512x512: ×0.880 < ×1.050
 *   - igemm32x64:wg8x4:tas-tb-un:dec: geometric mean ×1.126 (below the adopted ×1.488)
 *   - igemm32x64:wg8x4:tb-un: screened out by the quick look in conv2d-c96-512x512 (no timing to compare)
 *   - igemm32x64:wg8x4:tb-un:dec: conv2d-c96-512x512: ×1.026 < ×1.050
 *   - igemm32x64:wg8x8: geometric mean ×1.095 (below the adopted ×1.488)
 *   - igemm32x64:wg8x8:dec: geometric mean ×1.360 (below the adopted ×1.488)
 *   - igemm32x64:wg8x8:tas-tb-un: geometric mean ×1.114 (below the adopted ×1.488)
 *   - igemm32x64:wg8x8:tas-tb-un:dec: geometric mean ×1.469 (below the adopted ×1.488)
 *   - igemm32x64:wg8x8:tb-un: geometric mean ×1.136 (below the adopted ×1.488)
 *   - igemm32x64:wg8x8:tb-un:dec: geometric mean ×1.473 (below the adopted ×1.488)
 *   - igemm64x128:wg16x16: conv2d-c96-512x512: ×0.992 < ×1.050
 *   - igemm64x128:wg16x16:dec: conv2d-c96-512x512: ×1.036 < ×1.050
 *   - igemm64x128:wg16x16:tas-tb-un: geometric mean ×1.088 (below the adopted ×1.488)
 *   - igemm64x128:wg16x16:tas-tb-un:dec: geometric mean ×1.197 (below the adopted ×1.488)
 *   - igemm64x128:wg16x16:tb-un: geometric mean ×1.079 (below the adopted ×1.488)
 *   - igemm64x128:wg16x16:tb-un:dec: geometric mean ×1.196 (below the adopted ×1.488)
 *   - igemm64x128:wg16x8: conv2d-c96-512x512: ×1.011 < ×1.050
 *   - igemm64x128:wg16x8:dec: geometric mean ×1.169 (below the adopted ×1.488)
 *   - igemm64x128:wg16x8:tas-tb-un: conv2d-c96-512x512: ×0.990 < ×1.050
 *   - igemm64x128:wg16x8:tas-tb-un:dec: geometric mean ×1.231 (below the adopted ×1.488)
 *   - igemm64x128:wg16x8:tb-un: conv2d-c96-512x512: ×0.921 < ×1.050
 *   - igemm64x128:wg16x8:tb-un:dec: conv2d-c96-512x512: ×1.046 < ×1.050
 *   - igemm64x64:wg16x16: conv2d-c96-512x512: ×0.931 < ×1.050
 *   - igemm64x64:wg16x16:dec: conv2d-c96-512x512: ×1.046 < ×1.050
 *   - igemm64x64:wg16x16:tas-tb-un: geometric mean ×1.053 (below the adopted ×1.488)
 *   - igemm64x64:wg16x16:tas-tb-un:dec: geometric mean ×1.195 (below the adopted ×1.488)
 *   - igemm64x64:wg16x16:tb-un: conv2d-c96-512x512: ×1.013 < ×1.050
 *   - igemm64x64:wg16x16:tb-un:dec: geometric mean ×1.134 (below the adopted ×1.488)
 *   - igemm64x64:wg16x8: conv2d-c96-512x512: ×1.029 < ×1.050
 *   - igemm64x64:wg16x8:dec: geometric mean ×1.137 (below the adopted ×1.488)
 *   - igemm64x64:wg16x8:tas-tb-un: geometric mean ×1.131 (below the adopted ×1.488)
 *   - igemm64x64:wg16x8:tas-tb-un:dec: geometric mean ×1.267 (below the adopted ×1.488)
 *   - igemm64x64:wg16x8:tb-un: geometric mean ×1.067 (below the adopted ×1.488)
 *   - igemm64x64:wg16x8:tb-un:dec: geometric mean ×1.218 (below the adopted ×1.488)
 *   - igemm64x64:wg8x16: conv2d-c96-512x512: ×1.018 < ×1.050
 *   - igemm64x64:wg8x16:dec: geometric mean ×1.135 (below the adopted ×1.488)
 *   - igemm64x64:wg8x16:tas-tb-un: geometric mean ×1.103 (below the adopted ×1.488)
 *   - igemm64x64:wg8x16:tas-tb-un:dec: geometric mean ×1.234 (below the adopted ×1.488)
 *   - igemm64x64:wg8x16:tb-un: geometric mean ×1.128 (below the adopted ×1.488)
 *   - igemm64x64:wg8x16:tb-un:dec: geometric mean ×1.213 (below the adopted ×1.488)
 *   - igemm64x64:wg8x8: conv2d-c96-512x512: ×1.036 < ×1.050
 *   - igemm64x64:wg8x8:dec: geometric mean ×1.197 (below the adopted ×1.488)
 *   - igemm64x64:wg8x8:tas-tb-un: conv2d-c96-512x512: ×1.008 < ×1.050
 *   - igemm64x64:wg8x8:tas-tb-un:dec: geometric mean ×1.196 (below the adopted ×1.488)
 *   - igemm64x64:wg8x8:tb-un: conv2d-c96-512x512: ×0.873 < ×1.050
 *   - igemm64x64:wg8x8:tb-un:dec: conv2d-c96-512x512: ×0.969 < ×1.050
 *   - igemm8x64:wg8x4: screened out by the quick look in conv2d-c96-512x512 (no timing to compare)
 *   - igemm8x64:wg8x4:dec: screened out by the quick look in conv2d-c96-512x512 (no timing to compare)
 *   - igemm8x64:wg8x4:tas-tb-un: screened out by the quick look in conv2d-c96-512x512 (no timing to compare)
 *   - igemm8x64:wg8x4:tas-tb-un:dec: conv2d-c96-512x512: ×0.840 < ×1.050
 *   - igemm8x64:wg8x4:tb-un: screened out by the quick look in conv2d-c96-512x512 (no timing to compare)
 *   - igemm8x64:wg8x4:tb-un:dec: conv2d-c96-512x512: ×0.845 < ×1.050
 * - i8a8.linear ≤ 256 (i8a8 linear row count ≤ 256, 3 cases): adopted tile16x64r4x4w16x4k32 ×2.188 (×1.060–×7.188)
 *   - tile128x128r16x8w16x8k16: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x128r16x8w16x8k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x128r16x8w16x8k32: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x128r16x8w16x8k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x128r16x8w16x8k64: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x128r16x8w16x8k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x128r8x8w16x16k16: i8a8-linear-m1-n18432-k3072: ×0.957 < ×1.050
 *   - tile128x128r8x8w16x16k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x128r8x8w16x16k32: i8a8-linear-m1-n18432-k3072: ×0.959 < ×1.050
 *   - tile128x128r8x8w16x16k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x128r8x8w16x16k64: i8a8-linear-m1-n18432-k3072: ×0.947 < ×1.050
 *   - tile128x128r8x8w16x16k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x32r16x4w8x8k16: i8a8-linear-m185-n1280-k1280: ×0.959 < ×1.050
 *   - tile128x32r16x4w8x8k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x32r16x4w8x8k32: i8a8-linear-m185-n1280-k1280: ×0.947 < ×1.050
 *   - tile128x32r16x4w8x8k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x32r16x4w8x8k64: i8a8-linear-m185-n1280-k1280: ×0.934 < ×1.050
 *   - tile128x32r16x4w8x8k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x32r8x4w8x16k16: geometric mean ×1.076 (below the adopted ×2.188)
 *   - tile128x32r8x4w8x16k16s: i8a8-linear-m1-n18432-k3072: ×1.039 < ×1.050
 *   - tile128x32r8x4w8x16k32: geometric mean ×1.073 (below the adopted ×2.188)
 *   - tile128x32r8x4w8x16k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x32r8x4w8x16k64: i8a8-linear-m1-n18432-k3072: ×1.020 < ×1.050
 *   - tile128x32r8x4w8x16k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x64r16x4w16x8k16: i8a8-linear-m1-n18432-k3072: ×0.990 < ×1.050
 *   - tile128x64r16x4w16x8k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x64r16x4w16x8k32: i8a8-linear-m185-n1280-k1280: ×0.992 < ×1.050
 *   - tile128x64r16x4w16x8k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x64r16x4w16x8k64: i8a8-linear-m185-n1280-k1280: ×0.975 < ×1.050
 *   - tile128x64r16x4w16x8k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x64r16x8w8x8k16: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x64r16x8w8x8k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x64r16x8w8x8k32: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x64r16x8w8x8k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x64r16x8w8x8k64: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x64r16x8w8x8k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x64r8x4w16x16k16: geometric mean ×1.087 (below the adopted ×2.188)
 *   - tile128x64r8x4w16x16k16s: i8a8-linear-m1-n18432-k3072: ×1.009 < ×1.050
 *   - tile128x64r8x4w16x16k32: i8a8-linear-m1-n18432-k3072: ×1.041 < ×1.050
 *   - tile128x64r8x4w16x16k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x64r8x4w16x16k64: i8a8-linear-m1-n18432-k3072: ×1.006 < ×1.050
 *   - tile128x64r8x4w16x16k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x64r8x8w8x16k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x64r8x8w8x16k32: i8a8-linear-m256-n1280-k1280: ×0.991 < ×1.050
 *   - tile128x64r8x8w8x16k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile128x64r8x8w8x16k64: i8a8-linear-m1-n18432-k3072: ×0.987 < ×1.050
 *   - tile128x64r8x8w8x16k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile16x128r4x8w16x4k16: i8a8-linear-m256-n1280-k1280: ×1.015 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-linear-m256-n1280-k1280: ×1.002 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-linear-m256-n1280-k1280: ×1.043 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-linear-m256-n1280-k1280: ×1.041 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-linear-m256-n1280-k1280: ×1.044 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-linear-m256-n1280-k1280: ×1.028 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-linear-m256-n1280-k1280: ×1.005 < ×1.050
 *   - tile192x128r12x8w16x16k16: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x128r12x8w16x16k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x128r12x8w16x16k32: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x128r12x8w16x16k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x32r12x4w8x16k16: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x32r12x4w8x16k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x32r12x4w8x16k32: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x32r12x4w8x16k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x32r12x4w8x16k64: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x32r12x4w8x16k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x64r12x4w16x16k16: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x64r12x4w16x16k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x64r12x4w16x16k32: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x64r12x4w16x16k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x64r12x4w16x16k64: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x64r12x4w16x16k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x64r12x8w8x16k16: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x64r12x8w8x16k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x64r12x8w8x16k32: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x64r12x8w8x16k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x64r12x8w8x16k64: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile192x64r12x8w8x16k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile256x128r16x8w16x16k16: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile256x128r16x8w16x16k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile256x128r16x8w16x16k32: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile256x128r16x8w16x16k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile256x32r16x4w8x16k16: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile256x32r16x4w8x16k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile256x32r16x4w8x16k32: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile256x32r16x4w8x16k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile256x64r16x4w16x16k16: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile256x64r16x4w16x16k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile256x64r16x4w16x16k32: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile256x64r16x4w16x16k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile256x64r16x8w8x16k16: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile256x64r16x8w8x16k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile256x64r16x8w8x16k32: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile256x64r16x8w8x16k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile32x128r4x8w16x8k16: geometric mean ×1.752 (below the adopted ×2.188)
 *   - tile32x128r4x8w16x8k32: geometric mean ×1.754 (below the adopted ×2.188)
 *   - tile32x128r8x8w16x4k16: i8a8-linear-m256-n1280-k1280: ×0.968 < ×1.050
 *   - tile32x128r8x8w16x4k16s: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile32x128r8x8w16x4k32: i8a8-linear-m256-n1280-k1280: ×0.965 < ×1.050
 *   - tile32x128r8x8w16x4k32s: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile32x128r8x8w16x4k64: i8a8-linear-m256-n1280-k1280: ×0.951 < ×1.050
 *   - tile32x128r8x8w16x4k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile32x32r4x4w8x8k16: i8a8-linear-m256-n1280-k1280: ×1.047 < ×1.050
 *   - tile32x32r4x4w8x8k32: geometric mean ×1.784 (below the adopted ×2.188)
 *   - tile32x32r8x4w8x4k16: geometric mean ×1.812 (below the adopted ×2.188)
 *   - tile32x32r8x4w8x4k16s: geometric mean ×1.865 (below the adopted ×2.188)
 *   - tile32x32r8x4w8x4k32: geometric mean ×1.785 (below the adopted ×2.188)
 *   - tile32x32r8x4w8x4k32s: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile32x32r8x4w8x4k64: i8a8-linear-m256-n1280-k1280: ×1.017 < ×1.050
 *   - tile32x32r8x4w8x4k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile32x64r4x4w16x8k16: geometric mean ×1.778 (below the adopted ×2.188)
 *   - tile32x64r4x4w16x8k32: geometric mean ×1.788 (below the adopted ×2.188)
 *   - tile32x64r4x8w8x8k16: geometric mean ×1.787 (below the adopted ×2.188)
 *   - tile32x64r4x8w8x8k32: i8a8-linear-m256-n1280-k1280: ×1.037 < ×1.050
 *   - tile32x64r8x4w16x4k16: geometric mean ×1.804 (below the adopted ×2.188)
 *   - tile32x64r8x4w16x4k16s: geometric mean ×1.829 (below the adopted ×2.188)
 *   - tile32x64r8x4w16x4k32: geometric mean ×1.786 (below the adopted ×2.188)
 *   - tile32x64r8x4w16x4k32s: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile32x64r8x4w16x4k64: i8a8-linear-m256-n1280-k1280: ×1.024 < ×1.050
 *   - tile32x64r8x4w16x4k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile32x64r8x8w8x4k16: i8a8-linear-m256-n1280-k1280: ×0.962 < ×1.050
 *   - tile32x64r8x8w8x4k16s: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile32x64r8x8w8x4k32: i8a8-linear-m256-n1280-k1280: ×0.956 < ×1.050
 *   - tile32x64r8x8w8x4k32s: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile32x64r8x8w8x4k64: i8a8-linear-m256-n1280-k1280: ×0.937 < ×1.050
 *   - tile32x64r8x8w8x4k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile48x128r12x8w16x4k16: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare)
 *   - tile48x128r12x8w16x4k16s: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile48x128r12x8w16x4k32: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare)
 *   - tile48x128r12x8w16x4k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile48x128r12x8w16x4k64: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare)
 *   - tile48x128r12x8w16x4k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile48x32r12x4w8x4k16: i8a8-linear-m256-n1280-k1280: ×0.932 < ×1.050
 *   - tile48x32r12x4w8x4k16s: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare)
 *   - tile48x32r12x4w8x4k32: i8a8-linear-m256-n1280-k1280: ×0.917 < ×1.050
 *   - tile48x32r12x4w8x4k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile48x32r12x4w8x4k64: i8a8-linear-m256-n1280-k1280: ×0.890 < ×1.050
 *   - tile48x32r12x4w8x4k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile48x64r12x4w16x4k16: i8a8-linear-m256-n1280-k1280: ×0.946 < ×1.050
 *   - tile48x64r12x4w16x4k16s: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare)
 *   - tile48x64r12x4w16x4k32: i8a8-linear-m256-n1280-k1280: ×0.930 < ×1.050
 *   - tile48x64r12x4w16x4k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile48x64r12x4w16x4k64: i8a8-linear-m256-n1280-k1280: ×0.923 < ×1.050
 *   - tile48x64r12x4w16x4k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile48x64r12x8w8x4k16: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare)
 *   - tile48x64r12x8w8x4k16s: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile48x64r12x8w8x4k32: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare)
 *   - tile48x64r12x8w8x4k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile48x64r12x8w8x4k64: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare)
 *   - tile48x64r12x8w8x4k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x128r16x8w16x4k16: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile64x128r16x8w16x4k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x128r16x8w16x4k32: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x128r16x8w16x4k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x128r16x8w16x4k64: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x128r16x8w16x4k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x128r4x8w16x16k16: geometric mean ×1.421 (below the adopted ×2.188)
 *   - tile64x128r4x8w16x16k32: geometric mean ×1.438 (below the adopted ×2.188)
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m256-n1280-k1280: ×0.997 < ×1.050
 *   - tile64x128r8x8w16x8k16s: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile64x128r8x8w16x8k32: i8a8-linear-m256-n1280-k1280: ×0.994 < ×1.050
 *   - tile64x128r8x8w16x8k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x128r8x8w16x8k64: i8a8-linear-m256-n1280-k1280: ×0.983 < ×1.050
 *   - tile64x128r8x8w16x8k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x32r16x4w8x4k16: i8a8-linear-m256-n1280-k1280: ×0.959 < ×1.050
 *   - tile64x32r16x4w8x4k16s: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile64x32r16x4w8x4k32: i8a8-linear-m256-n1280-k1280: ×0.947 < ×1.050
 *   - tile64x32r16x4w8x4k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x32r16x4w8x4k64: i8a8-linear-m256-n1280-k1280: ×0.931 < ×1.050
 *   - tile64x32r16x4w8x4k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x32r4x4w8x16k16: geometric mean ×1.426 (below the adopted ×2.188)
 *   - tile64x32r4x4w8x16k32: geometric mean ×1.432 (below the adopted ×2.188)
 *   - tile64x32r8x4w8x8k16: geometric mean ×1.452 (below the adopted ×2.188)
 *   - tile64x32r8x4w8x8k16s: geometric mean ×1.462 (below the adopted ×2.188)
 *   - tile64x32r8x4w8x8k32: geometric mean ×1.442 (below the adopted ×2.188)
 *   - tile64x32r8x4w8x8k32s: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile64x32r8x4w8x8k64: i8a8-linear-m256-n1280-k1280: ×1.018 < ×1.050
 *   - tile64x32r8x4w8x8k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x64r16x4w16x4k16: i8a8-linear-m256-n1280-k1280: ×0.968 < ×1.050
 *   - tile64x64r16x4w16x4k16s: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile64x64r16x4w16x4k32: i8a8-linear-m256-n1280-k1280: ×0.962 < ×1.050
 *   - tile64x64r16x4w16x4k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x64r16x4w16x4k64: i8a8-linear-m256-n1280-k1280: ×0.953 < ×1.050
 *   - tile64x64r16x4w16x4k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x64r16x8w8x4k16: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile64x64r16x8w8x4k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x64r16x8w8x4k32: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x64r16x8w8x4k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x64r16x8w8x4k64: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x64r16x8w8x4k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x64r4x4w16x16k16: geometric mean ×1.447 (below the adopted ×2.188)
 *   - tile64x64r4x4w16x16k32: geometric mean ×1.427 (below the adopted ×2.188)
 *   - tile64x64r4x8w8x16k16: geometric mean ×1.458 (below the adopted ×2.188)
 *   - tile64x64r4x8w8x16k32: geometric mean ×1.456 (below the adopted ×2.188)
 *   - tile64x64r8x4w16x8k16: geometric mean ×1.478 (below the adopted ×2.188)
 *   - tile64x64r8x4w16x8k16s: geometric mean ×1.462 (below the adopted ×2.188)
 *   - tile64x64r8x4w16x8k32: geometric mean ×1.481 (below the adopted ×2.188)
 *   - tile64x64r8x4w16x8k32s: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile64x64r8x4w16x8k64: geometric mean ×1.439 (below the adopted ×2.188)
 *   - tile64x64r8x4w16x8k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x64r8x8w8x8k16: i8a8-linear-m256-n1280-k1280: ×0.991 < ×1.050
 *   - tile64x64r8x8w8x8k16s: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile64x64r8x8w8x8k32: i8a8-linear-m256-n1280-k1280: ×0.980 < ×1.050
 *   - tile64x64r8x8w8x8k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile64x64r8x8w8x8k64: i8a8-linear-m256-n1280-k1280: ×0.972 < ×1.050
 *   - tile64x64r8x8w8x8k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile96x128r12x8w16x8k16: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare)
 *   - tile96x128r12x8w16x8k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile96x128r12x8w16x8k32: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare)
 *   - tile96x128r12x8w16x8k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile96x128r12x8w16x8k64: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare)
 *   - tile96x128r12x8w16x8k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile96x32r12x4w8x8k16: i8a8-linear-m256-n1280-k1280: ×0.945 < ×1.050
 *   - tile96x32r12x4w8x8k16s: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare)
 *   - tile96x32r12x4w8x8k32: i8a8-linear-m256-n1280-k1280: ×0.922 < ×1.050
 *   - tile96x32r12x4w8x8k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile96x32r12x4w8x8k64: i8a8-linear-m256-n1280-k1280: ×0.901 < ×1.050
 *   - tile96x32r12x4w8x8k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile96x64r12x4w16x8k16: i8a8-linear-m256-n1280-k1280: ×0.990 < ×1.050
 *   - tile96x64r12x4w16x8k16s: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare)
 *   - tile96x64r12x4w16x8k32: i8a8-linear-m256-n1280-k1280: ×0.966 < ×1.050
 *   - tile96x64r12x4w16x8k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile96x64r12x4w16x8k64: i8a8-linear-m256-n1280-k1280: ×0.943 < ×1.050
 *   - tile96x64r12x4w16x8k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile96x64r12x8w8x8k16: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare)
 *   - tile96x64r12x8w8x8k16s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile96x64r12x8w8x8k32: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare)
 *   - tile96x64r12x8w8x8k32s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 *   - tile96x64r12x8w8x8k64: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare)
 *   - tile96x64r12x8w8x8k64s: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare)
 * - i8a8.linear 257〜512 (i8a8 linear row count 257〜512, 1 cases): adopted tile32x64r8x4w16x4k16s ×1.073 (×1.073–×1.073)
 *   - tile128x128r16x8w16x8k16: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x128r16x8w16x8k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x128r16x8w16x8k32: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x128r16x8w16x8k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x128r16x8w16x8k64: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x128r16x8w16x8k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x128r8x8w16x16k16: i8a8-linear-m512-n3072-k3072: ×1.001 < ×1.050
 *   - tile128x128r8x8w16x16k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x128r8x8w16x16k32: i8a8-linear-m512-n3072-k3072: ×1.004 < ×1.050
 *   - tile128x128r8x8w16x16k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x128r8x8w16x16k64: i8a8-linear-m512-n3072-k3072: ×0.988 < ×1.050
 *   - tile128x128r8x8w16x16k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x32r16x4w8x8k16: i8a8-linear-m512-n3072-k3072: ×1.017 < ×1.050
 *   - tile128x32r16x4w8x8k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x32r16x4w8x8k32: i8a8-linear-m512-n3072-k3072: ×1.003 < ×1.050
 *   - tile128x32r16x4w8x8k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x32r16x4w8x8k64: i8a8-linear-m512-n3072-k3072: ×0.976 < ×1.050
 *   - tile128x32r16x4w8x8k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x32r8x4w8x16k16: geometric mean ×1.059 (below the adopted ×1.073)
 *   - tile128x32r8x4w8x16k16s: geometric mean ×1.065 (below the adopted ×1.073)
 *   - tile128x32r8x4w8x16k32: geometric mean ×1.063 (below the adopted ×1.073)
 *   - tile128x32r8x4w8x16k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x32r8x4w8x16k64: geometric mean ×1.051 (below the adopted ×1.073)
 *   - tile128x32r8x4w8x16k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x64r16x4w16x8k16: i8a8-linear-m512-n3072-k3072: ×0.998 < ×1.050
 *   - tile128x64r16x4w16x8k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x64r16x4w16x8k32: i8a8-linear-m512-n3072-k3072: ×0.993 < ×1.050
 *   - tile128x64r16x4w16x8k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x64r16x4w16x8k64: i8a8-linear-m512-n3072-k3072: ×0.983 < ×1.050
 *   - tile128x64r16x4w16x8k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x64r16x8w8x8k16: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x64r16x8w8x8k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x64r16x8w8x8k32: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x64r16x8w8x8k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x64r16x8w8x8k64: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x64r16x8w8x8k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x64r8x4w16x16k16: geometric mean ×1.058 (below the adopted ×1.073)
 *   - tile128x64r8x4w16x16k16s: i8a8-linear-m512-n3072-k3072: ×1.012 < ×1.050
 *   - tile128x64r8x4w16x16k32: i8a8-linear-m512-n3072-k3072: ×1.048 < ×1.050
 *   - tile128x64r8x4w16x16k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x64r8x4w16x16k64: i8a8-linear-m512-n3072-k3072: ×1.031 < ×1.050
 *   - tile128x64r8x4w16x16k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x64r8x8w8x16k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x64r8x8w8x16k32: i8a8-linear-m512-n3072-k3072: ×0.992 < ×1.050
 *   - tile128x64r8x8w8x16k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile128x64r8x8w8x16k64: i8a8-linear-m512-n3072-k3072: ×0.987 < ×1.050
 *   - tile128x64r8x8w8x16k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile16x128r4x8w16x4k16: i8a8-linear-m512-n3072-k3072: ×1.005 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-linear-m512-n3072-k3072: ×0.997 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-linear-m512-n3072-k3072: ×1.020 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-linear-m512-n3072-k3072: ×1.005 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-linear-m512-n3072-k3072: ×0.996 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-linear-m512-n3072-k3072: ×1.006 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-linear-m512-n3072-k3072: ×1.015 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.995 < ×1.050
 *   - tile192x128r12x8w16x16k16: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile192x128r12x8w16x16k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile192x128r12x8w16x16k32: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile192x128r12x8w16x16k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile192x32r12x4w8x16k16: i8a8-linear-m512-n3072-k3072: ×0.972 < ×1.050
 *   - tile192x32r12x4w8x16k16s: i8a8-linear-m512-n3072-k3072: ×0.847 < ×1.050
 *   - tile192x32r12x4w8x16k32: i8a8-linear-m512-n3072-k3072: ×0.937 < ×1.050
 *   - tile192x32r12x4w8x16k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile192x32r12x4w8x16k64: i8a8-linear-m512-n3072-k3072: ×0.929 < ×1.050
 *   - tile192x32r12x4w8x16k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile192x64r12x4w16x16k16: i8a8-linear-m512-n3072-k3072: ×0.949 < ×1.050
 *   - tile192x64r12x4w16x16k16s: i8a8-linear-m512-n3072-k3072: ×0.798 < ×1.050
 *   - tile192x64r12x4w16x16k32: i8a8-linear-m512-n3072-k3072: ×0.937 < ×1.050
 *   - tile192x64r12x4w16x16k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile192x64r12x4w16x16k64: i8a8-linear-m512-n3072-k3072: ×0.906 < ×1.050
 *   - tile192x64r12x4w16x16k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile192x64r12x8w8x16k16: i8a8-linear-m512-n3072-k3072: ×0.772 < ×1.050
 *   - tile192x64r12x8w8x16k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile192x64r12x8w8x16k32: i8a8-linear-m512-n3072-k3072: ×0.788 < ×1.050
 *   - tile192x64r12x8w8x16k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile192x64r12x8w8x16k64: i8a8-linear-m512-n3072-k3072: ×0.776 < ×1.050
 *   - tile192x64r12x8w8x16k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile256x128r16x8w16x16k16: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile256x128r16x8w16x16k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile256x128r16x8w16x16k32: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile256x128r16x8w16x16k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile256x32r16x4w8x16k16: i8a8-linear-m512-n3072-k3072: ×0.979 < ×1.050
 *   - tile256x32r16x4w8x16k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile256x32r16x4w8x16k32: i8a8-linear-m512-n3072-k3072: ×0.969 < ×1.050
 *   - tile256x32r16x4w8x16k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile256x64r16x4w16x16k16: i8a8-linear-m512-n3072-k3072: ×0.965 < ×1.050
 *   - tile256x64r16x4w16x16k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile256x64r16x4w16x16k32: i8a8-linear-m512-n3072-k3072: ×0.969 < ×1.050
 *   - tile256x64r16x4w16x16k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile256x64r16x8w8x16k16: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile256x64r16x8w8x16k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile256x64r16x8w8x16k32: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile256x64r16x8w8x16k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile32x128r4x8w16x8k16: i8a8-linear-m512-n3072-k3072: ×1.029 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-linear-m512-n3072-k3072: ×1.031 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-linear-m512-n3072-k3072: ×1.014 < ×1.050
 *   - tile32x128r8x8w16x4k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile32x128r8x8w16x4k32: i8a8-linear-m512-n3072-k3072: ×1.007 < ×1.050
 *   - tile32x128r8x8w16x4k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile32x128r8x8w16x4k64: i8a8-linear-m512-n3072-k3072: ×0.942 < ×1.050
 *   - tile32x128r8x8w16x4k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile32x32r4x4w8x8k16: i8a8-linear-m512-n3072-k3072: ×1.026 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-linear-m512-n3072-k3072: ×1.031 < ×1.050
 *   - tile32x32r8x4w8x4k16: geometric mean ×1.069 (below the adopted ×1.073)
 *   - tile32x32r8x4w8x4k16s: i8a8-linear-m512-n3072-k3072: ×1.005 < ×1.050
 *   - tile32x32r8x4w8x4k32: geometric mean ×1.063 (below the adopted ×1.073)
 *   - tile32x32r8x4w8x4k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile32x32r8x4w8x4k64: i8a8-linear-m512-n3072-k3072: ×1.014 < ×1.050
 *   - tile32x32r8x4w8x4k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile32x64r4x4w16x8k16: i8a8-linear-m512-n3072-k3072: ×1.015 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-linear-m512-n3072-k3072: ×1.015 < ×1.050
 *   - tile32x64r4x8w8x8k16: geometric mean ×1.055 (below the adopted ×1.073)
 *   - tile32x64r4x8w8x8k32: i8a8-linear-m512-n3072-k3072: ×1.036 < ×1.050
 *   - tile32x64r8x4w16x4k16: geometric mean ×1.065 (below the adopted ×1.073)
 *   - tile32x64r8x4w16x4k32: geometric mean ×1.054 (below the adopted ×1.073)
 *   - tile32x64r8x4w16x4k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile32x64r8x4w16x4k64: i8a8-linear-m512-n3072-k3072: ×1.040 < ×1.050
 *   - tile32x64r8x4w16x4k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile32x64r8x8w8x4k16: i8a8-linear-m512-n3072-k3072: ×1.001 < ×1.050
 *   - tile32x64r8x8w8x4k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile32x64r8x8w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.987 < ×1.050
 *   - tile32x64r8x8w8x4k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile32x64r8x8w8x4k64: i8a8-linear-m512-n3072-k3072: ×0.915 < ×1.050
 *   - tile32x64r8x8w8x4k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile48x128r12x8w16x4k16: i8a8-linear-m512-n3072-k3072: ×0.893 < ×1.050
 *   - tile48x128r12x8w16x4k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile48x128r12x8w16x4k32: i8a8-linear-m512-n3072-k3072: ×0.853 < ×1.050
 *   - tile48x128r12x8w16x4k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile48x128r12x8w16x4k64: i8a8-linear-m512-n3072-k3072: ×0.825 < ×1.050
 *   - tile48x128r12x8w16x4k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile48x32r12x4w8x4k16: geometric mean ×1.055 (below the adopted ×1.073)
 *   - tile48x32r12x4w8x4k16s: i8a8-linear-m512-n3072-k3072: ×0.942 < ×1.050
 *   - tile48x32r12x4w8x4k32: i8a8-linear-m512-n3072-k3072: ×1.037 < ×1.050
 *   - tile48x32r12x4w8x4k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile48x32r12x4w8x4k64: i8a8-linear-m512-n3072-k3072: ×1.020 < ×1.050
 *   - tile48x32r12x4w8x4k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile48x64r12x4w16x4k16: i8a8-linear-m512-n3072-k3072: ×1.046 < ×1.050
 *   - tile48x64r12x4w16x4k16s: i8a8-linear-m512-n3072-k3072: ×0.949 < ×1.050
 *   - tile48x64r12x4w16x4k32: i8a8-linear-m512-n3072-k3072: ×1.016 < ×1.050
 *   - tile48x64r12x4w16x4k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile48x64r12x4w16x4k64: i8a8-linear-m512-n3072-k3072: ×1.034 < ×1.050
 *   - tile48x64r12x4w16x4k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile48x64r12x8w8x4k16: i8a8-linear-m512-n3072-k3072: ×0.858 < ×1.050
 *   - tile48x64r12x8w8x4k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile48x64r12x8w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.825 < ×1.050
 *   - tile48x64r12x8w8x4k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile48x64r12x8w8x4k64: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile48x64r12x8w8x4k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x128r16x8w16x4k16: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x128r16x8w16x4k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x128r16x8w16x4k32: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x128r16x8w16x4k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x128r16x8w16x4k64: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x128r16x8w16x4k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x128r4x8w16x16k16: i8a8-linear-m512-n3072-k3072: ×1.027 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-linear-m512-n3072-k3072: ×1.034 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m512-n3072-k3072: ×1.000 < ×1.050
 *   - tile64x128r8x8w16x8k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x128r8x8w16x8k32: i8a8-linear-m512-n3072-k3072: ×0.998 < ×1.050
 *   - tile64x128r8x8w16x8k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x128r8x8w16x8k64: i8a8-linear-m512-n3072-k3072: ×0.980 < ×1.050
 *   - tile64x128r8x8w16x8k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x32r16x4w8x4k16: i8a8-linear-m512-n3072-k3072: ×0.994 < ×1.050
 *   - tile64x32r16x4w8x4k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x32r16x4w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.971 < ×1.050
 *   - tile64x32r16x4w8x4k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x32r16x4w8x4k64: i8a8-linear-m512-n3072-k3072: ×0.931 < ×1.050
 *   - tile64x32r16x4w8x4k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x32r4x4w8x16k16: i8a8-linear-m512-n3072-k3072: ×1.025 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-linear-m512-n3072-k3072: ×1.026 < ×1.050
 *   - tile64x32r8x4w8x8k16: geometric mean ×1.067 (below the adopted ×1.073)
 *   - tile64x32r8x4w8x8k16s: i8a8-linear-m512-n3072-k3072: ×1.04996 < ×1.05000
 *   - tile64x32r8x4w8x8k32: geometric mean ×1.062 (below the adopted ×1.073)
 *   - tile64x32r8x4w8x8k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x32r8x4w8x8k64: i8a8-linear-m512-n3072-k3072: ×1.035 < ×1.050
 *   - tile64x32r8x4w8x8k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x64r16x4w16x4k16: i8a8-linear-m512-n3072-k3072: ×1.021 < ×1.050
 *   - tile64x64r16x4w16x4k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x64r16x4w16x4k32: i8a8-linear-m512-n3072-k3072: ×1.011 < ×1.050
 *   - tile64x64r16x4w16x4k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x64r16x4w16x4k64: i8a8-linear-m512-n3072-k3072: ×1.000 < ×1.050
 *   - tile64x64r16x4w16x4k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x64r16x8w8x4k16: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x64r16x8w8x4k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x64r16x8w8x4k32: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x64r16x8w8x4k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x64r16x8w8x4k64: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x64r16x8w8x4k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x64r4x4w16x16k16: i8a8-linear-m512-n3072-k3072: ×1.023 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-linear-m512-n3072-k3072: ×1.012 < ×1.050
 *   - tile64x64r4x8w8x16k16: geometric mean ×1.060 (below the adopted ×1.073)
 *   - tile64x64r4x8w8x16k32: geometric mean ×1.062 (below the adopted ×1.073)
 *   - tile64x64r8x4w16x8k16: geometric mean ×1.0727 (below the adopted ×1.0735)
 *   - tile64x64r8x4w16x8k16s: geometric mean ×1.052 (below the adopted ×1.073)
 *   - tile64x64r8x4w16x8k32: geometric mean ×1.072 (below the adopted ×1.073)
 *   - tile64x64r8x4w16x8k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x64r8x4w16x8k64: geometric mean ×1.060 (below the adopted ×1.073)
 *   - tile64x64r8x4w16x8k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x64r8x8w8x8k16: i8a8-linear-m512-n3072-k3072: ×1.032 < ×1.050
 *   - tile64x64r8x8w8x8k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x64r8x8w8x8k32: i8a8-linear-m512-n3072-k3072: ×1.020 < ×1.050
 *   - tile64x64r8x8w8x8k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x64r8x8w8x8k64: i8a8-linear-m512-n3072-k3072: ×0.996 < ×1.050
 *   - tile64x64r8x8w8x8k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile96x128r12x8w16x8k16: i8a8-linear-m512-n3072-k3072: ×0.802 < ×1.050
 *   - tile96x128r12x8w16x8k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile96x128r12x8w16x8k32: i8a8-linear-m512-n3072-k3072: ×0.799 < ×1.050
 *   - tile96x128r12x8w16x8k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile96x128r12x8w16x8k64: i8a8-linear-m512-n3072-k3072: ×0.763 < ×1.050
 *   - tile96x128r12x8w16x8k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile96x32r12x4w8x8k16: i8a8-linear-m512-n3072-k3072: ×0.977 < ×1.050
 *   - tile96x32r12x4w8x8k16s: i8a8-linear-m512-n3072-k3072: ×0.848 < ×1.050
 *   - tile96x32r12x4w8x8k32: i8a8-linear-m512-n3072-k3072: ×0.957 < ×1.050
 *   - tile96x32r12x4w8x8k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile96x32r12x4w8x8k64: i8a8-linear-m512-n3072-k3072: ×0.948 < ×1.050
 *   - tile96x32r12x4w8x8k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile96x64r12x4w16x8k16: i8a8-linear-m512-n3072-k3072: ×0.977 < ×1.050
 *   - tile96x64r12x4w16x8k16s: i8a8-linear-m512-n3072-k3072: ×0.864 < ×1.050
 *   - tile96x64r12x4w16x8k32: i8a8-linear-m512-n3072-k3072: ×0.952 < ×1.050
 *   - tile96x64r12x4w16x8k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile96x64r12x4w16x8k64: i8a8-linear-m512-n3072-k3072: ×0.949 < ×1.050
 *   - tile96x64r12x4w16x8k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile96x64r12x8w8x8k16: i8a8-linear-m512-n3072-k3072: ×0.800 < ×1.050
 *   - tile96x64r12x8w8x8k16s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile96x64r12x8w8x8k32: i8a8-linear-m512-n3072-k3072: ×0.792 < ×1.050
 *   - tile96x64r12x8w8x8k32s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile96x64r12x8w8x8k64: i8a8-linear-m512-n3072-k3072: ×0.781 < ×1.050
 *   - tile96x64r12x8w8x8k64s: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 * - i8a8.linear > 512 (i8a8 linear row count > 512, 9 cases): keeps the default tile128x64r8x8w8x16k16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - tile128x128r16x8w16x8k16: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x128r16x8w16x8k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x128r16x8w16x8k32: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x128r16x8w16x8k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x128r16x8w16x8k64: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x128r16x8w16x8k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x128r8x8w16x16k16: i8a8-linear-m7920-n3072-k3072: ×0.951 < ×1.050
 *   - tile128x128r8x8w16x16k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x128r8x8w16x16k32: i8a8-linear-m7920-n3072-k14336: ×0.972 < ×1.050
 *   - tile128x128r8x8w16x16k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x128r8x8w16x16k64: i8a8-linear-m4096-n2048-k8192: ×0.963 < ×1.050
 *   - tile128x128r8x8w16x16k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x32r16x4w8x8k16: i8a8-linear-m7920-n3072-k14336: ×0.960 < ×1.050
 *   - tile128x32r16x4w8x8k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x32r16x4w8x8k32: i8a8-linear-m7920-n3072-k14336: ×0.951 < ×1.050
 *   - tile128x32r16x4w8x8k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x32r16x4w8x8k64: i8a8-linear-m7920-n3072-k14336: ×0.924 < ×1.050
 *   - tile128x32r16x4w8x8k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x32r8x4w8x16k16: i8a8-linear-m7920-n3072-k14336: ×0.990 < ×1.050
 *   - tile128x32r8x4w8x16k16s: i8a8-linear-m7920-n3072-k14336: ×0.984 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-linear-m7920-n3072-k14336: ×0.994 < ×1.050
 *   - tile128x32r8x4w8x16k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x32r8x4w8x16k64: i8a8-linear-m7920-n3072-k14336: ×0.971 < ×1.050
 *   - tile128x32r8x4w8x16k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x64r16x4w16x8k16: i8a8-linear-m7920-n3072-k14336: ×0.968 < ×1.050
 *   - tile128x64r16x4w16x8k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x64r16x4w16x8k32: i8a8-linear-m7920-n3072-k14336: ×0.962 < ×1.050
 *   - tile128x64r16x4w16x8k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x64r16x4w16x8k64: i8a8-linear-m7920-n3072-k14336: ×0.951 < ×1.050
 *   - tile128x64r16x4w16x8k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x64r16x8w8x8k16: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x64r16x8w8x8k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x64r16x8w8x8k32: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x64r16x8w8x8k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x64r16x8w8x8k64: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x64r16x8w8x8k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x64r8x4w16x16k16: i8a8-linear-m7920-n3072-k14336: ×0.997 < ×1.050
 *   - tile128x64r8x4w16x16k16s: i8a8-linear-m7920-n3072-k14336: ×0.960 < ×1.050
 *   - tile128x64r8x4w16x16k32: i8a8-linear-m1024-n2048-k8192: ×0.964 < ×1.050
 *   - tile128x64r8x4w16x16k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x64r8x4w16x16k64: i8a8-linear-m7920-n3072-k14336: ×0.967 < ×1.050
 *   - tile128x64r8x4w16x16k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x64r8x8w8x16k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x64r8x8w8x16k32: i8a8-linear-m7920-n3072-k14336: ×0.973 < ×1.050
 *   - tile128x64r8x8w8x16k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile128x64r8x8w8x16k64: i8a8-linear-m7920-n3072-k14336: ×0.967 < ×1.050
 *   - tile128x64r8x8w8x16k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile16x128r4x8w16x4k16: i8a8-linear-m7920-n3072-k14336: ×0.932 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-linear-m7920-n3072-k14336: ×0.890 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-linear-m7920-n3072-k14336: ×0.918 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-linear-m7920-n3072-k14336: ×0.893 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-linear-m7920-n3072-k14336: ×0.902 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-linear-m7920-n3072-k14336: ×0.909 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-linear-m7920-n3072-k14336: ×0.913 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-linear-m7920-n3072-k14336: ×0.876 < ×1.050
 *   - tile192x128r12x8w16x16k16: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile192x128r12x8w16x16k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile192x128r12x8w16x16k32: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile192x128r12x8w16x16k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile192x32r12x4w8x16k16: i8a8-linear-m1024-n2048-k8192: ×0.915 < ×1.050
 *   - tile192x32r12x4w8x16k16s: i8a8-linear-m1024-n2048-k2048: ×0.821 < ×1.050
 *   - tile192x32r12x4w8x16k32: i8a8-linear-m1024-n2048-k8192: ×0.887 < ×1.050
 *   - tile192x32r12x4w8x16k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile192x32r12x4w8x16k64: i8a8-linear-m1024-n8192-k2048: ×0.892 < ×1.050
 *   - tile192x32r12x4w8x16k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile192x64r12x4w16x16k16: i8a8-linear-m1024-n2048-k8192: ×0.920 < ×1.050
 *   - tile192x64r12x4w16x16k16s: i8a8-linear-m1024-n2048-k2048: ×0.781 < ×1.050
 *   - tile192x64r12x4w16x16k32: i8a8-linear-m1024-n2048-k2048: ×0.896 < ×1.050
 *   - tile192x64r12x4w16x16k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile192x64r12x4w16x16k64: i8a8-linear-m1024-n2048-k2048: ×0.876 < ×1.050
 *   - tile192x64r12x4w16x16k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile192x64r12x8w8x16k16: i8a8-linear-m1024-n2048-k2048: ×0.747 < ×1.050
 *   - tile192x64r12x8w8x16k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile192x64r12x8w8x16k32: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile192x64r12x8w8x16k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile192x64r12x8w8x16k64: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile192x64r12x8w8x16k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile256x128r16x8w16x16k16: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile256x128r16x8w16x16k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile256x128r16x8w16x16k32: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile256x128r16x8w16x16k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile256x32r16x4w8x16k16: i8a8-linear-m7920-n3072-k14336: ×0.949 < ×1.050
 *   - tile256x32r16x4w8x16k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile256x32r16x4w8x16k32: i8a8-linear-m7920-n3072-k14336: ×0.932 < ×1.050
 *   - tile256x32r16x4w8x16k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile256x64r16x4w16x16k16: i8a8-linear-m4096-n2048-k8192: ×0.935 < ×1.050
 *   - tile256x64r16x4w16x16k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile256x64r16x4w16x16k32: i8a8-linear-m4096-n2048-k8192: ×0.937 < ×1.050
 *   - tile256x64r16x4w16x16k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile256x64r16x8w8x16k16: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile256x64r16x8w8x16k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile256x64r16x8w8x16k32: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile256x64r16x8w8x16k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile32x128r4x8w16x8k16: i8a8-linear-m7920-n3072-k14336: ×0.950 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-linear-m7920-n3072-k14336: ×0.939 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-linear-m7920-n3072-k14336: ×0.949 < ×1.050
 *   - tile32x128r8x8w16x4k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile32x128r8x8w16x4k32: i8a8-linear-m7920-n3072-k14336: ×0.933 < ×1.050
 *   - tile32x128r8x8w16x4k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile32x128r8x8w16x4k64: i8a8-linear-m7920-n3072-k14336: ×0.869 < ×1.050
 *   - tile32x128r8x8w16x4k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile32x32r4x4w8x8k16: i8a8-linear-m7920-n3072-k14336: ×0.919 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-linear-m7920-n3072-k14336: ×0.923 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-linear-m7920-n3072-k14336: ×0.963 < ×1.050
 *   - tile32x32r8x4w8x4k16s: i8a8-linear-m1024-n8192-k2048: ×0.987 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-linear-m7920-n3072-k14336: ×0.966 < ×1.050
 *   - tile32x32r8x4w8x4k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile32x32r8x4w8x4k64: i8a8-linear-m7920-n3072-k14336: ×0.945 < ×1.050
 *   - tile32x32r8x4w8x4k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile32x64r4x4w16x8k16: i8a8-linear-m7920-n3072-k14336: ×0.908 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-linear-m7920-n3072-k14336: ×0.921 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-linear-m4096-n2048-k8192: ×0.983 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-linear-m7920-n3072-k14336: ×0.947 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-linear-m7920-n3072-k14336: ×0.978 < ×1.050
 *   - tile32x64r8x4w16x4k16s: i8a8-linear-m7920-n3072-k14336: ×0.966 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-linear-m4096-n2048-k8192: ×0.988 < ×1.050
 *   - tile32x64r8x4w16x4k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile32x64r8x4w16x4k64: i8a8-linear-m7920-n3072-k14336: ×0.938 < ×1.050
 *   - tile32x64r8x4w16x4k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile32x64r8x8w8x4k16: i8a8-linear-m7920-n3072-k14336: ×0.940 < ×1.050
 *   - tile32x64r8x8w8x4k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile32x64r8x8w8x4k32: i8a8-linear-m1024-n2048-k2048: ×0.923 < ×1.050
 *   - tile32x64r8x8w8x4k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile32x64r8x8w8x4k64: i8a8-linear-m1024-n2048-k8192: ×0.834 < ×1.050
 *   - tile32x64r8x8w8x4k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile48x128r12x8w16x4k16: i8a8-linear-m1024-n2048-k8192: ×0.851 < ×1.050
 *   - tile48x128r12x8w16x4k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile48x128r12x8w16x4k32: i8a8-linear-m1024-n8192-k2048: ×0.816 < ×1.050
 *   - tile48x128r12x8w16x4k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile48x128r12x8w16x4k64: i8a8-linear-m1024-n2048-k2048: ×0.762 < ×1.050
 *   - tile48x128r12x8w16x4k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile48x32r12x4w8x4k16: i8a8-linear-m1024-n2048-k8192: ×1.001 < ×1.050
 *   - tile48x32r12x4w8x4k16s: i8a8-linear-m7920-n14336-k3072: ×0.920 < ×1.050
 *   - tile48x32r12x4w8x4k32: i8a8-linear-m1024-n2048-k8192: ×0.977 < ×1.050
 *   - tile48x32r12x4w8x4k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile48x32r12x4w8x4k64: i8a8-linear-m7920-n3072-k14336: ×0.957 < ×1.050
 *   - tile48x32r12x4w8x4k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile48x64r12x4w16x4k16: i8a8-linear-m7920-n3072-k14336: ×1.008 < ×1.050
 *   - tile48x64r12x4w16x4k16s: i8a8-linear-m1024-n2048-k2048: ×0.912 < ×1.050
 *   - tile48x64r12x4w16x4k32: i8a8-linear-m1024-n8192-k2048: ×0.980 < ×1.050
 *   - tile48x64r12x4w16x4k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile48x64r12x4w16x4k64: i8a8-linear-m7920-n3072-k14336: ×0.992 < ×1.050
 *   - tile48x64r12x4w16x4k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile48x64r12x8w8x4k16: i8a8-linear-m1024-n2048-k8192: ×0.835 < ×1.050
 *   - tile48x64r12x8w8x4k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile48x64r12x8w8x4k32: i8a8-linear-m1024-n2048-k8192: ×0.793 < ×1.050
 *   - tile48x64r12x8w8x4k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile48x64r12x8w8x4k64: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile48x64r12x8w8x4k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x128r16x8w16x4k16: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x128r16x8w16x4k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x128r16x8w16x4k32: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x128r16x8w16x4k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x128r16x8w16x4k64: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x128r16x8w16x4k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x128r4x8w16x16k16: i8a8-linear-m7920-n3072-k14336: ×0.971 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-linear-m7920-n3072-k14336: ×0.979 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m7920-n3072-k14336: ×0.957 < ×1.050
 *   - tile64x128r8x8w16x8k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x128r8x8w16x8k32: i8a8-linear-m4096-n2048-k8192: ×0.969 < ×1.050
 *   - tile64x128r8x8w16x8k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x128r8x8w16x8k64: i8a8-linear-m7920-n3072-k14336: ×0.947 < ×1.050
 *   - tile64x128r8x8w16x8k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x32r16x4w8x4k16: i8a8-linear-m1024-n2048-k8192: ×0.946 < ×1.050
 *   - tile64x32r16x4w8x4k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x32r16x4w8x4k32: i8a8-linear-m7920-n3072-k14336: ×0.939 < ×1.050
 *   - tile64x32r16x4w8x4k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x32r16x4w8x4k64: i8a8-linear-m7920-n3072-k14336: ×0.876 < ×1.050
 *   - tile64x32r16x4w8x4k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x32r4x4w8x16k16: i8a8-linear-m7920-n3072-k14336: ×0.925 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-linear-m7920-n3072-k14336: ×0.927 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-linear-m7920-n3072-k14336: ×0.985 < ×1.050
 *   - tile64x32r8x4w8x8k16s: i8a8-linear-m7920-n3072-k14336: ×0.994 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-linear-m7920-n3072-k14336: ×0.983 < ×1.050
 *   - tile64x32r8x4w8x8k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x32r8x4w8x8k64: i8a8-linear-m7920-n3072-k14336: ×0.955 < ×1.050
 *   - tile64x32r8x4w8x8k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x64r16x4w16x4k16: i8a8-linear-m7920-n3072-k14336: ×0.943 < ×1.050
 *   - tile64x64r16x4w16x4k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x64r16x4w16x4k32: i8a8-linear-m7920-n3072-k14336: ×0.946 < ×1.050
 *   - tile64x64r16x4w16x4k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x64r16x4w16x4k64: i8a8-linear-m7920-n3072-k14336: ×0.935 < ×1.050
 *   - tile64x64r16x4w16x4k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x64r16x8w8x4k16: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x64r16x8w8x4k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x64r16x8w8x4k32: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x64r16x8w8x4k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x64r16x8w8x4k64: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x64r16x8w8x4k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x64r4x4w16x16k16: i8a8-linear-m7920-n3072-k14336: ×0.932 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-linear-m7920-n3072-k14336: ×0.924 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-linear-m7920-n3072-k14336: ×0.986 < ×1.050
 *   - tile64x64r4x8w8x16k32: i8a8-linear-m7920-n3072-k14336: ×0.966 < ×1.050
 *   - tile64x64r8x4w16x8k16: i8a8-linear-m7920-n3072-k14336: ×1.001 < ×1.050
 *   - tile64x64r8x4w16x8k16s: i8a8-linear-m7920-n3072-k14336: ×0.981 < ×1.050
 *   - tile64x64r8x4w16x8k32: i8a8-linear-m7920-n3072-k14336: ×1.003 < ×1.050
 *   - tile64x64r8x4w16x8k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x64r8x4w16x8k64: i8a8-linear-m7920-n3072-k14336: ×0.972 < ×1.050
 *   - tile64x64r8x4w16x8k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x64r8x8w8x8k16: i8a8-linear-m7920-n3072-k14336: ×0.973 < ×1.050
 *   - tile64x64r8x8w8x8k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x64r8x8w8x8k32: i8a8-linear-m7920-n3072-k14336: ×0.958 < ×1.050
 *   - tile64x64r8x8w8x8k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x64r8x8w8x8k64: i8a8-linear-m7920-n3072-k14336: ×0.951 < ×1.050
 *   - tile64x64r8x8w8x8k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile96x128r12x8w16x8k16: i8a8-linear-m1024-n2048-k8192: ×0.863 < ×1.050
 *   - tile96x128r12x8w16x8k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile96x128r12x8w16x8k32: i8a8-linear-m1024-n2048-k2048: ×0.828 < ×1.050
 *   - tile96x128r12x8w16x8k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile96x128r12x8w16x8k64: i8a8-linear-m1024-n2048-k2048: ×0.826 < ×1.050
 *   - tile96x128r12x8w16x8k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile96x32r12x4w8x8k16: i8a8-linear-m7920-n3072-k14336: ×0.989 < ×1.050
 *   - tile96x32r12x4w8x8k16s: i8a8-linear-m1024-n2048-k8192: ×0.895 < ×1.050
 *   - tile96x32r12x4w8x8k32: i8a8-linear-m1024-n2048-k8192: ×0.970 < ×1.050
 *   - tile96x32r12x4w8x8k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile96x32r12x4w8x8k64: i8a8-linear-m1024-n2048-k8192: ×0.955 < ×1.050
 *   - tile96x32r12x4w8x8k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile96x64r12x4w16x8k16: i8a8-linear-m1024-n2048-k2048: ×0.998 < ×1.050
 *   - tile96x64r12x4w16x8k16s: i8a8-linear-m1024-n2048-k8192: ×0.895 < ×1.050
 *   - tile96x64r12x4w16x8k32: i8a8-linear-m1024-n2048-k2048: ×1.000 < ×1.050
 *   - tile96x64r12x4w16x8k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile96x64r12x4w16x8k64: i8a8-linear-m1024-n8192-k2048: ×0.981 < ×1.050
 *   - tile96x64r12x4w16x8k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile96x64r12x8w8x8k16: i8a8-linear-m1024-n2048-k8192: ×0.835 < ×1.050
 *   - tile96x64r12x8w8x8k16s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile96x64r12x8w8x8k32: i8a8-linear-m1024-n2048-k8192: ×0.831 < ×1.050
 *   - tile96x64r12x8w8x8k32s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile96x64r12x8w8x8k64: i8a8-linear-m7920-n3072-k3072: ×0.806 < ×1.050
 *   - tile96x64r12x8w8x8k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 * - i8a8.attentionQk (i8a8 attention ①QK, 4 cases): keeps the default tile128x64r8x8w8x16k16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - tile128x128r8x8w16x16k16: i8a8-attention-qk-cross-m1024-n512: ×0.934 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-attention-qk-cross-m1024-n512: ×0.946 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-attention-qk-self-m1024-n1024: ×1.011 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-attention-qk-self-m1024-n1024: ×1.000 < ×1.050
 *   - tile128x64r8x4w16x16k16: i8a8-attention-qk-self-m1024-n1024: ×0.999 < ×1.050
 *   - tile128x64r8x4w16x16k32: i8a8-attention-qk-cross-m4096-n512: ×1.003 < ×1.050
 *   - tile128x64r8x8w8x16k32: i8a8-attention-qk-cross-m4096-n512: ×0.961 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-attention-qk-self-m1024-n1024: ×0.947 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-attention-qk-self-m1024-n1024: ×0.946 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-attention-qk-self-m1024-n1024: ×0.956 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-attention-qk-self-m1024-n1024: ×0.944 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-attention-qk-self-m1024-n1024: ×0.925 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-attention-qk-self-m1024-n1024: ×0.942 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-attention-qk-self-m1024-n1024: ×0.955 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-attention-qk-self-m1024-n1024: ×0.940 < ×1.050
 *   - tile32x128r4x8w16x8k16: i8a8-attention-qk-self-m1024-n1024: ×0.968 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-attention-qk-self-m1024-n1024: ×0.969 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-attention-qk-self-m1024-n1024: ×0.969 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-attention-qk-cross-m4096-n512: ×0.968 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-attention-qk-self-m1024-n1024: ×0.969 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-attention-qk-self-m1024-n1024: ×0.966 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-attention-qk-self-m1024-n1024: ×1.001 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-attention-qk-self-m1024-n1024: ×0.992 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-attention-qk-self-m1024-n1024: ×0.955 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-attention-qk-self-m1024-n1024: ×0.945 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-attention-qk-self-m1024-n1024: ×0.991 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-attention-qk-self-m1024-n1024: ×0.978 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-attention-qk-self-m1024-n1024: ×0.996 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-attention-qk-self-m1024-n1024: ×0.993 < ×1.050
 *   - tile32x64r8x8w8x4k16: i8a8-attention-qk-self-m1024-n1024: ×0.962 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-attention-qk-self-m1024-n1024: ×0.953 < ×1.050
 *   - tile64x128r4x8w16x16k16: i8a8-attention-qk-cross-m4096-n512: ×0.979 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-attention-qk-self-m1024-n1024: ×0.986 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-attention-qk-cross-m4096-n512: ×0.962 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-attention-qk-self-m1024-n1024: ×0.974 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-attention-qk-self-m1024-n1024: ×0.962 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-attention-qk-self-m1024-n1024: ×0.956 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-attention-qk-cross-m4096-n512: ×1.003 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-attention-qk-cross-m4096-n512: ×0.997 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-attention-qk-self-m1024-n1024: ×0.953 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-attention-qk-self-m1024-n1024: ×0.962 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-attention-qk-self-m1024-n1024: ×1.002 < ×1.050
 *   - tile64x64r4x8w8x16k32: i8a8-attention-qk-self-m1024-n1024: ×0.990 < ×1.050
 *   - tile64x64r8x4w16x8k16: i8a8-attention-qk-self-m1024-n1024: ×1.001 < ×1.050
 *   - tile64x64r8x4w16x8k32: i8a8-attention-qk-self-m1024-n1024: ×1.012 < ×1.050
 *   - tile64x64r8x8w8x8k16: i8a8-attention-qk-cross-m4096-n512: ×0.990 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-attention-qk-cross-m4096-n512: ×0.997 < ×1.050
 * - i8a8.attentionPv (i8a8 attention ③PV, 4 cases): keeps the default tile64x128r8x8w16x8k16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - tile128x128r8x8w16x16k16: i8a8-attention-pv-cross-m4096-n512: ×0.959 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-attention-pv-cross-m4096-n512: ×0.966 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-attention-pv-cross-m4096-n512: ×0.963 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-attention-pv-cross-m4096-n512: ×0.952 < ×1.050
 *   - tile128x64r8x4w16x16k16: i8a8-attention-pv-cross-m4096-n512: ×0.999 < ×1.050
 *   - tile128x64r8x4w16x16k32: i8a8-attention-pv-cross-m4096-n512: ×0.984 < ×1.050
 *   - tile128x64r8x8w8x16k16: i8a8-attention-pv-cross-m4096-n512: ×0.956 < ×1.050
 *   - tile128x64r8x8w8x16k32: i8a8-attention-pv-cross-m4096-n512: ×0.957 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-attention-pv-self-m4096-n4096: ×0.975 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-attention-pv-self-m4096-n4096: ×0.955 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-attention-pv-self-m4096-n4096: ×0.952 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-attention-pv-cross-m4096-n512: ×0.944 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-attention-pv-cross-m4096-n512: ×0.949 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-attention-pv-self-m4096-n4096: ×0.944 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-attention-pv-self-m4096-n4096: ×0.962 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-attention-pv-self-m4096-n4096: ×0.948 < ×1.050
 *   - tile32x128r4x8w16x8k16: i8a8-attention-pv-self-m4096-n4096: ×1.009 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-attention-pv-cross-m4096-n512: ×0.990 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-attention-pv-cross-m4096-n512: ×0.979 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-attention-pv-self-m1024-n1024: ×0.963 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-attention-pv-cross-m4096-n512: ×0.956 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-attention-pv-cross-m4096-n512: ×0.946 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-attention-pv-cross-m4096-n512: ×0.955 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-attention-pv-self-m4096-n4096: ×0.949 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-attention-pv-self-m4096-n4096: ×0.954 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-attention-pv-cross-m4096-n512: ×0.953 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-attention-pv-cross-m4096-n512: ×0.980 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-attention-pv-self-m4096-n4096: ×0.960 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-attention-pv-self-m4096-n4096: ×1.010 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-attention-pv-self-m4096-n4096: ×0.994 < ×1.050
 *   - tile32x64r8x8w8x4k16: i8a8-attention-pv-self-m4096-n4096: ×0.949 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-attention-pv-self-m4096-n4096: ×0.936 < ×1.050
 *   - tile64x128r4x8w16x16k16: i8a8-attention-pv-self-m4096-n4096: ×0.996 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-attention-pv-cross-m4096-n512: ×0.988 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-attention-pv-cross-m4096-n512: ×0.957 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-attention-pv-cross-m4096-n512: ×0.933 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-attention-pv-cross-m4096-n512: ×0.926 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-attention-pv-self-m4096-n4096: ×0.965 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-attention-pv-cross-m4096-n512: ×0.960 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-attention-pv-cross-m4096-n512: ×0.952 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-attention-pv-cross-m4096-n512: ×0.965 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-attention-pv-self-m4096-n4096: ×1.002 < ×1.050
 *   - tile64x64r4x8w8x16k32: i8a8-attention-pv-cross-m4096-n512: ×0.993 < ×1.050
 *   - tile64x64r8x4w16x8k16: i8a8-attention-pv-cross-m4096-n512: ×1.006 < ×1.050
 *   - tile64x64r8x4w16x8k32: i8a8-attention-pv-cross-m4096-n512: ×0.997 < ×1.050
 *   - tile64x64r8x8w8x8k16: i8a8-attention-pv-self-m4096-n4096: ×0.911 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-attention-pv-self-m4096-n4096: ×0.957 < ×1.050
 * - i8a8.conv3d (i8a8 conv3d (w8a8 implicit GEMM), 14 cases): keeps the default tile128x64r8x8w8x16k16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - tile128x128r8x8w16x16k16: i8a8-conv3d-cin1024-cout1024-k333-in3x32x32: ×0.934 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-conv3d-cin1024-cout1024-k333-in3x32x32: ×0.926 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-conv3d-cin1024-cout1024-k333-in4x64x64: ×0.966 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.969 < ×1.050
 *   - tile128x64r8x4w16x16k16: i8a8-conv3d-cin1024-cout1024-k333-in4x64x64: ×0.946 < ×1.050
 *   - tile128x64r8x4w16x16k32: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.938 < ×1.050
 *   - tile128x64r8x8w8x16k32: i8a8-conv3d-cin512-cout256-k333-in6x128x128: ×0.983 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.912 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.871 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.879 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.863 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-conv3d-cin512-cout256-k333-in6x256x256: ×0.863 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.865 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.919 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.880 < ×1.050
 *   - tile32x128r4x8w16x8k16: i8a8-conv3d-cin1024-cout1024-k333-in4x64x64: ×0.951 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.941 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.943 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.909 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.909 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-conv3d-cin512-cout256-k333-in6x256x256: ×0.870 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.982 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.949 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-conv3d-cin512-cout256-k333-in6x256x256: ×0.900 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.883 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.963 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.946 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.973 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.956 < ×1.050
 *   - tile32x64r8x8w8x4k16: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.934 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.888 < ×1.050
 *   - tile64x128r4x8w16x16k16: i8a8-conv3d-cin1024-cout1024-k333-in4x64x64: ×0.955 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.956 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-conv3d-cin1024-cout1024-k333-in4x64x64: ×0.978 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.968 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.912 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-conv3d-cin512-cout512-k333-in6x128x128: ×0.891 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-conv3d-cin512-cout512-k333-in6x64x64: ×0.967 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-conv3d-cin512-cout512-k333-in6x128x128: ×0.979 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-conv3d-cin512-cout256-k333-in6x128x128: ×0.902 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.903 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.982 < ×1.050
 *   - tile64x64r4x8w8x16k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.973 < ×1.050
 *   - tile64x64r8x4w16x8k16: i8a8-conv3d-cin1024-cout1024-k333-in4x64x64: ×0.986 < ×1.050
 *   - tile64x64r8x4w16x8k32: i8a8-conv3d-cin512-cout256-k333-in6x128x128: ×0.991 < ×1.050
 *   - tile64x64r8x8w8x8k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.980 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.965 < ×1.050
 */
import type { GeometryProfile } from "../geometry-profile.ts";

export const APPLE_M5: GeometryProfile = {
  id: "apple-m5",
  match: { vendor: "apple", architecture: "metal-3", description: "Apple M5" },
  gemmRows: [
    { maxRows: 16, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 32, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 8, form: "tas-tb-un" } },
    { maxRows: 64, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16, form: "tas-tb-un" } },
    { maxRows: 128, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16, form: "tas-tb-un" } },
    { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16, form: "tas-tb-un" } },
    { maxRows: 512, geometry: { regM: 8, regN: 4, wgX: 8, wgY: 16, form: "tas-tb-un" } },
    { maxRows: Number.POSITIVE_INFINITY, geometry: { regM: 8, regN: 8, wgX: 16, wgY: 16 } },
  ],
  attention: {
    qk: { regM: 8, regN: 8, wgX: 16, wgY: 16 },
    pv: { regM: 8, regN: 8, wgX: 16, wgY: 16 },
  },
  conv2d: {
    rows64: { regM: 8, regN: 4, wgX: 16, wgY: 8, form: "tas-tb-un", decode: "counter" },
    rows32: { regM: 4, regN: 4, wgX: 16, wgY: 8, form: "tas-tb-un", decode: "counter" },
  },
  i8a8: {
    linear: [
      { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 4, tileK: 32 } },
      {
        maxRows: 512,
        geometry: { regM: 8, regN: 4, wgX: 16, wgY: 4, tileK: 16, skeleton: "vec4" },
      },
      {
        maxRows: Number.POSITIVE_INFINITY,
        geometry: { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 },
      },
    ],
    attentionQk: { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 },
    attentionPv: { regM: 8, regN: 8, wgX: 16, wgY: 8, tileK: 16 },
    conv3d: { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 },
  },
  provenance: {
    adapter: {
      vendor: "apple",
      architecture: "metal-3",
      device: "0x0000",
      description: "Apple M5",
    },
    families: {
      gemm: {
        sweep: "outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json",
        sha256: "afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad",
        date: "2026-11-23T15:06:51.706Z",
        candidateSet: "full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36",
        ],
        kernels: "743eba3965aa8f5d",
        caseSet: "e6ddd45226ab2155",
      },
      attention: {
        sweep: "outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json",
        sha256: "afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad",
        date: "2026-11-23T15:06:51.706Z",
        candidateSet: "full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36",
        ],
        kernels: "c3f4a946ccbf1df9",
        caseSet: "9ee211cf69674928",
      },
      conv2d: {
        sweep: "outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json",
        sha256: "afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad",
        date: "2026-11-23T15:06:51.706Z",
        candidateSet: "full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36",
        ],
        kernels: "82887096d13feac2",
        caseSet: "02186e45558e4151",
      },
      "i8a8-linear": {
        sweep: "outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json",
        sha256: "afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad",
        date: "2026-11-23T15:06:51.706Z",
        candidateSet: "full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36",
        ],
        kernels: "221915f80d9eaabf",
        caseSet: "3d32dc6f10559af2",
      },
      "i8a8-attention": {
        sweep: "outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json",
        sha256: "afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad",
        date: "2026-11-23T15:06:51.706Z",
        candidateSet: "full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36",
        ],
        kernels: "43c3180afc0bc5cd",
        caseSet: "8ab1a86b13087640",
      },
      "i8a8-conv3d": {
        sweep: "outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json",
        sha256: "afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad",
        date: "2026-11-23T15:06:51.706Z",
        candidateSet: "full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36",
        ],
        kernels: "a1666319e42c7dcd",
        caseSet: "27ce9da45b359b14",
      },
    },
  },
};
