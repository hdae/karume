/**
 * Geometry profile `apple-metal-3-shared` (**generated — do not edit by hand**).
 *
 * A static table of tile geometries for the adapter match: vendor apple, architecture metal-3.
 * A shared table: derived from the sweep records of 2 devices of that vendor and architecture,
 * Apple M5 (device 0x0000) and Apple M2 (device 0x0000).
 * It is matched without a description, so it predicts the geometries for every adapter of that
 * vendor and architecture no table names by its description (Chrome without its
 * developer-features flag reports none).
 * The runtime only selects the table by the adapter's match; nothing is measured at run time.
 * To change a value, take a sweep with the workbench's `geometry-sweep` entry and regenerate the
 * table with `node tools/geometry-profiles/generate.ts apple-metal-3-shared --shared --records=<path>,<path>[,…]`.
 *
 * Derived in this generation: gemm, attention, conv2d, i8a8-linear, i8a8-attention, i8a8-conv3d.
 *
 * Records of Apple M5 (device 0x0000):
 * - outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json (sha256 afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad, 2026-11-23T15:06:51.706Z)
 *   workbench; Chrome 158 on macOS; backend metal (declared); timestamps not quantised; checkout 8c9364d8481f83e45f18cc29d813ffd3350d1841
 *
 * Records of Apple M2 (device 0x0000):
 * - outputs/workbench/browser-apple-m2/geometry-sweep_2026-11-23T15-53-18.json (sha256 b85e0950e55ff2b9eb14f788614c5f9e02663539da0bce34bb3fe45f7cd62d8e, 2026-11-23T15:53:17.976Z)
 *   workbench; Chrome 158 on macOS; backend metal (declared); timestamps not quantised; checkout 8c9364d8481f83e45f18cc29d813ffd3350d1841
 * - outputs/workbench/browser-apple-m2/geometry-sweep_2026-11-23T15-54-00.json (sha256 f0509a5a73be99ae19c8d7935cbbbec7a192d717c18193b883cd301fb4568db1, 2026-11-23T15:54:00.903Z)
 *   workbench; Chrome 158 on macOS; backend metal (declared); timestamps not quantised; checkout 8c9364d8481f83e45f18cc29d813ffd3350d1841
 * - outputs/workbench/browser-apple-m2/geometry-sweep_2026-11-23T16-05-49.json (sha256 549662a1b508bb221cdb9a92300b3931c423f11b7f660fa460580f3adedf6b20, 2026-11-23T16:05:49.220Z)
 *   workbench; Chrome 158 on macOS; backend metal (declared); timestamps not quantised; checkout 8c9364d8481f83e45f18cc29d813ffd3350d1841
 *
 * Adoption (shared): a geometry replaces the default of a field only when, on every device, on
 * every case of the field that device's records hold, its output is identical to the default
 * geometry's and it is at least ×1.050 faster than the default on that device; among those,
 * the one with the largest geometric mean over the devices of each device's geometric mean is
 * taken, else the default stays. A field some device has no case of keeps the default. No timing
 * of one device is compared with another's.
 * Within one device, a case whose default re-measurement ratio lies outside 0.9–1.1 (or whose
 * re-measurement failed or is missing) is dropped from that record's ratios, as is an observation
 * whose timestamp rounding could move its ratio by more than 1%; output mismatches and failures
 * still count.
 *
 * - gemmRows ≤ 16 (linear / matmul / bmm row count ≤ 16; Apple M5 4 cases, Apple M2 4 cases): keeps the default reg16x16r1x4w4 (no geometry matches the output in every case at ×1.050 or faster on every device)
 *   - reg128x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg128x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg128x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg128x16r8x4w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg128x16r8x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg128x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg128x32r8x4w8: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg128x32r8x4w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg128x32r8x8w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg128x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg128x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg128x64r8x4w16: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg128x64r8x4w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg128x64r8x4w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg128x64r8x8w8: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg128x64r8x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg128x64r8x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x16r1x4w4:tas-tb-un: Apple M5: linear-m16-n3072-k1024: ×0.981 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x16r1x4w4:tb-un: Apple M5: linear-m16-n3072-k1024: ×1.019 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x16r2x4w4: Apple M5: bmm-b16-m16-n64-k128: ×0.826 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x16r2x4w4:tas-tb-un: Apple M5: bmm-b16-m16-n64-k64: ×0.889 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x16r2x4w4:tb-un: Apple M5: bmm-b16-m16-n64-k64: ×0.879 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x16r4x4w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x16r4x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x16r4x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x32r1x8w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x32r1x8w4:tas-tb-un: Apple M5: bmm-b16-m16-n64-k64: ×0.964 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x32r1x8w4:tb-un: Apple M5: bmm-b16-m16-n64-k64: ×0.955 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x32r2x4w8: Apple M5: bmm-b16-m16-n64-k128: ×0.886 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x32r2x4w8:tas-tb-un: Apple M5: bmm-b16-m16-n64-k64: ×1.007 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x32r2x4w8:tb-un: Apple M5: bmm-b16-m16-n64-k64: ×0.977 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x32r2x8w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x32r2x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x32r2x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x32r4x4w8: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x32r4x4w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x32r4x4w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x32r4x8w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x32r4x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x32r4x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x64r2x8w8: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x64r2x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x64r4x4w16: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x64r4x4w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x64r4x4w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x64r4x8w8: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x64r4x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg16x64r4x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x16r2x4w4: Apple M5: screened out by the quick look in linear-m16-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x16r2x4w4:tas-tb-un: Apple M5: linear-m16-n3072-k1024: ×0.851 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x16r2x4w4:tb-un: Apple M5: linear-m16-n3072-k1024: ×0.819 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x16r4x4w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x16r4x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x16r4x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x16r8x4w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x16r8x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r2x4w8: Apple M5: bmm-b16-m16-n64-k128: ×0.810 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r2x4w8:tas-tb-un: Apple M5: bmm-b16-m16-n64-k128: ×0.860 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r2x4w8:tb-un: Apple M5: bmm-b16-m16-n64-k128: ×0.843 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r2x8w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r2x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r2x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r4x4w8: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r4x4w8:tas-tb-un: Apple M5: bmm-b16-m16-n64-k64: ×0.816 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r4x4w8:tb-un: Apple M5: bmm-b16-m16-n64-k128: ×0.803 < ×1.050; Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r4x8w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r4x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r4x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r8x4w8: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r8x4w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r8x4w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r8x8w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x64r2x8w8: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x64r2x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x64r4x4w16: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x64r4x4w16:tas-tb-un: Apple M5: bmm-b16-m16-n64-k128: ×0.829 < ×1.050; Apple M2: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x64r4x4w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x64r4x8w8: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x64r4x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x64r4x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x64r8x4w16: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x64r8x4w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x64r8x4w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x64r8x8w8: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x64r8x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg32x64r8x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg4x16r1x4w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg4x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg4x16r1x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg4x32r1x8w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg4x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg4x32r1x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x16r4x4w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x16r4x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x16r4x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x16r8x4w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x16r8x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x32r4x4w8: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x32r4x4w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x32r4x8w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x32r4x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x32r4x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x32r8x4w8: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x32r8x4w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x32r8x4w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x32r8x8w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x64r4x4w16: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x64r4x4w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x64r4x4w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x64r4x8w8: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x64r4x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x64r4x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x64r8x4w16: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x64r8x4w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x64r8x4w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x64r8x8w8: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x64r8x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg64x64r8x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x16r1x4w4: Apple M5: screened out by the quick look in linear-m16-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m16-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x16r1x4w4:tb-un: Apple M5: screened out by the quick look in linear-m16-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x16r2x4w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x16r2x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x16r2x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x32r1x8w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m16-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x32r1x8w4:tb-un: Apple M5: screened out by the quick look in linear-m16-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x32r2x4w8: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x32r2x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m16-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x32r2x4w8:tb-un: Apple M5: screened out by the quick look in linear-m16-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x32r2x8w4: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x32r2x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x32r2x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x64r2x8w8: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 *   - reg8x64r2x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m16-n64-k128
 * - gemmRows 17〜32 (linear / matmul / bmm row count 17〜32; Apple M5 4 cases, Apple M2 4 cases): keeps the default reg16x16r1x4w4 (no geometry matches the output in every case at ×1.050 or faster on every device)
 *   - reg128x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg128x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg128x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg128x16r8x4w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg128x16r8x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg128x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg128x32r8x4w8: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg128x32r8x4w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg128x32r8x8w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg128x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg128x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg128x64r8x4w16: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg128x64r8x4w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg128x64r8x4w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg128x64r8x8w8: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg128x64r8x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg128x64r8x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x16r1x4w4:tas-tb-un: Apple M5: bmm-b16-m32-n64-k128: ×0.966 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x16r1x4w4:tb-un: Apple M5: bmm-b16-m32-n64-k128: ×0.990 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x16r2x4w4: Apple M5: bmm-b16-m32-n64-k128: ×0.984 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x16r2x4w4:tas-tb-un: Apple M5: linear-m32-n3072-k1024: ×1.039 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x16r2x4w4:tb-un: Apple M5: bmm-b16-m32-n64-k128: ×1.014 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x16r4x4w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x16r4x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x16r4x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x32r1x8w4: Apple M5: bmm-b16-m32-n64-k128: ×0.857 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x32r1x8w4:tas-tb-un: Apple M5: linear-m32-n3072-k1024: ×0.949 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x32r1x8w4:tb-un: Apple M5: linear-m32-n3072-k1024: ×0.972 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x32r2x4w8: Apple M5: bmm-b16-m32-n64-k128: ×1.044 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x32r2x4w8:tas-tb-un: Apple M5: ×1.174 (×1.057–×1.374); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x32r2x4w8:tb-un: Apple M5: ×1.179 (×1.067–×1.425); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x32r2x8w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x32r2x8w4:tas-tb-un: Apple M5: bmm-b16-m32-n64-k128: ×0.843 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x32r2x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x32r4x4w8: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x32r4x4w8:tas-tb-un: Apple M5: bmm-b16-m32-n64-k128: ×0.896 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x32r4x4w8:tb-un: Apple M5: bmm-b16-m32-n64-k128: ×0.840 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x32r4x8w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x32r4x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x32r4x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x64r2x8w8: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x64r2x8w8:tas-tb-un: Apple M5: bmm-b16-m32-n64-k128: ×0.895 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x64r2x8w8:tb-un: Apple M5: bmm-b16-m32-n64-k128: ×0.920 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x64r4x4w16: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x64r4x4w16:tas-tb-un: Apple M5: bmm-b16-m32-n64-k64: ×0.934 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x64r4x4w16:tb-un: Apple M5: bmm-b16-m32-n64-k64: ×0.940 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x64r4x8w8: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x64r4x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg16x64r4x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x16r2x4w4: Apple M5: bmm-b16-m32-n64-k64: ×0.993 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x16r2x4w4:tas-tb-un: Apple M5: ×1.252 (×1.166–×1.484); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x16r2x4w4:tb-un: Apple M5: ×1.237 (×1.123–×1.434); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x16r4x4w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x16r4x4w4:tas-tb-un: Apple M5: bmm-b16-m32-n64-k128: ×0.868 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x16r4x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x16r8x4w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x16r8x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r2x4w8: Apple M5: ×1.260 (×1.061–×1.520); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r2x4w8:tas-tb-un: Apple M5: ×1.297 (×1.136–×1.527); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r2x4w8:tb-un: Apple M5: ×1.270 (×1.117–×1.526); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r2x8w4: Apple M5: bmm-b16-m32-n64-k128: ×0.806 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r2x8w4:tas-tb-un: Apple M5: bmm-b16-m32-n64-k128: ×0.970 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r2x8w4:tb-un: Apple M5: bmm-b16-m32-n64-k128: ×0.972 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r4x4w8: Apple M5: bmm-b16-m32-n64-k128: ×0.804 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r4x4w8:tas-tb-un: Apple M5: bmm-b16-m32-n64-k128: ×1.025 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r4x4w8:tb-un: Apple M5: bmm-b16-m32-n64-k128: ×1.015 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r4x8w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r4x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r4x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r8x4w8: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r8x4w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r8x4w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r8x8w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x64r2x8w8: Apple M5: bmm-b16-m32-n64-k128: ×0.845 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x64r2x8w8:tas-tb-un: Apple M5: bmm-b16-m32-n64-k128: ×1.040 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x64r2x8w8:tb-un: Apple M5: bmm-b16-m32-n64-k128: ×1.022 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x64r4x4w16: Apple M5: bmm-b16-m32-n64-k128: ×0.853 < ×1.050; Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x64r4x4w16:tas-tb-un: Apple M5: ×1.317 (×1.067–×1.656); Apple M2: bmm-b16-m32-n64-k128: ×0.932 < ×1.050
 *   - reg32x64r4x4w16:tb-un: Apple M5: ×1.278 (×1.059–×1.544); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x64r4x8w8: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x64r4x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x64r4x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x64r8x4w16: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x64r8x4w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x64r8x4w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x64r8x8w8: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x64r8x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg32x64r8x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg4x16r1x4w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg4x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg4x16r1x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg4x32r1x8w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg4x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg4x32r1x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x16r4x4w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x16r4x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x16r4x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x16r8x4w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x16r8x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x32r4x4w8: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x32r4x4w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x32r4x8w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x32r4x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x32r4x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x32r8x4w8: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x32r8x4w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x32r8x4w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x32r8x8w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x64r4x4w16: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x64r4x4w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x64r4x4w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x64r4x8w8: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x64r4x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x64r4x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x64r8x4w16: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x64r8x4w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x64r8x4w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x64r8x8w8: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x64r8x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg64x64r8x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x16r1x4w4: Apple M5: screened out by the quick look in linear-m32-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m32-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x16r1x4w4:tb-un: Apple M5: screened out by the quick look in linear-m32-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x16r2x4w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x16r2x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x16r2x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x32r1x8w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m32-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x32r1x8w4:tb-un: Apple M5: screened out by the quick look in linear-m32-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x32r2x4w8: Apple M5: screened out by the quick look in linear-m32-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x32r2x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m32-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x32r2x4w8:tb-un: Apple M5: screened out by the quick look in linear-m32-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x32r2x8w4: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x32r2x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x32r2x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x64r2x8w8: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 *   - reg8x64r2x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m32-n64-k128
 * - gemmRows 33〜64 (linear / matmul / bmm row count 33〜64; Apple M5 4 cases, Apple M2 4 cases): adopted reg64x32r4x4w8:tas-tb-un ×1.503 over the devices (Apple M5 ×1.688 (×1.442–×2.104), Apple M2 ×1.337 (×1.146–×1.750))
 *   - reg128x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg128x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg128x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg128x16r8x4w4: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg128x16r8x4w4:tas-tb-un: Apple M5: bmm-b16-m64-n64-k128: ×0.815 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg128x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg128x32r8x4w8: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg128x32r8x4w8:tas-tb-un: Apple M5: bmm-b16-m64-n64-k128: ×0.824 < ×1.050; Apple M2: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x32r8x4w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg128x32r8x8w4: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg128x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg128x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg128x64r8x4w16: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg128x64r8x4w16:tas-tb-un: Apple M5: bmm-b16-m64-n64-k128: ×0.836 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg128x64r8x4w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg128x64r8x8w8: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg128x64r8x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg128x64r8x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x128r4x8w16:tb-un: Apple M5: bmm-b16-m64-n64-k128: ×0.802 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x16r1x4w4:tas-tb-un: Apple M5: linear-m64-n3072-k1024: ×0.983 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x16r1x4w4:tb-un: Apple M5: linear-m64-n3072-k1024: ×0.998 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x16r2x4w4: Apple M5: ×1.188 (×1.078–×1.247); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x16r2x4w4:tas-tb-un: Apple M5: ×1.219 (×1.103–×1.281); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x16r2x4w4:tb-un: Apple M5: ×1.189 (×1.070–×1.238); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x16r4x4w4: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x16r4x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x16r4x4w4:tb-un: Apple M5: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x32r1x8w4: Apple M5: linear-m64-n3072-k1024: ×1.002 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x32r1x8w4:tas-tb-un: Apple M5: linear-m64-n3072-k1024: ×1.011 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x32r1x8w4:tb-un: Apple M5: linear-m64-n3072-k1024: ×1.029 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x32r2x4w8: Apple M5: ×1.277 (×1.116–×1.372); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x32r2x4w8:tas-tb-un: Apple M5: ×1.277 (×1.121–×1.370); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x32r2x4w8:tb-un: Apple M5: ×1.290 (×1.124–×1.362); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x32r2x8w4: Apple M5: linear-m64-n3072-k1024: ×0.902 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x32r2x8w4:tas-tb-un: Apple M5: linear-m64-n3072-k1024: ×0.935 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x32r2x8w4:tb-un: Apple M5: linear-m64-n3072-k1024: ×0.926 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x32r4x4w8: Apple M5: linear-m64-n3072-k1024: ×0.919 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x32r4x4w8:tas-tb-un: Apple M5: linear-m64-n3072-k1024: ×0.951 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x32r4x4w8:tb-un: Apple M5: linear-m64-n3072-k1024: ×0.923 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x32r4x8w4: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x32r4x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x32r4x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x64r2x8w8: Apple M5: linear-m64-n3072-k1024: ×0.906 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x64r2x8w8:tas-tb-un: Apple M5: linear-m64-n3072-k1024: ×0.909 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x64r2x8w8:tb-un: Apple M5: linear-m64-n3072-k1024: ×0.916 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x64r4x4w16: Apple M5: linear-m64-n3072-k1024: ×0.913 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x64r4x4w16:tas-tb-un: Apple M5: linear-m64-n3072-k1024: ×0.934 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x64r4x4w16:tb-un: Apple M5: linear-m64-n3072-k1024: ×0.935 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x64r4x8w8: Apple M5: bmm-b16-m64-n64-k128: ×0.908 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x64r4x8w8:tas-tb-un: Apple M5: bmm-b16-m64-n64-k128: ×0.959 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg16x64r4x8w8:tb-un: Apple M5: bmm-b16-m64-n64-k128: ×0.922 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x16r2x4w4: Apple M5: ×1.338 (×1.277–×1.466); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x16r2x4w4:tas-tb-un: Apple M5: ×1.408 (×1.349–×1.539); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x16r2x4w4:tb-un: Apple M5: ×1.358 (×1.302–×1.492); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x16r4x4w4: Apple M5: ×1.243 (×1.163–×1.286); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x16r4x4w4:tas-tb-un: Apple M5: ×1.418 (×1.355–×1.462); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x16r4x4w4:tb-un: Apple M5: ×1.290 (×1.219–×1.335); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x16r8x4w4: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x16r8x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r2x4w8: Apple M5: ×1.434 (×1.372–×1.522); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r2x4w8:tas-tb-un: Apple M5: ×1.458 (×1.382–×1.533); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r2x4w8:tb-un: Apple M5: ×1.444 (×1.375–×1.535); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r2x8w4: Apple M5: ×1.402 (×1.291–×1.476); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r2x8w4:tas-tb-un: Apple M5: ×1.505 (×1.427–×1.561); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r2x8w4:tb-un: Apple M5: ×1.505 (×1.422–×1.599); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r4x4w8: Apple M5: ×1.439 (×1.335–×1.529); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r4x4w8:tas-tb-un: Apple M5: ×1.587 (×1.516–×1.656); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r4x4w8:tb-un: Apple M5: ×1.569 (×1.501–×1.657); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r4x8w4: Apple M5: bmm-b16-m64-n64-k128: ×0.976 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r4x8w4:tas-tb-un: Apple M5: bmm-b16-m64-n64-k128: ×1.046 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r4x8w4:tb-un: Apple M5: bmm-b16-m64-n64-k128: ×0.960 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r8x4w8: Apple M5: bmm-b16-m64-n64-k128: ×1.003 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r8x4w8:tas-tb-un: Apple M5: bmm-b16-m64-n64-k128: ×1.022 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r8x4w8:tb-un: Apple M5: bmm-b16-m64-n64-k128: ×0.930 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r8x8w4: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x64r2x8w8: Apple M5: ×1.383 (×1.266–×1.484); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x64r2x8w8:tas-tb-un: Apple M5: ×1.445 (×1.335–×1.501); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x64r2x8w8:tb-un: Apple M5: ×1.450 (×1.329–×1.519); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x64r4x4w16: Apple M5: ×1.437 (×1.316–×1.520); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x64r4x4w16:tas-tb-un: geometric mean ×1.382 (below the adopted ×1.503) over the devices — Apple M5: ×1.589 (×1.443–×1.772); Apple M2: ×1.202 (×1.166–×1.242)
 *   - reg32x64r4x4w16:tb-un: Apple M5: ×1.541 (×1.404–×1.636); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x64r4x8w8: Apple M5: ×1.381 (×1.069–×1.594); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x64r4x8w8:tas-tb-un: Apple M5: ×1.488 (×1.199–×1.718); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x64r4x8w8:tb-un: Apple M5: ×1.462 (×1.145–×1.727); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x64r8x4w16: Apple M5: ×1.371 (×1.094–×1.563); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x64r8x4w16:tas-tb-un: Apple M5: ×1.519 (×1.207–×1.778); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x64r8x4w16:tb-un: Apple M5: bmm-b16-m64-n64-k128: ×1.027 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x64r8x8w8: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x64r8x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg32x64r8x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg4x16r1x4w4: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg4x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg4x16r1x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg4x32r1x8w4: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg4x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg4x32r1x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x16r4x4w4: Apple M5: ×1.385 (×1.184–×1.709); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x16r4x4w4:tas-tb-un: Apple M5: ×1.546 (×1.296–×1.975); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x16r4x4w4:tb-un: Apple M5: ×1.429 (×1.170–×1.785); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x16r8x4w4: Apple M5: bmm-b16-m64-n64-k128: ×0.871 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x16r8x4w4:tas-tb-un: Apple M5: bmm-b16-m64-n64-k128: ×0.910 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x32r4x4w8: Apple M5: ×1.532 (×1.306–×1.863); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x32r4x4w8:tb-un: Apple M5: ×1.603 (×1.381–×2.044); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x32r4x8w4: Apple M5: bmm-b16-m64-n64-k128: ×1.024 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x32r4x8w4:tas-tb-un: Apple M5: ×1.549 (×1.180–×2.133); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x32r4x8w4:tb-un: Apple M5: ×1.512 (×1.095–×1.978); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x32r8x4w8: Apple M5: ×1.426 (×1.081–×1.934); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x32r8x4w8:tas-tb-un: Apple M5: ×1.557 (×1.197–×2.152); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x32r8x4w8:tb-un: Apple M5: bmm-b16-m64-n64-k128: ×0.998 < ×1.050; Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x32r8x8w4: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x64r4x4w16: Apple M5: ×1.519 (×1.284–×1.956); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x64r4x4w16:tas-tb-un: Apple M5: ×1.677 (×1.432–×2.073); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x64r4x4w16:tb-un: Apple M5: ×1.607 (×1.353–×2.072); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x64r4x8w8: Apple M5: ×1.498 (×1.147–×2.044); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x64r4x8w8:tas-tb-un: Apple M5: ×1.615 (×1.316–×2.127); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x64r4x8w8:tb-un: Apple M5: ×1.556 (×1.173–×2.052); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x64r8x4w16: Apple M5: ×1.457 (×1.143–×1.940); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x64r8x4w16:tas-tb-un: Apple M5: ×1.638 (×1.262–×2.130); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x64r8x4w16:tb-un: Apple M5: ×1.494 (×1.060–×2.112); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x64r8x8w8: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x64r8x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg64x64r8x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x16r1x4w4: Apple M5: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x16r1x4w4:tb-un: Apple M5: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x16r2x4w4: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x16r2x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x16r2x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x32r1x8w4: Apple M5: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x32r1x8w4:tb-un: Apple M5: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x32r2x4w8: Apple M5: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x32r2x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x32r2x4w8:tb-un: Apple M5: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x32r2x8w4: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x32r2x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x32r2x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x64r2x8w8: Apple M5: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 *   - reg8x64r2x8w8:tb-un: Apple M5: screened out by the quick look in linear-m64-n3072-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m64-n128-k64
 * - gemmRows 65〜128 (linear / matmul / bmm row count 65〜128; Apple M5 4 cases, Apple M2 4 cases): keeps the default reg64x32r4x4w8 (no geometry matches the output in every case at ×1.050 or faster on every device)
 *   - reg128x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg128x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg128x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg128x16r8x4w4: Apple M5: screened out by the quick look in matmul-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg128x16r8x4w4:tas-tb-un: Apple M5: matmul-m128-n2048-k1024: ×0.893 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg128x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg128x32r8x4w8: Apple M5: bmm-b16-m128-n64-k128: ×0.860 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg128x32r8x4w8:tas-tb-un: Apple M5: bmm-b16-m128-n64-k64: ×0.961 < ×1.050; Apple M2: screened out by the quick look in bmm-b16-m128-n64-k64 (no timing to compare)
 *   - reg128x32r8x4w8:tb-un: Apple M5: bmm-b16-m128-n64-k128: ×0.883 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg128x32r8x8w4: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg128x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg128x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg128x64r8x4w16: Apple M5: bmm-b16-m128-n64-k128: ×0.872 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg128x64r8x4w16:tas-tb-un: Apple M5: bmm-b16-m128-n64-k64: ×0.960 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg128x64r8x4w16:tb-un: Apple M5: matmul-m128-n2048-k1024: ×0.869 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg128x64r8x8w8: Apple M5: bmm-b16-m128-n64-k64: ×0.853 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg128x64r8x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg128x64r8x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x16r1x4w4: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x16r1x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x16r2x4w4: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x16r2x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x16r2x4w4:tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x16r4x4w4: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x16r4x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x16r4x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x32r1x8w4: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x32r1x8w4:tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x32r2x4w8: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x32r2x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x32r2x4w8:tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x32r2x8w4: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x32r2x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x32r2x8w4:tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x32r4x4w8: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x32r4x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x32r4x4w8:tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x32r4x8w4: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x32r4x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x32r4x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x64r2x8w8: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x64r2x8w8:tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x64r4x4w16: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x64r4x4w16:tas-tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x64r4x4w16:tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x64r4x8w8: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x64r4x8w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg16x64r4x8w8:tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x16r2x4w4: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x16r2x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x16r2x4w4:tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x16r4x4w4: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x16r4x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x16r4x4w4:tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x16r8x4w4: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x16r8x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r2x4w8: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r2x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r2x4w8:tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r2x8w4: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r2x8w4:tas-tb-un: Apple M5: linear-m128-n2048-k1024: ×0.822 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r2x8w4:tb-un: Apple M5: linear-m128-n2048-k1024: ×0.806 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r4x4w8: Apple M5: linear-m128-n2048-k1024: ×0.788 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r4x4w8:tas-tb-un: Apple M5: linear-m128-n2048-k1024: ×0.860 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r4x4w8:tb-un: Apple M5: linear-m128-n2048-k1024: ×0.835 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r4x8w4: Apple M5: linear-m128-n2048-k1024: ×0.824 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r4x8w4:tas-tb-un: Apple M5: linear-m128-n2048-k1024: ×0.852 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r4x8w4:tb-un: Apple M5: linear-m128-n2048-k1024: ×0.814 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r8x4w8: Apple M5: linear-m128-n2048-k1024: ×0.808 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r8x4w8:tas-tb-un: Apple M5: linear-m128-n2048-k1024: ×0.838 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r8x4w8:tb-un: Apple M5: linear-m128-n2048-k1024: ×0.856 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r8x8w4: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x64r2x8w8: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x64r2x8w8:tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x64r4x4w16: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x64r4x4w16:tas-tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x64r4x4w16:tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x64r4x8w8: Apple M5: linear-m128-n2048-k1024: ×0.814 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x64r4x8w8:tas-tb-un: Apple M5: linear-m128-n2048-k1024: ×0.843 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x64r4x8w8:tb-un: Apple M5: linear-m128-n2048-k1024: ×0.824 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x64r8x4w16: Apple M5: linear-m128-n2048-k1024: ×0.814 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x64r8x4w16:tas-tb-un: Apple M5: linear-m128-n2048-k1024: ×0.849 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x64r8x4w16:tb-un: Apple M5: linear-m128-n2048-k1024: ×0.856 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x64r8x8w8: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x64r8x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg32x64r8x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg4x16r1x4w4: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg4x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg4x16r1x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg4x32r1x8w4: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg4x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg4x32r1x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x16r4x4w4: Apple M5: matmul-m128-n2048-k1024: ×0.908 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x16r4x4w4:tas-tb-un: Apple M5: bmm-b16-m128-n64-k64: ×1.031 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x16r4x4w4:tb-un: Apple M5: linear-m128-n2048-k1024: ×0.929 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x16r8x4w4: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x16r8x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x32r4x4w8:tas-tb-un: Apple M5: ×1.110 (×1.077–×1.138); Apple M2: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg64x32r4x4w8:tb-un: Apple M5: bmm-b16-m128-n64-k128: ×1.038 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x32r4x8w4: Apple M5: bmm-b16-m128-n64-k128: ×0.942 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x32r4x8w4:tas-tb-un: Apple M5: bmm-b16-m128-n64-k64: ×0.982 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x32r4x8w4:tb-un: Apple M5: bmm-b16-m128-n64-k128: ×0.996 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x32r8x4w8: Apple M5: matmul-m128-n2048-k1024: ×0.885 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x32r8x4w8:tas-tb-un: Apple M5: matmul-m128-n2048-k1024: ×0.928 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x32r8x4w8:tb-un: Apple M5: bmm-b16-m128-n64-k128: ×0.917 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x32r8x8w4: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x64r4x4w16: Apple M5: bmm-b16-m128-n64-k128: ×0.897 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x64r4x4w16:tas-tb-un: Apple M5: linear-m128-n2048-k1024: ×0.989 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x64r4x4w16:tb-un: Apple M5: bmm-b16-m128-n64-k128: ×0.951 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x64r4x8w8: Apple M5: bmm-b16-m128-n64-k128: ×0.915 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x64r4x8w8:tas-tb-un: Apple M5: bmm-b16-m128-n64-k128: ×0.992 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x64r4x8w8:tb-un: Apple M5: bmm-b16-m128-n64-k64: ×0.950 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x64r8x4w16: Apple M5: bmm-b16-m128-n64-k128: ×0.896 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x64r8x4w16:tas-tb-un: Apple M5: bmm-b16-m128-n64-k64: ×1.011 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x64r8x4w16:tb-un: Apple M5: bmm-b16-m128-n64-k128: ×0.923 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x64r8x8w8: Apple M5: bmm-b16-m128-n64-k64: ×0.833 < ×1.050; Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x64r8x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg64x64r8x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x16r1x4w4: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x16r1x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x16r2x4w4: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x16r2x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x16r2x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x32r1x8w4: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x32r1x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x32r2x4w8: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x32r2x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x32r2x4w8:tb-un: Apple M5: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x32r2x8w4: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x32r2x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x32r2x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x64r2x8w8: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 *   - reg8x64r2x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m128-n64-k128
 * - gemmRows 129〜256 (linear / matmul / bmm row count 129〜256; Apple M5 4 cases, Apple M2 4 cases): keeps the default reg64x32r4x4w8 (no geometry matches the output in every case at ×1.050 or faster on every device)
 *   - reg128x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg128x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg128x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg128x16r8x4w4: Apple M5: bmm-b16-m256-n64-k64: ×0.791 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg128x16r8x4w4:tas-tb-un: Apple M5: bmm-b16-m256-n64-k64: ×0.895 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg128x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg128x32r8x4w8: Apple M5: bmm-b16-m256-n64-k64: ×0.910 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg128x32r8x4w8:tas-tb-un: Apple M5: bmm-b16-m256-n64-k64: ×1.022 < ×1.050; Apple M2: bmm-b16-m256-n64-k64: ×0.899 < ×1.050
 *   - reg128x32r8x4w8:tb-un: Apple M5: bmm-b16-m256-n64-k64: ×0.899 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg128x32r8x8w4: Apple M5: bmm-b16-m256-n64-k64: ×0.818 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg128x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg128x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg128x64r8x4w16: Apple M5: bmm-b16-m256-n64-k64: ×0.764 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg128x64r8x4w16:tas-tb-un: Apple M5: bmm-b16-m256-n64-k64: ×0.925 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg128x64r8x4w16:tb-un: Apple M5: bmm-b16-m256-n64-k64: ×0.819 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg128x64r8x8w8: Apple M5: bmm-b16-m256-n64-k64: ×0.821 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg128x64r8x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg128x64r8x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x16r1x4w4: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x16r1x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x16r2x4w4: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x16r2x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x16r2x4w4:tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x16r4x4w4: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x16r4x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x16r4x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x32r1x8w4: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x32r1x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x32r2x4w8: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x32r2x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x32r2x4w8:tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x32r2x8w4: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x32r2x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x32r2x8w4:tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x32r4x4w8: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x32r4x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x32r4x4w8:tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x32r4x8w4: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x32r4x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x32r4x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x64r2x8w8: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x64r2x8w8:tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x64r4x4w16: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x64r4x4w16:tas-tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x64r4x4w16:tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x64r4x8w8: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x64r4x8w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg16x64r4x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x16r2x4w4: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x16r2x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x16r2x4w4:tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x16r4x4w4: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x16r4x4w4:tas-tb-un: Apple M5: linear-m256-n2048-k1024: ×0.819 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x16r4x4w4:tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x16r8x4w4: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x16r8x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r2x4w8: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r2x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r2x4w8:tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r2x8w4: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r2x8w4:tas-tb-un: Apple M5: linear-m256-n2048-k1024: ×0.834 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r2x8w4:tb-un: Apple M5: linear-m256-n2048-k1024: ×0.819 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r4x4w8: Apple M5: linear-m256-n2048-k1024: ×0.808 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r4x4w8:tas-tb-un: Apple M5: linear-m256-n2048-k1024: ×0.885 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r4x4w8:tb-un: Apple M5: linear-m256-n2048-k1024: ×0.852 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r4x8w4: Apple M5: linear-m256-n2048-k1024: ×0.813 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r4x8w4:tas-tb-un: Apple M5: linear-m256-n2048-k1024: ×0.835 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r4x8w4:tb-un: Apple M5: linear-m256-n2048-k1024: ×0.804 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r8x4w8: Apple M5: linear-m256-n2048-k1024: ×0.806 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r8x4w8:tas-tb-un: Apple M5: linear-m256-n2048-k1024: ×0.827 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r8x4w8:tb-un: Apple M5: linear-m256-n2048-k1024: ×0.842 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r8x8w4: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x64r2x8w8: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x64r2x8w8:tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x64r4x4w16: Apple M5: linear-m256-n2048-k1024: ×0.787 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x64r4x4w16:tas-tb-un: Apple M5: linear-m256-n2048-k1024: ×0.815 < ×1.050; Apple M2: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x64r4x4w16:tb-un: Apple M5: linear-m256-n2048-k1024: ×0.813 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x64r4x8w8: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x64r4x8w8:tas-tb-un: Apple M5: linear-m256-n2048-k1024: ×0.821 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x64r4x8w8:tb-un: Apple M5: linear-m256-n2048-k1024: ×0.803 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x64r8x4w16: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x64r8x4w16:tas-tb-un: Apple M5: linear-m256-n2048-k1024: ×0.814 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x64r8x4w16:tb-un: Apple M5: linear-m256-n2048-k1024: ×0.809 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x64r8x8w8: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x64r8x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg32x64r8x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg4x16r1x4w4: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg4x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg4x16r1x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg4x32r1x8w4: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg4x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg4x32r1x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x16r4x4w4: Apple M5: matmul-m256-n2048-k1024: ×0.898 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x16r4x4w4:tas-tb-un: Apple M5: matmul-m256-n2048-k1024: ×1.043 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x16r4x4w4:tb-un: Apple M5: matmul-m256-n2048-k1024: ×0.922 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x16r8x4w4: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x16r8x4w4:tas-tb-un: Apple M5: bmm-b16-m256-n64-k128: ×0.783 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x32r4x4w8:tas-tb-un: Apple M5: ×1.138 (×1.100–×1.194); Apple M2: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg64x32r4x4w8:tb-un: Apple M5: ×1.070 (×1.052–×1.091); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x32r4x8w4: Apple M5: bmm-b16-m256-n64-k64: ×0.956 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x32r4x8w4:tas-tb-un: Apple M5: bmm-b16-m256-n64-k64: ×1.025 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x32r4x8w4:tb-un: Apple M5: linear-m256-n2048-k1024: ×0.996 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x32r8x4w8: Apple M5: matmul-m256-n2048-k1024: ×0.921 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x32r8x4w8:tas-tb-un: Apple M5: ×1.107 (×1.065–×1.153); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x32r8x4w8:tb-un: Apple M5: bmm-b16-m256-n64-k64: ×0.972 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x32r8x8w4: Apple M5: bmm-b16-m256-n64-k64: ×0.804 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x64r4x4w16: Apple M5: bmm-b16-m256-n64-k64: ×0.936 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x64r4x4w16:tas-tb-un: Apple M5: bmm-b16-m256-n64-k64: ×1.047 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x64r4x4w16:tb-un: Apple M5: bmm-b16-m256-n64-k64: ×1.020 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x64r4x8w8: Apple M5: bmm-b16-m256-n64-k64: ×0.936 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x64r4x8w8:tas-tb-un: Apple M5: bmm-b16-m256-n64-k64: ×0.979 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x64r4x8w8:tb-un: Apple M5: bmm-b16-m256-n64-k64: ×0.950 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x64r8x4w16: Apple M5: bmm-b16-m256-n64-k64: ×0.919 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x64r8x4w16:tas-tb-un: Apple M5: bmm-b16-m256-n64-k64: ×1.048 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x64r8x4w16:tb-un: Apple M5: bmm-b16-m256-n64-k64: ×0.964 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x64r8x8w8: Apple M5: bmm-b16-m256-n64-k64: ×0.867 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x64r8x8w8:tas-tb-un: Apple M5: bmm-b16-m256-n64-k64: ×0.867 < ×1.050; Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg64x64r8x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x16r1x4w4: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x16r1x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x16r2x4w4: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x16r2x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x16r2x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x32r1x8w4: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x32r1x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x32r2x4w8: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x32r2x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x32r2x4w8:tb-un: Apple M5: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x32r2x8w4: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x32r2x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x32r2x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x64r2x8w8: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 *   - reg8x64r2x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare); Apple M2: not measured in bmm-b16-m256-n64-k128
 * - gemmRows 257〜512 (linear / matmul / bmm row count 257〜512; Apple M5 5 cases, Apple M2 5 cases): keeps the default reg64x32r4x4w8 (no geometry matches the output in every case at ×1.050 or faster on every device)
 *   - reg128x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg128x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg128x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg128x16r8x4w4: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg128x16r8x4w4:tas-tb-un: Apple M5: bmm-b16-m512-n64-k512: ×0.903 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg128x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg128x32r8x4w8: Apple M5: bmm-b16-m512-n64-k64: ×0.921 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg128x32r8x4w8:tas-tb-un: Apple M5: ×1.135 (×1.057–×1.262); Apple M2: bmm-b16-m512-n64-k512: ×0.830 < ×1.050
 *   - reg128x32r8x4w8:tb-un: Apple M5: bmm-b16-m512-n64-k64: ×0.832 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg128x32r8x8w4: Apple M5: bmm-b16-m512-n64-k512: ×0.924 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg128x32r8x8w4:tas-tb-un: Apple M5: bmm-b16-m512-n64-k512: ×0.877 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg128x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg128x64r8x4w16: Apple M5: bmm-b16-m512-n512-k64: ×0.861 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg128x64r8x4w16:tas-tb-un: Apple M5: bmm-b16-m512-n64-k64: ×0.875 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg128x64r8x4w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg128x64r8x8w8: Apple M5: bmm-b16-m512-n64-k64: ×0.946 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg128x64r8x8w8:tas-tb-un: Apple M5: bmm-b16-m512-n64-k64: ×0.934 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg128x64r8x8w8:tb-un: Apple M5: bmm-b16-m512-n64-k64: ×0.875 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x16r1x4w4: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x16r1x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x16r2x4w4: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x16r2x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x16r2x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x16r4x4w4: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x16r4x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x16r4x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x32r1x8w4: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x32r1x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x32r2x4w8: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x32r2x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x32r2x4w8:tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x32r2x8w4: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x32r2x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x32r2x8w4:tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x32r4x4w8: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x32r4x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x32r4x4w8:tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x32r4x8w4: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x32r4x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x32r4x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x64r2x8w8: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x64r2x8w8:tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x64r4x4w16: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x64r4x4w16:tas-tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x64r4x4w16:tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x64r4x8w8: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x64r4x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg16x64r4x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x16r2x4w4: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x16r2x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x16r2x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x16r4x4w4: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x16r4x4w4:tas-tb-un: Apple M5: linear-m512-n2048-k1024: ×0.811 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x16r4x4w4:tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x16r8x4w4: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x16r8x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r2x4w8: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r2x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r2x4w8:tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r2x8w4: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r2x8w4:tas-tb-un: Apple M5: linear-m512-n2048-k1024: ×0.830 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r2x8w4:tb-un: Apple M5: linear-m512-n2048-k1024: ×0.807 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r4x4w8: Apple M5: linear-m512-n2048-k1024: ×0.805 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r4x4w8:tas-tb-un: Apple M5: linear-m512-n2048-k1024: ×0.882 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r4x4w8:tb-un: Apple M5: linear-m512-n2048-k1024: ×0.842 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r4x8w4: Apple M5: matmul-m512-n2048-k1024: ×0.785 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r4x8w4:tas-tb-un: Apple M5: linear-m512-n2048-k1024: ×0.816 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r4x8w4:tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r8x4w8: Apple M5: linear-m512-n2048-k1024: ×0.785 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r8x4w8:tas-tb-un: Apple M5: linear-m512-n2048-k1024: ×0.822 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r8x4w8:tb-un: Apple M5: linear-m512-n2048-k1024: ×0.827 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r8x8w4: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x64r2x8w8: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x64r2x8w8:tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x64r4x4w16: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x64r4x4w16:tas-tb-un: Apple M5: linear-m512-n2048-k1024: ×0.797 < ×1.050; Apple M2: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg32x64r4x4w16:tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x64r4x8w8: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x64r4x8w8:tas-tb-un: Apple M5: linear-m512-n2048-k1024: ×0.830 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x64r4x8w8:tb-un: Apple M5: linear-m512-n2048-k1024: ×0.797 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x64r8x4w16: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x64r8x4w16:tas-tb-un: Apple M5: linear-m512-n2048-k1024: ×0.859 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x64r8x4w16:tb-un: Apple M5: linear-m512-n2048-k1024: ×0.850 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x64r8x8w8: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x64r8x8w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg32x64r8x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg4x16r1x4w4: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg4x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg4x16r1x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg4x32r1x8w4: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg4x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg4x32r1x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x128r4x8w16: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x128r4x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x128r8x8w16: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x128r8x8w16:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x16r4x4w4: Apple M5: bmm-b16-m512-n512-k64: ×0.833 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x16r4x4w4:tas-tb-un: Apple M5: bmm-b16-m512-n512-k64: ×0.976 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x16r4x4w4:tb-un: Apple M5: bmm-b16-m512-n64-k64: ×0.913 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x16r8x4w4: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x16r8x4w4:tas-tb-un: Apple M5: bmm-b16-m512-n64-k512: ×0.722 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x16r8x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x32r4x4w8:tas-tb-un: Apple M5: bmm-b16-m512-n64-k512: ×1.049 < ×1.050; Apple M2: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare)
 *   - reg64x32r4x4w8:tb-un: Apple M5: bmm-b16-m512-n64-k512: ×1.034 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x32r4x8w4: Apple M5: matmul-m512-n2048-k1024: ×0.974 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x32r4x8w4:tas-tb-un: Apple M5: bmm-b16-m512-n64-k512: ×0.920 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x32r4x8w4:tb-un: Apple M5: linear-m512-n2048-k1024: ×1.009 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x32r8x4w8: Apple M5: bmm-b16-m512-n64-k512: ×0.905 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x32r8x4w8:tas-tb-un: Apple M5: matmul-m512-n2048-k1024: ×1.046 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x32r8x4w8:tb-un: Apple M5: bmm-b16-m512-n64-k512: ×0.812 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x32r8x8w4: Apple M5: bmm-b16-m512-n64-k512: ×0.878 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x32r8x8w4:tas-tb-un: Apple M5: bmm-b16-m512-n64-k512: ×0.854 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x32r8x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x64r4x4w16: Apple M5: bmm-b16-m512-n64-k64: ×0.969 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x64r4x4w16:tas-tb-un: Apple M5: bmm-b16-m512-n64-k64: ×0.994 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x64r4x4w16:tb-un: Apple M5: bmm-b16-m512-n64-k64: ×1.016 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x64r4x8w8: Apple M5: matmul-m512-n2048-k1024: ×0.989 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x64r4x8w8:tas-tb-un: Apple M5: bmm-b16-m512-n64-k64: ×1.008 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x64r4x8w8:tb-un: Apple M5: bmm-b16-m512-n64-k64: ×0.914 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x64r8x4w16: Apple M5: bmm-b16-m512-n64-k512: ×0.955 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x64r8x4w16:tas-tb-un: Apple M5: bmm-b16-m512-n64-k64: ×1.015 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x64r8x4w16:tb-un: Apple M5: bmm-b16-m512-n64-k64: ×0.825 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x64r8x8w8: Apple M5: bmm-b16-m512-n512-k64: ×0.881 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x64r8x8w8:tas-tb-un: Apple M5: bmm-b16-m512-n64-k64: ×0.980 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg64x64r8x8w8:tb-un: Apple M5: bmm-b16-m512-n64-k512: ×0.863 < ×1.050; Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x16r1x4w4: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x16r1x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x16r2x4w4: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x16r2x4w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x16r2x4w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x32r1x8w4: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x32r1x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x32r2x4w8: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x32r2x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m512-n2048-k1024 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x32r2x4w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x32r2x8w4: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x32r2x8w4:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x32r2x8w4:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x64r2x8w8: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 *   - reg8x64r2x8w8:tb-un: Apple M5: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare); Apple M2: not measured in bmm-b16-m512-n512-k64
 * - gemmRows > 512 (linear / matmul / bmm row count > 512; Apple M5 7 cases, Apple M2 7 cases): keeps the default reg128x128r8x8w16 (no geometry matches the output in every case at ×1.050 or faster on every device)
 *   - Sweep outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json: linear-m4096-n2048-k2048 dropped from the ratio material due to default re-measurement ratio ×1.108 out of range (output matches and failures are still checked)
 *   - reg128x128r8x8w16:tas-tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg128x128r8x8w16:tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg128x16r8x4w4: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg128x16r8x4w4:tas-tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg128x16r8x4w4:tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg128x32r8x4w8: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg128x32r8x4w8:tas-tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: linear-m1024-n2048-k8192: ×0.983 < ×1.050
 *   - reg128x32r8x4w8:tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg128x32r8x8w4: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg128x32r8x8w4:tas-tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg128x32r8x8w4:tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg128x64r8x4w16: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg128x64r8x4w16:tas-tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg128x64r8x4w16:tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg128x64r8x8w8: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg128x64r8x8w8:tas-tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg128x64r8x8w8:tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x128r4x8w16: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x128r4x8w16:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x16r1x4w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x16r1x4w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x16r2x4w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x16r2x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x16r2x4w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x16r4x4w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x16r4x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x16r4x4w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x32r1x8w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x32r1x8w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x32r2x4w8: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x32r2x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x32r2x4w8:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x32r2x8w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x32r2x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x32r2x8w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x32r4x4w8: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x32r4x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x32r4x4w8:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x32r4x8w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x32r4x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x32r4x8w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x64r2x8w8: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x64r2x8w8:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x64r4x4w16: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x64r4x4w16:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x64r4x4w16:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x64r4x8w8: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x64r4x8w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg16x64r4x8w8:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x128r4x8w16: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x128r4x8w16:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x128r4x8w16:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x128r8x8w16: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x128r8x8w16:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x128r8x8w16:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x16r2x4w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x16r2x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x16r2x4w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x16r4x4w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x16r4x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x16r4x4w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x16r8x4w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x16r8x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x16r8x4w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r2x4w8: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r2x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r2x4w8:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r2x8w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r2x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r2x8w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r4x4w8: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r4x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r4x4w8:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r4x8w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r4x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r4x8w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r8x4w8: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r8x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r8x4w8:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r8x8w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x32r8x8w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x64r2x8w8: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x64r2x8w8:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x64r4x4w16: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x64r4x4w16:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x64r4x4w16:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x64r4x8w8: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x64r4x8w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x64r4x8w8:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x64r8x4w16: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x64r8x4w16:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x64r8x4w16:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x64r8x8w8: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x64r8x8w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg32x64r8x8w8:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg4x16r1x4w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg4x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg4x16r1x4w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg4x32r1x8w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg4x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg4x32r1x8w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x128r4x8w16: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x128r4x8w16:tas-tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x128r4x8w16:tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x128r8x8w16: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x128r8x8w16:tas-tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x128r8x8w16:tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x16r4x4w4: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x16r4x4w4:tas-tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x16r4x4w4:tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x16r8x4w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x16r8x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x16r8x4w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x32r4x4w8: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x32r4x4w8:tas-tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg64x32r4x4w8:tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x32r4x8w4: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x32r4x8w4:tas-tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x32r4x8w4:tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x32r8x4w8: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x32r8x4w8:tas-tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x32r8x4w8:tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x32r8x8w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x32r8x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x32r8x8w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x64r4x4w16: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x64r4x4w16:tas-tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x64r4x4w16:tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x64r4x8w8: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x64r4x8w8:tas-tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x64r4x8w8:tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x64r8x4w16: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x64r8x4w16:tas-tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x64r8x4w16:tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x64r8x8w8: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x64r8x8w8:tas-tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg64x64r8x8w8:tb-un: Apple M5: not measured in linear-m4096-n2048-k2048 (dropped from the ratio material in every sweep); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x16r1x4w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x16r1x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x16r1x4w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x16r2x4w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x16r2x4w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x16r2x4w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x32r1x8w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x32r1x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x32r1x8w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x32r2x4w8: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x32r2x4w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x32r2x4w8:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x32r2x8w4: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x32r2x8w4:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x32r2x8w4:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x64r2x8w8: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x64r2x8w8:tas-tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 *   - reg8x64r2x8w8:tb-un: Apple M5: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in linear-m1024-n2048-k2048
 * - attention.qk (fused attention f32 ①QK; Apple M5 4 cases, Apple M2 0 cases): keeps the default reg128x128r8x8w16 (no case of this class in the records of Apple M2 (a shared table takes a geometry only for a class every device measured))
 * - attention.pv (fused attention f32 ③PV; Apple M5 4 cases, Apple M2 0 cases): keeps the default reg128x128r8x8w16 (no case of this class in the records of Apple M2 (a shared table takes a geometry only for a class every device measured))
 * - conv2d.rows64 (conv2d implicit GEMM m tile of 64 rows; Apple M5 2 cases, Apple M2 2 cases): adopted igemm64x64:wg16x8:tas-tb-un:dec ×1.495 over the devices (Apple M5 ×1.368 (×1.311–×1.427), Apple M2 ×1.634 (×1.582–×1.688))
 *   - igemm128x128:wg16x16: Apple M5: conv2d-c192-256x256: ×0.861 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x128:wg16x16:dec: Apple M5: conv2d-c192-256x256: ×0.961 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x128:wg16x16:tas-tb-un: Apple M5: conv2d-c192-256x256: ×0.866 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x128:wg16x16:tas-tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.922 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x128:wg16x16:tb-un: Apple M5: conv2d-c192-256x256: ×0.815 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x128:wg16x16:tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.882 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x64:wg16x16: Apple M5: conv2d-c192-256x256: ×0.821 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x64:wg16x16:dec: Apple M5: conv2d-c192-256x256: ×0.904 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x64:wg16x16:tas-tb-un: Apple M5: conv2d-c192-256x256: ×0.945 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x64:wg16x16:tas-tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.980 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x64:wg16x16:tb-un: Apple M5: conv2d-c192-256x256: ×0.860 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x64:wg16x16:tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.947 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x64:wg8x16: Apple M5: conv2d-c192-256x256: ×0.900 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x64:wg8x16:dec: Apple M5: conv2d-c192-256x256: ×0.967 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x64:wg8x16:tas-tb-un: Apple M5: conv2d-c192-256x256: ×0.892 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x64:wg8x16:tas-tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.969 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x64:wg8x16:tb-un: Apple M5: conv2d-c192-256x256: ×0.813 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm128x64:wg8x16:tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.852 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x128:wg16x4: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x128:wg16x4:dec: Apple M5: conv2d-c192-256x256: ×0.861 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x128:wg16x4:tas-tb-un: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x128:wg16x4:tas-tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.921 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x128:wg16x4:tb-un: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x128:wg16x4:tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.877 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg16x4: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg16x4:dec: Apple M5: conv2d-c192-256x256: ×0.841 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg16x4:tas-tb-un: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg16x4:tas-tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.985 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg16x4:tb-un: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg16x4:tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.961 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg8x4: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg8x4:dec: Apple M5: conv2d-c192-256x256: ×0.886 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg8x4:tas-tb-un: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg8x4:tas-tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.894 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg8x4:tb-un: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg8x4:tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.859 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg8x8: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg8x8:dec: Apple M5: conv2d-c192-256x256: ×0.849 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg8x8:tas-tb-un: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg8x8:tas-tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.921 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg8x8:tb-un: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm16x64:wg8x8:tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.928 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x128:wg16x4: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x128:wg16x4:dec: Apple M5: ×1.142 (×1.092–×1.194); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x128:wg16x4:tas-tb-un: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x128:wg16x4:tas-tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.951 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x128:wg16x4:tb-un: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x128:wg16x4:tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.847 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x128:wg16x8: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x128:wg16x8:dec: Apple M5: ×1.116 (×1.055–×1.180); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x128:wg16x8:tas-tb-un: Apple M5: screened out by the quick look in conv2d-c384-128x128 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x128:wg16x8:tas-tb-un:dec: Apple M5: ×1.184 (×1.145–×1.225); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x128:wg16x8:tb-un: Apple M5: screened out by the quick look in conv2d-c384-128x128 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x128:wg16x8:tb-un:dec: Apple M5: ×1.240 (×1.200–×1.281); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg16x4: Apple M5: conv2d-c384-128x128: ×0.765 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg16x4:dec: Apple M5: ×1.151 (×1.116–×1.187); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg16x4:tas-tb-un: Apple M5: screened out by the quick look in conv2d-c384-128x128 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg16x4:tas-tb-un:dec: Apple M5: ×1.217 (×1.173–×1.262); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg16x4:tb-un: Apple M5: screened out by the quick look in conv2d-c384-128x128 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg16x4:tb-un:dec: Apple M5: ×1.230 (×1.189–×1.273); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg16x8: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg16x8:dec: Apple M5: conv2d-c192-256x256: ×1.000 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg16x8:tas-tb-un: Apple M5: conv2d-c384-128x128: ×0.886 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg16x8:tas-tb-un:dec: geometric mean ×1.260 (below the adopted ×1.495) over the devices — Apple M5: ×1.202 (×1.162–×1.244); Apple M2: ×1.321 (×1.282–×1.361)
 *   - igemm32x64:wg16x8:tb-un: Apple M5: conv2d-c384-128x128: ×0.873 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg16x8:tb-un:dec: Apple M5: ×1.177 (×1.136–×1.219); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x16: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x16:dec: Apple M5: conv2d-c192-256x256: ×0.986 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x16:tas-tb-un: Apple M5: conv2d-c192-256x256: ×0.843 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x16:tas-tb-un:dec: Apple M5: ×1.097 (×1.058–×1.137); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x16:tb-un: Apple M5: conv2d-c192-256x256: ×0.854 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x16:tb-un:dec: Apple M5: ×1.105 (×1.064–×1.147); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x4: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x4:dec: Apple M5: ×1.144 (×1.106–×1.182); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x4:tas-tb-un: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x4:tas-tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.865 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x4:tb-un: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x4:tb-un:dec: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x8: Apple M5: screened out by the quick look in conv2d-c384-128x128 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x8:dec: Apple M5: ×1.101 (×1.066–×1.137); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x8:tas-tb-un: Apple M5: screened out by the quick look in conv2d-c384-128x128 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x8:tas-tb-un:dec: Apple M5: ×1.214 (×1.168–×1.262); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x8:tb-un: Apple M5: screened out by the quick look in conv2d-c384-128x128 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm32x64:wg8x8:tb-un:dec: Apple M5: ×1.215 (×1.197–×1.233); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x128:wg16x16: Apple M5: conv2d-c384-128x128: ×0.987 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x128:wg16x16:dec: Apple M5: ×1.204 (×1.132–×1.281); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x128:wg16x16:tas-tb-un: Apple M5: ×1.084 (×1.058–×1.112); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x128:wg16x16:tas-tb-un:dec: Apple M5: ×1.249 (×1.235–×1.264); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x128:wg16x16:tb-un: Apple M5: ×1.131 (×1.115–×1.147); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x128:wg16x16:tb-un:dec: Apple M5: ×1.271 (×1.186–×1.363); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x128:wg16x8:dec: Apple M5: ×1.247 (×1.182–×1.317); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x128:wg16x8:tas-tb-un: Apple M5: conv2d-c192-256x256: ×0.984 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x128:wg16x8:tas-tb-un:dec: Apple M5: ×1.335 (×1.272–×1.401); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x128:wg16x8:tb-un: Apple M5: conv2d-c384-128x128: ×0.851 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x128:wg16x8:tb-un:dec: Apple M5: ×1.131 (×1.072–×1.193); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg16x16: Apple M5: conv2d-c192-256x256: ×0.937 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg16x16:dec: Apple M5: ×1.126 (×1.079–×1.175); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg16x16:tas-tb-un: Apple M5: ×1.073 (×1.052–×1.094); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg16x16:tas-tb-un:dec: Apple M5: ×1.283 (×1.219–×1.350); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg16x16:tb-un: Apple M5: conv2d-c192-256x256: ×1.016 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg16x16:tb-un:dec: Apple M5: ×1.225 (×1.140–×1.315); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg16x8: Apple M5: conv2d-c192-256x256: ×0.967 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg16x8:dec: Apple M5: ×1.226 (×1.179–×1.275); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg16x8:tas-tb-un: Apple M5: ×1.157 (×1.131–×1.184); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg16x8:tb-un: Apple M5: ×1.075 (×1.067–×1.084); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg16x8:tb-un:dec: Apple M5: ×1.312 (×1.265–×1.362); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg8x16: Apple M5: conv2d-c192-256x256: ×1.010 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg8x16:dec: Apple M5: ×1.228 (×1.176–×1.282); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg8x16:tas-tb-un: Apple M5: conv2d-c192-256x256: ×1.021 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg8x16:tas-tb-un:dec: Apple M5: ×1.322 (×1.270–×1.376); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg8x16:tb-un: Apple M5: ×1.119 (×1.108–×1.131); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg8x16:tb-un:dec: Apple M5: ×1.337 (×1.282–×1.394); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg8x8: Apple M5: conv2d-c384-128x128: ×1.027 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg8x8:dec: Apple M5: ×1.243 (×1.166–×1.325); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg8x8:tas-tb-un: Apple M5: conv2d-c192-256x256: ×0.969 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg8x8:tas-tb-un:dec: Apple M5: ×1.285 (×1.231–×1.342); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg8x8:tb-un: Apple M5: conv2d-c384-128x128: ×0.842 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm64x64:wg8x8:tb-un:dec: Apple M5: conv2d-c192-256x256: ×0.997 < ×1.050; Apple M2: not measured in conv2d-c192-256x256
 *   - igemm8x64:wg8x4: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm8x64:wg8x4:dec: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm8x64:wg8x4:tas-tb-un: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm8x64:wg8x4:tas-tb-un:dec: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm8x64:wg8x4:tb-un: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 *   - igemm8x64:wg8x4:tb-un:dec: Apple M5: screened out by the quick look in conv2d-c192-256x256 (no timing to compare); Apple M2: not measured in conv2d-c192-256x256
 * - conv2d.rows32 (conv2d implicit GEMM m tile of 32 rows; Apple M5 1 cases, Apple M2 1 cases): adopted igemm32x64:wg16x8:tas-tb-un:dec ×1.601 over the devices (Apple M5 ×1.488 (×1.488–×1.488), Apple M2 ×1.722 (×1.722–×1.722))
 *   - igemm128x128:wg16x16: Apple M5: ×1.137 (×1.137–×1.137); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x128:wg16x16:dec: Apple M5: ×1.159 (×1.159–×1.159); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x128:wg16x16:tas-tb-un: Apple M5: ×1.121 (×1.121–×1.121); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x128:wg16x16:tas-tb-un:dec: Apple M5: ×1.209 (×1.209–×1.209); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x128:wg16x16:tb-un: Apple M5: ×1.105 (×1.105–×1.105); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x128:wg16x16:tb-un:dec: Apple M5: ×1.133 (×1.133–×1.133); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x64:wg16x16: Apple M5: conv2d-c96-512x512: ×1.042 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x64:wg16x16:dec: Apple M5: ×1.110 (×1.110–×1.110); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x64:wg16x16:tas-tb-un: Apple M5: ×1.156 (×1.156–×1.156); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x64:wg16x16:tas-tb-un:dec: Apple M5: ×1.253 (×1.253–×1.253); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x64:wg16x16:tb-un: Apple M5: ×1.138 (×1.138–×1.138); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x64:wg16x16:tb-un:dec: Apple M5: ×1.192 (×1.192–×1.192); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x64:wg8x16: Apple M5: ×1.175 (×1.175–×1.175); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x64:wg8x16:dec: Apple M5: ×1.144 (×1.144–×1.144); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x64:wg8x16:tas-tb-un: Apple M5: ×1.152 (×1.152–×1.152); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x64:wg8x16:tas-tb-un:dec: Apple M5: ×1.249 (×1.249–×1.249); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x64:wg8x16:tb-un: Apple M5: ×1.122 (×1.122–×1.122); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm128x64:wg8x16:tb-un:dec: Apple M5: ×1.089 (×1.089–×1.089); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x128:wg16x4: Apple M5: screened out by the quick look in conv2d-c96-512x512 (no timing to compare); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x128:wg16x4:dec: Apple M5: ×1.211 (×1.211–×1.211); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x128:wg16x4:tas-tb-un: Apple M5: screened out by the quick look in conv2d-c96-512x512 (no timing to compare); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x128:wg16x4:tas-tb-un:dec: Apple M5: ×1.193 (×1.193–×1.193); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x128:wg16x4:tb-un: Apple M5: screened out by the quick look in conv2d-c96-512x512 (no timing to compare); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x128:wg16x4:tb-un:dec: Apple M5: ×1.134 (×1.134–×1.134); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg16x4: Apple M5: screened out by the quick look in conv2d-c96-512x512 (no timing to compare); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg16x4:dec: Apple M5: ×1.094 (×1.094–×1.094); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg16x4:tas-tb-un: Apple M5: conv2d-c96-512x512: ×0.828 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg16x4:tas-tb-un:dec: Apple M5: ×1.278 (×1.278–×1.278); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg16x4:tb-un: Apple M5: screened out by the quick look in conv2d-c96-512x512 (no timing to compare); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg16x4:tb-un:dec: Apple M5: ×1.244 (×1.244–×1.244); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg8x4: Apple M5: screened out by the quick look in conv2d-c96-512x512 (no timing to compare); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg8x4:dec: Apple M5: ×1.145 (×1.145–×1.145); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg8x4:tas-tb-un: Apple M5: screened out by the quick look in conv2d-c96-512x512 (no timing to compare); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg8x4:tas-tb-un:dec: Apple M5: ×1.152 (×1.152–×1.152); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg8x4:tb-un: Apple M5: screened out by the quick look in conv2d-c96-512x512 (no timing to compare); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg8x4:tb-un:dec: Apple M5: ×1.085 (×1.085–×1.085); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg8x8: Apple M5: screened out by the quick look in conv2d-c96-512x512 (no timing to compare); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg8x8:dec: Apple M5: ×1.098 (×1.098–×1.098); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg8x8:tas-tb-un: Apple M5: screened out by the quick look in conv2d-c96-512x512 (no timing to compare); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg8x8:tas-tb-un:dec: Apple M5: ×1.194 (×1.194–×1.194); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg8x8:tb-un: Apple M5: screened out by the quick look in conv2d-c96-512x512 (no timing to compare); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm16x64:wg8x8:tb-un:dec: Apple M5: ×1.200 (×1.200–×1.200); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x128:wg16x4:dec: Apple M5: ×1.384 (×1.384–×1.384); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x128:wg16x4:tas-tb-un: Apple M5: conv2d-c96-512x512: ×1.009 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x128:wg16x4:tas-tb-un:dec: Apple M5: ×1.213 (×1.213–×1.213); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x128:wg16x4:tb-un: Apple M5: conv2d-c96-512x512: ×0.812 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x128:wg16x4:tb-un:dec: Apple M5: ×1.081 (×1.081–×1.081); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x128:wg16x8: Apple M5: ×1.125 (×1.125–×1.125); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x128:wg16x8:dec: Apple M5: ×1.355 (×1.355–×1.355); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x128:wg16x8:tas-tb-un: Apple M5: ×1.121 (×1.121–×1.121); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x128:wg16x8:tas-tb-un:dec: Apple M5: ×1.472 (×1.472–×1.472); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x128:wg16x8:tb-un: Apple M5: ×1.159 (×1.159–×1.159); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x128:wg16x8:tb-un:dec: Apple M5: ×1.363 (×1.363–×1.363); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg16x4: Apple M5: ×1.137 (×1.137–×1.137); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg16x4:dec: Apple M5: ×1.434 (×1.434–×1.434); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg16x4:tas-tb-un: Apple M5: ×1.152 (×1.152–×1.152); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg16x4:tas-tb-un:dec: Apple M5: ×1.460 (×1.460–×1.460); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg16x4:tb-un: Apple M5: conv2d-c96-512x512: ×1.021 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg16x4:tb-un:dec: Apple M5: ×1.471 (×1.471–×1.471); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg16x8: Apple M5: ×1.050 (×1.050–×1.050); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg16x8:dec: Apple M5: ×1.286 (×1.286–×1.286); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg16x8:tas-tb-un: Apple M5: ×1.186 (×1.186–×1.186); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg16x8:tb-un: Apple M5: ×1.166 (×1.166–×1.166); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg16x8:tb-un:dec: Apple M5: ×1.448 (×1.448–×1.448); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x16: Apple M5: ×1.052 (×1.052–×1.052); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x16:dec: Apple M5: ×1.263 (×1.263–×1.263); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x16:tas-tb-un: Apple M5: ×1.125 (×1.125–×1.125); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x16:tas-tb-un:dec: Apple M5: ×1.355 (×1.355–×1.355); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x16:tb-un: Apple M5: ×1.141 (×1.141–×1.141); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x16:tb-un:dec: Apple M5: ×1.363 (×1.363–×1.363); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x4: Apple M5: conv2d-c96-512x512: ×0.986 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x4:dec: Apple M5: ×1.375 (×1.375–×1.375); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x4:tas-tb-un: Apple M5: conv2d-c96-512x512: ×0.880 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x4:tas-tb-un:dec: Apple M5: ×1.126 (×1.126–×1.126); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x4:tb-un: Apple M5: screened out by the quick look in conv2d-c96-512x512 (no timing to compare); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x4:tb-un:dec: Apple M5: conv2d-c96-512x512: ×1.026 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x8: Apple M5: ×1.095 (×1.095–×1.095); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x8:dec: Apple M5: ×1.360 (×1.360–×1.360); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x8:tas-tb-un: Apple M5: ×1.114 (×1.114–×1.114); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x8:tas-tb-un:dec: Apple M5: ×1.469 (×1.469–×1.469); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x8:tb-un: Apple M5: ×1.136 (×1.136–×1.136); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm32x64:wg8x8:tb-un:dec: Apple M5: ×1.473 (×1.473–×1.473); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x128:wg16x16: Apple M5: conv2d-c96-512x512: ×0.992 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x128:wg16x16:dec: Apple M5: conv2d-c96-512x512: ×1.036 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x128:wg16x16:tas-tb-un: Apple M5: ×1.088 (×1.088–×1.088); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x128:wg16x16:tas-tb-un:dec: Apple M5: ×1.197 (×1.197–×1.197); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x128:wg16x16:tb-un: Apple M5: ×1.079 (×1.079–×1.079); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x128:wg16x16:tb-un:dec: Apple M5: ×1.196 (×1.196–×1.196); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x128:wg16x8: Apple M5: conv2d-c96-512x512: ×1.011 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x128:wg16x8:dec: Apple M5: ×1.169 (×1.169–×1.169); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x128:wg16x8:tas-tb-un: Apple M5: conv2d-c96-512x512: ×0.990 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x128:wg16x8:tas-tb-un:dec: Apple M5: ×1.231 (×1.231–×1.231); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x128:wg16x8:tb-un: Apple M5: conv2d-c96-512x512: ×0.921 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x128:wg16x8:tb-un:dec: Apple M5: conv2d-c96-512x512: ×1.046 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg16x16: Apple M5: conv2d-c96-512x512: ×0.931 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg16x16:dec: Apple M5: conv2d-c96-512x512: ×1.046 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg16x16:tas-tb-un: Apple M5: ×1.053 (×1.053–×1.053); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg16x16:tas-tb-un:dec: Apple M5: ×1.195 (×1.195–×1.195); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg16x16:tb-un: Apple M5: conv2d-c96-512x512: ×1.013 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg16x16:tb-un:dec: Apple M5: ×1.134 (×1.134–×1.134); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg16x8: Apple M5: conv2d-c96-512x512: ×1.029 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg16x8:dec: Apple M5: ×1.137 (×1.137–×1.137); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg16x8:tas-tb-un: Apple M5: ×1.131 (×1.131–×1.131); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg16x8:tas-tb-un:dec: geometric mean ×1.436 (below the adopted ×1.601) over the devices — Apple M5: ×1.267 (×1.267–×1.267); Apple M2: ×1.627 (×1.627–×1.627)
 *   - igemm64x64:wg16x8:tb-un: Apple M5: ×1.067 (×1.067–×1.067); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg16x8:tb-un:dec: Apple M5: ×1.218 (×1.218–×1.218); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg8x16: Apple M5: conv2d-c96-512x512: ×1.018 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg8x16:dec: Apple M5: ×1.135 (×1.135–×1.135); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg8x16:tas-tb-un: Apple M5: ×1.103 (×1.103–×1.103); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg8x16:tas-tb-un:dec: Apple M5: ×1.234 (×1.234–×1.234); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg8x16:tb-un: Apple M5: ×1.128 (×1.128–×1.128); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg8x16:tb-un:dec: Apple M5: ×1.213 (×1.213–×1.213); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg8x8: Apple M5: conv2d-c96-512x512: ×1.036 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg8x8:dec: Apple M5: ×1.197 (×1.197–×1.197); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg8x8:tas-tb-un: Apple M5: conv2d-c96-512x512: ×1.008 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg8x8:tas-tb-un:dec: Apple M5: ×1.196 (×1.196–×1.196); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg8x8:tb-un: Apple M5: conv2d-c96-512x512: ×0.873 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm64x64:wg8x8:tb-un:dec: Apple M5: conv2d-c96-512x512: ×0.969 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm8x64:wg8x4: Apple M5: screened out by the quick look in conv2d-c96-512x512 (no timing to compare); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm8x64:wg8x4:dec: Apple M5: screened out by the quick look in conv2d-c96-512x512 (no timing to compare); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm8x64:wg8x4:tas-tb-un: Apple M5: screened out by the quick look in conv2d-c96-512x512 (no timing to compare); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm8x64:wg8x4:tas-tb-un:dec: Apple M5: conv2d-c96-512x512: ×0.840 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 *   - igemm8x64:wg8x4:tb-un: Apple M5: screened out by the quick look in conv2d-c96-512x512 (no timing to compare); Apple M2: not measured in conv2d-c96-512x512
 *   - igemm8x64:wg8x4:tb-un:dec: Apple M5: conv2d-c96-512x512: ×0.845 < ×1.050; Apple M2: not measured in conv2d-c96-512x512
 * - i8a8.linear ≤ 256 (i8a8 linear row count ≤ 256; Apple M5 3 cases, Apple M2 3 cases): adopted tile16x64r4x4w16x4k32 ×2.231 over the devices (Apple M5 ×2.188 (×1.060–×7.188), Apple M2 ×2.274 (×1.087–×7.599))
 *   - tile128x128r16x8w16x8k16: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x128r16x8w16x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x128r16x8w16x8k32: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x128r16x8w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x128r16x8w16x8k64: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x128r16x8w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x128r8x8w16x16k16: Apple M5: i8a8-linear-m1-n18432-k3072: ×0.957 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x128r8x8w16x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x128r8x8w16x16k32: Apple M5: i8a8-linear-m1-n18432-k3072: ×0.959 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x128r8x8w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x128r8x8w16x16k64: Apple M5: i8a8-linear-m1-n18432-k3072: ×0.947 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x128r8x8w16x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x32r16x4w8x8k16: Apple M5: i8a8-linear-m185-n1280-k1280: ×0.959 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x32r16x4w8x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x32r16x4w8x8k32: Apple M5: i8a8-linear-m185-n1280-k1280: ×0.947 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x32r16x4w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x32r16x4w8x8k64: Apple M5: i8a8-linear-m185-n1280-k1280: ×0.934 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x32r16x4w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x32r8x4w8x16k16: Apple M5: ×1.076 (×1.055–×1.089); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x32r8x4w8x16k16s: Apple M5: i8a8-linear-m1-n18432-k3072: ×1.039 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x32r8x4w8x16k32: Apple M5: ×1.073 (×1.058–×1.084); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x32r8x4w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x32r8x4w8x16k64: Apple M5: i8a8-linear-m1-n18432-k3072: ×1.020 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x32r8x4w8x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r16x4w16x8k16: Apple M5: i8a8-linear-m1-n18432-k3072: ×0.990 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r16x4w16x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r16x4w16x8k32: Apple M5: i8a8-linear-m185-n1280-k1280: ×0.992 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r16x4w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r16x4w16x8k64: Apple M5: i8a8-linear-m185-n1280-k1280: ×0.975 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r16x4w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r16x8w8x8k16: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r16x8w8x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r16x8w8x8k32: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r16x8w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r16x8w8x8k64: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r16x8w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r8x4w16x16k16: Apple M5: ×1.087 (×1.050–×1.110); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r8x4w16x16k16s: Apple M5: i8a8-linear-m1-n18432-k3072: ×1.009 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r8x4w16x16k32: Apple M5: i8a8-linear-m1-n18432-k3072: ×1.041 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r8x4w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r8x4w16x16k64: Apple M5: i8a8-linear-m1-n18432-k3072: ×1.006 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r8x4w16x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r8x8w8x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r8x8w8x16k32: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.991 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r8x8w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r8x8w8x16k64: Apple M5: i8a8-linear-m1-n18432-k3072: ×0.987 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile128x64r8x8w8x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile16x128r4x8w16x4k16: Apple M5: i8a8-linear-m256-n1280-k1280: ×1.015 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile16x128r4x8w16x4k32: Apple M5: i8a8-linear-m256-n1280-k1280: ×1.002 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile16x32r4x4w8x4k16: Apple M5: i8a8-linear-m256-n1280-k1280: ×1.043 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile16x32r4x4w8x4k32: Apple M5: i8a8-linear-m256-n1280-k1280: ×1.041 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile16x64r4x4w16x4k16: Apple M5: i8a8-linear-m256-n1280-k1280: ×1.044 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile16x64r4x8w8x4k16: Apple M5: i8a8-linear-m256-n1280-k1280: ×1.028 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile16x64r4x8w8x4k32: Apple M5: i8a8-linear-m256-n1280-k1280: ×1.005 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x128r12x8w16x16k16: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x128r12x8w16x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x128r12x8w16x16k32: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x128r12x8w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x32r12x4w8x16k16: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x32r12x4w8x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x32r12x4w8x16k32: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x32r12x4w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x32r12x4w8x16k64: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x32r12x4w8x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x64r12x4w16x16k16: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x64r12x4w16x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x64r12x4w16x16k32: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x64r12x4w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x64r12x4w16x16k64: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x64r12x4w16x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x64r12x8w8x16k16: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x64r12x8w8x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x64r12x8w8x16k32: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x64r12x8w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x64r12x8w8x16k64: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile192x64r12x8w8x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile256x128r16x8w16x16k16: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile256x128r16x8w16x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile256x128r16x8w16x16k32: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile256x128r16x8w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile256x32r16x4w8x16k16: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile256x32r16x4w8x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile256x32r16x4w8x16k32: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile256x32r16x4w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile256x64r16x4w16x16k16: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile256x64r16x4w16x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile256x64r16x4w16x16k32: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile256x64r16x4w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile256x64r16x8w8x16k16: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile256x64r16x8w8x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile256x64r16x8w8x16k32: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile256x64r16x8w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x128r4x8w16x8k16: Apple M5: ×1.752 (×1.059–×3.711); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x128r4x8w16x8k32: Apple M5: ×1.754 (×1.056–×3.738); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x128r8x8w16x4k16: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.968 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x128r8x8w16x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x128r8x8w16x4k32: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.965 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x128r8x8w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x128r8x8w16x4k64: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.951 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x128r8x8w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x32r4x4w8x8k16: Apple M5: i8a8-linear-m256-n1280-k1280: ×1.047 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x32r4x4w8x8k32: Apple M5: ×1.784 (×1.066–×3.858); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x32r8x4w8x4k16: Apple M5: ×1.812 (×1.076–×4.021); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x32r8x4w8x4k16s: Apple M5: ×1.865 (×1.109–×4.123); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x32r8x4w8x4k32: Apple M5: ×1.785 (×1.060–×3.978); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x32r8x4w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x32r8x4w8x4k64: Apple M5: i8a8-linear-m256-n1280-k1280: ×1.017 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x32r8x4w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x64r4x4w16x8k16: Apple M5: ×1.778 (×1.065–×3.816); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x64r4x4w16x8k32: Apple M5: ×1.788 (×1.067–×3.861); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x64r4x8w8x8k16: Apple M5: ×1.787 (×1.063–×3.937); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x64r4x8w8x8k32: Apple M5: i8a8-linear-m256-n1280-k1280: ×1.037 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x64r8x4w16x4k16: Apple M5: ×1.804 (×1.074–×3.964); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x64r8x4w16x4k16s: Apple M5: ×1.829 (×1.087–×4.022); Apple M2: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare)
 *   - tile32x64r8x4w16x4k32: Apple M5: ×1.786 (×1.064–×3.926); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x64r8x4w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x64r8x4w16x4k64: Apple M5: i8a8-linear-m256-n1280-k1280: ×1.024 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x64r8x4w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x64r8x8w8x4k16: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.962 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x64r8x8w8x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x64r8x8w8x4k32: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.956 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x64r8x8w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x64r8x8w8x4k64: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.937 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile32x64r8x8w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x128r12x8w16x4k16: Apple M5: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x128r12x8w16x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x128r12x8w16x4k32: Apple M5: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x128r12x8w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x128r12x8w16x4k64: Apple M5: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x128r12x8w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x32r12x4w8x4k16: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.932 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x32r12x4w8x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x32r12x4w8x4k32: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.917 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x32r12x4w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x32r12x4w8x4k64: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.890 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x32r12x4w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x64r12x4w16x4k16: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.946 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x64r12x4w16x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x64r12x4w16x4k32: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.930 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x64r12x4w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x64r12x4w16x4k64: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.923 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x64r12x4w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x64r12x8w8x4k16: Apple M5: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x64r12x8w8x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x64r12x8w8x4k32: Apple M5: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x64r12x8w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x64r12x8w8x4k64: Apple M5: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile48x64r12x8w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x128r16x8w16x4k16: Apple M5: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x128r16x8w16x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x128r16x8w16x4k32: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x128r16x8w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x128r16x8w16x4k64: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x128r16x8w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x128r4x8w16x16k16: Apple M5: ×1.421 (×1.076–×1.928); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x128r4x8w16x16k32: Apple M5: ×1.438 (×1.082–×1.965); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x128r8x8w16x8k16: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.997 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x128r8x8w16x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x128r8x8w16x8k32: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.994 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x128r8x8w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x128r8x8w16x8k64: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.983 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x128r8x8w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x32r16x4w8x4k16: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.959 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x32r16x4w8x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x32r16x4w8x4k32: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.947 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x32r16x4w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x32r16x4w8x4k64: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.931 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x32r16x4w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x32r4x4w8x16k16: Apple M5: ×1.426 (×1.063–×1.971); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x32r4x4w8x16k32: Apple M5: ×1.432 (×1.065–×1.990); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x32r8x4w8x8k16: Apple M5: ×1.452 (×1.074–×2.071); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x32r8x4w8x8k16s: Apple M5: ×1.462 (×1.084–×2.064); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x32r8x4w8x8k32: Apple M5: ×1.442 (×1.063–×2.062); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x32r8x4w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x32r8x4w8x8k64: Apple M5: i8a8-linear-m256-n1280-k1280: ×1.018 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x32r8x4w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r16x4w16x4k16: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.968 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r16x4w16x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r16x4w16x4k32: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.962 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r16x4w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r16x4w16x4k64: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.953 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r16x4w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r16x8w8x4k16: Apple M5: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r16x8w8x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r16x8w8x4k32: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r16x8w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r16x8w8x4k64: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r16x8w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r4x4w16x16k16: Apple M5: ×1.447 (×1.080–×1.996); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r4x4w16x16k32: Apple M5: ×1.427 (×1.066–×1.963); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r4x8w8x16k16: Apple M5: ×1.458 (×1.083–×2.043); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r4x8w8x16k32: Apple M5: ×1.456 (×1.076–×2.049); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r8x4w16x8k16: Apple M5: ×1.478 (×1.098–×2.070); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r8x4w16x8k16s: Apple M5: ×1.462 (×1.089–×2.040); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r8x4w16x8k32: Apple M5: ×1.481 (×1.098–×2.079); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r8x4w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r8x4w16x8k64: Apple M5: ×1.439 (×1.069–×2.017); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r8x4w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r8x8w8x8k16: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.991 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r8x8w8x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m185-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r8x8w8x8k32: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.980 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r8x8w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r8x8w8x8k64: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.972 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile64x64r8x8w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x128r12x8w16x8k16: Apple M5: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x128r12x8w16x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x128r12x8w16x8k32: Apple M5: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x128r12x8w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x128r12x8w16x8k64: Apple M5: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x128r12x8w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x32r12x4w8x8k16: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.945 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x32r12x4w8x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x32r12x4w8x8k32: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.922 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x32r12x4w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x32r12x4w8x8k64: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.901 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x32r12x4w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x64r12x4w16x8k16: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.990 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x64r12x4w16x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x64r12x4w16x8k32: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.966 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x64r12x4w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x64r12x4w16x8k64: Apple M5: i8a8-linear-m256-n1280-k1280: ×0.943 < ×1.050; Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x64r12x4w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x64r12x8w8x8k16: Apple M5: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x64r12x8w8x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x64r12x8w8x8k32: Apple M5: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x64r12x8w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x64r12x8w8x8k64: Apple M5: screened out by the quick look in i8a8-linear-m256-n1280-k1280 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 *   - tile96x64r12x8w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1-n18432-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m1-n18432-k3072
 * - i8a8.linear 257〜512 (i8a8 linear row count 257〜512; Apple M5 1 cases, Apple M2 1 cases): keeps the default tile128x64r8x8w8x16k16 (no geometry matches the output in every case at ×1.050 or faster on every device)
 *   - tile128x128r16x8w16x8k16: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x128r16x8w16x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x128r16x8w16x8k32: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x128r16x8w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x128r16x8w16x8k64: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x128r16x8w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x128r8x8w16x16k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.001 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x128r8x8w16x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x128r8x8w16x16k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.004 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x128r8x8w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x128r8x8w16x16k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.988 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x128r8x8w16x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x32r16x4w8x8k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.017 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x32r16x4w8x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x32r16x4w8x8k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.003 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x32r16x4w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x32r16x4w8x8k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.976 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x32r16x4w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x32r8x4w8x16k16: Apple M5: ×1.059 (×1.059–×1.059); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x32r8x4w8x16k16s: Apple M5: ×1.065 (×1.065–×1.065); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x32r8x4w8x16k32: Apple M5: ×1.063 (×1.063–×1.063); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x32r8x4w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x32r8x4w8x16k64: Apple M5: ×1.051 (×1.051–×1.051); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x32r8x4w8x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r16x4w16x8k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.998 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r16x4w16x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r16x4w16x8k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.993 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r16x4w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r16x4w16x8k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.983 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r16x4w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r16x8w8x8k16: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r16x8w8x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r16x8w8x8k32: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r16x8w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r16x8w8x8k64: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r16x8w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r8x4w16x16k16: Apple M5: ×1.058 (×1.058–×1.058); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r8x4w16x16k16s: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.012 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r8x4w16x16k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.048 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r8x4w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r8x4w16x16k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.031 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r8x4w16x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r8x8w8x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r8x8w8x16k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.992 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r8x8w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r8x8w8x16k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.987 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile128x64r8x8w8x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile16x128r4x8w16x4k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.005 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile16x128r4x8w16x4k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.997 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile16x32r4x4w8x4k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.020 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile16x32r4x4w8x4k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.005 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile16x64r4x4w16x4k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.996 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile16x64r4x4w16x4k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.006 < ×1.050; Apple M2: i8a8-linear-m512-n3072-k3072: ×1.002 < ×1.050
 *   - tile16x64r4x8w8x4k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.015 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile16x64r4x8w8x4k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.995 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x128r12x8w16x16k16: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x128r12x8w16x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x128r12x8w16x16k32: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x128r12x8w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x32r12x4w8x16k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.972 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x32r12x4w8x16k16s: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.847 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x32r12x4w8x16k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.937 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x32r12x4w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x32r12x4w8x16k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.929 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x32r12x4w8x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x64r12x4w16x16k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.949 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x64r12x4w16x16k16s: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.798 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x64r12x4w16x16k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.937 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x64r12x4w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x64r12x4w16x16k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.906 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x64r12x4w16x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x64r12x8w8x16k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.772 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x64r12x8w8x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x64r12x8w8x16k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.788 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x64r12x8w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x64r12x8w8x16k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.776 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile192x64r12x8w8x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile256x128r16x8w16x16k16: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile256x128r16x8w16x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile256x128r16x8w16x16k32: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile256x128r16x8w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile256x32r16x4w8x16k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.979 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile256x32r16x4w8x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile256x32r16x4w8x16k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.969 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile256x32r16x4w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile256x64r16x4w16x16k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.965 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile256x64r16x4w16x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile256x64r16x4w16x16k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.969 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile256x64r16x4w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile256x64r16x8w8x16k16: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile256x64r16x8w8x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile256x64r16x8w8x16k32: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile256x64r16x8w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x128r4x8w16x8k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.029 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x128r4x8w16x8k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.031 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x128r8x8w16x4k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.014 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x128r8x8w16x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x128r8x8w16x4k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.007 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x128r8x8w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x128r8x8w16x4k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.942 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x128r8x8w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x32r4x4w8x8k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.026 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x32r4x4w8x8k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.031 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x32r8x4w8x4k16: Apple M5: ×1.069 (×1.069–×1.069); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x32r8x4w8x4k16s: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.005 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x32r8x4w8x4k32: Apple M5: ×1.063 (×1.063–×1.063); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x32r8x4w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x32r8x4w8x4k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.014 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x32r8x4w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x64r4x4w16x8k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.015 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x64r4x4w16x8k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.015 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x64r4x8w8x8k16: Apple M5: ×1.055 (×1.055–×1.055); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x64r4x8w8x8k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.036 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x64r8x4w16x4k16: Apple M5: ×1.065 (×1.065–×1.065); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x64r8x4w16x4k16s: Apple M5: ×1.073 (×1.073–×1.073); Apple M2: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile32x64r8x4w16x4k32: Apple M5: ×1.054 (×1.054–×1.054); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x64r8x4w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x64r8x4w16x4k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.040 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x64r8x4w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x64r8x8w8x4k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.001 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x64r8x8w8x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x64r8x8w8x4k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.987 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x64r8x8w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x64r8x8w8x4k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.915 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile32x64r8x8w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x128r12x8w16x4k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.893 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x128r12x8w16x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x128r12x8w16x4k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.853 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x128r12x8w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x128r12x8w16x4k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.825 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x128r12x8w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x32r12x4w8x4k16: Apple M5: ×1.055 (×1.055–×1.055); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x32r12x4w8x4k16s: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.942 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x32r12x4w8x4k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.037 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x32r12x4w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x32r12x4w8x4k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.020 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x32r12x4w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x64r12x4w16x4k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.046 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x64r12x4w16x4k16s: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.949 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x64r12x4w16x4k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.016 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x64r12x4w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x64r12x4w16x4k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.034 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x64r12x4w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x64r12x8w8x4k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.858 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x64r12x8w8x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x64r12x8w8x4k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.825 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x64r12x8w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x64r12x8w8x4k64: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile48x64r12x8w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x128r16x8w16x4k16: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x128r16x8w16x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x128r16x8w16x4k32: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x128r16x8w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x128r16x8w16x4k64: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x128r16x8w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x128r4x8w16x16k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.027 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x128r4x8w16x16k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.034 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x128r8x8w16x8k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.000 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x128r8x8w16x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x128r8x8w16x8k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.998 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x128r8x8w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x128r8x8w16x8k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.980 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x128r8x8w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x32r16x4w8x4k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.994 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x32r16x4w8x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x32r16x4w8x4k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.971 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x32r16x4w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x32r16x4w8x4k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.931 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x32r16x4w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x32r4x4w8x16k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.025 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x32r4x4w8x16k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.026 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x32r8x4w8x8k16: Apple M5: ×1.067 (×1.067–×1.067); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x32r8x4w8x8k16s: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.04996 < ×1.05000; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x32r8x4w8x8k32: Apple M5: ×1.062 (×1.062–×1.062); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x32r8x4w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x32r8x4w8x8k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.035 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x32r8x4w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r16x4w16x4k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.021 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r16x4w16x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r16x4w16x4k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.011 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r16x4w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r16x4w16x4k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.000 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r16x4w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r16x8w8x4k16: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r16x8w8x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r16x8w8x4k32: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r16x8w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r16x8w8x4k64: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r16x8w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r4x4w16x16k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.023 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r4x4w16x16k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.012 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r4x8w8x16k16: Apple M5: ×1.060 (×1.060–×1.060); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r4x8w8x16k32: Apple M5: ×1.062 (×1.062–×1.062); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r8x4w16x8k16: Apple M5: ×1.073 (×1.073–×1.073); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r8x4w16x8k16s: Apple M5: ×1.052 (×1.052–×1.052); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r8x4w16x8k32: Apple M5: ×1.072 (×1.072–×1.072); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r8x4w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r8x4w16x8k64: Apple M5: ×1.060 (×1.060–×1.060); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r8x4w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r8x8w8x8k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.032 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r8x8w8x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r8x8w8x8k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×1.020 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r8x8w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r8x8w8x8k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.996 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile64x64r8x8w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x128r12x8w16x8k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.802 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x128r12x8w16x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x128r12x8w16x8k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.799 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x128r12x8w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x128r12x8w16x8k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.763 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x128r12x8w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x32r12x4w8x8k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.977 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x32r12x4w8x8k16s: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.848 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x32r12x4w8x8k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.957 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x32r12x4w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x32r12x4w8x8k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.948 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x32r12x4w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x64r12x4w16x8k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.977 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x64r12x4w16x8k16s: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.864 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x64r12x4w16x8k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.952 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x64r12x4w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x64r12x4w16x8k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.949 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x64r12x4w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x64r12x8w8x8k16: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.800 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x64r12x8w8x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x64r12x8w8x8k32: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.792 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x64r12x8w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x64r12x8w8x8k64: Apple M5: i8a8-linear-m512-n3072-k3072: ×0.781 < ×1.050; Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 *   - tile96x64r12x8w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare); Apple M2: not measured in i8a8-linear-m512-n3072-k3072
 * - i8a8.linear > 512 (i8a8 linear row count > 512; Apple M5 9 cases, Apple M2 9 cases): keeps the default tile128x64r8x8w8x16k16 (no geometry matches the output in every case at ×1.050 or faster on every device)
 *   - tile128x128r16x8w16x8k16: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x128r16x8w16x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x128r16x8w16x8k32: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x128r16x8w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x128r16x8w16x8k64: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x128r16x8w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x128r8x8w16x16k16: Apple M5: i8a8-linear-m7920-n3072-k3072: ×0.951 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x128r8x8w16x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x128r8x8w16x16k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.972 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x128r8x8w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x128r8x8w16x16k64: Apple M5: i8a8-linear-m4096-n2048-k8192: ×0.963 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x128r8x8w16x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x32r16x4w8x8k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.960 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x32r16x4w8x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x32r16x4w8x8k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.951 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x32r16x4w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x32r16x4w8x8k64: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.924 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x32r16x4w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x32r8x4w8x16k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.990 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x32r8x4w8x16k16s: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.984 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x32r8x4w8x16k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.994 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x32r8x4w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x32r8x4w8x16k64: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.971 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x32r8x4w8x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r16x4w16x8k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.968 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r16x4w16x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r16x4w16x8k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.962 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r16x4w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r16x4w16x8k64: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.951 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r16x4w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r16x8w8x8k16: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r16x8w8x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r16x8w8x8k32: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r16x8w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r16x8w8x8k64: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r16x8w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r8x4w16x16k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.997 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r8x4w16x16k16s: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.960 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r8x4w16x16k32: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.964 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r8x4w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r8x4w16x16k64: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.967 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r8x4w16x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r8x8w8x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r8x8w8x16k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.973 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r8x8w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r8x8w8x16k64: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.967 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile128x64r8x8w8x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile16x128r4x8w16x4k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.932 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile16x128r4x8w16x4k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.890 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile16x32r4x4w8x4k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.918 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile16x32r4x4w8x4k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.893 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile16x64r4x4w16x4k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.902 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile16x64r4x4w16x4k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.909 < ×1.050; Apple M2: i8a8-linear-m4096-n8192-k2048: ×0.982 < ×1.050
 *   - tile16x64r4x8w8x4k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.913 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile16x64r4x8w8x4k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.876 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x128r12x8w16x16k16: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x128r12x8w16x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x128r12x8w16x16k32: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x128r12x8w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x32r12x4w8x16k16: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.915 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x32r12x4w8x16k16s: Apple M5: i8a8-linear-m1024-n2048-k2048: ×0.821 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x32r12x4w8x16k32: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.887 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x32r12x4w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x32r12x4w8x16k64: Apple M5: i8a8-linear-m1024-n8192-k2048: ×0.892 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x32r12x4w8x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x64r12x4w16x16k16: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.920 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x64r12x4w16x16k16s: Apple M5: i8a8-linear-m1024-n2048-k2048: ×0.781 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x64r12x4w16x16k32: Apple M5: i8a8-linear-m1024-n2048-k2048: ×0.896 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x64r12x4w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x64r12x4w16x16k64: Apple M5: i8a8-linear-m1024-n2048-k2048: ×0.876 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x64r12x4w16x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x64r12x8w8x16k16: Apple M5: i8a8-linear-m1024-n2048-k2048: ×0.747 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x64r12x8w8x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x64r12x8w8x16k32: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x64r12x8w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x64r12x8w8x16k64: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile192x64r12x8w8x16k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile256x128r16x8w16x16k16: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile256x128r16x8w16x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile256x128r16x8w16x16k32: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile256x128r16x8w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile256x32r16x4w8x16k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.949 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile256x32r16x4w8x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile256x32r16x4w8x16k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.932 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile256x32r16x4w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile256x64r16x4w16x16k16: Apple M5: i8a8-linear-m4096-n2048-k8192: ×0.935 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile256x64r16x4w16x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile256x64r16x4w16x16k32: Apple M5: i8a8-linear-m4096-n2048-k8192: ×0.937 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile256x64r16x4w16x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile256x64r16x8w8x16k16: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile256x64r16x8w8x16k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile256x64r16x8w8x16k32: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile256x64r16x8w8x16k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x128r4x8w16x8k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.950 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x128r4x8w16x8k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.939 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x128r8x8w16x4k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.949 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x128r8x8w16x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x128r8x8w16x4k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.933 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x128r8x8w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x128r8x8w16x4k64: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.869 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x128r8x8w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x32r4x4w8x8k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.919 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x32r4x4w8x8k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.923 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x32r8x4w8x4k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.963 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x32r8x4w8x4k16s: Apple M5: i8a8-linear-m1024-n8192-k2048: ×0.987 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x32r8x4w8x4k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.966 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x32r8x4w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x32r8x4w8x4k64: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.945 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x32r8x4w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x64r4x4w16x8k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.908 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x64r4x4w16x8k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.921 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x64r4x8w8x8k16: Apple M5: i8a8-linear-m4096-n2048-k8192: ×0.983 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x64r4x8w8x8k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.947 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x64r8x4w16x4k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.978 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x64r8x4w16x4k16s: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.966 < ×1.050; Apple M2: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile32x64r8x4w16x4k32: Apple M5: i8a8-linear-m4096-n2048-k8192: ×0.988 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x64r8x4w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x64r8x4w16x4k64: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.938 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x64r8x4w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x64r8x8w8x4k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.940 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x64r8x8w8x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x64r8x8w8x4k32: Apple M5: i8a8-linear-m1024-n2048-k2048: ×0.923 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x64r8x8w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x64r8x8w8x4k64: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.834 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile32x64r8x8w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x128r12x8w16x4k16: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.851 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x128r12x8w16x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x128r12x8w16x4k32: Apple M5: i8a8-linear-m1024-n8192-k2048: ×0.816 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x128r12x8w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x128r12x8w16x4k64: Apple M5: i8a8-linear-m1024-n2048-k2048: ×0.762 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x128r12x8w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x32r12x4w8x4k16: Apple M5: i8a8-linear-m1024-n2048-k8192: ×1.001 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x32r12x4w8x4k16s: Apple M5: i8a8-linear-m7920-n14336-k3072: ×0.920 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x32r12x4w8x4k32: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.977 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x32r12x4w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x32r12x4w8x4k64: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.957 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x32r12x4w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x64r12x4w16x4k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×1.008 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x64r12x4w16x4k16s: Apple M5: i8a8-linear-m1024-n2048-k2048: ×0.912 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x64r12x4w16x4k32: Apple M5: i8a8-linear-m1024-n8192-k2048: ×0.980 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x64r12x4w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x64r12x4w16x4k64: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.992 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x64r12x4w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x64r12x8w8x4k16: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.835 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x64r12x8w8x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x64r12x8w8x4k32: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.793 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x64r12x8w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x64r12x8w8x4k64: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile48x64r12x8w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x128r16x8w16x4k16: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x128r16x8w16x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x128r16x8w16x4k32: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x128r16x8w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x128r16x8w16x4k64: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x128r16x8w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x128r4x8w16x16k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.971 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x128r4x8w16x16k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.979 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x128r8x8w16x8k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.957 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x128r8x8w16x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x128r8x8w16x8k32: Apple M5: i8a8-linear-m4096-n2048-k8192: ×0.969 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x128r8x8w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x128r8x8w16x8k64: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.947 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x128r8x8w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x32r16x4w8x4k16: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.946 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x32r16x4w8x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x32r16x4w8x4k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.939 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x32r16x4w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x32r16x4w8x4k64: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.876 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x32r16x4w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x32r4x4w8x16k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.925 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x32r4x4w8x16k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.927 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x32r8x4w8x8k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.985 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x32r8x4w8x8k16s: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.994 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x32r8x4w8x8k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.983 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x32r8x4w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x32r8x4w8x8k64: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.955 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x32r8x4w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r16x4w16x4k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.943 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r16x4w16x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r16x4w16x4k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.946 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r16x4w16x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r16x4w16x4k64: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.935 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r16x4w16x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r16x8w8x4k16: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r16x8w8x4k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r16x8w8x4k32: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r16x8w8x4k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r16x8w8x4k64: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r16x8w8x4k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r4x4w16x16k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.932 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r4x4w16x16k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.924 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r4x8w8x16k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.986 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r4x8w8x16k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.966 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r8x4w16x8k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×1.001 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r8x4w16x8k16s: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.981 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r8x4w16x8k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×1.003 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r8x4w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r8x4w16x8k64: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.972 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r8x4w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r8x8w8x8k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.973 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r8x8w8x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r8x8w8x8k32: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.958 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r8x8w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r8x8w8x8k64: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.951 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile64x64r8x8w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x128r12x8w16x8k16: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.863 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x128r12x8w16x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x128r12x8w16x8k32: Apple M5: i8a8-linear-m1024-n2048-k2048: ×0.828 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x128r12x8w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x128r12x8w16x8k64: Apple M5: i8a8-linear-m1024-n2048-k2048: ×0.826 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x128r12x8w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x32r12x4w8x8k16: Apple M5: i8a8-linear-m7920-n3072-k14336: ×0.989 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x32r12x4w8x8k16s: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.895 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x32r12x4w8x8k32: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.970 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x32r12x4w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x32r12x4w8x8k64: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.955 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x32r12x4w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x64r12x4w16x8k16: Apple M5: i8a8-linear-m1024-n2048-k2048: ×0.998 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x64r12x4w16x8k16s: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.895 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x64r12x4w16x8k32: Apple M5: i8a8-linear-m1024-n2048-k2048: ×1.000 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x64r12x4w16x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x64r12x4w16x8k64: Apple M5: i8a8-linear-m1024-n8192-k2048: ×0.981 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x64r12x4w16x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x64r12x8w8x8k16: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.835 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x64r12x8w8x8k16s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x64r12x8w8x8k32: Apple M5: i8a8-linear-m1024-n2048-k8192: ×0.831 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x64r12x8w8x8k32s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x64r12x8w8x8k64: Apple M5: i8a8-linear-m7920-n3072-k3072: ×0.806 < ×1.050; Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 *   - tile96x64r12x8w8x8k64s: Apple M5: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare); Apple M2: not measured in i8a8-linear-m1024-n2048-k2048
 * - i8a8.attentionQk (i8a8 attention ①QK; Apple M5 4 cases, Apple M2 0 cases): keeps the default tile128x64r8x8w8x16k16 (no case of this class in the records of Apple M2 (a shared table takes a geometry only for a class every device measured))
 * - i8a8.attentionPv (i8a8 attention ③PV; Apple M5 4 cases, Apple M2 0 cases): keeps the default tile64x128r8x8w16x8k16 (no case of this class in the records of Apple M2 (a shared table takes a geometry only for a class every device measured))
 * - i8a8.conv3d (i8a8 conv3d (w8a8 implicit GEMM); Apple M5 14 cases, Apple M2 0 cases): keeps the default tile128x64r8x8w8x16k16 (no case of this class in the records of Apple M2 (a shared table takes a geometry only for a class every device measured))
 */
import type { GeometryProfile } from "../geometry-profile.ts";

export const APPLE_METAL_3_SHARED: GeometryProfile = {
  id: "apple-metal-3-shared",
  match: { vendor: "apple", architecture: "metal-3" },
  gemmRows: [
    { maxRows: 16, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 32, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 64, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16, form: "tas-tb-un" } },
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
    rows64: { regM: 8, regN: 4, wgX: 16, wgY: 8, form: "tas-tb-un", decode: "counter" },
    rows32: { regM: 4, regN: 4, wgX: 16, wgY: 8, form: "tas-tb-un", decode: "counter" },
  },
  i8a8: {
    linear: [
      { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 4, tileK: 32 } },
      { maxRows: 512, geometry: { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 } },
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
    adapters: [
      { vendor: "apple", architecture: "metal-3", device: "0x0000", description: "Apple M5" },
      { vendor: "apple", architecture: "metal-3", device: "0x0000", description: "Apple M2" },
    ],
    families: {
      gemm: {
        sweep:
          "outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json, outputs/workbench/browser-apple-m2/geometry-sweep_2026-11-23T16-05-49.json",
        sha256:
          "afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad, 549662a1b508bb221cdb9a92300b3931c423f11b7f660fa460580f3adedf6b20",
        date: "2026-11-23T15:06:51.706Z, 2026-11-23T16:05:49.220Z",
        candidateSet: "full, full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36",
        ],
        kernels: "aca9655ab3d023d9",
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
        sweep:
          "outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json, outputs/workbench/browser-apple-m2/geometry-sweep_2026-11-23T15-54-00.json",
        sha256:
          "afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad, f0509a5a73be99ae19c8d7935cbbbec7a192d717c18193b883cd301fb4568db1",
        date: "2026-11-23T15:06:51.706Z, 2026-11-23T15:54:00.903Z",
        candidateSet: "full, full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36",
        ],
        kernels: "82887096d13feac2",
        caseSet: "02186e45558e4151",
      },
      "i8a8-linear": {
        sweep:
          "outputs/workbench/browser-apple-m5/geometry-sweep_2026-11-23T15-06-54.json, outputs/workbench/browser-apple-m2/geometry-sweep_2026-11-23T15-53-18.json",
        sha256:
          "afc2d8e55efba83fe92e394f78bc9f065d5597d3ad3cc88071b8d0bdb46f36ad, b85e0950e55ff2b9eb14f788614c5f9e02663539da0bce34bb3fe45f7cd62d8e",
        date: "2026-11-23T15:06:51.706Z, 2026-11-23T15:53:17.976Z",
        candidateSet: "full, full",
        userAgent: [
          "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36",
        ],
        kernels: "83a036f74a2f08bd",
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
