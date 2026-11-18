/**
 * Geometry profile `nvidia-blackwell` (**generated — do not edit by hand**).
 *
 * A static table of tile geometries for the adapter match: vendor nvidia, architecture blackwell.
 * The runtime only selects the table by the adapter's match; nothing is measured at run time.
 * To change a value, take a sweep with the workbench's `geometry-sweep` entry and regenerate the
 * table with `node tools/geometry-profiles/generate.ts nvidia-blackwell --records=<path>[,<path>…]`.
 *
 * Derived in this generation: gemm, attention, conv2d, i8a8-linear, i8a8-attention, i8a8-conv3d.
 *
 * - outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/geometry-sweep_2026-11-18T08-51-46.json (sha256 b9c253ebc734730800b6efb5fed85c650c2926f1aa665b20f7abfb32aaf623d5, 2026-11-18T08:51:56.313Z)
 *   workbench; Chrome 157 on Windows; backend d3d12 (declared); timestamps not quantised; checkout 1a42151a20e6cf806ec8f8127a762725bec294c7
 *
 * Carried over from the registered table at commit 9483242ae575a613d6efc158f985446d7bbbaa78: none.
 * Their adoption lines are in `git show 9483242ae575a613d6efc158f985446d7bbbaa78:packages/runtime/src/kernels/geometry-profiles/nvidia-blackwell.ts`.
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
 *   - reg128x32r8x4w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x16r1x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg16x16r1x4w4:tb-un: bmm-b16-m16-n64-k64: ×0.981 < ×1.050
 *   - reg32x32r2x4w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r2x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r2x4w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r4x4w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r4x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg32x32r4x4w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r4x8w8: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r4x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r4x8w8:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r8x4w16: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r8x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 *   - reg64x64r8x4w16:tb-un: screened out by the quick look in bmm-b16-m16-n64-k128 (no timing to compare)
 * - gemmRows 17〜32 (linear / matmul / bmm row count 17〜32, 4 cases): keeps the default reg16x16r1x4w4 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg16x16r1x4w4:tas-tb-un: bmm-b16-m32-n64-k128: ×0.827 < ×1.050
 *   - reg16x16r1x4w4:tb-un: matmul-m32-n3072-k1024: ×0.984 < ×1.050
 *   - reg32x32r2x4w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x32r2x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x32r2x4w8:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x32r4x4w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x32r4x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg32x32r4x4w8:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r4x8w8: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r4x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r4x8w8:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r8x4w16: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r8x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 *   - reg64x64r8x4w16:tb-un: screened out by the quick look in bmm-b16-m32-n64-k128 (no timing to compare)
 * - gemmRows 33〜64 (linear / matmul / bmm row count 33〜64, 4 cases): keeps the default reg16x16r1x4w4 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x32r8x4w8: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x32r8x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg128x32r8x4w8:tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg16x16r1x4w4:tas-tb-un: matmul-m64-n3072-k1024: ×0.837 < ×1.050
 *   - reg16x16r1x4w4:tb-un: bmm-b16-m64-n128-k64: ×0.973 < ×1.050
 *   - reg32x32r2x4w8: bmm-b16-m64-n64-k128: ×0.855 < ×1.050
 *   - reg32x32r2x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg32x32r2x4w8:tb-un: bmm-b16-m64-n64-k128: ×0.857 < ×1.050
 *   - reg32x32r4x4w8: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg32x32r4x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg32x32r4x4w8:tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x32r4x4w8:tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg64x64r4x4w16:tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x64r4x8w8: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg64x64r4x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x64r4x8w8:tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg64x64r8x4w16: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 *   - reg64x64r8x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m64-n64-k128 (no timing to compare)
 *   - reg64x64r8x4w16:tb-un: screened out by the quick look in bmm-b16-m64-n128-k64 (no timing to compare)
 * - gemmRows 65〜128 (linear / matmul / bmm row count 65〜128, 4 cases): adopted reg64x32r4x4w8:tas-tb-un ×1.073 (×1.067–×1.081)
 *   - reg128x128r8x8w16: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg128x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg128x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg16x16r1x4w4: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x16r1x4w4:tas-tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg16x16r1x4w4:tb-un: screened out by the quick look in linear-m128-n2048-k1024 (no timing to compare)
 *   - reg32x32r2x4w8: linear-m128-n2048-k1024: ×1.020 < ×1.050
 *   - reg32x32r2x4w8:tas-tb-un: linear-m128-n2048-k1024: ×0.995 < ×1.050
 *   - reg32x32r2x4w8:tb-un: linear-m128-n2048-k1024: ×1.021 < ×1.050
 *   - reg32x32r4x4w8: linear-m128-n2048-k1024: ×0.928 < ×1.050
 *   - reg32x32r4x4w8:tas-tb-un: bmm-b16-m128-n64-k128: ×1.029 < ×1.050
 *   - reg32x32r4x4w8:tb-un: linear-m128-n2048-k1024: ×0.995 < ×1.050
 *   - reg64x32r4x4w8:tb-un: bmm-b16-m128-n64-k64: ×1.001 < ×1.050
 *   - reg64x64r4x4w16: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x64r4x4w16:tb-un: bmm-b16-m128-n64-k128: ×0.831 < ×1.050
 *   - reg64x64r4x8w8: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x64r4x8w8:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x64r4x8w8:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x64r8x4w16: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x64r8x4w16:tas-tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 *   - reg64x64r8x4w16:tb-un: screened out by the quick look in bmm-b16-m128-n64-k128 (no timing to compare)
 * - gemmRows 129〜256 (linear / matmul / bmm row count 129〜256, 4 cases): adopted reg64x64r4x4w16:tb-un ×1.091 (×1.060–×1.116)
 *   - reg128x128r8x8w16: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg128x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg128x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg128x32r8x4w8:tas-tb-un: bmm-b16-m256-n64-k128: ×0.857 < ×1.050
 *   - reg128x32r8x4w8:tb-un: matmul-m256-n2048-k1024: ×0.837 < ×1.050
 *   - reg16x16r1x4w4: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x16r1x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg16x16r1x4w4:tb-un: screened out by the quick look in bmm-b16-m256-n64-k128 (no timing to compare)
 *   - reg32x32r2x4w8: linear-m256-n2048-k1024: ×0.805 < ×1.050
 *   - reg32x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m256-n2048-k1024 (no timing to compare)
 *   - reg32x32r2x4w8:tb-un: linear-m256-n2048-k1024: ×0.805 < ×1.050
 *   - reg32x32r4x4w8: linear-m256-n2048-k1024: ×0.853 < ×1.050
 *   - reg32x32r4x4w8:tas-tb-un: linear-m256-n2048-k1024: ×0.875 < ×1.050
 *   - reg32x32r4x4w8:tb-un: linear-m256-n2048-k1024: ×0.863 < ×1.050
 *   - reg64x32r4x4w8:tas-tb-un: linear-m256-n2048-k1024: ×1.046 < ×1.050
 *   - reg64x32r4x4w8:tb-un: bmm-b16-m256-n64-k64: ×0.983 < ×1.050
 *   - reg64x64r4x4w16: linear-m256-n2048-k1024: ×1.026 < ×1.050
 *   - reg64x64r4x4w16:tas-tb-un: bmm-b16-m256-n64-k64: ×0.972 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m256-n64-k64: ×0.876 < ×1.050
 *   - reg64x64r4x8w8:tas-tb-un: matmul-m256-n2048-k1024: ×1.005 < ×1.050
 *   - reg64x64r4x8w8:tb-un: linear-m256-n2048-k1024: ×0.931 < ×1.050
 *   - reg64x64r8x4w16: linear-m256-n2048-k1024: ×0.973 < ×1.050
 *   - reg64x64r8x4w16:tas-tb-un: matmul-m256-n2048-k1024: ×0.997 < ×1.050
 *   - reg64x64r8x4w16:tb-un: linear-m256-n2048-k1024: ×0.949 < ×1.050
 * - gemmRows 257〜512 (linear / matmul / bmm row count 257〜512, 5 cases): adopted reg64x64r4x4w16:tas-tb-un ×1.090 (×1.051–×1.133)
 *   - reg128x128r8x8w16: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg128x128r8x8w16:tas-tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg128x128r8x8w16:tb-un: screened out by the quick look in bmm-b16-m512-n64-k512 (no timing to compare)
 *   - reg128x32r8x4w8: bmm-b16-m512-n64-k512: ×0.807 < ×1.050
 *   - reg128x32r8x4w8:tas-tb-un: bmm-b16-m512-n64-k64: ×0.877 < ×1.050
 *   - reg128x32r8x4w8:tb-un: bmm-b16-m512-n64-k512: ×0.833 < ×1.050
 *   - reg16x16r1x4w4: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg16x16r1x4w4:tas-tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg16x16r1x4w4:tb-un: screened out by the quick look in bmm-b16-m512-n512-k64 (no timing to compare)
 *   - reg32x32r2x4w8: linear-m512-n2048-k1024: ×0.836 < ×1.050
 *   - reg32x32r2x4w8:tas-tb-un: linear-m512-n2048-k1024: ×0.827 < ×1.050
 *   - reg32x32r2x4w8:tb-un: linear-m512-n2048-k1024: ×0.836 < ×1.050
 *   - reg32x32r4x4w8: linear-m512-n2048-k1024: ×0.824 < ×1.050
 *   - reg32x32r4x4w8:tas-tb-un: linear-m512-n2048-k1024: ×0.852 < ×1.050
 *   - reg32x32r4x4w8:tb-un: linear-m512-n2048-k1024: ×0.839 < ×1.050
 *   - reg64x32r4x4w8:tas-tb-un: bmm-b16-m512-n64-k64: ×0.956 < ×1.050
 *   - reg64x32r4x4w8:tb-un: bmm-b16-m512-n64-k64: ×0.880 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m512-n64-k512: ×1.040 < ×1.050
 *   - reg64x64r4x4w16:tb-un: geometric mean ×1.076 (below the adopted ×1.090)
 *   - reg64x64r4x8w8: bmm-b16-m512-n64-k64: ×0.821 < ×1.050
 *   - reg64x64r4x8w8:tas-tb-un: bmm-b16-m512-n64-k64: ×0.965 < ×1.050
 *   - reg64x64r4x8w8:tb-un: bmm-b16-m512-n64-k64: ×0.951 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m512-n64-k64: ×0.968 < ×1.050
 *   - reg64x64r8x4w16:tas-tb-un: bmm-b16-m512-n64-k64: ×0.970 < ×1.050
 *   - reg64x64r8x4w16:tb-un: bmm-b16-m512-n64-k64: ×0.941 < ×1.050
 * - gemmRows > 512 (linear / matmul / bmm row count > 512, 7 cases): adopted reg128x128r8x8w16:tb-un ×1.105 (×1.079–×1.176)
 *   - reg128x128r8x8w16:tas-tb-un: linear-m4096-n8192-k2048: ×1.040 < ×1.050
 *   - reg128x32r8x4w8: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg128x32r8x4w8:tas-tb-un: linear-m1024-n2048-k2048: ×0.904 < ×1.050
 *   - reg128x32r8x4w8:tb-un: matmul-m4096-n2048-k2048: ×0.864 < ×1.050
 *   - reg16x16r1x4w4: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x16r1x4w4:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg16x16r1x4w4:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r2x4w8: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r2x4w8:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r2x4w8:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r4x4w8: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r4x4w8:tas-tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg32x32r4x4w8:tb-un: screened out by the quick look in linear-m1024-n2048-k2048 (no timing to compare)
 *   - reg64x32r4x4w8: screened out by the quick look in linear-m4096-n8192-k2048 (no timing to compare)
 *   - reg64x32r4x4w8:tas-tb-un: linear-m4096-n8192-k2048: ×0.805 < ×1.050
 *   - reg64x32r4x4w8:tb-un: linear-m4096-n8192-k2048: ×0.792 < ×1.050
 *   - reg64x64r4x4w16: linear-m1024-n2048-k2048: ×0.816 < ×1.050
 *   - reg64x64r4x4w16:tas-tb-un: linear-m1024-n2048-k2048: ×0.890 < ×1.050
 *   - reg64x64r4x4w16:tb-un: linear-m1024-n2048-k2048: ×0.850 < ×1.050
 *   - reg64x64r4x8w8: linear-m1024-n2048-k2048: ×0.821 < ×1.050
 *   - reg64x64r4x8w8:tas-tb-un: linear-m1024-n2048-k2048: ×0.935 < ×1.050
 *   - reg64x64r4x8w8:tb-un: linear-m1024-n2048-k8192: ×0.881 < ×1.050
 *   - reg64x64r8x4w16: linear-m1024-n2048-k2048: ×0.887 < ×1.050
 *   - reg64x64r8x4w16:tas-tb-un: linear-m1024-n2048-k2048: ×0.931 < ×1.050
 *   - reg64x64r8x4w16:tb-un: linear-m1024-n2048-k8192: ×0.888 < ×1.050
 * - attention.qk (fused attention f32 ①QK, 4 cases): keeps the default reg128x128r8x8w16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x32r8x4w8: attention-qk-self-m4096-n4096: ×0.813 < ×1.050
 *   - reg16x16r1x4w4: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x32r2x4w8: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg32x32r4x4w8: screened out by the quick look in attention-qk-cross-m1024-n512 (no timing to compare)
 *   - reg64x32r4x4w8: screened out by the quick look in attention-qk-self-m4096-n4096 (no timing to compare)
 *   - reg64x64r4x4w16: attention-qk-self-m4096-n4096: ×0.822 < ×1.050
 *   - reg64x64r4x8w8: attention-qk-self-m4096-n4096: ×0.821 < ×1.050
 *   - reg64x64r8x4w16: attention-qk-self-m4096-n4096: ×0.895 < ×1.050
 * - attention.pv (fused attention f32 ③PV, 4 cases): keeps the default reg128x128r8x8w16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x32r8x4w8: screened out by the quick look in attention-pv-self-m1024-n1024 (no timing to compare)
 *   - reg16x16r1x4w4: screened out by the quick look in attention-pv-cross-m1024-n512 (no timing to compare)
 *   - reg32x32r2x4w8: attention-pv-self-m4096-n4096: ×0.796 < ×1.050
 *   - reg32x32r4x4w8: attention-pv-self-m1024-n1024: ×0.847 < ×1.050
 *   - reg64x32r4x4w8: attention-pv-cross-m1024-n512: ×0.896 < ×1.050
 *   - reg64x64r4x4w16: attention-pv-cross-m1024-n512: ×0.924 < ×1.050
 *   - reg64x64r4x8w8: attention-pv-self-m1024-n1024: ×0.880 < ×1.050
 *   - reg64x64r8x4w16: attention-pv-self-m1024-n1024: ×0.946 < ×1.050
 * - conv2d.rows64 (conv2d implicit GEMM m tile of 64 rows, 2 cases): adopted igemm64x64:wg16x8:tas-tb-un:dec ×1.202 (×1.172–×1.232)
 *   - igemm128x128:wg16x16: conv2d-c192-256x256: ×0.864 < ×1.050
 *   - igemm128x128:wg16x16:dec: conv2d-c192-256x256: ×0.935 < ×1.050
 *   - igemm128x128:wg16x16:tas-tb-un: conv2d-c192-256x256: ×0.889 < ×1.050
 *   - igemm128x128:wg16x16:tas-tb-un:dec: conv2d-c192-256x256: ×0.971 < ×1.050
 *   - igemm128x128:wg16x16:tb-un: conv2d-c192-256x256: ×0.910 < ×1.050
 *   - igemm128x128:wg16x16:tb-un:dec: conv2d-c192-256x256: ×0.946 < ×1.050
 *   - igemm128x64:wg16x16: conv2d-c192-256x256: ×0.864 < ×1.050
 *   - igemm128x64:wg16x16:dec: conv2d-c192-256x256: ×0.916 < ×1.050
 *   - igemm128x64:wg16x16:tas-tb-un: conv2d-c192-256x256: ×0.913 < ×1.050
 *   - igemm128x64:wg16x16:tas-tb-un:dec: conv2d-c192-256x256: ×0.921 < ×1.050
 *   - igemm128x64:wg16x16:tb-un: conv2d-c192-256x256: ×0.856 < ×1.050
 *   - igemm128x64:wg16x16:tb-un:dec: conv2d-c192-256x256: ×0.909 < ×1.050
 *   - igemm32x128:wg16x4: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm32x128:wg16x4:dec: conv2d-c192-256x256: ×0.911 < ×1.050
 *   - igemm32x128:wg16x4:tas-tb-un: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm32x128:wg16x4:tas-tb-un:dec: conv2d-c192-256x256: ×0.968 < ×1.050
 *   - igemm32x128:wg16x4:tb-un: screened out by the quick look in conv2d-c192-256x256 (no timing to compare)
 *   - igemm32x128:wg16x4:tb-un:dec: conv2d-c192-256x256: ×0.829 < ×1.050
 *   - igemm32x64:wg16x8: conv2d-c384-128x128: ×0.827 < ×1.050
 *   - igemm32x64:wg16x8:dec: conv2d-c192-256x256: ×0.981 < ×1.050
 *   - igemm32x64:wg16x8:tas-tb-un: conv2d-c384-128x128: ×0.887 < ×1.050
 *   - igemm32x64:wg16x8:tas-tb-un:dec: conv2d-c192-256x256: ×1.028 < ×1.050
 *   - igemm32x64:wg16x8:tb-un: conv2d-c384-128x128: ×0.829 < ×1.050
 *   - igemm32x64:wg16x8:tb-un:dec: conv2d-c192-256x256: ×0.983 < ×1.050
 *   - igemm64x128:wg16x8:dec: geometric mean ×1.131 (below the adopted ×1.202)
 *   - igemm64x128:wg16x8:tas-tb-un: conv2d-c384-128x128: ×0.980 < ×1.050
 *   - igemm64x128:wg16x8:tas-tb-un:dec: geometric mean ×1.177 (below the adopted ×1.202)
 *   - igemm64x128:wg16x8:tb-un: conv2d-c192-256x256: ×1.009 < ×1.050
 *   - igemm64x128:wg16x8:tb-un:dec: geometric mean ×1.144 (below the adopted ×1.202)
 *   - igemm64x64:wg16x16: conv2d-c192-256x256: ×0.963 < ×1.050
 *   - igemm64x64:wg16x16:dec: geometric mean ×1.094 (below the adopted ×1.202)
 *   - igemm64x64:wg16x16:tas-tb-un: conv2d-c192-256x256: ×1.042 < ×1.050
 *   - igemm64x64:wg16x16:tas-tb-un:dec: geometric mean ×1.174 (below the adopted ×1.202)
 *   - igemm64x64:wg16x16:tb-un: conv2d-c192-256x256: ×0.990 < ×1.050
 *   - igemm64x64:wg16x16:tb-un:dec: geometric mean ×1.126 (below the adopted ×1.202)
 *   - igemm64x64:wg16x8: conv2d-c192-256x256: ×1.033 < ×1.050
 *   - igemm64x64:wg16x8:dec: geometric mean ×1.174 (below the adopted ×1.202)
 *   - igemm64x64:wg16x8:tas-tb-un: geometric mean ×1.062 (below the adopted ×1.202)
 *   - igemm64x64:wg16x8:tb-un: conv2d-c192-256x256: ×1.003 < ×1.050
 *   - igemm64x64:wg16x8:tb-un:dec: geometric mean ×1.133 (below the adopted ×1.202)
 *   - igemm64x64:wg8x16: conv2d-c192-256x256: ×0.989 < ×1.050
 *   - igemm64x64:wg8x16:dec: geometric mean ×1.140 (below the adopted ×1.202)
 *   - igemm64x64:wg8x16:tas-tb-un: conv2d-c192-256x256: ×1.018 < ×1.050
 *   - igemm64x64:wg8x16:tas-tb-un:dec: geometric mean ×1.154 (below the adopted ×1.202)
 *   - igemm64x64:wg8x16:tb-un: conv2d-c192-256x256: ×1.015 < ×1.050
 *   - igemm64x64:wg8x16:tb-un:dec: geometric mean ×1.134 (below the adopted ×1.202)
 * - conv2d.rows32 (conv2d implicit GEMM m tile of 32 rows, 1 cases): adopted igemm32x64:wg16x8:tas-tb-un:dec ×1.256 (×1.256–×1.256)
 *   - igemm128x128:wg16x16: geometric mean ×1.060 (below the adopted ×1.256)
 *   - igemm128x128:wg16x16:dec: geometric mean ×1.126 (below the adopted ×1.256)
 *   - igemm128x128:wg16x16:tas-tb-un: geometric mean ×1.093 (below the adopted ×1.256)
 *   - igemm128x128:wg16x16:tas-tb-un:dec: geometric mean ×1.170 (below the adopted ×1.256)
 *   - igemm128x128:wg16x16:tb-un: geometric mean ×1.104 (below the adopted ×1.256)
 *   - igemm128x128:wg16x16:tb-un:dec: geometric mean ×1.137 (below the adopted ×1.256)
 *   - igemm128x64:wg16x16: geometric mean ×1.069 (below the adopted ×1.256)
 *   - igemm128x64:wg16x16:dec: geometric mean ×1.122 (below the adopted ×1.256)
 *   - igemm128x64:wg16x16:tas-tb-un: geometric mean ×1.119 (below the adopted ×1.256)
 *   - igemm128x64:wg16x16:tas-tb-un:dec: geometric mean ×1.122 (below the adopted ×1.256)
 *   - igemm128x64:wg16x16:tb-un: geometric mean ×1.056 (below the adopted ×1.256)
 *   - igemm128x64:wg16x16:tb-un:dec: geometric mean ×1.109 (below the adopted ×1.256)
 *   - igemm32x128:wg16x4:dec: geometric mean ×1.115 (below the adopted ×1.256)
 *   - igemm32x128:wg16x4:tas-tb-un: conv2d-c96-512x512: ×0.929 < ×1.050
 *   - igemm32x128:wg16x4:tas-tb-un:dec: geometric mean ×1.187 (below the adopted ×1.256)
 *   - igemm32x128:wg16x4:tb-un: conv2d-c96-512x512: ×0.927 < ×1.050
 *   - igemm32x128:wg16x4:tb-un:dec: conv2d-c96-512x512: ×0.997 < ×1.050
 *   - igemm32x64:wg16x8: geometric mean ×1.052 (below the adopted ×1.256)
 *   - igemm32x64:wg16x8:dec: geometric mean ×1.189 (below the adopted ×1.256)
 *   - igemm32x64:wg16x8:tas-tb-un: geometric mean ×1.129 (below the adopted ×1.256)
 *   - igemm32x64:wg16x8:tb-un: geometric mean ×1.054 (below the adopted ×1.256)
 *   - igemm32x64:wg16x8:tb-un:dec: geometric mean ×1.197 (below the adopted ×1.256)
 *   - igemm64x128:wg16x8: conv2d-c96-512x512: ×0.940 < ×1.050
 *   - igemm64x128:wg16x8:dec: conv2d-c96-512x512: ×1.023 < ×1.050
 *   - igemm64x128:wg16x8:tas-tb-un: conv2d-c96-512x512: ×0.970 < ×1.050
 *   - igemm64x128:wg16x8:tas-tb-un:dec: geometric mean ×1.067 (below the adopted ×1.256)
 *   - igemm64x128:wg16x8:tb-un: conv2d-c96-512x512: ×0.960 < ×1.050
 *   - igemm64x128:wg16x8:tb-un:dec: conv2d-c96-512x512: ×1.042 < ×1.050
 *   - igemm64x64:wg16x16: conv2d-c96-512x512: ×0.918 < ×1.050
 *   - igemm64x64:wg16x16:dec: conv2d-c96-512x512: ×0.998 < ×1.050
 *   - igemm64x64:wg16x16:tas-tb-un: conv2d-c96-512x512: ×0.997 < ×1.050
 *   - igemm64x64:wg16x16:tas-tb-un:dec: geometric mean ×1.068 (below the adopted ×1.256)
 *   - igemm64x64:wg16x16:tb-un: conv2d-c96-512x512: ×0.943 < ×1.050
 *   - igemm64x64:wg16x16:tb-un:dec: conv2d-c96-512x512: ×1.022 < ×1.050
 *   - igemm64x64:wg16x8: conv2d-c96-512x512: ×0.995 < ×1.050
 *   - igemm64x64:wg16x8:dec: geometric mean ×1.074 (below the adopted ×1.256)
 *   - igemm64x64:wg16x8:tas-tb-un: conv2d-c96-512x512: ×1.019 < ×1.050
 *   - igemm64x64:wg16x8:tas-tb-un:dec: geometric mean ×1.101 (below the adopted ×1.256)
 *   - igemm64x64:wg16x8:tb-un: conv2d-c96-512x512: ×0.964 < ×1.050
 *   - igemm64x64:wg16x8:tb-un:dec: conv2d-c96-512x512: ×1.029 < ×1.050
 *   - igemm64x64:wg8x16: conv2d-c96-512x512: ×0.950 < ×1.050
 *   - igemm64x64:wg8x16:dec: conv2d-c96-512x512: ×1.046 < ×1.050
 *   - igemm64x64:wg8x16:tas-tb-un: conv2d-c96-512x512: ×0.984 < ×1.050
 *   - igemm64x64:wg8x16:tas-tb-un:dec: geometric mean ×1.068 (below the adopted ×1.256)
 *   - igemm64x64:wg8x16:tb-un: conv2d-c96-512x512: ×0.980 < ×1.050
 *   - igemm64x64:wg8x16:tb-un:dec: conv2d-c96-512x512: ×1.043 < ×1.050
 * - i8a8.linear ≤ 256 (i8a8 linear row count ≤ 256, 3 cases): adopted tile32x64r4x4w16x8k32 ×2.336 (×1.899–×2.827)
 *   - tile128x64r16x4w16x8k64s: geometric mean ×1.428 (below the adopted ×2.336)
 *   - tile128x64r8x8w8x16k64s: geometric mean ×1.388 (below the adopted ×2.336)
 *   - tile16x128r4x8w16x4k16: geometric mean ×1.716 (below the adopted ×2.336)
 *   - tile32x32r4x4w8x8k16: geometric mean ×1.939 (below the adopted ×2.336)
 *   - tile32x32r8x4w8x4k64s: geometric mean ×1.967 (below the adopted ×2.336)
 *   - tile32x64r4x8w8x8k16: geometric mean ×1.870 (below the adopted ×2.336)
 *   - tile64x128r4x8w16x16k32: geometric mean ×1.412 (below the adopted ×2.336)
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m256-n1280-k1280: ×0.959 < ×1.050
 *   - tile64x32r4x4w8x16k16: geometric mean ×1.873 (below the adopted ×2.336)
 *   - tile64x64r4x4w16x16k16: geometric mean ×1.703 (below the adopted ×2.336)
 *   - tile64x64r4x8w8x16k16: geometric mean ×1.747 (below the adopted ×2.336)
 *   - tile64x64r8x4w16x8k16: geometric mean ×1.794 (below the adopted ×2.336)
 *   - tile64x64r8x4w16x8k64s: geometric mean ×2.080 (below the adopted ×2.336)
 * - i8a8.linear 257〜512 (i8a8 linear row count 257〜512, 1 cases): adopted tile128x64r16x4w16x8k64s ×1.200 (×1.200–×1.200)
 *   - tile128x64r8x8w8x16k64s: geometric mean ×1.196 (below the adopted ×1.200)
 *   - tile16x128r4x8w16x4k16: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile32x32r4x4w8x8k16: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile32x32r8x4w8x4k64s: i8a8-linear-m512-n3072-k3072: ×0.845 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-linear-m512-n3072-k3072: ×1.005 < ×1.050
 *   - tile32x64r4x8w8x8k16: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x128r4x8w16x16k32: i8a8-linear-m512-n3072-k3072: ×1.036 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m512-n3072-k3072: ×0.982 < ×1.050
 *   - tile64x32r4x4w8x16k16: screened out by the quick look in i8a8-linear-m512-n3072-k3072 (no timing to compare)
 *   - tile64x64r4x4w16x16k16: i8a8-linear-m512-n3072-k3072: ×1.033 < ×1.050
 *   - tile64x64r4x8w8x16k16: geometric mean ×1.098 (below the adopted ×1.200)
 *   - tile64x64r8x4w16x8k16: i8a8-linear-m512-n3072-k3072: ×0.973 < ×1.050
 *   - tile64x64r8x4w16x8k64s: geometric mean ×1.162 (below the adopted ×1.200)
 * - i8a8.linear > 512 (i8a8 linear row count > 512, 9 cases): adopted tile128x64r16x4w16x8k64s ×1.210 (×1.171–×1.244)
 *   - tile128x64r8x8w8x16k64s: geometric mean ×1.206 (below the adopted ×1.210)
 *   - tile16x128r4x8w16x4k16: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile32x32r4x4w8x8k16: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile32x32r8x4w8x4k64s: screened out by the quick look in i8a8-linear-m1024-n2048-k8192 (no timing to compare)
 *   - tile32x64r4x4w16x8k32: i8a8-linear-m4096-n8192-k2048: ×0.917 < ×1.050
 *   - tile32x64r4x8w8x8k16: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x128r4x8w16x16k32: i8a8-linear-m1024-n2048-k2048: ×1.017 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m1024-n8192-k2048: ×0.967 < ×1.050
 *   - tile64x32r4x4w8x16k16: screened out by the quick look in i8a8-linear-m1024-n2048-k2048 (no timing to compare)
 *   - tile64x64r4x4w16x16k16: i8a8-linear-m1024-n2048-k8192: ×0.999 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-linear-m1024-n2048-k2048: ×1.041 < ×1.050
 *   - tile64x64r8x4w16x8k16: i8a8-linear-m1024-n2048-k8192: ×0.830 < ×1.050
 *   - tile64x64r8x4w16x8k64s: geometric mean ×1.127 (below the adopted ×1.210)
 * - i8a8.attentionQk (i8a8 attention ①QK, 4 cases): adopted tile64x64r8x4w16x8k16 ×1.096 (×1.070–×1.120)
 *   - tile16x128r4x8w16x4k16: screened out by the quick look in i8a8-attention-qk-cross-m1024-n512 (no timing to compare)
 *   - tile32x32r4x4w8x8k16: screened out by the quick look in i8a8-attention-qk-cross-m1024-n512 (no timing to compare)
 *   - tile32x64r4x4w16x8k32: screened out by the quick look in i8a8-attention-qk-cross-m1024-n512 (no timing to compare)
 *   - tile32x64r4x8w8x8k16: i8a8-attention-qk-cross-m1024-n512: ×0.839 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-attention-qk-self-m4096-n4096: ×0.985 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-attention-qk-cross-m1024-n512: ×0.926 < ×1.050
 *   - tile64x32r4x4w8x16k16: screened out by the quick look in i8a8-attention-qk-cross-m1024-n512 (no timing to compare)
 *   - tile64x64r4x4w16x16k16: i8a8-attention-qk-self-m4096-n4096: ×0.980 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-attention-qk-self-m4096-n4096: ×1.021 < ×1.050
 * - i8a8.attentionPv (i8a8 attention ③PV, 4 cases): adopted tile64x128r4x8w16x16k32 ×1.159 (×1.137–×1.187)
 *   - tile128x64r8x8w8x16k16: i8a8-attention-pv-self-m4096-n4096: ×0.847 < ×1.050
 *   - tile16x128r4x8w16x4k16: screened out by the quick look in i8a8-attention-pv-cross-m1024-n512 (no timing to compare)
 *   - tile32x32r4x4w8x8k16: screened out by the quick look in i8a8-attention-pv-cross-m1024-n512 (no timing to compare)
 *   - tile32x64r4x4w16x8k32: i8a8-attention-pv-self-m4096-n4096: ×1.004 < ×1.050
 *   - tile32x64r4x8w8x8k16: screened out by the quick look in i8a8-attention-pv-cross-m1024-n512 (no timing to compare)
 *   - tile64x32r4x4w8x16k16: screened out by the quick look in i8a8-attention-pv-cross-m1024-n512 (no timing to compare)
 *   - tile64x64r4x4w16x16k16: i8a8-attention-pv-self-m1024-n1024: ×1.0496 < ×1.0500
 *   - tile64x64r4x8w8x16k16: i8a8-attention-pv-self-m1024-n1024: ×0.942 < ×1.050
 *   - tile64x64r8x4w16x8k16: i8a8-attention-pv-self-m1024-n1024: ×0.982 < ×1.050
 * - i8a8.conv3d (i8a8 conv3d (w8a8 implicit GEMM), 14 cases): adopted tile64x64r8x4w16x8k16 ×1.345 (×1.186–×2.368)
 *   - tile16x128r4x8w16x4k16: screened out by the quick look in i8a8-conv3d-cin1024-cout1024-k333-in3x32x32 (no timing to compare)
 *   - tile32x32r4x4w8x8k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.872 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.953 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-conv3d-cin512-cout256-k333-in6x256x256: ×0.980 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×1.014 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.976 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.977 < ×1.050
 *   - tile64x64r4x4w16x16k16: geometric mean ×1.216 (below the adopted ×1.345)
 *   - tile64x64r4x8w8x16k16: geometric mean ×1.261 (below the adopted ×1.345)
 */
import type { GeometryProfile } from "../geometry-profile.ts";

export const NVIDIA_BLACKWELL: GeometryProfile = {
  id: "nvidia-blackwell",
  match: { vendor: "nvidia", architecture: "blackwell" },
  gemmRows: [
    { maxRows: 16, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 32, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 64, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 128, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16, form: "tas-tb-un" } },
    { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 16, form: "tb-un" } },
    { maxRows: 512, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 16, form: "tas-tb-un" } },
    {
      maxRows: Number.POSITIVE_INFINITY,
      geometry: { regM: 8, regN: 8, wgX: 16, wgY: 16, form: "tb-un" },
    },
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
      { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 8, tileK: 32 } },
      {
        maxRows: 512,
        geometry: { regM: 16, regN: 4, wgX: 16, wgY: 8, tileK: 64, skeleton: "vec4" },
      },
      {
        maxRows: Number.POSITIVE_INFINITY,
        geometry: { regM: 16, regN: 4, wgX: 16, wgY: 8, tileK: 64, skeleton: "vec4" },
      },
    ],
    attentionQk: { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 },
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
          "outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/geometry-sweep_2026-11-18T08-51-46.json",
        sha256: "b9c253ebc734730800b6efb5fed85c650c2926f1aa665b20f7abfb32aaf623d5",
        date: "2026-11-18T08:51:56.313Z",
        candidateSet: "quick+",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36",
        ],
        kernels: "5beadfe7409496c8",
        caseSet: "e6ddd45226ab2155",
      },
      attention: {
        sweep:
          "outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/geometry-sweep_2026-11-18T08-51-46.json",
        sha256: "b9c253ebc734730800b6efb5fed85c650c2926f1aa665b20f7abfb32aaf623d5",
        date: "2026-11-18T08:51:56.313Z",
        candidateSet: "quick+",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36",
        ],
        kernels: "c3f4a946ccbf1df9",
        caseSet: "9ee211cf69674928",
      },
      conv2d: {
        sweep:
          "outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/geometry-sweep_2026-11-18T08-51-46.json",
        sha256: "b9c253ebc734730800b6efb5fed85c650c2926f1aa665b20f7abfb32aaf623d5",
        date: "2026-11-18T08:51:56.313Z",
        candidateSet: "quick+",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36",
        ],
        kernels: "82887096d13feac2",
        caseSet: "02186e45558e4151",
      },
      "i8a8-linear": {
        sweep:
          "outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/geometry-sweep_2026-11-18T08-51-46.json",
        sha256: "b9c253ebc734730800b6efb5fed85c650c2926f1aa665b20f7abfb32aaf623d5",
        date: "2026-11-18T08:51:56.313Z",
        candidateSet: "quick+",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36",
        ],
        kernels: "834e8806e13400a9",
        caseSet: "3d32dc6f10559af2",
      },
      "i8a8-attention": {
        sweep:
          "outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/geometry-sweep_2026-11-18T08-51-46.json",
        sha256: "b9c253ebc734730800b6efb5fed85c650c2926f1aa665b20f7abfb32aaf623d5",
        date: "2026-11-18T08:51:56.313Z",
        candidateSet: "quick+",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36",
        ],
        kernels: "b605a1cbd123789d",
        caseSet: "8ab1a86b13087640",
      },
      "i8a8-conv3d": {
        sweep:
          "outputs/workbench/browser-nvidia-geforce-rtx-5070-ti/geometry-sweep_2026-11-18T08-51-46.json",
        sha256: "b9c253ebc734730800b6efb5fed85c650c2926f1aa665b20f7abfb32aaf623d5",
        date: "2026-11-18T08:51:56.313Z",
        candidateSet: "quick+",
        userAgent: [
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0.0.0 Safari/537.36",
        ],
        kernels: "c91b3368411e71d9",
        caseSet: "27ce9da45b359b14",
      },
    },
  },
};
