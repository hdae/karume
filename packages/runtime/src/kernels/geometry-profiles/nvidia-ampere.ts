/**
 * Geometry profile `nvidia-ampere` (**generated — do not edit by hand**).
 *
 * A static table of tile geometries for adapter `nvidia / ampere`, derived from the
 * sweep records below with `deriveGeometryProfile` of the public surface `@karume/runtime/tune` (ADR 0117
 * decision 2 — the CLI and the GPU lab that used to write these files were removed in 0.15; the short Node
 * scripts that ran the sweep and wrote this file are kept, untracked, in
 * `outputs/diag/wave-b-2026-10-26/v5/`). The runtime only selects this table by the adapter's
 * `match` (exact match on vendor / architecture). Nothing is measured at run time (no autotuning — the
 * MUST NOT of the ADR 0022 addendum of 2026-08-10 (later), reaffirmed by ADR 0115 decision 8 and ADR 0117 decision 9).
 * To change a value, take the sweep again and regenerate the table.
 *
 * Sweeps of every family except `i8a8-linear` (whose records are described at the end of this comment)
 * (adapter nvidia / ampere / nvidia-geforce-rtx-3080-ti / NVIDIA: 615.78.08 615.78.8.0; Node + Dawn, Vulkan;
 * one process per case, the GPU cooled before each case):
 *
 * - outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run1-2026-10-26T19-31-37.851Z.json (sha256 aa978a9021e1cc0aea4f123727e36a039cec161bb55d0c715d7e3e80189f763f, 2026-10-26T19:31:37.851Z)
 * - outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run2-2026-10-26T20-20-58.449Z.json (sha256 d265e379dc2d7ca5c7036a59d8f2d55b9351465937d743a59574a1ad642c65d8, 2026-10-26T20:20:58.449Z)
 *
 * Adoption criteria (the rules of `deriveProfile` — ADR 0115 decision 4, ADR 0116, ADR 0117 decision 3): among
 * the geometries whose output matches the default on every case of the class and whose ratio to the default is
 * ×1.050 or more, the one with the largest geometric mean across the cases; if there is none, the default. When
 * several sweeps measured the same case, the ratio is the geometric mean of those observations. A case
 * whose default re-measurement ratio is outside 0.9–1.1, or whose re-measurement failed, is dropped from that
 * sweep's ratio material (output match and failures are still checked).
 * Timestamps are GPU timestamp queries in ns (Dawn, the 65,536 ns quantisation disabled), so no rounding-error
 * bound applies.
 *
 * Adoption:
 *
 * - gemmRows ≤ 16 (linear / matmul / bmm row count ≤ 16, 4 cases): keeps the default reg16x16r1x4w4 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: bmm-b16-m16-n64-k128: ×0.183 < ×1.050
 *   - reg128x32r8x4w8: matmul-m16-n3072-k1024: ×0.221 < ×1.050
 *   - reg32x32r2x4w8: linear-m16-n3072-k1024: ×0.745 < ×1.050
 *   - reg32x32r4x4w8: linear-m16-n3072-k1024: ×0.598 < ×1.050
 *   - reg64x32r4x4w8: matmul-m16-n3072-k1024: ×0.608 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m16-n64-k128: ×0.521 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m16-n64-k128: ×0.396 < ×1.050
 *   - reg64x64r8x4w16: matmul-m16-n3072-k1024: ×0.319 < ×1.050
 * - gemmRows 17〜32 (linear / matmul / bmm row count 17〜32, 4 cases): keeps the default reg16x16r1x4w4 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: bmm-b16-m32-n64-k128: ×0.197 < ×1.050
 *   - reg128x32r8x4w8: matmul-m32-n3072-k1024: ×0.246 < ×1.050
 *   - reg32x32r2x4w8: bmm-b16-m32-n64-k128: ×0.826 < ×1.050
 *   - reg32x32r4x4w8: linear-m32-n3072-k1024: ×0.742 < ×1.050
 *   - reg64x32r4x4w8: bmm-b16-m32-n64-k128: ×0.677 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m32-n64-k128: ×0.551 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m32-n64-k128: ×0.420 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m32-n64-k128: ×0.345 < ×1.050
 * - gemmRows 33〜64 (linear / matmul / bmm row count 33〜64, 4 cases): keeps the default reg16x16r1x4w4 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: bmm-b16-m64-n64-k128: ×0.247 < ×1.050
 *   - reg128x32r8x4w8: bmm-b16-m64-n64-k128: ×0.318 < ×1.050
 *   - reg32x32r2x4w8: bmm-b16-m64-n64-k128: ×1.034 < ×1.050
 *   - reg32x32r4x4w8: bmm-b16-m64-n64-k128: ×0.961 < ×1.050
 *   - reg64x32r4x4w8: bmm-b16-m64-n64-k128: ×0.832 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m64-n64-k128: ×0.679 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m64-n64-k128: ×0.512 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m64-n64-k128: ×0.441 < ×1.050
 * - gemmRows 65〜128 (linear / matmul / bmm row count 65〜128, 4 cases): keeps the default reg64x32r4x4w8 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: bmm-b16-m128-n64-k128: ×0.292 < ×1.050
 *   - reg128x32r8x4w8: bmm-b16-m128-n64-k128: ×0.387 < ×1.050
 *   - reg16x16r1x4w4: linear-m128-n2048-k1024: ×0.505 < ×1.050
 *   - reg32x32r2x4w8: linear-m128-n2048-k1024: ×0.751 < ×1.050
 *   - reg32x32r4x4w8: linear-m128-n2048-k1024: ×0.855 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m128-n64-k128: ×0.831 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m128-n64-k128: ×0.629 < ×1.050
 *   - reg64x64r8x4w16: matmul-m128-n2048-k1024: ×0.467 < ×1.050
 * - gemmRows 129〜256 (linear / matmul / bmm row count 129〜256, 4 cases): adopted reg64x64r4x4w16 ×1.114 (×1.105–×1.125)
 *   - reg128x128r8x8w16: bmm-b16-m256-n64-k128: ×0.414 < ×1.050
 *   - reg128x32r8x4w8: matmul-m256-n2048-k1024: ×0.539 < ×1.050
 *   - reg16x16r1x4w4: linear-m256-n2048-k1024: ×0.440 < ×1.050
 *   - reg32x32r2x4w8: linear-m256-n2048-k1024: ×0.729 < ×1.050
 *   - reg32x32r4x4w8: linear-m256-n2048-k1024: ×0.812 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m256-n64-k64: ×0.816 < ×1.050
 *   - reg64x64r8x4w16: matmul-m256-n2048-k1024: ×0.686 < ×1.050
 * - gemmRows 257〜512 (linear / matmul / bmm row count 257〜512, 5 cases): keeps the default reg64x32r4x4w8 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: bmm-b16-m512-n64-k64: ×0.529 < ×1.050
 *   - reg128x32r8x4w8: bmm-b16-m512-n64-k512: ×0.529 < ×1.050
 *   - reg16x16r1x4w4: linear-m512-n2048-k1024: ×0.401 < ×1.050
 *   - reg32x32r2x4w8: linear-m512-n2048-k1024: ×0.707 < ×1.050
 *   - reg32x32r4x4w8: linear-m512-n2048-k1024: ×0.807 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m512-n64-k64: ×1.025 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m512-n64-k64: ×0.688 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m512-n64-k512: ×0.782 < ×1.050
 * - gemmRows > 512 (linear / matmul / bmm row count > 512, 7 cases): keeps the default reg128x128r8x8w16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x32r8x4w8: matmul-m4096-n2048-k2048: ×0.532 < ×1.050
 *   - reg16x16r1x4w4: linear-m4096-n8192-k2048: ×0.269 < ×1.050
 *   - reg32x32r2x4w8: linear-m4096-n8192-k2048: ×0.481 < ×1.050
 *   - reg32x32r4x4w8: linear-m4096-n8192-k2048: ×0.567 < ×1.050
 *   - reg64x32r4x4w8: linear-m4096-n8192-k2048: ×0.749 < ×1.050
 *   - reg64x64r4x4w16: linear-m4096-n8192-k2048: ×0.863 < ×1.050
 *   - reg64x64r4x8w8: linear-m4096-n8192-k2048: ×0.723 < ×1.050
 *   - reg64x64r8x4w16: linear-m4096-n8192-k2048: ×0.719 < ×1.050
 * - attention.qk (fused attention f32 ①QK, 4 cases): keeps the default reg128x128r8x8w16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x32r8x4w8: attention-qk-self-m4096-n4096: ×0.570 < ×1.050
 *   - reg16x16r1x4w4: attention-qk-self-m4096-n4096: ×0.296 < ×1.050
 *   - reg32x32r2x4w8: attention-qk-self-m4096-n4096: ×0.520 < ×1.050
 *   - reg32x32r4x4w8: attention-qk-self-m4096-n4096: ×0.630 < ×1.050
 *   - reg64x32r4x4w8: attention-qk-self-m4096-n4096: ×0.796 < ×1.050
 *   - reg64x64r4x4w16: attention-qk-self-m4096-n4096: ×0.938 < ×1.050
 *   - reg64x64r4x8w8: attention-qk-self-m4096-n4096: ×0.758 < ×1.050
 *   - reg64x64r8x4w16: attention-qk-self-m4096-n4096: ×0.774 < ×1.050
 * - attention.pv (fused attention f32 ③PV, 4 cases): adopted reg64x64r4x4w16 ×1.113 (×1.062–×1.162)
 *   - reg128x32r8x4w8: attention-pv-self-m4096-n4096: ×0.547 < ×1.050
 *   - reg16x16r1x4w4: attention-pv-self-m4096-n4096: ×0.378 < ×1.050
 *   - reg32x32r2x4w8: attention-pv-self-m4096-n4096: ×0.645 < ×1.050
 *   - reg32x32r4x4w8: attention-pv-self-m4096-n4096: ×0.845 < ×1.050
 *   - reg64x32r4x4w8: attention-pv-self-m4096-n4096: ×0.870 < ×1.050
 *   - reg64x64r4x8w8: attention-pv-self-m4096-n4096: ×0.865 < ×1.050
 *   - reg64x64r8x4w16: attention-pv-self-m4096-n4096: ×0.859 < ×1.050
 * - conv2d.rows64 (conv2d implicit GEMM m tile of 64 rows, 2 cases): keeps the default igemm64x128:wg16x8 (no geometry matches the output in every case at ×1.050 or faster)
 *   - igemm128x128:wg16x16: conv2d-c192-256x256: ×0.855 < ×1.050
 *   - igemm128x64:wg16x16: conv2d-c192-256x256: ×0.730 < ×1.050
 *   - igemm32x128:wg16x4: conv2d-c384-128x128: ×0.765 < ×1.050
 *   - igemm64x64:wg16x16: conv2d-c192-256x256: ×1.019 < ×1.050
 *   - igemm64x64:wg16x8: conv2d-c384-128x128: ×0.875 < ×1.050
 *   - igemm64x64:wg8x16: conv2d-c384-128x128: ×0.886 < ×1.050
 * - conv2d.rows32 (conv2d implicit GEMM m tile of 32 rows, 1 cases): keeps the default igemm32x128:wg16x4 (no geometry matches the output in every case at ×1.050 or faster)
 *   - igemm128x128:wg16x16: conv2d-c96-512x512: ×1.035 < ×1.050
 *   - igemm128x64:wg16x16: conv2d-c96-512x512: ×0.877 < ×1.050
 *   - igemm64x128:wg16x8: conv2d-c96-512x512: ×0.936 < ×1.050
 *   - igemm64x64:wg16x16: conv2d-c96-512x512: ×1.010 < ×1.050
 *   - igemm64x64:wg16x8: conv2d-c96-512x512: ×0.828 < ×1.050
 *   - igemm64x64:wg8x16: conv2d-c96-512x512: ×0.823 < ×1.050
 * - i8a8.linear ≤ 256 (i8a8 linear row count ≤ 256, 3 cases): adopted tile32x32r8x4w8x4k64s ×3.377 (×3.208–×3.537)
 *   - tile128x128r16x8w16x8k16: i8a8-linear-m185-n1280-k1280: ×0.761 < ×1.050
 *   - tile128x128r16x8w16x8k16s: i8a8-linear-m185-n1280-k1280: ×0.674 < ×1.050
 *   - tile128x128r16x8w16x8k32: i8a8-linear-m256-n1280-k1280: ×0.838 < ×1.050
 *   - tile128x128r16x8w16x8k32s: i8a8-linear-m185-n1280-k1280: ×0.746 < ×1.050
 *   - tile128x128r16x8w16x8k64: i8a8-linear-m185-n1280-k1280: ×0.781 < ×1.050
 *   - tile128x128r16x8w16x8k64s: i8a8-linear-m185-n1280-k1280: ×0.776 < ×1.050
 *   - tile128x128r8x8w16x16k16: i8a8-linear-m256-n1280-k1280: ×0.810 < ×1.050
 *   - tile128x128r8x8w16x16k16s: i8a8-linear-m256-n1280-k1280: ×0.825 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-linear-m185-n1280-k1280: ×0.836 < ×1.050
 *   - tile128x128r8x8w16x16k32s: geometric mean ×1.178 (below the adopted ×3.377)
 *   - tile128x128r8x8w16x16k64: i8a8-linear-m256-n1280-k1280: ×0.780 < ×1.050
 *   - tile128x128r8x8w16x16k64s: geometric mean ×1.273 (below the adopted ×3.377)
 *   - tile128x32r16x4w8x8k16: i8a8-linear-m1-n18432-k3072: ×0.895 < ×1.050
 *   - tile128x32r16x4w8x8k16s: geometric mean ×1.312 (below the adopted ×3.377)
 *   - tile128x32r16x4w8x8k32: i8a8-linear-m1-n18432-k3072: ×0.914 < ×1.050
 *   - tile128x32r16x4w8x8k32s: geometric mean ×1.743 (below the adopted ×3.377)
 *   - tile128x32r16x4w8x8k64: i8a8-linear-m1-n18432-k3072: ×0.814 < ×1.050
 *   - tile128x32r16x4w8x8k64s: geometric mean ×1.855 (below the adopted ×3.377)
 *   - tile128x32r8x4w8x16k16: geometric mean ×1.624 (below the adopted ×3.377)
 *   - tile128x32r8x4w8x16k16s: geometric mean ×1.400 (below the adopted ×3.377)
 *   - tile128x32r8x4w8x16k32: i8a8-linear-m1-n18432-k3072: ×0.930 < ×1.050
 *   - tile128x32r8x4w8x16k32s: geometric mean ×1.708 (below the adopted ×3.377)
 *   - tile128x32r8x4w8x16k64: i8a8-linear-m1-n18432-k3072: ×0.728 < ×1.050
 *   - tile128x32r8x4w8x16k64s: geometric mean ×2.118 (below the adopted ×3.377)
 *   - tile128x64r16x4w16x8k16: i8a8-linear-m1-n18432-k3072: ×1.044 < ×1.050
 *   - tile128x64r16x4w16x8k16s: geometric mean ×1.285 (below the adopted ×3.377)
 *   - tile128x64r16x4w16x8k32: i8a8-linear-m1-n18432-k3072: ×0.955 < ×1.050
 *   - tile128x64r16x4w16x8k32s: geometric mean ×1.697 (below the adopted ×3.377)
 *   - tile128x64r16x4w16x8k64: i8a8-linear-m1-n18432-k3072: ×0.893 < ×1.050
 *   - tile128x64r16x4w16x8k64s: geometric mean ×1.893 (below the adopted ×3.377)
 *   - tile128x64r16x8w8x8k16: i8a8-linear-m256-n1280-k1280: ×0.809 < ×1.050
 *   - tile128x64r16x8w8x8k16s: i8a8-linear-m256-n1280-k1280: ×0.727 < ×1.050
 *   - tile128x64r16x8w8x8k32: i8a8-linear-m185-n1280-k1280: ×0.883 < ×1.050
 *   - tile128x64r16x8w8x8k32s: i8a8-linear-m256-n1280-k1280: ×0.764 < ×1.050
 *   - tile128x64r16x8w8x8k64: i8a8-linear-m256-n1280-k1280: ×0.785 < ×1.050
 *   - tile128x64r16x8w8x8k64s: i8a8-linear-m185-n1280-k1280: ×0.780 < ×1.050
 *   - tile128x64r8x4w16x16k16: geometric mean ×1.329 (below the adopted ×3.377)
 *   - tile128x64r8x4w16x16k16s: geometric mean ×1.190 (below the adopted ×3.377)
 *   - tile128x64r8x4w16x16k32: i8a8-linear-m1-n18432-k3072: ×0.986 < ×1.050
 *   - tile128x64r8x4w16x16k32s: geometric mean ×1.566 (below the adopted ×3.377)
 *   - tile128x64r8x4w16x16k64: i8a8-linear-m1-n18432-k3072: ×0.827 < ×1.050
 *   - tile128x64r8x4w16x16k64s: geometric mean ×1.832 (below the adopted ×3.377)
 *   - tile128x64r8x8w8x16k16s: geometric mean ×1.222 (below the adopted ×3.377)
 *   - tile128x64r8x8w8x16k32: i8a8-linear-m1-n18432-k3072: ×0.926 < ×1.050
 *   - tile128x64r8x8w8x16k32s: geometric mean ×1.640 (below the adopted ×3.377)
 *   - tile128x64r8x8w8x16k64: i8a8-linear-m1-n18432-k3072: ×0.846 < ×1.050
 *   - tile128x64r8x8w8x16k64s: geometric mean ×1.823 (below the adopted ×3.377)
 *   - tile16x128r4x8w16x4k16: geometric mean ×1.731 (below the adopted ×3.377)
 *   - tile16x128r4x8w16x4k32: geometric mean ×1.487 (below the adopted ×3.377)
 *   - tile16x32r4x4w8x4k16: geometric mean ×2.203 (below the adopted ×3.377)
 *   - tile16x32r4x4w8x4k32: geometric mean ×2.274 (below the adopted ×3.377)
 *   - tile16x64r4x4w16x4k16: geometric mean ×2.163 (below the adopted ×3.377)
 *   - tile16x64r4x4w16x4k32: geometric mean ×2.274 (below the adopted ×3.377)
 *   - tile16x64r4x8w8x4k16: geometric mean ×2.076 (below the adopted ×3.377)
 *   - tile16x64r4x8w8x4k32: geometric mean ×1.739 (below the adopted ×3.377)
 *   - tile192x128r12x8w16x16k16: i8a8-linear-m256-n1280-k1280: ×0.588 < ×1.050
 *   - tile192x128r12x8w16x16k16s: i8a8-linear-m256-n1280-k1280: ×0.649 < ×1.050
 *   - tile192x128r12x8w16x16k32: i8a8-linear-m256-n1280-k1280: ×0.543 < ×1.050
 *   - tile192x128r12x8w16x16k32s: i8a8-linear-m185-n1280-k1280: ×0.754 < ×1.050
 *   - tile192x32r12x4w8x16k16: i8a8-linear-m1-n18432-k3072: ×0.669 < ×1.050
 *   - tile192x32r12x4w8x16k16s: i8a8-linear-m1-n18432-k3072: ×0.909 < ×1.050
 *   - tile192x32r12x4w8x16k32: i8a8-linear-m1-n18432-k3072: ×0.627 < ×1.050
 *   - tile192x32r12x4w8x16k32s: i8a8-linear-m1-n18432-k3072: ×0.950 < ×1.050
 *   - tile192x32r12x4w8x16k64: i8a8-linear-m1-n18432-k3072: ×0.537 < ×1.050
 *   - tile192x32r12x4w8x16k64s: i8a8-linear-m1-n18432-k3072: ×0.954 < ×1.050
 *   - tile192x64r12x4w16x16k16: i8a8-linear-m1-n18432-k3072: ×0.697 < ×1.050
 *   - tile192x64r12x4w16x16k16s: i8a8-linear-m1-n18432-k3072: ×0.829 < ×1.050
 *   - tile192x64r12x4w16x16k32: i8a8-linear-m1-n18432-k3072: ×0.700 < ×1.050
 *   - tile192x64r12x4w16x16k32s: i8a8-linear-m1-n18432-k3072: ×0.850 < ×1.050
 *   - tile192x64r12x4w16x16k64: i8a8-linear-m1-n18432-k3072: ×0.599 < ×1.050
 *   - tile192x64r12x4w16x16k64s: i8a8-linear-m1-n18432-k3072: ×0.948 < ×1.050
 *   - tile192x64r12x8w8x16k16: i8a8-linear-m1-n18432-k3072: ×0.752 < ×1.050
 *   - tile192x64r12x8w8x16k16s: i8a8-linear-m1-n18432-k3072: ×0.889 < ×1.050
 *   - tile192x64r12x8w8x16k32: i8a8-linear-m1-n18432-k3072: ×0.679 < ×1.050
 *   - tile192x64r12x8w8x16k32s: i8a8-linear-m1-n18432-k3072: ×0.949 < ×1.050
 *   - tile192x64r12x8w8x16k64: i8a8-linear-m1-n18432-k3072: ×0.580 < ×1.050
 *   - tile192x64r12x8w8x16k64s: i8a8-linear-m1-n18432-k3072: ×0.990 < ×1.050
 *   - tile256x128r16x8w16x16k16: i8a8-linear-m256-n1280-k1280: ×0.463 < ×1.050
 *   - tile256x128r16x8w16x16k16s: i8a8-linear-m256-n1280-k1280: ×0.459 < ×1.050
 *   - tile256x128r16x8w16x16k32: i8a8-linear-m256-n1280-k1280: ×0.531 < ×1.050
 *   - tile256x128r16x8w16x16k32s: i8a8-linear-m256-n1280-k1280: ×0.552 < ×1.050
 *   - tile256x32r16x4w8x16k16: i8a8-linear-m1-n18432-k3072: ×0.482 < ×1.050
 *   - tile256x32r16x4w8x16k16s: i8a8-linear-m1-n18432-k3072: ×0.713 < ×1.050
 *   - tile256x32r16x4w8x16k32: i8a8-linear-m1-n18432-k3072: ×0.485 < ×1.050
 *   - tile256x32r16x4w8x16k32s: i8a8-linear-m1-n18432-k3072: ×0.759 < ×1.050
 *   - tile256x64r16x4w16x16k16: i8a8-linear-m1-n18432-k3072: ×0.561 < ×1.050
 *   - tile256x64r16x4w16x16k16s: i8a8-linear-m1-n18432-k3072: ×0.660 < ×1.050
 *   - tile256x64r16x4w16x16k32: i8a8-linear-m1-n18432-k3072: ×0.548 < ×1.050
 *   - tile256x64r16x4w16x16k32s: i8a8-linear-m1-n18432-k3072: ×0.712 < ×1.050
 *   - tile256x64r16x8w8x16k16: i8a8-linear-m1-n18432-k3072: ×0.580 < ×1.050
 *   - tile256x64r16x8w8x16k16s: i8a8-linear-m1-n18432-k3072: ×0.587 < ×1.050
 *   - tile256x64r16x8w8x16k32: i8a8-linear-m1-n18432-k3072: ×0.544 < ×1.050
 *   - tile256x64r16x8w8x16k32s: i8a8-linear-m1-n18432-k3072: ×0.638 < ×1.050
 *   - tile32x128r4x8w16x8k16: geometric mean ×1.819 (below the adopted ×3.377)
 *   - tile32x128r4x8w16x8k32: geometric mean ×1.758 (below the adopted ×3.377)
 *   - tile32x128r8x8w16x4k16: geometric mean ×1.414 (below the adopted ×3.377)
 *   - tile32x128r8x8w16x4k16s: geometric mean ×1.640 (below the adopted ×3.377)
 *   - tile32x128r8x8w16x4k32: geometric mean ×1.462 (below the adopted ×3.377)
 *   - tile32x128r8x8w16x4k32s: geometric mean ×1.951 (below the adopted ×3.377)
 *   - tile32x128r8x8w16x4k64: geometric mean ×1.444 (below the adopted ×3.377)
 *   - tile32x128r8x8w16x4k64s: geometric mean ×2.476 (below the adopted ×3.377)
 *   - tile32x32r4x4w8x8k16: geometric mean ×2.173 (below the adopted ×3.377)
 *   - tile32x32r4x4w8x8k32: geometric mean ×2.295 (below the adopted ×3.377)
 *   - tile32x32r8x4w8x4k16: geometric mean ×2.132 (below the adopted ×3.377)
 *   - tile32x32r8x4w8x4k16s: geometric mean ×2.162 (below the adopted ×3.377)
 *   - tile32x32r8x4w8x4k32: geometric mean ×1.853 (below the adopted ×3.377)
 *   - tile32x32r8x4w8x4k32s: geometric mean ×2.509 (below the adopted ×3.377)
 *   - tile32x32r8x4w8x4k64: geometric mean ×1.649 (below the adopted ×3.377)
 *   - tile32x64r4x4w16x8k16: geometric mean ×2.153 (below the adopted ×3.377)
 *   - tile32x64r4x4w16x8k32: geometric mean ×2.310 (below the adopted ×3.377)
 *   - tile32x64r4x8w8x8k16: geometric mean ×1.914 (below the adopted ×3.377)
 *   - tile32x64r4x8w8x8k32: geometric mean ×1.748 (below the adopted ×3.377)
 *   - tile32x64r8x4w16x4k16: geometric mean ×2.085 (below the adopted ×3.377)
 *   - tile32x64r8x4w16x4k16s: geometric mean ×1.854 (below the adopted ×3.377)
 *   - tile32x64r8x4w16x4k32: geometric mean ×1.882 (below the adopted ×3.377)
 *   - tile32x64r8x4w16x4k32s: geometric mean ×2.404 (below the adopted ×3.377)
 *   - tile32x64r8x4w16x4k64: geometric mean ×1.752 (below the adopted ×3.377)
 *   - tile32x64r8x4w16x4k64s: geometric mean ×3.151 (below the adopted ×3.377)
 *   - tile32x64r8x8w8x4k16: geometric mean ×1.479 (below the adopted ×3.377)
 *   - tile32x64r8x8w8x4k16s: geometric mean ×1.767 (below the adopted ×3.377)
 *   - tile32x64r8x8w8x4k32: geometric mean ×1.409 (below the adopted ×3.377)
 *   - tile32x64r8x8w8x4k32s: geometric mean ×1.998 (below the adopted ×3.377)
 *   - tile32x64r8x8w8x4k64: geometric mean ×1.344 (below the adopted ×3.377)
 *   - tile32x64r8x8w8x4k64s: geometric mean ×2.525 (below the adopted ×3.377)
 *   - tile48x128r12x8w16x4k16: i8a8-linear-m256-n1280-k1280: ×1.011 < ×1.050
 *   - tile48x128r12x8w16x4k16s: geometric mean ×1.406 (below the adopted ×3.377)
 *   - tile48x128r12x8w16x4k32: i8a8-linear-m256-n1280-k1280: ×0.996 < ×1.050
 *   - tile48x128r12x8w16x4k32s: geometric mean ×1.602 (below the adopted ×3.377)
 *   - tile48x128r12x8w16x4k64: i8a8-linear-m256-n1280-k1280: ×1.009 < ×1.050
 *   - tile48x128r12x8w16x4k64s: geometric mean ×1.695 (below the adopted ×3.377)
 *   - tile48x32r12x4w8x4k16: geometric mean ×1.570 (below the adopted ×3.377)
 *   - tile48x32r12x4w8x4k16s: geometric mean ×1.909 (below the adopted ×3.377)
 *   - tile48x32r12x4w8x4k32: geometric mean ×1.636 (below the adopted ×3.377)
 *   - tile48x32r12x4w8x4k32s: geometric mean ×2.191 (below the adopted ×3.377)
 *   - tile48x32r12x4w8x4k64: geometric mean ×1.613 (below the adopted ×3.377)
 *   - tile48x32r12x4w8x4k64s: geometric mean ×2.768 (below the adopted ×3.377)
 *   - tile48x64r12x4w16x4k16: geometric mean ×1.442 (below the adopted ×3.377)
 *   - tile48x64r12x4w16x4k16s: geometric mean ×1.649 (below the adopted ×3.377)
 *   - tile48x64r12x4w16x4k32: geometric mean ×1.637 (below the adopted ×3.377)
 *   - tile48x64r12x4w16x4k32s: geometric mean ×2.224 (below the adopted ×3.377)
 *   - tile48x64r12x4w16x4k64: geometric mean ×1.596 (below the adopted ×3.377)
 *   - tile48x64r12x4w16x4k64s: geometric mean ×2.711 (below the adopted ×3.377)
 *   - tile48x64r12x8w8x4k16: i8a8-linear-m256-n1280-k1280: ×1.040 < ×1.050
 *   - tile48x64r12x8w8x4k16s: geometric mean ×1.492 (below the adopted ×3.377)
 *   - tile48x64r12x8w8x4k32: i8a8-linear-m256-n1280-k1280: ×0.932 < ×1.050
 *   - tile48x64r12x8w8x4k32s: geometric mean ×1.588 (below the adopted ×3.377)
 *   - tile48x64r12x8w8x4k64: i8a8-linear-m256-n1280-k1280: ×0.929 < ×1.050
 *   - tile48x64r12x8w8x4k64s: geometric mean ×1.484 (below the adopted ×3.377)
 *   - tile64x128r16x8w16x4k16: i8a8-linear-m256-n1280-k1280: ×0.807 < ×1.050
 *   - tile64x128r16x8w16x4k16s: i8a8-linear-m256-n1280-k1280: ×0.722 < ×1.050
 *   - tile64x128r16x8w16x4k32: i8a8-linear-m256-n1280-k1280: ×0.897 < ×1.050
 *   - tile64x128r16x8w16x4k32s: geometric mean ×1.416 (below the adopted ×3.377)
 *   - tile64x128r16x8w16x4k64: i8a8-linear-m185-n1280-k1280: ×0.799 < ×1.050
 *   - tile64x128r16x8w16x4k64s: i8a8-linear-m256-n1280-k1280: ×0.785 < ×1.050
 *   - tile64x128r4x8w16x16k16: geometric mean ×1.424 (below the adopted ×3.377)
 *   - tile64x128r4x8w16x16k32: geometric mean ×1.328 (below the adopted ×3.377)
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m256-n1280-k1280: ×1.033 < ×1.050
 *   - tile64x128r8x8w16x8k16s: geometric mean ×1.403 (below the adopted ×3.377)
 *   - tile64x128r8x8w16x8k32: geometric mean ×1.373 (below the adopted ×3.377)
 *   - tile64x128r8x8w16x8k32s: geometric mean ×1.846 (below the adopted ×3.377)
 *   - tile64x128r8x8w16x8k64: geometric mean ×1.291 (below the adopted ×3.377)
 *   - tile64x128r8x8w16x8k64s: geometric mean ×2.164 (below the adopted ×3.377)
 *   - tile64x32r16x4w8x4k16: geometric mean ×1.367 (below the adopted ×3.377)
 *   - tile64x32r16x4w8x4k16s: geometric mean ×1.722 (below the adopted ×3.377)
 *   - tile64x32r16x4w8x4k32: geometric mean ×1.297 (below the adopted ×3.377)
 *   - tile64x32r16x4w8x4k32s: geometric mean ×1.963 (below the adopted ×3.377)
 *   - tile64x32r16x4w8x4k64: geometric mean ×1.174 (below the adopted ×3.377)
 *   - tile64x32r16x4w8x4k64s: geometric mean ×2.204 (below the adopted ×3.377)
 *   - tile64x32r4x4w8x16k16: geometric mean ×2.034 (below the adopted ×3.377)
 *   - tile64x32r4x4w8x16k32: geometric mean ×2.192 (below the adopted ×3.377)
 *   - tile64x32r8x4w8x8k16: geometric mean ×1.987 (below the adopted ×3.377)
 *   - tile64x32r8x4w8x8k16s: geometric mean ×1.753 (below the adopted ×3.377)
 *   - tile64x32r8x4w8x8k32: geometric mean ×1.750 (below the adopted ×3.377)
 *   - tile64x32r8x4w8x8k32s: geometric mean ×2.249 (below the adopted ×3.377)
 *   - tile64x32r8x4w8x8k64: geometric mean ×1.543 (below the adopted ×3.377)
 *   - tile64x32r8x4w8x8k64s: geometric mean ×2.596 (below the adopted ×3.377)
 *   - tile64x64r16x4w16x4k16: geometric mean ×1.262 (below the adopted ×3.377)
 *   - tile64x64r16x4w16x4k16s: geometric mean ×1.739 (below the adopted ×3.377)
 *   - tile64x64r16x4w16x4k32: geometric mean ×1.368 (below the adopted ×3.377)
 *   - tile64x64r16x4w16x4k32s: geometric mean ×2.022 (below the adopted ×3.377)
 *   - tile64x64r16x4w16x4k64: geometric mean ×1.363 (below the adopted ×3.377)
 *   - tile64x64r16x4w16x4k64s: geometric mean ×2.323 (below the adopted ×3.377)
 *   - tile64x64r16x8w8x4k16: i8a8-linear-m256-n1280-k1280: ×0.871 < ×1.050
 *   - tile64x64r16x8w8x4k16s: i8a8-linear-m256-n1280-k1280: ×0.752 < ×1.050
 *   - tile64x64r16x8w8x4k32: i8a8-linear-m256-n1280-k1280: ×0.843 < ×1.050
 *   - tile64x64r16x8w8x4k32s: i8a8-linear-m185-n1280-k1280: ×0.770 < ×1.050
 *   - tile64x64r16x8w8x4k64: i8a8-linear-m185-n1280-k1280: ×0.463 < ×1.050
 *   - tile64x64r16x8w8x4k64s: i8a8-linear-m256-n1280-k1280: ×0.785 < ×1.050
 *   - tile64x64r4x4w16x16k16: geometric mean ×2.042 (below the adopted ×3.377)
 *   - tile64x64r4x4w16x16k32: geometric mean ×2.305 (below the adopted ×3.377)
 *   - tile64x64r4x8w8x16k16: geometric mean ×1.887 (below the adopted ×3.377)
 *   - tile64x64r4x8w8x16k32: geometric mean ×1.691 (below the adopted ×3.377)
 *   - tile64x64r8x4w16x8k16: geometric mean ×2.007 (below the adopted ×3.377)
 *   - tile64x64r8x4w16x8k16s: geometric mean ×1.719 (below the adopted ×3.377)
 *   - tile64x64r8x4w16x8k32: geometric mean ×1.792 (below the adopted ×3.377)
 *   - tile64x64r8x4w16x8k32s: geometric mean ×2.309 (below the adopted ×3.377)
 *   - tile64x64r8x4w16x8k64: geometric mean ×1.696 (below the adopted ×3.377)
 *   - tile64x64r8x4w16x8k64s: geometric mean ×2.932 (below the adopted ×3.377)
 *   - tile64x64r8x8w8x8k16: geometric mean ×1.372 (below the adopted ×3.377)
 *   - tile64x64r8x8w8x8k16s: geometric mean ×1.717 (below the adopted ×3.377)
 *   - tile64x64r8x8w8x8k32: geometric mean ×1.315 (below the adopted ×3.377)
 *   - tile64x64r8x8w8x8k32s: geometric mean ×1.980 (below the adopted ×3.377)
 *   - tile64x64r8x8w8x8k64: geometric mean ×1.325 (below the adopted ×3.377)
 *   - tile64x64r8x8w8x8k64s: geometric mean ×2.247 (below the adopted ×3.377)
 *   - tile96x128r12x8w16x8k16: i8a8-linear-m185-n1280-k1280: ×0.798 < ×1.050
 *   - tile96x128r12x8w16x8k16s: i8a8-linear-m256-n1280-k1280: ×1.012 < ×1.050
 *   - tile96x128r12x8w16x8k32: i8a8-linear-m185-n1280-k1280: ×0.850 < ×1.050
 *   - tile96x128r12x8w16x8k32s: geometric mean ×1.443 (below the adopted ×3.377)
 *   - tile96x128r12x8w16x8k64: i8a8-linear-m185-n1280-k1280: ×0.902 < ×1.050
 *   - tile96x128r12x8w16x8k64s: geometric mean ×1.614 (below the adopted ×3.377)
 *   - tile96x32r12x4w8x8k16: geometric mean ×1.217 (below the adopted ×3.377)
 *   - tile96x32r12x4w8x8k16s: geometric mean ×1.702 (below the adopted ×3.377)
 *   - tile96x32r12x4w8x8k32: geometric mean ×1.305 (below the adopted ×3.377)
 *   - tile96x32r12x4w8x8k32s: geometric mean ×2.087 (below the adopted ×3.377)
 *   - tile96x32r12x4w8x8k64: geometric mean ×1.257 (below the adopted ×3.377)
 *   - tile96x32r12x4w8x8k64s: geometric mean ×2.238 (below the adopted ×3.377)
 *   - tile96x64r12x4w16x8k16: geometric mean ×1.170 (below the adopted ×3.377)
 *   - tile96x64r12x4w16x8k16s: geometric mean ×1.489 (below the adopted ×3.377)
 *   - tile96x64r12x4w16x8k32: geometric mean ×1.266 (below the adopted ×3.377)
 *   - tile96x64r12x4w16x8k32s: geometric mean ×1.703 (below the adopted ×3.377)
 *   - tile96x64r12x4w16x8k64: geometric mean ×1.363 (below the adopted ×3.377)
 *   - tile96x64r12x4w16x8k64s: geometric mean ×2.318 (below the adopted ×3.377)
 *   - tile96x64r12x8w8x8k16: i8a8-linear-m185-n1280-k1280: ×1.021 < ×1.050
 *   - tile96x64r12x8w8x8k16s: geometric mean ×1.339 (below the adopted ×3.377)
 *   - tile96x64r12x8w8x8k32: i8a8-linear-m256-n1280-k1280: ×1.016 < ×1.050
 *   - tile96x64r12x8w8x8k32s: geometric mean ×1.542 (below the adopted ×3.377)
 *   - tile96x64r12x8w8x8k64: i8a8-linear-m256-n1280-k1280: ×1.009 < ×1.050
 *   - tile96x64r12x8w8x8k64s: geometric mean ×1.577 (below the adopted ×3.377)
 * - i8a8.linear 257〜512 (i8a8 linear row count 257〜512, 1 cases): adopted tile64x64r8x4w16x8k64s ×1.571 (×1.571–×1.571)
 *   - tile128x128r16x8w16x8k16: i8a8-linear-m512-n3072-k3072: ×0.908 < ×1.050
 *   - tile128x128r16x8w16x8k16s: i8a8-linear-m512-n3072-k3072: ×0.997 < ×1.050
 *   - tile128x128r16x8w16x8k32: i8a8-linear-m512-n3072-k3072: ×0.948 < ×1.050
 *   - tile128x128r16x8w16x8k32s: geometric mean ×1.076 (below the adopted ×1.571)
 *   - tile128x128r16x8w16x8k64: i8a8-linear-m512-n3072-k3072: ×0.819 < ×1.050
 *   - tile128x128r16x8w16x8k64s: geometric mean ×1.085 (below the adopted ×1.571)
 *   - tile128x128r8x8w16x16k16: i8a8-linear-m512-n3072-k3072: ×0.894 < ×1.050
 *   - tile128x128r8x8w16x16k16s: i8a8-linear-m512-n3072-k3072: ×1.021 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-linear-m512-n3072-k3072: ×0.903 < ×1.050
 *   - tile128x128r8x8w16x16k32s: geometric mean ×1.220 (below the adopted ×1.571)
 *   - tile128x128r8x8w16x16k64: i8a8-linear-m512-n3072-k3072: ×0.802 < ×1.050
 *   - tile128x128r8x8w16x16k64s: geometric mean ×1.194 (below the adopted ×1.571)
 *   - tile128x32r16x4w8x8k16: i8a8-linear-m512-n3072-k3072: ×0.908 < ×1.050
 *   - tile128x32r16x4w8x8k16s: geometric mean ×1.223 (below the adopted ×1.571)
 *   - tile128x32r16x4w8x8k32: i8a8-linear-m512-n3072-k3072: ×0.959 < ×1.050
 *   - tile128x32r16x4w8x8k32s: geometric mean ×1.414 (below the adopted ×1.571)
 *   - tile128x32r16x4w8x8k64: i8a8-linear-m512-n3072-k3072: ×0.911 < ×1.050
 *   - tile128x32r16x4w8x8k64s: geometric mean ×1.393 (below the adopted ×1.571)
 *   - tile128x32r8x4w8x16k16: geometric mean ×1.331 (below the adopted ×1.571)
 *   - tile128x32r8x4w8x16k16s: geometric mean ×1.318 (below the adopted ×1.571)
 *   - tile128x32r8x4w8x16k32: i8a8-linear-m512-n3072-k3072: ×1.027 < ×1.050
 *   - tile128x32r8x4w8x16k32s: geometric mean ×1.462 (below the adopted ×1.571)
 *   - tile128x32r8x4w8x16k64: i8a8-linear-m512-n3072-k3072: ×0.897 < ×1.050
 *   - tile128x32r8x4w8x16k64s: geometric mean ×1.487 (below the adopted ×1.571)
 *   - tile128x64r16x4w16x8k16: geometric mean ×1.058 (below the adopted ×1.571)
 *   - tile128x64r16x4w16x8k16s: geometric mean ×1.227 (below the adopted ×1.571)
 *   - tile128x64r16x4w16x8k32: i8a8-linear-m512-n3072-k3072: ×0.974 < ×1.050
 *   - tile128x64r16x4w16x8k32s: geometric mean ×1.472 (below the adopted ×1.571)
 *   - tile128x64r16x4w16x8k64: i8a8-linear-m512-n3072-k3072: ×0.955 < ×1.050
 *   - tile128x64r16x4w16x8k64s: geometric mean ×1.500 (below the adopted ×1.571)
 *   - tile128x64r16x8w8x8k16: i8a8-linear-m512-n3072-k3072: ×0.913 < ×1.050
 *   - tile128x64r16x8w8x8k16s: i8a8-linear-m512-n3072-k3072: ×1.010 < ×1.050
 *   - tile128x64r16x8w8x8k32: i8a8-linear-m512-n3072-k3072: ×0.902 < ×1.050
 *   - tile128x64r16x8w8x8k32s: geometric mean ×1.069 (below the adopted ×1.571)
 *   - tile128x64r16x8w8x8k64: i8a8-linear-m512-n3072-k3072: ×0.817 < ×1.050
 *   - tile128x64r16x8w8x8k64s: geometric mean ×1.098 (below the adopted ×1.571)
 *   - tile128x64r8x4w16x16k16: geometric mean ×1.279 (below the adopted ×1.571)
 *   - tile128x64r8x4w16x16k16s: geometric mean ×1.210 (below the adopted ×1.571)
 *   - tile128x64r8x4w16x16k32: i8a8-linear-m512-n3072-k3072: ×1.025 < ×1.050
 *   - tile128x64r8x4w16x16k32s: geometric mean ×1.358 (below the adopted ×1.571)
 *   - tile128x64r8x4w16x16k64: i8a8-linear-m512-n3072-k3072: ×0.898 < ×1.050
 *   - tile128x64r8x4w16x16k64s: geometric mean ×1.424 (below the adopted ×1.571)
 *   - tile128x64r8x8w8x16k16s: geometric mean ×1.167 (below the adopted ×1.571)
 *   - tile128x64r8x8w8x16k32: i8a8-linear-m512-n3072-k3072: ×0.937 < ×1.050
 *   - tile128x64r8x8w8x16k32s: geometric mean ×1.401 (below the adopted ×1.571)
 *   - tile128x64r8x8w8x16k64: i8a8-linear-m512-n3072-k3072: ×0.900 < ×1.050
 *   - tile128x64r8x8w8x16k64s: geometric mean ×1.432 (below the adopted ×1.571)
 *   - tile16x128r4x8w16x4k16: i8a8-linear-m512-n3072-k3072: ×0.863 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-linear-m512-n3072-k3072: ×0.868 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-linear-m512-n3072-k3072: ×0.871 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.892 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-linear-m512-n3072-k3072: ×0.857 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-linear-m512-n3072-k3072: ×0.908 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-linear-m512-n3072-k3072: ×0.808 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.807 < ×1.050
 *   - tile192x128r12x8w16x16k16: i8a8-linear-m512-n3072-k3072: ×0.993 < ×1.050
 *   - tile192x128r12x8w16x16k16s: geometric mean ×1.056 (below the adopted ×1.571)
 *   - tile192x128r12x8w16x16k32: i8a8-linear-m512-n3072-k3072: ×0.911 < ×1.050
 *   - tile192x128r12x8w16x16k32s: geometric mean ×1.186 (below the adopted ×1.571)
 *   - tile192x32r12x4w8x16k16: i8a8-linear-m512-n3072-k3072: ×0.957 < ×1.050
 *   - tile192x32r12x4w8x16k16s: geometric mean ×1.254 (below the adopted ×1.571)
 *   - tile192x32r12x4w8x16k32: i8a8-linear-m512-n3072-k3072: ×0.921 < ×1.050
 *   - tile192x32r12x4w8x16k32s: geometric mean ×1.348 (below the adopted ×1.571)
 *   - tile192x32r12x4w8x16k64: i8a8-linear-m512-n3072-k3072: ×0.830 < ×1.050
 *   - tile192x32r12x4w8x16k64s: geometric mean ×1.335 (below the adopted ×1.571)
 *   - tile192x64r12x4w16x16k16: i8a8-linear-m512-n3072-k3072: ×0.949 < ×1.050
 *   - tile192x64r12x4w16x16k16s: geometric mean ×1.270 (below the adopted ×1.571)
 *   - tile192x64r12x4w16x16k32: i8a8-linear-m512-n3072-k3072: ×0.993 < ×1.050
 *   - tile192x64r12x4w16x16k32s: geometric mean ×1.369 (below the adopted ×1.571)
 *   - tile192x64r12x4w16x16k64: i8a8-linear-m512-n3072-k3072: ×0.941 < ×1.050
 *   - tile192x64r12x4w16x16k64s: geometric mean ×1.414 (below the adopted ×1.571)
 *   - tile192x64r12x8w8x16k16: i8a8-linear-m512-n3072-k3072: ×1.016 < ×1.050
 *   - tile192x64r12x8w8x16k16s: geometric mean ×1.225 (below the adopted ×1.571)
 *   - tile192x64r12x8w8x16k32: i8a8-linear-m512-n3072-k3072: ×0.947 < ×1.050
 *   - tile192x64r12x8w8x16k32s: geometric mean ×1.290 (below the adopted ×1.571)
 *   - tile192x64r12x8w8x16k64: i8a8-linear-m512-n3072-k3072: ×0.921 < ×1.050
 *   - tile192x64r12x8w8x16k64s: geometric mean ×1.488 (below the adopted ×1.571)
 *   - tile256x128r16x8w16x16k16: i8a8-linear-m512-n3072-k3072: ×0.821 < ×1.050
 *   - tile256x128r16x8w16x16k16s: i8a8-linear-m512-n3072-k3072: ×0.779 < ×1.050
 *   - tile256x128r16x8w16x16k32: i8a8-linear-m512-n3072-k3072: ×0.940 < ×1.050
 *   - tile256x128r16x8w16x16k32s: i8a8-linear-m512-n3072-k3072: ×0.961 < ×1.050
 *   - tile256x32r16x4w8x16k16: i8a8-linear-m512-n3072-k3072: ×0.888 < ×1.050
 *   - tile256x32r16x4w8x16k16s: geometric mean ×1.181 (below the adopted ×1.571)
 *   - tile256x32r16x4w8x16k32: i8a8-linear-m512-n3072-k3072: ×0.936 < ×1.050
 *   - tile256x32r16x4w8x16k32s: geometric mean ×1.320 (below the adopted ×1.571)
 *   - tile256x64r16x4w16x16k16: i8a8-linear-m512-n3072-k3072: ×0.889 < ×1.050
 *   - tile256x64r16x4w16x16k16s: geometric mean ×1.086 (below the adopted ×1.571)
 *   - tile256x64r16x4w16x16k32: i8a8-linear-m512-n3072-k3072: ×0.887 < ×1.050
 *   - tile256x64r16x4w16x16k32s: geometric mean ×1.093 (below the adopted ×1.571)
 *   - tile256x64r16x8w8x16k16: i8a8-linear-m512-n3072-k3072: ×0.869 < ×1.050
 *   - tile256x64r16x8w8x16k16s: i8a8-linear-m512-n3072-k3072: ×0.889 < ×1.050
 *   - tile256x64r16x8w8x16k32: i8a8-linear-m512-n3072-k3072: ×0.886 < ×1.050
 *   - tile256x64r16x8w8x16k32s: geometric mean ×1.068 (below the adopted ×1.571)
 *   - tile32x128r4x8w16x8k16: geometric mean ×1.165 (below the adopted ×1.571)
 *   - tile32x128r4x8w16x8k32: i8a8-linear-m512-n3072-k3072: ×1.016 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-linear-m512-n3072-k3072: ×0.972 < ×1.050
 *   - tile32x128r8x8w16x4k16s: geometric mean ×1.303 (below the adopted ×1.571)
 *   - tile32x128r8x8w16x4k32: i8a8-linear-m512-n3072-k3072: ×0.940 < ×1.050
 *   - tile32x128r8x8w16x4k32s: geometric mean ×1.403 (below the adopted ×1.571)
 *   - tile32x128r8x8w16x4k64: i8a8-linear-m512-n3072-k3072: ×0.881 < ×1.050
 *   - tile32x128r8x8w16x4k64s: geometric mean ×1.349 (below the adopted ×1.571)
 *   - tile32x32r4x4w8x8k16: i8a8-linear-m512-n3072-k3072: ×0.977 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-linear-m512-n3072-k3072: ×0.985 < ×1.050
 *   - tile32x32r8x4w8x4k16: geometric mean ×1.097 (below the adopted ×1.571)
 *   - tile32x32r8x4w8x4k16s: geometric mean ×1.232 (below the adopted ×1.571)
 *   - tile32x32r8x4w8x4k32: i8a8-linear-m512-n3072-k3072: ×1.035 < ×1.050
 *   - tile32x32r8x4w8x4k32s: geometric mean ×1.422 (below the adopted ×1.571)
 *   - tile32x32r8x4w8x4k64: i8a8-linear-m512-n3072-k3072: ×0.798 < ×1.050
 *   - tile32x32r8x4w8x4k64s: geometric mean ×1.451 (below the adopted ×1.571)
 *   - tile32x64r4x4w16x8k16: i8a8-linear-m512-n3072-k3072: ×1.026 < ×1.050
 *   - tile32x64r4x4w16x8k32: geometric mean ×1.114 (below the adopted ×1.571)
 *   - tile32x64r4x8w8x8k16: geometric mean ×1.072 (below the adopted ×1.571)
 *   - tile32x64r4x8w8x8k32: i8a8-linear-m512-n3072-k3072: ×0.990 < ×1.050
 *   - tile32x64r8x4w16x4k16: geometric mean ×1.305 (below the adopted ×1.571)
 *   - tile32x64r8x4w16x4k16s: geometric mean ×1.421 (below the adopted ×1.571)
 *   - tile32x64r8x4w16x4k32: geometric mean ×1.073 (below the adopted ×1.571)
 *   - tile32x64r8x4w16x4k32s: geometric mean ×1.513 (below the adopted ×1.571)
 *   - tile32x64r8x4w16x4k64: i8a8-linear-m512-n3072-k3072: ×0.855 < ×1.050
 *   - tile32x64r8x4w16x4k64s: geometric mean ×1.467 (below the adopted ×1.571)
 *   - tile32x64r8x8w8x4k16: i8a8-linear-m512-n3072-k3072: ×0.979 < ×1.050
 *   - tile32x64r8x8w8x4k16s: geometric mean ×1.299 (below the adopted ×1.571)
 *   - tile32x64r8x8w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.924 < ×1.050
 *   - tile32x64r8x8w8x4k32s: geometric mean ×1.375 (below the adopted ×1.571)
 *   - tile32x64r8x8w8x4k64: i8a8-linear-m512-n3072-k3072: ×0.738 < ×1.050
 *   - tile32x64r8x8w8x4k64s: geometric mean ×1.375 (below the adopted ×1.571)
 *   - tile48x128r12x8w16x4k16: i8a8-linear-m512-n3072-k3072: ×1.020 < ×1.050
 *   - tile48x128r12x8w16x4k16s: geometric mean ×1.310 (below the adopted ×1.571)
 *   - tile48x128r12x8w16x4k32: i8a8-linear-m512-n3072-k3072: ×0.988 < ×1.050
 *   - tile48x128r12x8w16x4k32s: geometric mean ×1.383 (below the adopted ×1.571)
 *   - tile48x128r12x8w16x4k64: i8a8-linear-m512-n3072-k3072: ×0.940 < ×1.050
 *   - tile48x128r12x8w16x4k64s: geometric mean ×1.387 (below the adopted ×1.571)
 *   - tile48x32r12x4w8x4k16: i8a8-linear-m512-n3072-k3072: ×0.953 < ×1.050
 *   - tile48x32r12x4w8x4k16s: geometric mean ×1.337 (below the adopted ×1.571)
 *   - tile48x32r12x4w8x4k32: i8a8-linear-m512-n3072-k3072: ×1.010 < ×1.050
 *   - tile48x32r12x4w8x4k32s: geometric mean ×1.405 (below the adopted ×1.571)
 *   - tile48x32r12x4w8x4k64: i8a8-linear-m512-n3072-k3072: ×0.845 < ×1.050
 *   - tile48x32r12x4w8x4k64s: geometric mean ×1.422 (below the adopted ×1.571)
 *   - tile48x64r12x4w16x4k16: i8a8-linear-m512-n3072-k3072: ×0.977 < ×1.050
 *   - tile48x64r12x4w16x4k16s: geometric mean ×1.235 (below the adopted ×1.571)
 *   - tile48x64r12x4w16x4k32: i8a8-linear-m512-n3072-k3072: ×0.993 < ×1.050
 *   - tile48x64r12x4w16x4k32s: geometric mean ×1.409 (below the adopted ×1.571)
 *   - tile48x64r12x4w16x4k64: i8a8-linear-m512-n3072-k3072: ×0.919 < ×1.050
 *   - tile48x64r12x4w16x4k64s: geometric mean ×1.392 (below the adopted ×1.571)
 *   - tile48x64r12x8w8x4k16: geometric mean ×1.052 (below the adopted ×1.571)
 *   - tile48x64r12x8w8x4k16s: geometric mean ×1.320 (below the adopted ×1.571)
 *   - tile48x64r12x8w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.933 < ×1.050
 *   - tile48x64r12x8w8x4k32s: geometric mean ×1.301 (below the adopted ×1.571)
 *   - tile48x64r12x8w8x4k64: i8a8-linear-m512-n3072-k3072: ×0.931 < ×1.050
 *   - tile48x64r12x8w8x4k64s: geometric mean ×1.341 (below the adopted ×1.571)
 *   - tile64x128r16x8w16x4k16: i8a8-linear-m512-n3072-k3072: ×0.893 < ×1.050
 *   - tile64x128r16x8w16x4k16s: i8a8-linear-m512-n3072-k3072: ×0.998 < ×1.050
 *   - tile64x128r16x8w16x4k32: i8a8-linear-m512-n3072-k3072: ×0.914 < ×1.050
 *   - tile64x128r16x8w16x4k32s: geometric mean ×1.212 (below the adopted ×1.571)
 *   - tile64x128r16x8w16x4k64: i8a8-linear-m512-n3072-k3072: ×0.819 < ×1.050
 *   - tile64x128r16x8w16x4k64s: geometric mean ×1.096 (below the adopted ×1.571)
 *   - tile64x128r4x8w16x16k16: geometric mean ×1.094 (below the adopted ×1.571)
 *   - tile64x128r4x8w16x16k32: i8a8-linear-m512-n3072-k3072: ×0.953 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m512-n3072-k3072: ×0.977 < ×1.050
 *   - tile64x128r8x8w16x8k16s: geometric mean ×1.200 (below the adopted ×1.571)
 *   - tile64x128r8x8w16x8k32: i8a8-linear-m512-n3072-k3072: ×1.029 < ×1.050
 *   - tile64x128r8x8w16x8k32s: geometric mean ×1.396 (below the adopted ×1.571)
 *   - tile64x128r8x8w16x8k64: i8a8-linear-m512-n3072-k3072: ×0.923 < ×1.050
 *   - tile64x128r8x8w16x8k64s: geometric mean ×1.433 (below the adopted ×1.571)
 *   - tile64x32r16x4w8x4k16: i8a8-linear-m512-n3072-k3072: ×1.011 < ×1.050
 *   - tile64x32r16x4w8x4k16s: geometric mean ×1.344 (below the adopted ×1.571)
 *   - tile64x32r16x4w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.952 < ×1.050
 *   - tile64x32r16x4w8x4k32s: geometric mean ×1.423 (below the adopted ×1.571)
 *   - tile64x32r16x4w8x4k64: i8a8-linear-m512-n3072-k3072: ×0.733 < ×1.050
 *   - tile64x32r16x4w8x4k64s: geometric mean ×1.419 (below the adopted ×1.571)
 *   - tile64x32r4x4w8x16k16: i8a8-linear-m512-n3072-k3072: ×1.027 < ×1.050
 *   - tile64x32r4x4w8x16k32: geometric mean ×1.119 (below the adopted ×1.571)
 *   - tile64x32r8x4w8x8k16: geometric mean ×1.301 (below the adopted ×1.571)
 *   - tile64x32r8x4w8x8k16s: geometric mean ×1.421 (below the adopted ×1.571)
 *   - tile64x32r8x4w8x8k32: geometric mean ×1.074 (below the adopted ×1.571)
 *   - tile64x32r8x4w8x8k32s: geometric mean ×1.538 (below the adopted ×1.571)
 *   - tile64x32r8x4w8x8k64: i8a8-linear-m512-n3072-k3072: ×0.884 < ×1.050
 *   - tile64x32r8x4w8x8k64s: geometric mean ×1.461 (below the adopted ×1.571)
 *   - tile64x64r16x4w16x4k16: i8a8-linear-m512-n3072-k3072: ×0.922 < ×1.050
 *   - tile64x64r16x4w16x4k16s: geometric mean ×1.335 (below the adopted ×1.571)
 *   - tile64x64r16x4w16x4k32: i8a8-linear-m512-n3072-k3072: ×0.992 < ×1.050
 *   - tile64x64r16x4w16x4k32s: geometric mean ×1.417 (below the adopted ×1.571)
 *   - tile64x64r16x4w16x4k64: i8a8-linear-m512-n3072-k3072: ×0.943 < ×1.050
 *   - tile64x64r16x4w16x4k64s: geometric mean ×1.472 (below the adopted ×1.571)
 *   - tile64x64r16x8w8x4k16: i8a8-linear-m512-n3072-k3072: ×0.912 < ×1.050
 *   - tile64x64r16x8w8x4k16s: i8a8-linear-m512-n3072-k3072: ×1.028 < ×1.050
 *   - tile64x64r16x8w8x4k32: i8a8-linear-m512-n3072-k3072: ×0.837 < ×1.050
 *   - tile64x64r16x8w8x4k32s: i8a8-linear-m512-n3072-k3072: ×1.029 < ×1.050
 *   - tile64x64r16x8w8x4k64: i8a8-linear-m512-n3072-k3072: ×0.695 < ×1.050
 *   - tile64x64r16x8w8x4k64s: geometric mean ×1.088 (below the adopted ×1.571)
 *   - tile64x64r4x4w16x16k16: geometric mean ×1.207 (below the adopted ×1.571)
 *   - tile64x64r4x4w16x16k32: geometric mean ×1.309 (below the adopted ×1.571)
 *   - tile64x64r4x8w8x16k16: geometric mean ×1.254 (below the adopted ×1.571)
 *   - tile64x64r4x8w8x16k32: geometric mean ×1.085 (below the adopted ×1.571)
 *   - tile64x64r8x4w16x8k16: geometric mean ×1.393 (below the adopted ×1.571)
 *   - tile64x64r8x4w16x8k16s: geometric mean ×1.437 (below the adopted ×1.571)
 *   - tile64x64r8x4w16x8k32: geometric mean ×1.146 (below the adopted ×1.571)
 *   - tile64x64r8x4w16x8k32s: geometric mean ×1.547 (below the adopted ×1.571)
 *   - tile64x64r8x4w16x8k64: i8a8-linear-m512-n3072-k3072: ×1.003 < ×1.050
 *   - tile64x64r8x8w8x8k16: i8a8-linear-m512-n3072-k3072: ×1.013 < ×1.050
 *   - tile64x64r8x8w8x8k16s: geometric mean ×1.347 (below the adopted ×1.571)
 *   - tile64x64r8x8w8x8k32: i8a8-linear-m512-n3072-k3072: ×0.918 < ×1.050
 *   - tile64x64r8x8w8x8k32s: geometric mean ×1.414 (below the adopted ×1.571)
 *   - tile64x64r8x8w8x8k64: i8a8-linear-m512-n3072-k3072: ×0.946 < ×1.050
 *   - tile64x64r8x8w8x8k64s: geometric mean ×1.439 (below the adopted ×1.571)
 *   - tile96x128r12x8w16x8k16: i8a8-linear-m512-n3072-k3072: ×1.036 < ×1.050
 *   - tile96x128r12x8w16x8k16s: geometric mean ×1.383 (below the adopted ×1.571)
 *   - tile96x128r12x8w16x8k32: i8a8-linear-m512-n3072-k3072: ×0.920 < ×1.050
 *   - tile96x128r12x8w16x8k32s: geometric mean ×1.318 (below the adopted ×1.571)
 *   - tile96x128r12x8w16x8k64: i8a8-linear-m512-n3072-k3072: ×0.939 < ×1.050
 *   - tile96x128r12x8w16x8k64s: geometric mean ×1.415 (below the adopted ×1.571)
 *   - tile96x32r12x4w8x8k16: i8a8-linear-m512-n3072-k3072: ×0.887 < ×1.050
 *   - tile96x32r12x4w8x8k16s: geometric mean ×1.254 (below the adopted ×1.571)
 *   - tile96x32r12x4w8x8k32: i8a8-linear-m512-n3072-k3072: ×0.902 < ×1.050
 *   - tile96x32r12x4w8x8k32s: geometric mean ×1.376 (below the adopted ×1.571)
 *   - tile96x32r12x4w8x8k64: i8a8-linear-m512-n3072-k3072: ×0.844 < ×1.050
 *   - tile96x32r12x4w8x8k64s: geometric mean ×1.337 (below the adopted ×1.571)
 *   - tile96x64r12x4w16x8k16: i8a8-linear-m512-n3072-k3072: ×0.916 < ×1.050
 *   - tile96x64r12x4w16x8k16s: geometric mean ×1.272 (below the adopted ×1.571)
 *   - tile96x64r12x4w16x8k32: i8a8-linear-m512-n3072-k3072: ×0.939 < ×1.050
 *   - tile96x64r12x4w16x8k32s: geometric mean ×1.357 (below the adopted ×1.571)
 *   - tile96x64r12x4w16x8k64: i8a8-linear-m512-n3072-k3072: ×0.926 < ×1.050
 *   - tile96x64r12x4w16x8k64s: geometric mean ×1.461 (below the adopted ×1.571)
 *   - tile96x64r12x8w8x8k16: i8a8-linear-m512-n3072-k3072: ×1.006 < ×1.050
 *   - tile96x64r12x8w8x8k16s: geometric mean ×1.202 (below the adopted ×1.571)
 *   - tile96x64r12x8w8x8k32: i8a8-linear-m512-n3072-k3072: ×1.030 < ×1.050
 *   - tile96x64r12x8w8x8k32s: geometric mean ×1.406 (below the adopted ×1.571)
 *   - tile96x64r12x8w8x8k64: i8a8-linear-m512-n3072-k3072: ×0.913 < ×1.050
 *   - tile96x64r12x8w8x8k64s: geometric mean ×1.364 (below the adopted ×1.571)
 * - i8a8.linear > 512 (i8a8 linear row count > 512, 9 cases): adopted tile128x64r16x4w16x8k64s ×1.406 (×1.379–×1.440)
 *   - tile128x128r16x8w16x8k16: i8a8-linear-m4096-n2048-k2048: ×1.022 < ×1.050
 *   - tile128x128r16x8w16x8k16s: i8a8-linear-m4096-n2048-k2048: ×0.909 < ×1.050
 *   - tile128x128r16x8w16x8k32: i8a8-linear-m1024-n8192-k2048: ×0.995 < ×1.050
 *   - tile128x128r16x8w16x8k32s: i8a8-linear-m4096-n2048-k2048: ×0.958 < ×1.050
 *   - tile128x128r16x8w16x8k64: i8a8-linear-m4096-n2048-k2048: ×0.900 < ×1.050
 *   - tile128x128r16x8w16x8k64s: i8a8-linear-m4096-n2048-k2048: ×1.015 < ×1.050
 *   - tile128x128r8x8w16x16k16: i8a8-linear-m4096-n2048-k2048: ×0.928 < ×1.050
 *   - tile128x128r8x8w16x16k16s: geometric mean ×1.136 (below the adopted ×1.406)
 *   - tile128x128r8x8w16x16k32: i8a8-linear-m4096-n2048-k2048: ×0.950 < ×1.050
 *   - tile128x128r8x8w16x16k32s: geometric mean ×1.295 (below the adopted ×1.406)
 *   - tile128x128r8x8w16x16k64: i8a8-linear-m4096-n2048-k2048: ×0.878 < ×1.050
 *   - tile128x128r8x8w16x16k64s: geometric mean ×1.326 (below the adopted ×1.406)
 *   - tile128x32r16x4w8x8k16: i8a8-linear-m7920-n3072-k14336: ×0.770 < ×1.050
 *   - tile128x32r16x4w8x8k16s: i8a8-linear-m7920-n3072-k14336: ×0.913 < ×1.050
 *   - tile128x32r16x4w8x8k32: i8a8-linear-m4096-n2048-k8192: ×0.887 < ×1.050
 *   - tile128x32r16x4w8x8k32s: geometric mean ×1.297 (below the adopted ×1.406)
 *   - tile128x32r16x4w8x8k64: i8a8-linear-m7920-n14336-k3072: ×0.773 < ×1.050
 *   - tile128x32r16x4w8x8k64s: geometric mean ×1.324 (below the adopted ×1.406)
 *   - tile128x32r8x4w8x16k16: i8a8-linear-m7920-n3072-k14336: ×0.725 < ×1.050
 *   - tile128x32r8x4w8x16k16s: i8a8-linear-m7920-n3072-k14336: ×0.698 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-linear-m4096-n2048-k8192: ×0.875 < ×1.050
 *   - tile128x32r8x4w8x16k32s: geometric mean ×1.272 (below the adopted ×1.406)
 *   - tile128x32r8x4w8x16k64: i8a8-linear-m7920-n14336-k3072: ×0.735 < ×1.050
 *   - tile128x32r8x4w8x16k64s: geometric mean ×1.291 (below the adopted ×1.406)
 *   - tile128x64r16x4w16x8k16: i8a8-linear-m7920-n14336-k3072: ×1.021 < ×1.050
 *   - tile128x64r16x4w16x8k16s: geometric mean ×1.261 (below the adopted ×1.406)
 *   - tile128x64r16x4w16x8k32: i8a8-linear-m1024-n2048-k2048: ×0.941 < ×1.050
 *   - tile128x64r16x4w16x8k32s: geometric mean ×1.373 (below the adopted ×1.406)
 *   - tile128x64r16x4w16x8k64: i8a8-linear-m7920-n14336-k3072: ×0.875 < ×1.050
 *   - tile128x64r16x8w8x8k16: i8a8-linear-m1024-n8192-k2048: ×1.021 < ×1.050
 *   - tile128x64r16x8w8x8k16s: i8a8-linear-m1024-n8192-k2048: ×0.935 < ×1.050
 *   - tile128x64r16x8w8x8k32: i8a8-linear-m4096-n2048-k2048: ×0.957 < ×1.050
 *   - tile128x64r16x8w8x8k32s: i8a8-linear-m4096-n2048-k2048: ×0.948 < ×1.050
 *   - tile128x64r16x8w8x8k64: i8a8-linear-m4096-n2048-k2048: ×0.821 < ×1.050
 *   - tile128x64r16x8w8x8k64s: i8a8-linear-m4096-n2048-k2048: ×0.985 < ×1.050
 *   - tile128x64r8x4w16x16k16: geometric mean ×1.147 (below the adopted ×1.406)
 *   - tile128x64r8x4w16x16k16s: geometric mean ×1.120 (below the adopted ×1.406)
 *   - tile128x64r8x4w16x16k32: i8a8-linear-m4096-n8192-k2048: ×0.951 < ×1.050
 *   - tile128x64r8x4w16x16k32s: geometric mean ×1.304 (below the adopted ×1.406)
 *   - tile128x64r8x4w16x16k64: i8a8-linear-m4096-n2048-k2048: ×0.840 < ×1.050
 *   - tile128x64r8x4w16x16k64s: geometric mean ×1.313 (below the adopted ×1.406)
 *   - tile128x64r8x8w8x16k16s: geometric mean ×1.205 (below the adopted ×1.406)
 *   - tile128x64r8x8w8x16k32: i8a8-linear-m1024-n2048-k2048: ×0.897 < ×1.050
 *   - tile128x64r8x8w8x16k32s: geometric mean ×1.347 (below the adopted ×1.406)
 *   - tile128x64r8x8w8x16k64: i8a8-linear-m7920-n14336-k3072: ×0.871 < ×1.050
 *   - tile128x64r8x8w8x16k64s: geometric mean ×1.368 (below the adopted ×1.406)
 *   - tile16x128r4x8w16x4k16: i8a8-linear-m7920-n3072-k14336: ×0.128 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-linear-m7920-n3072-k14336: ×0.253 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-linear-m7920-n3072-k14336: ×0.213 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-linear-m7920-n14336-k3072: ×0.244 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-linear-m7920-n3072-k14336: ×0.126 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-linear-m7920-n3072-k14336: ×0.247 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-linear-m7920-n3072-k14336: ×0.129 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-linear-m7920-n14336-k3072: ×0.258 < ×1.050
 *   - tile192x128r12x8w16x16k16: i8a8-linear-m1024-n2048-k2048: ×0.790 < ×1.050
 *   - tile192x128r12x8w16x16k16s: i8a8-linear-m1024-n2048-k2048: ×0.697 < ×1.050
 *   - tile192x128r12x8w16x16k32: i8a8-linear-m1024-n2048-k2048: ×0.684 < ×1.050
 *   - tile192x128r12x8w16x16k32s: i8a8-linear-m1024-n2048-k2048: ×0.766 < ×1.050
 *   - tile192x32r12x4w8x16k16: i8a8-linear-m7920-n3072-k14336: ×0.674 < ×1.050
 *   - tile192x32r12x4w8x16k16s: i8a8-linear-m7920-n3072-k14336: ×0.764 < ×1.050
 *   - tile192x32r12x4w8x16k32: i8a8-linear-m1024-n8192-k2048: ×0.792 < ×1.050
 *   - tile192x32r12x4w8x16k32s: geometric mean ×1.254 (below the adopted ×1.406)
 *   - tile192x32r12x4w8x16k64: i8a8-linear-m1024-n8192-k2048: ×0.682 < ×1.050
 *   - tile192x32r12x4w8x16k64s: geometric mean ×1.278 (below the adopted ×1.406)
 *   - tile192x64r12x4w16x16k16: i8a8-linear-m1024-n8192-k2048: ×0.878 < ×1.050
 *   - tile192x64r12x4w16x16k16s: i8a8-linear-m1024-n2048-k2048: ×0.967 < ×1.050
 *   - tile192x64r12x4w16x16k32: i8a8-linear-m1024-n2048-k2048: ×0.913 < ×1.050
 *   - tile192x64r12x4w16x16k32s: i8a8-linear-m1024-n2048-k8192: ×1.017 < ×1.050
 *   - tile192x64r12x4w16x16k64: i8a8-linear-m1024-n8192-k2048: ×0.781 < ×1.050
 *   - tile192x64r12x4w16x16k64s: geometric mean ×1.231 (below the adopted ×1.406)
 *   - tile192x64r12x8w8x16k16: i8a8-linear-m1024-n8192-k2048: ×0.934 < ×1.050
 *   - tile192x64r12x8w8x16k16s: geometric mean ×1.215 (below the adopted ×1.406)
 *   - tile192x64r12x8w8x16k32: i8a8-linear-m1024-n2048-k2048: ×0.855 < ×1.050
 *   - tile192x64r12x8w8x16k32s: geometric mean ×1.264 (below the adopted ×1.406)
 *   - tile192x64r12x8w8x16k64: i8a8-linear-m1024-n8192-k2048: ×0.774 < ×1.050
 *   - tile192x64r12x8w8x16k64s: geometric mean ×1.318 (below the adopted ×1.406)
 *   - tile256x128r16x8w16x16k16: i8a8-linear-m4096-n2048-k2048: ×0.781 < ×1.050
 *   - tile256x128r16x8w16x16k16s: i8a8-linear-m4096-n2048-k2048: ×0.750 < ×1.050
 *   - tile256x128r16x8w16x16k32: i8a8-linear-m1024-n8192-k2048: ×0.880 < ×1.050
 *   - tile256x128r16x8w16x16k32s: i8a8-linear-m4096-n2048-k2048: ×0.906 < ×1.050
 *   - tile256x32r16x4w8x16k16: i8a8-linear-m7920-n3072-k14336: ×0.683 < ×1.050
 *   - tile256x32r16x4w8x16k16s: i8a8-linear-m7920-n3072-k14336: ×0.776 < ×1.050
 *   - tile256x32r16x4w8x16k32: i8a8-linear-m4096-n2048-k8192: ×0.884 < ×1.050
 *   - tile256x32r16x4w8x16k32s: geometric mean ×1.297 (below the adopted ×1.406)
 *   - tile256x64r16x4w16x16k16: i8a8-linear-m4096-n2048-k2048: ×0.951 < ×1.050
 *   - tile256x64r16x4w16x16k16s: geometric mean ×1.160 (below the adopted ×1.406)
 *   - tile256x64r16x4w16x16k32: i8a8-linear-m4096-n2048-k2048: ×0.941 < ×1.050
 *   - tile256x64r16x4w16x16k32s: geometric mean ×1.230 (below the adopted ×1.406)
 *   - tile256x64r16x8w8x16k16: i8a8-linear-m4096-n2048-k2048: ×0.996 < ×1.050
 *   - tile256x64r16x8w8x16k16s: i8a8-linear-m4096-n2048-k2048: ×0.829 < ×1.050
 *   - tile256x64r16x8w8x16k32: i8a8-linear-m4096-n2048-k2048: ×0.970 < ×1.050
 *   - tile256x64r16x8w8x16k32s: i8a8-linear-m4096-n2048-k2048: ×0.952 < ×1.050
 *   - tile32x128r4x8w16x8k16: i8a8-linear-m7920-n3072-k14336: ×0.317 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-linear-m7920-n14336-k3072: ×0.531 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-linear-m7920-n3072-k14336: ×0.250 < ×1.050
 *   - tile32x128r8x8w16x4k16s: i8a8-linear-m7920-n3072-k14336: ×0.247 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-linear-m7920-n14336-k3072: ×0.528 < ×1.050
 *   - tile32x128r8x8w16x4k32s: i8a8-linear-m7920-n3072-k14336: ×0.487 < ×1.050
 *   - tile32x128r8x8w16x4k64: i8a8-linear-m7920-n14336-k3072: ×0.709 < ×1.050
 *   - tile32x128r8x8w16x4k64s: i8a8-linear-m7920-n14336-k3072: ×0.999 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-linear-m7920-n3072-k14336: ×0.362 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-linear-m7920-n14336-k3072: ×0.488 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-linear-m7920-n3072-k14336: ×0.348 < ×1.050
 *   - tile32x32r8x4w8x4k16s: i8a8-linear-m7920-n3072-k14336: ×0.330 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-linear-m7920-n14336-k3072: ×0.493 < ×1.050
 *   - tile32x32r8x4w8x4k32s: i8a8-linear-m7920-n14336-k3072: ×0.488 < ×1.050
 *   - tile32x32r8x4w8x4k64: i8a8-linear-m7920-n14336-k3072: ×0.613 < ×1.050
 *   - tile32x32r8x4w8x4k64s: i8a8-linear-m7920-n14336-k3072: ×0.928 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-linear-m7920-n3072-k14336: ×0.403 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-linear-m7920-n3072-k14336: ×0.498 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-linear-m7920-n3072-k14336: ×0.262 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-linear-m7920-n3072-k14336: ×0.521 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-linear-m7920-n3072-k14336: ×0.260 < ×1.050
 *   - tile32x64r8x4w16x4k16s: i8a8-linear-m7920-n3072-k14336: ×0.242 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-linear-m7920-n3072-k14336: ×0.479 < ×1.050
 *   - tile32x64r8x4w16x4k32s: i8a8-linear-m7920-n3072-k14336: ×0.463 < ×1.050
 *   - tile32x64r8x4w16x4k64: i8a8-linear-m7920-n14336-k3072: ×0.667 < ×1.050
 *   - tile32x64r8x4w16x4k64s: i8a8-linear-m7920-n14336-k3072: ×0.989 < ×1.050
 *   - tile32x64r8x8w8x4k16: i8a8-linear-m7920-n3072-k14336: ×0.241 < ×1.050
 *   - tile32x64r8x8w8x4k16s: i8a8-linear-m7920-n3072-k14336: ×0.239 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-linear-m7920-n14336-k3072: ×0.518 < ×1.050
 *   - tile32x64r8x8w8x4k32s: i8a8-linear-m7920-n3072-k14336: ×0.458 < ×1.050
 *   - tile32x64r8x8w8x4k64: i8a8-linear-m7920-n14336-k3072: ×0.638 < ×1.050
 *   - tile32x64r8x8w8x4k64s: i8a8-linear-m7920-n14336-k3072: ×1.011 < ×1.050
 *   - tile48x128r12x8w16x4k16: i8a8-linear-m7920-n3072-k14336: ×0.414 < ×1.050
 *   - tile48x128r12x8w16x4k16s: i8a8-linear-m7920-n3072-k14336: ×0.394 < ×1.050
 *   - tile48x128r12x8w16x4k32: i8a8-linear-m7920-n14336-k3072: ×0.849 < ×1.050
 *   - tile48x128r12x8w16x4k32s: i8a8-linear-m7920-n14336-k3072: ×0.781 < ×1.050
 *   - tile48x128r12x8w16x4k64: i8a8-linear-m1024-n2048-k2048: ×0.801 < ×1.050
 *   - tile48x128r12x8w16x4k64s: geometric mean ×1.211 (below the adopted ×1.406)
 *   - tile48x32r12x4w8x4k16: i8a8-linear-m7920-n3072-k14336: ×0.446 < ×1.050
 *   - tile48x32r12x4w8x4k16s: i8a8-linear-m7920-n3072-k14336: ×0.438 < ×1.050
 *   - tile48x32r12x4w8x4k32: i8a8-linear-m7920-n14336-k3072: ×0.718 < ×1.050
 *   - tile48x32r12x4w8x4k32s: i8a8-linear-m7920-n3072-k14336: ×0.694 < ×1.050
 *   - tile48x32r12x4w8x4k64: i8a8-linear-m7920-n14336-k3072: ×0.762 < ×1.050
 *   - tile48x32r12x4w8x4k64s: geometric mean ×1.243 (below the adopted ×1.406)
 *   - tile48x64r12x4w16x4k16: i8a8-linear-m7920-n3072-k14336: ×0.338 < ×1.050
 *   - tile48x64r12x4w16x4k16s: i8a8-linear-m7920-n3072-k14336: ×0.379 < ×1.050
 *   - tile48x64r12x4w16x4k32: i8a8-linear-m7920-n14336-k3072: ×0.810 < ×1.050
 *   - tile48x64r12x4w16x4k32s: i8a8-linear-m7920-n3072-k14336: ×0.685 < ×1.050
 *   - tile48x64r12x4w16x4k64: i8a8-linear-m1024-n8192-k2048: ×0.811 < ×1.050
 *   - tile48x64r12x4w16x4k64s: geometric mean ×1.277 (below the adopted ×1.406)
 *   - tile48x64r12x8w8x4k16: i8a8-linear-m7920-n3072-k14336: ×0.362 < ×1.050
 *   - tile48x64r12x8w8x4k16s: i8a8-linear-m7920-n3072-k14336: ×0.354 < ×1.050
 *   - tile48x64r12x8w8x4k32: i8a8-linear-m7920-n14336-k3072: ×0.746 < ×1.050
 *   - tile48x64r12x8w8x4k32s: i8a8-linear-m7920-n3072-k14336: ×0.700 < ×1.050
 *   - tile48x64r12x8w8x4k64: i8a8-linear-m7920-n14336-k3072: ×0.808 < ×1.050
 *   - tile48x64r12x8w8x4k64s: geometric mean ×1.258 (below the adopted ×1.406)
 *   - tile64x128r16x8w16x4k16: i8a8-linear-m7920-n14336-k3072: ×0.600 < ×1.050
 *   - tile64x128r16x8w16x4k16s: i8a8-linear-m1024-n8192-k2048: ×0.912 < ×1.050
 *   - tile64x128r16x8w16x4k32: i8a8-linear-m1024-n8192-k2048: ×0.971 < ×1.050
 *   - tile64x128r16x8w16x4k32s: geometric mean ×1.310 (below the adopted ×1.406)
 *   - tile64x128r16x8w16x4k64: i8a8-linear-m7920-n14336-k3072: ×0.853 < ×1.050
 *   - tile64x128r16x8w16x4k64s: i8a8-linear-m4096-n2048-k2048: ×0.956 < ×1.050
 *   - tile64x128r4x8w16x16k16: i8a8-linear-m7920-n14336-k3072: ×0.928 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-linear-m4096-n8192-k2048: ×0.876 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m7920-n3072-k14336: ×0.741 < ×1.050
 *   - tile64x128r8x8w16x8k16s: i8a8-linear-m7920-n3072-k14336: ×0.993 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-linear-m7920-n14336-k3072: ×0.964 < ×1.050
 *   - tile64x128r8x8w16x8k32s: i8a8-linear-m7920-n14336-k3072: ×1.046 < ×1.050
 *   - tile64x128r8x8w16x8k64: i8a8-linear-m7920-n14336-k3072: ×0.875 < ×1.050
 *   - tile64x128r8x8w16x8k64s: geometric mean ×1.344 (below the adopted ×1.406)
 *   - tile64x32r16x4w8x4k16: i8a8-linear-m7920-n3072-k14336: ×0.506 < ×1.050
 *   - tile64x32r16x4w8x4k16s: i8a8-linear-m7920-n3072-k14336: ×0.495 < ×1.050
 *   - tile64x32r16x4w8x4k32: i8a8-linear-m7920-n14336-k3072: ×0.865 < ×1.050
 *   - tile64x32r16x4w8x4k32s: i8a8-linear-m7920-n3072-k14336: ×0.817 < ×1.050
 *   - tile64x32r16x4w8x4k64: i8a8-linear-m7920-n14336-k3072: ×0.667 < ×1.050
 *   - tile64x32r16x4w8x4k64s: geometric mean ×1.283 (below the adopted ×1.406)
 *   - tile64x32r4x4w8x16k16: i8a8-linear-m7920-n3072-k14336: ×0.638 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-linear-m7920-n3072-k14336: ×0.864 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-linear-m7920-n3072-k14336: ×0.601 < ×1.050
 *   - tile64x32r8x4w8x8k16s: i8a8-linear-m7920-n3072-k14336: ×0.512 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-linear-m1024-n8192-k2048: ×0.871 < ×1.050
 *   - tile64x32r8x4w8x8k32s: i8a8-linear-m7920-n3072-k14336: ×0.874 < ×1.050
 *   - tile64x32r8x4w8x8k64: i8a8-linear-m7920-n14336-k3072: ×0.715 < ×1.050
 *   - tile64x32r8x4w8x8k64s: geometric mean ×1.256 (below the adopted ×1.406)
 *   - tile64x64r16x4w16x4k16: i8a8-linear-m7920-n3072-k14336: ×0.507 < ×1.050
 *   - tile64x64r16x4w16x4k16s: i8a8-linear-m7920-n3072-k14336: ×0.618 < ×1.050
 *   - tile64x64r16x4w16x4k32: i8a8-linear-m7920-n14336-k3072: ×0.919 < ×1.050
 *   - tile64x64r16x4w16x4k32s: i8a8-linear-m7920-n3072-k14336: ×0.989 < ×1.050
 *   - tile64x64r16x4w16x4k64: i8a8-linear-m7920-n14336-k3072: ×0.801 < ×1.050
 *   - tile64x64r16x4w16x4k64s: geometric mean ×1.354 (below the adopted ×1.406)
 *   - tile64x64r16x8w8x4k16: i8a8-linear-m7920-n3072-k14336: ×0.729 < ×1.050
 *   - tile64x64r16x8w8x4k16s: i8a8-linear-m1024-n8192-k2048: ×0.913 < ×1.050
 *   - tile64x64r16x8w8x4k32: i8a8-linear-m7920-n14336-k3072: ×0.917 < ×1.050
 *   - tile64x64r16x8w8x4k32s: i8a8-linear-m1024-n8192-k2048: ×0.944 < ×1.050
 *   - tile64x64r16x8w8x4k64: i8a8-linear-m4096-n2048-k2048: ×0.574 < ×1.050
 *   - tile64x64r16x8w8x4k64s: i8a8-linear-m1024-n8192-k2048: ×0.990 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-linear-m7920-n3072-k14336: ×0.900 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-linear-m7920-n3072-k14336: ×0.946 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-linear-m7920-n3072-k14336: ×0.878 < ×1.050
 *   - tile64x64r4x8w8x16k32: i8a8-linear-m7920-n14336-k3072: ×0.872 < ×1.050
 *   - tile64x64r8x4w16x8k16: i8a8-linear-m7920-n3072-k14336: ×0.885 < ×1.050
 *   - tile64x64r8x4w16x8k16s: i8a8-linear-m7920-n3072-k14336: ×0.741 < ×1.050
 *   - tile64x64r8x4w16x8k32: i8a8-linear-m4096-n8192-k2048: ×0.923 < ×1.050
 *   - tile64x64r8x4w16x8k32s: i8a8-linear-m7920-n3072-k14336: ×0.943 < ×1.050
 *   - tile64x64r8x4w16x8k64: i8a8-linear-m7920-n14336-k3072: ×0.808 < ×1.050
 *   - tile64x64r8x4w16x8k64s: geometric mean ×1.340 (below the adopted ×1.406)
 *   - tile64x64r8x8w8x8k16: i8a8-linear-m7920-n3072-k14336: ×0.756 < ×1.050
 *   - tile64x64r8x8w8x8k16s: i8a8-linear-m7920-n3072-k14336: ×0.616 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-linear-m1024-n8192-k2048: ×0.897 < ×1.050
 *   - tile64x64r8x8w8x8k32s: i8a8-linear-m7920-n3072-k14336: ×0.999 < ×1.050
 *   - tile64x64r8x8w8x8k64: i8a8-linear-m7920-n14336-k3072: ×0.796 < ×1.050
 *   - tile64x64r8x8w8x8k64s: geometric mean ×1.295 (below the adopted ×1.406)
 *   - tile96x128r12x8w16x8k16: i8a8-linear-m1024-n2048-k2048: ×0.971 < ×1.050
 *   - tile96x128r12x8w16x8k16s: geometric mean ×1.240 (below the adopted ×1.406)
 *   - tile96x128r12x8w16x8k32: i8a8-linear-m1024-n2048-k2048: ×0.811 < ×1.050
 *   - tile96x128r12x8w16x8k32s: geometric mean ×1.308 (below the adopted ×1.406)
 *   - tile96x128r12x8w16x8k64: i8a8-linear-m1024-n2048-k2048: ×0.839 < ×1.050
 *   - tile96x128r12x8w16x8k64s: geometric mean ×1.358 (below the adopted ×1.406)
 *   - tile96x32r12x4w8x8k16: i8a8-linear-m7920-n3072-k14336: ×0.590 < ×1.050
 *   - tile96x32r12x4w8x8k16s: i8a8-linear-m7920-n3072-k14336: ×0.713 < ×1.050
 *   - tile96x32r12x4w8x8k32: i8a8-linear-m1024-n8192-k2048: ×0.850 < ×1.050
 *   - tile96x32r12x4w8x8k32s: geometric mean ×1.276 (below the adopted ×1.406)
 *   - tile96x32r12x4w8x8k64: i8a8-linear-m1024-n8192-k2048: ×0.779 < ×1.050
 *   - tile96x32r12x4w8x8k64s: geometric mean ×1.288 (below the adopted ×1.406)
 *   - tile96x64r12x4w16x8k16: i8a8-linear-m4096-n2048-k8192: ×0.890 < ×1.050
 *   - tile96x64r12x4w16x8k16s: geometric mean ×1.133 (below the adopted ×1.406)
 *   - tile96x64r12x4w16x8k32: i8a8-linear-m4096-n2048-k2048: ×0.909 < ×1.050
 *   - tile96x64r12x4w16x8k32s: geometric mean ×1.258 (below the adopted ×1.406)
 *   - tile96x64r12x4w16x8k64: i8a8-linear-m4096-n2048-k2048: ×0.853 < ×1.050
 *   - tile96x64r12x4w16x8k64s: geometric mean ×1.377 (below the adopted ×1.406)
 *   - tile96x64r12x8w8x8k16: i8a8-linear-m1024-n2048-k2048: ×0.997 < ×1.050
 *   - tile96x64r12x8w8x8k16s: geometric mean ×1.234 (below the adopted ×1.406)
 *   - tile96x64r12x8w8x8k32: i8a8-linear-m1024-n2048-k2048: ×0.930 < ×1.050
 *   - tile96x64r12x8w8x8k32s: geometric mean ×1.333 (below the adopted ×1.406)
 *   - tile96x64r12x8w8x8k64: i8a8-linear-m1024-n2048-k2048: ×0.782 < ×1.050
 *   - tile96x64r12x8w8x8k64s: geometric mean ×1.230 (below the adopted ×1.406)
 * - i8a8.attentionQk (i8a8 attention ①QK, 4 cases): adopted tile64x64r8x4w16x8k16 ×1.103 (×1.091–×1.120)
 *   - tile128x64r8x4w16x16k16: i8a8-attention-qk-cross-m1024-n512: ×1.050 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-attention-qk-self-m4096-n4096: ×0.593 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-attention-qk-self-m4096-n4096: ×0.763 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-attention-qk-self-m4096-n4096: ×0.773 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-attention-qk-cross-m1024-n512: ×0.976 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-attention-qk-cross-m1024-n512: ×0.858 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-attention-qk-self-m4096-n4096: ×0.970 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-attention-qk-self-m4096-n4096: ×0.880 < ×1.050
 * - i8a8.attentionPv (i8a8 attention ③PV, 4 cases): keeps the default tile64x128r8x8w16x8k16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - tile128x64r8x4w16x16k16: i8a8-attention-pv-self-m1024-n1024: ×0.764 < ×1.050
 *   - tile128x64r8x8w8x16k16: i8a8-attention-pv-cross-m4096-n512: ×0.800 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-attention-pv-self-m1024-n1024: ×0.761 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-attention-pv-self-m1024-n1024: ×0.806 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-attention-pv-self-m1024-n1024: ×0.783 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-attention-pv-self-m1024-n1024: ×0.797 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-attention-pv-self-m1024-n1024: ×0.976 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-attention-pv-self-m1024-n1024: ×0.732 < ×1.050
 *   - tile64x64r8x4w16x8k16: i8a8-attention-pv-self-m1024-n1024: ×0.804 < ×1.050
 * - i8a8.conv3d (i8a8 conv3d (w8a8 implicit GEMM), 14 cases): adopted tile64x64r8x4w16x8k16 ×1.339 (×1.139–×2.472)
 *   - tile128x64r8x4w16x16k16: geometric mean ×1.211 (below the adopted ×1.339)
 *   - tile16x128r4x8w16x4k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.149 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.730 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-conv3d-cin1024-cout512-k333-in6x128x128: ×0.408 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-conv3d-cin256-cout256-k333-in6x256x256: ×0.951 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-conv3d-cin1024-cout1024-k333-in4x64x64: ×0.821 < ×1.050
 *   - tile64x64r4x4w16x16k16: geometric mean ×1.238 (below the adopted ×1.339)
 *   - tile64x64r4x8w8x16k16: i8a8-conv3d-cin512-cout256-k333-in6x256x256: ×1.002 < ×1.050
 *
 * Provenance converted on 2026-11-12 to one entry per op family (ADR 0117 addendum 2026-11-12), no field
 * value changed: both records have rows of every family, so every family's entry names both, and its
 * `kernels` / `caseSet` are the current runtime's keys over the family's cases.
 *
 * `i8a8.linear` (the row-count rules of ADR 0115 addendum decision 14) and the `i8a8-linear` entry come from
 * two `full` sweeps of that family alone (its 13 cases; Node + Dawn, timestamp quantisation off), derived on
 * 2026-11-13 with `deriveGeometryProfile` and `base` = this table: every other family's fields, verdicts and
 * provenance entries are this table's as they were (the records above). The RTX 3080 Ti heats within one
 * case of 230 candidates, so each sweep measured every case in chunks of 8 candidates (the `geometries`
 * option of `runGeometrySweep`, ADR 0117 addendum 2026-11-12), one process per call, the GPU cooled to
 * 56 °C before each; the per-case records of one chunk are merged into one record (the rule of the
 * Playground's merge), so the provenance names 29 records per sweep (`merged/rep1-chunkNN.json` and
 * `merged/rep2-chunkNN.json`). Three of the 754 calls were not kept (their default re-measurement stayed
 * outside ×1.05 after three tries), so in each of those the other sweep's rows stand alone. The scripts
 * that ran the sweep, merged the records, derived and wrote these parts are kept, untracked, in
 * `outputs/diag/k85-buckets-2026-11-12/`.
 */
import type { GeometryProfile } from "../geometry-profile.ts";

export const NVIDIA_AMPERE: GeometryProfile = {
  id: "nvidia-ampere",
  match: { vendor: "nvidia", architecture: "ampere" },
  gemmRows: [
    { maxRows: 16, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 32, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 64, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 128, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 16 } },
    { maxRows: 512, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: Number.POSITIVE_INFINITY, geometry: { regM: 8, regN: 8, wgX: 16, wgY: 16 } },
  ],
  attention: {
    qk: { regM: 8, regN: 8, wgX: 16, wgY: 16 },
    pv: { regM: 4, regN: 4, wgX: 16, wgY: 16 },
  },
  conv2d: {
    rows64: { regM: 8, regN: 8, wgX: 16, wgY: 8 },
    rows32: { regM: 8, regN: 8, wgX: 16, wgY: 4 },
  },
  i8a8: {
    linear: [
      { maxRows: 256, geometry: { regM: 8, regN: 4, wgX: 8, wgY: 4, tileK: 64, skeleton: "vec4" } },
      {
        maxRows: 512,
        geometry: { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 64, skeleton: "vec4" },
      },
      {
        maxRows: Number.POSITIVE_INFINITY,
        geometry: { regM: 16, regN: 4, wgX: 16, wgY: 8, tileK: 64, skeleton: "vec4" },
      },
    ],
    attentionQk: { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 },
    attentionPv: { regM: 8, regN: 8, wgX: 16, wgY: 8, tileK: 16 },
    conv3d: { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 },
  },
  provenance: {
    adapter: {
      vendor: "nvidia",
      architecture: "ampere",
      device: "nvidia-geforce-rtx-3080-ti",
      description: "NVIDIA: 615.78.08 615.78.8.0",
    },
    families: {
      gemm: {
        sweep:
          "outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run1-2026-10-26T19-31-37.851Z.json, outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run2-2026-10-26T20-20-58.449Z.json",
        sha256:
          "aa978a9021e1cc0aea4f123727e36a039cec161bb55d0c715d7e3e80189f763f, d265e379dc2d7ca5c7036a59d8f2d55b9351465937d743a59574a1ad642c65d8",
        date: "2026-10-26T19:31:37.851Z, 2026-10-26T20:20:58.449Z",
        candidateSet: "quick+, quick+",
        userAgent: ["Node.js/24"],
        kernels: "ca7e78e7f763ea6a",
        caseSet: "e6ddd45226ab2155",
      },
      attention: {
        sweep:
          "outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run1-2026-10-26T19-31-37.851Z.json, outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run2-2026-10-26T20-20-58.449Z.json",
        sha256:
          "aa978a9021e1cc0aea4f123727e36a039cec161bb55d0c715d7e3e80189f763f, d265e379dc2d7ca5c7036a59d8f2d55b9351465937d743a59574a1ad642c65d8",
        date: "2026-10-26T19:31:37.851Z, 2026-10-26T20:20:58.449Z",
        candidateSet: "quick+, quick+",
        userAgent: ["Node.js/24"],
        kernels: "427da13d86d8f3f7",
        caseSet: "9ee211cf69674928",
      },
      conv2d: {
        sweep:
          "outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run1-2026-10-26T19-31-37.851Z.json, outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run2-2026-10-26T20-20-58.449Z.json",
        sha256:
          "aa978a9021e1cc0aea4f123727e36a039cec161bb55d0c715d7e3e80189f763f, d265e379dc2d7ca5c7036a59d8f2d55b9351465937d743a59574a1ad642c65d8",
        date: "2026-10-26T19:31:37.851Z, 2026-10-26T20:20:58.449Z",
        candidateSet: "quick+, quick+",
        userAgent: ["Node.js/24"],
        kernels: "e6004bc50ba66635",
        caseSet: "02186e45558e4151",
      },
      "i8a8-linear": {
        sweep:
          "outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk00.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk01.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk02.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk03.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk04.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk05.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk06.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk07.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk08.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk09.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk10.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk11.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk12.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk13.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk14.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk15.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk16.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk17.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk18.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk19.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk20.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk21.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk22.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk23.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk24.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk25.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk26.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk27.json, outputs/diag/k85-buckets-2026-11-12/merged/rep1-chunk28.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk00.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk01.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk02.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk03.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk04.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk05.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk06.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk07.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk08.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk09.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk10.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk11.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk12.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk13.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk14.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk15.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk16.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk17.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk18.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk19.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk20.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk21.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk22.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk23.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk24.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk25.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk26.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk27.json, outputs/diag/k85-buckets-2026-11-12/merged/rep2-chunk28.json",
        sha256:
          "0329bb33aab749a4e0ce4dc31410c43a29e6555f3409c1e140d7fb1fc689ef23, d37547bee2af0de170337a5d3055fefc0c099ee84c61c11ff97b5e8589daf25f, 21c4d32d4c28e24dd6fe793c63502ce65e4ae63399f79a630e46e4ee09161ac9, 3e0f6e03cc01746ca21130630e428122713d3bb1a3210eef1bb3b859dbe672ba, 327dbd0d37467e919df7f9c67f29f084df54ac7896127f02e46d4d2100f90c94, c3dda389f5817bc33981345a2d145fb29d01f63d8430a0d1d2f0675645d7f9b9, bedd36845b17bb55b940af92f71c03e9ef25d58cf216c8fb93cc3215c6238f3e, 0957c9a30f8762878280cfd630375fc249ce1f59ef606db3a409e23b1967376d, d862d190d365fc8fe715350b39c1950c4943cbc37f6cb579c10d84462c3d67df, 4d35dbe7c5f9d520e67031fa388b9399d5b25ab78c5ed09280e75d0cee871f25, c8faa4655b378a019d4494877171b6f19cddea9f94681f3a2087e7ddeb516bb3, e01484781866705a1b9c4684f783d2135f83294a64f7b6c92ef0cd7d4b1e3b19, d7ebacbeaf55d4d71cd0e1d9a43769c313b9064699ead206963b7c8ff59a608f, 78c20112462dc9b88fbeba6e148a3e2a3cd8f483c55006966bc85a6dd3771532, d8427fdbcf5e3a2bfcd531ff5d25b20320b3c42ede1ec88daad2173c8596dc67, 2e4ecae8a55e1688bb65a6c6c8bfd4597d75eaad041b55091b1cd337b2070362, cc33762f678409cff2c49708e40f40886bbaacbd30c88d8c76d0d1abb34cd5ca, 7c53855ea4137ebf0ae8ecff3f8294da595da865719d170ab7ee10adfb5b29d8, 78bfc9ee01790bbb1ceec01e1a08883b28f1607ba4f85fa9685b59cfd6661318, 7cee21018aa72bfdd447d8be2a4484b0cc13da3877cdd8e198ba3ad75d3720d0, bd41470bee17e9b9b1baf318e94efddca2d0986a272c963063d57e8f974b7699, 63f5ad6727c53da5a74a46df8dabb35cba2fb084565e38cb8ef0e270f387b60f, 2187c03e311268ab7b0192206845d17ff8d95937ccd75292ff7318fb74283b2a, 8d6400cbeedd4bc3435c15df892a895cffe88c7a05799aaa1101e69854725b4a, 954f11d3882c0a28c7aa9e3e71c4f687b0f707d92cda050583c72b2fd61cc828, 63baef9fe699384280a4fb8445f66475005965aac5932af6b38e7f9f46784eee, 89ae2d654e4a8626404d587802b7580e27376eb456a56ff9493fbb865057fd10, b3bd81a935800ef67cf0b43a6b4247c6d392e48bf68894a5b5cef9499b21ed51, 7b842f9c18d5e0958a1a430305339c7c2e3fd5775fbac986affd0d051e6577b9, 6f336d35408a8cf96e1826c9eac70d9b9d8ce0a674e083d1ac6f61798c764c39, 152363090860f00f4c1caf3f1e2216f8f0380c5731d26d4824ad12de19343953, 76fd26809b5aaa5481185f95cdfebd087f255e940f6e515986324d6189bcb2a2, 53d4d8aa77864d7dbb7ef74013806a51242d694cdd06a70062fbe82f0e262a2a, 59dc2fcc29408f5b8ced409700ab52eca170b0119426e1f9724b37908b8d44af, 50e40590ec838283e6ff44ef73c124d20fd34340bfd44c12a339a12d0ab749b5, 0f183b2b5e6b34e70e78e3dc4f2d44f19f9fa24f80d9c6718eff17c69d5faa8b, 4a54ea752118e0dfcba6f7e7ff0169466e692b1195e2a6299d0e09ece94a13e7, bc03ce41e9a4fb28eedddcb4ed9991049fcd7e3c9ddb5b9f1cf970d1d5b4ef11, d0a72b63f0e95c6319e7ee43ccf6d0fe43084e770ae08b24195211f5a0852996, 0ffec9fecaca5edb40a4d56739fac289308f4b9dc110c17ecc367dc4cf270ea9, 0cae1e5e2b32b5c1f86d63487312c1e0677d0d78ffccb376fd36f728d8c8e7fc, 7699d6c2172b26354f3ec069dcdc9f07fde49c3b879866b2cf037e6578bfd18a, 2a28702a04faecafb79386307f0f585d3c2c474e6f45bc9bb0c090d044edc5de, ea69d254b5f5aa794cc4ebcac012022e5ee9acb7357dc535286c9d50fb741853, 1f9d6b3708cce7017a20c8faff39e4a31fe30cfa0f5ce9a50bc50e4d3081fc08, 89f24c9980cf4eeacb3a301d3651f99fe6a52d4429931984de1b336bb01f185f, 1a535a4880f5f922fe17e939684da5132febc2964140fad9947559e8366a40a5, 9e758cabb23aa1b809c2791dec4e290a67391a57ccfb5198606767b029fc4370, 51677686cf19906a5e02612c3b7d463554b49cf6b53ad0c58288ef22e65f5772, 74c68e0ce7e84b9d8600ba4add66610e47825b4e7260355c972c9ce9d78c3b9c, 378d29fad7a6e6ad87c7694b5d008f9edfc64a3c5b164c267ead42af08f3360d, 08e93a743c7f28a7962582081a532d1dc65d037a95aee963f469fc0a5acc93e3, b186145fe83b72e39ca5e7043f45239fd39b6195f30d881db7b8cb26ff672c41, fc31d7f840c3ff4cc601e69d60ef3bc855c8d5f4a67254ff6beefc6b7be15797, f0c0d37c7094ddb8c569c7bd4e7d41f0449710c756123cb55e20e926f4337eea, cbec8948b28d7922eaff803dc94c0318be72a52cae57597e2c72395464860df3, 180f4a42f6561d0df0ba63a14548af306160af5066599f13dec067580bd78356, a73c8067767a11233e756b095140099f175d19aa37824d5f9ccf65791a1417b1",
        date: "2026-11-12T19:07:45.413Z, 2026-11-12T19:20:16.703Z, 2026-11-12T19:35:33.963Z, 2026-11-12T19:48:16.473Z, 2026-11-12T20:00:36.426Z, 2026-11-12T20:12:57.272Z, 2026-11-12T20:27:46.518Z, 2026-11-12T20:40:22.082Z, 2026-11-12T20:51:50.705Z, 2026-11-12T21:04:50.877Z, 2026-11-12T21:18:30.344Z, 2026-11-12T21:29:43.985Z, 2026-11-12T21:41:35.965Z, 2026-11-12T21:54:25.953Z, 2026-11-12T22:06:13.631Z, 2026-11-12T22:17:12.106Z, 2026-11-12T22:28:09.315Z, 2026-11-12T22:41:30.022Z, 2026-11-12T22:53:05.183Z, 2026-11-12T23:05:59.394Z, 2026-11-12T23:21:02.324Z, 2026-11-12T23:32:39.244Z, 2026-11-12T23:45:54.276Z, 2026-11-12T23:57:26.676Z, 2026-11-13T00:09:59.560Z, 2026-11-13T00:22:12.553Z, 2026-11-13T00:35:35.441Z, 2026-11-13T00:49:26.776Z, 2026-11-13T00:59:38.687Z, 2026-11-13T01:10:44.869Z, 2026-11-13T01:23:12.386Z, 2026-11-13T01:36:36.635Z, 2026-11-13T01:47:36.702Z, 2026-11-13T01:59:50.752Z, 2026-11-13T02:11:17.364Z, 2026-11-13T02:21:54.699Z, 2026-11-13T02:32:39.393Z, 2026-11-13T02:43:08.546Z, 2026-11-13T02:54:15.361Z, 2026-11-13T03:05:31.815Z, 2026-11-13T03:17:55.116Z, 2026-11-13T03:30:26.736Z, 2026-11-13T03:43:21.147Z, 2026-11-13T03:56:13.458Z, 2026-11-13T04:09:14.912Z, 2026-11-13T04:21:24.188Z, 2026-11-13T04:33:27.546Z, 2026-11-13T04:46:57.493Z, 2026-11-13T04:58:25.466Z, 2026-11-13T05:11:54.581Z, 2026-11-13T05:25:18.915Z, 2026-11-13T05:38:55.566Z, 2026-11-13T05:51:50.755Z, 2026-11-13T06:03:39.432Z, 2026-11-13T06:15:29.737Z, 2026-11-13T06:27:12.039Z, 2026-11-13T06:39:08.540Z, 2026-11-13T06:48:24.510Z",
        candidateSet:
          "full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full, full",
        userAgent: ["Node.js/24"],
        kernels: "67d915bfeae0cc15",
        caseSet: "3d32dc6f10559af2",
      },
      "i8a8-attention": {
        sweep:
          "outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run1-2026-10-26T19-31-37.851Z.json, outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run2-2026-10-26T20-20-58.449Z.json",
        sha256:
          "aa978a9021e1cc0aea4f123727e36a039cec161bb55d0c715d7e3e80189f763f, d265e379dc2d7ca5c7036a59d8f2d55b9351465937d743a59574a1ad642c65d8",
        date: "2026-10-26T19:31:37.851Z, 2026-10-26T20:20:58.449Z",
        candidateSet: "quick+, quick+",
        userAgent: ["Node.js/24"],
        kernels: "d5b184a653a1b381",
        caseSet: "8ab1a86b13087640",
      },
      "i8a8-conv3d": {
        sweep:
          "outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run1-2026-10-26T19-31-37.851Z.json, outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run2-2026-10-26T20-20-58.449Z.json",
        sha256:
          "aa978a9021e1cc0aea4f123727e36a039cec161bb55d0c715d7e3e80189f763f, d265e379dc2d7ca5c7036a59d8f2d55b9351465937d743a59574a1ad642c65d8",
        date: "2026-10-26T19:31:37.851Z, 2026-10-26T20:20:58.449Z",
        candidateSet: "quick+, quick+",
        userAgent: ["Node.js/24"],
        kernels: "c91b3368411e71d9",
        caseSet: "27ce9da45b359b14",
      },
    },
  },
};
