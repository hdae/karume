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
 * Sweeps (adapter nvidia / ampere / nvidia-geforce-rtx-3080-ti / NVIDIA: 615.78.08 615.78.8.0; Node + Dawn, Vulkan;
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
 * - i8a8.linear (i8a8 linear, 6 cases): adopted tile64x64r8x4w16x8k16 ×1.170 (×1.080–×1.290)
 *   - tile128x64r8x4w16x16k16: geometric mean ×1.148 (below the adopted ×1.170)
 *   - tile16x128r4x8w16x4k16: i8a8-linear-m4096-n8192-k2048: ×0.187 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-linear-m4096-n8192-k2048: ×0.568 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-linear-m4096-n8192-k2048: ×0.417 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m4096-n8192-k2048: ×0.977 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-linear-m4096-n2048-k8192: ×0.843 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-linear-m4096-n8192-k2048: ×0.980 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-linear-m4096-n8192-k2048: ×0.931 < ×1.050
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
    linear: { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 },
    attentionQk: { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 },
    attentionPv: { regM: 8, regN: 8, wgX: 16, wgY: 8, tileK: 16 },
    conv3d: { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 },
  },
  provenance: {
    sweep:
      "outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run1-2026-10-26T19-31-37.851Z.json, outputs/diag/wave-b-2026-10-26/v5/geometry-sweep-node-percase-run2-2026-10-26T20-20-58.449Z.json",
    sha256:
      "aa978a9021e1cc0aea4f123727e36a039cec161bb55d0c715d7e3e80189f763f, d265e379dc2d7ca5c7036a59d8f2d55b9351465937d743a59574a1ad642c65d8",
    date: "2026-10-26T19:31:37.851Z, 2026-10-26T20:20:58.449Z",
    candidateSet: "quick+, quick+",
    userAgent: ["Node.js/24"],
    adapter: {
      vendor: "nvidia",
      architecture: "ampere",
      device: "nvidia-geforce-rtx-3080-ti",
      description: "NVIDIA: 615.78.08 615.78.8.0",
    },
    kernels: "c1b6cb7ea2ede6d2",
    caseSet: "0c9e4c7b7202eb2a",
  },
};
