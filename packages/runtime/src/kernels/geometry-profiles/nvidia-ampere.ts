/**
 * Geometry profile `nvidia-ampere` (**generated — do not edit by hand**).
 *
 * A static table of tile geometries for adapter `nvidia / ampere`, derived from the
 * sweep record below with `deriveGeometryProfile` of the public surface `@karume/runtime/tune` (ADR 0117
 * decision 2 — the CLI and the GPU lab that used to write these files were removed in 0.15; the short Node
 * scripts that ran the sweep and wrote this file are kept, untracked, in
 * `.claude/reviews/2026-10-11_wan-attention-recon/`). The runtime only selects this table by the adapter's
 * `match` (exact match on vendor / architecture). Nothing is measured at run time (no autotuning — ADR 0022
 * decision 3). To change a value, take the sweep again and regenerate the table.
 *
 * Sweep (adapter nvidia / ampere / nvidia-geforce-rtx-3080-ti / NVIDIA: 615.78.08 615.78.8.0; Node + Dawn, Vulkan):
 *
 * - outputs/bench/geometry-sweep-node-percase-2026-10-23T14-19-39.869Z.json (sha256 f0ee26d405f34146e0c10b758caf1c41fcf72484c19decacf566e23b02205490, 2026-10-23T14:19:39.869Z)
 *
 * Adoption criteria (the rules of `deriveProfile` — ADR 0115 decision 4, ADR 0116, ADR 0117 decision 3): among
 * the geometries whose output matches the default on every case of the class and whose ratio to the default is
 * ×1.050 or more, the one with the largest geometric mean across the cases; if there is none, the default. A case
 * whose default re-measurement ratio is outside 0.9–1.1, or whose re-measurement failed, is dropped from the ratio
 * material (output match and failures are still checked). Timestamps are GPU timestamp queries in ns (Dawn, the
 * 65,536 ns quantisation disabled), so no rounding-error bound applies.
 *
 * Adoption:
 *
 * - gemmRows ≤ 16 (linear / matmul / bmm row count ≤ 16, 4 cases): keeps the default reg16x16r1x4w4 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: bmm-b16-m16-n64-k128: ×0.183 < ×1.050
 *   - reg128x32r8x4w8: matmul-m16-n3072-k1024: ×0.219 < ×1.050
 *   - reg32x32r2x4w8: linear-m16-n3072-k1024: ×0.735 < ×1.050
 *   - reg32x32r4x4w8: linear-m16-n3072-k1024: ×0.594 < ×1.050
 *   - reg64x32r4x4w8: matmul-m16-n3072-k1024: ×0.605 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m16-n64-k128: ×0.521 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m16-n64-k128: ×0.395 < ×1.050
 *   - reg64x64r8x4w16: matmul-m16-n3072-k1024: ×0.317 < ×1.050
 * - gemmRows 17〜32 (linear / matmul / bmm row count 17〜32, 4 cases): keeps the default reg16x16r1x4w4 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: bmm-b16-m32-n64-k128: ×0.197 < ×1.050
 *   - reg128x32r8x4w8: matmul-m32-n3072-k1024: ×0.247 < ×1.050
 *   - reg32x32r2x4w8: bmm-b16-m32-n64-k128: ×0.823 < ×1.050
 *   - reg32x32r4x4w8: linear-m32-n3072-k1024: ×0.736 < ×1.050
 *   - reg64x32r4x4w8: bmm-b16-m32-n64-k128: ×0.674 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m32-n64-k128: ×0.553 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m32-n64-k128: ×0.419 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m32-n64-k128: ×0.346 < ×1.050
 * - gemmRows 33〜64 (linear / matmul / bmm row count 33〜64, 4 cases): keeps the default reg16x16r1x4w4 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: bmm-b16-m64-n64-k128: ×0.248 < ×1.050
 *   - reg128x32r8x4w8: bmm-b16-m64-n64-k128: ×0.319 < ×1.050
 *   - reg32x32r2x4w8: bmm-b16-m64-n64-k128: ×1.040 < ×1.050
 *   - reg32x32r4x4w8: bmm-b16-m64-n64-k128: ×0.967 < ×1.050
 *   - reg64x32r4x4w8: bmm-b16-m64-n64-k128: ×0.836 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m64-n64-k128: ×0.682 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m64-n64-k128: ×0.515 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m64-n64-k128: ×0.443 < ×1.050
 * - gemmRows 65〜128 (linear / matmul / bmm row count 65〜128, 4 cases): keeps the default reg64x32r4x4w8 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: bmm-b16-m128-n64-k128: ×0.292 < ×1.050
 *   - reg128x32r8x4w8: bmm-b16-m128-n64-k128: ×0.387 < ×1.050
 *   - reg16x16r1x4w4: linear-m128-n2048-k1024: ×0.507 < ×1.050
 *   - reg32x32r2x4w8: linear-m128-n2048-k1024: ×0.751 < ×1.050
 *   - reg32x32r4x4w8: linear-m128-n2048-k1024: ×0.857 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m128-n64-k128: ×0.834 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m128-n64-k128: ×0.630 < ×1.050
 *   - reg64x64r8x4w16: matmul-m128-n2048-k1024: ×0.458 < ×1.050
 * - gemmRows 129〜256 (linear / matmul / bmm row count 129〜256, 4 cases): adopted reg64x64r4x4w16 ×1.118 (×1.109–×1.125)
 *   - reg128x128r8x8w16: bmm-b16-m256-n64-k128: ×0.414 < ×1.050
 *   - reg128x32r8x4w8: matmul-m256-n2048-k1024: ×0.535 < ×1.050
 *   - reg16x16r1x4w4: linear-m256-n2048-k1024: ×0.434 < ×1.050
 *   - reg32x32r2x4w8: linear-m256-n2048-k1024: ×0.720 < ×1.050
 *   - reg32x32r4x4w8: linear-m256-n2048-k1024: ×0.808 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m256-n64-k64: ×0.820 < ×1.050
 *   - reg64x64r8x4w16: matmul-m256-n2048-k1024: ×0.678 < ×1.050
 * - gemmRows 257〜512 (linear / matmul / bmm row count 257〜512, 5 cases): keeps the default reg64x32r4x4w8 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x128r8x8w16: bmm-b16-m512-n64-k64: ×0.519 < ×1.050
 *   - reg128x32r8x4w8: bmm-b16-m512-n64-k512: ×0.525 < ×1.050
 *   - reg16x16r1x4w4: linear-m512-n2048-k1024: ×0.400 < ×1.050
 *   - reg32x32r2x4w8: linear-m512-n2048-k1024: ×0.708 < ×1.050
 *   - reg32x32r4x4w8: linear-m512-n2048-k1024: ×0.809 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m512-n64-k64: ×1.029 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m512-n64-k64: ×0.681 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m512-n64-k512: ×0.774 < ×1.050
 * - gemmRows > 512 (linear / matmul / bmm row count > 512, 7 cases): keeps the default reg128x128r8x8w16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x32r8x4w8: matmul-m4096-n2048-k2048: ×0.524 < ×1.050
 *   - reg16x16r1x4w4: linear-m4096-n8192-k2048: ×0.269 < ×1.050
 *   - reg32x32r2x4w8: linear-m4096-n8192-k2048: ×0.484 < ×1.050
 *   - reg32x32r4x4w8: linear-m4096-n8192-k2048: ×0.570 < ×1.050
 *   - reg64x32r4x4w8: linear-m4096-n8192-k2048: ×0.748 < ×1.050
 *   - reg64x64r4x4w16: linear-m4096-n8192-k2048: ×0.860 < ×1.050
 *   - reg64x64r4x8w8: linear-m4096-n8192-k2048: ×0.720 < ×1.050
 *   - reg64x64r8x4w16: linear-m4096-n8192-k2048: ×0.720 < ×1.050
 * - attention.qk (fused attention f32 ①QK, 4 cases): keeps the default reg128x128r8x8w16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - reg128x32r8x4w8: attention-qk-self-m4096-n4096: ×0.572 < ×1.050
 *   - reg16x16r1x4w4: attention-qk-self-m4096-n4096: ×0.296 < ×1.050
 *   - reg32x32r2x4w8: attention-qk-self-m4096-n4096: ×0.526 < ×1.050
 *   - reg32x32r4x4w8: attention-qk-self-m4096-n4096: ×0.629 < ×1.050
 *   - reg64x32r4x4w8: attention-qk-self-m4096-n4096: ×0.809 < ×1.050
 *   - reg64x64r4x4w16: attention-qk-self-m4096-n4096: ×0.945 < ×1.050
 *   - reg64x64r4x8w8: attention-qk-self-m4096-n4096: ×0.755 < ×1.050
 *   - reg64x64r8x4w16: attention-qk-self-m4096-n4096: ×0.775 < ×1.050
 * - attention.pv (fused attention f32 ③PV, 4 cases): adopted reg64x64r4x4w16 ×1.095 (×1.072–×1.112)
 *   - reg128x32r8x4w8: attention-pv-self-m4096-n4096: ×0.543 < ×1.050
 *   - reg16x16r1x4w4: attention-pv-self-m4096-n4096: ×0.384 < ×1.050
 *   - reg32x32r2x4w8: attention-pv-self-m4096-n4096: ×0.666 < ×1.050
 *   - reg32x32r4x4w8: attention-pv-self-m4096-n4096: ×0.841 < ×1.050
 *   - reg64x32r4x4w8: attention-pv-self-m4096-n4096: ×0.879 < ×1.050
 *   - reg64x64r4x8w8: attention-pv-self-m4096-n4096: ×0.869 < ×1.050
 *   - reg64x64r8x4w16: attention-pv-self-m4096-n4096: ×0.860 < ×1.050
 * - conv2d.rows64 (conv2d implicit GEMM m tile of 64 rows, 2 cases): keeps the default igemm64x128:wg16x8 (no geometry matches the output in every case at ×1.050 or faster)
 *   - igemm128x128:wg16x16: conv2d-c192-256x256: ×0.855 < ×1.050
 *   - igemm128x64:wg16x16: conv2d-c192-256x256: ×0.718 < ×1.050
 *   - igemm32x128:wg16x4: conv2d-c384-128x128: ×0.760 < ×1.050
 *   - igemm64x64:wg16x16: conv2d-c384-128x128: ×1.009 < ×1.050
 *   - igemm64x64:wg16x8: conv2d-c384-128x128: ×0.891 < ×1.050
 *   - igemm64x64:wg8x16: conv2d-c384-128x128: ×0.892 < ×1.050
 * - conv2d.rows32 (conv2d implicit GEMM m tile of 32 rows, 1 cases): keeps the default igemm32x128:wg16x4 (no geometry matches the output in every case at ×1.050 or faster)
 *   - igemm128x128:wg16x16: conv2d-c96-512x512: ×1.033 < ×1.050
 *   - igemm128x64:wg16x16: conv2d-c96-512x512: ×0.867 < ×1.050
 *   - igemm64x128:wg16x8: conv2d-c96-512x512: ×0.941 < ×1.050
 *   - igemm64x64:wg16x16: conv2d-c96-512x512: ×0.985 < ×1.050
 *   - igemm64x64:wg16x8: conv2d-c96-512x512: ×0.829 < ×1.050
 *   - igemm64x64:wg8x16: conv2d-c96-512x512: ×0.824 < ×1.050
 * - i8a8.linear (i8a8 linear, 6 cases): adopted tile64x64r8x4w16x8k16 ×1.160 (×1.073–×1.247)
 *   - tile128x64r8x4w16x16k16: geometric mean ×1.141 (below the adopted ×1.160)
 *   - tile16x128r4x8w16x4k16: i8a8-linear-m4096-n8192-k2048: ×0.184 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-linear-m4096-n8192-k2048: ×0.556 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-linear-m4096-n8192-k2048: ×0.407 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m4096-n8192-k2048: ×0.968 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-linear-m4096-n8192-k2048: ×0.829 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-linear-m4096-n8192-k2048: ×0.965 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-linear-m4096-n8192-k2048: ×0.914 < ×1.050
 * - i8a8.attentionQk (i8a8 attention ①QK, 4 cases): keeps the default tile128x64r8x8w8x16k16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - Sweep outputs/bench/geometry-sweep-node-percase-2026-10-23T14-19-39.869Z.json: i8a8-attention-qk-self-m1024-n1024 dropped from the ratio material due to default re-measurement ratio ×1.115 out of range (output matches and failures are still checked)
 *   - tile128x64r8x4w16x16k16: not measured in i8a8-attention-qk-self-m1024-n1024 (dropped from the ratio material in every sweep)
 *   - tile16x128r4x8w16x4k16: not measured in i8a8-attention-qk-self-m1024-n1024 (dropped from the ratio material in every sweep)
 *   - tile32x32r4x4w8x8k16: not measured in i8a8-attention-qk-self-m1024-n1024 (dropped from the ratio material in every sweep)
 *   - tile32x64r4x8w8x8k16: not measured in i8a8-attention-qk-self-m1024-n1024 (dropped from the ratio material in every sweep)
 *   - tile64x128r8x8w16x8k16: not measured in i8a8-attention-qk-self-m1024-n1024 (dropped from the ratio material in every sweep)
 *   - tile64x32r4x4w8x16k16: not measured in i8a8-attention-qk-self-m1024-n1024 (dropped from the ratio material in every sweep)
 *   - tile64x64r4x4w16x16k16: not measured in i8a8-attention-qk-self-m1024-n1024 (dropped from the ratio material in every sweep)
 *   - tile64x64r4x8w8x16k16: not measured in i8a8-attention-qk-self-m1024-n1024 (dropped from the ratio material in every sweep)
 *   - tile64x64r8x4w16x8k16: not measured in i8a8-attention-qk-self-m1024-n1024 (dropped from the ratio material in every sweep)
 * - i8a8.attentionPv (i8a8 attention ③PV, 4 cases): keeps the default tile64x128r8x8w16x8k16 (no geometry matches the output in every case at ×1.050 or faster)
 *   - tile128x64r8x4w16x16k16: i8a8-attention-pv-self-m1024-n1024: ×0.762 < ×1.050
 *   - tile128x64r8x8w8x16k16: i8a8-attention-pv-cross-m4096-n512: ×0.793 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-attention-pv-self-m1024-n1024: ×0.772 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-attention-pv-self-m1024-n1024: ×0.798 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-attention-pv-self-m1024-n1024: ×0.775 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-attention-pv-self-m1024-n1024: ×0.795 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-attention-pv-self-m1024-n1024: ×0.944 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-attention-pv-self-m1024-n1024: ×0.727 < ×1.050
 *   - tile64x64r8x4w16x8k16: i8a8-attention-pv-self-m1024-n1024: ×0.802 < ×1.050
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
    attentionQk: { regM: 8, regN: 8, wgX: 8, wgY: 16, tileK: 16 },
    attentionPv: { regM: 8, regN: 8, wgX: 16, wgY: 8, tileK: 16 },
  },
  provenance: {
    sweep: "outputs/bench/geometry-sweep-node-percase-2026-10-23T14-19-39.869Z.json",
    sha256: "f0ee26d405f34146e0c10b758caf1c41fcf72484c19decacf566e23b02205490",
    date: "2026-10-23T14:19:39.869Z",
    candidateSet: "quick+",
    userAgent: ["Node.js/24"],
    adapter: {
      vendor: "nvidia",
      architecture: "ampere",
      device: "nvidia-geforce-rtx-3080-ti",
      description: "NVIDIA: 615.78.08 615.78.8.0",
    },
    kernels: "578f92d98518569a",
    caseSet: "c1060d8829f3b9de",
  },
};
