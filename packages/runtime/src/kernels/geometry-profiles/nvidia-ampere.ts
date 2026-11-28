/**
 * Geometry profile `nvidia-ampere` (**generated — do not edit by hand**).
 *
 * A static table of tile geometries for the adapter match: vendor nvidia, architecture ampere.
 * The runtime only selects the table by the adapter's match; nothing is measured at run time.
 * To change a value, take a sweep with the workbench's `geometry-sweep` entry and regenerate the
 * table with `node tools/geometry-profiles/generate.ts nvidia-ampere --records=<path>[,<path>…]`.
 * The GEMV choices of linear (`linearGemv`) come from records of the `gemv-sweep` entry
 * (`--gemv-records=<path>,<path>[,…]`).
 * The grid-fill rule of the GEMM (`gemmGrid`) comes from records of the `geometry-sweep` entry's
 * `grid` set (`--grid-records=<path>[,<path>…]`); where the table carries it, it replaces the
 * row-count buckets for linear / matmul / bmm.
 *
 * Derived in this generation: the grid-fill rule of the GEMM (grid records).
 *
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (sha256 bd4a705dc7667f6072b0489e860c5ff7ecc78d810aa734f5db4c90de46f3e208, 2026-11-28T05:13:00.523Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 98866c56e9326c5a575871f9477828633d1817f0
 * - outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json (sha256 d5955ac66b40174e5920a70315ec68332d84a5a189f6ede12ecb0757cd21eaf1, 2026-11-28T05:21:20.825Z)
 *   workbench; Node; backend vulkan (dawn-default); timestamps not quantised; checkout 98866c56e9326c5a575871f9477828633d1817f0 (dirty)
 *
 * Carried over from the registered table at commit 8c9364d8481f83e45f18cc29d813ffd3350d1841: gemm, attention, conv2d, i8a8-linear, i8a8-attention, i8a8-conv3d, the GEMV choices of linear.
 * Their adoption lines are in `git show 8c9364d8481f83e45f18cc29d813ffd3350d1841:packages/runtime/src/kernels/geometry-profiles/nvidia-ampere.ts`.
 *
 * Grid-fill rule: the ladder (largest first)
 * reg128x128r8x8w16, reg64x64r4x4w16, reg64x32r4x4w8, reg32x32r2x4w8, reg16x16r1x4w4,
 * the K floor 256 and the padding bound 1.5 are the code's; the two integers are
 * the pair, of a fixed grid, that qualifies with the lowest census-weighted GEMM time over the
 * grid set's cases. A pair qualifies when, on every case, the tile it picks wrote the default's
 * words in every record that timed it, is timed against the table's tile for the case (by a
 * steady record alone, else beside a second record: the GEMV rule's class of a record) and is not
 * slower than it in any such reading beyond 1 ÷ (×1.065 × the two rows' gaps), and when the
 * weighted time is the table's ÷ ×1.050 or less. No pair qualifying, the table carries no rule.
 *
 * - gemmGrid (the grid-fill rule of the GEMM; 33 cases; 4 of 35 pairs qualify; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json unsteady (the default spread beyond ×1.05 in 9 of 33 cases), evidence beside a second record; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json unsteady (the default spread beyond ×1.05 in 8 of 33 cases), evidence beside a second record): adopted (256, 48) — weighted GEMM time ×0.926 of today's (gain ×1.080 ≥ ×1.050)
 *   - (128, 32): refused, weighted time ×0.931 — grid-linear-m64-n512-k512-wbf16: reg32x32r2x4w8 is slower than today's reg16x16r1x4w4 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.839, below 1 ÷ (×1.065 × gaps ×1.000 × ×1.004) = ×0.935)
 *   - (128, 48): qualifies, weighted time ×0.929
 *   - (128, 64): refused, weighted time ×0.929 — grid-bmm-b12-m196-n64-k196: reg64x32r4x4w8 is slower than today's reg64x64r4x4w16 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json (×0.882, below 1 ÷ (×1.065 × gaps ×1.000 × ×1.001) = ×0.938)
 *   - (128, 96): refused, weighted time ×0.929 — grid-linear-m128-n1024-k1024-wi4g32: reg32x32r2x4w8 is slower than today's reg64x32r4x4w8 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.901, below 1 ÷ (×1.065 × gaps ×1.002 × ×1.000) = ×0.937)
 *   - (128, 128): refused, weighted time ×0.929 — grid-linear-m128-n1024-k1024-wi4g32: reg32x32r2x4w8 is slower than today's reg64x32r4x4w8 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.901, below 1 ÷ (×1.065 × gaps ×1.002 × ×1.000) = ×0.937)
 *   - (192, 32): refused, weighted time ×0.931 — grid-linear-m64-n512-k512-wbf16: reg32x32r2x4w8 is slower than today's reg16x16r1x4w4 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.839, below 1 ÷ (×1.065 × gaps ×1.000 × ×1.004) = ×0.935)
 *   - (192, 48): qualifies, weighted time ×0.929
 *   - (192, 64): refused, weighted time ×0.929 — grid-bmm-b12-m196-n64-k196: reg64x32r4x4w8 is slower than today's reg64x64r4x4w16 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json (×0.882, below 1 ÷ (×1.065 × gaps ×1.000 × ×1.001) = ×0.938)
 *   - (192, 96): refused, weighted time ×0.929 — grid-linear-m128-n1024-k1024-wi4g32: reg32x32r2x4w8 is slower than today's reg64x32r4x4w8 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.901, below 1 ÷ (×1.065 × gaps ×1.002 × ×1.000) = ×0.937)
 *   - (192, 128): refused, weighted time ×0.929 — grid-linear-m128-n1024-k1024-wi4g32: reg32x32r2x4w8 is slower than today's reg64x32r4x4w8 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.901, below 1 ÷ (×1.065 × gaps ×1.002 × ×1.000) = ×0.937)
 *   - (256, 32): refused, weighted time ×0.928 — grid-linear-m64-n512-k512-wbf16: reg32x32r2x4w8 is slower than today's reg16x16r1x4w4 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.839, below 1 ÷ (×1.065 × gaps ×1.000 × ×1.004) = ×0.935)
 *   - (256, 48): qualifies, weighted time ×0.926
 *   - (256, 64): refused, weighted time ×0.926 — grid-bmm-b12-m196-n64-k196: reg64x32r4x4w8 is slower than today's reg64x64r4x4w16 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json (×0.882, below 1 ÷ (×1.065 × gaps ×1.000 × ×1.001) = ×0.938)
 *   - (256, 96): refused, weighted time ×0.927 — grid-linear-m128-n1024-k1024-wi4g32: reg32x32r2x4w8 is slower than today's reg64x32r4x4w8 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.901, below 1 ÷ (×1.065 × gaps ×1.002 × ×1.000) = ×0.937)
 *   - (256, 128): refused, weighted time ×0.926 — grid-linear-m128-n1024-k1024-wi4g32: reg32x32r2x4w8 is slower than today's reg64x32r4x4w8 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.901, below 1 ÷ (×1.065 × gaps ×1.002 × ×1.000) = ×0.937)
 *   - (288, 32): refused, weighted time ×0.928 — grid-linear-m64-n512-k512-wbf16: reg32x32r2x4w8 is slower than today's reg16x16r1x4w4 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.839, below 1 ÷ (×1.065 × gaps ×1.000 × ×1.004) = ×0.935)
 *   - (288, 48): qualifies, weighted time ×0.926
 *   - (288, 64): refused, weighted time ×0.926 — grid-bmm-b12-m196-n64-k196: reg64x32r4x4w8 is slower than today's reg64x64r4x4w16 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json (×0.882, below 1 ÷ (×1.065 × gaps ×1.000 × ×1.001) = ×0.938)
 *   - (288, 96): refused, weighted time ×0.927 — grid-linear-m128-n1024-k1024-wi4g32: reg32x32r2x4w8 is slower than today's reg64x32r4x4w8 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.901, below 1 ÷ (×1.065 × gaps ×1.002 × ×1.000) = ×0.937)
 *   - (288, 128): refused, weighted time ×0.926 — grid-linear-m128-n1024-k1024-wi4g32: reg32x32r2x4w8 is slower than today's reg64x32r4x4w8 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.901, below 1 ÷ (×1.065 × gaps ×1.002 × ×1.000) = ×0.937)
 *   - (320, 32): refused, weighted time ×0.931 — grid-linear-m64-n512-k512-wbf16: reg32x32r2x4w8 is slower than today's reg16x16r1x4w4 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.839, below 1 ÷ (×1.065 × gaps ×1.000 × ×1.004) = ×0.935)
 *   - (320, 48): refused, weighted time ×0.929 — grid-linear-m768-n6144-k1536-wbf16: reg64x64r4x4w16 is slower than today's reg128x128r8x8w16 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.912, below 1 ÷ (×1.065 × gaps ×1.003 × ×1.018) = ×0.920)
 *   - (320, 64): refused, weighted time ×0.929 — grid-linear-m768-n6144-k1536-wbf16: reg64x64r4x4w16 is slower than today's reg128x128r8x8w16 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.912, below 1 ÷ (×1.065 × gaps ×1.003 × ×1.018) = ×0.920)
 *   - (320, 96): refused, weighted time ×0.930 — grid-linear-m768-n6144-k1536-wbf16: reg64x64r4x4w16 is slower than today's reg128x128r8x8w16 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.912, below 1 ÷ (×1.065 × gaps ×1.003 × ×1.018) = ×0.920)
 *   - (320, 128): refused, weighted time ×0.929 — grid-linear-m768-n6144-k1536-wbf16: reg64x64r4x4w16 is slower than today's reg128x128r8x8w16 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.912, below 1 ÷ (×1.065 × gaps ×1.003 × ×1.018) = ×0.920)
 *   - (384, 32): refused, weighted time ×0.931 — grid-linear-m64-n512-k512-wbf16: reg32x32r2x4w8 is slower than today's reg16x16r1x4w4 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.839, below 1 ÷ (×1.065 × gaps ×1.000 × ×1.004) = ×0.935)
 *   - (384, 48): refused, weighted time ×0.929 — grid-linear-m768-n6144-k1536-wbf16: reg64x64r4x4w16 is slower than today's reg128x128r8x8w16 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.912, below 1 ÷ (×1.065 × gaps ×1.003 × ×1.018) = ×0.920)
 *   - (384, 64): refused, weighted time ×0.929 — grid-linear-m768-n6144-k1536-wbf16: reg64x64r4x4w16 is slower than today's reg128x128r8x8w16 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.912, below 1 ÷ (×1.065 × gaps ×1.003 × ×1.018) = ×0.920)
 *   - (384, 96): refused, weighted time ×0.930 — grid-linear-m768-n6144-k1536-wbf16: reg64x64r4x4w16 is slower than today's reg128x128r8x8w16 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.912, below 1 ÷ (×1.065 × gaps ×1.003 × ×1.018) = ×0.920)
 *   - (384, 128): refused, weighted time ×0.929 — grid-linear-m768-n6144-k1536-wbf16: reg64x64r4x4w16 is slower than today's reg128x128r8x8w16 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.912, below 1 ÷ (×1.065 × gaps ×1.003 × ×1.018) = ×0.920)
 *   - (480, 32): refused, weighted time ×0.931 — grid-linear-m64-n512-k512-wbf16: reg32x32r2x4w8 is slower than today's reg16x16r1x4w4 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.839, below 1 ÷ (×1.065 × gaps ×1.000 × ×1.004) = ×0.935)
 *   - (480, 48): refused, weighted time ×0.929 — grid-linear-m768-n6144-k1536-wbf16: reg64x64r4x4w16 is slower than today's reg128x128r8x8w16 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.912, below 1 ÷ (×1.065 × gaps ×1.003 × ×1.018) = ×0.920)
 *   - (480, 64): refused, weighted time ×0.929 — grid-linear-m768-n6144-k1536-wbf16: reg64x64r4x4w16 is slower than today's reg128x128r8x8w16 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.912, below 1 ÷ (×1.065 × gaps ×1.003 × ×1.018) = ×0.920)
 *   - (480, 96): refused, weighted time ×0.930 — grid-linear-m768-n6144-k1536-wbf16: reg64x64r4x4w16 is slower than today's reg128x128r8x8w16 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.912, below 1 ÷ (×1.065 × gaps ×1.003 × ×1.018) = ×0.920)
 *   - (480, 128): refused, weighted time ×0.929 — grid-linear-m768-n6144-k1536-wbf16: reg64x64r4x4w16 is slower than today's reg128x128r8x8w16 in outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json (×0.912, below 1 ÷ (×1.065 × gaps ×1.003 × ×1.018) = ×0.920)
 *   - the cases of (256, 48) (a gain is today's time ÷ the tile's, the geometric mean over the readings):
 *     - grid-linear-m1370-n384-k1536-wf32 (×12): reg64x64r4x4w16 against today's reg128x128r8x8w16 ×2.020 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×2.023, gaps ×1.005 / ×1.012; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×2.017, gaps ×1.005 / ×1.006)
 *     - grid-linear-m1370-n384-k384-wf32 (×48): reg64x64r4x4w16 against today's reg128x128r8x8w16 ×1.806 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.820, gaps ×1.003 / ×1.001; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.793, gaps ×1.001 / ×1.006)
 *     - grid-linear-m196-n768-k3072-wf32 (×12): today's reg64x64r4x4w16 ×1.000
 *     - grid-linear-m729-n1152-k1152-wf32 (×110): reg64x64r4x4w16 against today's reg128x128r8x8w16 ×1.403 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.376, gaps ×1.003 / ×1.006; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.430, gaps ×1.010 / ×1.001)
 *     - grid-linear-m729-n4304-k1152-wf32 (×27): reg64x64r4x4w16 against today's reg128x128r8x8w16 ×1.023 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.016, gaps ×1.001 / ×1.019; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.030, gaps ×1.002 / ×1.005)
 *     - grid-linear-m1024-n768-k3072-wf32 (×18): reg64x64r4x4w16 against today's reg128x128r8x8w16 ×1.356 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.359, gaps ×1.037 / ×1.038; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.353, gaps ×1.003 / ×1.001)
 *     - grid-linear-m4096-n3072-k768-wf32 (×18): today's reg128x128r8x8w16 ×1.000
 *     - grid-linear-m17424-n384-k384-wf32 (×2): today's reg128x128r8x8w16 ×1.000
 *     - grid-linear-m65536-n192-k768-wf32 (×2): today's reg128x128r8x8w16 ×1.000
 *     - grid-linear-m256-n1536-k6144-wf32 (×2): today's reg64x64r4x4w16 ×1.000
 *     - grid-linear-m1024-n768-k3072-wf16 (×18): reg64x64r4x4w16 against today's reg128x128r8x8w16 ×1.284 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.263, gaps ×1.013 / ×1.019; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.304, gaps ×1.030 / ×1.018)
 *     - grid-linear-m4096-n768-k3072-wf16 (×18): reg64x64r4x4w16 against today's reg128x128r8x8w16 ×1.092 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.087, gaps ×1.015 / ×1.008; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.096, gaps ×1.022 / ×1.003)
 *     - grid-linear-m256-n512-k512-wbf16 (×96): reg64x32r4x4w8 against today's reg64x64r4x4w16 ×1.265 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.267, gaps ×1.000 / ×1.000; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.262, gaps ×1.001 / ×1.006)
 *     - grid-linear-m64-n2048-k512-wbf16 (×52): reg64x32r4x4w8 against today's reg16x16r1x4w4 ×1.589 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.589, gaps ×1.004 / ×1.000; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.590, gaps ×1.000 / ×1.000)
 *     - grid-linear-m64-n512-k512-wbf16 (×96): today's reg16x16r1x4w4 ×1.000
 *     - grid-linear-m768-n256-k1536-wbf16 (×59): reg64x64r4x4w16 against today's reg128x128r8x8w16 ×2.843 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×2.827, gaps ×1.002 / ×1.004; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×2.859, gaps ×1.000 / ×1.000)
 *     - grid-linear-m768-n6144-k1536-wbf16 (×30): today's reg128x128r8x8w16 ×1.000
 *     - grid-linear-m768-n12288-k1536-wbf16 (×40): today's reg128x128r8x8w16 ×1.000
 *     - grid-linear-m64-n3072-k1024-wbf16 (×56): reg64x64r4x4w16 against today's reg16x16r1x4w4 ×1.531 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.535, gaps ×1.000 / ×1.003; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.527, gaps ×1.002 / ×1.002)
 *     - grid-linear-m512-n1024-k1024-wbf16 (×37): reg64x64r4x4w16 against today's reg64x32r4x4w8 ×1.114 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.113, gaps ×1.007 / ×1.007; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.115, gaps ×1.008 / ×1.000)
 *     - grid-linear-m768-n2048-k1536-wi4g32 (×28): reg64x64r4x4w16 against today's reg128x128r8x8w16 ×1.360 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.341, gaps ×1.001 / ×1.012; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.379, gaps ×1.007 / ×1.002)
 *     - grid-linear-m768-n2560-k10240-wi4g32 (×42): reg64x64r4x4w16 against today's reg128x128r8x8w16 ×1.118 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.091, gaps ×1.004 / ×1.015; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.147, gaps ×1.024 / ×1.010)
 *     - grid-linear-m768-n512-k2560-wi4g32 (×40): reg64x64r4x4w16 against today's reg128x128r8x8w16 ×1.909 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.897, gaps ×1.002 / ×1.001; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.921, gaps ×1.021 / ×1.010)
 *     - grid-linear-m768-n15360-k3840-wi4g32 (×96): today's reg128x128r8x8w16 ×1.000
 *     - grid-linear-m768-n128-k2816-wi4g32 (×30): reg64x32r4x4w8 against today's reg128x128r8x8w16 ×3.135 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×3.135, gaps ×1.000 / ×1.000; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×3.135, gaps ×1.000 / ×1.000)
 *     - grid-linear-m128-n1024-k1024-wi4g32 (×88): today's reg64x32r4x4w8 ×1.000
 *     - grid-bmm-b6-m1370-n64-k1370 (×12): reg64x64r4x4w16 against today's reg128x128r8x8w16 ×2.254 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×2.230, gaps ×1.010 / ×1.017; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×2.278, gaps ×1.000 / ×1.001)
 *     - grid-bmm-b6-m1370-n1370-k64 (×12): reg64x64r4x4w16 against today's reg128x128r8x8w16 ×1.617 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.649, gaps ×1.003 / ×1.012; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.586, gaps ×1.006 / ×1.059)
 *     - grid-bmm-b12-m196-n64-k196 (×12): today's reg64x64r4x4w16 ×1.000
 *     - grid-bmm-b16-m729-n72-k729 (×27): reg64x32r4x4w8 against today's reg128x128r8x8w16 ×1.541 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.507, gaps ×1.010 / ×1.023; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.576, gaps ×1.010 / ×1.009)
 *     - grid-bmm-b864-m144-n32-k144 (×18): reg64x32r4x4w8 against today's reg64x64r4x4w16 ×1.599 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.593, gaps ×1.005 / ×1.004; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.606, gaps ×1.003 / ×1.001)
 *     - grid-bmm-b16-m64-n64-k128 (×28): reg32x32r2x4w8 against today's reg16x16r1x4w4 ×1.053 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.045, gaps ×1.000 / ×1.000; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.061, gaps ×1.004 / ×1.003)
 *     - grid-bmm-b64-m512-n64-k512 (×24): reg64x64r4x4w16 against today's reg64x32r4x4w8 ×1.092 (outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-13-00.json ×1.098, gaps ×1.002 / ×1.002; outputs/workbench/node-nvidia-geforce-rtx-3080-ti/geometry-sweep_2026-11-28T05-21-20.json ×1.085, gaps ×1.004 / ×1.001)
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
  linearGemv: {
    sequential: {
      variant: { cols: 32, unroll: 4 },
      shapes: [
        { storage: "bf16", n: 8960, k: 1536, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 2048, k: 1536, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 256, k: 1536, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 1536, k: 2048, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 6144, k: 1536, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 1536, k: 6144, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 1536, k: 256, variant: { cols: 32, unroll: 16 } },
        { storage: "bf16", n: 4096, k: 1536, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 512, k: 1536, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 1536, k: 4096, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 12288, k: 1536, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 1536, k: 12288, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 10752, k: 2560, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 2048, k: 2560, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 512, k: 2560, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 256, k: 2560, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 2560, k: 2048, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 2560, k: 10240, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 2560, k: 256, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 4096, k: 2560, variant: { cols: 16, unroll: 8 } },
        { storage: "bf16", n: 1024, k: 2560, variant: { cols: 16, unroll: 16 } },
        { storage: "bf16", n: 2560, k: 4096, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 256, k: 1536, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 1536, k: 6144, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 1536, k: 4096, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 12288, k: 1536, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 1536, k: 12288, variant: { cols: 32, unroll: 16 } },
        { storage: "i8", n: 262144, k: 1536, variant: { cols: 16, unroll: 4 } },
        { storage: "i4", n: 262144, k: 1536, variant: { cols: 16, unroll: 1 } },
        { storage: "i4", n: 256, k: 3072, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 256, k: 1024, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 256, k: 2048, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 256, k: 3072, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 1024, k: 256, variant: { cols: 32, unroll: 16 } },
        { storage: "i8", n: 256, k: 1024, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 2048, k: 256, variant: { cols: 32, unroll: 16 } },
        { storage: "i8", n: 256, k: 2048, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 256, k: 1536, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 1536, k: 256, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 2048, k: 2560, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 256, k: 2560, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 2560, k: 2048, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 2560, k: 10240, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 1024, k: 2560, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 262144, k: 2560, variant: { cols: 16, unroll: 4 } },
        { storage: "i4", n: 262144, k: 2560, variant: { cols: 16, unroll: 2 } },
        { storage: "i4", n: 256, k: 5120, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 256, k: 2560, variant: { cols: 16, unroll: 16 } },
        { storage: "i8", n: 2560, k: 256, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 4096, k: 3840, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 2048, k: 3840, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 512, k: 3840, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 3840, k: 4096, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 3840, k: 8192, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 262144, k: 3840, variant: { cols: 16, unroll: 2 } },
        { storage: "i4", n: 1024, k: 7680, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 4096, k: 1024, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 1024, k: 4096, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 8192, k: 1024, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 1024, k: 8192, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 3840, k: 1024, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 262144, k: 1024, variant: { cols: 16, unroll: 8 } },
        { storage: "i4", n: 2048, k: 2816, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 2112, k: 2816, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 1024, k: 2816, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 128, k: 2816, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 2816, k: 4096, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 2816, k: 8192, variant: { cols: 32, unroll: 16 } },
        { storage: "i4", n: 2816, k: 2112, variant: { cols: 16, unroll: 16 } },
        { storage: "i4", n: 262144, k: 2816, variant: { cols: 16, unroll: 1 } },
        { storage: "bf16", n: 262144, k: 1536, variant: { cols: 16, unroll: 4 } },
        { storage: "bf16", n: 10240, k: 2560, variant: { cols: 32, unroll: 8 } },
      ],
    },
    rowsThreadTarget: 16384,
    parallelShapes: [
      {
        storage: "i4",
        n: 8960,
        k: 1536,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 2048,
        k: 1536,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 256,
        k: 1536,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1536,
        k: 2048,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 6144,
        k: 1536,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 1536,
        k: 6144,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1536,
        k: 256,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 4096,
        k: 1536,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 512,
        k: 1536,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1536,
        k: 4096,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 12288,
        k: 1536,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 1536,
        k: 12288,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      { storage: "i8", n: 262144, k: 1536, lanes: 16, packedActivations: false, rowBlock: 4 },
      {
        storage: "i4",
        n: 2048,
        k: 1536,
        group: 512,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 256,
        k: 1536,
        group: 512,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      { storage: "i4", n: 1536, k: 2048, group: 2048, lanes: 32, packedActivations: true },
      {
        storage: "i4",
        n: 6144,
        k: 1536,
        group: 512,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      { storage: "i4", n: 1536, k: 6144, group: 2048, lanes: 32, packedActivations: true },
      { storage: "i8", n: 256, k: 1536, lanes: 32, packedActivations: false },
      { storage: "i8", n: 1536, k: 256, lanes: 4, packedActivations: false },
      {
        storage: "i4",
        n: 4096,
        k: 1536,
        group: 512,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 512,
        k: 1536,
        group: 512,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      { storage: "i4", n: 1536, k: 4096, group: 4096, lanes: 32, packedActivations: true },
      {
        storage: "i2",
        n: 12288,
        k: 1536,
        lanes: 4,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 8,
      },
      { storage: "i2", n: 1536, k: 12288, lanes: 32, packedActivations: true },
      {
        storage: "i4",
        n: 10752,
        k: 2560,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 2048,
        k: 2560,
        group: 32,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 512,
        k: 2560,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 256,
        k: 2560,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2560,
        k: 2048,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 10240,
        k: 2560,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 2560,
        k: 10240,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2560,
        k: 256,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 4096,
        k: 2560,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1024,
        k: 2560,
        group: 32,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2560,
        k: 4096,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      { storage: "i8", n: 262144, k: 2560, lanes: 16, packedActivations: false, rowBlock: 4 },
      {
        storage: "i4",
        n: 2048,
        k: 2560,
        group: 512,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 512,
        k: 2560,
        group: 512,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      { storage: "i4", n: 2560, k: 2048, group: 2048, lanes: 32, packedActivations: true },
      {
        storage: "i4",
        n: 10240,
        k: 2560,
        group: 512,
        lanes: 8,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      { storage: "i4", n: 2560, k: 10240, group: 2048, lanes: 32, packedActivations: true },
      { storage: "i8", n: 256, k: 2560, lanes: 32, packedActivations: false },
      { storage: "i8", n: 2560, k: 256, lanes: 4, packedActivations: false },
      {
        storage: "i4",
        n: 4096,
        k: 2560,
        group: 512,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 1024,
        k: 2560,
        group: 512,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      { storage: "i4", n: 2560, k: 4096, group: 4096, lanes: 32, packedActivations: true },
      {
        storage: "i4",
        n: 256,
        k: 3072,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 256,
        k: 1024,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 15360,
        k: 3840,
        group: 32,
        lanes: 8,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 8,
      },
      { storage: "i4", n: 8192, k: 3840, group: 32, lanes: 4, packedActivations: false },
      {
        storage: "i4",
        n: 4096,
        k: 3840,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2048,
        k: 3840,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 512,
        k: 3840,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 3840,
        k: 4096,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 3840,
        k: 8192,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 3840,
        k: 15360,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 262144,
        k: 3840,
        group: 32,
        lanes: 8,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 1024,
        k: 7680,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1024,
        k: 4096,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1024,
        k: 8192,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 8192,
        k: 2816,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 4096,
        k: 2816,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2048,
        k: 2816,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2112,
        k: 2816,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 1024,
        k: 2816,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 128,
        k: 2816,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2816,
        k: 4096,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2816,
        k: 8192,
        group: 32,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 2816,
        k: 2112,
        group: 32,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      {
        storage: "i4",
        n: 262144,
        k: 2816,
        group: 32,
        lanes: 8,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 8,
      },
      { storage: "bf16", n: 2048, k: 1536, lanes: 16, packedActivations: false },
      { storage: "bf16", n: 256, k: 1536, lanes: 32, packedActivations: false },
      { storage: "bf16", n: 6144, k: 1536, lanes: 8, packedActivations: false },
      { storage: "bf16", n: 1536, k: 6144, lanes: 16, packedActivations: false },
      { storage: "bf16", n: 1536, k: 256, lanes: 8, packedActivations: false },
      { storage: "bf16", n: 4096, k: 1536, lanes: 8, packedActivations: false },
      { storage: "bf16", n: 512, k: 1536, lanes: 32, packedActivations: false },
      {
        storage: "bf16",
        n: 1536,
        k: 4096,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      { storage: "bf16", n: 1536, k: 12288, lanes: 16, packedActivations: false },
      {
        storage: "bf16",
        n: 2048,
        k: 2560,
        lanes: 32,
        packedActivations: false,
        stagedActivations: true,
      },
      { storage: "bf16", n: 512, k: 2560, lanes: 32, packedActivations: false },
      { storage: "bf16", n: 256, k: 2560, lanes: 32, packedActivations: false },
      { storage: "bf16", n: 2560, k: 2048, lanes: 8, packedActivations: false },
      { storage: "bf16", n: 2560, k: 10240, lanes: 16, packedActivations: false },
      { storage: "bf16", n: 4096, k: 2560, lanes: 8, packedActivations: false },
      { storage: "bf16", n: 1024, k: 2560, lanes: 16, packedActivations: false },
      { storage: "bf16", n: 2560, k: 4096, lanes: 16, packedActivations: false },
      {
        storage: "i4",
        n: 2048,
        k: 256,
        group: 32,
        lanes: 4,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 256,
        k: 2048,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      { storage: "i8", n: 256, k: 3072, lanes: 32, packedActivations: false },
      { storage: "i8", n: 1024, k: 256, lanes: 4, packedActivations: false },
      { storage: "i8", n: 256, k: 1024, lanes: 32, packedActivations: false },
      { storage: "i8", n: 2048, k: 256, lanes: 4, packedActivations: false },
      { storage: "i8", n: 256, k: 2048, lanes: 32, packedActivations: false },
      {
        storage: "i4",
        n: 256,
        k: 5120,
        group: 32,
        lanes: 32,
        packedActivations: false,
        rowBlock: 2,
      },
      {
        storage: "i4",
        n: 3840,
        k: 1024,
        group: 32,
        lanes: 16,
        packedActivations: false,
        stagedActivations: true,
        rowBlock: 4,
      },
      { storage: "bf16", n: 1536, k: 2048, lanes: 32, packedActivations: false },
      { storage: "bf16", n: 2560, k: 256, lanes: 8, packedActivations: false },
    ],
  },
  gemmGrid: {
    ladder: [
      { regM: 8, regN: 8, wgX: 16, wgY: 16 },
      { regM: 4, regN: 4, wgX: 16, wgY: 16 },
      { regM: 4, regN: 4, wgX: 8, wgY: 16 },
      { regM: 2, regN: 4, wgX: 8, wgY: 16 },
      { regM: 1, regN: 4, wgX: 4, wgY: 16 },
    ],
    keepLargest: { minWorkgroups: 256, minK: 256 },
    minWorkgroups: 48,
    maxPadding: 1.5,
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
