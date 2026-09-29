/**
 * 幾何プロファイル `nvidia-blackwell`（**生成物 — 手で編集しない**）。
 *
 * tools/geometry-sweep の `profile` が掃引の記録から書いた、adapter `nvidia / blackwell` 用のタイル幾何の
 * 静的な表（perf-ledger K-71）。runtime は adapter の (vendor, architecture) でこの表を選ぶだけで、
 * 実行時には測らない（オートチューン禁止 — ADR 0022 決定 3）。値を変えるときは掃引を取り直して
 * 下のコマンドで再生成する。
 *
 * 再生成（リポ直下から・`--check` を足すと再生成とバイト同一かだけを見る）:
 *
 *   deno run -A tools/geometry-sweep/main.ts profile \
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-37-27.914Z.json \
 *     --id nvidia-blackwell --vendor nvidia --architecture blackwell \
 *     --out packages/runtime/src/kernels/geometry-profiles/nvidia-blackwell.ts --min-speedup 1.05
 *
 * 掃引（adapter nvidia / blackwell / NVIDIA GeForce RTX 5070 Ti）:
 *
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-37-27.914Z.json（sha256 084cd0efd7ea7780dfcdab5b53ee4be8177638c7d8984920096a4851b767aa90・2026-09-27T18:37:27.914Z）
 *
 * 採否の基準: クラスの全ケースで出力が既定と一致し、既定比が ×1.050 以上の幾何のうち、
 * ケース間の幾何平均が最大のもの。無ければ既定（掃引の既定の行の幾何）。同じケースを複数の掃引が
 * 測っていれば、比はその観測の幾何平均。gemmRows は掃引にある linear / matmul / bmm のケースで決め、3 経路に同じ表が効く。
 *
 * 採否:
 *
 * - gemmRows[0]（linear / matmul / bmm の行数 ≤ 64・1 ケース）: 採用 reg32x32r2x4w8 ×1.352（×1.352〜×1.352）
 *   - reg128x128r8x8w16: linear-m64-n3072-k1024 で ×0.501 < ×1.050
 *   - reg128x16r8x4w4: linear-m64-n3072-k1024 で ×0.532 < ×1.050
 *   - reg128x32r8x4w8: linear-m64-n3072-k1024 で ×0.735 < ×1.050
 *   - reg128x32r8x8w4: linear-m64-n3072-k1024 で ×0.588 < ×1.050
 *   - reg128x64r8x4w16: linear-m64-n3072-k1024 で ×0.841 < ×1.050
 *   - reg128x64r8x8w8: linear-m64-n3072-k1024 で ×0.763 < ×1.050
 *   - reg16x128r4x8w16: linear-m64-n3072-k1024 で ×0.540 < ×1.050
 *   - reg16x16r2x4w4: linear-m64-n3072-k1024 で ×0.835 < ×1.050
 *   - reg16x16r4x4w4: linear-m64-n3072-k1024 で ×0.614 < ×1.050
 *   - reg16x32r1x8w4: 幾何平均 ×1.119（採用 ×1.352 に届かない）
 *   - reg16x32r2x4w8: linear-m64-n3072-k1024 で ×1.050 < ×1.050
 *   - reg16x32r2x8w4: linear-m64-n3072-k1024 で ×0.824 < ×1.050
 *   - reg16x32r4x4w8: linear-m64-n3072-k1024 で ×0.847 < ×1.050
 *   - reg16x32r4x8w4: linear-m64-n3072-k1024 で ×0.469 < ×1.050
 *   - reg16x64r2x8w8: linear-m64-n3072-k1024 で ×0.925 < ×1.050
 *   - reg16x64r4x4w16: linear-m64-n3072-k1024 で ×0.964 < ×1.050
 *   - reg16x64r4x8w8: linear-m64-n3072-k1024 で ×0.537 < ×1.050
 *   - reg32x128r4x8w16: linear-m64-n3072-k1024 で ×0.894 < ×1.050
 *   - reg32x128r8x8w16: linear-m64-n3072-k1024 で ×0.466 < ×1.050
 *   - reg32x16r2x4w4: linear-m64-n3072-k1024 で ×0.965 < ×1.050
 *   - reg32x16r4x4w4: linear-m64-n3072-k1024 で ×0.865 < ×1.050
 *   - reg32x16r8x4w4: linear-m64-n3072-k1024 で ×0.596 < ×1.050
 *   - reg32x32r2x8w4: 幾何平均 ×1.167（採用 ×1.352 に届かない）
 *   - reg32x32r4x4w8: 幾何平均 ×1.208（採用 ×1.352 に届かない）
 *   - reg32x32r4x8w4: linear-m64-n3072-k1024 で ×0.769 < ×1.050
 *   - reg32x32r8x4w8: linear-m64-n3072-k1024 で ×0.782 < ×1.050
 *   - reg32x32r8x8w4: linear-m64-n3072-k1024 で ×0.433 < ×1.050
 *   - reg32x64r2x8w8: linear-m64-n3072-k1024 で ×1.036 < ×1.050
 *   - reg32x64r4x4w16: 幾何平均 ×1.091（採用 ×1.352 に届かない）
 *   - reg32x64r4x8w8: linear-m64-n3072-k1024 で ×0.881 < ×1.050
 *   - reg32x64r8x4w16: linear-m64-n3072-k1024 で ×0.933 < ×1.050
 *   - reg32x64r8x8w8: linear-m64-n3072-k1024 で ×0.426 < ×1.050
 *   - reg4x16r1x4w4: linear-m64-n3072-k1024 で ×0.275 < ×1.050
 *   - reg4x32r1x8w4: linear-m64-n3072-k1024 で ×0.273 < ×1.050
 *   - reg64x128r4x8w16: linear-m64-n3072-k1024 で ×0.765 < ×1.050
 *   - reg64x128r8x8w16: linear-m64-n3072-k1024 で ×0.654 < ×1.050
 *   - reg64x16r4x4w4: linear-m64-n3072-k1024 で ×0.980 < ×1.050
 *   - reg64x16r8x4w4: linear-m64-n3072-k1024 で ×0.650 < ×1.050
 *   - reg64x32r4x4w8: 幾何平均 ×1.186（採用 ×1.352 に届かない）
 *   - reg64x32r4x8w4: linear-m64-n3072-k1024 で ×0.936 < ×1.050
 *   - reg64x32r8x4w8: linear-m64-n3072-k1024 で ×0.971 < ×1.050
 *   - reg64x32r8x8w4: linear-m64-n3072-k1024 で ×0.518 < ×1.050
 *   - reg64x64r4x4w16: 幾何平均 ×1.306（採用 ×1.352 に届かない）
 *   - reg64x64r4x8w8: 幾何平均 ×1.061（採用 ×1.352 に届かない）
 *   - reg64x64r8x4w16: linear-m64-n3072-k1024 で ×0.936 < ×1.050
 *   - reg64x64r8x8w8: linear-m64-n3072-k1024 で ×0.631 < ×1.050
 *   - reg8x16r1x4w4: linear-m64-n3072-k1024 で ×0.695 < ×1.050
 *   - reg8x16r2x4w4: linear-m64-n3072-k1024 で ×0.418 < ×1.050
 *   - reg8x32r1x8w4: linear-m64-n3072-k1024 で ×0.791 < ×1.050
 *   - reg8x32r2x4w8: linear-m64-n3072-k1024 で ×0.752 < ×1.050
 *   - reg8x32r2x8w4: linear-m64-n3072-k1024 で ×0.474 < ×1.050
 *   - reg8x64r2x8w8: linear-m64-n3072-k1024 で ×0.534 < ×1.050
 * - gemmRows[1]（linear / matmul / bmm の行数 65〜512・1 ケース）: 採用 reg128x128r8x8w16 ×1.475（×1.475〜×1.475）
 *   - reg128x16r8x4w4: linear-m512-n2048-k1024 で ×0.611 < ×1.050
 *   - reg128x32r8x4w8: 幾何平均 ×1.073（採用 ×1.475 に届かない）
 *   - reg128x32r8x8w4: linear-m512-n2048-k1024 で ×1.042 < ×1.050
 *   - reg128x64r8x4w16: 幾何平均 ×1.412（採用 ×1.475 に届かない）
 *   - reg128x64r8x8w8: 幾何平均 ×1.420（採用 ×1.475 に届かない）
 *   - reg16x128r4x8w16: linear-m512-n2048-k1024 で ×0.832 < ×1.050
 *   - reg16x16r1x4w4: linear-m512-n2048-k1024 で ×0.624 < ×1.050
 *   - reg16x16r2x4w4: linear-m512-n2048-k1024 で ×0.511 < ×1.050
 *   - reg16x16r4x4w4: linear-m512-n2048-k1024 で ×0.376 < ×1.050
 *   - reg16x32r1x8w4: linear-m512-n2048-k1024 で ×0.749 < ×1.050
 *   - reg16x32r2x4w8: linear-m512-n2048-k1024 で ×0.692 < ×1.050
 *   - reg16x32r2x8w4: linear-m512-n2048-k1024 で ×0.649 < ×1.050
 *   - reg16x32r4x4w8: linear-m512-n2048-k1024 で ×0.633 < ×1.050
 *   - reg16x32r4x8w4: linear-m512-n2048-k1024 で ×0.441 < ×1.050
 *   - reg16x64r2x8w8: linear-m512-n2048-k1024 で ×0.718 < ×1.050
 *   - reg16x64r4x4w16: linear-m512-n2048-k1024 で ×0.793 < ×1.050
 *   - reg16x64r4x8w8: linear-m512-n2048-k1024 で ×0.768 < ×1.050
 *   - reg32x128r4x8w16: 幾何平均 ×1.198（採用 ×1.475 に届かない）
 *   - reg32x128r8x8w16: linear-m512-n2048-k1024 で ×1.026 < ×1.050
 *   - reg32x16r2x4w4: linear-m512-n2048-k1024 で ×0.628 < ×1.050
 *   - reg32x16r4x4w4: linear-m512-n2048-k1024 で ×0.548 < ×1.050
 *   - reg32x16r8x4w4: linear-m512-n2048-k1024 で ×0.442 < ×1.050
 *   - reg32x32r2x4w8: linear-m512-n2048-k1024 で ×0.907 < ×1.050
 *   - reg32x32r2x8w4: linear-m512-n2048-k1024 で ×0.875 < ×1.050
 *   - reg32x32r4x4w8: linear-m512-n2048-k1024 で ×0.854 < ×1.050
 *   - reg32x32r4x8w4: linear-m512-n2048-k1024 で ×0.859 < ×1.050
 *   - reg32x32r8x4w8: linear-m512-n2048-k1024 で ×0.848 < ×1.050
 *   - reg32x32r8x8w4: linear-m512-n2048-k1024 で ×0.491 < ×1.050
 *   - reg32x64r2x8w8: linear-m512-n2048-k1024 で ×0.969 < ×1.050
 *   - reg32x64r4x4w16: 幾何平均 ×1.065（採用 ×1.475 に届かない）
 *   - reg32x64r4x8w8: linear-m512-n2048-k1024 で ×1.047 < ×1.050
 *   - reg32x64r8x4w16: 幾何平均 ×1.101（採用 ×1.475 に届かない）
 *   - reg32x64r8x8w8: linear-m512-n2048-k1024 で ×0.964 < ×1.050
 *   - reg4x16r1x4w4: linear-m512-n2048-k1024 で ×0.167 < ×1.050
 *   - reg4x32r1x8w4: linear-m512-n2048-k1024 で ×0.174 < ×1.050
 *   - reg64x128r4x8w16: 幾何平均 ×1.417（採用 ×1.475 に届かない）
 *   - reg64x128r8x8w16: 幾何平均 ×1.320（採用 ×1.475 に届かない）
 *   - reg64x16r4x4w4: linear-m512-n2048-k1024 で ×0.616 < ×1.050
 *   - reg64x16r8x4w4: linear-m512-n2048-k1024 で ×0.603 < ×1.050
 *   - reg64x32r4x8w4: linear-m512-n2048-k1024 で ×0.978 < ×1.050
 *   - reg64x32r8x4w8: linear-m512-n2048-k1024 で ×0.976 < ×1.050
 *   - reg64x32r8x8w4: linear-m512-n2048-k1024 で ×0.948 < ×1.050
 *   - reg64x64r4x4w16: 幾何平均 ×1.262（採用 ×1.475 に届かない）
 *   - reg64x64r4x8w8: 幾何平均 ×1.303（採用 ×1.475 に届かない）
 *   - reg64x64r8x4w16: 幾何平均 ×1.330（採用 ×1.475 に届かない）
 *   - reg64x64r8x8w8: 幾何平均 ×1.224（採用 ×1.475 に届かない）
 *   - reg8x16r1x4w4: linear-m512-n2048-k1024 で ×0.428 < ×1.050
 *   - reg8x16r2x4w4: linear-m512-n2048-k1024 で ×0.261 < ×1.050
 *   - reg8x32r1x8w4: linear-m512-n2048-k1024 で ×0.490 < ×1.050
 *   - reg8x32r2x4w8: linear-m512-n2048-k1024 で ×0.460 < ×1.050
 *   - reg8x32r2x8w4: linear-m512-n2048-k1024 で ×0.301 < ×1.050
 *   - reg8x64r2x8w8: linear-m512-n2048-k1024 で ×0.455 < ×1.050
 * - gemmRows[2]（linear / matmul / bmm の行数 > 512・6 ケース）: 既定 reg128x128r8x8w16 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - reg128x16r8x4w4: linear-m1024-n2048-k2048 で ×0.352 < ×1.050
 *   - reg128x32r8x4w8: linear-m1024-n2048-k2048 で ×0.623 < ×1.050
 *   - reg128x32r8x8w4: linear-m1024-n2048-k8192 で ×0.574 < ×1.050
 *   - reg128x64r8x4w16: linear-m1024-n2048-k8192 で ×0.839 < ×1.050
 *   - reg128x64r8x8w8: linear-m1024-n2048-k2048 で ×0.905 < ×1.050
 *   - reg16x128r4x8w16: linear-m1024-n2048-k2048 で ×0.485 < ×1.050
 *   - reg16x16r1x4w4: linear-m4096-n8192-k2048 で ×0.340 < ×1.050
 *   - reg16x16r2x4w4: linear-m4096-n8192-k2048 で ×0.278 < ×1.050
 *   - reg16x16r4x4w4: linear-m4096-n8192-k2048 で ×0.206 < ×1.050
 *   - reg16x32r1x8w4: linear-m4096-n8192-k2048 で ×0.413 < ×1.050
 *   - reg16x32r2x4w8: linear-m4096-n8192-k2048 で ×0.376 < ×1.050
 *   - reg16x32r2x8w4: linear-m4096-n8192-k2048 で ×0.382 < ×1.050
 *   - reg16x32r4x4w8: linear-m4096-n8192-k2048 で ×0.371 < ×1.050
 *   - reg16x32r4x8w4: linear-m4096-n8192-k2048 で ×0.262 < ×1.050
 *   - reg16x64r2x8w8: linear-m4096-n8192-k2048 で ×0.418 < ×1.050
 *   - reg16x64r4x4w16: linear-m4096-n8192-k2048 で ×0.467 < ×1.050
 *   - reg16x64r4x8w8: linear-m1024-n2048-k2048 で ×0.429 < ×1.050
 *   - reg32x128r4x8w16: linear-m1024-n2048-k2048 で ×0.708 < ×1.050
 *   - reg32x128r8x8w16: linear-m1024-n2048-k8192 で ×0.573 < ×1.050
 *   - reg32x16r2x4w4: linear-m4096-n8192-k2048 で ×0.343 < ×1.050
 *   - reg32x16r4x4w4: linear-m4096-n8192-k2048 で ×0.318 < ×1.050
 *   - reg32x16r8x4w4: linear-m4096-n8192-k2048 で ×0.260 < ×1.050
 *   - reg32x32r2x4w8: linear-m4096-n8192-k2048 で ×0.501 < ×1.050
 *   - reg32x32r2x8w4: linear-m4096-n8192-k2048 で ×0.514 < ×1.050
 *   - reg32x32r4x4w8: linear-m4096-n8192-k2048 で ×0.494 < ×1.050
 *   - reg32x32r4x8w4: linear-m1024-n2048-k2048 で ×0.480 < ×1.050
 *   - reg32x32r8x4w8: linear-m1024-n2048-k2048 で ×0.473 < ×1.050
 *   - reg32x32r8x8w4: linear-m1024-n2048-k8192 で ×0.336 < ×1.050
 *   - reg32x64r2x8w8: linear-m4096-n8192-k2048 で ×0.578 < ×1.050
 *   - reg32x64r4x4w16: linear-m4096-n8192-k2048 で ×0.652 < ×1.050
 *   - reg32x64r4x8w8: linear-m1024-n2048-k2048 で ×0.615 < ×1.050
 *   - reg32x64r8x4w16: linear-m1024-n2048-k2048 で ×0.652 < ×1.050
 *   - reg32x64r8x8w8: linear-m1024-n2048-k8192 で ×0.519 < ×1.050
 *   - reg4x16r1x4w4: linear-m4096-n8192-k2048 で ×0.089 < ×1.050
 *   - reg4x32r1x8w4: linear-m4096-n8192-k2048 で ×0.095 < ×1.050
 *   - reg64x128r4x8w16: linear-m1024-n2048-k8192 で ×0.835 < ×1.050
 *   - reg64x128r8x8w16: linear-m1024-n2048-k2048 で ×0.929 < ×1.050
 *   - reg64x16r4x4w4: linear-m4096-n8192-k2048 で ×0.358 < ×1.050
 *   - reg64x16r8x4w4: linear-m1024-n2048-k2048 で ×0.331 < ×1.050
 *   - reg64x32r4x4w8: linear-m4096-n8192-k2048 で ×0.589 < ×1.050
 *   - reg64x32r4x8w4: linear-m1024-n2048-k2048 で ×0.575 < ×1.050
 *   - reg64x32r8x4w8: linear-m1024-n2048-k2048 で ×0.568 < ×1.050
 *   - reg64x32r8x8w4: linear-m1024-n2048-k8192 で ×0.504 < ×1.050
 *   - reg64x64r4x4w16: linear-m1024-n2048-k2048 で ×0.748 < ×1.050
 *   - reg64x64r4x8w8: linear-m1024-n2048-k2048 で ×0.768 < ×1.050
 *   - reg64x64r8x4w16: linear-m1024-n2048-k2048 で ×0.791 < ×1.050
 *   - reg64x64r8x8w8: linear-m1024-n2048-k8192 で ×0.689 < ×1.050
 *   - reg8x16r1x4w4: linear-m4096-n8192-k2048 で ×0.234 < ×1.050
 *   - reg8x16r2x4w4: linear-m4096-n8192-k2048 で ×0.142 < ×1.050
 *   - reg8x32r1x8w4: linear-m4096-n8192-k2048 で ×0.269 < ×1.050
 *   - reg8x32r2x4w8: linear-m4096-n8192-k2048 で ×0.253 < ×1.050
 *   - reg8x32r2x8w4: linear-m4096-n8192-k2048 で ×0.167 < ×1.050
 *   - reg8x64r2x8w8: linear-m4096-n8192-k2048 で ×0.269 < ×1.050
 * - attention.qk（融合 attention f32 ①QK・4 ケース）: 既定 reg128x128r8x8w16 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - reg128x16r8x4w4: attention-qk-self-m4096-n4096 で ×0.365 < ×1.050
 *   - reg128x32r8x4w8: attention-qk-self-m4096-n4096 で ×0.642 < ×1.050
 *   - reg128x32r8x8w4: attention-qk-self-m4096-n4096 で ×0.622 < ×1.050
 *   - reg128x64r8x4w16: attention-qk-self-m4096-n4096 で ×0.854 < ×1.050
 *   - reg128x64r8x8w8: attention-qk-cross-m4096-n512 で ×0.909 < ×1.050
 *   - reg16x128r4x8w16: attention-qk-self-m4096-n4096 で ×0.459 < ×1.050
 *   - reg16x16r1x4w4: attention-qk-self-m4096-n4096 で ×0.321 < ×1.050
 *   - reg16x16r2x4w4: attention-qk-self-m4096-n4096 で ×0.267 < ×1.050
 *   - reg16x16r4x4w4: attention-qk-self-m4096-n4096 で ×0.202 < ×1.050
 *   - reg16x32r1x8w4: attention-qk-self-m4096-n4096 で ×0.383 < ×1.050
 *   - reg16x32r2x4w8: attention-qk-self-m4096-n4096 で ×0.353 < ×1.050
 *   - reg16x32r2x8w4: attention-qk-self-m4096-n4096 で ×0.357 < ×1.050
 *   - reg16x32r4x4w8: attention-qk-self-m4096-n4096 で ×0.351 < ×1.050
 *   - reg16x32r4x8w4: attention-qk-self-m4096-n4096 で ×0.254 < ×1.050
 *   - reg16x64r2x8w8: attention-qk-self-m4096-n4096 で ×0.390 < ×1.050
 *   - reg16x64r4x4w16: attention-qk-self-m4096-n4096 で ×0.429 < ×1.050
 *   - reg16x64r4x8w8: attention-qk-self-m4096-n4096 で ×0.409 < ×1.050
 *   - reg32x128r4x8w16: attention-qk-self-m4096-n4096 で ×0.685 < ×1.050
 *   - reg32x128r8x8w16: attention-qk-self-m1024-n1024 で ×0.686 < ×1.050
 *   - reg32x16r2x4w4: attention-qk-self-m4096-n4096 で ×0.340 < ×1.050
 *   - reg32x16r4x4w4: attention-qk-self-m4096-n4096 で ×0.316 < ×1.050
 *   - reg32x16r8x4w4: attention-qk-self-m4096-n4096 で ×0.259 < ×1.050
 *   - reg32x32r2x4w8: attention-qk-self-m4096-n4096 で ×0.484 < ×1.050
 *   - reg32x32r2x8w4: attention-qk-self-m4096-n4096 で ×0.496 < ×1.050
 *   - reg32x32r4x4w8: attention-qk-self-m4096-n4096 で ×0.481 < ×1.050
 *   - reg32x32r4x8w4: attention-qk-self-m4096-n4096 で ×0.471 < ×1.050
 *   - reg32x32r8x4w8: attention-qk-self-m4096-n4096 で ×0.468 < ×1.050
 *   - reg32x32r8x8w4: attention-qk-self-m4096-n4096 で ×0.349 < ×1.050
 *   - reg32x64r2x8w8: attention-qk-self-m4096-n4096 で ×0.552 < ×1.050
 *   - reg32x64r4x4w16: attention-qk-self-m4096-n4096 で ×0.616 < ×1.050
 *   - reg32x64r4x8w8: attention-qk-self-m4096-n4096 で ×0.602 < ×1.050
 *   - reg32x64r8x4w16: attention-qk-self-m4096-n4096 で ×0.640 < ×1.050
 *   - reg32x64r8x8w8: attention-qk-cross-m4096-n512 で ×0.614 < ×1.050
 *   - reg4x16r1x4w4: attention-qk-self-m4096-n4096 で ×0.085 < ×1.050
 *   - reg4x32r1x8w4: attention-qk-self-m4096-n4096 で ×0.090 < ×1.050
 *   - reg64x128r4x8w16: attention-qk-self-m4096-n4096 で ×0.849 < ×1.050
 *   - reg64x128r8x8w16: attention-qk-cross-m4096-n512 で ×0.922 < ×1.050
 *   - reg64x16r4x4w4: attention-qk-self-m4096-n4096 で ×0.362 < ×1.050
 *   - reg64x16r8x4w4: attention-qk-self-m4096-n4096 で ×0.341 < ×1.050
 *   - reg64x32r4x4w8: attention-qk-self-m4096-n4096 で ×0.589 < ×1.050
 *   - reg64x32r4x8w4: attention-qk-self-m4096-n4096 で ×0.581 < ×1.050
 *   - reg64x32r8x4w8: attention-qk-self-m4096-n4096 で ×0.572 < ×1.050
 *   - reg64x32r8x8w4: attention-qk-self-m4096-n4096 で ×0.557 < ×1.050
 *   - reg64x64r4x4w16: attention-qk-self-m4096-n4096 で ×0.728 < ×1.050
 *   - reg64x64r4x8w8: attention-qk-self-m4096-n4096 で ×0.763 < ×1.050
 *   - reg64x64r8x4w16: attention-qk-self-m4096-n4096 で ×0.793 < ×1.050
 *   - reg64x64r8x8w8: attention-qk-cross-m4096-n512 で ×0.774 < ×1.050
 *   - reg8x16r1x4w4: attention-qk-self-m4096-n4096 で ×0.216 < ×1.050
 *   - reg8x16r2x4w4: attention-qk-self-m4096-n4096 で ×0.138 < ×1.050
 *   - reg8x32r1x8w4: attention-qk-self-m4096-n4096 で ×0.246 < ×1.050
 *   - reg8x32r2x4w8: attention-qk-self-m4096-n4096 で ×0.230 < ×1.050
 *   - reg8x32r2x8w4: attention-qk-self-m4096-n4096 で ×0.159 < ×1.050
 *   - reg8x64r2x8w8: attention-qk-self-m4096-n4096 で ×0.244 < ×1.050
 * - attention.pv（融合 attention f32 ③PV・4 ケース）: 既定 reg128x128r8x8w16 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - reg128x16r8x4w4: attention-pv-cross-m1024-n512 で ×0.333 < ×1.050
 *   - reg128x32r8x4w8: attention-pv-cross-m1024-n512 で ×0.592 < ×1.050
 *   - reg128x32r8x8w4: attention-pv-self-m1024-n1024 で ×0.459 < ×1.050
 *   - reg128x64r8x4w16: attention-pv-self-m1024-n1024 で ×0.813 < ×1.050
 *   - reg128x64r8x8w8: attention-pv-cross-m1024-n512 で ×0.880 < ×1.050
 *   - reg16x128r4x8w16: attention-pv-cross-m1024-n512 で ×0.752 < ×1.050
 *   - reg16x16r1x4w4: attention-pv-cross-m1024-n512 で ×0.444 < ×1.050
 *   - reg16x16r2x4w4: attention-pv-cross-m1024-n512 で ×0.330 < ×1.050
 *   - reg16x16r4x4w4: attention-pv-cross-m1024-n512 で ×0.302 < ×1.050
 *   - reg16x32r1x8w4: attention-pv-cross-m1024-n512 で ×0.616 < ×1.050
 *   - reg16x32r2x4w8: attention-pv-cross-m1024-n512 で ×0.530 < ×1.050
 *   - reg16x32r2x8w4: attention-pv-cross-m1024-n512 で ×0.572 < ×1.050
 *   - reg16x32r4x4w8: attention-pv-cross-m1024-n512 で ×0.532 < ×1.050
 *   - reg16x32r4x8w4: attention-pv-cross-m1024-n512 で ×0.431 < ×1.050
 *   - reg16x64r2x8w8: attention-pv-cross-m1024-n512 で ×0.635 < ×1.050
 *   - reg16x64r4x4w16: attention-pv-cross-m1024-n512 で ×0.718 < ×1.050
 *   - reg16x64r4x8w8: attention-pv-cross-m1024-n512 で ×0.652 < ×1.050
 *   - reg32x128r4x8w16: attention-pv-cross-m1024-n512 で ×0.879 < ×1.050
 *   - reg32x128r8x8w16: attention-pv-self-m1024-n1024 で ×0.667 < ×1.050
 *   - reg32x16r2x4w4: attention-pv-cross-m1024-n512 で ×0.378 < ×1.050
 *   - reg32x16r4x4w4: attention-pv-cross-m1024-n512 で ×0.346 < ×1.050
 *   - reg32x16r8x4w4: attention-pv-cross-m1024-n512 で ×0.293 < ×1.050
 *   - reg32x32r2x4w8: attention-pv-cross-m1024-n512 で ×0.622 < ×1.050
 *   - reg32x32r2x8w4: attention-pv-cross-m1024-n512 で ×0.642 < ×1.050
 *   - reg32x32r4x4w8: attention-pv-cross-m1024-n512 で ×0.605 < ×1.050
 *   - reg32x32r4x8w4: attention-pv-cross-m1024-n512 で ×0.564 < ×1.050
 *   - reg32x32r8x4w8: attention-pv-cross-m1024-n512 で ×0.539 < ×1.050
 *   - reg32x32r8x8w4: attention-pv-self-m1024-n1024 で ×0.364 < ×1.050
 *   - reg32x64r2x8w8: attention-pv-cross-m1024-n512 で ×0.726 < ×1.050
 *   - reg32x64r4x4w16: attention-pv-cross-m1024-n512 で ×0.824 < ×1.050
 *   - reg32x64r4x8w8: attention-pv-cross-m1024-n512 で ×0.755 < ×1.050
 *   - reg32x64r8x4w16: attention-pv-cross-m1024-n512 で ×0.789 < ×1.050
 *   - reg32x64r8x8w8: attention-pv-self-m1024-n1024 で ×0.625 < ×1.050
 *   - reg4x16r1x4w4: attention-pv-cross-m1024-n512 で ×0.203 < ×1.050
 *   - reg4x32r1x8w4: attention-pv-self-m4096-n4096 で ×0.284 < ×1.050
 *   - reg64x128r4x8w16: attention-pv-cross-m1024-n512 で ×0.916 < ×1.050
 *   - reg64x128r8x8w16: attention-pv-cross-m1024-n512 で ×1.022 < ×1.050
 *   - reg64x16r4x4w4: attention-pv-cross-m1024-n512 で ×0.369 < ×1.050
 *   - reg64x16r8x4w4: attention-pv-cross-m1024-n512 で ×0.324 < ×1.050
 *   - reg64x32r4x4w8: attention-pv-cross-m1024-n512 で ×0.642 < ×1.050
 *   - reg64x32r4x8w4: attention-pv-cross-m1024-n512 で ×0.605 < ×1.050
 *   - reg64x32r8x4w8: attention-pv-cross-m1024-n512 で ×0.581 < ×1.050
 *   - reg64x32r8x8w4: attention-pv-self-m1024-n1024 で ×0.463 < ×1.050
 *   - reg64x64r4x4w16: attention-pv-cross-m1024-n512 で ×0.773 < ×1.050
 *   - reg64x64r4x8w8: attention-pv-cross-m1024-n512 で ×0.816 < ×1.050
 *   - reg64x64r8x4w16: attention-pv-cross-m1024-n512 で ×0.827 < ×1.050
 *   - reg64x64r8x8w8: attention-pv-self-m1024-n1024 で ×0.611 < ×1.050
 *   - reg8x16r1x4w4: attention-pv-cross-m1024-n512 で ×0.339 < ×1.050
 *   - reg8x16r2x4w4: attention-pv-cross-m1024-n512 で ×0.268 < ×1.050
 *   - reg8x32r1x8w4: attention-pv-self-m4096-n4096 で ×0.512 < ×1.050
 *   - reg8x32r2x4w8: attention-pv-cross-m1024-n512 で ×0.438 < ×1.050
 *   - reg8x32r2x8w4: attention-pv-self-m4096-n4096 で ×0.389 < ×1.050
 *   - reg8x64r2x8w8: attention-pv-self-m4096-n4096 で ×0.500 < ×1.050
 * - conv2d.rows64（conv2d implicit GEMM の m タイル 64 行・2 ケース）: 既定 igemm64x128:wg16x8 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - igemm128x128:wg16x16: conv2d-c192-256x256 で ×0.848 < ×1.050
 *   - igemm128x64:wg16x16: conv2d-c192-256x256 で ×0.782 < ×1.050
 *   - igemm128x64:wg8x16: conv2d-c192-256x256 で ×0.806 < ×1.050
 *   - igemm16x128:wg16x4: conv2d-c384-128x128 で ×0.599 < ×1.050
 *   - igemm16x64:wg16x4: conv2d-c384-128x128 で ×0.599 < ×1.050
 *   - igemm16x64:wg8x4: conv2d-c384-128x128 で ×0.580 < ×1.050
 *   - igemm16x64:wg8x8: conv2d-c384-128x128 で ×0.586 < ×1.050
 *   - igemm32x128:wg16x4: conv2d-c384-128x128 で ×0.800 < ×1.050
 *   - igemm32x128:wg16x8: conv2d-c384-128x128 で ×0.839 < ×1.050
 *   - igemm32x64:wg16x4: conv2d-c192-256x256 で ×0.749 < ×1.050
 *   - igemm32x64:wg16x8: conv2d-c384-128x128 で ×0.796 < ×1.050
 *   - igemm32x64:wg8x16: conv2d-c192-256x256 で ×0.774 < ×1.050
 *   - igemm32x64:wg8x4: conv2d-c192-256x256 で ×0.744 < ×1.050
 *   - igemm32x64:wg8x8: conv2d-c192-256x256 で ×0.811 < ×1.050
 *   - igemm64x128:wg16x16: conv2d-c192-256x256 で ×1.003 < ×1.050
 *   - igemm64x64:wg16x16: conv2d-c192-256x256 で ×0.948 < ×1.050
 *   - igemm64x64:wg16x8: conv2d-c192-256x256 で ×0.934 < ×1.050
 *   - igemm64x64:wg8x16: conv2d-c192-256x256 で ×0.979 < ×1.050
 *   - igemm64x64:wg8x8: conv2d-c192-256x256 で ×0.927 < ×1.050
 *   - igemm8x64:wg8x4: conv2d-c384-128x128 で ×0.364 < ×1.050
 * - conv2d.rows32（conv2d implicit GEMM の m タイル 32 行・1 ケース）: 既定 igemm32x128:wg16x4 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - igemm128x128:wg16x16: conv2d-c96-512x512 で ×1.038 < ×1.050
 *   - igemm128x64:wg16x16: conv2d-c96-512x512 で ×0.978 < ×1.050
 *   - igemm128x64:wg8x16: conv2d-c96-512x512 で ×0.995 < ×1.050
 *   - igemm16x128:wg16x4: conv2d-c96-512x512 で ×0.781 < ×1.050
 *   - igemm16x64:wg16x4: conv2d-c96-512x512 で ×0.759 < ×1.050
 *   - igemm16x64:wg8x4: conv2d-c96-512x512 で ×0.729 < ×1.050
 *   - igemm16x64:wg8x8: conv2d-c96-512x512 で ×0.744 < ×1.050
 *   - igemm32x128:wg16x8: conv2d-c96-512x512 で ×1.041 < ×1.050
 *   - igemm32x64:wg16x4: conv2d-c96-512x512 で ×0.922 < ×1.050
 *   - igemm32x64:wg16x8: conv2d-c96-512x512 で ×1.003 < ×1.050
 *   - igemm32x64:wg8x16: conv2d-c96-512x512 で ×0.963 < ×1.050
 *   - igemm32x64:wg8x4: conv2d-c96-512x512 で ×0.940 < ×1.050
 *   - igemm32x64:wg8x8: conv2d-c96-512x512 で ×0.995 < ×1.050
 *   - igemm64x128:wg16x16: conv2d-c96-512x512 で ×0.946 < ×1.050
 *   - igemm64x128:wg16x8: conv2d-c96-512x512 で ×0.929 < ×1.050
 *   - igemm64x64:wg16x16: conv2d-c96-512x512 で ×0.906 < ×1.050
 *   - igemm64x64:wg16x8: conv2d-c96-512x512 で ×0.897 < ×1.050
 *   - igemm64x64:wg8x16: conv2d-c96-512x512 で ×0.945 < ×1.050
 *   - igemm64x64:wg8x8: conv2d-c96-512x512 で ×0.890 < ×1.050
 *   - igemm8x64:wg8x4: conv2d-c96-512x512 で ×0.479 < ×1.050
 * - i8a8.linear（i8a8 linear・6 ケース）: 採用 tile128x64r8x4w16x16k16 ×1.194（×1.156〜×1.256）
 *   - tile128x128r8x8w16x16k16: i8a8-linear-m4096-n2048-k8192 で ×0.845 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-linear-m4096-n2048-k8192 で ×0.880 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-linear-m4096-n8192-k2048 で ×0.885 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-linear-m4096-n8192-k2048 で ×1.029 < ×1.050
 *   - tile128x64r8x4w16x16k32: 幾何平均 ×1.154（採用 ×1.194 に届かない）
 *   - tile128x64r8x8w8x16k32: 幾何平均 ×1.166（採用 ×1.194 に届かない）
 *   - tile16x128r4x8w16x4k16: i8a8-linear-m4096-n2048-k8192 で ×0.582 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-linear-m4096-n8192-k2048 で ×0.633 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-linear-m1024-n2048-k8192 で ×0.596 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-linear-m4096-n8192-k2048 で ×0.675 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-linear-m4096-n8192-k2048 で ×0.551 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-linear-m4096-n8192-k2048 で ×0.685 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-linear-m1024-n2048-k2048 で ×0.510 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-linear-m4096-n8192-k2048 で ×0.747 < ×1.050
 *   - tile32x128r4x8w16x8k16: i8a8-linear-m4096-n8192-k2048 で ×0.744 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-linear-m4096-n8192-k2048 で ×0.771 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-linear-m4096-n2048-k8192 で ×0.858 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-linear-m4096-n2048-k8192 で ×0.848 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-linear-m1024-n2048-k8192 で ×0.714 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-linear-m4096-n8192-k2048 で ×0.768 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-linear-m1024-n2048-k8192 で ×0.610 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-linear-m4096-n2048-k8192 で ×0.884 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-linear-m4096-n8192-k2048 で ×0.823 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-linear-m4096-n8192-k2048 で ×0.812 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-linear-m1024-n2048-k8192 で ×0.798 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-linear-m4096-n8192-k2048 で ×0.928 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-linear-m4096-n2048-k2048 で ×0.881 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-linear-m4096-n2048-k8192 で ×0.960 < ×1.050
 *   - tile32x64r8x8w8x4k16: i8a8-linear-m4096-n2048-k2048 で ×0.883 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-linear-m4096-n2048-k8192 で ×0.860 < ×1.050
 *   - tile64x128r4x8w16x16k16: i8a8-linear-m4096-n8192-k2048 で ×0.842 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-linear-m4096-n2048-k8192 で ×0.829 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m1024-n2048-k2048 で ×0.929 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-linear-m4096-n2048-k8192 で ×0.915 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-linear-m4096-n8192-k2048 で ×0.859 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-linear-m4096-n8192-k2048 で ×0.824 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-linear-m1024-n2048-k2048 で ×0.873 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-linear-m4096-n8192-k2048 で ×0.998 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-linear-m4096-n8192-k2048 で ×0.891 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-linear-m4096-n8192-k2048 で ×0.905 < ×1.050
 *   - tile64x64r4x8w8x16k16: 幾何平均 ×1.082（採用 ×1.194 に届かない）
 *   - tile64x64r4x8w8x16k32: i8a8-linear-m4096-n8192-k2048 で ×1.047 < ×1.050
 *   - tile64x64r8x4w16x8k16: 幾何平均 ×1.122（採用 ×1.194 に届かない）
 *   - tile64x64r8x4w16x8k32: 幾何平均 ×1.108（採用 ×1.194 に届かない）
 *   - tile64x64r8x8w8x8k16: i8a8-linear-m4096-n2048-k8192 で ×0.953 < ×1.050
 *   - tile64x64r8x8w8x8k32: 幾何平均 ×1.101（採用 ×1.194 に届かない）
 * - i8a8.attentionQk（i8a8 attention ①QK・4 ケース）: 採用 tile128x64r8x4w16x16k16 ×1.214（×1.199〜×1.238）
 *   - tile128x128r8x8w16x16k16: i8a8-attention-qk-cross-m1024-n512 で ×0.953 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-attention-qk-cross-m1024-n512 で ×0.891 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-attention-qk-cross-m4096-n512 で ×0.976 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-attention-qk-cross-m4096-n512 で ×1.039 < ×1.050
 *   - tile128x64r8x4w16x16k32: 幾何平均 ×1.190（採用 ×1.214 に届かない）
 *   - tile128x64r8x8w8x16k32: i8a8-attention-qk-cross-m4096-n512 で ×1.013 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile16x128r4x8w16x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile16x32r4x4w8x4k16: i8a8-attention-qk-self-m4096-n4096 で ×0.720 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile16x64r4x4w16x4k16: i8a8-attention-qk-self-m4096-n4096 で ×0.724 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile16x64r4x8w8x4k16: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile16x64r4x8w8x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x128r4x8w16x8k16: i8a8-attention-qk-self-m4096-n4096 で ×0.818 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x128r8x8w16x4k16: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x128r8x8w16x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x32r4x4w8x8k16: i8a8-attention-qk-self-m4096-n4096 で ×0.829 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-attention-qk-cross-m4096-n512 で ×0.807 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-attention-qk-cross-m1024-n512 で ×0.905 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x64r4x4w16x8k16: i8a8-attention-qk-self-m4096-n4096 で ×0.887 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-attention-qk-self-m4096-n4096 で ×0.878 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-attention-qk-self-m4096-n4096 で ×0.943 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x64r8x4w16x4k16: i8a8-attention-qk-cross-m1024-n512 で ×1.001 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x64r8x8w8x4k16: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x64r8x8w8x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile64x128r4x8w16x16k16: i8a8-attention-qk-cross-m1024-n512 で ×0.878 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-attention-qk-cross-m1024-n512 で ×0.885 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-attention-qk-cross-m1024-n512 で ×0.995 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile64x32r4x4w8x16k16: i8a8-attention-qk-cross-m4096-n512 で ×0.892 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-attention-qk-cross-m4096-n512 で ×0.878 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-attention-qk-cross-m1024-n512 で ×0.991 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-attention-qk-cross-m4096-n512 で ×0.972 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-attention-qk-self-m4096-n4096 で ×0.946 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-attention-qk-self-m4096-n4096 で ×0.967 < ×1.050
 *   - tile64x64r4x8w8x16k16: 幾何平均 ×1.083（採用 ×1.214 に届かない）
 *   - tile64x64r4x8w8x16k32: i8a8-attention-qk-self-m4096-n4096 で ×1.010 < ×1.050
 *   - tile64x64r8x4w16x8k16: 幾何平均 ×1.164（採用 ×1.214 に届かない）
 *   - tile64x64r8x4w16x8k32: 幾何平均 ×1.166（採用 ×1.214 に届かない）
 *   - tile64x64r8x8w8x8k16: i8a8-attention-qk-cross-m4096-n512 で ×0.968 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 * - i8a8.attentionPv（i8a8 attention ③PV・4 ケース）: 採用 tile128x64r8x4w16x16k16 ×1.212（×1.134〜×1.306）
 *   - tile128x128r8x8w16x16k16: i8a8-attention-pv-cross-m4096-n512 で ×0.876 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-attention-pv-cross-m4096-n512 で ×0.886 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-attention-pv-self-m4096-n4096 で ×0.815 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-attention-pv-self-m4096-n4096 で ×0.754 < ×1.050
 *   - tile128x64r8x4w16x16k32: 幾何平均 ×1.176（採用 ×1.212 に届かない）
 *   - tile128x64r8x8w8x16k16: i8a8-attention-pv-cross-m4096-n512 で ×0.965 < ×1.050
 *   - tile128x64r8x8w8x16k32: i8a8-attention-pv-cross-m4096-n512 で ×0.840 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.730 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.743 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.651 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.646 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.680 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.769 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.756 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.764 < ×1.050
 *   - tile32x128r4x8w16x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.881 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-attention-pv-self-m4096-n4096 で ×0.895 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.933 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.896 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.770 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-attention-pv-self-m4096-n4096 で ×0.790 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-attention-pv-self-m1024-n1024 で ×0.758 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-attention-pv-cross-m4096-n512 で ×0.714 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.896 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-attention-pv-self-m4096-n4096 で ×0.915 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.937 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-attention-pv-cross-m4096-n512 で ×0.935 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.964 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.944 < ×1.050
 *   - tile32x64r8x8w8x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.892 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-attention-pv-cross-m4096-n512 で ×0.779 < ×1.050
 *   - tile64x128r4x8w16x16k16: i8a8-attention-pv-cross-m4096-n512 で ×0.942 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-attention-pv-self-m4096-n4096 で ×0.941 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-attention-pv-cross-m4096-n512 で ×0.976 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-attention-pv-self-m4096-n4096 で ×0.841 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-attention-pv-self-m4096-n4096 で ×0.844 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.815 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-attention-pv-cross-m4096-n512 で ×0.747 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-attention-pv-self-m4096-n4096 で ×0.966 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-attention-pv-self-m4096-n4096 で ×0.986 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-attention-pv-self-m4096-n4096 で ×1.046 < ×1.050
 *   - tile64x64r4x8w8x16k32: i8a8-attention-pv-self-m4096-n4096 で ×1.017 < ×1.050
 *   - tile64x64r8x4w16x8k16: 幾何平均 ×1.172（採用 ×1.212 に届かない）
 *   - tile64x64r8x4w16x8k32: 幾何平均 ×1.100（採用 ×1.212 に届かない）
 *   - tile64x64r8x8w8x8k16: i8a8-attention-pv-cross-m4096-n512 で ×0.941 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-attention-pv-cross-m4096-n512 で ×0.840 < ×1.050
 */
import type { GeometryProfile } from "../geometry-profile.ts";

export const NVIDIA_BLACKWELL: GeometryProfile = {
  id: "nvidia-blackwell",
  match: { vendor: "nvidia", architecture: "blackwell" },
  gemmRows: [
    { maxRows: 64, geometry: { regM: 2, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: 512, geometry: { regM: 8, regN: 8, wgX: 16, wgY: 16 } },
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
    sweep: "outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-37-27.914Z.json",
    sha256: "084cd0efd7ea7780dfcdab5b53ee4be8177638c7d8984920096a4851b767aa90",
    date: "2026-09-27T18:37:27.914Z",
    adapter: "nvidia / blackwell / NVIDIA GeForce RTX 5070 Ti",
  },
};
