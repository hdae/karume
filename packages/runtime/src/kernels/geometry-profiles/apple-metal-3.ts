/**
 * 幾何プロファイル `apple-metal-3`（**生成物 — 手で編集しない**）。
 *
 * tools/geometry-sweep の `profile` が掃引の記録から書いた、adapter `apple / metal-3` 用のタイル幾何の
 * 静的な表（perf-ledger K-71）。runtime は adapter の (vendor, architecture) でこの表を選ぶだけで、
 * 実行時には測らない（オートチューン禁止 — ADR 0022 決定 3）。値を変えるときは掃引を取り直して
 * 下のコマンドで再生成する。
 *
 * 再生成（リポ直下から・`--check` を足すと再生成とバイト同一かだけを見る）:
 *
 *   deno run -A tools/geometry-sweep/main.ts profile \
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-09-27T16-28-00.787Z.json \
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json \
 *     --id apple-metal-3 --vendor apple --architecture metal-3 \
 *     --out packages/runtime/src/kernels/geometry-profiles/apple-metal-3.ts --min-speedup 1.05
 *
 * 掃引（adapter apple / metal-3 / Apple M2）:
 *
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-27T16-28-00.787Z.json（sha256 921e89ba70b34ef68ec21c153c2a2c313c80bbbc98d53da720942c2919ead034・2026-09-27T16:28:00.787Z）
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json（sha256 f464ca82eb61f4c8fd60efa06365815eb25cec5e1f8ed905cccd242cadacfc2a・2026-09-27T18:31:46.471Z）
 *
 * 採否の基準: クラスの全ケースで出力が既定と一致し、既定比が ×1.050 以上の幾何のうち、
 * ケース間の幾何平均が最大のもの。無ければ既定（掃引の既定の行の幾何）。同じケースを複数の掃引が
 * 測っていれば、比はその観測の幾何平均。gemmRows は linear の実測で決め、同じ骨格の matmul / bmm にも効く。
 *
 * 採否:
 *
 * - gemmRows[0]（linear / matmul / bmm の行数 ≤ 64・1 ケース）: 採用 reg64x64r4x4w16 ×2.261（×2.261〜×2.261）
 *   - reg128x128r8x8w16: linear-m64-n3072-k1024 で ×0.795 < ×1.050
 *   - reg128x16r8x4w4: linear-m64-n3072-k1024 で ×0.924 < ×1.050
 *   - reg128x32r8x4w8: 幾何平均 ×1.223（採用 ×2.261 に届かない）
 *   - reg128x32r8x8w4: linear-m64-n3072-k1024 で ×0.992 < ×1.050
 *   - reg128x64r8x4w16: 幾何平均 ×1.189（採用 ×2.261 に届かない）
 *   - reg128x64r8x8w8: linear-m64-n3072-k1024 で ×1.040 < ×1.050
 *   - reg16x128r4x8w16: linear-m64-n3072-k1024 で ×0.533 < ×1.050
 *   - reg16x16r2x4w4: linear-m64-n3072-k1024 で ×1.027 < ×1.050
 *   - reg16x16r4x4w4: linear-m64-n3072-k1024 で ×0.503 < ×1.050
 *   - reg16x32r1x8w4: linear-m64-n3072-k1024 で ×0.942 < ×1.050
 *   - reg16x32r2x4w8: 幾何平均 ×1.085（採用 ×2.261 に届かない）
 *   - reg16x32r2x8w4: linear-m64-n3072-k1024 で ×0.682 < ×1.050
 *   - reg16x32r4x4w8: linear-m64-n3072-k1024 で ×0.701 < ×1.050
 *   - reg16x32r4x8w4: linear-m64-n3072-k1024 で ×0.370 < ×1.050
 *   - reg16x64r2x8w8: linear-m64-n3072-k1024 で ×0.873 < ×1.050
 *   - reg16x64r4x4w16: linear-m64-n3072-k1024 で ×0.889 < ×1.050
 *   - reg16x64r4x8w8: linear-m64-n3072-k1024 で ×0.509 < ×1.050
 *   - reg32x128r4x8w16: 幾何平均 ×1.239（採用 ×2.261 に届かない）
 *   - reg32x128r8x8w16: linear-m64-n3072-k1024 で ×0.833 < ×1.050
 *   - reg32x16r2x4w4: 幾何平均 ×1.482（採用 ×2.261 に届かない）
 *   - reg32x16r4x4w4: linear-m64-n3072-k1024 で ×1.017 < ×1.050
 *   - reg32x16r8x4w4: linear-m64-n3072-k1024 で ×0.555 < ×1.050
 *   - reg32x32r2x4w8: 幾何平均 ×1.607（採用 ×2.261 に届かない）
 *   - reg32x32r2x8w4: 幾何平均 ×1.603（採用 ×2.261 に届かない）
 *   - reg32x32r4x4w8: 幾何平均 ×1.709（採用 ×2.261 に届かない）
 *   - reg32x32r4x8w4: 幾何平均 ×1.159（採用 ×2.261 に届かない）
 *   - reg32x32r8x4w8: 幾何平均 ×1.140（採用 ×2.261 に届かない）
 *   - reg32x32r8x8w4: linear-m64-n3072-k1024 で ×0.397 < ×1.050
 *   - reg32x64r2x8w8: 幾何平均 ×1.703（採用 ×2.261 に届かない）
 *   - reg32x64r4x4w16: 幾何平均 ×1.757（採用 ×2.261 に届かない）
 *   - reg32x64r4x8w8: 幾何平均 ×1.238（採用 ×2.261 に届かない）
 *   - reg32x64r8x4w16: 幾何平均 ×1.193（採用 ×2.261 に届かない）
 *   - reg32x64r8x8w8: linear-m64-n3072-k1024 で ×0.612 < ×1.050
 *   - reg4x16r1x4w4: linear-m64-n3072-k1024 で ×0.187 < ×1.050
 *   - reg4x32r1x8w4: linear-m64-n3072-k1024 で ×0.128 < ×1.050
 *   - reg64x128r4x8w16: 幾何平均 ×1.696（採用 ×2.261 に届かない）
 *   - reg64x128r8x8w16: 幾何平均 ×1.660（採用 ×2.261 に届かない）
 *   - reg64x16r4x4w4: 幾何平均 ×1.618（採用 ×2.261 に届かない）
 *   - reg64x16r8x4w4: linear-m64-n3072-k1024 で ×1.011 < ×1.050
 *   - reg64x32r4x4w8: 幾何平均 ×2.221（採用 ×2.261 に届かない）
 *   - reg64x32r4x8w4: 幾何平均 ×1.682（採用 ×2.261 に届かない）
 *   - reg64x32r8x4w8: 幾何平均 ×1.633（採用 ×2.261 に届かない）
 *   - reg64x32r8x8w4: linear-m64-n3072-k1024 で ×0.885 < ×1.050
 *   - reg64x64r4x8w8: 幾何平均 ×1.796（採用 ×2.261 に届かない）
 *   - reg64x64r8x4w16: 幾何平均 ×1.719（採用 ×2.261 に届かない）
 *   - reg64x64r8x8w8: 幾何平均 ×1.317（採用 ×2.261 に届かない）
 *   - reg8x16r1x4w4: linear-m64-n3072-k1024 で ×0.596 < ×1.050
 *   - reg8x16r2x4w4: linear-m64-n3072-k1024 で ×0.314 < ×1.050
 *   - reg8x32r1x8w4: linear-m64-n3072-k1024 で ×0.424 < ×1.050
 *   - reg8x32r2x4w8: linear-m64-n3072-k1024 で ×0.446 < ×1.050
 *   - reg8x32r2x8w4: linear-m64-n3072-k1024 で ×0.230 < ×1.050
 *   - reg8x64r2x8w8: linear-m64-n3072-k1024 で ×0.305 < ×1.050
 * - gemmRows[1]（linear / matmul / bmm の行数 65〜512・1 ケース）: 採用 reg128x32r8x4w8 ×1.102（×1.102〜×1.102）
 *   - reg128x128r8x8w16: linear-m512-n2048-k1024 で ×0.653 < ×1.050
 *   - reg128x16r8x4w4: linear-m512-n2048-k1024 で ×0.729 < ×1.050
 *   - reg128x32r8x8w4: linear-m512-n2048-k1024 で ×0.775 < ×1.050
 *   - reg128x64r8x4w16: linear-m512-n2048-k1024 で ×1.020 < ×1.050
 *   - reg128x64r8x8w8: linear-m512-n2048-k1024 で ×0.868 < ×1.050
 *   - reg16x128r4x8w16: linear-m512-n2048-k1024 で ×0.311 < ×1.050
 *   - reg16x16r1x4w4: linear-m512-n2048-k1024 で ×0.432 < ×1.050
 *   - reg16x16r2x4w4: linear-m512-n2048-k1024 で ×0.452 < ×1.050
 *   - reg16x16r4x4w4: linear-m512-n2048-k1024 で ×0.225 < ×1.050
 *   - reg16x32r1x8w4: linear-m512-n2048-k1024 で ×0.460 < ×1.050
 *   - reg16x32r2x4w8: linear-m512-n2048-k1024 で ×0.484 < ×1.050
 *   - reg16x32r2x8w4: linear-m512-n2048-k1024 で ×0.348 < ×1.050
 *   - reg16x32r4x4w8: linear-m512-n2048-k1024 で ×0.365 < ×1.050
 *   - reg16x32r4x8w4: linear-m512-n2048-k1024 で ×0.176 < ×1.050
 *   - reg16x64r2x8w8: linear-m512-n2048-k1024 で ×0.433 < ×1.050
 *   - reg16x64r4x4w16: linear-m512-n2048-k1024 で ×0.449 < ×1.050
 *   - reg16x64r4x8w8: linear-m512-n2048-k1024 で ×0.265 < ×1.050
 *   - reg32x128r4x8w16: linear-m512-n2048-k1024 で ×0.640 < ×1.050
 *   - reg32x128r8x8w16: linear-m512-n2048-k1024 で ×0.457 < ×1.050
 *   - reg32x16r2x4w4: linear-m512-n2048-k1024 で ×0.656 < ×1.050
 *   - reg32x16r4x4w4: linear-m512-n2048-k1024 で ×0.521 < ×1.050
 *   - reg32x16r8x4w4: linear-m512-n2048-k1024 で ×0.291 < ×1.050
 *   - reg32x32r2x4w8: linear-m512-n2048-k1024 で ×0.739 < ×1.050
 *   - reg32x32r2x8w4: linear-m512-n2048-k1024 で ×0.716 < ×1.050
 *   - reg32x32r4x4w8: linear-m512-n2048-k1024 で ×0.765 < ×1.050
 *   - reg32x32r4x8w4: linear-m512-n2048-k1024 で ×0.570 < ×1.050
 *   - reg32x32r8x4w8: linear-m512-n2048-k1024 で ×0.571 < ×1.050
 *   - reg32x32r8x8w4: linear-m512-n2048-k1024 で ×0.224 < ×1.050
 *   - reg32x64r2x8w8: linear-m512-n2048-k1024 で ×0.731 < ×1.050
 *   - reg32x64r4x4w16: linear-m512-n2048-k1024 で ×0.795 < ×1.050
 *   - reg32x64r4x8w8: linear-m512-n2048-k1024 で ×0.608 < ×1.050
 *   - reg32x64r8x4w16: linear-m512-n2048-k1024 で ×0.652 < ×1.050
 *   - reg32x64r8x8w8: linear-m512-n2048-k1024 で ×0.385 < ×1.050
 *   - reg4x16r1x4w4: linear-m512-n2048-k1024 で ×0.080 < ×1.050
 *   - reg4x32r1x8w4: linear-m512-n2048-k1024 で ×0.057 < ×1.050
 *   - reg64x128r4x8w16: linear-m512-n2048-k1024 で ×0.860 < ×1.050
 *   - reg64x128r8x8w16: linear-m512-n2048-k1024 で ×0.699 < ×1.050
 *   - reg64x16r4x4w4: linear-m512-n2048-k1024 で ×0.801 < ×1.050
 *   - reg64x16r8x4w4: linear-m512-n2048-k1024 で ×0.505 < ×1.050
 *   - reg64x32r4x8w4: linear-m512-n2048-k1024 で ×0.877 < ×1.050
 *   - reg64x32r8x4w8: linear-m512-n2048-k1024 で ×0.873 < ×1.050
 *   - reg64x32r8x8w4: linear-m512-n2048-k1024 で ×0.551 < ×1.050
 *   - reg64x64r4x4w16: linear-m512-n2048-k1024 で ×0.981 < ×1.050
 *   - reg64x64r4x8w8: linear-m512-n2048-k1024 で ×0.907 < ×1.050
 *   - reg64x64r8x4w16: linear-m512-n2048-k1024 で ×0.981 < ×1.050
 *   - reg64x64r8x8w8: linear-m512-n2048-k1024 で ×0.686 < ×1.050
 *   - reg8x16r1x4w4: linear-m512-n2048-k1024 で ×0.268 < ×1.050
 *   - reg8x16r2x4w4: linear-m512-n2048-k1024 で ×0.140 < ×1.050
 *   - reg8x32r1x8w4: linear-m512-n2048-k1024 で ×0.199 < ×1.050
 *   - reg8x32r2x4w8: linear-m512-n2048-k1024 で ×0.211 < ×1.050
 *   - reg8x32r2x8w4: linear-m512-n2048-k1024 で ×0.109 < ×1.050
 *   - reg8x64r2x8w8: linear-m512-n2048-k1024 で ×0.143 < ×1.050
 * - gemmRows[2]（linear / matmul / bmm の行数 > 512・6 ケース）: 採用 reg128x32r8x4w8 ×1.740（×1.678〜×1.773）
 *   - reg128x16r8x4w4: linear-m1024-n2048-k8192 で ×1.039 < ×1.050
 *   - reg128x32r8x8w4: 幾何平均 ×1.315（採用 ×1.740 に届かない）
 *   - reg128x64r8x4w16: 幾何平均 ×1.581（採用 ×1.740 に届かない）
 *   - reg128x64r8x8w8: 幾何平均 ×1.382（採用 ×1.740 に届かない）
 *   - reg16x128r4x8w16: linear-m1024-n2048-k8192 で ×0.481 < ×1.050
 *   - reg16x16r1x4w4: linear-m1024-n2048-k8192 で ×0.662 < ×1.050
 *   - reg16x16r2x4w4: linear-m1024-n2048-k8192 で ×0.681 < ×1.050
 *   - reg16x16r4x4w4: linear-m1024-n2048-k8192 で ×0.331 < ×1.050
 *   - reg16x32r1x8w4: linear-m1024-n2048-k8192 で ×0.702 < ×1.050
 *   - reg16x32r2x4w8: linear-m1024-n2048-k8192 で ×0.736 < ×1.050
 *   - reg16x32r2x8w4: linear-m1024-n2048-k8192 で ×0.534 < ×1.050
 *   - reg16x32r4x4w8: linear-m1024-n2048-k8192 で ×0.548 < ×1.050
 *   - reg16x32r4x8w4: linear-m1024-n2048-k8192 で ×0.263 < ×1.050
 *   - reg16x64r2x8w8: linear-m4096-n2048-k8192 で ×0.664 < ×1.050
 *   - reg16x64r4x4w16: linear-m1024-n2048-k8192 で ×0.647 < ×1.050
 *   - reg16x64r4x8w8: linear-m1024-n2048-k8192 で ×0.412 < ×1.050
 *   - reg32x128r4x8w16: linear-m1024-n2048-k8192 で ×0.961 < ×1.050
 *   - reg32x128r8x8w16: linear-m1024-n2048-k8192 で ×0.768 < ×1.050
 *   - reg32x16r2x4w4: linear-m1024-n2048-k8192 で ×0.998 < ×1.050
 *   - reg32x16r4x4w4: linear-m1024-n2048-k8192 で ×0.794 < ×1.050
 *   - reg32x16r8x4w4: linear-m4096-n2048-k8192 で ×0.432 < ×1.050
 *   - reg32x32r2x4w8: 幾何平均 ×1.158（採用 ×1.740 に届かない）
 *   - reg32x32r2x8w4: 幾何平均 ×1.113（採用 ×1.740 に届かない）
 *   - reg32x32r4x4w8: 幾何平均 ×1.195（採用 ×1.740 に届かない）
 *   - reg32x32r4x8w4: linear-m1024-n2048-k8192 で ×0.887 < ×1.050
 *   - reg32x32r8x4w8: linear-m1024-n2048-k8192 で ×0.903 < ×1.050
 *   - reg32x32r8x8w4: linear-m1024-n2048-k8192 で ×0.341 < ×1.050
 *   - reg32x64r2x8w8: 幾何平均 ×1.141（採用 ×1.740 に届かない）
 *   - reg32x64r4x4w16: 幾何平均 ×1.249（採用 ×1.740 に届かない）
 *   - reg32x64r4x8w8: linear-m1024-n2048-k8192 で ×0.942 < ×1.050
 *   - reg32x64r8x4w16: linear-m1024-n2048-k8192 で ×1.018 < ×1.050
 *   - reg32x64r8x8w8: linear-m1024-n2048-k8192 で ×0.638 < ×1.050
 *   - reg4x16r1x4w4: linear-m1024-n2048-k8192 で ×0.120 < ×1.050
 *   - reg4x32r1x8w4: linear-m1024-n2048-k8192 で ×0.084 < ×1.050
 *   - reg64x128r4x8w16: 幾何平均 ×1.318（採用 ×1.740 に届かない）
 *   - reg64x128r8x8w16: 幾何平均 ×1.118（採用 ×1.740 に届かない）
 *   - reg64x16r4x4w4: 幾何平均 ×1.224（採用 ×1.740 に届かない）
 *   - reg64x16r8x4w4: linear-m1024-n2048-k8192 で ×0.768 < ×1.050
 *   - reg64x32r4x4w8: 幾何平均 ×1.583（採用 ×1.740 に届かない）
 *   - reg64x32r4x8w4: 幾何平均 ×1.420（採用 ×1.740 に届かない）
 *   - reg64x32r8x4w8: 幾何平均 ×1.431（採用 ×1.740 に届かない）
 *   - reg64x32r8x8w4: linear-m1024-n2048-k8192 で ×0.877 < ×1.050
 *   - reg64x64r4x4w16: 幾何平均 ×1.535（採用 ×1.740 に届かない）
 *   - reg64x64r4x8w8: 幾何平均 ×1.406（採用 ×1.740 に届かない）
 *   - reg64x64r8x4w16: 幾何平均 ×1.565（採用 ×1.740 に届かない）
 *   - reg64x64r8x8w8: 幾何平均 ×1.106（採用 ×1.740 に届かない）
 *   - reg8x16r1x4w4: linear-m1024-n2048-k2048 で ×0.373 < ×1.050
 *   - reg8x16r2x4w4: linear-m1024-n2048-k8192 で ×0.205 < ×1.050
 *   - reg8x32r1x8w4: linear-m1024-n2048-k8192 で ×0.300 < ×1.050
 *   - reg8x32r2x4w8: linear-m1024-n2048-k8192 で ×0.322 < ×1.050
 *   - reg8x32r2x8w4: linear-m1024-n2048-k8192 で ×0.161 < ×1.050
 *   - reg8x64r2x8w8: linear-m1024-n2048-k8192 で ×0.212 < ×1.050
 * - attention.qk（融合 attention f32 ①QK・4 ケース）: 採用 reg128x32r8x4w8 ×1.734（×1.571〜×1.830）
 *   - reg128x16r8x4w4: 幾何平均 ×1.160（採用 ×1.734 に届かない）
 *   - reg128x32r8x8w4: 幾何平均 ×1.314（採用 ×1.734 に届かない）
 *   - reg128x64r8x4w16: 幾何平均 ×1.590（採用 ×1.734 に届かない）
 *   - reg128x64r8x8w8: 幾何平均 ×1.385（採用 ×1.734 に届かない）
 *   - reg16x128r4x8w16: attention-qk-self-m1024-n1024 で ×0.495 < ×1.050
 *   - reg16x16r1x4w4: attention-qk-cross-m4096-n512 で ×0.603 < ×1.050
 *   - reg16x16r2x4w4: attention-qk-cross-m4096-n512 で ×0.694 < ×1.050
 *   - reg16x16r4x4w4: attention-qk-self-m1024-n1024 で ×0.381 < ×1.050
 *   - reg16x32r1x8w4: attention-qk-cross-m4096-n512 で ×0.699 < ×1.050
 *   - reg16x32r2x4w8: attention-qk-cross-m4096-n512 で ×0.746 < ×1.050
 *   - reg16x32r2x8w4: attention-qk-self-m1024-n1024 で ×0.580 < ×1.050
 *   - reg16x32r4x4w8: attention-qk-self-m1024-n1024 で ×0.603 < ×1.050
 *   - reg16x32r4x8w4: attention-qk-self-m1024-n1024 で ×0.282 < ×1.050
 *   - reg16x64r2x8w8: attention-qk-self-m1024-n1024 で ×0.704 < ×1.050
 *   - reg16x64r4x4w16: attention-qk-cross-m4096-n512 で ×0.667 < ×1.050
 *   - reg16x64r4x8w8: attention-qk-self-m1024-n1024 で ×0.419 < ×1.050
 *   - reg32x128r4x8w16: attention-qk-self-m1024-n1024 で ×1.021 < ×1.050
 *   - reg32x128r8x8w16: attention-qk-self-m1024-n1024 で ×0.791 < ×1.050
 *   - reg32x16r2x4w4: attention-qk-cross-m4096-n512 で ×0.955 < ×1.050
 *   - reg32x16r4x4w4: attention-qk-self-m1024-n1024 で ×0.837 < ×1.050
 *   - reg32x16r8x4w4: attention-qk-self-m1024-n1024 で ×0.468 < ×1.050
 *   - reg32x32r2x4w8: attention-qk-cross-m4096-n512 で ×1.041 < ×1.050
 *   - reg32x32r2x8w4: 幾何平均 ×1.133（採用 ×1.734 に届かない）
 *   - reg32x32r4x4w8: 幾何平均 ×1.194（採用 ×1.734 に届かない）
 *   - reg32x32r4x8w4: attention-qk-cross-m4096-n512 で ×0.940 < ×1.050
 *   - reg32x32r8x4w8: attention-qk-cross-m1024-n512 で ×0.969 < ×1.050
 *   - reg32x32r8x8w4: attention-qk-self-m1024-n1024 で ×0.373 < ×1.050
 *   - reg32x64r2x8w8: 幾何平均 ×1.139（採用 ×1.734 に届かない）
 *   - reg32x64r4x4w16: 幾何平均 ×1.237（採用 ×1.734 に届かない）
 *   - reg32x64r4x8w8: 幾何平均 ×1.086（採用 ×1.734 に届かない）
 *   - reg32x64r8x4w16: attention-qk-cross-m4096-n512 で ×1.022 < ×1.050
 *   - reg32x64r8x8w8: attention-qk-self-m1024-n1024 で ×0.677 < ×1.050
 *   - reg4x16r1x4w4: attention-qk-self-m1024-n1024 で ×0.133 < ×1.050
 *   - reg4x32r1x8w4: attention-qk-self-m1024-n1024 で ×0.093 < ×1.050
 *   - reg64x128r4x8w16: 幾何平均 ×1.288（採用 ×1.734 に届かない）
 *   - reg64x128r8x8w16: 幾何平均 ×1.163（採用 ×1.734 に届かない）
 *   - reg64x16r4x4w4: 幾何平均 ×1.198（採用 ×1.734 に届かない）
 *   - reg64x16r8x4w4: attention-qk-cross-m1024-n512 で ×0.822 < ×1.050
 *   - reg64x32r4x4w8: 幾何平均 ×1.545（採用 ×1.734 に届かない）
 *   - reg64x32r4x8w4: 幾何平均 ×1.408（採用 ×1.734 に届かない）
 *   - reg64x32r8x4w8: 幾何平均 ×1.422（採用 ×1.734 に届かない）
 *   - reg64x32r8x8w4: attention-qk-cross-m1024-n512 で ×0.921 < ×1.050
 *   - reg64x64r4x4w16: 幾何平均 ×1.489（採用 ×1.734 に届かない）
 *   - reg64x64r4x8w8: 幾何平均 ×1.407（採用 ×1.734 に届かない）
 *   - reg64x64r8x4w16: 幾何平均 ×1.552（採用 ×1.734 に届かない）
 *   - reg64x64r8x8w8: 幾何平均 ×1.273（採用 ×1.734 に届かない）
 *   - reg8x16r1x4w4: attention-qk-cross-m4096-n512 で ×0.401 < ×1.050
 *   - reg8x16r2x4w4: attention-qk-cross-m4096-n512 で ×0.222 < ×1.050
 *   - reg8x32r1x8w4: attention-qk-self-m1024-n1024 で ×0.329 < ×1.050
 *   - reg8x32r2x4w8: attention-qk-self-m1024-n1024 で ×0.346 < ×1.050
 *   - reg8x32r2x8w4: attention-qk-self-m1024-n1024 で ×0.179 < ×1.050
 *   - reg8x64r2x8w8: attention-qk-self-m1024-n1024 で ×0.229 < ×1.050
 * - attention.pv（融合 attention f32 ③PV・4 ケース）: 採用 reg64x64r8x4w16 ×1.494（×1.403〜×1.671）
 *   - reg128x16r8x4w4: attention-pv-cross-m1024-n512 で ×0.831 < ×1.050
 *   - reg128x32r8x4w8: 幾何平均 ×1.249（採用 ×1.494 に届かない）
 *   - reg128x32r8x8w4: attention-pv-self-m1024-n1024 で ×0.962 < ×1.050
 *   - reg128x64r8x4w16: 幾何平均 ×1.292（採用 ×1.494 に届かない）
 *   - reg128x64r8x8w8: attention-pv-self-m1024-n1024 で ×1.013 < ×1.050
 *   - reg16x128r4x8w16: attention-pv-cross-m1024-n512 で ×0.917 < ×1.050
 *   - reg16x16r1x4w4: attention-pv-cross-m4096-n512 で ×0.833 < ×1.050
 *   - reg16x16r2x4w4: 幾何平均 ×1.184（採用 ×1.494 に届かない）
 *   - reg16x16r4x4w4: attention-pv-cross-m1024-n512 で ×0.580 < ×1.050
 *   - reg16x32r1x8w4: attention-pv-cross-m4096-n512 で ×0.886 < ×1.050
 *   - reg16x32r2x4w8: 幾何平均 ×1.266（採用 ×1.494 に届かない）
 *   - reg16x32r2x8w4: attention-pv-cross-m1024-n512 で ×0.975 < ×1.050
 *   - reg16x32r4x4w8: attention-pv-cross-m1024-n512 で ×0.915 < ×1.050
 *   - reg16x32r4x8w4: attention-pv-cross-m1024-n512 で ×0.618 < ×1.050
 *   - reg16x64r2x8w8: attention-pv-cross-m1024-n512 で ×1.031 < ×1.050
 *   - reg16x64r4x4w16: attention-pv-cross-m1024-n512 で ×0.971 < ×1.050
 *   - reg16x64r4x8w8: attention-pv-cross-m1024-n512 で ×0.852 < ×1.050
 *   - reg32x128r4x8w16: 幾何平均 ×1.358（採用 ×1.494 に届かない）
 *   - reg32x128r8x8w16: attention-pv-self-m1024-n1024 で ×0.949 < ×1.050
 *   - reg32x16r2x4w4: attention-pv-cross-m4096-n512 で ×1.037 < ×1.050
 *   - reg32x16r4x4w4: attention-pv-cross-m1024-n512 で ×0.893 < ×1.050
 *   - reg32x16r8x4w4: attention-pv-cross-m1024-n512 で ×0.534 < ×1.050
 *   - reg32x32r2x4w8: 幾何平均 ×1.288（採用 ×1.494 に届かない）
 *   - reg32x32r2x8w4: 幾何平均 ×1.273（採用 ×1.494 に届かない）
 *   - reg32x32r4x4w8: 幾何平均 ×1.400（採用 ×1.494 に届かない）
 *   - reg32x32r4x8w4: 幾何平均 ×1.282（採用 ×1.494 に届かない）
 *   - reg32x32r8x4w8: 幾何平均 ×1.204（採用 ×1.494 に届かない）
 *   - reg32x32r8x8w4: attention-pv-cross-m1024-n512 で ×0.508 < ×1.050
 *   - reg32x64r2x8w8: 幾何平均 ×1.265（採用 ×1.494 に届かない）
 *   - reg32x64r4x4w16: 幾何平均 ×1.304（採用 ×1.494 に届かない）
 *   - reg32x64r4x8w8: 幾何平均 ×1.397（採用 ×1.494 に届かない）
 *   - reg32x64r8x4w16: 幾何平均 ×1.253（採用 ×1.494 に届かない）
 *   - reg32x64r8x8w8: attention-pv-self-m1024-n1024 で ×0.927 < ×1.050
 *   - reg4x16r1x4w4: attention-pv-self-m4096-n4096 で ×0.432 < ×1.050
 *   - reg4x32r1x8w4: attention-pv-self-m4096-n4096 で ×0.284 < ×1.050
 *   - reg64x128r4x8w16: 幾何平均 ×1.335（採用 ×1.494 に届かない）
 *   - reg64x128r8x8w16: attention-pv-self-m1024-n1024 で ×0.941 < ×1.050
 *   - reg64x16r4x4w4: attention-pv-cross-m1024-n512 で ×0.949 < ×1.050
 *   - reg64x16r8x4w4: attention-pv-cross-m1024-n512 で ×0.767 < ×1.050
 *   - reg64x32r4x4w8: 幾何平均 ×1.392（採用 ×1.494 に届かない）
 *   - reg64x32r4x8w4: 幾何平均 ×1.350（採用 ×1.494 に届かない）
 *   - reg64x32r8x4w8: 幾何平均 ×1.250（採用 ×1.494 に届かない）
 *   - reg64x32r8x8w4: attention-pv-self-m1024-n1024 で ×0.863 < ×1.050
 *   - reg64x64r4x4w16: 幾何平均 ×1.417（採用 ×1.494 に届かない）
 *   - reg64x64r4x8w8: 幾何平均 ×1.365（採用 ×1.494 に届かない）
 *   - reg64x64r8x8w8: attention-pv-self-m1024-n1024 で ×1.035 < ×1.050
 *   - reg8x16r1x4w4: attention-pv-cross-m4096-n512 で ×0.800 < ×1.050
 *   - reg8x16r2x4w4: attention-pv-cross-m4096-n512 で ×0.576 < ×1.050
 *   - reg8x32r1x8w4: attention-pv-self-m4096-n4096 で ×0.712 < ×1.050
 *   - reg8x32r2x4w8: attention-pv-self-m4096-n4096 で ×0.806 < ×1.050
 *   - reg8x32r2x8w4: attention-pv-self-m4096-n4096 で ×0.448 < ×1.050
 *   - reg8x64r2x8w8: attention-pv-self-m4096-n4096 で ×0.616 < ×1.050
 * - conv2d.rows64（conv2d implicit GEMM の m タイル 64 行・2 ケース）: 採用 igemm128x64:wg16x16 ×1.444（×1.219〜×1.711）
 *   - igemm128x128:wg16x16: conv2d-c192-256x256 で ×0.806 < ×1.050
 *   - igemm128x64:wg8x16: conv2d-c192-256x256 で ×1.044 < ×1.050
 *   - igemm16x128:wg16x4: conv2d-c384-128x128 で ×0.392 < ×1.050
 *   - igemm16x64:wg16x4: conv2d-c384-128x128 で ×0.525 < ×1.050
 *   - igemm16x64:wg8x4: conv2d-c384-128x128 で ×0.360 < ×1.050
 *   - igemm16x64:wg8x8: conv2d-c384-128x128 で ×0.515 < ×1.050
 *   - igemm32x128:wg16x4: conv2d-c384-128x128 で ×0.640 < ×1.050
 *   - igemm32x128:wg16x8: conv2d-c384-128x128 で ×0.779 < ×1.050
 *   - igemm32x64:wg16x4: conv2d-c384-128x128 で ×0.827 < ×1.050
 *   - igemm32x64:wg16x8: conv2d-c384-128x128 で ×0.921 < ×1.050
 *   - igemm32x64:wg8x16: conv2d-c384-128x128 で ×0.938 < ×1.050
 *   - igemm32x64:wg8x4: conv2d-c384-128x128 で ×0.583 < ×1.050
 *   - igemm32x64:wg8x8: conv2d-c384-128x128 で ×0.841 < ×1.050
 *   - igemm64x128:wg16x16: 幾何平均 ×1.234（採用 ×1.444 に届かない）
 *   - igemm64x64:wg16x16: 幾何平均 ×1.394（採用 ×1.444 に届かない）
 *   - igemm64x64:wg16x8: 幾何平均 ×1.297（採用 ×1.444 に届かない）
 *   - igemm64x64:wg8x16: 幾何平均 ×1.281（採用 ×1.444 に届かない）
 *   - igemm64x64:wg8x8: conv2d-c384-128x128 で ×1.005 < ×1.050
 *   - igemm8x64:wg8x4: conv2d-c384-128x128 で ×0.189 < ×1.050
 * - conv2d.rows32（conv2d implicit GEMM の m タイル 32 行・1 ケース）: 採用 igemm128x64:wg16x16 ×1.648（×1.648〜×1.648）
 *   - igemm128x128:wg16x16: 幾何平均 ×1.187（採用 ×1.648 に届かない）
 *   - igemm128x64:wg8x16: 幾何平均 ×1.504（採用 ×1.648 に届かない）
 *   - igemm16x128:wg16x4: conv2d-c96-512x512 で ×0.662 < ×1.050
 *   - igemm16x64:wg16x4: conv2d-c96-512x512 で ×0.815 < ×1.050
 *   - igemm16x64:wg8x4: conv2d-c96-512x512 で ×0.597 < ×1.050
 *   - igemm16x64:wg8x8: conv2d-c96-512x512 で ×0.829 < ×1.050
 *   - igemm32x128:wg16x8: 幾何平均 ×1.274（採用 ×1.648 に届かない）
 *   - igemm32x64:wg16x4: 幾何平均 ×1.341（採用 ×1.648 に届かない）
 *   - igemm32x64:wg16x8: 幾何平均 ×1.333（採用 ×1.648 に届かない）
 *   - igemm32x64:wg8x16: 幾何平均 ×1.307（採用 ×1.648 に届かない）
 *   - igemm32x64:wg8x4: conv2d-c96-512x512 で ×0.944 < ×1.050
 *   - igemm32x64:wg8x8: 幾何平均 ×1.352（採用 ×1.648 に届かない）
 *   - igemm64x128:wg16x16: 幾何平均 ×1.366（採用 ×1.648 に届かない）
 *   - igemm64x128:wg16x8: 幾何平均 ×1.185（採用 ×1.648 に届かない）
 *   - igemm64x64:wg16x16: 幾何平均 ×1.379（採用 ×1.648 に届かない）
 *   - igemm64x64:wg16x8: 幾何平均 ×1.430（採用 ×1.648 に届かない）
 *   - igemm64x64:wg8x16: 幾何平均 ×1.356（採用 ×1.648 に届かない）
 *   - igemm64x64:wg8x8: 幾何平均 ×1.194（採用 ×1.648 に届かない）
 *   - igemm8x64:wg8x4: conv2d-c96-512x512 で ×0.321 < ×1.050
 * - i8a8.linear（i8a8 linear・6 ケース）: 採用 tile64x64r8x4w16x8k16 ×1.112（×1.061〜×1.127）
 *   - tile32x32r4x4w8x8k16: i8a8-linear-m1024-n8192-k2048 で ×1.002 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m1024-n2048-k8192 で ×0.975 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-linear-m1024-n8192-k2048 で ×1.000 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-linear-m4096-n2048-k8192 で ×0.967 < ×1.050
 * - i8a8.attentionQk（i8a8 attention ①QK・4 ケース）: 採用 tile64x64r8x4w16x8k16 ×1.128（×1.127〜×1.130）
 *   - tile32x32r4x4w8x8k16: i8a8-attention-qk-cross-m4096-n512 で ×1.031 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-attention-qk-self-m4096-n4096 で ×0.998 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-attention-qk-self-m4096-n4096 で ×1.025 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-attention-qk-cross-m1024-n512 で ×0.990 < ×1.050
 * - i8a8.attentionPv（i8a8 attention ③PV・4 ケース）: 採用 tile64x64r8x4w16x8k16 ×1.098（×1.090〜×1.106）
 *   - tile128x64r8x8w8x16k16: i8a8-attention-pv-cross-m4096-n512 で ×0.975 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.959 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-attention-pv-cross-m4096-n512 で ×0.957 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-attention-pv-self-m4096-n4096 で ×0.967 < ×1.050
 */
import type { GeometryProfile } from "../geometry-profile.ts";

export const APPLE_METAL_3: GeometryProfile = {
  id: "apple-metal-3",
  match: { vendor: "apple", architecture: "metal-3" },
  gemmRows: [
    { maxRows: 64, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 16 } },
    { maxRows: 512, geometry: { regM: 8, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: Number.POSITIVE_INFINITY, geometry: { regM: 8, regN: 4, wgX: 8, wgY: 16 } },
  ],
  attention: {
    qk: { regM: 8, regN: 4, wgX: 8, wgY: 16 },
    pv: { regM: 8, regN: 4, wgX: 16, wgY: 8 },
  },
  conv2d: {
    rows64: { regM: 8, regN: 4, wgX: 16, wgY: 16 },
    rows32: { regM: 8, regN: 4, wgX: 16, wgY: 16 },
  },
  i8a8: {
    linear: { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 },
    attentionQk: { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 },
    attentionPv: { regM: 8, regN: 4, wgX: 16, wgY: 8, tileK: 16 },
  },
  provenance: {
    sweep:
      "outputs/bench-browser/geometry-sweep-browser-2026-09-27T16-28-00.787Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json",
    sha256:
      "921e89ba70b34ef68ec21c153c2a2c313c80bbbc98d53da720942c2919ead034, f464ca82eb61f4c8fd60efa06365815eb25cec5e1f8ed905cccd242cadacfc2a",
    date: "2026-09-27T16:28:00.787Z, 2026-09-27T18:31:46.471Z",
    adapter: "apple / metal-3 / Apple M2",
  },
};
