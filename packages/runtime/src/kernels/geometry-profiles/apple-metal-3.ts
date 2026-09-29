/**
 * 幾何プロファイル `apple-metal-3`（**生成物 — 手で編集しない**）。
 *
 * tools/geometry-sweep の `profile` が掃引の記録から書いた、adapter `apple / metal-3 / Apple M2` 用のタイル幾何の
 * 静的な表（perf-ledger K-71）。runtime は adapter の `match`（vendor / architecture / description の
 * 完全一致）でこの表を選ぶだけ。
 * 実行時には測らない（オートチューン禁止 — ADR 0022 決定 3）。値を変えるときは掃引を取り直して
 * 下のコマンドで再生成する。
 *
 * 再生成（リポ直下から・`--check` を足すと再生成とバイト同一かだけを見る）:
 *
 *   deno run -A tools/geometry-sweep/main.ts profile \
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-09-27T16-28-00.787Z.json \
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json \
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-09-29T13-08-11.329Z.json \
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json \
 *     --id apple-metal-3 --vendor apple --architecture metal-3 --description 'Apple M2' \
 *     --out packages/runtime/src/kernels/geometry-profiles/apple-metal-3.ts --min-speedup 1.05
 *
 * 掃引（adapter apple / metal-3 / Apple M2）:
 *
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-27T16-28-00.787Z.json（sha256 921e89ba70b34ef68ec21c153c2a2c313c80bbbc98d53da720942c2919ead034・2026-09-27T16:28:00.787Z）
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json（sha256 f464ca82eb61f4c8fd60efa06365815eb25cec5e1f8ed905cccd242cadacfc2a・2026-09-27T18:31:46.471Z）
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-29T13-08-11.329Z.json（sha256 38d97dac5f04a7e80fb6591ab437da7dd2b3cee66b0d03e40cd1475c0f398550・2026-09-29T13:08:11.329Z）
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json（sha256 d9e5e7814370c741562819474f5e0837012d05f233869688d28bb91902ac8621・2026-09-29T21:04:14.586Z）
 *
 * 採否の基準: クラスの全ケースで出力が既定と一致し、既定比が ×1.050 以上の幾何のうち、
 * ケース間の幾何平均が最大のもの。無ければ既定（掃引の既定の行の幾何）。同じケースを複数の掃引が
 * 測っていれば、比はその観測の幾何平均。gemmRows は掃引にある linear / matmul / bmm のケースで決め、3 経路に同じ表が効く。
 * 材料の門: 掃引ごとに、既定の再測定比（cases[].defaultRepeat.driftRatio）が 0.9〜1.1 の外か、
 * 再測定が失敗 / 無いケースはその掃引の比の材料から外す（出力の一致と失敗は見る — 外した掃引で不一致 /
 * 失敗の幾何は採らない。比は同じケースを他の掃引が測っていればそちらで判定し、どの掃引にも残らなければ
 * 測っていない扱い）。外したケースは採否の欄ごとに「掃引 …」の行で示す。
 *
 * 採否:
 *
 * - gemmRows[0]（linear / matmul / bmm の行数 ≤ 64・6 ケース）: 既定 reg16x16r1x4w4 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - 掃引 outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json: bmm-b16-m64-n128-k64 は既定の再測定比 ×0.800 が範囲外のため比の材料から外した（出力の一致と失敗は見る）
 *   - reg128x128r8x8w16: bmm-b16-m64-n64-k128 で ×0.188 < ×1.050
 *   - reg128x16r8x4w4: linear-m16-n3072-k1024 で ×0.276 < ×1.050
 *   - reg128x32r8x4w8: linear-m16-n3072-k1024 で ×0.351 < ×1.050
 *   - reg128x32r8x8w4: linear-m16-n3072-k1024 で ×0.285 < ×1.050
 *   - reg128x64r8x4w16: linear-m16-n3072-k1024 で ×0.340 < ×1.050
 *   - reg128x64r8x8w8: linear-m16-n3072-k1024 で ×0.297 < ×1.050
 *   - reg16x128r4x8w16: linear-m16-n3072-k1024 で ×0.438 < ×1.050
 *   - reg16x16r2x4w4: bmm-b16-m64-n64-k128 で ×0.800 < ×1.050
 *   - reg16x16r4x4w4: linear-m16-n3072-k1024 で ×0.391 < ×1.050
 *   - reg16x32r1x8w4: bmm-b16-m64-n64-k128 で ×0.889 < ×1.050
 *   - reg16x32r2x4w8: bmm-b16-m64-n64-k128 で ×1.045 < ×1.050
 *   - reg16x32r2x8w4: linear-m64-n3072-k1024 で ×0.678 < ×1.050
 *   - reg16x32r4x4w8: linear-m64-n3072-k1024 で ×0.694 < ×1.050
 *   - reg16x32r4x8w4: linear-m64-n3072-k1024 で ×0.371 < ×1.050
 *   - reg16x64r2x8w8: linear-m32-n3072-k1024 で ×0.694 < ×1.050
 *   - reg16x64r4x4w16: linear-m32-n3072-k1024 で ×0.710 < ×1.050
 *   - reg16x64r4x8w8: linear-m32-n3072-k1024 で ×0.389 < ×1.050
 *   - reg32x128r4x8w16: bmm-b16-m64-n64-k128 で ×0.641 < ×1.050
 *   - reg32x128r8x8w16: linear-m16-n3072-k1024 で ×0.361 < ×1.050
 *   - reg32x16r2x4w4: linear-m16-n3072-k1024 で ×0.851 < ×1.050
 *   - reg32x16r4x4w4: linear-m16-n3072-k1024 で ×0.590 < ×1.050
 *   - reg32x16r8x4w4: linear-m16-n3072-k1024 で ×0.328 < ×1.050
 *   - reg32x32r2x4w8: linear-m16-n3072-k1024 で ×0.931 < ×1.050
 *   - reg32x32r2x8w4: linear-m16-n3072-k1024 で ×0.905 < ×1.050
 *   - reg32x32r4x4w8: linear-m16-n3072-k1024 で ×0.957 < ×1.050
 *   - reg32x32r4x8w4: linear-m16-n3072-k1024 で ×0.660 < ×1.050
 *   - reg32x32r8x4w8: linear-m16-n3072-k1024 で ×0.649 < ×1.050
 *   - reg32x32r8x8w4: linear-m16-n3072-k1024 で ×0.256 < ×1.050
 *   - reg32x64r2x8w8: linear-m16-n3072-k1024 で ×0.965 < ×1.050
 *   - reg32x64r4x4w16: linear-m16-n3072-k1024 で ×0.992 < ×1.050
 *   - reg32x64r4x8w8: linear-m16-n3072-k1024 で ×0.680 < ×1.050
 *   - reg32x64r8x4w16: linear-m16-n3072-k1024 で ×0.663 < ×1.050
 *   - reg32x64r8x8w8: linear-m16-n3072-k1024 で ×0.350 < ×1.050
 *   - reg4x16r1x4w4: linear-m32-n3072-k1024 で ×0.183 < ×1.050
 *   - reg4x32r1x8w4: linear-m16-n3072-k1024 で ×0.120 < ×1.050
 *   - reg64x128r4x8w16: bmm-b16-m64-n64-k128 で ×0.462 < ×1.050
 *   - reg64x128r8x8w16: bmm-b16-m64-n64-k128 で ×0.412 < ×1.050
 *   - reg64x16r4x4w4: linear-m16-n3072-k1024 で ×0.500 < ×1.050
 *   - reg64x16r8x4w4: linear-m16-n3072-k1024 で ×0.326 < ×1.050
 *   - reg64x32r4x4w8: linear-m16-n3072-k1024 で ×0.639 < ×1.050
 *   - reg64x32r4x8w4: linear-m16-n3072-k1024 で ×0.497 < ×1.050
 *   - reg64x32r8x4w8: linear-m16-n3072-k1024 で ×0.490 < ×1.050
 *   - reg64x32r8x8w4: linear-m16-n3072-k1024 で ×0.274 < ×1.050
 *   - reg64x64r4x4w16: linear-m16-n3072-k1024 で ×0.649 < ×1.050
 *   - reg64x64r4x8w8: linear-m16-n3072-k1024 で ×0.518 < ×1.050
 *   - reg64x64r8x4w16: linear-m16-n3072-k1024 で ×0.503 < ×1.050
 *   - reg64x64r8x8w8: linear-m16-n3072-k1024 で ×0.413 < ×1.050
 *   - reg8x16r1x4w4: linear-m32-n3072-k1024 で ×0.556 < ×1.050
 *   - reg8x16r2x4w4: linear-m16-n3072-k1024 で ×0.287 < ×1.050
 *   - reg8x32r1x8w4: linear-m32-n3072-k1024 で ×0.380 < ×1.050
 *   - reg8x32r2x4w8: linear-m32-n3072-k1024 で ×0.399 < ×1.050
 *   - reg8x32r2x8w4: linear-m32-n3072-k1024 で ×0.206 < ×1.050
 *   - reg8x64r2x8w8: linear-m16-n3072-k1024 で ×0.233 < ×1.050
 * - gemmRows[1]（linear / matmul / bmm の行数 65〜512・7 ケース）: 既定 reg64x32r4x4w8 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - 掃引 outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json: bmm-b16-m512-n512-k64 は既定の再測定比 ×1.176 が範囲外のため比の材料から外した（出力の一致と失敗は見る）
 *   - reg128x128r8x8w16: bmm-b16-m512-n64-k512 で ×0.357 < ×1.050
 *   - reg128x16r8x4w4: bmm-b16-m512-n64-k64 で ×0.678 < ×1.050
 *   - reg128x32r8x4w8: bmm-b16-m512-n64-k512 で ×0.983 < ×1.050
 *   - reg128x32r8x8w4: linear-m128-n2048-k1024 で ×0.688 < ×1.050
 *   - reg128x64r8x4w16: bmm-b16-m512-n512-k64 で ×0.939 < ×1.050
 *   - reg128x64r8x8w8: linear-m128-n2048-k1024 で ×0.759 < ×1.050
 *   - reg16x128r4x8w16: linear-m256-n2048-k1024 で ×0.291 < ×1.050
 *   - reg16x16r1x4w4: linear-m512-n2048-k1024 で ×0.432 < ×1.050
 *   - reg16x16r2x4w4: linear-m512-n2048-k1024 で ×0.453 < ×1.050
 *   - reg16x16r4x4w4: linear-m512-n2048-k1024 で ×0.225 < ×1.050
 *   - reg16x32r1x8w4: linear-m256-n2048-k1024 で ×0.453 < ×1.050
 *   - reg16x32r2x4w8: linear-m512-n2048-k1024 で ×0.484 < ×1.050
 *   - reg16x32r2x8w4: linear-m128-n2048-k1024 で ×0.331 < ×1.050
 *   - reg16x32r4x4w8: linear-m128-n2048-k1024 で ×0.340 < ×1.050
 *   - reg16x32r4x8w4: linear-m256-n2048-k1024 で ×0.172 < ×1.050
 *   - reg16x64r2x8w8: linear-m128-n2048-k1024 で ×0.405 < ×1.050
 *   - reg16x64r4x4w16: linear-m128-n2048-k1024 で ×0.418 < ×1.050
 *   - reg16x64r4x8w8: linear-m128-n2048-k1024 で ×0.237 < ×1.050
 *   - reg32x128r4x8w16: bmm-b16-m512-n64-k512 で ×0.488 < ×1.050
 *   - reg32x128r8x8w16: linear-m128-n2048-k1024 で ×0.384 < ×1.050
 *   - reg32x16r2x4w4: linear-m512-n2048-k1024 で ×0.657 < ×1.050
 *   - reg32x16r4x4w4: linear-m128-n2048-k1024 で ×0.494 < ×1.050
 *   - reg32x16r8x4w4: linear-m128-n2048-k1024 で ×0.274 < ×1.050
 *   - reg32x32r2x4w8: linear-m256-n2048-k1024 で ×0.735 < ×1.050
 *   - reg32x32r2x8w4: linear-m128-n2048-k1024 で ×0.707 < ×1.050
 *   - reg32x32r4x4w8: linear-m128-n2048-k1024 で ×0.756 < ×1.050
 *   - reg32x32r4x8w4: linear-m128-n2048-k1024 で ×0.486 < ×1.050
 *   - reg32x32r8x4w8: linear-m128-n2048-k1024 で ×0.480 < ×1.050
 *   - reg32x32r8x8w4: linear-m256-n2048-k1024 で ×0.216 < ×1.050
 *   - reg32x64r2x8w8: linear-m256-n2048-k1024 で ×0.699 < ×1.050
 *   - reg32x64r4x4w16: linear-m128-n2048-k1024 で ×0.772 < ×1.050
 *   - reg32x64r4x8w8: linear-m128-n2048-k1024 で ×0.529 < ×1.050
 *   - reg32x64r8x4w16: linear-m256-n2048-k1024 で ×0.647 < ×1.050
 *   - reg32x64r8x8w8: linear-m256-n2048-k1024 で ×0.347 < ×1.050
 *   - reg4x16r1x4w4: linear-m512-n2048-k1024 で ×0.081 < ×1.050
 *   - reg4x32r1x8w4: linear-m512-n2048-k1024 で ×0.057 < ×1.050
 *   - reg64x128r4x8w16: bmm-b16-m512-n64-k512 で ×0.488 < ×1.050
 *   - reg64x128r8x8w16: bmm-b16-m512-n64-k512 で ×0.433 < ×1.050
 *   - reg64x16r4x4w4: bmm-b16-m512-n64-k512 で ×0.686 < ×1.050
 *   - reg64x16r8x4w4: linear-m128-n2048-k1024 で ×0.475 < ×1.050
 *   - reg64x32r4x8w4: linear-m256-n2048-k1024 で ×0.869 < ×1.050
 *   - reg64x32r8x4w8: linear-m512-n2048-k1024 で ×0.877 < ×1.050
 *   - reg64x32r8x8w4: linear-m256-n2048-k1024 で ×0.493 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m512-n64-k512 で ×0.896 < ×1.050
 *   - reg64x64r4x8w8: linear-m512-n2048-k1024 で ×0.900 < ×1.050
 *   - reg64x64r8x4w16: linear-m256-n2048-k1024 で ×0.923 < ×1.050
 *   - reg64x64r8x8w8: linear-m128-n2048-k1024 で ×0.574 < ×1.050
 *   - reg8x16r1x4w4: linear-m256-n2048-k1024 で ×0.268 < ×1.050
 *   - reg8x16r2x4w4: linear-m256-n2048-k1024 で ×0.139 < ×1.050
 *   - reg8x32r1x8w4: linear-m256-n2048-k1024 で ×0.198 < ×1.050
 *   - reg8x32r2x4w8: linear-m256-n2048-k1024 で ×0.210 < ×1.050
 *   - reg8x32r2x8w4: linear-m256-n2048-k1024 で ×0.108 < ×1.050
 *   - reg8x64r2x8w8: linear-m256-n2048-k1024 で ×0.142 < ×1.050
 * - gemmRows[2]（linear / matmul / bmm の行数 > 512・7 ケース）: 採用 reg128x32r8x4w8 ×1.647（×1.171〜×1.765）
 *   - 掃引 outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json: matmul-m4096-n2048-k2048 は既定の再測定比 ×1.160 が範囲外のため比の材料から外した（出力の一致と失敗は見る）
 *   - reg128x16r8x4w4: matmul-m4096-n2048-k2048 で ×0.707 < ×1.050
 *   - reg128x32r8x8w4: matmul-m4096-n2048-k2048 で ×0.914 < ×1.050
 *   - reg128x64r8x4w16: 幾何平均 ×1.494（採用 ×1.647 に届かない）
 *   - reg128x64r8x8w8: matmul-m4096-n2048-k2048 で ×0.958 < ×1.050
 *   - reg16x128r4x8w16: linear-m1024-n2048-k8192 で ×0.478 < ×1.050
 *   - reg16x16r1x4w4: linear-m1024-n2048-k8192 で ×0.671 < ×1.050
 *   - reg16x16r2x4w4: linear-m1024-n2048-k8192 で ×0.695 < ×1.050
 *   - reg16x16r4x4w4: linear-m1024-n2048-k8192 で ×0.337 < ×1.050
 *   - reg16x32r1x8w4: linear-m1024-n2048-k8192 で ×0.717 < ×1.050
 *   - reg16x32r2x4w8: linear-m1024-n2048-k8192 で ×0.746 < ×1.050
 *   - reg16x32r2x8w4: linear-m1024-n2048-k8192 で ×0.546 < ×1.050
 *   - reg16x32r4x4w8: linear-m1024-n2048-k8192 で ×0.561 < ×1.050
 *   - reg16x32r4x8w4: linear-m1024-n2048-k8192 で ×0.271 < ×1.050
 *   - reg16x64r2x8w8: linear-m1024-n2048-k8192 で ×0.674 < ×1.050
 *   - reg16x64r4x4w16: linear-m1024-n2048-k8192 で ×0.682 < ×1.050
 *   - reg16x64r4x8w8: linear-m1024-n2048-k8192 で ×0.409 < ×1.050
 *   - reg32x128r4x8w16: linear-m1024-n2048-k8192 で ×0.969 < ×1.050
 *   - reg32x128r8x8w16: linear-m1024-n2048-k8192 で ×0.765 < ×1.050
 *   - reg32x16r2x4w4: linear-m1024-n2048-k8192 で ×1.009 < ×1.050
 *   - reg32x16r4x4w4: linear-m1024-n2048-k8192 で ×0.798 < ×1.050
 *   - reg32x16r8x4w4: linear-m1024-n2048-k8192 で ×0.439 < ×1.050
 *   - reg32x32r2x4w8: 幾何平均 ×1.157（採用 ×1.647 に届かない）
 *   - reg32x32r2x8w4: 幾何平均 ×1.108（採用 ×1.647 に届かない）
 *   - reg32x32r4x4w8: 幾何平均 ×1.182（採用 ×1.647 に届かない）
 *   - reg32x32r4x8w4: linear-m1024-n2048-k8192 で ×0.904 < ×1.050
 *   - reg32x32r8x4w8: linear-m1024-n2048-k8192 で ×0.907 < ×1.050
 *   - reg32x32r8x8w4: linear-m1024-n2048-k8192 で ×0.347 < ×1.050
 *   - reg32x64r2x8w8: 幾何平均 ×1.131（採用 ×1.647 に届かない）
 *   - reg32x64r4x4w16: 幾何平均 ×1.240（採用 ×1.647 に届かない）
 *   - reg32x64r4x8w8: linear-m1024-n2048-k8192 で ×0.967 < ×1.050
 *   - reg32x64r8x4w16: linear-m1024-n2048-k8192 で ×1.036 < ×1.050
 *   - reg32x64r8x8w8: linear-m1024-n2048-k8192 で ×0.636 < ×1.050
 *   - reg4x16r1x4w4: linear-m1024-n2048-k8192 で ×0.123 < ×1.050
 *   - reg4x32r1x8w4: linear-m1024-n2048-k8192 で ×0.087 < ×1.050
 *   - reg64x128r4x8w16: 幾何平均 ×1.283（採用 ×1.647 に届かない）
 *   - reg64x128r8x8w16: matmul-m4096-n2048-k2048 で ×0.952 < ×1.050
 *   - reg64x16r4x4w4: matmul-m4096-n2048-k2048 で ×0.978 < ×1.050
 *   - reg64x16r8x4w4: matmul-m4096-n2048-k2048 で ×0.625 < ×1.050
 *   - reg64x32r4x4w8: 幾何平均 ×1.511（採用 ×1.647 に届かない）
 *   - reg64x32r4x8w4: 幾何平均 ×1.373（採用 ×1.647 に届かない）
 *   - reg64x32r8x4w8: 幾何平均 ×1.391（採用 ×1.647 に届かない）
 *   - reg64x32r8x8w4: matmul-m4096-n2048-k2048 で ×0.792 < ×1.050
 *   - reg64x64r4x4w16: 幾何平均 ×1.474（採用 ×1.647 に届かない）
 *   - reg64x64r4x8w8: 幾何平均 ×1.365（採用 ×1.647 に届かない）
 *   - reg64x64r8x4w16: 幾何平均 ×1.503（採用 ×1.647 に届かない）
 *   - reg64x64r8x8w8: matmul-m4096-n2048-k2048 で ×1.013 < ×1.050
 *   - reg8x16r1x4w4: linear-m1024-n2048-k2048 で ×0.403 < ×1.050
 *   - reg8x16r2x4w4: linear-m1024-n2048-k8192 で ×0.211 < ×1.050
 *   - reg8x32r1x8w4: linear-m1024-n2048-k8192 で ×0.306 < ×1.050
 *   - reg8x32r2x4w8: linear-m1024-n2048-k8192 で ×0.328 < ×1.050
 *   - reg8x32r2x8w4: linear-m1024-n2048-k8192 で ×0.166 < ×1.050
 *   - reg8x64r2x8w8: linear-m1024-n2048-k8192 で ×0.216 < ×1.050
 * - attention.qk（融合 attention f32 ①QK・4 ケース）: 採用 reg128x32r8x4w8 ×1.754（×1.664〜×1.789）
 *   - reg128x16r8x4w4: 幾何平均 ×1.160（採用 ×1.754 に届かない）
 *   - reg128x32r8x8w4: 幾何平均 ×1.314（採用 ×1.754 に届かない）
 *   - reg128x64r8x4w16: 幾何平均 ×1.603（採用 ×1.754 に届かない）
 *   - reg128x64r8x8w8: 幾何平均 ×1.383（採用 ×1.754 に届かない）
 *   - reg16x128r4x8w16: attention-qk-cross-m4096-n512 で ×0.498 < ×1.050
 *   - reg16x16r1x4w4: attention-qk-cross-m4096-n512 で ×0.647 < ×1.050
 *   - reg16x16r2x4w4: attention-qk-cross-m4096-n512 で ×0.706 < ×1.050
 *   - reg16x16r4x4w4: attention-qk-cross-m4096-n512 で ×0.382 < ×1.050
 *   - reg16x32r1x8w4: attention-qk-cross-m4096-n512 で ×0.725 < ×1.050
 *   - reg16x32r2x4w8: attention-qk-cross-m4096-n512 で ×0.762 < ×1.050
 *   - reg16x32r2x8w4: attention-qk-cross-m4096-n512 で ×0.582 < ×1.050
 *   - reg16x32r4x4w8: attention-qk-cross-m4096-n512 で ×0.609 < ×1.050
 *   - reg16x32r4x8w4: attention-qk-cross-m4096-n512 で ×0.282 < ×1.050
 *   - reg16x64r2x8w8: attention-qk-cross-m4096-n512 で ×0.707 < ×1.050
 *   - reg16x64r4x4w16: attention-qk-cross-m4096-n512 で ×0.698 < ×1.050
 *   - reg16x64r4x8w8: attention-qk-cross-m4096-n512 で ×0.422 < ×1.050
 *   - reg32x128r4x8w16: attention-qk-cross-m4096-n512 で ×1.028 < ×1.050
 *   - reg32x128r8x8w16: attention-qk-self-m4096-n4096 で ×0.794 < ×1.050
 *   - reg32x16r2x4w4: attention-qk-cross-m4096-n512 で ×0.997 < ×1.050
 *   - reg32x16r4x4w4: attention-qk-cross-m4096-n512 で ×0.841 < ×1.050
 *   - reg32x16r8x4w4: attention-qk-cross-m4096-n512 で ×0.474 < ×1.050
 *   - reg32x32r2x4w8: 幾何平均 ×1.167（採用 ×1.754 に届かない）
 *   - reg32x32r2x8w4: 幾何平均 ×1.144（採用 ×1.754 に届かない）
 *   - reg32x32r4x4w8: 幾何平均 ×1.200（採用 ×1.754 に届かない）
 *   - reg32x32r4x8w4: attention-qk-cross-m4096-n512 で ×0.943 < ×1.050
 *   - reg32x32r8x4w8: attention-qk-cross-m4096-n512 で ×0.973 < ×1.050
 *   - reg32x32r8x8w4: attention-qk-cross-m4096-n512 で ×0.375 < ×1.050
 *   - reg32x64r2x8w8: 幾何平均 ×1.146（採用 ×1.754 に届かない）
 *   - reg32x64r4x4w16: 幾何平均 ×1.252（採用 ×1.754 に届かない）
 *   - reg32x64r4x8w8: 幾何平均 ×1.086（採用 ×1.754 に届かない）
 *   - reg32x64r8x4w16: 幾何平均 ×1.095（採用 ×1.754 に届かない）
 *   - reg32x64r8x8w8: attention-qk-cross-m4096-n512 で ×0.678 < ×1.050
 *   - reg4x16r1x4w4: attention-qk-cross-m4096-n512 で ×0.133 < ×1.050
 *   - reg4x32r1x8w4: attention-qk-cross-m4096-n512 で ×0.093 < ×1.050
 *   - reg64x128r4x8w16: 幾何平均 ×1.284（採用 ×1.754 に届かない）
 *   - reg64x128r8x8w16: 幾何平均 ×1.158（採用 ×1.754 に届かない）
 *   - reg64x16r4x4w4: 幾何平均 ×1.209（採用 ×1.754 に届かない）
 *   - reg64x16r8x4w4: attention-qk-cross-m4096-n512 で ×0.826 < ×1.050
 *   - reg64x32r4x4w8: 幾何平均 ×1.558（採用 ×1.754 に届かない）
 *   - reg64x32r4x8w4: 幾何平均 ×1.418（採用 ×1.754 に届かない）
 *   - reg64x32r8x4w8: 幾何平均 ×1.430（採用 ×1.754 に届かない）
 *   - reg64x32r8x8w4: attention-qk-cross-m4096-n512 で ×0.927 < ×1.050
 *   - reg64x64r4x4w16: 幾何平均 ×1.498（採用 ×1.754 に届かない）
 *   - reg64x64r4x8w8: 幾何平均 ×1.412（採用 ×1.754 に届かない）
 *   - reg64x64r8x4w16: 幾何平均 ×1.560（採用 ×1.754 に届かない）
 *   - reg64x64r8x8w8: 幾何平均 ×1.272（採用 ×1.754 に届かない）
 *   - reg8x16r1x4w4: attention-qk-cross-m4096-n512 で ×0.413 < ×1.050
 *   - reg8x16r2x4w4: attention-qk-cross-m4096-n512 で ×0.222 < ×1.050
 *   - reg8x32r1x8w4: attention-qk-cross-m4096-n512 で ×0.332 < ×1.050
 *   - reg8x32r2x4w8: attention-qk-cross-m4096-n512 で ×0.350 < ×1.050
 *   - reg8x32r2x8w4: attention-qk-cross-m4096-n512 で ×0.180 < ×1.050
 *   - reg8x64r2x8w8: attention-qk-cross-m4096-n512 で ×0.229 < ×1.050
 * - attention.pv（融合 attention f32 ③PV・4 ケース）: 採用 reg32x64r4x8w8 ×1.635（×1.495〜×1.693）
 *   - 掃引 outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json: attention-pv-cross-m1024-n512 は既定の再測定比 ×1.126 が範囲外のため比の材料から外した（出力の一致と失敗は見る）
 *   - 掃引 outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json: attention-pv-self-m1024-n1024 は既定の再測定比 ×1.228 が範囲外のため比の材料から外した（出力の一致と失敗は見る）
 *   - reg128x16r8x4w4: attention-pv-cross-m4096-n512 で ×0.945 < ×1.050
 *   - reg128x32r8x4w8: 幾何平均 ×1.458（採用 ×1.635 に届かない）
 *   - reg128x32r8x8w4: 幾何平均 ×1.154（採用 ×1.635 に届かない）
 *   - reg128x64r8x4w16: 幾何平均 ×1.511（採用 ×1.635 に届かない）
 *   - reg128x64r8x8w8: 幾何平均 ×1.281（採用 ×1.635 に届かない）
 *   - reg16x128r4x8w16: 幾何平均 ×1.173（採用 ×1.635 に届かない）
 *   - reg16x16r1x4w4: attention-pv-cross-m4096-n512 で ×0.988 < ×1.050
 *   - reg16x16r2x4w4: 幾何平均 ×1.307（採用 ×1.635 に届かない）
 *   - reg16x16r4x4w4: attention-pv-cross-m4096-n512 で ×0.651 < ×1.050
 *   - reg16x32r1x8w4: attention-pv-cross-m4096-n512 で ×1.012 < ×1.050
 *   - reg16x32r2x4w8: 幾何平均 ×1.453（採用 ×1.635 に届かない）
 *   - reg16x32r2x8w4: 幾何平均 ×1.147（採用 ×1.635 に届かない）
 *   - reg16x32r4x4w8: 幾何平均 ×1.190（採用 ×1.635 に届かない）
 *   - reg16x32r4x8w4: attention-pv-self-m4096-n4096 で ×0.636 < ×1.050
 *   - reg16x64r2x8w8: 幾何平均 ×1.338（採用 ×1.635 に届かない）
 *   - reg16x64r4x4w16: 幾何平均 ×1.368（採用 ×1.635 に届かない）
 *   - reg16x64r4x8w8: attention-pv-self-m4096-n4096 で ×0.951 < ×1.050
 *   - reg32x128r4x8w16: 幾何平均 ×1.574（採用 ×1.635 に届かない）
 *   - reg32x128r8x8w16: 幾何平均 ×1.294（採用 ×1.635 に届かない）
 *   - reg32x16r2x4w4: 幾何平均 ×1.340（採用 ×1.635 に届かない）
 *   - reg32x16r4x4w4: attention-pv-cross-m4096-n512 で ×1.042 < ×1.050
 *   - reg32x16r8x4w4: attention-pv-self-m4096-n4096 で ×0.561 < ×1.050
 *   - reg32x32r2x4w8: 幾何平均 ×1.477（採用 ×1.635 に届かない）
 *   - reg32x32r2x8w4: 幾何平均 ×1.438（採用 ×1.635 に届かない）
 *   - reg32x32r4x4w8: 幾何平均 ×1.510（採用 ×1.635 に届かない）
 *   - reg32x32r4x8w4: 幾何平均 ×1.461（採用 ×1.635 に届かない）
 *   - reg32x32r8x4w8: 幾何平均 ×1.365（採用 ×1.635 に届かない）
 *   - reg32x32r8x8w4: attention-pv-self-m4096-n4096 で ×0.535 < ×1.050
 *   - reg32x64r2x8w8: 幾何平均 ×1.460（採用 ×1.635 に届かない）
 *   - reg32x64r4x4w16: 幾何平均 ×1.557（採用 ×1.635 に届かない）
 *   - reg32x64r8x4w16: 幾何平均 ×1.501（採用 ×1.635 に届かない）
 *   - reg32x64r8x8w8: 幾何平均 ×1.144（採用 ×1.635 に届かない）
 *   - reg4x16r1x4w4: attention-pv-self-m4096-n4096 で ×0.464 < ×1.050
 *   - reg4x32r1x8w4: attention-pv-self-m4096-n4096 で ×0.296 < ×1.050
 *   - reg64x128r4x8w16: 幾何平均 ×1.531（採用 ×1.635 に届かない）
 *   - reg64x128r8x8w16: 幾何平均 ×1.298（採用 ×1.635 に届かない）
 *   - reg64x16r4x4w4: 幾何平均 ×1.255（採用 ×1.635 に届かない）
 *   - reg64x16r8x4w4: attention-pv-cross-m4096-n512 で ×0.847 < ×1.050
 *   - reg64x32r4x4w8: 幾何平均 ×1.501（採用 ×1.635 に届かない）
 *   - reg64x32r4x8w4: 幾何平均 ×1.579（採用 ×1.635 に届かない）
 *   - reg64x32r8x4w8: 幾何平均 ×1.454（採用 ×1.635 に届かない）
 *   - reg64x32r8x8w4: attention-pv-cross-m4096-n512 で ×0.979 < ×1.050
 *   - reg64x64r4x4w16: 幾何平均 ×1.531（採用 ×1.635 に届かない）
 *   - reg64x64r4x8w8: 幾何平均 ×1.610（採用 ×1.635 に届かない）
 *   - reg64x64r8x4w16: 幾何平均 ×1.632（採用 ×1.635 に届かない）
 *   - reg64x64r8x8w8: 幾何平均 ×1.303（採用 ×1.635 に届かない）
 *   - reg8x16r1x4w4: attention-pv-cross-m4096-n512 で ×0.907 < ×1.050
 *   - reg8x16r2x4w4: attention-pv-cross-m4096-n512 で ×0.624 < ×1.050
 *   - reg8x32r1x8w4: attention-pv-self-m4096-n4096 で ×0.719 < ×1.050
 *   - reg8x32r2x4w8: attention-pv-self-m4096-n4096 で ×0.830 < ×1.050
 *   - reg8x32r2x8w4: attention-pv-self-m4096-n4096 で ×0.456 < ×1.050
 *   - reg8x64r2x8w8: attention-pv-self-m4096-n4096 で ×0.608 < ×1.050
 * - conv2d.rows64（conv2d implicit GEMM の m タイル 64 行・2 ケース）: 採用 igemm128x64:wg16x16 ×1.448（×1.218〜×1.722）
 *   - igemm128x128:wg16x16: conv2d-c192-256x256 で ×0.808 < ×1.050
 *   - igemm128x64:wg8x16: conv2d-c192-256x256 で ×1.039 < ×1.050
 *   - igemm16x128:wg16x4: conv2d-c384-128x128 で ×0.395 < ×1.050
 *   - igemm16x64:wg16x4: conv2d-c384-128x128 で ×0.527 < ×1.050
 *   - igemm16x64:wg8x4: conv2d-c384-128x128 で ×0.357 < ×1.050
 *   - igemm16x64:wg8x8: conv2d-c384-128x128 で ×0.514 < ×1.050
 *   - igemm32x128:wg16x4: conv2d-c384-128x128 で ×0.635 < ×1.050
 *   - igemm32x128:wg16x8: conv2d-c384-128x128 で ×0.784 < ×1.050
 *   - igemm32x64:wg16x4: conv2d-c384-128x128 で ×0.836 < ×1.050
 *   - igemm32x64:wg16x8: conv2d-c384-128x128 で ×0.926 < ×1.050
 *   - igemm32x64:wg8x16: conv2d-c384-128x128 で ×0.940 < ×1.050
 *   - igemm32x64:wg8x4: conv2d-c384-128x128 で ×0.581 < ×1.050
 *   - igemm32x64:wg8x8: conv2d-c384-128x128 で ×0.840 < ×1.050
 *   - igemm64x128:wg16x16: 幾何平均 ×1.234（採用 ×1.448 に届かない）
 *   - igemm64x64:wg16x16: 幾何平均 ×1.393（採用 ×1.448 に届かない）
 *   - igemm64x64:wg16x8: 幾何平均 ×1.298（採用 ×1.448 に届かない）
 *   - igemm64x64:wg8x16: 幾何平均 ×1.278（採用 ×1.448 に届かない）
 *   - igemm64x64:wg8x8: conv2d-c384-128x128 で ×1.016 < ×1.050
 *   - igemm8x64:wg8x4: conv2d-c384-128x128 で ×0.187 < ×1.050
 * - conv2d.rows32（conv2d implicit GEMM の m タイル 32 行・1 ケース）: 採用 igemm128x64:wg16x16 ×1.662（×1.662〜×1.662）
 *   - igemm128x128:wg16x16: 幾何平均 ×1.174（採用 ×1.662 に届かない）
 *   - igemm128x64:wg8x16: 幾何平均 ×1.480（採用 ×1.662 に届かない）
 *   - igemm16x128:wg16x4: conv2d-c96-512x512 で ×0.638 < ×1.050
 *   - igemm16x64:wg16x4: conv2d-c96-512x512 で ×0.848 < ×1.050
 *   - igemm16x64:wg8x4: conv2d-c96-512x512 で ×0.575 < ×1.050
 *   - igemm16x64:wg8x8: conv2d-c96-512x512 で ×0.839 < ×1.050
 *   - igemm32x128:wg16x8: 幾何平均 ×1.276（採用 ×1.662 に届かない）
 *   - igemm32x64:wg16x4: 幾何平均 ×1.352（採用 ×1.662 に届かない）
 *   - igemm32x64:wg16x8: 幾何平均 ×1.390（採用 ×1.662 に届かない）
 *   - igemm32x64:wg8x16: 幾何平均 ×1.376（採用 ×1.662 に届かない）
 *   - igemm32x64:wg8x4: conv2d-c96-512x512 で ×0.910 < ×1.050
 *   - igemm32x64:wg8x8: 幾何平均 ×1.366（採用 ×1.662 に届かない）
 *   - igemm64x128:wg16x16: 幾何平均 ×1.371（採用 ×1.662 に届かない）
 *   - igemm64x128:wg16x8: 幾何平均 ×1.160（採用 ×1.662 に届かない）
 *   - igemm64x64:wg16x16: 幾何平均 ×1.400（採用 ×1.662 に届かない）
 *   - igemm64x64:wg16x8: 幾何平均 ×1.439（採用 ×1.662 に届かない）
 *   - igemm64x64:wg8x16: 幾何平均 ×1.388（採用 ×1.662 に届かない）
 *   - igemm64x64:wg8x8: 幾何平均 ×1.180（採用 ×1.662 に届かない）
 *   - igemm8x64:wg8x4: conv2d-c96-512x512 で ×0.308 < ×1.050
 * - i8a8.linear（i8a8 linear・6 ケース）: 採用 tile64x64r4x8w8x16k16 ×1.128（×1.115〜×1.154）
 *   - tile128x128r8x8w16x16k16: i8a8-linear-m4096-n2048-k8192 で ×0.761 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-linear-m4096-n2048-k8192 で ×0.808 < ×1.050
 *   - tile128x32r8x4w8x16k16: 幾何平均 ×1.122（採用 ×1.128 に届かない）
 *   - tile128x32r8x4w8x16k32: 幾何平均 ×1.102（採用 ×1.128 に届かない）
 *   - tile128x64r8x4w16x16k16: 幾何平均 ×1.098（採用 ×1.128 に届かない）
 *   - tile128x64r8x4w16x16k32: i8a8-linear-m4096-n2048-k8192 で ×0.931 < ×1.050
 *   - tile128x64r8x8w8x16k32: i8a8-linear-m4096-n2048-k8192 で ×1.035 < ×1.050
 *   - tile16x128r4x8w16x4k16: 幾何平均 ×1.115（採用 ×1.128 に届かない）
 *   - tile16x128r4x8w16x4k32: 幾何平均 ×1.100（採用 ×1.128 に届かない）
 *   - tile16x32r4x4w8x4k16: i8a8-linear-m1024-n2048-k2048 で ×0.987 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-linear-m4096-n8192-k2048 で ×0.994 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-linear-m4096-n8192-k2048 で ×0.996 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-linear-m4096-n8192-k2048 で ×0.995 < ×1.050
 *   - tile16x64r4x8w8x4k16: 幾何平均 ×1.117（採用 ×1.128 に届かない）
 *   - tile16x64r4x8w8x4k32: 幾何平均 ×1.096（採用 ×1.128 に届かない）
 *   - tile32x128r4x8w16x8k16: 幾何平均 ×1.122（採用 ×1.128 に届かない）
 *   - tile32x128r4x8w16x8k32: 幾何平均 ×1.100（採用 ×1.128 に届かない）
 *   - tile32x128r8x8w16x4k16: i8a8-linear-m1024-n2048-k8192 で ×0.966 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-linear-m1024-n2048-k8192 で ×0.990 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-linear-m4096-n8192-k2048 で ×0.999 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-linear-m4096-n8192-k2048 で ×0.994 < ×1.050
 *   - tile32x32r8x4w8x4k16: 幾何平均 ×1.121（採用 ×1.128 に届かない）
 *   - tile32x32r8x4w8x4k32: 幾何平均 ×1.114（採用 ×1.128 に届かない）
 *   - tile32x64r4x4w16x8k16: i8a8-linear-m4096-n8192-k2048 で ×0.992 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-linear-m4096-n8192-k2048 で ×0.998 < ×1.050
 *   - tile32x64r4x8w8x8k16: 幾何平均 ×1.125（採用 ×1.128 に届かない）
 *   - tile32x64r4x8w8x8k32: 幾何平均 ×1.120（採用 ×1.128 に届かない）
 *   - tile32x64r8x4w16x4k16: 幾何平均 ×1.123（採用 ×1.128 に届かない）
 *   - tile32x64r8x4w16x4k32: 幾何平均 ×1.118（採用 ×1.128 に届かない）
 *   - tile32x64r8x8w8x4k16: i8a8-linear-m1024-n2048-k8192 で ×0.961 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-linear-m4096-n2048-k8192 で ×0.957 < ×1.050
 *   - tile64x128r4x8w16x16k16: 幾何平均 ×1.097（採用 ×1.128 に届かない）
 *   - tile64x128r4x8w16x16k32: i8a8-linear-m4096-n2048-k8192 で ×0.924 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m1024-n2048-k8192 で ×0.975 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-linear-m4096-n2048-k8192 で ×1.018 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-linear-m4096-n8192-k2048 で ×0.998 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-linear-m4096-n8192-k2048 で ×0.999 < ×1.050
 *   - tile64x32r8x4w8x8k16: 幾何平均 ×1.123（採用 ×1.128 に届かない）
 *   - tile64x32r8x4w8x8k32: 幾何平均 ×1.119（採用 ×1.128 に届かない）
 *   - tile64x64r4x4w16x16k16: i8a8-linear-m4096-n8192-k2048 で ×0.965 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-linear-m4096-n8192-k2048 で ×0.975 < ×1.050
 *   - tile64x64r4x8w8x16k32: 幾何平均 ×1.113（採用 ×1.128 に届かない）
 *   - tile64x64r8x4w16x8k16: 幾何平均 ×1.119（採用 ×1.128 に届かない）
 *   - tile64x64r8x4w16x8k32: 幾何平均 ×1.111（採用 ×1.128 に届かない）
 *   - tile64x64r8x8w8x8k16: i8a8-linear-m1024-n2048-k8192 で ×1.015 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-linear-m1024-n2048-k8192 で ×1.033 < ×1.050
 * - i8a8.attentionQk（i8a8 attention ①QK・4 ケース）: 採用 tile32x64r4x8w8x8k16 ×1.156（×1.154〜×1.157）
 *   - tile128x128r8x8w16x16k16: i8a8-attention-qk-cross-m1024-n512 で ×0.727 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-attention-qk-self-m1024-n1024 で ×0.815 < ×1.050
 *   - tile128x32r8x4w8x16k16: 幾何平均 ×1.116（採用 ×1.156 に届かない）
 *   - tile128x32r8x4w8x16k32: 幾何平均 ×1.121（採用 ×1.156 に届かない）
 *   - tile128x64r8x4w16x16k16: 幾何平均 ×1.119（採用 ×1.156 に届かない）
 *   - tile128x64r8x4w16x16k32: i8a8-attention-qk-cross-m4096-n512 で ×0.938 < ×1.050
 *   - tile128x64r8x8w8x16k32: i8a8-attention-qk-cross-m1024-n512 で ×1.004 < ×1.050
 *   - tile16x128r4x8w16x4k16: 幾何平均 ×1.145（採用 ×1.156 に届かない）
 *   - tile16x128r4x8w16x4k32: 幾何平均 ×1.116（採用 ×1.156 に届かない）
 *   - tile16x32r4x4w8x4k16: i8a8-attention-qk-cross-m4096-n512 で ×1.023 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-attention-qk-cross-m4096-n512 で ×1.007 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-attention-qk-cross-m4096-n512 で ×1.029 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-attention-qk-cross-m4096-n512 で ×1.018 < ×1.050
 *   - tile16x64r4x8w8x4k16: 幾何平均 ×1.135（採用 ×1.156 に届かない）
 *   - tile16x64r4x8w8x4k32: 幾何平均 ×1.099（採用 ×1.156 に届かない）
 *   - tile32x128r4x8w16x8k16: 幾何平均 ×1.114（採用 ×1.156 に届かない）
 *   - tile32x128r4x8w16x8k32: 幾何平均 ×1.122（採用 ×1.156 に届かない）
 *   - tile32x128r8x8w16x4k16: i8a8-attention-qk-cross-m4096-n512 で ×0.952 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-attention-qk-cross-m4096-n512 で ×0.937 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-attention-qk-cross-m4096-n512 で ×1.031 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-attention-qk-cross-m4096-n512 で ×1.027 < ×1.050
 *   - tile32x32r8x4w8x4k16: 幾何平均 ×1.152（採用 ×1.156 に届かない）
 *   - tile32x32r8x4w8x4k32: 幾何平均 ×1.140（採用 ×1.156 に届かない）
 *   - tile32x64r4x4w16x8k16: i8a8-attention-qk-cross-m4096-n512 で ×1.023 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-attention-qk-cross-m4096-n512 で ×0.992 < ×1.050
 *   - tile32x64r4x8w8x8k32: 幾何平均 ×1.145（採用 ×1.156 に届かない）
 *   - tile32x64r8x4w16x4k16: 幾何平均 ×1.153（採用 ×1.156 に届かない）
 *   - tile32x64r8x4w16x4k32: 幾何平均 ×1.142（採用 ×1.156 に届かない）
 *   - tile32x64r8x8w8x4k16: i8a8-attention-qk-cross-m4096-n512 で ×0.944 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-attention-qk-self-m4096-n4096 で ×0.911 < ×1.050
 *   - tile64x128r4x8w16x16k16: 幾何平均 ×1.118（採用 ×1.156 に届かない）
 *   - tile64x128r4x8w16x16k32: i8a8-attention-qk-cross-m4096-n512 で ×0.950 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-attention-qk-cross-m1024-n512 で ×0.998 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-attention-qk-cross-m1024-n512 で ×0.998 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-attention-qk-cross-m4096-n512 で ×1.025 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-attention-qk-cross-m4096-n512 で ×0.992 < ×1.050
 *   - tile64x32r8x4w8x8k16: 幾何平均 ×1.153（採用 ×1.156 に届かない）
 *   - tile64x32r8x4w8x8k32: 幾何平均 ×1.142（採用 ×1.156 に届かない）
 *   - tile64x64r4x4w16x16k16: i8a8-attention-qk-cross-m4096-n512 で ×0.988 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-attention-qk-cross-m4096-n512 で ×0.989 < ×1.050
 *   - tile64x64r4x8w8x16k16: 幾何平均 ×1.129（採用 ×1.156 に届かない）
 *   - tile64x64r4x8w8x16k32: 幾何平均 ×1.137（採用 ×1.156 に届かない）
 *   - tile64x64r8x4w16x8k16: 幾何平均 ×1.128（採用 ×1.156 に届かない）
 *   - tile64x64r8x4w16x8k32: 幾何平均 ×1.137（採用 ×1.156 に届かない）
 *   - tile64x64r8x8w8x8k16: i8a8-attention-qk-self-m4096-n4096 で ×1.012 < ×1.050
 *   - tile64x64r8x8w8x8k32: 幾何平均 ×1.072（採用 ×1.156 に届かない）
 * - i8a8.attentionPv（i8a8 attention ③PV・4 ケース）: 採用 tile16x128r4x8w16x4k16 ×1.133（×1.121〜×1.144）
 *   - tile128x128r8x8w16x16k16: i8a8-attention-pv-cross-m4096-n512 で ×0.747 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-attention-pv-cross-m4096-n512 で ×0.799 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-attention-pv-self-m4096-n4096 で ×1.029 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-attention-pv-self-m4096-n4096 で ×1.028 < ×1.050
 *   - tile128x64r8x4w16x16k16: 幾何平均 ×1.087（採用 ×1.133 に届かない）
 *   - tile128x64r8x4w16x16k32: i8a8-attention-pv-cross-m4096-n512 で ×0.901 < ×1.050
 *   - tile128x64r8x8w8x16k16: i8a8-attention-pv-cross-m4096-n512 で ×0.975 < ×1.050
 *   - tile128x64r8x8w8x16k32: i8a8-attention-pv-cross-m4096-n512 で ×1.033 < ×1.050
 *   - tile16x128r4x8w16x4k32: 幾何平均 ×1.112（採用 ×1.133 に届かない）
 *   - tile16x32r4x4w8x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.948 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.939 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.992 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.990 < ×1.050
 *   - tile16x64r4x8w8x4k16: 幾何平均 ×1.098（採用 ×1.133 に届かない）
 *   - tile16x64r4x8w8x4k32: 幾何平均 ×1.078（採用 ×1.133 に届かない）
 *   - tile32x128r4x8w16x8k16: 幾何平均 ×1.114（採用 ×1.133 に届かない）
 *   - tile32x128r4x8w16x8k32: 幾何平均 ×1.114（採用 ×1.133 に届かない）
 *   - tile32x128r8x8w16x4k16: i8a8-attention-pv-cross-m1024-n512 で ×0.980 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-attention-pv-cross-m4096-n512 で ×1.008 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.956 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-attention-pv-self-m4096-n4096 で ×0.950 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-attention-pv-self-m4096-n4096 で ×1.045 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-attention-pv-self-m4096-n4096 で ×1.033 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.987 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-attention-pv-cross-m4096-n512 で ×0.968 < ×1.050
 *   - tile32x64r4x8w8x8k16: 幾何平均 ×1.116（採用 ×1.133 に届かない）
 *   - tile32x64r4x8w8x8k32: 幾何平均 ×1.109（採用 ×1.133 に届かない）
 *   - tile32x64r8x4w16x4k16: 幾何平均 ×1.114（採用 ×1.133 に届かない）
 *   - tile32x64r8x4w16x4k32: 幾何平均 ×1.104（採用 ×1.133 に届かない）
 *   - tile32x64r8x8w8x4k16: i8a8-attention-pv-cross-m1024-n512 で ×0.979 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-attention-pv-cross-m4096-n512 で ×0.937 < ×1.050
 *   - tile64x128r4x8w16x16k16: i8a8-attention-pv-self-m4096-n4096 で ×0.938 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-attention-pv-self-m4096-n4096 で ×0.943 < ×1.050
 *   - tile64x128r8x8w16x8k32: 幾何平均 ×1.060（採用 ×1.133 に届かない）
 *   - tile64x32r4x4w8x16k16: i8a8-attention-pv-self-m4096-n4096 で ×0.952 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-attention-pv-self-m4096-n4096 で ×0.929 < ×1.050
 *   - tile64x32r8x4w8x8k16: 幾何平均 ×1.065（採用 ×1.133 に届かない）
 *   - tile64x32r8x4w8x8k32: i8a8-attention-pv-self-m4096-n4096 で ×1.044 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-attention-pv-self-m4096-n4096 で ×0.964 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-attention-pv-self-m4096-n4096 で ×0.967 < ×1.050
 *   - tile64x64r4x8w8x16k16: 幾何平均 ×1.099（採用 ×1.133 に届かない）
 *   - tile64x64r4x8w8x16k32: 幾何平均 ×1.099（採用 ×1.133 に届かない）
 *   - tile64x64r8x4w16x8k16: 幾何平均 ×1.097（採用 ×1.133 に届かない）
 *   - tile64x64r8x4w16x8k32: 幾何平均 ×1.096（採用 ×1.133 に届かない）
 *   - tile64x64r8x8w8x8k16: i8a8-attention-pv-cross-m1024-n512 で ×0.988 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-attention-pv-cross-m1024-n512 で ×0.998 < ×1.050
 */
import type { GeometryProfile } from "../geometry-profile.ts";

export const APPLE_METAL_3: GeometryProfile = {
  id: "apple-metal-3",
  match: { vendor: "apple", architecture: "metal-3", description: "Apple M2" },
  gemmRows: [
    { maxRows: 64, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 512, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: Number.POSITIVE_INFINITY, geometry: { regM: 8, regN: 4, wgX: 8, wgY: 16 } },
  ],
  attention: {
    qk: { regM: 8, regN: 4, wgX: 8, wgY: 16 },
    pv: { regM: 4, regN: 8, wgX: 8, wgY: 8 },
  },
  conv2d: {
    rows64: { regM: 8, regN: 4, wgX: 16, wgY: 16 },
    rows32: { regM: 8, regN: 4, wgX: 16, wgY: 16 },
  },
  i8a8: {
    linear: { regM: 4, regN: 8, wgX: 8, wgY: 16, tileK: 16 },
    attentionQk: { regM: 4, regN: 8, wgX: 8, wgY: 8, tileK: 16 },
    attentionPv: { regM: 4, regN: 8, wgX: 16, wgY: 4, tileK: 16 },
  },
  provenance: {
    sweep:
      "outputs/bench-browser/geometry-sweep-browser-2026-09-27T16-28-00.787Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-31-46.471Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-29T13-08-11.329Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-29T21-04-14.586Z.json",
    sha256:
      "921e89ba70b34ef68ec21c153c2a2c313c80bbbc98d53da720942c2919ead034, f464ca82eb61f4c8fd60efa06365815eb25cec5e1f8ed905cccd242cadacfc2a, 38d97dac5f04a7e80fb6591ab437da7dd2b3cee66b0d03e40cd1475c0f398550, d9e5e7814370c741562819474f5e0837012d05f233869688d28bb91902ac8621",
    date:
      "2026-09-27T16:28:00.787Z, 2026-09-27T18:31:46.471Z, 2026-09-29T13:08:11.329Z, 2026-09-29T21:04:14.586Z",
    adapter: "apple / metal-3 / Apple M2",
  },
};
