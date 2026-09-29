/**
 * 幾何プロファイル `nvidia-blackwell`（**生成物 — 手で編集しない**）。
 *
 * tools/geometry-sweep の `profile` が掃引の記録から書いた、adapter `nvidia / blackwell` 用のタイル幾何の
 * 静的な表（perf-ledger K-71）。runtime は adapter の `match`（vendor / architecture / description の
 * 完全一致）でこの表を選ぶだけ。
 * 実行時には測らない（オートチューン禁止 — ADR 0022 決定 3）。値を変えるときは掃引を取り直して
 * 下のコマンドで再生成する。
 *
 * 再生成（リポ直下から・`--check` を足すと再生成とバイト同一かだけを見る）:
 *
 *   deno run -A tools/geometry-sweep/main.ts profile \
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-37-27.914Z.json \
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-09-29T20-42-08.545Z.json \
 *     --id nvidia-blackwell --vendor nvidia --architecture blackwell \
 *     --out packages/runtime/src/kernels/geometry-profiles/nvidia-blackwell.ts --min-speedup 1.05
 *
 * 掃引（adapter nvidia / blackwell / NVIDIA GeForce RTX 5070 Ti）:
 *
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-37-27.914Z.json（sha256 084cd0efd7ea7780dfcdab5b53ee4be8177638c7d8984920096a4851b767aa90・2026-09-27T18:37:27.914Z）
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-29T20-42-08.545Z.json（sha256 1ae3a8f6d9a1270d8e3b3ad41aa7fe13757c976a323de7d2b0b99139800cb2e5・2026-09-29T20:42:08.545Z）
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
 *   - reg128x128r8x8w16: linear-m16-n3072-k1024 で ×0.178 < ×1.050
 *   - reg128x16r8x4w4: linear-m16-n3072-k1024 で ×0.200 < ×1.050
 *   - reg128x32r8x4w8: linear-m16-n3072-k1024 で ×0.270 < ×1.050
 *   - reg128x32r8x8w4: linear-m16-n3072-k1024 で ×0.223 < ×1.050
 *   - reg128x64r8x4w16: linear-m16-n3072-k1024 で ×0.311 < ×1.050
 *   - reg128x64r8x8w8: linear-m16-n3072-k1024 で ×0.263 < ×1.050
 *   - reg16x128r4x8w16: linear-m16-n3072-k1024 で ×0.209 < ×1.050
 *   - reg16x16r2x4w4: linear-m16-n3072-k1024 で ×0.702 < ×1.050
 *   - reg16x16r4x4w4: linear-m16-n3072-k1024 で ×0.373 < ×1.050
 *   - reg16x32r1x8w4: linear-m16-n3072-k1024 で ×0.698 < ×1.050
 *   - reg16x32r2x4w8: linear-m16-n3072-k1024 で ×0.773 < ×1.050
 *   - reg16x32r2x8w4: linear-m16-n3072-k1024 で ×0.402 < ×1.050
 *   - reg16x32r4x4w8: linear-m16-n3072-k1024 で ×0.414 < ×1.050
 *   - reg16x32r4x8w4: linear-m16-n3072-k1024 で ×0.212 < ×1.050
 *   - reg16x64r2x8w8: linear-m16-n3072-k1024 で ×0.418 < ×1.050
 *   - reg16x64r4x4w16: linear-m16-n3072-k1024 で ×0.432 < ×1.050
 *   - reg16x64r4x8w8: linear-m16-n3072-k1024 で ×0.216 < ×1.050
 *   - reg32x128r4x8w16: linear-m16-n3072-k1024 で ×0.319 < ×1.050
 *   - reg32x128r8x8w16: linear-m16-n3072-k1024 で ×0.162 < ×1.050
 *   - reg32x16r2x4w4: linear-m16-n3072-k1024 で ×0.715 < ×1.050
 *   - reg32x16r4x4w4: linear-m16-n3072-k1024 で ×0.465 < ×1.050
 *   - reg32x16r8x4w4: linear-m16-n3072-k1024 で ×0.280 < ×1.050
 *   - reg32x32r2x4w8: linear-m16-n3072-k1024 で ×0.658 < ×1.050
 *   - reg32x32r2x8w4: linear-m16-n3072-k1024 で ×0.555 < ×1.050
 *   - reg32x32r4x4w8: linear-m16-n3072-k1024 で ×0.585 < ×1.050
 *   - reg32x32r4x8w4: linear-m16-n3072-k1024 で ×0.298 < ×1.050
 *   - reg32x32r8x4w8: linear-m16-n3072-k1024 で ×0.305 < ×1.050
 *   - reg32x32r8x8w4: linear-m16-n3072-k1024 で ×0.162 < ×1.050
 *   - reg32x64r2x8w8: linear-m16-n3072-k1024 で ×0.546 < ×1.050
 *   - reg32x64r4x4w16: linear-m16-n3072-k1024 で ×0.524 < ×1.050
 *   - reg32x64r4x8w8: linear-m16-n3072-k1024 で ×0.335 < ×1.050
 *   - reg32x64r8x4w16: linear-m16-n3072-k1024 で ×0.349 < ×1.050
 *   - reg32x64r8x8w8: linear-m16-n3072-k1024 で ×0.156 < ×1.050
 *   - reg4x16r1x4w4: linear-m32-n3072-k1024 で ×0.278 < ×1.050
 *   - reg4x32r1x8w4: linear-m16-n3072-k1024 で ×0.238 < ×1.050
 *   - reg64x128r4x8w16: linear-m16-n3072-k1024 で ×0.286 < ×1.050
 *   - reg64x128r8x8w16: linear-m16-n3072-k1024 で ×0.234 < ×1.050
 *   - reg64x16r4x4w4: linear-m16-n3072-k1024 で ×0.390 < ×1.050
 *   - reg64x16r8x4w4: linear-m16-n3072-k1024 で ×0.270 < ×1.050
 *   - reg64x32r4x4w8: linear-m16-n3072-k1024 で ×0.467 < ×1.050
 *   - reg64x32r4x8w4: linear-m16-n3072-k1024 で ×0.356 < ×1.050
 *   - reg64x32r8x4w8: linear-m16-n3072-k1024 で ×0.368 < ×1.050
 *   - reg64x32r8x8w4: linear-m16-n3072-k1024 で ×0.207 < ×1.050
 *   - reg64x64r4x4w16: linear-m16-n3072-k1024 で ×0.504 < ×1.050
 *   - reg64x64r4x8w8: linear-m16-n3072-k1024 で ×0.399 < ×1.050
 *   - reg64x64r8x4w16: linear-m16-n3072-k1024 で ×0.349 < ×1.050
 *   - reg64x64r8x8w8: linear-m16-n3072-k1024 で ×0.220 < ×1.050
 *   - reg8x16r1x4w4: linear-m64-n3072-k1024 で ×0.686 < ×1.050
 *   - reg8x16r2x4w4: linear-m64-n3072-k1024 で ×0.425 < ×1.050
 *   - reg8x32r1x8w4: linear-m16-n3072-k1024 で ×0.464 < ×1.050
 *   - reg8x32r2x4w8: linear-m16-n3072-k1024 で ×0.504 < ×1.050
 *   - reg8x32r2x8w4: linear-m16-n3072-k1024 で ×0.259 < ×1.050
 *   - reg8x64r2x8w8: linear-m16-n3072-k1024 で ×0.233 < ×1.050
 * - gemmRows[1]（linear / matmul / bmm の行数 65〜512・7 ケース）: 既定 reg64x32r4x4w8 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - reg128x128r8x8w16: linear-m128-n2048-k1024 で ×0.409 < ×1.050
 *   - reg128x16r8x4w4: linear-m128-n2048-k1024 で ×0.544 < ×1.050
 *   - reg128x32r8x4w8: linear-m128-n2048-k1024 で ×0.857 < ×1.050
 *   - reg128x32r8x8w4: linear-m128-n2048-k1024 で ×0.529 < ×1.050
 *   - reg128x64r8x4w16: linear-m128-n2048-k1024 で ×0.675 < ×1.050
 *   - reg128x64r8x8w8: linear-m128-n2048-k1024 で ×0.590 < ×1.050
 *   - reg16x128r4x8w16: linear-m128-n2048-k1024 で ×0.448 < ×1.050
 *   - reg16x16r1x4w4: linear-m256-n2048-k1024 で ×0.622 < ×1.050
 *   - reg16x16r2x4w4: linear-m256-n2048-k1024 で ×0.466 < ×1.050
 *   - reg16x16r4x4w4: linear-m256-n2048-k1024 で ×0.351 < ×1.050
 *   - reg16x32r1x8w4: linear-m256-n2048-k1024 で ×0.705 < ×1.050
 *   - reg16x32r2x4w8: linear-m256-n2048-k1024 で ×0.654 < ×1.050
 *   - reg16x32r2x8w4: linear-m128-n2048-k1024 で ×0.635 < ×1.050
 *   - reg16x32r4x4w8: linear-m512-n2048-k1024 で ×0.634 < ×1.050
 *   - reg16x32r4x8w4: linear-m128-n2048-k1024 で ×0.377 < ×1.050
 *   - reg16x64r2x8w8: linear-m256-n2048-k1024 で ×0.701 < ×1.050
 *   - reg16x64r4x4w16: linear-m128-n2048-k1024 で ×0.751 < ×1.050
 *   - reg16x64r4x8w8: linear-m128-n2048-k1024 で ×0.435 < ×1.050
 *   - reg32x128r4x8w16: linear-m128-n2048-k1024 で ×0.733 < ×1.050
 *   - reg32x128r8x8w16: linear-m128-n2048-k1024 で ×0.396 < ×1.050
 *   - reg32x16r2x4w4: bmm-b16-m512-n512-k64 で ×0.601 < ×1.050
 *   - reg32x16r4x4w4: matmul-m512-n2048-k1024 で ×0.540 < ×1.050
 *   - reg32x16r8x4w4: linear-m128-n2048-k1024 で ×0.432 < ×1.050
 *   - reg32x32r2x4w8: linear-m256-n2048-k1024 で ×0.822 < ×1.050
 *   - reg32x32r2x8w4: linear-m256-n2048-k1024 で ×0.854 < ×1.050
 *   - reg32x32r4x4w8: linear-m512-n2048-k1024 で ×0.852 < ×1.050
 *   - reg32x32r4x8w4: linear-m128-n2048-k1024 で ×0.567 < ×1.050
 *   - reg32x32r8x4w8: linear-m128-n2048-k1024 で ×0.600 < ×1.050
 *   - reg32x32r8x8w4: linear-m128-n2048-k1024 で ×0.359 < ×1.050
 *   - reg32x64r2x8w8: linear-m128-n2048-k1024 で ×0.871 < ×1.050
 *   - reg32x64r4x4w16: linear-m128-n2048-k1024 で ×0.876 < ×1.050
 *   - reg32x64r4x8w8: linear-m128-n2048-k1024 で ×0.730 < ×1.050
 *   - reg32x64r8x4w16: linear-m128-n2048-k1024 で ×0.788 < ×1.050
 *   - reg32x64r8x8w8: linear-m128-n2048-k1024 で ×0.355 < ×1.050
 *   - reg4x16r1x4w4: linear-m256-n2048-k1024 で ×0.165 < ×1.050
 *   - reg4x32r1x8w4: linear-m128-n2048-k1024 で ×0.171 < ×1.050
 *   - reg64x128r4x8w16: linear-m128-n2048-k1024 で ×0.635 < ×1.050
 *   - reg64x128r8x8w16: linear-m128-n2048-k1024 で ×0.553 < ×1.050
 *   - reg64x16r4x4w4: matmul-m512-n2048-k1024 で ×0.541 < ×1.050
 *   - reg64x16r8x4w4: linear-m128-n2048-k1024 で ×0.486 < ×1.050
 *   - reg64x32r4x8w4: linear-m128-n2048-k1024 で ×0.735 < ×1.050
 *   - reg64x32r8x4w8: linear-m128-n2048-k1024 で ×0.819 < ×1.050
 *   - reg64x32r8x8w4: linear-m128-n2048-k1024 で ×0.425 < ×1.050
 *   - reg64x64r4x4w16: linear-m128-n2048-k1024 で ×1.035 < ×1.050
 *   - reg64x64r4x8w8: linear-m128-n2048-k1024 で ×0.883 < ×1.050
 *   - reg64x64r8x4w16: linear-m128-n2048-k1024 で ×0.789 < ×1.050
 *   - reg64x64r8x8w8: linear-m128-n2048-k1024 で ×0.534 < ×1.050
 *   - reg8x16r1x4w4: linear-m256-n2048-k1024 で ×0.419 < ×1.050
 *   - reg8x16r2x4w4: linear-m256-n2048-k1024 で ×0.253 < ×1.050
 *   - reg8x32r1x8w4: linear-m256-n2048-k1024 で ×0.461 < ×1.050
 *   - reg8x32r2x4w8: linear-m256-n2048-k1024 で ×0.422 < ×1.050
 *   - reg8x32r2x8w4: linear-m256-n2048-k1024 で ×0.274 < ×1.050
 *   - reg8x64r2x8w8: linear-m128-n2048-k1024 で ×0.427 < ×1.050
 * - gemmRows[2]（linear / matmul / bmm の行数 > 512・7 ケース）: 既定 reg128x128r8x8w16 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - reg128x16r8x4w4: matmul-m4096-n2048-k2048 で ×0.349 < ×1.050
 *   - reg128x32r8x4w8: linear-m4096-n8192-k2048 で ×0.621 < ×1.050
 *   - reg128x32r8x8w4: linear-m1024-n2048-k8192 で ×0.571 < ×1.050
 *   - reg128x64r8x4w16: linear-m1024-n2048-k8192 で ×0.834 < ×1.050
 *   - reg128x64r8x8w8: linear-m1024-n2048-k2048 で ×0.904 < ×1.050
 *   - reg16x128r4x8w16: linear-m1024-n2048-k2048 で ×0.485 < ×1.050
 *   - reg16x16r1x4w4: linear-m4096-n8192-k2048 で ×0.340 < ×1.050
 *   - reg16x16r2x4w4: linear-m4096-n8192-k2048 で ×0.278 < ×1.050
 *   - reg16x16r4x4w4: linear-m4096-n8192-k2048 で ×0.206 < ×1.050
 *   - reg16x32r1x8w4: linear-m4096-n8192-k2048 で ×0.413 < ×1.050
 *   - reg16x32r2x4w8: linear-m4096-n8192-k2048 で ×0.376 < ×1.050
 *   - reg16x32r2x8w4: linear-m4096-n8192-k2048 で ×0.381 < ×1.050
 *   - reg16x32r4x4w8: linear-m4096-n8192-k2048 で ×0.371 < ×1.050
 *   - reg16x32r4x8w4: linear-m4096-n8192-k2048 で ×0.254 < ×1.050
 *   - reg16x64r2x8w8: linear-m4096-n8192-k2048 で ×0.418 < ×1.050
 *   - reg16x64r4x4w16: linear-m4096-n8192-k2048 で ×0.466 < ×1.050
 *   - reg16x64r4x8w8: linear-m1024-n2048-k2048 で ×0.429 < ×1.050
 *   - reg32x128r4x8w16: linear-m1024-n2048-k2048 で ×0.704 < ×1.050
 *   - reg32x128r8x8w16: linear-m1024-n2048-k8192 で ×0.574 < ×1.050
 *   - reg32x16r2x4w4: linear-m4096-n8192-k2048 で ×0.342 < ×1.050
 *   - reg32x16r4x4w4: linear-m4096-n8192-k2048 で ×0.318 < ×1.050
 *   - reg32x16r8x4w4: linear-m4096-n8192-k2048 で ×0.256 < ×1.050
 *   - reg32x32r2x4w8: linear-m4096-n8192-k2048 で ×0.501 < ×1.050
 *   - reg32x32r2x8w4: linear-m4096-n8192-k2048 で ×0.514 < ×1.050
 *   - reg32x32r4x4w8: linear-m4096-n8192-k2048 で ×0.494 < ×1.050
 *   - reg32x32r4x8w4: linear-m4096-n8192-k2048 で ×0.476 < ×1.050
 *   - reg32x32r8x4w8: linear-m4096-n8192-k2048 で ×0.470 < ×1.050
 *   - reg32x32r8x8w4: linear-m1024-n2048-k8192 で ×0.327 < ×1.050
 *   - reg32x64r2x8w8: linear-m4096-n8192-k2048 で ×0.578 < ×1.050
 *   - reg32x64r4x4w16: linear-m4096-n8192-k2048 で ×0.630 < ×1.050
 *   - reg32x64r4x8w8: linear-m1024-n2048-k2048 で ×0.613 < ×1.050
 *   - reg32x64r8x4w16: linear-m1024-n2048-k2048 で ×0.649 < ×1.050
 *   - reg32x64r8x8w8: linear-m1024-n2048-k8192 で ×0.512 < ×1.050
 *   - reg4x16r1x4w4: linear-m4096-n8192-k2048 で ×0.089 < ×1.050
 *   - reg4x32r1x8w4: linear-m4096-n8192-k2048 で ×0.095 < ×1.050
 *   - reg64x128r4x8w16: linear-m1024-n2048-k8192 で ×0.830 < ×1.050
 *   - reg64x128r8x8w16: linear-m1024-n2048-k2048 で ×0.929 < ×1.050
 *   - reg64x16r4x4w4: linear-m4096-n8192-k2048 で ×0.357 < ×1.050
 *   - reg64x16r8x4w4: linear-m1024-n2048-k2048 で ×0.332 < ×1.050
 *   - reg64x32r4x4w8: linear-m4096-n8192-k2048 で ×0.589 < ×1.050
 *   - reg64x32r4x8w4: linear-m4096-n8192-k2048 で ×0.567 < ×1.050
 *   - reg64x32r8x4w8: linear-m4096-n8192-k2048 で ×0.563 < ×1.050
 *   - reg64x32r8x8w4: linear-m1024-n2048-k8192 で ×0.500 < ×1.050
 *   - reg64x64r4x4w16: linear-m1024-n2048-k2048 で ×0.742 < ×1.050
 *   - reg64x64r4x8w8: linear-m1024-n2048-k2048 で ×0.759 < ×1.050
 *   - reg64x64r8x4w16: linear-m1024-n2048-k2048 で ×0.789 < ×1.050
 *   - reg64x64r8x8w8: linear-m1024-n2048-k8192 で ×0.685 < ×1.050
 *   - reg8x16r1x4w4: linear-m4096-n8192-k2048 で ×0.233 < ×1.050
 *   - reg8x16r2x4w4: linear-m4096-n8192-k2048 で ×0.142 < ×1.050
 *   - reg8x32r1x8w4: linear-m4096-n8192-k2048 で ×0.269 < ×1.050
 *   - reg8x32r2x4w8: linear-m4096-n8192-k2048 で ×0.252 < ×1.050
 *   - reg8x32r2x8w4: linear-m4096-n8192-k2048 で ×0.167 < ×1.050
 *   - reg8x64r2x8w8: linear-m4096-n8192-k2048 で ×0.268 < ×1.050
 * - attention.qk（融合 attention f32 ①QK・4 ケース）: 既定 reg128x128r8x8w16 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - reg128x16r8x4w4: attention-qk-self-m4096-n4096 で ×0.367 < ×1.050
 *   - reg128x32r8x4w8: attention-qk-self-m4096-n4096 で ×0.645 < ×1.050
 *   - reg128x32r8x8w4: attention-qk-self-m4096-n4096 で ×0.625 < ×1.050
 *   - reg128x64r8x4w16: attention-qk-self-m4096-n4096 で ×0.855 < ×1.050
 *   - reg128x64r8x8w8: attention-qk-cross-m4096-n512 で ×0.910 < ×1.050
 *   - reg16x128r4x8w16: attention-qk-self-m4096-n4096 で ×0.461 < ×1.050
 *   - reg16x16r1x4w4: attention-qk-self-m4096-n4096 で ×0.321 < ×1.050
 *   - reg16x16r2x4w4: attention-qk-self-m4096-n4096 で ×0.268 < ×1.050
 *   - reg16x16r4x4w4: attention-qk-self-m4096-n4096 で ×0.203 < ×1.050
 *   - reg16x32r1x8w4: attention-qk-self-m4096-n4096 で ×0.385 < ×1.050
 *   - reg16x32r2x4w8: attention-qk-self-m4096-n4096 で ×0.354 < ×1.050
 *   - reg16x32r2x8w4: attention-qk-self-m4096-n4096 で ×0.359 < ×1.050
 *   - reg16x32r4x4w8: attention-qk-self-m4096-n4096 で ×0.352 < ×1.050
 *   - reg16x32r4x8w4: attention-qk-self-m4096-n4096 で ×0.254 < ×1.050
 *   - reg16x64r2x8w8: attention-qk-self-m4096-n4096 で ×0.392 < ×1.050
 *   - reg16x64r4x4w16: attention-qk-self-m4096-n4096 で ×0.431 < ×1.050
 *   - reg16x64r4x8w8: attention-qk-self-m4096-n4096 で ×0.411 < ×1.050
 *   - reg32x128r4x8w16: attention-qk-self-m4096-n4096 で ×0.688 < ×1.050
 *   - reg32x128r8x8w16: attention-qk-self-m1024-n1024 で ×0.685 < ×1.050
 *   - reg32x16r2x4w4: attention-qk-self-m4096-n4096 で ×0.340 < ×1.050
 *   - reg32x16r4x4w4: attention-qk-self-m4096-n4096 で ×0.317 < ×1.050
 *   - reg32x16r8x4w4: attention-qk-self-m4096-n4096 で ×0.260 < ×1.050
 *   - reg32x32r2x4w8: attention-qk-self-m4096-n4096 で ×0.485 < ×1.050
 *   - reg32x32r2x8w4: attention-qk-self-m4096-n4096 で ×0.499 < ×1.050
 *   - reg32x32r4x4w8: attention-qk-self-m4096-n4096 で ×0.481 < ×1.050
 *   - reg32x32r4x8w4: attention-qk-self-m4096-n4096 で ×0.473 < ×1.050
 *   - reg32x32r8x4w8: attention-qk-self-m4096-n4096 で ×0.469 < ×1.050
 *   - reg32x32r8x8w4: attention-qk-self-m1024-n1024 で ×0.349 < ×1.050
 *   - reg32x64r2x8w8: attention-qk-self-m4096-n4096 で ×0.554 < ×1.050
 *   - reg32x64r4x4w16: attention-qk-self-m4096-n4096 で ×0.617 < ×1.050
 *   - reg32x64r4x8w8: attention-qk-self-m4096-n4096 で ×0.603 < ×1.050
 *   - reg32x64r8x4w16: attention-qk-self-m4096-n4096 で ×0.642 < ×1.050
 *   - reg32x64r8x8w8: attention-qk-self-m1024-n1024 で ×0.614 < ×1.050
 *   - reg4x16r1x4w4: attention-qk-self-m4096-n4096 で ×0.085 < ×1.050
 *   - reg4x32r1x8w4: attention-qk-self-m4096-n4096 で ×0.090 < ×1.050
 *   - reg64x128r4x8w16: attention-qk-self-m4096-n4096 で ×0.849 < ×1.050
 *   - reg64x128r8x8w16: attention-qk-cross-m4096-n512 で ×0.924 < ×1.050
 *   - reg64x16r4x4w4: attention-qk-self-m4096-n4096 で ×0.363 < ×1.050
 *   - reg64x16r8x4w4: attention-qk-self-m4096-n4096 で ×0.343 < ×1.050
 *   - reg64x32r4x4w8: attention-qk-self-m4096-n4096 で ×0.589 < ×1.050
 *   - reg64x32r4x8w4: attention-qk-self-m4096-n4096 で ×0.583 < ×1.050
 *   - reg64x32r8x4w8: attention-qk-self-m4096-n4096 で ×0.575 < ×1.050
 *   - reg64x32r8x8w4: attention-qk-self-m4096-n4096 で ×0.559 < ×1.050
 *   - reg64x64r4x4w16: attention-qk-self-m4096-n4096 で ×0.731 < ×1.050
 *   - reg64x64r4x8w8: attention-qk-self-m4096-n4096 で ×0.763 < ×1.050
 *   - reg64x64r8x4w16: attention-qk-self-m4096-n4096 で ×0.795 < ×1.050
 *   - reg64x64r8x8w8: attention-qk-cross-m4096-n512 で ×0.775 < ×1.050
 *   - reg8x16r1x4w4: attention-qk-self-m4096-n4096 で ×0.216 < ×1.050
 *   - reg8x16r2x4w4: attention-qk-self-m4096-n4096 で ×0.137 < ×1.050
 *   - reg8x32r1x8w4: attention-qk-self-m4096-n4096 で ×0.246 < ×1.050
 *   - reg8x32r2x4w8: attention-qk-self-m4096-n4096 で ×0.231 < ×1.050
 *   - reg8x32r2x8w4: attention-qk-self-m4096-n4096 で ×0.159 < ×1.050
 *   - reg8x64r2x8w8: attention-qk-self-m4096-n4096 で ×0.245 < ×1.050
 * - attention.pv（融合 attention f32 ③PV・4 ケース）: 既定 reg128x128r8x8w16 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - reg128x16r8x4w4: attention-pv-cross-m1024-n512 で ×0.328 < ×1.050
 *   - reg128x32r8x4w8: attention-pv-cross-m1024-n512 で ×0.590 < ×1.050
 *   - reg128x32r8x8w4: attention-pv-self-m1024-n1024 で ×0.446 < ×1.050
 *   - reg128x64r8x4w16: attention-pv-self-m1024-n1024 で ×0.791 < ×1.050
 *   - reg128x64r8x8w8: attention-pv-cross-m1024-n512 で ×0.881 < ×1.050
 *   - reg16x128r4x8w16: attention-pv-cross-m1024-n512 で ×0.748 < ×1.050
 *   - reg16x16r1x4w4: attention-pv-cross-m1024-n512 で ×0.438 < ×1.050
 *   - reg16x16r2x4w4: attention-pv-cross-m1024-n512 で ×0.324 < ×1.050
 *   - reg16x16r4x4w4: attention-pv-cross-m1024-n512 で ×0.300 < ×1.050
 *   - reg16x32r1x8w4: attention-pv-cross-m1024-n512 で ×0.614 < ×1.050
 *   - reg16x32r2x4w8: attention-pv-cross-m1024-n512 で ×0.523 < ×1.050
 *   - reg16x32r2x8w4: attention-pv-cross-m1024-n512 で ×0.569 < ×1.050
 *   - reg16x32r4x4w8: attention-pv-cross-m1024-n512 で ×0.529 < ×1.050
 *   - reg16x32r4x8w4: attention-pv-cross-m1024-n512 で ×0.435 < ×1.050
 *   - reg16x64r2x8w8: attention-pv-cross-m1024-n512 で ×0.630 < ×1.050
 *   - reg16x64r4x4w16: attention-pv-cross-m1024-n512 で ×0.714 < ×1.050
 *   - reg16x64r4x8w8: attention-pv-cross-m1024-n512 で ×0.649 < ×1.050
 *   - reg32x128r4x8w16: attention-pv-cross-m1024-n512 で ×0.877 < ×1.050
 *   - reg32x128r8x8w16: attention-pv-self-m1024-n1024 で ×0.669 < ×1.050
 *   - reg32x16r2x4w4: attention-pv-cross-m1024-n512 で ×0.372 < ×1.050
 *   - reg32x16r4x4w4: attention-pv-cross-m1024-n512 で ×0.344 < ×1.050
 *   - reg32x16r8x4w4: attention-pv-cross-m1024-n512 で ×0.298 < ×1.050
 *   - reg32x32r2x4w8: attention-pv-cross-m1024-n512 で ×0.617 < ×1.050
 *   - reg32x32r2x8w4: attention-pv-cross-m1024-n512 で ×0.641 < ×1.050
 *   - reg32x32r4x4w8: attention-pv-cross-m1024-n512 で ×0.602 < ×1.050
 *   - reg32x32r4x8w4: attention-pv-cross-m1024-n512 で ×0.561 < ×1.050
 *   - reg32x32r8x4w8: attention-pv-cross-m1024-n512 で ×0.539 < ×1.050
 *   - reg32x32r8x8w4: attention-pv-self-m1024-n1024 で ×0.354 < ×1.050
 *   - reg32x64r2x8w8: attention-pv-cross-m1024-n512 で ×0.719 < ×1.050
 *   - reg32x64r4x4w16: attention-pv-cross-m1024-n512 で ×0.820 < ×1.050
 *   - reg32x64r4x8w8: attention-pv-cross-m1024-n512 で ×0.753 < ×1.050
 *   - reg32x64r8x4w16: attention-pv-cross-m1024-n512 で ×0.788 < ×1.050
 *   - reg32x64r8x8w8: attention-pv-self-m1024-n1024 で ×0.606 < ×1.050
 *   - reg4x16r1x4w4: attention-pv-cross-m1024-n512 で ×0.201 < ×1.050
 *   - reg4x32r1x8w4: attention-pv-cross-m1024-n512 で ×0.284 < ×1.050
 *   - reg64x128r4x8w16: attention-pv-cross-m1024-n512 で ×0.911 < ×1.050
 *   - reg64x128r8x8w16: attention-pv-cross-m1024-n512 で ×1.015 < ×1.050
 *   - reg64x16r4x4w4: attention-pv-cross-m1024-n512 で ×0.365 < ×1.050
 *   - reg64x16r8x4w4: attention-pv-cross-m1024-n512 で ×0.320 < ×1.050
 *   - reg64x32r4x4w8: attention-pv-cross-m1024-n512 で ×0.637 < ×1.050
 *   - reg64x32r4x8w4: attention-pv-cross-m1024-n512 で ×0.599 < ×1.050
 *   - reg64x32r8x4w8: attention-pv-cross-m1024-n512 で ×0.578 < ×1.050
 *   - reg64x32r8x8w4: attention-pv-self-m1024-n1024 で ×0.449 < ×1.050
 *   - reg64x64r4x4w16: attention-pv-cross-m1024-n512 で ×0.763 < ×1.050
 *   - reg64x64r4x8w8: attention-pv-cross-m1024-n512 で ×0.815 < ×1.050
 *   - reg64x64r8x4w16: attention-pv-self-m1024-n1024 で ×0.816 < ×1.050
 *   - reg64x64r8x8w8: attention-pv-self-m1024-n1024 で ×0.594 < ×1.050
 *   - reg8x16r1x4w4: attention-pv-cross-m1024-n512 で ×0.332 < ×1.050
 *   - reg8x16r2x4w4: attention-pv-cross-m1024-n512 で ×0.264 < ×1.050
 *   - reg8x32r1x8w4: attention-pv-cross-m1024-n512 で ×0.510 < ×1.050
 *   - reg8x32r2x4w8: attention-pv-cross-m1024-n512 で ×0.433 < ×1.050
 *   - reg8x32r2x8w4: attention-pv-cross-m1024-n512 で ×0.387 < ×1.050
 *   - reg8x64r2x8w8: attention-pv-cross-m1024-n512 で ×0.497 < ×1.050
 * - conv2d.rows64（conv2d implicit GEMM の m タイル 64 行・2 ケース）: 既定 igemm64x128:wg16x8 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - igemm128x128:wg16x16: conv2d-c192-256x256 で ×0.848 < ×1.050
 *   - igemm128x64:wg16x16: conv2d-c192-256x256 で ×0.782 < ×1.050
 *   - igemm128x64:wg8x16: conv2d-c192-256x256 で ×0.806 < ×1.050
 *   - igemm16x128:wg16x4: conv2d-c384-128x128 で ×0.601 < ×1.050
 *   - igemm16x64:wg16x4: conv2d-c384-128x128 で ×0.599 < ×1.050
 *   - igemm16x64:wg8x4: conv2d-c384-128x128 で ×0.583 < ×1.050
 *   - igemm16x64:wg8x8: conv2d-c384-128x128 で ×0.587 < ×1.050
 *   - igemm32x128:wg16x4: conv2d-c384-128x128 で ×0.799 < ×1.050
 *   - igemm32x128:wg16x8: conv2d-c384-128x128 で ×0.836 < ×1.050
 *   - igemm32x64:wg16x4: conv2d-c192-256x256 で ×0.747 < ×1.050
 *   - igemm32x64:wg16x8: conv2d-c384-128x128 で ×0.795 < ×1.050
 *   - igemm32x64:wg8x16: conv2d-c384-128x128 で ×0.773 < ×1.050
 *   - igemm32x64:wg8x4: conv2d-c192-256x256 で ×0.741 < ×1.050
 *   - igemm32x64:wg8x8: conv2d-c192-256x256 で ×0.807 < ×1.050
 *   - igemm64x128:wg16x16: conv2d-c192-256x256 で ×1.002 < ×1.050
 *   - igemm64x64:wg16x16: conv2d-c192-256x256 で ×0.947 < ×1.050
 *   - igemm64x64:wg16x8: conv2d-c384-128x128 で ×0.935 < ×1.050
 *   - igemm64x64:wg8x16: conv2d-c192-256x256 で ×0.980 < ×1.050
 *   - igemm64x64:wg8x8: conv2d-c192-256x256 で ×0.928 < ×1.050
 *   - igemm8x64:wg8x4: conv2d-c384-128x128 で ×0.363 < ×1.050
 * - conv2d.rows32（conv2d implicit GEMM の m タイル 32 行・1 ケース）: 既定 igemm32x128:wg16x4 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - igemm128x128:wg16x16: conv2d-c96-512x512 で ×1.034 < ×1.050
 *   - igemm128x64:wg16x16: conv2d-c96-512x512 で ×0.974 < ×1.050
 *   - igemm128x64:wg8x16: conv2d-c96-512x512 で ×0.996 < ×1.050
 *   - igemm16x128:wg16x4: conv2d-c96-512x512 で ×0.777 < ×1.050
 *   - igemm16x64:wg16x4: conv2d-c96-512x512 で ×0.759 < ×1.050
 *   - igemm16x64:wg8x4: conv2d-c96-512x512 で ×0.731 < ×1.050
 *   - igemm16x64:wg8x8: conv2d-c96-512x512 で ×0.744 < ×1.050
 *   - igemm32x128:wg16x8: conv2d-c96-512x512 で ×1.040 < ×1.050
 *   - igemm32x64:wg16x4: conv2d-c96-512x512 で ×0.923 < ×1.050
 *   - igemm32x64:wg16x8: conv2d-c96-512x512 で ×1.003 < ×1.050
 *   - igemm32x64:wg8x16: conv2d-c96-512x512 で ×0.959 < ×1.050
 *   - igemm32x64:wg8x4: conv2d-c96-512x512 で ×0.937 < ×1.050
 *   - igemm32x64:wg8x8: conv2d-c96-512x512 で ×0.993 < ×1.050
 *   - igemm64x128:wg16x16: conv2d-c96-512x512 で ×0.942 < ×1.050
 *   - igemm64x128:wg16x8: conv2d-c96-512x512 で ×0.930 < ×1.050
 *   - igemm64x64:wg16x16: conv2d-c96-512x512 で ×0.905 < ×1.050
 *   - igemm64x64:wg16x8: conv2d-c96-512x512 で ×0.898 < ×1.050
 *   - igemm64x64:wg8x16: conv2d-c96-512x512 で ×0.941 < ×1.050
 *   - igemm64x64:wg8x8: conv2d-c96-512x512 で ×0.892 < ×1.050
 *   - igemm8x64:wg8x4: conv2d-c96-512x512 で ×0.479 < ×1.050
 * - i8a8.linear（i8a8 linear・6 ケース）: 採用 tile128x64r8x4w16x16k16 ×1.193（×1.164〜×1.239）
 *   - tile128x128r8x8w16x16k16: i8a8-linear-m4096-n2048-k2048 で ×0.815 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-linear-m4096-n2048-k2048 で ×0.865 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-linear-m4096-n8192-k2048 で ×0.903 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-linear-m4096-n2048-k8192 で ×1.037 < ×1.050
 *   - tile128x64r8x4w16x16k32: 幾何平均 ×1.147（採用 ×1.193 に届かない）
 *   - tile128x64r8x8w8x16k32: 幾何平均 ×1.158（採用 ×1.193 に届かない）
 *   - tile16x128r4x8w16x4k16: i8a8-linear-m1024-n2048-k2048 で ×0.556 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-linear-m4096-n2048-k8192 で ×0.617 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-linear-m4096-n2048-k8192 で ×0.559 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-linear-m4096-n8192-k2048 で ×0.670 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-linear-m4096-n8192-k2048 で ×0.544 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-linear-m4096-n8192-k2048 で ×0.673 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-linear-m1024-n2048-k2048 で ×0.498 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-linear-m4096-n8192-k2048 で ×0.738 < ×1.050
 *   - tile32x128r4x8w16x8k16: i8a8-linear-m4096-n2048-k8192 で ×0.753 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-linear-m4096-n2048-k8192 で ×0.756 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-linear-m4096-n2048-k2048 で ×0.833 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-linear-m4096-n2048-k2048 で ×0.816 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-linear-m1024-n2048-k8192 で ×0.699 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-linear-m4096-n8192-k2048 で ×0.766 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-linear-m1024-n2048-k2048 で ×0.601 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-linear-m4096-n2048-k8192 で ×0.884 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-linear-m4096-n8192-k2048 で ×0.808 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-linear-m4096-n8192-k2048 で ×0.796 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-linear-m1024-n2048-k2048 で ×0.791 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-linear-m4096-n2048-k8192 で ×0.925 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-linear-m1024-n2048-k2048 で ×0.864 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-linear-m4096-n2048-k8192 で ×0.959 < ×1.050
 *   - tile32x64r8x8w8x4k16: i8a8-linear-m4096-n2048-k2048 で ×0.888 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-linear-m4096-n2048-k8192 で ×0.853 < ×1.050
 *   - tile64x128r4x8w16x16k16: i8a8-linear-m4096-n2048-k8192 で ×0.835 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-linear-m4096-n2048-k8192 で ×0.802 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m1024-n2048-k2048 で ×0.898 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-linear-m4096-n2048-k2048 で ×0.883 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-linear-m4096-n8192-k2048 で ×0.853 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-linear-m4096-n8192-k2048 で ×0.810 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-linear-m1024-n2048-k2048 で ×0.847 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-linear-m4096-n2048-k8192 で ×0.999 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-linear-m4096-n8192-k2048 で ×0.874 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-linear-m4096-n8192-k2048 で ×0.890 < ×1.050
 *   - tile64x64r4x8w8x16k16: 幾何平均 ×1.083（採用 ×1.193 に届かない）
 *   - tile64x64r4x8w8x16k32: i8a8-linear-m4096-n2048-k8192 で ×1.029 < ×1.050
 *   - tile64x64r8x4w16x8k16: 幾何平均 ×1.126（採用 ×1.193 に届かない）
 *   - tile64x64r8x4w16x8k32: 幾何平均 ×1.111（採用 ×1.193 に届かない）
 *   - tile64x64r8x8w8x8k16: i8a8-linear-m4096-n2048-k8192 で ×0.956 < ×1.050
 *   - tile64x64r8x8w8x8k32: 幾何平均 ×1.104（採用 ×1.193 に届かない）
 * - i8a8.attentionQk（i8a8 attention ①QK・4 ケース）: 採用 tile128x64r8x4w16x16k16 ×1.222（×1.201〜×1.243）
 *   - tile128x128r8x8w16x16k16: i8a8-attention-qk-cross-m1024-n512 で ×0.952 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-attention-qk-cross-m1024-n512 で ×0.894 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-attention-qk-cross-m4096-n512 で ×0.974 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-attention-qk-cross-m4096-n512 で ×1.039 < ×1.050
 *   - tile128x64r8x4w16x16k32: 幾何平均 ×1.199（採用 ×1.222 に届かない）
 *   - tile128x64r8x8w8x16k32: i8a8-attention-qk-cross-m1024-n512 で ×1.015 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile16x128r4x8w16x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile16x32r4x4w8x4k16: i8a8-attention-qk-self-m4096-n4096 で ×0.717 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile16x64r4x4w16x4k16: i8a8-attention-qk-self-m4096-n4096 で ×0.721 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile16x64r4x8w8x4k16: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile16x64r4x8w8x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x128r4x8w16x8k16: i8a8-attention-qk-self-m4096-n4096 で ×0.816 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x128r8x8w16x4k16: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x128r8x8w16x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x32r4x4w8x8k16: i8a8-attention-qk-self-m4096-n4096 で ×0.827 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-attention-qk-self-m4096-n4096 で ×0.805 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-attention-qk-cross-m1024-n512 で ×0.903 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x64r4x4w16x8k16: i8a8-attention-qk-self-m4096-n4096 で ×0.887 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-attention-qk-self-m4096-n4096 で ×0.878 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-attention-qk-self-m4096-n4096 で ×0.941 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x64r8x4w16x4k16: i8a8-attention-qk-cross-m1024-n512 で ×1.000 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x64r8x8w8x4k16: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x64r8x8w8x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile64x128r4x8w16x16k16: i8a8-attention-qk-cross-m1024-n512 で ×0.878 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-attention-qk-cross-m1024-n512 で ×0.885 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-attention-qk-cross-m1024-n512 で ×0.995 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile64x32r4x4w8x16k16: i8a8-attention-qk-cross-m4096-n512 で ×0.894 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-attention-qk-cross-m4096-n512 で ×0.881 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-attention-qk-cross-m1024-n512 で ×0.990 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-attention-qk-cross-m4096-n512 で ×0.974 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-attention-qk-cross-m4096-n512 で ×0.963 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-attention-qk-self-m4096-n4096 で ×0.967 < ×1.050
 *   - tile64x64r4x8w8x16k16: 幾何平均 ×1.090（採用 ×1.222 に届かない）
 *   - tile64x64r4x8w8x16k32: i8a8-attention-qk-self-m4096-n4096 で ×1.012 < ×1.050
 *   - tile64x64r8x4w16x8k16: 幾何平均 ×1.171（採用 ×1.222 に届かない）
 *   - tile64x64r8x4w16x8k32: 幾何平均 ×1.173（採用 ×1.222 に届かない）
 *   - tile64x64r8x8w8x8k16: i8a8-attention-qk-cross-m4096-n512 で ×0.970 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 * - i8a8.attentionPv（i8a8 attention ③PV・4 ケース）: 採用 tile128x64r8x4w16x16k16 ×1.210（×1.135〜×1.301）
 *   - tile128x128r8x8w16x16k16: i8a8-attention-pv-cross-m4096-n512 で ×0.878 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-attention-pv-cross-m4096-n512 で ×0.885 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-attention-pv-self-m4096-n4096 で ×0.814 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-attention-pv-self-m4096-n4096 で ×0.752 < ×1.050
 *   - tile128x64r8x4w16x16k32: 幾何平均 ×1.175（採用 ×1.210 に届かない）
 *   - tile128x64r8x8w8x16k16: i8a8-attention-pv-cross-m4096-n512 で ×0.966 < ×1.050
 *   - tile128x64r8x8w8x16k32: i8a8-attention-pv-cross-m4096-n512 で ×0.839 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.727 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.743 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.649 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.643 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.678 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.769 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.758 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.763 < ×1.050
 *   - tile32x128r4x8w16x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.880 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-attention-pv-self-m4096-n4096 で ×0.894 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.932 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.896 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.764 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-attention-pv-self-m4096-n4096 で ×0.788 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-attention-pv-self-m1024-n1024 で ×0.749 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-attention-pv-cross-m4096-n512 で ×0.716 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.896 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-attention-pv-self-m4096-n4096 で ×0.911 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.934 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-attention-pv-self-m4096-n4096 で ×0.934 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.957 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.952 < ×1.050
 *   - tile32x64r8x8w8x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.892 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-attention-pv-cross-m4096-n512 で ×0.777 < ×1.050
 *   - tile64x128r4x8w16x16k16: i8a8-attention-pv-self-m4096-n4096 で ×0.945 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-attention-pv-self-m4096-n4096 で ×0.942 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-attention-pv-cross-m4096-n512 で ×0.975 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-attention-pv-self-m4096-n4096 で ×0.838 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-attention-pv-self-m4096-n4096 で ×0.843 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.814 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-attention-pv-self-m4096-n4096 で ×0.748 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-attention-pv-self-m4096-n4096 で ×0.962 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-attention-pv-self-m4096-n4096 で ×0.983 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-attention-pv-self-m4096-n4096 で ×1.046 < ×1.050
 *   - tile64x64r4x8w8x16k32: i8a8-attention-pv-cross-m4096-n512 で ×1.013 < ×1.050
 *   - tile64x64r8x4w16x8k16: 幾何平均 ×1.172（採用 ×1.210 に届かない）
 *   - tile64x64r8x4w16x8k32: 幾何平均 ×1.102（採用 ×1.210 に届かない）
 *   - tile64x64r8x8w8x8k16: i8a8-attention-pv-cross-m4096-n512 で ×0.940 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-attention-pv-cross-m4096-n512 で ×0.837 < ×1.050
 */
import type { GeometryProfile } from "../geometry-profile.ts";

export const NVIDIA_BLACKWELL: GeometryProfile = {
  id: "nvidia-blackwell",
  match: { vendor: "nvidia", architecture: "blackwell" },
  gemmRows: [
    { maxRows: 64, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 512, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
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
    sweep:
      "outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-37-27.914Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-29T20-42-08.545Z.json",
    sha256:
      "084cd0efd7ea7780dfcdab5b53ee4be8177638c7d8984920096a4851b767aa90, 1ae3a8f6d9a1270d8e3b3ad41aa7fe13757c976a323de7d2b0b99139800cb2e5",
    date: "2026-09-27T18:37:27.914Z, 2026-09-29T20:42:08.545Z",
    adapter: "nvidia / blackwell / NVIDIA GeForce RTX 5070 Ti",
  },
};
