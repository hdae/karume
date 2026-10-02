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
 *     --from outputs/bench-browser/geometry-sweep-browser-2026-10-01T16-46-50.379Z.json \
 *     --id nvidia-blackwell --vendor nvidia --architecture blackwell \
 *     --out packages/runtime/src/kernels/geometry-profiles/nvidia-blackwell.ts --min-speedup 1.05
 *
 * 掃引（adapter nvidia / blackwell / 0x2c05 / NVIDIA GeForce RTX 5070 Ti）:
 *
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-37-27.914Z.json（sha256 084cd0efd7ea7780dfcdab5b53ee4be8177638c7d8984920096a4851b767aa90・2026-09-27T18:37:27.914Z）
 * - outputs/bench-browser/geometry-sweep-browser-2026-09-29T20-42-08.545Z.json（sha256 1ae3a8f6d9a1270d8e3b3ad41aa7fe13757c976a323de7d2b0b99139800cb2e5・2026-09-29T20:42:08.545Z）
 * - outputs/bench-browser/geometry-sweep-browser-2026-10-01T16-46-50.379Z.json（sha256 0e57d4efcb14efe0e01767066fce05e1784ca43660a4f8a78c5a1a2c48d1ae87・2026-10-01T16:46:50.379Z）
 *
 * 採否の基準: クラスの全ケースで出力が既定と一致し、既定比が ×1.050 以上の幾何のうち、
 * ケース間の幾何平均が最大のもの。無ければ既定（掃引の既定の行の幾何）。同じケースを複数の掃引が
 * 測っていれば、比はその観測の幾何平均。gemmRows は掃引にある linear / matmul / bmm のケースで決め、3 経路に同じ表が効く。
 * 材料の門: 掃引ごとに、既定の再測定比（cases[].defaultRepeat.driftRatio）が 0.9〜1.1 の外か、
 * 再測定が失敗 / 無いケースはその掃引の比の材料から外す（出力の一致と失敗は見る — 外した掃引で不一致 /
 * 失敗の幾何は採らない。比は同じケースを他の掃引が測っていればそちらで判定し、どの掃引にも残らなければ
 * 測っていない扱い）。外したケースは採否の欄ごとに「掃引 …」の行で示す。
 * 丸めの門: timestamp が丸められた掃引（Chrome のフラグ無しの 100 µs）では、観測（掃引 1 本の中の 1 行）
 * ごとに比の丸め誤差の上界 E = e(行) + e(既定の行)（e = 刻み ÷ 最小の round）を出し、E が 1% を超える観測を
 * その掃引の比の材料から外す（出力の一致と失敗は見る）。既定の行の e が超える（か出せない）ケースは全観測を
 * 外す。外した観測は「掃引 … の <幾何> は丸め誤差の上界 E …」の行で示す。
 *
 * 採否:
 *
 * - gemmRows ≤ 16（linear / matmul / bmm の行数 ≤ 16・4 ケース）: 既定 reg16x16r1x4w4 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - reg128x128r8x8w16: matmul-m16-n3072-k1024 で ×0.166 < ×1.050
 *   - reg128x16r8x4w4: matmul-m16-n3072-k1024 で ×0.168 < ×1.050
 *   - reg128x32r8x4w8: matmul-m16-n3072-k1024 で ×0.233 < ×1.050
 *   - reg128x32r8x8w4: matmul-m16-n3072-k1024 で ×0.206 < ×1.050
 *   - reg128x64r8x4w16: bmm-b16-m16-n64-k128 で ×0.268 < ×1.050
 *   - reg128x64r8x8w8: bmm-b16-m16-n64-k128 で ×0.264 < ×1.050
 *   - reg16x128r4x8w16: linear-m16-n3072-k1024 で ×0.209 < ×1.050
 *   - reg16x16r2x4w4: linear-m16-n3072-k1024 で ×0.703 < ×1.050
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
 *   - reg32x128r8x8w16: linear-m16-n3072-k1024 で ×0.166 < ×1.050
 *   - reg32x16r2x4w4: matmul-m16-n3072-k1024 で ×0.638 < ×1.050
 *   - reg32x16r4x4w4: linear-m16-n3072-k1024 で ×0.465 < ×1.050
 *   - reg32x16r8x4w4: linear-m16-n3072-k1024 で ×0.280 < ×1.050
 *   - reg32x32r2x4w8: matmul-m16-n3072-k1024 で ×0.644 < ×1.050
 *   - reg32x32r2x8w4: linear-m16-n3072-k1024 で ×0.555 < ×1.050
 *   - reg32x32r4x4w8: linear-m16-n3072-k1024 で ×0.585 < ×1.050
 *   - reg32x32r4x8w4: linear-m16-n3072-k1024 で ×0.298 < ×1.050
 *   - reg32x32r8x4w8: linear-m16-n3072-k1024 で ×0.305 < ×1.050
 *   - reg32x32r8x8w4: linear-m16-n3072-k1024 で ×0.162 < ×1.050
 *   - reg32x64r2x8w8: linear-m16-n3072-k1024 で ×0.547 < ×1.050
 *   - reg32x64r4x4w16: linear-m16-n3072-k1024 で ×0.524 < ×1.050
 *   - reg32x64r4x8w8: linear-m16-n3072-k1024 で ×0.336 < ×1.050
 *   - reg32x64r8x4w16: linear-m16-n3072-k1024 で ×0.349 < ×1.050
 *   - reg32x64r8x8w8: linear-m16-n3072-k1024 で ×0.160 < ×1.050
 *   - reg4x16r1x4w4: linear-m16-n3072-k1024 で ×0.358 < ×1.050
 *   - reg4x32r1x8w4: linear-m16-n3072-k1024 で ×0.239 < ×1.050
 *   - reg64x128r4x8w16: matmul-m16-n3072-k1024 で ×0.274 < ×1.050
 *   - reg64x128r8x8w16: linear-m16-n3072-k1024 で ×0.237 < ×1.050
 *   - reg64x16r4x4w4: matmul-m16-n3072-k1024 で ×0.347 < ×1.050
 *   - reg64x16r8x4w4: matmul-m16-n3072-k1024 で ×0.263 < ×1.050
 *   - reg64x32r4x4w8: matmul-m16-n3072-k1024 で ×0.430 < ×1.050
 *   - reg64x32r4x8w4: linear-m16-n3072-k1024 で ×0.356 < ×1.050
 *   - reg64x32r8x4w8: linear-m16-n3072-k1024 で ×0.368 < ×1.050
 *   - reg64x32r8x8w4: linear-m16-n3072-k1024 で ×0.207 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m16-n64-k128 で ×0.455 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m16-n64-k128 で ×0.399 < ×1.050
 *   - reg64x64r8x4w16: linear-m16-n3072-k1024 で ×0.349 < ×1.050
 *   - reg64x64r8x8w8: linear-m16-n3072-k1024 で ×0.226 < ×1.050
 *   - reg8x16r1x4w4: linear-m16-n3072-k1024 で ×0.778 < ×1.050
 *   - reg8x16r2x4w4: linear-m16-n3072-k1024 で ×0.445 < ×1.050
 *   - reg8x32r1x8w4: linear-m16-n3072-k1024 で ×0.464 < ×1.050
 *   - reg8x32r2x4w8: linear-m16-n3072-k1024 で ×0.504 < ×1.050
 *   - reg8x32r2x8w4: linear-m16-n3072-k1024 で ×0.259 < ×1.050
 *   - reg8x64r2x8w8: linear-m16-n3072-k1024 で ×0.233 < ×1.050
 * - gemmRows 17〜32（linear / matmul / bmm の行数 17〜32・4 ケース）: 既定 reg16x16r1x4w4 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - reg128x128r8x8w16: bmm-b16-m32-n64-k128 で ×0.194 < ×1.050
 *   - reg128x16r8x4w4: matmul-m32-n3072-k1024 で ×0.230 < ×1.050
 *   - reg128x32r8x4w8: matmul-m32-n3072-k1024 で ×0.318 < ×1.050
 *   - reg128x32r8x8w4: bmm-b16-m32-n64-k128 で ×0.272 < ×1.050
 *   - reg128x64r8x4w16: bmm-b16-m32-n64-k128 で ×0.299 < ×1.050
 *   - reg128x64r8x8w8: bmm-b16-m32-n64-k128 で ×0.294 < ×1.050
 *   - reg16x128r4x8w16: linear-m32-n3072-k1024 で ×0.330 < ×1.050
 *   - reg16x16r2x4w4: matmul-m32-n3072-k1024 で ×0.767 < ×1.050
 *   - reg16x16r4x4w4: linear-m32-n3072-k1024 で ×0.495 < ×1.050
 *   - reg16x32r1x8w4: linear-m32-n3072-k1024 で ×0.945 < ×1.050
 *   - reg16x32r2x4w8: linear-m32-n3072-k1024 で ×0.978 < ×1.050
 *   - reg16x32r2x8w4: linear-m32-n3072-k1024 で ×0.588 < ×1.050
 *   - reg16x32r4x4w8: linear-m32-n3072-k1024 で ×0.610 < ×1.050
 *   - reg16x32r4x8w4: linear-m32-n3072-k1024 で ×0.335 < ×1.050
 *   - reg16x64r2x8w8: linear-m32-n3072-k1024 で ×0.607 < ×1.050
 *   - reg16x64r4x4w16: linear-m32-n3072-k1024 で ×0.656 < ×1.050
 *   - reg16x64r4x8w8: linear-m32-n3072-k1024 で ×0.316 < ×1.050
 *   - reg32x128r4x8w16: bmm-b16-m32-n64-k128 で ×0.472 < ×1.050
 *   - reg32x128r8x8w16: linear-m32-n3072-k1024 で ×0.260 < ×1.050
 *   - reg32x16r2x4w4: matmul-m32-n3072-k1024 で ×0.859 < ×1.050
 *   - reg32x16r4x4w4: bmm-b16-m32-n64-k128 で ×0.658 < ×1.050
 *   - reg32x16r8x4w4: linear-m32-n3072-k1024 で ×0.399 < ×1.050
 *   - reg32x32r2x4w8: bmm-b16-m32-n64-k128 で ×0.750 < ×1.050
 *   - reg32x32r2x8w4: bmm-b16-m32-n64-k128 で ×0.724 < ×1.050
 *   - reg32x32r4x4w8: bmm-b16-m32-n64-k128 で ×0.740 < ×1.050
 *   - reg32x32r4x8w4: bmm-b16-m32-n64-k128 で ×0.456 < ×1.050
 *   - reg32x32r8x4w8: bmm-b16-m32-n64-k128 で ×0.464 < ×1.050
 *   - reg32x32r8x8w4: linear-m32-n3072-k1024 で ×0.242 < ×1.050
 *   - reg32x64r2x8w8: bmm-b16-m32-n64-k128 で ×0.628 < ×1.050
 *   - reg32x64r4x4w16: bmm-b16-m32-n64-k128 で ×0.671 < ×1.050
 *   - reg32x64r4x8w8: bmm-b16-m32-n64-k128 で ×0.484 < ×1.050
 *   - reg32x64r8x4w16: bmm-b16-m32-n64-k128 で ×0.503 < ×1.050
 *   - reg32x64r8x8w8: linear-m32-n3072-k1024 で ×0.253 < ×1.050
 *   - reg4x16r1x4w4: linear-m32-n3072-k1024 で ×0.280 < ×1.050
 *   - reg4x32r1x8w4: linear-m32-n3072-k1024 で ×0.295 < ×1.050
 *   - reg64x128r4x8w16: bmm-b16-m32-n64-k128 で ×0.316 < ×1.050
 *   - reg64x128r8x8w16: bmm-b16-m32-n64-k128 で ×0.307 < ×1.050
 *   - reg64x16r4x4w4: matmul-m32-n3072-k1024 で ×0.465 < ×1.050
 *   - reg64x16r8x4w4: matmul-m32-n3072-k1024 で ×0.353 < ×1.050
 *   - reg64x32r4x4w8: matmul-m32-n3072-k1024 で ×0.577 < ×1.050
 *   - reg64x32r4x8w4: bmm-b16-m32-n64-k128 で ×0.483 < ×1.050
 *   - reg64x32r8x4w8: bmm-b16-m32-n64-k128 で ×0.489 < ×1.050
 *   - reg64x32r8x8w4: bmm-b16-m32-n64-k128 で ×0.263 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m32-n64-k128 で ×0.498 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m32-n64-k128 で ×0.430 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m32-n64-k128 で ×0.409 < ×1.050
 *   - reg64x64r8x8w8: bmm-b16-m32-n64-k128 で ×0.294 < ×1.050
 *   - reg8x16r1x4w4: linear-m32-n3072-k1024 で ×0.749 < ×1.050
 *   - reg8x16r2x4w4: linear-m32-n3072-k1024 で ×0.456 < ×1.050
 *   - reg8x32r1x8w4: linear-m32-n3072-k1024 で ×0.623 < ×1.050
 *   - reg8x32r2x4w8: linear-m32-n3072-k1024 で ×0.655 < ×1.050
 *   - reg8x32r2x8w4: linear-m32-n3072-k1024 で ×0.343 < ×1.050
 *   - reg8x64r2x8w8: linear-m32-n3072-k1024 で ×0.330 < ×1.050
 * - gemmRows 33〜64（linear / matmul / bmm の行数 33〜64・4 ケース）: 既定 reg16x16r1x4w4 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - reg128x128r8x8w16: bmm-b16-m64-n64-k128 で ×0.224 < ×1.050
 *   - reg128x16r8x4w4: matmul-m64-n3072-k1024 で ×0.420 < ×1.050
 *   - reg128x32r8x4w8: bmm-b16-m64-n64-k128 で ×0.458 < ×1.050
 *   - reg128x32r8x8w4: bmm-b16-m64-n64-k128 で ×0.323 < ×1.050
 *   - reg128x64r8x4w16: bmm-b16-m64-n64-k128 で ×0.345 < ×1.050
 *   - reg128x64r8x8w8: bmm-b16-m64-n64-k128 で ×0.336 < ×1.050
 *   - reg16x128r4x8w16: linear-m64-n3072-k1024 で ×0.545 < ×1.050
 *   - reg16x16r2x4w4: bmm-b16-m64-n128-k64 で ×0.773 < ×1.050
 *   - reg16x16r4x4w4: linear-m64-n3072-k1024 で ×0.621 < ×1.050
 *   - reg16x32r1x8w4: bmm-b16-m64-n64-k128 で ×1.022 < ×1.050
 *   - reg16x32r2x4w8: bmm-b16-m64-n64-k128 で ×1.030 < ×1.050
 *   - reg16x32r2x8w4: linear-m64-n3072-k1024 で ×0.817 < ×1.050
 *   - reg16x32r4x4w8: bmm-b16-m64-n64-k128 で ×0.828 < ×1.050
 *   - reg16x32r4x8w4: linear-m64-n3072-k1024 で ×0.500 < ×1.050
 *   - reg16x64r2x8w8: bmm-b16-m64-n64-k128 で ×0.851 < ×1.050
 *   - reg16x64r4x4w16: bmm-b16-m64-n64-k128 で ×0.874 < ×1.050
 *   - reg16x64r4x8w8: bmm-b16-m64-n64-k128 で ×0.535 < ×1.050
 *   - reg32x128r4x8w16: bmm-b16-m64-n64-k128 で ×0.563 < ×1.050
 *   - reg32x128r8x8w16: bmm-b16-m64-n64-k128 で ×0.354 < ×1.050
 *   - reg32x16r2x4w4: matmul-m64-n3072-k1024 で ×0.835 < ×1.050
 *   - reg32x16r4x4w4: bmm-b16-m64-n64-k128 で ×0.719 < ×1.050
 *   - reg32x16r8x4w4: bmm-b16-m64-n64-k128 で ×0.499 < ×1.050
 *   - reg32x32r2x4w8: bmm-b16-m64-n64-k128 で ×0.886 < ×1.050
 *   - reg32x32r2x8w4: bmm-b16-m64-n64-k128 で ×0.858 < ×1.050
 *   - reg32x32r4x4w8: bmm-b16-m64-n64-k128 で ×0.875 < ×1.050
 *   - reg32x32r4x8w4: bmm-b16-m64-n64-k128 で ×0.545 < ×1.050
 *   - reg32x32r8x4w8: bmm-b16-m64-n64-k128 で ×0.554 < ×1.050
 *   - reg32x32r8x8w4: bmm-b16-m64-n64-k128 で ×0.308 < ×1.050
 *   - reg32x64r2x8w8: bmm-b16-m64-n64-k128 で ×0.749 < ×1.050
 *   - reg32x64r4x4w16: bmm-b16-m64-n64-k128 で ×0.800 < ×1.050
 *   - reg32x64r4x8w8: bmm-b16-m64-n64-k128 で ×0.578 < ×1.050
 *   - reg32x64r8x4w16: bmm-b16-m64-n64-k128 で ×0.601 < ×1.050
 *   - reg32x64r8x8w8: bmm-b16-m64-n64-k128 で ×0.326 < ×1.050
 *   - reg4x16r1x4w4: linear-m64-n3072-k1024 で ×0.278 < ×1.050
 *   - reg4x32r1x8w4: linear-m64-n3072-k1024 で ×0.276 < ×1.050
 *   - reg64x128r4x8w16: bmm-b16-m64-n64-k128 で ×0.360 < ×1.050
 *   - reg64x128r8x8w16: bmm-b16-m64-n64-k128 で ×0.349 < ×1.050
 *   - reg64x16r4x4w4: bmm-b16-m64-n64-k128 で ×0.735 < ×1.050
 *   - reg64x16r8x4w4: bmm-b16-m64-n64-k128 で ×0.430 < ×1.050
 *   - reg64x32r4x4w8: bmm-b16-m64-n64-k128 で ×0.733 < ×1.050
 *   - reg64x32r4x8w4: bmm-b16-m64-n64-k128 で ×0.542 < ×1.050
 *   - reg64x32r8x4w8: bmm-b16-m64-n64-k128 で ×0.557 < ×1.050
 *   - reg64x32r8x8w4: bmm-b16-m64-n64-k128 で ×0.311 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m64-n64-k128 で ×0.556 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m64-n64-k128 で ×0.481 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m64-n64-k128 で ×0.465 < ×1.050
 *   - reg64x64r8x8w8: bmm-b16-m64-n64-k128 で ×0.343 < ×1.050
 *   - reg8x16r1x4w4: linear-m64-n3072-k1024 で ×0.689 < ×1.050
 *   - reg8x16r2x4w4: linear-m64-n3072-k1024 で ×0.423 < ×1.050
 *   - reg8x32r1x8w4: linear-m64-n3072-k1024 で ×0.801 < ×1.050
 *   - reg8x32r2x4w8: linear-m64-n3072-k1024 で ×0.745 < ×1.050
 *   - reg8x32r2x8w4: linear-m64-n3072-k1024 で ×0.481 < ×1.050
 *   - reg8x64r2x8w8: linear-m64-n3072-k1024 で ×0.529 < ×1.050
 * - gemmRows 65〜128（linear / matmul / bmm の行数 65〜128・4 ケース）: 既定 reg64x32r4x4w8 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - reg128x128r8x8w16: bmm-b16-m128-n64-k128 で ×0.289 < ×1.050
 *   - reg128x16r8x4w4: linear-m128-n2048-k1024 で ×0.551 < ×1.050
 *   - reg128x32r8x4w8: bmm-b16-m128-n64-k128 で ×0.582 < ×1.050
 *   - reg128x32r8x8w4: bmm-b16-m128-n64-k128 で ×0.430 < ×1.050
 *   - reg128x64r8x4w16: bmm-b16-m128-n64-k128 で ×0.446 < ×1.050
 *   - reg128x64r8x8w8: bmm-b16-m128-n64-k128 で ×0.424 < ×1.050
 *   - reg16x128r4x8w16: linear-m128-n2048-k1024 で ×0.453 < ×1.050
 *   - reg16x16r1x4w4: linear-m128-n2048-k1024 で ×0.641 < ×1.050
 *   - reg16x16r2x4w4: linear-m128-n2048-k1024 で ×0.530 < ×1.050
 *   - reg16x16r4x4w4: linear-m128-n2048-k1024 で ×0.381 < ×1.050
 *   - reg16x32r1x8w4: linear-m128-n2048-k1024 で ×0.730 < ×1.050
 *   - reg16x32r2x4w8: linear-m128-n2048-k1024 で ×0.675 < ×1.050
 *   - reg16x32r2x8w4: linear-m128-n2048-k1024 で ×0.634 < ×1.050
 *   - reg16x32r4x4w8: linear-m128-n2048-k1024 で ×0.645 < ×1.050
 *   - reg16x32r4x8w4: linear-m128-n2048-k1024 で ×0.387 < ×1.050
 *   - reg16x64r2x8w8: linear-m128-n2048-k1024 で ×0.709 < ×1.050
 *   - reg16x64r4x4w16: linear-m128-n2048-k1024 で ×0.750 < ×1.050
 *   - reg16x64r4x8w8: linear-m128-n2048-k1024 で ×0.441 < ×1.050
 *   - reg32x128r4x8w16: linear-m128-n2048-k1024 で ×0.747 < ×1.050
 *   - reg32x128r8x8w16: linear-m128-n2048-k1024 で ×0.397 < ×1.050
 *   - reg32x16r2x4w4: matmul-m128-n2048-k1024 で ×0.623 < ×1.050
 *   - reg32x16r4x4w4: linear-m128-n2048-k1024 で ×0.580 < ×1.050
 *   - reg32x16r8x4w4: linear-m128-n2048-k1024 で ×0.437 < ×1.050
 *   - reg32x32r2x4w8: linear-m128-n2048-k1024 で ×0.888 < ×1.050
 *   - reg32x32r2x8w4: linear-m128-n2048-k1024 で ×0.864 < ×1.050
 *   - reg32x32r4x4w8: linear-m128-n2048-k1024 で ×0.883 < ×1.050
 *   - reg32x32r4x8w4: linear-m128-n2048-k1024 で ×0.581 < ×1.050
 *   - reg32x32r8x4w8: linear-m128-n2048-k1024 で ×0.602 < ×1.050
 *   - reg32x32r8x8w4: linear-m128-n2048-k1024 で ×0.361 < ×1.050
 *   - reg32x64r2x8w8: linear-m128-n2048-k1024 で ×0.870 < ×1.050
 *   - reg32x64r4x4w16: linear-m128-n2048-k1024 で ×0.898 < ×1.050
 *   - reg32x64r4x8w8: linear-m128-n2048-k1024 で ×0.741 < ×1.050
 *   - reg32x64r8x4w16: linear-m128-n2048-k1024 で ×0.790 < ×1.050
 *   - reg32x64r8x8w8: linear-m128-n2048-k1024 で ×0.357 < ×1.050
 *   - reg4x16r1x4w4: linear-m128-n2048-k1024 で ×0.174 < ×1.050
 *   - reg4x32r1x8w4: linear-m128-n2048-k1024 で ×0.171 < ×1.050
 *   - reg64x128r4x8w16: bmm-b16-m128-n64-k128 で ×0.494 < ×1.050
 *   - reg64x128r8x8w16: bmm-b16-m128-n64-k128 で ×0.480 < ×1.050
 *   - reg64x16r4x4w4: matmul-m128-n2048-k1024 で ×0.627 < ×1.050
 *   - reg64x16r8x4w4: linear-m128-n2048-k1024 で ×0.492 < ×1.050
 *   - reg64x32r4x8w4: bmm-b16-m128-n64-k64 で ×0.733 < ×1.050
 *   - reg64x32r8x4w8: bmm-b16-m128-n64-k128 で ×0.761 < ×1.050
 *   - reg64x32r8x8w4: linear-m128-n2048-k1024 で ×0.427 < ×1.050
 *   - reg64x64r4x4w16: bmm-b16-m128-n64-k128 で ×0.763 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m128-n64-k128 で ×0.659 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m128-n64-k128 で ×0.642 < ×1.050
 *   - reg64x64r8x8w8: bmm-b16-m128-n64-k128 で ×0.474 < ×1.050
 *   - reg8x16r1x4w4: linear-m128-n2048-k1024 で ×0.427 < ×1.050
 *   - reg8x16r2x4w4: linear-m128-n2048-k1024 で ×0.258 < ×1.050
 *   - reg8x32r1x8w4: linear-m128-n2048-k1024 で ×0.499 < ×1.050
 *   - reg8x32r2x4w8: linear-m128-n2048-k1024 で ×0.471 < ×1.050
 *   - reg8x32r2x8w4: linear-m128-n2048-k1024 で ×0.307 < ×1.050
 *   - reg8x64r2x8w8: linear-m128-n2048-k1024 で ×0.426 < ×1.050
 * - gemmRows 129〜256（linear / matmul / bmm の行数 129〜256・4 ケース）: 採用 reg64x64r4x4w16 ×1.203（×1.149〜×1.291）
 *   - reg128x128r8x8w16: bmm-b16-m256-n64-k128 で ×0.450 < ×1.050
 *   - reg128x16r8x4w4: matmul-m256-n2048-k1024 で ×0.566 < ×1.050
 *   - reg128x32r8x4w8: bmm-b16-m256-n64-k128 で ×0.898 < ×1.050
 *   - reg128x32r8x8w4: bmm-b16-m256-n64-k64 で ×0.655 < ×1.050
 *   - reg128x64r8x4w16: bmm-b16-m256-n64-k128 で ×0.694 < ×1.050
 *   - reg128x64r8x8w8: bmm-b16-m256-n64-k64 で ×0.645 < ×1.050
 *   - reg16x128r4x8w16: linear-m256-n2048-k1024 で ×0.693 < ×1.050
 *   - reg16x16r1x4w4: linear-m256-n2048-k1024 で ×0.628 < ×1.050
 *   - reg16x16r2x4w4: linear-m256-n2048-k1024 で ×0.476 < ×1.050
 *   - reg16x16r4x4w4: linear-m256-n2048-k1024 で ×0.353 < ×1.050
 *   - reg16x32r1x8w4: linear-m256-n2048-k1024 で ×0.719 < ×1.050
 *   - reg16x32r2x4w8: linear-m256-n2048-k1024 で ×0.668 < ×1.050
 *   - reg16x32r2x8w4: linear-m256-n2048-k1024 で ×0.664 < ×1.050
 *   - reg16x32r4x4w8: linear-m256-n2048-k1024 で ×0.658 < ×1.050
 *   - reg16x32r4x8w4: linear-m256-n2048-k1024 で ×0.462 < ×1.050
 *   - reg16x64r2x8w8: linear-m256-n2048-k1024 で ×0.709 < ×1.050
 *   - reg16x64r4x4w16: linear-m256-n2048-k1024 で ×0.791 < ×1.050
 *   - reg16x64r4x8w8: linear-m256-n2048-k1024 で ×0.678 < ×1.050
 *   - reg32x128r4x8w16: bmm-b16-m256-n64-k64 で ×0.752 < ×1.050
 *   - reg32x128r8x8w16: bmm-b16-m256-n64-k128 で ×0.684 < ×1.050
 *   - reg32x16r2x4w4: linear-m256-n2048-k1024 で ×0.619 < ×1.050
 *   - reg32x16r4x4w4: linear-m256-n2048-k1024 で ×0.584 < ×1.050
 *   - reg32x16r8x4w4: linear-m256-n2048-k1024 で ×0.464 < ×1.050
 *   - reg32x32r2x4w8: linear-m256-n2048-k1024 で ×0.837 < ×1.050
 *   - reg32x32r2x8w4: linear-m256-n2048-k1024 で ×0.867 < ×1.050
 *   - reg32x32r4x4w8: linear-m256-n2048-k1024 で ×0.855 < ×1.050
 *   - reg32x32r4x8w4: linear-m256-n2048-k1024 で ×0.806 < ×1.050
 *   - reg32x32r8x4w8: linear-m256-n2048-k1024 で ×0.813 < ×1.050
 *   - reg32x32r8x8w4: linear-m256-n2048-k1024 で ×0.520 < ×1.050
 *   - reg32x64r2x8w8: linear-m256-n2048-k1024 で ×0.960 < ×1.050
 *   - reg32x64r4x4w16: linear-m256-n2048-k1024 で ×1.035 < ×1.050
 *   - reg32x64r4x8w8: bmm-b16-m256-n64-k64 で ×0.969 < ×1.050
 *   - reg32x64r8x4w16: linear-m256-n2048-k1024 で ×0.996 < ×1.050
 *   - reg32x64r8x8w8: linear-m256-n2048-k1024 で ×0.657 < ×1.050
 *   - reg4x16r1x4w4: linear-m256-n2048-k1024 で ×0.166 < ×1.050
 *   - reg4x32r1x8w4: linear-m256-n2048-k1024 で ×0.175 < ×1.050
 *   - reg64x128r4x8w16: bmm-b16-m256-n64-k64 で ×0.726 < ×1.050
 *   - reg64x128r8x8w16: bmm-b16-m256-n64-k64 で ×0.713 < ×1.050
 *   - reg64x16r4x4w4: matmul-m256-n2048-k1024 で ×0.586 < ×1.050
 *   - reg64x16r8x4w4: matmul-m256-n2048-k1024 で ×0.549 < ×1.050
 *   - reg64x32r4x8w4: bmm-b16-m256-n64-k64 で ×0.879 < ×1.050
 *   - reg64x32r8x4w8: bmm-b16-m256-n64-k64 で ×0.939 < ×1.050
 *   - reg64x32r8x8w4: bmm-b16-m256-n64-k64 で ×0.639 < ×1.050
 *   - reg64x64r4x8w8: bmm-b16-m256-n64-k64 で ×0.945 < ×1.050
 *   - reg64x64r8x4w16: bmm-b16-m256-n64-k64 で ×0.981 < ×1.050
 *   - reg64x64r8x8w8: bmm-b16-m256-n64-k64 で ×0.705 < ×1.050
 *   - reg8x16r1x4w4: linear-m256-n2048-k1024 で ×0.428 < ×1.050
 *   - reg8x16r2x4w4: linear-m256-n2048-k1024 で ×0.259 < ×1.050
 *   - reg8x32r1x8w4: linear-m256-n2048-k1024 で ×0.464 < ×1.050
 *   - reg8x32r2x4w8: linear-m256-n2048-k1024 で ×0.430 < ×1.050
 *   - reg8x32r2x8w4: linear-m256-n2048-k1024 で ×0.279 < ×1.050
 *   - reg8x64r2x8w8: linear-m256-n2048-k1024 で ×0.462 < ×1.050
 * - gemmRows 257〜512（linear / matmul / bmm の行数 257〜512・5 ケース）: 採用 reg64x64r8x4w16 ×1.330（×1.184〜×1.451）
 *   - reg128x128r8x8w16: bmm-b16-m512-n64-k64 で ×0.709 < ×1.050
 *   - reg128x16r8x4w4: matmul-m512-n2048-k1024 で ×0.552 < ×1.050
 *   - reg128x32r8x4w8: bmm-b16-m512-n64-k64 で ×0.966 < ×1.050
 *   - reg128x32r8x8w4: bmm-b16-m512-n64-k64 で ×0.832 < ×1.050
 *   - reg128x64r8x4w16: 幾何平均 ×1.273（採用 ×1.330 に届かない）
 *   - reg128x64r8x8w8: bmm-b16-m512-n64-k64 で ×0.993 < ×1.050
 *   - reg16x128r4x8w16: bmm-b16-m512-n64-k512 で ×0.726 < ×1.050
 *   - reg16x16r1x4w4: linear-m512-n2048-k1024 で ×0.623 < ×1.050
 *   - reg16x16r2x4w4: linear-m512-n2048-k1024 で ×0.511 < ×1.050
 *   - reg16x16r4x4w4: linear-m512-n2048-k1024 で ×0.376 < ×1.050
 *   - reg16x32r1x8w4: linear-m512-n2048-k1024 で ×0.751 < ×1.050
 *   - reg16x32r2x4w8: linear-m512-n2048-k1024 で ×0.691 < ×1.050
 *   - reg16x32r2x8w4: linear-m512-n2048-k1024 で ×0.650 < ×1.050
 *   - reg16x32r4x4w8: linear-m512-n2048-k1024 で ×0.634 < ×1.050
 *   - reg16x32r4x8w4: linear-m512-n2048-k1024 で ×0.442 < ×1.050
 *   - reg16x64r2x8w8: linear-m512-n2048-k1024 で ×0.718 < ×1.050
 *   - reg16x64r4x4w16: linear-m512-n2048-k1024 で ×0.794 < ×1.050
 *   - reg16x64r4x8w8: linear-m512-n2048-k1024 で ×0.767 < ×1.050
 *   - reg32x128r4x8w16: bmm-b16-m512-n64-k64 で ×0.785 < ×1.050
 *   - reg32x128r8x8w16: bmm-b16-m512-n64-k64 で ×0.710 < ×1.050
 *   - reg32x16r2x4w4: bmm-b16-m512-n512-k64 で ×0.594 < ×1.050
 *   - reg32x16r4x4w4: matmul-m512-n2048-k1024 で ×0.540 < ×1.050
 *   - reg32x16r8x4w4: linear-m512-n2048-k1024 で ×0.441 < ×1.050
 *   - reg32x32r2x4w8: linear-m512-n2048-k1024 で ×0.908 < ×1.050
 *   - reg32x32r2x8w4: linear-m512-n2048-k1024 で ×0.875 < ×1.050
 *   - reg32x32r4x4w8: linear-m512-n2048-k1024 で ×0.852 < ×1.050
 *   - reg32x32r4x8w4: linear-m512-n2048-k1024 で ×0.859 < ×1.050
 *   - reg32x32r8x4w8: linear-m512-n2048-k1024 で ×0.849 < ×1.050
 *   - reg32x32r8x8w4: linear-m512-n2048-k1024 で ×0.489 < ×1.050
 *   - reg32x64r2x8w8: linear-m512-n2048-k1024 で ×0.970 < ×1.050
 *   - reg32x64r4x4w16: 幾何平均 ×1.209（採用 ×1.330 に届かない）
 *   - reg32x64r4x8w8: linear-m512-n2048-k1024 で ×1.048 < ×1.050
 *   - reg32x64r8x4w16: 幾何平均 ×1.236（採用 ×1.330 に届かない）
 *   - reg32x64r8x8w8: bmm-b16-m512-n64-k64 で ×0.930 < ×1.050
 *   - reg4x16r1x4w4: linear-m512-n2048-k1024 で ×0.167 < ×1.050
 *   - reg4x32r1x8w4: linear-m512-n2048-k1024 で ×0.174 < ×1.050
 *   - reg64x128r4x8w16: bmm-b16-m512-n64-k64 で ×0.759 < ×1.050
 *   - reg64x128r8x8w16: bmm-b16-m512-n64-k64 で ×0.732 < ×1.050
 *   - reg64x16r4x4w4: matmul-m512-n2048-k1024 で ×0.557 < ×1.050
 *   - reg64x16r8x4w4: bmm-b16-m512-n64-k512 で ×0.557 < ×1.050
 *   - reg64x32r4x8w4: bmm-b16-m512-n64-k64 で ×0.953 < ×1.050
 *   - reg64x32r8x4w8: linear-m512-n2048-k1024 で ×0.977 < ×1.050
 *   - reg64x32r8x8w4: bmm-b16-m512-n64-k64 で ×0.859 < ×1.050
 *   - reg64x64r4x4w16: 幾何平均 ×1.278（採用 ×1.330 に届かない）
 *   - reg64x64r4x8w8: bmm-b16-m512-n64-k64 で ×1.017 < ×1.050
 *   - reg64x64r8x8w8: bmm-b16-m512-n64-k64 で ×0.986 < ×1.050
 *   - reg8x16r1x4w4: linear-m512-n2048-k1024 で ×0.428 < ×1.050
 *   - reg8x16r2x4w4: linear-m512-n2048-k1024 で ×0.261 < ×1.050
 *   - reg8x32r1x8w4: linear-m512-n2048-k1024 で ×0.490 < ×1.050
 *   - reg8x32r2x4w8: linear-m512-n2048-k1024 で ×0.461 < ×1.050
 *   - reg8x32r2x8w4: linear-m512-n2048-k1024 で ×0.301 < ×1.050
 *   - reg8x64r2x8w8: linear-m512-n2048-k1024 で ×0.454 < ×1.050
 * - gemmRows > 512（linear / matmul / bmm の行数 > 512・7 ケース）: 既定 reg128x128r8x8w16 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - reg128x16r8x4w4: linear-m1024-n2048-k2048 で ×0.353 < ×1.050
 *   - reg128x32r8x4w8: linear-m4096-n8192-k2048 で ×0.625 < ×1.050
 *   - reg128x32r8x8w4: linear-m1024-n2048-k8192 で ×0.572 < ×1.050
 *   - reg128x64r8x4w16: linear-m1024-n2048-k8192 で ×0.836 < ×1.050
 *   - reg128x64r8x8w8: linear-m1024-n2048-k2048 で ×0.905 < ×1.050
 *   - reg16x128r4x8w16: linear-m1024-n2048-k2048 で ×0.485 < ×1.050
 *   - reg16x16r1x4w4: linear-m4096-n8192-k2048 で ×0.340 < ×1.050
 *   - reg16x16r2x4w4: linear-m4096-n8192-k2048 で ×0.277 < ×1.050
 *   - reg16x16r4x4w4: linear-m4096-n8192-k2048 で ×0.206 < ×1.050
 *   - reg16x32r1x8w4: linear-m4096-n8192-k2048 で ×0.412 < ×1.050
 *   - reg16x32r2x4w8: linear-m4096-n8192-k2048 で ×0.376 < ×1.050
 *   - reg16x32r2x8w4: linear-m4096-n8192-k2048 で ×0.381 < ×1.050
 *   - reg16x32r4x4w8: linear-m4096-n8192-k2048 で ×0.371 < ×1.050
 *   - reg16x32r4x8w4: linear-m4096-n8192-k2048 で ×0.257 < ×1.050
 *   - reg16x64r2x8w8: linear-m4096-n8192-k2048 で ×0.418 < ×1.050
 *   - reg16x64r4x4w16: linear-m4096-n8192-k2048 で ×0.466 < ×1.050
 *   - reg16x64r4x8w8: linear-m1024-n2048-k2048 で ×0.430 < ×1.050
 *   - reg32x128r4x8w16: linear-m1024-n2048-k2048 で ×0.706 < ×1.050
 *   - reg32x128r8x8w16: linear-m1024-n2048-k8192 で ×0.575 < ×1.050
 *   - reg32x16r2x4w4: linear-m4096-n8192-k2048 で ×0.342 < ×1.050
 *   - reg32x16r4x4w4: linear-m4096-n8192-k2048 で ×0.318 < ×1.050
 *   - reg32x16r8x4w4: linear-m4096-n8192-k2048 で ×0.256 < ×1.050
 *   - reg32x32r2x4w8: linear-m4096-n8192-k2048 で ×0.501 < ×1.050
 *   - reg32x32r2x8w4: linear-m4096-n8192-k2048 で ×0.513 < ×1.050
 *   - reg32x32r4x4w8: linear-m4096-n8192-k2048 で ×0.493 < ×1.050
 *   - reg32x32r4x8w4: linear-m4096-n8192-k2048 で ×0.478 < ×1.050
 *   - reg32x32r8x4w8: linear-m4096-n8192-k2048 で ×0.473 < ×1.050
 *   - reg32x32r8x8w4: linear-m1024-n2048-k8192 で ×0.330 < ×1.050
 *   - reg32x64r2x8w8: linear-m4096-n8192-k2048 で ×0.578 < ×1.050
 *   - reg32x64r4x4w16: linear-m4096-n8192-k2048 で ×0.637 < ×1.050
 *   - reg32x64r4x8w8: linear-m1024-n2048-k2048 で ×0.615 < ×1.050
 *   - reg32x64r8x4w16: linear-m1024-n2048-k2048 で ×0.651 < ×1.050
 *   - reg32x64r8x8w8: linear-m1024-n2048-k8192 で ×0.512 < ×1.050
 *   - reg4x16r1x4w4: linear-m4096-n8192-k2048 で ×0.089 < ×1.050
 *   - reg4x32r1x8w4: linear-m4096-n8192-k2048 で ×0.095 < ×1.050
 *   - reg64x128r4x8w16: linear-m1024-n2048-k8192 で ×0.832 < ×1.050
 *   - reg64x128r8x8w16: linear-m1024-n2048-k2048 で ×0.930 < ×1.050
 *   - reg64x16r4x4w4: linear-m4096-n8192-k2048 で ×0.357 < ×1.050
 *   - reg64x16r8x4w4: linear-m1024-n2048-k2048 で ×0.332 < ×1.050
 *   - reg64x32r4x4w8: linear-m4096-n8192-k2048 で ×0.589 < ×1.050
 *   - reg64x32r4x8w4: linear-m4096-n8192-k2048 で ×0.571 < ×1.050
 *   - reg64x32r8x4w8: linear-m4096-n8192-k2048 で ×0.566 < ×1.050
 *   - reg64x32r8x8w4: linear-m1024-n2048-k8192 で ×0.502 < ×1.050
 *   - reg64x64r4x4w16: linear-m1024-n2048-k2048 で ×0.744 < ×1.050
 *   - reg64x64r4x8w8: linear-m1024-n2048-k2048 で ×0.762 < ×1.050
 *   - reg64x64r8x4w16: linear-m1024-n2048-k2048 で ×0.790 < ×1.050
 *   - reg64x64r8x8w8: linear-m1024-n2048-k8192 で ×0.687 < ×1.050
 *   - reg8x16r1x4w4: linear-m4096-n8192-k2048 で ×0.233 < ×1.050
 *   - reg8x16r2x4w4: linear-m4096-n8192-k2048 で ×0.142 < ×1.050
 *   - reg8x32r1x8w4: linear-m4096-n8192-k2048 で ×0.269 < ×1.050
 *   - reg8x32r2x4w8: linear-m4096-n8192-k2048 で ×0.252 < ×1.050
 *   - reg8x32r2x8w4: linear-m4096-n8192-k2048 で ×0.166 < ×1.050
 *   - reg8x64r2x8w8: linear-m4096-n8192-k2048 で ×0.268 < ×1.050
 * - attention.qk（融合 attention f32 ①QK・4 ケース）: 既定 reg128x128r8x8w16 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - reg128x16r8x4w4: attention-qk-self-m4096-n4096 で ×0.368 < ×1.050
 *   - reg128x32r8x4w8: attention-qk-self-m4096-n4096 で ×0.647 < ×1.050
 *   - reg128x32r8x8w4: attention-qk-self-m4096-n4096 で ×0.627 < ×1.050
 *   - reg128x64r8x4w16: attention-qk-self-m4096-n4096 で ×0.857 < ×1.050
 *   - reg128x64r8x8w8: attention-qk-cross-m4096-n512 で ×0.910 < ×1.050
 *   - reg16x128r4x8w16: attention-qk-self-m4096-n4096 で ×0.463 < ×1.050
 *   - reg16x16r1x4w4: attention-qk-self-m4096-n4096 で ×0.322 < ×1.050
 *   - reg16x16r2x4w4: attention-qk-self-m4096-n4096 で ×0.269 < ×1.050
 *   - reg16x16r4x4w4: attention-qk-self-m4096-n4096 で ×0.203 < ×1.050
 *   - reg16x32r1x8w4: attention-qk-self-m4096-n4096 で ×0.386 < ×1.050
 *   - reg16x32r2x4w8: attention-qk-self-m4096-n4096 で ×0.355 < ×1.050
 *   - reg16x32r2x8w4: attention-qk-self-m4096-n4096 で ×0.360 < ×1.050
 *   - reg16x32r4x4w8: attention-qk-self-m4096-n4096 で ×0.353 < ×1.050
 *   - reg16x32r4x8w4: attention-qk-self-m4096-n4096 で ×0.254 < ×1.050
 *   - reg16x64r2x8w8: attention-qk-self-m4096-n4096 で ×0.393 < ×1.050
 *   - reg16x64r4x4w16: attention-qk-self-m4096-n4096 で ×0.432 < ×1.050
 *   - reg16x64r4x8w8: attention-qk-self-m4096-n4096 で ×0.412 < ×1.050
 *   - reg32x128r4x8w16: attention-qk-self-m4096-n4096 で ×0.689 < ×1.050
 *   - reg32x128r8x8w16: attention-qk-self-m1024-n1024 で ×0.686 < ×1.050
 *   - reg32x16r2x4w4: attention-qk-self-m4096-n4096 で ×0.341 < ×1.050
 *   - reg32x16r4x4w4: attention-qk-self-m4096-n4096 で ×0.318 < ×1.050
 *   - reg32x16r8x4w4: attention-qk-self-m4096-n4096 で ×0.261 < ×1.050
 *   - reg32x32r2x4w8: attention-qk-self-m4096-n4096 で ×0.487 < ×1.050
 *   - reg32x32r2x8w4: attention-qk-self-m4096-n4096 で ×0.500 < ×1.050
 *   - reg32x32r4x4w8: attention-qk-self-m4096-n4096 で ×0.482 < ×1.050
 *   - reg32x32r4x8w4: attention-qk-self-m4096-n4096 で ×0.475 < ×1.050
 *   - reg32x32r8x4w8: attention-qk-self-m4096-n4096 で ×0.471 < ×1.050
 *   - reg32x32r8x8w4: attention-qk-self-m1024-n1024 で ×0.349 < ×1.050
 *   - reg32x64r2x8w8: attention-qk-self-m4096-n4096 で ×0.556 < ×1.050
 *   - reg32x64r4x4w16: attention-qk-self-m4096-n4096 で ×0.619 < ×1.050
 *   - reg32x64r4x8w8: attention-qk-self-m4096-n4096 で ×0.604 < ×1.050
 *   - reg32x64r8x4w16: attention-qk-self-m4096-n4096 で ×0.643 < ×1.050
 *   - reg32x64r8x8w8: attention-qk-self-m1024-n1024 で ×0.614 < ×1.050
 *   - reg4x16r1x4w4: attention-qk-self-m4096-n4096 で ×0.085 < ×1.050
 *   - reg4x32r1x8w4: attention-qk-self-m4096-n4096 で ×0.090 < ×1.050
 *   - reg64x128r4x8w16: attention-qk-self-m4096-n4096 で ×0.851 < ×1.050
 *   - reg64x128r8x8w16: attention-qk-cross-m4096-n512 で ×0.923 < ×1.050
 *   - reg64x16r4x4w4: attention-qk-self-m4096-n4096 で ×0.364 < ×1.050
 *   - reg64x16r8x4w4: attention-qk-self-m4096-n4096 で ×0.344 < ×1.050
 *   - reg64x32r4x4w8: attention-qk-self-m4096-n4096 で ×0.590 < ×1.050
 *   - reg64x32r4x8w4: attention-qk-self-m4096-n4096 で ×0.584 < ×1.050
 *   - reg64x32r8x4w8: attention-qk-self-m4096-n4096 で ×0.576 < ×1.050
 *   - reg64x32r8x8w4: attention-qk-self-m4096-n4096 で ×0.561 < ×1.050
 *   - reg64x64r4x4w16: attention-qk-self-m4096-n4096 で ×0.733 < ×1.050
 *   - reg64x64r4x8w8: attention-qk-self-m4096-n4096 で ×0.765 < ×1.050
 *   - reg64x64r8x4w16: attention-qk-self-m4096-n4096 で ×0.796 < ×1.050
 *   - reg64x64r8x8w8: attention-qk-cross-m4096-n512 で ×0.775 < ×1.050
 *   - reg8x16r1x4w4: attention-qk-self-m4096-n4096 で ×0.217 < ×1.050
 *   - reg8x16r2x4w4: attention-qk-self-m4096-n4096 で ×0.138 < ×1.050
 *   - reg8x32r1x8w4: attention-qk-self-m4096-n4096 で ×0.247 < ×1.050
 *   - reg8x32r2x4w8: attention-qk-self-m4096-n4096 で ×0.232 < ×1.050
 *   - reg8x32r2x8w4: attention-qk-self-m4096-n4096 で ×0.160 < ×1.050
 *   - reg8x64r2x8w8: attention-qk-self-m4096-n4096 で ×0.246 < ×1.050
 * - attention.pv（融合 attention f32 ③PV・4 ケース）: 既定 reg128x128r8x8w16 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - reg128x16r8x4w4: attention-pv-cross-m1024-n512 で ×0.327 < ×1.050
 *   - reg128x32r8x4w8: attention-pv-cross-m1024-n512 で ×0.590 < ×1.050
 *   - reg128x32r8x8w4: attention-pv-self-m1024-n1024 で ×0.452 < ×1.050
 *   - reg128x64r8x4w16: attention-pv-self-m1024-n1024 で ×0.800 < ×1.050
 *   - reg128x64r8x8w8: attention-pv-cross-m1024-n512 で ×0.881 < ×1.050
 *   - reg16x128r4x8w16: attention-pv-cross-m1024-n512 で ×0.746 < ×1.050
 *   - reg16x16r1x4w4: attention-pv-cross-m1024-n512 で ×0.436 < ×1.050
 *   - reg16x16r2x4w4: attention-pv-cross-m1024-n512 で ×0.323 < ×1.050
 *   - reg16x16r4x4w4: attention-pv-cross-m1024-n512 で ×0.299 < ×1.050
 *   - reg16x32r1x8w4: attention-pv-cross-m1024-n512 で ×0.612 < ×1.050
 *   - reg16x32r2x4w8: attention-pv-cross-m1024-n512 で ×0.521 < ×1.050
 *   - reg16x32r2x8w4: attention-pv-cross-m1024-n512 で ×0.568 < ×1.050
 *   - reg16x32r4x4w8: attention-pv-cross-m1024-n512 で ×0.529 < ×1.050
 *   - reg16x32r4x8w4: attention-pv-cross-m1024-n512 で ×0.437 < ×1.050
 *   - reg16x64r2x8w8: attention-pv-cross-m1024-n512 で ×0.629 < ×1.050
 *   - reg16x64r4x4w16: attention-pv-cross-m1024-n512 で ×0.712 < ×1.050
 *   - reg16x64r4x8w8: attention-pv-cross-m1024-n512 で ×0.649 < ×1.050
 *   - reg32x128r4x8w16: attention-pv-cross-m1024-n512 で ×0.875 < ×1.050
 *   - reg32x128r8x8w16: attention-pv-self-m1024-n1024 で ×0.677 < ×1.050
 *   - reg32x16r2x4w4: attention-pv-cross-m1024-n512 で ×0.370 < ×1.050
 *   - reg32x16r4x4w4: attention-pv-cross-m1024-n512 で ×0.343 < ×1.050
 *   - reg32x16r8x4w4: attention-pv-cross-m1024-n512 で ×0.299 < ×1.050
 *   - reg32x32r2x4w8: attention-pv-cross-m1024-n512 で ×0.615 < ×1.050
 *   - reg32x32r2x8w4: attention-pv-cross-m1024-n512 で ×0.641 < ×1.050
 *   - reg32x32r4x4w8: attention-pv-cross-m1024-n512 で ×0.601 < ×1.050
 *   - reg32x32r4x8w4: attention-pv-cross-m1024-n512 で ×0.560 < ×1.050
 *   - reg32x32r8x4w8: attention-pv-cross-m1024-n512 で ×0.539 < ×1.050
 *   - reg32x32r8x8w4: attention-pv-self-m1024-n1024 で ×0.358 < ×1.050
 *   - reg32x64r2x8w8: attention-pv-cross-m1024-n512 で ×0.717 < ×1.050
 *   - reg32x64r4x4w16: attention-pv-cross-m1024-n512 で ×0.819 < ×1.050
 *   - reg32x64r4x8w8: attention-pv-cross-m1024-n512 で ×0.753 < ×1.050
 *   - reg32x64r8x4w16: attention-pv-cross-m1024-n512 で ×0.788 < ×1.050
 *   - reg32x64r8x8w8: attention-pv-self-m1024-n1024 で ×0.614 < ×1.050
 *   - reg4x16r1x4w4: attention-pv-cross-m1024-n512 で ×0.200 < ×1.050
 *   - reg4x32r1x8w4: attention-pv-cross-m1024-n512 で ×0.283 < ×1.050
 *   - reg64x128r4x8w16: attention-pv-cross-m1024-n512 で ×0.910 < ×1.050
 *   - reg64x128r8x8w16: attention-pv-cross-m1024-n512 で ×1.014 < ×1.050
 *   - reg64x16r4x4w4: attention-pv-cross-m1024-n512 で ×0.364 < ×1.050
 *   - reg64x16r8x4w4: attention-pv-cross-m1024-n512 で ×0.319 < ×1.050
 *   - reg64x32r4x4w8: attention-pv-cross-m1024-n512 で ×0.636 < ×1.050
 *   - reg64x32r4x8w4: attention-pv-cross-m1024-n512 で ×0.597 < ×1.050
 *   - reg64x32r8x4w8: attention-pv-cross-m1024-n512 で ×0.577 < ×1.050
 *   - reg64x32r8x8w4: attention-pv-self-m1024-n1024 で ×0.454 < ×1.050
 *   - reg64x64r4x4w16: attention-pv-cross-m1024-n512 で ×0.762 < ×1.050
 *   - reg64x64r4x8w8: attention-pv-cross-m1024-n512 で ×0.815 < ×1.050
 *   - reg64x64r8x4w16: attention-pv-self-m1024-n1024 で ×0.824 < ×1.050
 *   - reg64x64r8x8w8: attention-pv-self-m1024-n1024 で ×0.601 < ×1.050
 *   - reg8x16r1x4w4: attention-pv-cross-m1024-n512 で ×0.331 < ×1.050
 *   - reg8x16r2x4w4: attention-pv-cross-m1024-n512 で ×0.263 < ×1.050
 *   - reg8x32r1x8w4: attention-pv-cross-m1024-n512 で ×0.508 < ×1.050
 *   - reg8x32r2x4w8: attention-pv-cross-m1024-n512 で ×0.431 < ×1.050
 *   - reg8x32r2x8w4: attention-pv-cross-m1024-n512 で ×0.386 < ×1.050
 *   - reg8x64r2x8w8: attention-pv-cross-m1024-n512 で ×0.495 < ×1.050
 * - conv2d.rows64（conv2d implicit GEMM の m タイル 64 行・2 ケース）: 既定 igemm64x128:wg16x8 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - igemm128x128:wg16x16: conv2d-c192-256x256 で ×0.848 < ×1.050
 *   - igemm128x64:wg16x16: conv2d-c192-256x256 で ×0.783 < ×1.050
 *   - igemm128x64:wg8x16: conv2d-c192-256x256 で ×0.805 < ×1.050
 *   - igemm16x128:wg16x4: conv2d-c384-128x128 で ×0.602 < ×1.050
 *   - igemm16x64:wg16x4: conv2d-c384-128x128 で ×0.601 < ×1.050
 *   - igemm16x64:wg8x4: conv2d-c384-128x128 で ×0.581 < ×1.050
 *   - igemm16x64:wg8x8: conv2d-c384-128x128 で ×0.587 < ×1.050
 *   - igemm32x128:wg16x4: conv2d-c384-128x128 で ×0.800 < ×1.050
 *   - igemm32x128:wg16x8: conv2d-c384-128x128 で ×0.835 < ×1.050
 *   - igemm32x64:wg16x4: conv2d-c192-256x256 で ×0.747 < ×1.050
 *   - igemm32x64:wg16x8: conv2d-c384-128x128 で ×0.793 < ×1.050
 *   - igemm32x64:wg8x16: conv2d-c384-128x128 で ×0.773 < ×1.050
 *   - igemm32x64:wg8x4: conv2d-c192-256x256 で ×0.739 < ×1.050
 *   - igemm32x64:wg8x8: conv2d-c192-256x256 で ×0.806 < ×1.050
 *   - igemm64x128:wg16x16: conv2d-c192-256x256 で ×1.000 < ×1.050
 *   - igemm64x64:wg16x16: conv2d-c192-256x256 で ×0.949 < ×1.050
 *   - igemm64x64:wg16x8: conv2d-c384-128x128 で ×0.934 < ×1.050
 *   - igemm64x64:wg8x16: conv2d-c192-256x256 で ×0.982 < ×1.050
 *   - igemm64x64:wg8x8: conv2d-c192-256x256 で ×0.927 < ×1.050
 *   - igemm8x64:wg8x4: conv2d-c384-128x128 で ×0.363 < ×1.050
 * - conv2d.rows32（conv2d implicit GEMM の m タイル 32 行・1 ケース）: 既定 igemm32x128:wg16x4 のまま（全ケースで出力が一致し ×1.050 以上の幾何が無い）
 *   - igemm128x128:wg16x16: conv2d-c96-512x512 で ×1.032 < ×1.050
 *   - igemm128x64:wg16x16: conv2d-c96-512x512 で ×0.972 < ×1.050
 *   - igemm128x64:wg8x16: conv2d-c96-512x512 で ×0.995 < ×1.050
 *   - igemm16x128:wg16x4: conv2d-c96-512x512 で ×0.774 < ×1.050
 *   - igemm16x64:wg16x4: conv2d-c96-512x512 で ×0.758 < ×1.050
 *   - igemm16x64:wg8x4: conv2d-c96-512x512 で ×0.730 < ×1.050
 *   - igemm16x64:wg8x8: conv2d-c96-512x512 で ×0.744 < ×1.050
 *   - igemm32x128:wg16x8: conv2d-c96-512x512 で ×1.039 < ×1.050
 *   - igemm32x64:wg16x4: conv2d-c96-512x512 で ×0.922 < ×1.050
 *   - igemm32x64:wg16x8: conv2d-c96-512x512 で ×1.003 < ×1.050
 *   - igemm32x64:wg8x16: conv2d-c96-512x512 で ×0.957 < ×1.050
 *   - igemm32x64:wg8x4: conv2d-c96-512x512 で ×0.936 < ×1.050
 *   - igemm32x64:wg8x8: conv2d-c96-512x512 で ×0.993 < ×1.050
 *   - igemm64x128:wg16x16: conv2d-c96-512x512 で ×0.940 < ×1.050
 *   - igemm64x128:wg16x8: conv2d-c96-512x512 で ×0.930 < ×1.050
 *   - igemm64x64:wg16x16: conv2d-c96-512x512 で ×0.904 < ×1.050
 *   - igemm64x64:wg16x8: conv2d-c96-512x512 で ×0.898 < ×1.050
 *   - igemm64x64:wg8x16: conv2d-c96-512x512 で ×0.939 < ×1.050
 *   - igemm64x64:wg8x8: conv2d-c96-512x512 で ×0.891 < ×1.050
 *   - igemm8x64:wg8x4: conv2d-c96-512x512 で ×0.478 < ×1.050
 * - i8a8.linear（i8a8 linear・6 ケース）: 採用 tile128x64r8x4w16x16k16 ×1.194（×1.164〜×1.245）
 *   - tile128x128r8x8w16x16k16: i8a8-linear-m4096-n2048-k2048 で ×0.825 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-linear-m4096-n2048-k2048 で ×0.870 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-linear-m4096-n8192-k2048 で ×0.897 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-linear-m4096-n8192-k2048 で ×1.036 < ×1.050
 *   - tile128x64r8x4w16x16k32: 幾何平均 ×1.150（採用 ×1.194 に届かない）
 *   - tile128x64r8x8w8x16k32: 幾何平均 ×1.161（採用 ×1.194 に届かない）
 *   - tile16x128r4x8w16x4k16: i8a8-linear-m1024-n2048-k2048 で ×0.568 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-linear-m4096-n2048-k8192 で ×0.624 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-linear-m4096-n2048-k8192 で ×0.545 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-linear-m4096-n8192-k2048 で ×0.672 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-linear-m4096-n8192-k2048 で ×0.544 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-linear-m4096-n8192-k2048 で ×0.677 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-linear-m1024-n2048-k2048 で ×0.505 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-linear-m4096-n8192-k2048 で ×0.740 < ×1.050
 *   - tile32x128r4x8w16x8k16: i8a8-linear-m4096-n8192-k2048 で ×0.751 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-linear-m4096-n2048-k8192 で ×0.763 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-linear-m4096-n2048-k2048 で ×0.842 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-linear-m4096-n2048-k2048 で ×0.826 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-linear-m1024-n2048-k8192 で ×0.714 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-linear-m4096-n8192-k2048 で ×0.767 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-linear-m1024-n2048-k2048 で ×0.607 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-linear-m4096-n2048-k8192 で ×0.883 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-linear-m4096-n8192-k2048 で ×0.813 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-linear-m4096-n8192-k2048 で ×0.802 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-linear-m1024-n2048-k8192 で ×0.796 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-linear-m4096-n2048-k8192 で ×0.926 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-linear-m4096-n2048-k2048 で ×0.875 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-linear-m4096-n2048-k8192 で ×0.959 < ×1.050
 *   - tile32x64r8x8w8x4k16: i8a8-linear-m4096-n2048-k2048 で ×0.889 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-linear-m4096-n2048-k8192 で ×0.856 < ×1.050
 *   - tile64x128r4x8w16x16k16: i8a8-linear-m4096-n2048-k8192 で ×0.838 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-linear-m4096-n2048-k8192 で ×0.811 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-linear-m1024-n2048-k2048 で ×0.908 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-linear-m4096-n2048-k2048 で ×0.895 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-linear-m4096-n8192-k2048 で ×0.855 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-linear-m4096-n8192-k2048 で ×0.816 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-linear-m1024-n2048-k2048 で ×0.855 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-linear-m4096-n2048-k8192 で ×0.998 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-linear-m4096-n8192-k2048 で ×0.882 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-linear-m4096-n8192-k2048 で ×0.895 < ×1.050
 *   - tile64x64r4x8w8x16k16: 幾何平均 ×1.082（採用 ×1.194 に届かない）
 *   - tile64x64r4x8w8x16k32: i8a8-linear-m4096-n2048-k8192 で ×1.037 < ×1.050
 *   - tile64x64r8x4w16x8k16: 幾何平均 ×1.124（採用 ×1.194 に届かない）
 *   - tile64x64r8x4w16x8k32: 幾何平均 ×1.111（採用 ×1.194 に届かない）
 *   - tile64x64r8x8w8x8k16: i8a8-linear-m4096-n2048-k8192 で ×0.955 < ×1.050
 *   - tile64x64r8x8w8x8k32: 幾何平均 ×1.104（採用 ×1.194 に届かない）
 * - i8a8.attentionQk（i8a8 attention ①QK・4 ケース）: 採用 tile128x64r8x4w16x16k16 ×1.218（×1.200〜×1.238）
 *   - tile128x128r8x8w16x16k16: i8a8-attention-qk-cross-m1024-n512 で ×0.951 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-attention-qk-cross-m1024-n512 で ×0.894 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-attention-qk-cross-m4096-n512 で ×0.976 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-attention-qk-cross-m4096-n512 で ×1.038 < ×1.050
 *   - tile128x64r8x4w16x16k32: 幾何平均 ×1.196（採用 ×1.218 に届かない）
 *   - tile128x64r8x8w8x16k32: i8a8-attention-qk-cross-m1024-n512 で ×1.015 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile16x128r4x8w16x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile16x32r4x4w8x4k16: i8a8-attention-qk-self-m4096-n4096 で ×0.715 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile16x64r4x4w16x4k16: i8a8-attention-qk-self-m4096-n4096 で ×0.719 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile16x64r4x8w8x4k16: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile16x64r4x8w8x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x128r4x8w16x8k16: i8a8-attention-qk-self-m4096-n4096 で ×0.813 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x128r8x8w16x4k16: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x128r8x8w16x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x32r4x4w8x8k16: i8a8-attention-qk-self-m4096-n4096 で ×0.825 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-attention-qk-self-m4096-n4096 で ×0.802 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-attention-qk-cross-m1024-n512 で ×0.903 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x64r4x4w16x8k16: i8a8-attention-qk-self-m4096-n4096 で ×0.885 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-attention-qk-self-m4096-n4096 で ×0.875 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-attention-qk-self-m4096-n4096 で ×0.939 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x64r8x4w16x4k16: i8a8-attention-qk-cross-m1024-n512 で ×1.000 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x64r8x8w8x4k16: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile32x64r8x8w8x4k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile64x128r4x8w16x16k16: i8a8-attention-qk-cross-m1024-n512 で ×0.878 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-attention-qk-cross-m1024-n512 で ×0.885 < ×1.050
 *   - tile64x128r8x8w16x8k16: i8a8-attention-qk-self-m4096-n4096 で ×0.993 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 *   - tile64x32r4x4w8x16k16: i8a8-attention-qk-cross-m4096-n512 で ×0.894 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-attention-qk-cross-m4096-n512 で ×0.880 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-attention-qk-cross-m1024-n512 で ×0.990 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-attention-qk-cross-m4096-n512 で ×0.974 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-attention-qk-self-m4096-n4096 で ×0.958 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-attention-qk-self-m4096-n4096 で ×0.963 < ×1.050
 *   - tile64x64r4x8w8x16k16: 幾何平均 ×1.087（採用 ×1.218 に届かない）
 *   - tile64x64r4x8w8x16k32: i8a8-attention-qk-self-m4096-n4096 で ×1.008 < ×1.050
 *   - tile64x64r8x4w16x8k16: 幾何平均 ×1.168（採用 ×1.218 に届かない）
 *   - tile64x64r8x4w16x8k32: 幾何平均 ×1.169（採用 ×1.218 に届かない）
 *   - tile64x64r8x8w8x8k16: i8a8-attention-qk-cross-m4096-n512 で ×0.971 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-attention-qk-cross-m1024-n512 で出力が既定と不一致
 * - i8a8.attentionPv（i8a8 attention ③PV・4 ケース）: 採用 tile128x64r8x4w16x16k16 ×1.210（×1.135〜×1.300）
 *   - tile128x128r8x8w16x16k16: i8a8-attention-pv-cross-m4096-n512 で ×0.878 < ×1.050
 *   - tile128x128r8x8w16x16k32: i8a8-attention-pv-cross-m4096-n512 で ×0.883 < ×1.050
 *   - tile128x32r8x4w8x16k16: i8a8-attention-pv-self-m4096-n4096 で ×0.814 < ×1.050
 *   - tile128x32r8x4w8x16k32: i8a8-attention-pv-self-m4096-n4096 で ×0.752 < ×1.050
 *   - tile128x64r8x4w16x16k32: 幾何平均 ×1.175（採用 ×1.210 に届かない）
 *   - tile128x64r8x8w8x16k16: i8a8-attention-pv-cross-m4096-n512 で ×0.966 < ×1.050
 *   - tile128x64r8x8w8x16k32: i8a8-attention-pv-cross-m4096-n512 で ×0.837 < ×1.050
 *   - tile16x128r4x8w16x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.728 < ×1.050
 *   - tile16x128r4x8w16x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.743 < ×1.050
 *   - tile16x32r4x4w8x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.649 < ×1.050
 *   - tile16x32r4x4w8x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.643 < ×1.050
 *   - tile16x64r4x4w16x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.679 < ×1.050
 *   - tile16x64r4x4w16x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.768 < ×1.050
 *   - tile16x64r4x8w8x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.760 < ×1.050
 *   - tile16x64r4x8w8x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.764 < ×1.050
 *   - tile32x128r4x8w16x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.879 < ×1.050
 *   - tile32x128r4x8w16x8k32: i8a8-attention-pv-self-m4096-n4096 で ×0.894 < ×1.050
 *   - tile32x128r8x8w16x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.933 < ×1.050
 *   - tile32x128r8x8w16x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.896 < ×1.050
 *   - tile32x32r4x4w8x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.764 < ×1.050
 *   - tile32x32r4x4w8x8k32: i8a8-attention-pv-self-m4096-n4096 で ×0.789 < ×1.050
 *   - tile32x32r8x4w8x4k16: i8a8-attention-pv-self-m1024-n1024 で ×0.750 < ×1.050
 *   - tile32x32r8x4w8x4k32: i8a8-attention-pv-cross-m4096-n512 で ×0.716 < ×1.050
 *   - tile32x64r4x4w16x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.897 < ×1.050
 *   - tile32x64r4x4w16x8k32: i8a8-attention-pv-self-m4096-n4096 で ×0.910 < ×1.050
 *   - tile32x64r4x8w8x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.933 < ×1.050
 *   - tile32x64r4x8w8x8k32: i8a8-attention-pv-self-m4096-n4096 で ×0.934 < ×1.050
 *   - tile32x64r8x4w16x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.957 < ×1.050
 *   - tile32x64r8x4w16x4k32: i8a8-attention-pv-self-m4096-n4096 で ×0.950 < ×1.050
 *   - tile32x64r8x8w8x4k16: i8a8-attention-pv-self-m4096-n4096 で ×0.890 < ×1.050
 *   - tile32x64r8x8w8x4k32: i8a8-attention-pv-cross-m4096-n512 で ×0.776 < ×1.050
 *   - tile64x128r4x8w16x16k16: i8a8-attention-pv-self-m4096-n4096 で ×0.944 < ×1.050
 *   - tile64x128r4x8w16x16k32: i8a8-attention-pv-self-m4096-n4096 で ×0.941 < ×1.050
 *   - tile64x128r8x8w16x8k32: i8a8-attention-pv-cross-m4096-n512 で ×0.975 < ×1.050
 *   - tile64x32r4x4w8x16k16: i8a8-attention-pv-self-m4096-n4096 で ×0.840 < ×1.050
 *   - tile64x32r4x4w8x16k32: i8a8-attention-pv-self-m4096-n4096 で ×0.843 < ×1.050
 *   - tile64x32r8x4w8x8k16: i8a8-attention-pv-self-m4096-n4096 で ×0.814 < ×1.050
 *   - tile64x32r8x4w8x8k32: i8a8-attention-pv-self-m4096-n4096 で ×0.748 < ×1.050
 *   - tile64x64r4x4w16x16k16: i8a8-attention-pv-self-m4096-n4096 で ×0.964 < ×1.050
 *   - tile64x64r4x4w16x16k32: i8a8-attention-pv-self-m4096-n4096 で ×0.983 < ×1.050
 *   - tile64x64r4x8w8x16k16: i8a8-attention-pv-self-m4096-n4096 で ×1.048 < ×1.050
 *   - tile64x64r4x8w8x16k32: i8a8-attention-pv-cross-m4096-n512 で ×1.013 < ×1.050
 *   - tile64x64r8x4w16x8k16: 幾何平均 ×1.173（採用 ×1.210 に届かない）
 *   - tile64x64r8x4w16x8k32: 幾何平均 ×1.104（採用 ×1.210 に届かない）
 *   - tile64x64r8x8w8x8k16: i8a8-attention-pv-cross-m4096-n512 で ×0.939 < ×1.050
 *   - tile64x64r8x8w8x8k32: i8a8-attention-pv-cross-m4096-n512 で ×0.836 < ×1.050
 */
import type { GeometryProfile } from "../geometry-profile.ts";

export const NVIDIA_BLACKWELL: GeometryProfile = {
  id: "nvidia-blackwell",
  match: { vendor: "nvidia", architecture: "blackwell" },
  gemmRows: [
    { maxRows: 16, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 32, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 64, geometry: { regM: 1, regN: 4, wgX: 4, wgY: 16 } },
    { maxRows: 128, geometry: { regM: 4, regN: 4, wgX: 8, wgY: 16 } },
    { maxRows: 256, geometry: { regM: 4, regN: 4, wgX: 16, wgY: 16 } },
    { maxRows: 512, geometry: { regM: 8, regN: 4, wgX: 16, wgY: 8 } },
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
      "outputs/bench-browser/geometry-sweep-browser-2026-09-27T18-37-27.914Z.json, outputs/bench-browser/geometry-sweep-browser-2026-09-29T20-42-08.545Z.json, outputs/bench-browser/geometry-sweep-browser-2026-10-01T16-46-50.379Z.json",
    sha256:
      "084cd0efd7ea7780dfcdab5b53ee4be8177638c7d8984920096a4851b767aa90, 1ae3a8f6d9a1270d8e3b3ad41aa7fe13757c976a323de7d2b0b99139800cb2e5, 0e57d4efcb14efe0e01767066fce05e1784ca43660a4f8a78c5a1a2c48d1ae87",
    date: "2026-09-27T18:37:27.914Z, 2026-09-29T20:42:08.545Z, 2026-10-01T16:46:50.379Z",
    candidateSet: "full, full, full",
    userAgent: [
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36",
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
    ],
    adapter: {
      vendor: "nvidia",
      architecture: "blackwell",
      device: "0x2c05",
      description: "NVIDIA GeForce RTX 5070 Ti",
    },
    kernels: "21c1151685b4d48a",
    caseSet: "c1060d8829f3b9de",
  },
};
