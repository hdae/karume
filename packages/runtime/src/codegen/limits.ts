/**
 * カーネル実装に由来する値域上限のうち、**op 契約層も同じ門を張る**もの。
 *
 * MUST: 置き場はここ 1 か所。上限が params 側にしか無いと「契約検査は通ったのに実行段で
 * 内部エラー」になり、どの op のどの値が域外なのかが利用者の診断に残らない（strided 族の
 * rank 上限 {@link "./strided.ts"} `STRIDED_RANK` と同じ分担）。ops → kernels の依存を作らない
 * ための中立な場所でもある（ops / kernels は双方すでに codegen へ依存している）。
 */

/**
 * gru_scan の隠れ幅 H の上限（= カーネルの workgroup サイズ）。**1 lane = 1 隠れユニット**の
 * 割り当てで、生成 WGSL の `h_shared` / `stage` はこの長さの workgroup 配列として展開される
 * （kernels/gru-scan.ts）。
 *
 * MUST: 超過は契約層（ops/shapes.ts の gru_scan）と params（`gruScanParams`）の両方で
 * fail loudly にする — 黙って縮退させると workgroup 配列の範囲外書き込みになり、例外なしに
 * 別ユニットの状態が壊れる。上限を上げるには workgroup 内 grid-stride と `h` の二重化が要る
 * = 別の設計判断（ADR 0056 決定 5）。
 */
export const GRU_SCAN_MAX_HIDDEN = 256;
