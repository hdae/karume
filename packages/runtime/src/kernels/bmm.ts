/**
 * bmm（rank-3 バッチ matmul `[B,M,K] × [B,K,N] → [B,M,N]`、f32）。実体は GEMM 3 op 共通の
 * 64×64 レジスタブロッキング骨格（src/kernels/gemm.ts）で、matmul との違いはバッチ軸を
 * dispatch の z 軸に取り、3 入出力のオフセットを行列 1 枚ぶんから導く点だけ。
 *
 * uniform は matmul / linear と同じ 3 語 `{m,n,k}`。バッチ数はシェーダが `wid.z` から導くので
 * uniform に載せない（載せても一度も読まない死んだフィールドになる）。
 *
 * **行窓変種**（{@link GemmRowWindow}）だけが uniform を 5 語へ広げる。分解 attention の
 * 行ブロック実行（src/runtime/fusion.ts）専用で、共有の `gemmParams` には 1 語も足さない
 * — 足すと全 op の uniform レイアウトとスナップショットが総取っ替えになる。
 */

import {
  assertGemmRowWindow,
  gemmKeyPart,
  gemmParams,
  type GemmRowWindow,
  gemmWgsl,
} from "./gemm.ts";

export type { GemmRowWindow };

/**
 * `rows` は**行列 1 枚の M**（バッチ軸は含めない — バッチは dispatch の z で、タイル幾何とは
 * 独立）。タイル幾何のバケットは src/kernels/gemm-geometry.ts の `gemmGeometryForRows`。
 *
 * `window` は行窓変種の判別子（省略時は従来のキーと 1 バイトも変わらない）。**行オフセットと
 * 全 M は uniform 値なのでキーに載せない** — 載せるとブロックの本数だけ同じ WGSL が
 * パイプラインへ複製される。
 */
export const bmmKey = (v4: boolean, rows?: number, window?: GemmRowWindow): string =>
  `bmm:v2:f32:${gemmKeyPart(v4, rows)}${window === undefined ? "" : `:rw${window}`}`;

export const bmmWgsl = (v4: boolean, rows?: number, window?: GemmRowWindow): string =>
  gemmWgsl({ op: "bmm", v4, rows, rowWindow: window });

export const bmmParams = (m: number, n: number, k: number): Uint32Array<ArrayBuffer> =>
  gemmParams("bmm", m, n, k);

/**
 * 行窓変種の uniform（`{m,n,k}` + `{row_offset, rows_full}` の 5 語）。
 *
 * uniform アドレス空間の struct は 16 バイト整列なので、5 語ぶんの内容でも **32 バイト**確保
 * する（不足すると binding が validation で落ちる）。MUST: 並びは gemm.ts の
 * `ROW_WINDOW_DIMS_EXTRA` と対。
 *
 * `rowsFull` は行窓側の**元の全 M**、`rowOffset` はそのうちこのブロックが担当する先頭行
 * （門は {@link assertGemmRowWindow} — 融合 attention の params と共有）。
 */
export const bmmRowWindowParams = (
  m: number,
  n: number,
  k: number,
  rowOffset: number,
  rowsFull: number,
): Uint32Array<ArrayBuffer> => {
  assertGemmRowWindow("bmm 行窓 params", m, { offset: rowOffset, rowsFull });
  const base = gemmParams("bmm", m, n, k);
  const params = new Uint32Array(8);
  params.set(base.subarray(0, 3));
  params[3] = rowOffset;
  params[4] = rowsFull;
  return params;
};
