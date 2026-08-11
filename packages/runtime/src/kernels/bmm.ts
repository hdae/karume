/**
 * bmm（rank-3 バッチ matmul `[B,M,K] × [B,K,N] → [B,M,N]`、f32）。実体は GEMM 3 op 共通の
 * 64×64 レジスタブロッキング骨格（src/kernels/gemm.ts）で、matmul との違いはバッチ軸を
 * dispatch の z 軸に取り、3 入出力のオフセットを行列 1 枚ぶんから導く点だけ。
 *
 * uniform は matmul / linear と同じ 3 語 `{m,n,k}`。バッチ数はシェーダが `wid.z` から導くので
 * uniform に載せない（載せても一度も読まない死んだフィールドになる）。
 */

import { gemmKeyPart, gemmParams, gemmWgsl } from "./gemm.ts";

/**
 * `rows` は**行列 1 枚の M**（バッチ軸は含めない — バッチは dispatch の z で、タイル幾何とは
 * 独立）。タイル幾何のバケットは src/kernels/gemm-geometry.ts の `gemmGeometryForRows`。
 */
export const bmmKey = (v4: boolean, rows?: number): string => `bmm:v2:f32:${gemmKeyPart(v4, rows)}`;

export const bmmWgsl = (v4: boolean, rows?: number): string => gemmWgsl({ op: "bmm", v4, rows });

export const bmmParams = (m: number, n: number, k: number): Uint32Array<ArrayBuffer> =>
  gemmParams("bmm", m, n, k);
