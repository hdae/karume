/**
 * linear（`x[…,in] × W[out,in] + b[out]`、f32）。先行次元は平坦化して `[m,k] × [k,n]` の
 * 2 次元 GEMM に落とす（`m` = 先行次元の積、`k` = 入力特徴数、`n` = 出力特徴数）。
 *
 * 実体は GEMM 3 op 共通の 64×64 レジスタブロッキング骨格（src/kernels/gemm.ts）で、matmul /
 * bmm との違いは 3 点だけ:
 *
 * - **重みは転置レイアウト**（torch の `Linear.weight` は `[out, in]`）。連続方向が k なので、
 *   1 スレッドが k 連続 4 要素を読み、**共有メモリ側で転置して置く**（転置コピーは出さない）。
 * - 末尾で bias を 1 度だけ足す（accumulator に初期値として入れない — 縮約順序を matmul と
 *   同じ「k 昇順の逐次」に保つため。bias を先に積むと丸めの並びが変わる）。
 * - 重み格納の変種を持つ（`w=f32` / `w=f16` / `w=i8` — ADR 0018 / 0019）。差は共有メモリへの
 *   読み出し式（と i8 の scale 束縛）だけで、i8 の scale は**共有タイルへの読み込み時**に
 *   要素ごとに掛かる（MAC ごとでも縮約の外でもない — ADR 0019）。
 */

import { type GemmCompute, gemmComputeKeyPart, gemmKeyPart, gemmParams, gemmWgsl } from "./gemm.ts";
import { weightKeyPart, type WeightStorage } from "./weight-storage.ts";

export { LINEAR_SCALE_BINDING } from "./gemm.ts";

/**
 * `rows` は平坦化後の M（先行次元の積）。タイル幾何のバケット（src/kernels/gemm-geometry.ts の
 * `gemmGeometryForRows`）を決める形状由来の値で、MUST: キー・WGSL・dispatch に**同じ M** を通す。
 */
export const linearKey = (
  weight: WeightStorage,
  v4: boolean,
  compute: GemmCompute = "f32",
  rows?: number,
): string =>
  `linear:v2:f32:${gemmKeyPart(v4, rows)}${weightKeyPart(weight)}${gemmComputeKeyPart(compute)}`;

export const linearWgsl = (
  weight: WeightStorage,
  v4: boolean,
  compute: GemmCompute = "f32",
  rows?: number,
): string => gemmWgsl({ op: "linear", v4, weight, compute, rows });

export const linearParams = (m: number, n: number, k: number): Uint32Array<ArrayBuffer> =>
  gemmParams("linear", m, n, k);
