/**
 * matmul（rank-2 × rank-2、f32）。実体は GEMM 3 op 共通の 64×64 レジスタブロッキング骨格
 * （src/kernels/gemm.ts）で、ここはキー・WGSL・params の呼び出し面だけを持つ。
 *
 * v4 フラグは形状（`k % 4 == 0 && n % 4 == 0`）から executor が導き、キーと WGSL の両方へ
 * 渡す。不変条件（縮約順序・1 workgroup = 1 タイル・fail loudly）は骨格側の MUST が正本。
 */

import { gemmKeyPart, gemmParams, gemmWgsl } from "./gemm.ts";

export const matmulKey = (v4: boolean): string => `matmul:v2:f32:${gemmKeyPart(v4)}`;

export const matmulWgsl = (v4: boolean): string => gemmWgsl({ op: "matmul", v4 });

export const matmulParams = (m: number, n: number, k: number): Uint32Array<ArrayBuffer> =>
  gemmParams("matmul", m, n, k);
