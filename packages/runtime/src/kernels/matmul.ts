/**
 * matmul（rank-2 × rank-2、f32）。実体は GEMM 3 op 共通の 64×64 レジスタブロッキング骨格
 * （src/kernels/gemm.ts）で、ここはキー・WGSL・params の呼び出し面だけを持つ。
 *
 * v4 フラグは形状（`k % 4 == 0 && n % 4 == 0`）から executor が導き、キーと WGSL の両方へ
 * 渡す。不変条件（縮約順序・1 workgroup = 1 タイル・fail loudly）は骨格側の MUST が正本。
 *
 * `rows`（= M）も同じく形状由来で、タイル幾何のバケット（src/kernels/gemm-geometry.ts の
 * `gemmGeometryForRows`）を決める。MUST: キー・WGSL・dispatch の 3 つに**同じ M** を通す。
 */

import { gemmKeyPart, gemmParams, gemmWgsl } from "./gemm.ts";

export const matmulKey = (v4: boolean, rows?: number): string =>
  `matmul:v2:f32:${gemmKeyPart(v4, rows)}`;

export const matmulWgsl = (v4: boolean, rows?: number): string =>
  gemmWgsl({ op: "matmul", v4, rows });

export const matmulParams = (m: number, n: number, k: number): Uint32Array<ArrayBuffer> =>
  gemmParams("matmul", m, n, k);
