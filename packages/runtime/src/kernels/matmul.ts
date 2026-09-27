/**
 * matmul（rank-2 × rank-2、f32）。実体は GEMM 3 op 共通の 64×64 レジスタブロッキング骨格
 * （src/kernels/gemm.ts）で、ここはキー・WGSL・params の呼び出し面だけを持つ。
 *
 * v4 フラグは形状（`k % 4 == 0 && n % 4 == 0`）から executor が導き、キーと WGSL の両方へ
 * 渡す。不変条件（縮約順序・1 workgroup = 1 タイル・fail loudly）は骨格側の MUST が正本。
 *
 * `rows`（= M）も同じく形状由来で、タイル幾何のバケット（src/kernels/gemm-geometry.ts の
 * `gemmGeometryForRows`）を決める。MUST: キー・WGSL・dispatch の 3 つに**同じ M** を通す。
 *
 * `geometry` は明示の幾何（src/kernels/gemm.ts の `GemmSpec` — Session では adapter の
 * プロファイルが選んだ値）。渡すと `rows` のバケットより優先し、MUST: キーと WGSL へ**同じ値**を
 * 通す（`rows` と同じ規律）。
 */

import { gemmKeyPart, gemmParams, gemmWgsl } from "./gemm.ts";
import type { GemmGeometry } from "./gemm-geometry.ts";

export const matmulKey = (v4: boolean, rows?: number, geometry?: GemmGeometry): string =>
  `matmul:v2:f32:${gemmKeyPart(v4, rows, geometry)}`;

export const matmulWgsl = (v4: boolean, rows?: number, geometry?: GemmGeometry): string =>
  gemmWgsl({ op: "matmul", v4, rows, geometry });

export const matmulParams = (m: number, n: number, k: number): Uint32Array<ArrayBuffer> =>
  gemmParams("matmul", m, n, k);
