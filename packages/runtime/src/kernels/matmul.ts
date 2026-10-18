/**
 * matmul (rank-2 × rank-2, f32). The body is the register-blocking skeleton shared by the GEMM
 * family (src/kernels/gemm.ts — dimensions in GemmGeometry); this file only holds the key, WGSL and
 * params entry points.
 *
 * The executor derives the v4 flag from the shape (`k % 4 == 0 && n % 4 == 0`) and passes it to
 * both the key and the WGSL. The invariants (reduction order, 1 workgroup = 1 tile, fail loudly)
 * are owned by the MUSTs on the skeleton side.
 *
 * `rows` (= M) is likewise shape-derived and selects the tile-geometry bucket
 * (`gemmGeometryForRows` in src/kernels/gemm-geometry.ts). MUST: pass the **same M** to all three of
 * key, WGSL and dispatch.
 *
 * `geometry` is the explicit geometry (`GemmSpec` in src/kernels/gemm.ts — in a Session, the value
 * the adapter's profile chose). When given it takes precedence over the `rows` bucket, and MUST: pass
 * the **same value** to the key and the WGSL (the same discipline as `rows`).
 */

import { gemmKeyPart, gemmParams, gemmWgsl } from "./gemm.ts";
import type { GemmGeometry } from "./gemm-geometry.ts";

export const matmulKey = (v4: boolean, rows?: number, geometry?: GemmGeometry): string =>
  `matmul:v2:f32:${gemmKeyPart(v4, rows, geometry)}`;

export const matmulWgsl = (v4: boolean, rows?: number, geometry?: GemmGeometry): string =>
  gemmWgsl({ op: "matmul", v4, rows, geometry });

export const matmulParams = (m: number, n: number, k: number): Uint32Array<ArrayBuffer> =>
  gemmParams("matmul", m, n, k);
