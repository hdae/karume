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

import { assertU32Params } from "../codegen/params.ts";
import { gemmKeyPart, gemmParams, gemmWgsl } from "./gemm.ts";
import type { GemmGeometry } from "./gemm-geometry.ts";

export const matmulKey = (v4: boolean, rows?: number, geometry?: GemmGeometry): string =>
  `matmul:v2:f32:${gemmKeyPart(v4, rows, geometry)}`;

export const matmulWgsl = (v4: boolean, rows?: number, geometry?: GemmGeometry): string =>
  gemmWgsl({ op: "matmul", v4, rows, geometry });

/**
 * MUST: the element counts of A `M·K`, B `K·N` and C `M·N` fit in u32. The skeleton builds every
 * operand index as a u32 flat index (`abase + arow·k + col`, `bbase + row·n`, `cbase + orow·n +
 * col`), so a product past 2^32 − 1 drops its upper bits and another element is read or written
 * without an exception — the u32 checks on m / n / k alone (in {@link gemmParams}) do not stop it.
 */
export const matmulParams = (m: number, n: number, k: number): Uint32Array<ArrayBuffer> => {
  const params = gemmParams("matmul", m, n, k);
  assertU32Params("matmul params", { "m * k": m * k, "k * n": k * n, "m * n": m * n });
  return params;
};
