/**
 * bmm (rank-3 batched matmul `[B,M,K] × [B,K,N] → [B,M,N]`, f32). The body is the
 * register-blocking skeleton shared by the GEMM family (src/kernels/gemm.ts — dimensions in
 * GemmGeometry); the only difference from matmul is that the batch axis is taken on the dispatch's z
 * axis and the offsets of the 3 inputs/outputs are derived from the size of one matrix.
 *
 * The uniform is the same 3 words `{m,n,k}` as matmul / linear. The shader derives the batch index
 * from `wid.z`, so it is not put in the uniform (it would be a dead field that is never read).
 *
 * Only the **row-window variant** ({@link GemmRowWindow}) widens the uniform to 5 words. It exists
 * solely for the row-block execution of decomposed attention (src/runtime/fusion.ts) and adds not a
 * single word to the shared `gemmParams` — adding one would change the uniform layout and the
 * snapshots of every op.
 */

import { assertU32Params } from "../codegen/params.ts";
import {
  assertGemmRowWindow,
  gemmKeyPart,
  gemmParams,
  gemmTakesForm,
  type GemmRowWindow,
  gemmWgsl,
} from "./gemm.ts";
import type { GemmGeometry } from "./gemm-geometry.ts";

export type { GemmRowWindow };

/**
 * `rows` is **the M of one matrix** (the batch axis is not included — the batch is the dispatch's z
 * and independent of the tile geometry). The tile-geometry bucket is `gemmGeometryForRows` in
 * src/kernels/gemm-geometry.ts.
 *
 * `window` is the discriminator of the row-window variant (when omitted, the key does not differ by
 * a single byte from the existing one). **The row offset and the full M are uniform values and are
 * not put in the key** — putting them in would duplicate the same WGSL into pipelines once per block.
 *
 * `geometry` is the explicit geometry (`GemmSpec` in src/kernels/gemm.ts — on a Session's bmm node,
 * the value the adapter's profile chose). When given it takes precedence over the `rows` bucket, and
 * MUST: pass the **same value** to the key and the WGSL (the same discipline as `rows`).
 */
export const bmmKey = (
  v4: boolean,
  rows?: number,
  window?: GemmRowWindow,
  geometry?: GemmGeometry,
): string =>
  `bmm:v2:f32:${gemmKeyPart(v4, rows, geometry, gemmTakesForm("bmm", v4))}${window === undefined ? "" : `:rw${window}`}`;

export const bmmWgsl = (
  v4: boolean,
  rows?: number,
  window?: GemmRowWindow,
  geometry?: GemmGeometry,
): string => gemmWgsl({ op: "bmm", v4, rows, rowWindow: window, geometry });

/**
 * The element counts of the three operands including the batch.
 *
 * MUST: each fits in u32. The batch base is the u32 product `wid.z · rows · k` (and the like —
 * `batchPrologue` in gemm.ts), and the in-matrix indices are added to it, so a count past
 * 2^32 − 1 wraps and the last batches read or write the first ones (or one matrix another element)
 * without an exception — the u32 checks on m / n / k alone do not stop it. `rows` is per side: the
 * row-window variant strides the windowed side by the full M (`rows_full`), the other side by the
 * block's m.
 */
const assertBmmExtents = (
  where: string,
  batch: number,
  rows: { readonly a: number; readonly c: number },
  n: number,
  k: number,
): void => {
  assertU32Params(where, { batch });
  assertU32Params(where, {
    "A elements (batch * rows * k)": batch * rows.a * k,
    "B elements (batch * k * n)": batch * k * n,
    "C elements (batch * rows * n)": batch * rows.c * n,
  });
};

/**
 * `batch` is not written to the uniform (the shader takes it from `wid.z`); it is taken only for
 * the flat-index check ({@link assertBmmExtents}). MUST: pass the dispatch's z count.
 */
export const bmmParams = (
  m: number,
  n: number,
  k: number,
  batch: number,
): Uint32Array<ArrayBuffer> => {
  const params = gemmParams("bmm", m, n, k);
  assertBmmExtents("bmm params", batch, { a: m, c: m }, n, k);
  return params;
};

/**
 * The uniform of the row-window variant (5 words: `{m,n,k}` + `{row_offset, rows_full}`).
 *
 * The 5 words of content are rounded up to a 16-byte unit and **32 bytes** are allocated (allocating
 * in 16-byte units is a project convention (the fusion pass fixes the same shape in ADR 0040), not a
 * requirement of the spec). MUST: the order pairs with `ROW_WINDOW_DIMS_EXTRA` in gemm.ts.
 *
 * `rowsFull` is the **original full M** of the row window, and `rowOffset` is the first row of it
 * that this block handles (the guard is {@link assertGemmRowWindow} — shared with the params of fused
 * attention).
 *
 * `window` is the windowed side (the same value as the key's and the WGSL's) and `batch` the
 * dispatch's z count: the windowed side's flat index strides by the full M, the other side by the
 * block's m ({@link assertBmmExtents}). Neither goes into the uniform.
 */
export const bmmRowWindowParams = (
  m: number,
  n: number,
  k: number,
  rowOffset: number,
  rowsFull: number,
  window: GemmRowWindow,
  batch: number,
): Uint32Array<ArrayBuffer> => {
  assertGemmRowWindow("bmm row window params", m, { offset: rowOffset, rowsFull });
  const base = gemmParams("bmm", m, n, k);
  assertBmmExtents(
    "bmm row window params",
    batch,
    window === "a" ? { a: rowsFull, c: m } : { a: m, c: rowsFull },
    n,
    k,
  );
  const params = new Uint32Array(8);
  params.set(base.subarray(0, 3));
  params[3] = rowOffset;
  params[4] = rowsFull;
  return params;
};
