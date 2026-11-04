/**
 * linear (`x[…,in] × W[out,in] + b[out]`, f32). The leading dimensions are flattened down to a 2-D
 * GEMM `[m,k] × [k,n]` (`m` = product of the leading dimensions, `k` = input features, `n` = output
 * features).
 *
 * The body is the register-blocking skeleton shared by the GEMM family (src/kernels/gemm.ts —
 * dimensions in GemmGeometry); it differs from matmul / bmm in only 3 points:
 *
 * - **The weight is in transposed layout** (torch's `Linear.weight` is `[out, in]`). The contiguous
 *   direction is k, so one thread reads 4 k-contiguous elements and **places them transposed on the
 *   shared-memory side** (no transposed copy is emitted).
 * - The bias is added exactly once at the end (it is not put into the accumulator as the initial
 *   value — to keep the reduction order the same "sequential in ascending k" as matmul. Adding the
 *   bias first changes the order of rounding).
 * - It has weight-storage variants (`w=f32` / `w=f16` / `w=i8` / `w=i4` / `w=i2` — ADR 0018 / 0019 /
 *   0069 / 0097). The only difference is the expression that reads into shared memory (and the scale
 *   bindings of i8 / i4 / i2); the i8 scale is applied per element **when loading into the shared
 *   tile** (neither per MAC nor outside the reduction — ADR 0019).
 */

import { assertU32Params } from "../codegen/params.ts";
import { type GemmCompute, gemmComputeKeyPart, gemmKeyPart, gemmParams, gemmWgsl } from "./gemm.ts";
import type { GemmGeometry } from "./gemm-geometry.ts";
import {
  i4GroupKeyPart,
  i4GroupShift,
  weightKeyPart,
  type WeightStorage,
} from "./weight-storage.ts";

export { LINEAR_SCALE_BINDING } from "./gemm.ts";

/**
 * `rows` is the M after flattening (the product of the leading dimensions). It is the shape-derived
 * value that selects the tile-geometry bucket (`gemmGeometryForRows` in
 * src/kernels/gemm-geometry.ts), and MUST: pass the **same M** to the key, the WGSL and the dispatch.
 *
 * `groupSize` is the i4 group length (required for i4). The shift is baked into the WGSL, so **it
 * also goes into the key** (`:wi4g32` — codegen determinism: same key → byte-identical WGSL).
 *
 * `geometry` is the explicit geometry (`GemmSpec` in src/kernels/gemm.ts — in a Session the
 * profile's value, in measurement a sweep candidate). When given it takes precedence over the `rows`
 * bucket, and MUST: pass the **same value** to the key and the WGSL (the same discipline as `rows`).
 */
export const linearKey = (
  weight: WeightStorage,
  v4: boolean,
  compute: GemmCompute = "f32",
  rows?: number,
  groupSize?: number,
  geometry?: GemmGeometry,
): string => {
  i4GroupShift("linear", weight, groupSize);
  // The generation that guarded the i8 scale read. Keys of unchanged storage formats are kept.
  return `linear:v${weight === "i8" ? 3 : 2}:f32:${gemmKeyPart(v4, rows, geometry)}${weightKeyPart(
    weight,
  )}${i4GroupKeyPart(groupSize)}${gemmComputeKeyPart(compute)}`;
};

export const linearWgsl = (
  weight: WeightStorage,
  v4: boolean,
  compute: GemmCompute = "f32",
  rows?: number,
  groupSize?: number,
  geometry?: GemmGeometry,
): string =>
  gemmWgsl({
    op: "linear",
    v4,
    weight,
    compute,
    rows,
    weightGroupShift: i4GroupShift("linear", weight, groupSize),
    geometry,
  });

/**
 * The uniform Dims (the u32 range of m / n / k is checked by {@link gemmParams}).
 *
 * MUST: the logical element count of the weight `n·k ≤ 2^32 − 1`. The GEMM path first builds the row
 * start as the logical element index `wcol·dims.k` in u32, and the loader turns it into a word
 * position (i2 `>> 4`, i4 `>> 3`). If the product wraps first, the high bits do not come back and the
 * weights of another output channel are read without any error. With i2 the storage reaches this at
 * about 1 GiB, so neither the binding limit nor the residency plan's compressed-byte check stops it
 * (the index is the same expression regardless of storage format, so the check is not split by
 * storage format either).
 *
 * MUST: the activation element count `m·k ≤ 2^32 − 1` too — the A index `abase + arow·k + col` is a
 * u32 flat index, so a wrapped product reads another row of the activation without an exception.
 * NOTE: the output `m·n` is not checked yet — an open gap: the accepted set the linear params
 * tests pin includes `[65, 268435455, 16]`, whose `m·n` passes u32.
 */
export const linearParams = (m: number, n: number, k: number): Uint32Array<ArrayBuffer> => {
  const params = gemmParams("linear", m, n, k);
  assertU32Params("linear params", { "m * k": m * k, "n * k": n * k });
  return params;
};
