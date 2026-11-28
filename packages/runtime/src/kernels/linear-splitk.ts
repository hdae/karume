/**
 * linear with the K reduction split into slices (`x[m,k] · wᵀ[k,n] + bias[n]`, f32 compute, every
 * weight storage the GEMM skeleton reads: f32 / f16 / bf16 / i8 / i4 / i2) — a **reorder** strategy
 * (ADR 0152 decision 2): the products are the reference kernel's, only the order in which they are
 * added differs. The storage's decode happens at the tile fill, which the split does not touch.
 *
 * Two dispatches, the form of the split-K variants of the published `@huggingface/kernels` MatMul
 * (a partial pass with the slice on the dispatch's z axis, then a combine pass):
 *
 * 1. **Partial sums: the split of the GEMM skeleton** (src/kernels/gemm.ts, `splitSlices`). The
 *    linear GEMM's own kernel — the same tile geometry, fills and inner-product loop — runs over the
 *    K tiles of slice `wid.z` only, each slice from 0 in the reference kernel's ascending order,
 *    and stores its accumulators to `partials[slice][row][col]` without the bias. The tiles are
 *    dealt out in whole tiles (`kTileLoop` in gemm.ts), so a tile count the slice count does not
 *    divide gives slices one tile apart.
 * 2. **Combine.** One thread adds an element's partial sums in ascending slice order, then the bias,
 *    once, last (as the reference kernel does — src/kernels/linear.ts).
 *
 * What it changes: a single accumulator over K terms rounds K − 1 times against a growing sum, so its
 * error grows with K; `slices` accumulators of about `K / slices` terms each round against sums that
 * are about `slices` times smaller, and the combine adds `slices − 1` more roundings. With one slice
 * the two dispatches add the same terms in the same order as the reference kernel and give its words
 * (the structural check of the kernel; the selection never takes one slice).
 *
 * Keys: the partial's key is the linear GEMM's key parts (geometry, form, storage) with the slice
 * count, which the WGSL bakes in — two slice counts are two kernels whose numbers differ, so a
 * Session's kernel set names the one it ran. The combine reads the count from the uniform and has
 * one key.
 *
 * SPEC: docs/spec/numerics.md#strategies-admitted-only-by-a-private-key
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import { type GemmCompute, gemmKeyPart, gemmTakesForm, gemmWgsl } from "./gemm.ts";
import { GEMM_TILE_K, type GemmGeometry } from "./gemm-geometry.ts";
import {
  i4GroupKeyPart,
  i4GroupShift,
  weightKeyPart,
  type WeightStorage,
} from "./weight-storage.ts";

/** How many slices the selection cuts the K reduction into. */
export type LinearSplitKSlices = 2 | 4 | 8 | 16;

/**
 * The fewest terms of whole K tiles one slice adds (below it the combine's extra roundings and its
 * pass buy too little): 16 K tiles.
 */
export const LINEAR_SPLITK_MIN_SLICE = 256;

/** The threads of one workgroup of the combine pass (grid-stride). */
export const LINEAR_SPLITK_COMBINE_GROUP = 256;

/** The compute of the split partial (the split of the f16 compute variant is not built). */
const SPLIT_COMPUTE: GemmCompute = "f32";

/**
 * The slice count for a reduction length: the largest of 16 / 8 / 4 / 2 whose shortest slice holds
 * at least {@link LINEAR_SPLITK_MIN_SLICE} terms' worth of whole K tiles; `undefined` when there is
 * none (fewer than 32 K tiles, `k ≤ 496`).
 *
 * The partial deals the K tiles out in whole tiles (`kTileLoop` in gemm.ts), so the count need not
 * divide the tile count (K = 4304, 269 tiles, takes 16 slices of 16 and 17 tiles) and K need not be a
 * multiple of the K tile: the reduction's last tile may be partial, so the last slice may add up to
 * 15 terms fewer than its tiles hold.
 */
export const linearSplitKSlices = (k: number): LinearSplitKSlices | undefined => {
  const tiles = Math.ceil(k / GEMM_TILE_K);
  return ([16, 8, 4, 2] as const).find(
    (slices) => Math.floor(tiles / slices) * GEMM_TILE_K >= LINEAR_SPLITK_MIN_SLICE,
  );
};

/** The bytes of the partial sums `[slices, m, n]` (f32) — the temporary between the two dispatches. */
export const linearSplitKPartialsBytes = (m: number, n: number, slices: number): number =>
  slices * m * n * 4;

/**
 * The uniform of both dispatches: `m`, `n`, `k`, `slices` (the partial reads the first three, as the
 * linear GEMM does; the combine reads all four).
 *
 * MUST: the u32 bounds of the linear GEMM (`m·k` and `n·k` — the A and W indices of the shared
 * skeleton) and of the partials (`slices·m·n` — the combine's and the partial's flat index), so a
 * wrapped index fails here instead of reading another row.
 */
export const linearSplitKParams = (
  m: number,
  n: number,
  k: number,
  slices: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("linear split-K params", { m, n, k, slices });
  if (slices < 1) throw new CodegenError(`linear split-K params: ${slices} slices`);
  assertU32Params("linear split-K params", {
    "m * k": m * k,
    "n * k": n * k,
    partials: slices * m * n,
  });
  return new Uint32Array([m, n, k, slices]);
};

/**
 * The key of the split partial: `linear_splitk_gemm`, the linear GEMM's geometry, form and storage
 * parts, and the slice count (`:s4`).
 *
 * MUST: the same `rows`, `groupSize` and `geometry` as {@link linearSplitKGemmWgsl} (the discipline
 * of `linearKey`).
 */
export const linearSplitKGemmKey = (
  weight: WeightStorage,
  v4: boolean,
  slices: number,
  rows?: number,
  groupSize?: number,
  geometry?: GemmGeometry,
): string => {
  i4GroupShift("linear split-K", weight, groupSize);
  return `linear_splitk_gemm:v1:f32:${gemmKeyPart(
    v4,
    rows,
    geometry,
    gemmTakesForm("linear", v4, SPLIT_COMPUTE),
  )}${weightKeyPart(weight)}${i4GroupKeyPart(groupSize)}:s${slices}`;
};

/** The WGSL of the split partial: the linear GEMM with `splitSlices` (src/kernels/gemm.ts). */
export const linearSplitKGemmWgsl = (
  weight: WeightStorage,
  v4: boolean,
  slices: number,
  rows?: number,
  groupSize?: number,
  geometry?: GemmGeometry,
): string =>
  gemmWgsl({
    op: "linear",
    v4,
    weight,
    compute: SPLIT_COMPUTE,
    rows,
    weightGroupShift: i4GroupShift("linear split-K", weight, groupSize),
    geometry,
    splitSlices: slices,
  });

const DIMS = `struct Dims {
  m: u32,
  n: u32,
  k: u32,
  slices: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;`;

export const linearSplitKCombineKey = (): string =>
  `linear_splitk_combine:v1:f32:wg${LINEAR_SPLITK_COMBINE_GROUP}`;

export const linearSplitKCombineWgsl =
  (): string => `// karume linear split-K, pass 2 (the K-slice partial sums added in slice order, then + bias[n], f32, grid-stride / wg ${LINEAR_SPLITK_COMBINE_GROUP})
${DIMS}
@group(0) @binding(1) var<storage, read> partials: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(${LINEAR_SPLITK_COMBINE_GROUP})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let total = dims.m * dims.n;
  let stride = nwg.x * ${LINEAR_SPLITK_COMBINE_GROUP}u;
  var i = gid.x;
  while (i < total) {
    var acc = partials[i];
    for (var slice = 1u; slice < dims.slices; slice = slice + 1u) {
      acc = acc + partials[slice * total + i];
    }
    // The bias is added once, last (not as the accumulator's initial value).
    out[i] = acc + bias[i % dims.n];
    i = i + stride;
  }
}
`;
