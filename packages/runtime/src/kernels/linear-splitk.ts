/**
 * linear with the K reduction split into slices (`x[m,k] · wᵀ[k,n] + bias[n]`, f32 weights, f32
 * compute) — a **reorder** strategy (ADR 0152 decision 2): the products are the reference kernel's,
 * only the order in which they are added differs.
 *
 * Two dispatches, the form of the split-K variants of the published `@huggingface/kernels` MatMul
 * (a partial pass with the slice on the dispatch's z axis, then a combine pass):
 *
 * 1. **Partial sums.** One thread owns one output element of one slice and adds the slice's products
 *    in ascending k into one f32 accumulator starting at 0 — the reference kernel's order inside a
 *    slice. It writes `partials[slice][row][col]`.
 * 2. **Combine.** One thread adds an element's partial sums in ascending slice order, then the bias,
 *    once, last (as the reference kernel does — src/kernels/linear.ts).
 *
 * What it changes: a single accumulator over K terms rounds K − 1 times against a growing sum, so its
 * error grows with K; `slices` accumulators of `K / slices` terms each round against sums that are
 * `slices` times smaller, and the combine adds `slices − 1` more roundings. With one slice the two
 * dispatches add the same terms in the same order as the reference kernel.
 *
 * Neither WGSL text carries a shape or the slice count (they are uniforms), so each has one key.
 *
 * NOTE: one thread reads its own `k / slices` elements of `x` and of `w` — two loads per
 * multiply-add, no shared tile. It is written for the accuracy of the sum, not for speed; the GEMM
 * skeleton's register blocking (src/kernels/gemm.ts) is what a fast split-K kernel would reuse.
 */

import { assertU32Params } from "../codegen/params.ts";

/** How many slices the K reduction is cut into. */
export type LinearSplitKSlices = 2 | 4 | 8 | 16;

/** The fewest terms one slice adds (below it the combine's extra roundings buy too little). */
export const LINEAR_SPLITK_MIN_SLICE = 256;

/** The edge of the square of output elements one workgroup of the partial pass covers. */
export const LINEAR_SPLITK_TILE = 8;

/** The threads of one workgroup of the combine pass (grid-stride). */
export const LINEAR_SPLITK_COMBINE_GROUP = 256;

/**
 * The slice count for a reduction length: the largest of 16 / 8 / 4 / 2 that divides `k` and leaves
 * each slice at least {@link LINEAR_SPLITK_MIN_SLICE} terms; `undefined` when there is none
 * (`k < 512`, or `k` odd).
 */
export const linearSplitKSlices = (k: number): LinearSplitKSlices | undefined =>
  ([16, 8, 4, 2] as const).find(
    (slices) => k % slices === 0 && k / slices >= LINEAR_SPLITK_MIN_SLICE,
  );

/** The bytes of the partial sums `[slices, m, n]` (f32) — the temporary between the two dispatches. */
export const linearSplitKPartialsBytes = (m: number, n: number, slices: number): number =>
  slices * m * n * 4;

/**
 * The uniform of both dispatches: `m`, `n`, `k`, `slices`.
 * MUST: `slices` divides `k` — a slice length that is not an integer would leave the last terms of
 * the reduction unread (a silent wrong value).
 */
export const linearSplitKParams = (
  m: number,
  n: number,
  k: number,
  slices: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("linear split-K params", { m, n, k, slices });
  if (slices < 1 || k % slices !== 0) {
    throw new RangeError(`linear split-K: ${slices} slices do not divide k = ${k}`);
  }
  // `partials` is indexed in u32: its element count has to be one.
  assertU32Params("linear split-K params", { partials: slices * m * n });
  return new Uint32Array([m, n, k, slices]);
};

const DIMS = `struct Dims {
  m: u32,
  n: u32,
  k: u32,
  slices: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;`;

export const linearSplitKPartialKey = (): string =>
  `linear_splitk_partial:v1:f32:wg${LINEAR_SPLITK_TILE}x${LINEAR_SPLITK_TILE}`;

export const linearSplitKPartialWgsl =
  (): string => `// karume linear split-K, pass 1 (partial sums of x[m,k] · wᵀ[k,n] over K slices, f32, one output per thread / wg ${LINEAR_SPLITK_TILE}x${LINEAR_SPLITK_TILE})
${DIMS}
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;

@compute @workgroup_size(${LINEAR_SPLITK_TILE}, ${LINEAR_SPLITK_TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let row = gid.y;
  let slice = gid.z;
  if (col >= dims.n || row >= dims.m) {
    return;
  }
  let per = dims.k / dims.slices;
  let first = slice * per;
  // Ascending k inside the slice, one accumulator from 0: the reference kernel's order.
  var acc = 0.0;
  for (var kk = first; kk < first + per; kk = kk + 1u) {
    acc = acc + x[row * dims.k + kk] * w[col * dims.k + kk];
  }
  partials[(slice * dims.m + row) * dims.n + col] = acc;
}
`;

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
