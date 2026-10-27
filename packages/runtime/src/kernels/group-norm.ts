/**
 * The fixed kernels of group_norm (`x[N,C,…]`, channel groups, per-channel affine, f32 — ADR NNNN
 * decision 7).
 *
 * Group g of sample n is the contiguous run of `C/G` channels with every trailing element, so the
 * input is treated as `rows = N·G` contiguous rows of `length = (C/G)·inner` elements
 * (`inner` = the product of the trailing extents). One node is **five dispatches**:
 *
 * | Step | Key                         | Role                                                         |
 * | ---- | --------------------------- | ------------------------------------------------------------ |
 * | ①    | {@link GROUP_NORM_PARTIAL_KEY} pass 0 | per 1,024-element chunk: `Σ x` → `partial[row, chunk]` |
 * | ②    | {@link GROUP_NORM_TOTAL_KEY} pass 0   | per row: `Σ partial` → `mean = Σ · (1/length)`          |
 * | ③    | {@link GROUP_NORM_PARTIAL_KEY} pass 1 | per chunk: `Σ (x − mean)²` → `partial[row, chunk]`     |
 * | ④    | {@link GROUP_NORM_TOTAL_KEY} pass 1   | per row: `rstd = 1 / sqrt(Σ partial · (1/length) + eps)` |
 * | ⑤    | {@link GROUP_NORM_APPLY_KEY}          | every element: `(x − mean) · rstd · weight[c] + bias[c]` |
 *
 * ## Why two stages (for correctness and for the shape, not only for speed)
 *
 * A VAE at full resolution has 32 rows of millions of elements (`[192, H, W]` at the area cap of
 * the MOSS decoder: 6 channels × 2.79M pixels ≈ 16.8M elements per group). layer_norm's form (one
 * row = one workgroup) would run 32 workgroups that each loop millions of times — the "group_norm =
 * plain sum" counter-example of docs/op-vocabulary.md. The partial stage spreads a row over
 * `ceil(length / 1024)` workgroups (grid-stride over `rows · chunks` — more than the 65,535
 * workgroups of one dispatch dimension at that size), and the total stage folds a row's partials in
 * one workgroup.
 *
 * ## Two passes (MUST — the layer_norm discipline)
 *
 * Mean first, then the sum of squared deviations (population variance, divide by `length`). The
 * one-pass `E[x²] − E[x]²` cancels catastrophically when the mean is large against the spread (an
 * input with mean 100 and spread 1 keeps about one significant digit of the variance in f32), and
 * can give a negative variance.
 *
 * ## Determinism (MUST — fixed tree)
 *
 * The addition order depends on the shape only, never on the device or on the dispatch count:
 *
 * - a chunk is 1,024 elements at fixed offsets; thread `t` adds elements `t, t+256, t+512, t+768`
 *   of the chunk in that order, then the 256 values meet in a halving tree (`s[t] += s[t+stride]`,
 *   stride 128 → 1) — the same tree as layer_norm;
 * - a row's partials are added by thread `t` in the order `t, t+256, …`, then the same tree.
 *
 * The grid-stride loops only choose **which workgroup** computes a chunk or a row; the value of
 * each is the same whichever does it (the geometry invariance the grid-stride test checks with a
 * lowered dispatch limit).
 *
 * ## Scale and eps
 *
 * `1/length` is computed on the host in f64 and rounded once to f32 (the shader does not divide —
 * WGSL f32 division may be 2.5 ULP, and `f32(length)` is inexact above 2^24). eps is an f32 word
 * (`codegen/f32-eps.ts` — the same accepted set as the op contract).
 *
 * ## The f32 hazard range (docs/op-vocabulary.md "Check for numerically hazardous classes")
 *
 * - `1 / sqrt(variance + eps)`: eps is a normal f32 (≥ 2⁻¹²⁶ — the contract), so the argument is
 *   ≥ eps and the result ≤ 1/sqrt(eps) is finite; a zero-variance group gives exactly the bias
 *   (`x − mean = 0` when the mean is exact — canary in `gpu_group_norm_test.ts` ③).
 * - The sums overflow only beyond |x| ≈ 3.4e38 / length (pass 0) or |x − mean| ≈ 1.8e19 / √length
 *   (pass 1) — far outside normalisation inputs; torch's f32 group norm overflows there too. An
 *   overflowed variance gives rstd = 0 (the bias), not NaN.
 * - NaN or ±Inf in a group makes that group's statistics NaN (`Inf − Inf`) and leaves the other
 *   groups untouched (canaries in ③).
 *
 * MUST: params are uniforms. Every loop that contains a barrier has a workgroup-uniform condition
 * (derived from `workgroup_id`, `num_workgroups` and the uniform params — the uniformity analysis
 * does not prove storage loads uniform).
 */

import { CodegenError } from "../codegen/errors.ts";
import { f32EpsDefect } from "../codegen/f32-eps.ts";
import { assertU32Params } from "../codegen/params.ts";

export const GROUP_NORM_WORKGROUP_SIZE = 256;

/** Elements per partial sum (each of the 256 threads adds 4 — module doc "Determinism"). */
export const GROUP_NORM_CHUNK = 1024;

const ITEMS_PER_THREAD = GROUP_NORM_CHUNK / GROUP_NORM_WORKGROUP_SIZE;

export const GROUP_NORM_PARTIAL_KEY = `group_norm_partial:v1:f32:chunk${GROUP_NORM_CHUNK}:wg${GROUP_NORM_WORKGROUP_SIZE}`;
export const GROUP_NORM_TOTAL_KEY = `group_norm_total:v1:f32:wg${GROUP_NORM_WORKGROUP_SIZE}`;
export const GROUP_NORM_APPLY_KEY = `group_norm_apply:v1:f32:wg${GROUP_NORM_WORKGROUP_SIZE}`;

/** Pass 0 sums the values (→ the mean), pass 1 the squared deviations (→ the variance). */
export type GroupNormPass = 0 | 1;

/** The 256 → 1 halving tree over `scratch` (shared by both reducing kernels — one tree shape). */
const TREE_WGSL = `    workgroupBarrier();
    var stride = ${GROUP_NORM_WORKGROUP_SIZE / 2}u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = scratch[lid] + scratch[lid + stride];
      }
      workgroupBarrier();
      stride = stride / 2u;
    }`;

export const GROUP_NORM_PARTIAL_WGSL: string = `// karume group_norm partial sums (pass 0: x, pass 1: (x - mean)^2; f32, ${GROUP_NORM_CHUNK}-element chunks, fixed tree)
struct Params {
  rows: u32,
  length: u32,
  chunks: u32,
  phase: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read_write> partial: array<f32>;

var<workgroup> scratch: array<f32, ${GROUP_NORM_WORKGROUP_SIZE}>;

@compute @workgroup_size(${GROUP_NORM_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let total = params.rows * params.chunks;
  var p = wid.x;
  while (p < total) {
    let row = p / params.chunks;
    let begin = (p - row * params.chunks) * ${GROUP_NORM_CHUNK}u;
    let base = row * params.length;
    var mean = 0.0;
    if (params.phase == 1u) {
      mean = stats[2u * row];
    }
    var acc = 0.0;
    for (var k = 0u; k < ${ITEMS_PER_THREAD}u; k = k + 1u) {
      let i = begin + k * ${GROUP_NORM_WORKGROUP_SIZE}u + lid;
      if (i < params.length) {
        let v = x[base + i];
        if (params.phase == 0u) {
          acc = acc + v;
        } else {
          let d = v - mean;
          acc = acc + d * d;
        }
      }
    }
    scratch[lid] = acc;
${TREE_WGSL}
    if (lid == 0u) {
      partial[p] = scratch[0u];
    }
    // make every read of scratch[0] finish before the next chunk overwrites scratch[lid]
    workgroupBarrier();
    p = p + nwg.x;
  }
}
`;

export const GROUP_NORM_TOTAL_WGSL: string = `// karume group_norm totals (pass 0: mean, pass 1: 1 / sqrt(variance + eps); f32, fixed tree)
struct Params {
  rows: u32,
  chunks: u32,
  phase: u32,
  inv_length: f32,
  eps: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> partial: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;

var<workgroup> scratch: array<f32, ${GROUP_NORM_WORKGROUP_SIZE}>;

@compute @workgroup_size(${GROUP_NORM_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  var row = wid.x;
  while (row < params.rows) {
    let base = row * params.chunks;
    var acc = 0.0;
    var j = lid;
    while (j < params.chunks) {
      acc = acc + partial[base + j];
      j = j + ${GROUP_NORM_WORKGROUP_SIZE}u;
    }
    scratch[lid] = acc;
${TREE_WGSL}
    if (lid == 0u) {
      if (params.phase == 0u) {
        stats[2u * row] = scratch[0u] * params.inv_length;
      } else {
        stats[2u * row + 1u] = 1.0 / sqrt(scratch[0u] * params.inv_length + params.eps);
      }
    }
    // make every read of scratch[0] finish before the next row overwrites scratch[lid]
    workgroupBarrier();
    row = row + nwg.x;
  }
}
`;

export const GROUP_NORM_APPLY_WGSL: string = `// karume group_norm apply ((x - mean) * rstd * weight[c] + bias[c]; f32, grid-stride, full-write)
struct Params {
  count: u32,
  length: u32,
  inner: u32,
  groups: u32,
  per_group: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> weight: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(${GROUP_NORM_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let step = nwg.x * ${GROUP_NORM_WORKGROUP_SIZE}u;
  var e = wid.x * ${GROUP_NORM_WORKGROUP_SIZE}u + lid3.x;
  while (e < params.count) {
    let row = e / params.length;
    let o = e - row * params.length;
    let c = (row % params.groups) * params.per_group + o / params.inner;
    out[e] = (x[e] - stats[2u * row]) * stats[2u * row + 1u] * weight[c] + bias[c];
    e = e + step;
  }
}
`;

/** The row view of one group_norm node ({@link groupNormLayout}). */
export type GroupNormLayout = {
  /** `N·G` — one row per group of each sample. */
  readonly rows: number;
  /** Elements per row (`(C/G)·inner`). */
  readonly length: number;
  /** The product of the trailing extents (elements per channel). */
  readonly inner: number;
  readonly groups: number;
  /** Channels per group (`C/G`). */
  readonly perGroup: number;
  /** Partial sums per row (`ceil(length / 1024)`). */
  readonly chunks: number;
  /** Every element (`rows · length`). */
  readonly count: number;
};

/**
 * The row view of `x[N,C,…]` with `groups` groups. MUST: the single source of the layout for the
 * recipe builder and the estimate (the temporaries' bytes — {@link groupNormTempBytes}).
 */
export const groupNormLayout = (shape: readonly number[], groups: number): GroupNormLayout => {
  if (shape.length < 2 || shape.some((dim) => !Number.isInteger(dim) || dim < 1)) {
    throw new CodegenError(
      `group_norm layout: x must be [N,C,…] with positive extents ([${shape.join(",")}])`,
    );
  }
  const [batch, channels] = shape;
  if (!Number.isInteger(groups) || groups < 1 || channels % groups !== 0) {
    throw new CodegenError(
      `group_norm layout: ${groups} groups do not divide ${channels} channels`,
    );
  }
  const inner = shape.slice(2).reduce((count, dim) => count * dim, 1);
  const perGroup = channels / groups;
  const length = perGroup * inner;
  const rows = batch * groups;
  const count = rows * length;
  const chunks = Math.ceil(length / GROUP_NORM_CHUNK);
  assertU32Params("group_norm layout", {
    rows,
    length,
    inner,
    chunks,
    count,
    "rows * chunks": rows * chunks,
  });
  return { rows, length, inner, groups, perGroup, chunks, count };
};

/** The bytes of the node's two temporaries: the partial sums and the per-row (mean, rstd) pairs. */
export const groupNormTempBytes = (
  layout: GroupNormLayout,
): { readonly partials: number; readonly stats: number } => ({
  partials: layout.rows * layout.chunks * 4,
  stats: layout.rows * 2 * 4,
});

/** The uniform of the partial kernel (16 bytes: rows / length / chunks / pass). */
export const groupNormPartialParams = (
  layout: GroupNormLayout,
  pass: GroupNormPass,
): Uint32Array<ArrayBuffer> => {
  const params = new Uint32Array(4);
  params[0] = layout.rows;
  params[1] = layout.length;
  params[2] = layout.chunks;
  params[3] = pass;
  return params;
};

/**
 * The uniform of the total kernel (32 bytes — rows / chunks / pass / `1/length` / eps, padded to
 * the project's 16-byte units).
 *
 * MUST: eps is stored as an f32 bit pattern and checked as an f32 (the layer_norm params rule —
 * `1e-50` would become 0 and `1e39` infinity in f32, both silently diverging from the f64 CPU
 * reference).
 */
export const groupNormTotalParams = (
  layout: GroupNormLayout,
  pass: GroupNormPass,
  eps: number,
): Uint32Array<ArrayBuffer> => {
  const defect = f32EpsDefect(eps);
  if (defect !== undefined) {
    throw new CodegenError(
      `group_norm params: eps must be a finite positive f32 value that rounds to a normal number (at least 2^-126) (${eps}, ${defect})`,
    );
  }
  const params = new Uint32Array(8);
  const floats = new Float32Array(params.buffer);
  params[0] = layout.rows;
  params[1] = layout.chunks;
  params[2] = pass;
  floats[3] = 1 / layout.length;
  floats[4] = eps;
  return params;
};

/** The uniform of the apply kernel (32 bytes — count / length / inner / groups / per_group). */
export const groupNormApplyParams = (layout: GroupNormLayout): Uint32Array<ArrayBuffer> => {
  const params = new Uint32Array(8);
  params[0] = layout.count;
  params[1] = layout.length;
  params[2] = layout.inner;
  params[3] = layout.groups;
  params[4] = layout.perGroup;
  return params;
};
