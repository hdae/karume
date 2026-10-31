/**
 * The kernel of rms_norm (last dimension, weight only, f32) (ADR 0017).
 * `out[r, i] = x[r, i] · rsqrt(mean_r(x²) + eps) · weight[i]`
 *
 * The input is treated as a contiguous `[rows, dim]` layout with the leading dimensions flattened
 * (the contract is normalisation over the last dimension only — src/ops.ts).
 *
 * ## Difference from layer_norm (MUST)
 *
 * **The mean is not subtracted**. layer_norm is forced into "two passes (mean → sum of squared
 * deviations)" to avoid cancellation in the population variance; rms_norm's reduction is **a
 * single sum of squares**, so one pass suffices (no cancellation like `E[x²] − E[x]²` can arise
 * structurally). There is no bias either — by ADR 0017's arity-2 contract (two inputs, x and
 * weight).
 *
 * ## Scan and addition order
 *
 * One row = one workgroup (128 threads for dim ≤ 128, 256 otherwise) with a tree reduction; rows
 * use the same grid-stride as reduce.ts / layer-norm.ts, so a row count above the dispatch limit
 * (spec default 65535) still works (measured: Qwen3's [1,512,1024] is 512 rows, but the DiT's QK
 * norm multiplies the row count by the head axis).
 *
 * For dim ≤ 128, the upper half of the 256 version holds only zero sums of squares. Only the first
 * +0 stage is skipped, keeping the lower half's addition order and the output expression (ADR 0017
 * addendum 1).
 *
 * ## The prefetch form (v2 — perf-ledger K-99, ADR 0017 addendum 2)
 *
 * Where a row is a multiple of 256 elements up to {@link RMS_NORM_PREFETCH_MAX_DIM}, the 256-wide
 * kernel and the RMS → add fusion take v2: the output loop runs a constant `dim / 256` trips (the
 * trip count is in the WGSL and the key), the weight words are loaded into registers once per
 * workgroup before the row loop, and each row's x words (and the residual words of the add) are
 * loaded before its reduction. v1's output loop re-read them after the reduction behind a runtime
 * trip count, which made a lone row latency-bound (RTX 3080 Ti, dependent chain, M = 1: 5.0 → 2.8 µs
 * at dim 1536, 9.9 → 3.8 µs at 4096). Only the source of each operand changes: the reduction is the
 * same text and the value is {@link rmsNormValueWgsl} with the registers as its operands, so v2
 * writes v1's words (checked word for word against v1 on the GPU — gpu_rms_norm_prefetch_test.ts).
 * Other widths keep v1's text and key.
 *
 * MUST: params are passed as a uniform. There is a workgroupBarrier inside the row loop, and the
 * loop condition must be uniform within the workgroup (loads from storage are not proven uniform by
 * the uniformity analysis).
 * MUST: the reciprocal square root is `inverseSqrt` (the same single operation as torch's
 * `rsqrt`). The CPU reference is written in the different form `1 / Math.sqrt(…)` — copying the
 * same expression would let a rounding error cancel out on both sides.
 */

import { CodegenError } from "../codegen/errors.ts";
import { f32EpsDefect } from "../codegen/f32-eps.ts";
import { assertU32Params } from "../codegen/params.ts";

export const RMS_NORM_WORKGROUP_SIZE = 256;

/** MUST: bump the key when the WGSL changes (do not mix different sources under one key). */
export const RMS_NORM_KEY = `rms_norm:v1:f32:lastdim:wg${RMS_NORM_WORKGROUP_SIZE}`;

/** Input order of the following add. Code generation keeps the left / right of the addition too. */
export type RmsNormAddOrder = "norm-residual" | "residual-norm";

/**
 * The reduction of one row, from the sum of squares to `inv` and the barrier that ends the reads of
 * `scratch[0]` (inside the row loop; it reads `x`, `base`, `dim`, `scale`, `lid` and `params.eps`).
 *
 * MUST: a fused kernel that normalises a row takes this text, never a copy. The bit identity of the
 * plain kernel rests on the driver contracting `acc + v * v` exactly as here (research
 * 2026-10-25-speed-recon-3080ti §6: a register-array rewrite of the same loop moved 2,317 scales on
 * Vulkan), so only the same string compiles to the same arithmetic.
 */
export const rmsNormInvWgsl = (workgroupSize: 128 | 256): string => `    // sum of squares (1 pass)
    var acc = 0.0;
    var i = lid;
    while (i < dim) {
      let v = x[base + i];
      acc = acc + v * v;
      i = i + ${workgroupSize}u;
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride = ${workgroupSize / 2}u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = scratch[lid] + scratch[lid + stride];
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let inv = inverseSqrt(scratch[0u] * scale + params.eps);
    // make every read of scratch[0] finish before the next row overwrites scratch[lid]
    workgroupBarrier();
`;

/**
 * The normalised value of element `index` of the row (`x · inv · weight`, left to right) — shared by
 * every form that writes or uses it, under the MUST of {@link rmsNormInvWgsl}. `registers` names
 * the registers that already hold the element's x and weight words (the prefetch form); only the
 * operands' source changes, never the expression.
 */
export const rmsNormValueWgsl = (
  index: string,
  registers: { readonly x?: string; readonly weight?: string } = {},
): string =>
  `${registers.x ?? `x[base + ${index}]`} * inv * ${registers.weight ?? `weight[${index}]`}`;

const rmsNormWgsl = (workgroupSize: 128 | 256): string =>
  `// karume rms_norm (last dim, weight only, f32, sum of squares in 1 pass)
struct Params {
  rows: u32,
  dim: u32,
  eps: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, ${workgroupSize}>;

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let scale = 1.0 / f32(dim);
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

${rmsNormInvWgsl(workgroupSize)}
    var o = lid;
    while (o < dim) {
      out[base + o] = ${rmsNormValueWgsl("o")};
      o = o + ${workgroupSize}u;
    }
    row = row + nwg.x;
  }
}
`;

// The 256 version keeps its existing source and key. This is pure WGSL generation; no GPU work.
export const RMS_NORM_WGSL: string = rmsNormWgsl(RMS_NORM_WORKGROUP_SIZE);
export const RMS_NORM_128_KEY = "rms_norm:v1:f32:lastdim:wg128";
export const RMS_NORM_128_WGSL: string = rmsNormWgsl(128);

/**
 * The widest row the prefetch form takes (16 trips). It was measured faster than v1 at every
 * multiple of 256 from 256 to 4096 and every row count from 1 to 32,768 (perf-ledger K-99); wider
 * rows hold more registers per thread and are unmeasured, so they keep v1.
 */
export const RMS_NORM_PREFETCH_MAX_DIM = 4096;

/**
 * The constant trip count of the prefetch form's output loop for a row of `dim` elements, or
 * `undefined` where the 256-wide kernel keeps v1 (`dim` not a multiple of 256, or above
 * {@link RMS_NORM_PREFETCH_MAX_DIM}).
 */
export const rmsNormPrefetchTrip = (dim: number): number | undefined =>
  Number.isSafeInteger(dim) &&
  dim >= RMS_NORM_WORKGROUP_SIZE &&
  dim <= RMS_NORM_PREFETCH_MAX_DIM &&
  dim % RMS_NORM_WORKGROUP_SIZE === 0
    ? dim / RMS_NORM_WORKGROUP_SIZE
    : undefined;

/**
 * The prefetch form (v2) for rows of exactly `trip · 256` elements: the weight words in registers
 * before the row loop, the row's x (and residual) words in registers before its reduction, and a
 * constant-trip output loop. Element `o = lid + t · 256` is register `t` of thread `lid`, so
 * every index is in bounds and the loops are workgroup-uniform.
 */
const rmsNormPrefetchWgsl = (trip: number, addOrder?: RmsNormAddOrder): string => {
  const value = rmsNormValueWgsl("o", { x: "xv[t]", weight: "wv[t]" });
  const store =
    addOrder === undefined
      ? `out[base + o] = ${value};`
      : addOrder === "norm-residual"
        ? `out[base + o] = bitcast<f32>(bitcast<u32>(${value}) ^ params.rounding_mask) + rv[t];`
        : `out[base + o] = rv[t] + bitcast<f32>(bitcast<u32>(${value}) ^ params.rounding_mask);`;
  const loop = (body: string): string => `for (var t = 0u; t < ${trip}u; t = t + 1u) { ${body} }`;
  const step = `t * ${RMS_NORM_WORKGROUP_SIZE}u`;
  return `// karume rms_norm (last dim, weight only, f32, sum of squares in 1 pass; weight, x${addOrder === undefined ? "" : ", residual"} prefetched, ${trip} elements per thread)
struct Params {
  rows: u32,
  dim: u32,
  eps: f32,${addOrder === undefined ? "" : "\n  rounding_mask: u32,"}
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(${addOrder === undefined ? 3 : 4}) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, ${RMS_NORM_WORKGROUP_SIZE}>;

${
  addOrder === undefined ? "" : "@group(0) @binding(3) var<storage, read> residual: array<f32>;\n"
}@compute @workgroup_size(${RMS_NORM_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
  let scale = 1.0 / f32(dim);
  var wv: array<f32, ${trip}>;
  ${loop(`wv[t] = weight[lid + ${step}];`)}
  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;
    var xv: array<f32, ${trip}>;
    ${loop(`xv[t] = x[base + lid + ${step}];`)}${
      addOrder === undefined
        ? ""
        : `
    var rv: array<f32, ${trip}>;
    ${loop(`rv[t] = residual[base + lid + ${step}];`)}`
    }

${rmsNormInvWgsl(RMS_NORM_WORKGROUP_SIZE)}
    for (var t = 0u; t < ${trip}u; t = t + 1u) {
      let o = lid + ${step};
      ${store}
    }
    row = row + nwg.x;
  }
}
`;
};

const requirePrefetchTrip = (dim: number, where: string): number => {
  const trip = rmsNormPrefetchTrip(dim);
  if (trip === undefined) {
    throw new CodegenError(
      `${where}: no prefetch form for a row of ${dim} elements (a multiple of ${RMS_NORM_WORKGROUP_SIZE} up to ${RMS_NORM_PREFETCH_MAX_DIM})`,
    );
  }
  return trip;
};

/**
 * The key of the 256-wide rms_norm for rows of `dim` elements: the prefetch form's
 * (`rms_norm:v2:…:trip<dim / 256>` — the trip count is in the WGSL) where
 * {@link rmsNormPrefetchTrip} has one, else v1's {@link RMS_NORM_KEY}.
 */
export const rmsNormWorkgroupKey = (dim: number): string => {
  const trip = rmsNormPrefetchTrip(dim);
  return trip === undefined
    ? RMS_NORM_KEY
    : `rms_norm:v2:f32:lastdim:wg${RMS_NORM_WORKGROUP_SIZE}:trip${trip}`;
};

/** The WGSL of {@link rmsNormWorkgroupKey}. */
export const rmsNormWorkgroupWgsl = (dim: number): string => {
  const trip = rmsNormPrefetchTrip(dim);
  return trip === undefined ? RMS_NORM_WGSL : rmsNormPrefetchWgsl(trip);
};

/** The plain rms_norm kernels (the 128- / 256-wide workgroup forms here, the subgroup form of ADR 0100). */
export type RmsNormKernel = "workgroup128" | "workgroup256" | "subgroup32";

/**
 * The plain rms_norm kernel a row of `dim` elements runs: the 128-wide workgroup form for dim ≤ 128,
 * else the subgroup form when the Session asks for it (`rmsNormReduce: "subgroup32"`), else the
 * 256-wide workgroup form ({@link rmsNormWorkgroupKey} — v1 or its bit-identical prefetch form).
 *
 * MUST: this is the one decision — `buildRmsNorm` (runtime/recipe-builders/norm.ts) and the fusion
 * rule `rmsNormRope` (whose kernel shares the 256-wide form's reduction text and is bit-identical to
 * it only) both ask it, so the fused form exists exactly where the plain path runs that kernel.
 */
export const rmsNormKernel = (
  dim: number,
  reduce: "workgroup" | "subgroup32" | undefined,
): RmsNormKernel =>
  dim > 0 && dim <= 128 ? "workgroup128" : reduce === "subgroup32" ? "subgroup32" : "workgroup256";

/**
 * Opt-in RMS → add fusion (DECIDED: docs/decisions/0099-rms-norm-add-fusion.md).
 * Connected directly, the last multiply and the add fuse and the rounding changes, so an integer
 * XOR with a uniform 0 is inserted. The value is the fourth word of rmsNormParams (0). Do not
 * simplify it into a constant-expression bitcast round trip.
 * It matches the reference in measurement, but WGSL does not guarantee bit identity on every
 * backend. The workgroup reduction and the grid-stride are shared with the standalone RMS, and the
 * public default path does not change.
 *
 * The fused kernel is the prefetch form (v2) of the standalone RMS, so it exists only for the rows
 * {@link rmsNormPrefetchTrip} accepts (the rule's measured widths are all among them); any other
 * `dim` throws. v1 (the runtime-trip output loop) is kept test-only in
 * tests/fixtures/wgsl-baseline/ for the GPU identity test.
 */
export const rmsNormAddWgsl = (order: RmsNormAddOrder, dim: number): string => {
  assertRmsNormAddOrder(order);
  return rmsNormPrefetchWgsl(requirePrefetchTrip(dim, "rms_norm_add"), order);
};

export const rmsNormAddKey = (order: RmsNormAddOrder, dim: number): string => {
  assertRmsNormAddOrder(order);
  requirePrefetchTrip(dim, "rms_norm_add");
  return `rms_norm_add:v2:${rmsNormWorkgroupKey(dim)}:${order}:xor-round`;
};

const assertRmsNormAddOrder = (order: RmsNormAddOrder): void => {
  if (order !== "norm-residual" && order !== "residual-norm") {
    throw new CodegenError(`rms_norm_add: unsupported addition order '${String(order)}'`);
  }
};

/**
 * The uniform Params (rows / dim / eps. 16 bytes are allocated even for three words of content —
 * allocating in 16-byte units is a project convention (the fusion pass fixes the same form in ADR
 * 0040), not a requirement of the specification).
 *
 * MUST: eps is stored as an f32 bit pattern. Written as a u32, the exponent turns into an integer
 * value, and it runs with "eps ≈ 0" without an exception.
 *
 * MUST: the range of eps is checked **as an f32** (the same double check as layer_norm params).
 * Even when finite and positive in f64, `1e-50` becomes 0 and `1e39` becomes `+Inf` in f32, and
 * both diverge from the CPU reference, which computes in f64. A non-zero subnormal is also flushed
 * to 0 on FTZ GPUs, so it is rejected (the test is shared in codegen/f32-eps.ts).
 */
export const rmsNormParams = (rows: number, dim: number, eps: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("rms_norm params", { rows, dim });
  if (dim < 1) {
    throw new CodegenError(`rms_norm params: dim must be a positive integer (${dim})`);
  }
  // The test is shared with the op contract (assertEps in ops/attrs.ts) — so that neither side
  // alone lets a subnormal through.
  const defect = f32EpsDefect(eps);
  if (defect !== undefined) {
    throw new CodegenError(
      `rms_norm params: eps must be a finite positive f32 value that rounds to a normal number (at least 2^-126) (${eps}, ${defect})`,
    );
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  new Float32Array(params.buffer)[2] = eps;
  return params;
};
