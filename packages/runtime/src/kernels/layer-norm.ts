/**
 * The fixed kernel of layer_norm (last dimension, with affine, f32).
 * `out[r, i] = (x[r, i] − mean_r) / sqrt(var_r + eps) · weight[i] + bias[i]`
 *
 * The input is treated as a contiguous `[rows, dim]` layout with the leading dimensions flattened
 * (the contract is normalisation over the last dimension only — src/ops.ts).
 *
 * ## Variance convention (MUST)
 *
 * **Population variance (divide by N, correction = 0)**. `torch.var` defaults to
 * `correction = 1` (divide by N−1), but the internal variance of normalisation layers divides by
 * N; mixing them up grows the error as the dimension gets smaller (the reverse trap of the
 * ledger's counterexample "`var` = `sum((x−mean)²)/N` is wrong").
 *
 * ## The scan decision (for correctness, not performance)
 *
 * One row = one workgroup (256 threads) in **two passes**: (1) sum → mean (2) sum of squared
 * deviations → variance. The one-pass form `E[x²] − E[x]²` can give a negative variance through
 * cancellation (rows with a large `mean²` in f32), so it is not used. Rows use the same
 * grid-stride as reduce.ts, so a row count above the dispatch limit (spec default 65535) still
 * works.
 *
 * NOTE: the ledger's ruling "may the sequential row scan be copied" is answered here with **not
 * sequential** (a 256-thread tree reduction for a row length of 1024). reduce.ts is already a tree,
 * and copying the same form keeps the invariant "one row = one workgroup" consistent across the
 * family. As a target for replacement in the performance milestone it is treated like the matmul
 * family.
 *
 * MUST: params are passed as a uniform. There is a workgroupBarrier inside the row loop, and the
 * loop condition must be uniform within the workgroup (loads from storage are not proven uniform by
 * the uniformity analysis).
 */

import { CodegenError } from "../codegen/errors.ts";
import { f32EpsDefect } from "../codegen/f32-eps.ts";
import { assertU32Params } from "../codegen/params.ts";

export const LAYER_NORM_WORKGROUP_SIZE = 256;

export const LAYER_NORM_KEY = `layer_norm:v1:f32:lastdim:wg${LAYER_NORM_WORKGROUP_SIZE}`;

/**
 * The body that produces one row's statistics (`base` / `mean` / `inv`). **It MUST share the same
 * string with the fused kernel** — written separately, the plain sequence and the fused version
 * split in reduction order, and a single difference in the order of summing `(x−mean)²` breaks bit
 * equality (the same reason `SIGMOID_STABLE_WGSL` is shared with silu.ts).
 *
 * Premises (identifiers the user provides): `params` (rows / dim / eps), `x`, `scratch`
 * (`array<f32, {@link LAYER_NORM_WORKGROUP_SIZE}>`), `lid`, `dim`, `scale` (= 1/dim) and the
 * workgroup-uniform row loop variable `row`. It contains barriers, so this block can only sit
 * inside **uniform control flow**.
 */
export const LAYER_NORM_ROW_STATS_WGSL: string = `    let base = row * dim;

    // ① sum → mean
    var acc = 0.0;
    var i = lid;
    while (i < dim) {
      acc = acc + x[base + i];
      i = i + ${LAYER_NORM_WORKGROUP_SIZE}u;
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride = ${LAYER_NORM_WORKGROUP_SIZE / 2}u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = scratch[lid] + scratch[lid + stride];
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
    let mean = scratch[0u] * scale;
    // make every read of scratch finish before ② overwrites it
    workgroupBarrier();

    // ② sum of squared deviations → population variance (correction = 0)
    var sq = 0.0;
    var j = lid;
    while (j < dim) {
      let d = x[base + j] - mean;
      sq = sq + d * d;
      j = j + ${LAYER_NORM_WORKGROUP_SIZE}u;
    }
    scratch[lid] = sq;
    workgroupBarrier();
    var stride2 = ${LAYER_NORM_WORKGROUP_SIZE / 2}u;
    while (stride2 > 0u) {
      if (lid < stride2) {
        scratch[lid] = scratch[lid] + scratch[lid + stride2];
      }
      workgroupBarrier();
      stride2 = stride2 / 2u;
    }
    let inv = 1.0 / sqrt(scratch[0u] * scale + params.eps);
    // make every read of scratch[0] finish before the next row overwrites scratch[lid]
    workgroupBarrier();`;

/**
 * The affine of one normalised element (the expression that applies `weight` / `bias`). For the
 * same reason as {@link LAYER_NORM_ROW_STATS_WGSL} it **MUST share the same string** with the fused
 * kernel (whether it contracts to an fma depends on how the expression is written). It assumes the
 * in-row index `base + o` and the weight index `o`.
 */
export const LAYER_NORM_AFFINE_WGSL = "(x[base + o] - mean) * inv * weight[o] + bias[o]";

export const LAYER_NORM_WGSL: string = `// karume layer_norm (last dim, affine, f32, 2 passes / population variance)
struct Params {
  rows: u32,
  dim: u32,
  eps: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> scratch: array<f32, ${LAYER_NORM_WORKGROUP_SIZE}>;

@compute @workgroup_size(${LAYER_NORM_WORKGROUP_SIZE})
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
${LAYER_NORM_ROW_STATS_WGSL}

    var o = lid;
    while (o < dim) {
      out[base + o] = ${LAYER_NORM_AFFINE_WGSL};
      o = o + ${LAYER_NORM_WORKGROUP_SIZE}u;
    }
    row = row + nwg.x;
  }
}
`;

/**
 * The uniform Params (rows / dim / eps. 16 bytes are allocated even for three words of content —
 * allocating in 16-byte units is a project convention (the fusion pass fixes the same form in ADR
 * 0040), not a requirement of the specification).
 *
 * MUST: eps is stored as an f32 bit pattern. Written as a u32, the exponent turns into an integer
 * value, and it runs with "eps ≈ 0" without an exception.
 *
 * MUST: the range of eps is checked **as an f32** (redundant with the contract check, but a check
 * that the direct kernel-call path also passes). Even when finite and positive in f64, `1e-50`
 * becomes 0 and `1e39` becomes `+Inf` in f32, and both silently diverge from the CPU reference
 * (reference/ops.ts), which computes in f64. A non-zero subnormal is also flushed to 0 on FTZ GPUs,
 * so it is rejected (the accepted set is the same as the contract check — the test is shared in
 * codegen/f32-eps.ts).
 */
export const layerNormParams = (
  rows: number,
  dim: number,
  eps: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("layer_norm params", { rows, dim });
  if (dim < 1) {
    throw new CodegenError(`layer_norm params: dim は正整数（${dim}）`);
  }
  // The test is shared with the op contract (assertEps in ops/attrs.ts) — so that neither side
  // alone lets a subnormal through.
  const defect = f32EpsDefect(eps);
  if (defect !== undefined) {
    throw new CodegenError(
      `layer_norm params: eps は f32 として有限の正数で、正規数（2^-126 以上）に丸まる値（${eps}・${defect}）`,
    );
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  new Float32Array(params.buffer)[2] = eps;
  return params;
};
