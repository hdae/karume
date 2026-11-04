/**
 * The fixed kernel of softmax / safe_softmax (last dimension, f32).
 *
 * ## safe-softmax (MUST)
 *
 * Built in the form `exp(x − amax_r)`. The naive `exp(x) / Σexp(x)` breaks exactly as in the
 * ledger's counterexamples:
 *
 * - Large positive values: `exp(89) > the largest finite f32` gives `inf`, and `inf / inf = NaN`.
 * - Large negative values: `exp(−104) < the smallest f32 subnormal` makes every term 0, and
 *   `0 / 0 = NaN`. A row that went through masked_fill's fill value −3.4028234663852886e+38
 *   **always** lands in this region.
 *
 * With amax subtracted, the exponent's argument is always in `(−∞, 0]`, and the largest element
 * always yields `exp(0) = 1`, so the denominator is never 0. A row whose elements are all the same
 * finite value (including a fully masked row filled with the finite sentinel −F32_MAX) gives the
 * uniform distribution `1/dim`, matching torch.
 * A row of all −inf is outside this guarantee. Plain softmax's identity is the finite −F32_MAX, so
 * `amax = −F32_MAX`, `Σ exp(−inf − amax) = 0`, and `1.0 / 0` is an indeterminate value in WGSL.
 * This input violates the contract and is not checked (ADR 0044 decision 3 — an all −inf row is
 * canonical only for the safe_softmax variant).
 *
 * The input is treated as a contiguous `[rows, dim]` layout with the leading dimensions flattened.
 * One row = one workgroup (256 threads) in **three passes**: (1) amax (2) `Σ exp(x − amax)` (3)
 * write-out. (2) and (3) evaluate `exp` twice to avoid an intermediate buffer of the row length
 * (correctness first — a target for replacement in the performance milestone). `exp` of the same
 * argument returns the same value, so determinism holds.
 *
 * MUST: params are passed as a uniform (there is a workgroupBarrier inside the row loop, and the
 * loop condition must be uniform within the workgroup).
 * MUST: the row max of (1) is `nan_max` (NaN test on the bit pattern — ADR 0020). WGSL's `max`
 * drops NaN at the specification level ("Returns e2 if e1 is less than e2, and e1 otherwise"), so
 * with a plain `max` **a row of all NaN passes safe_softmax's empty-row test, becomes exactly 0,
 * and the NaN disappears silently** (a row with some NaN already propagated before v2 through the sum of (2)
 * — the hole was only the all-NaN row). For non-NaN inputs nan_max = plain max, so the existing
 * numeric results do not move by a single bit (v2).
 *
 * ## The safe_softmax variant (ADR 0044)
 *
 * The contract is the same as softmax plus "**a row whose max is −inf is written as all 0**" (the
 * same semantics as the safe-softmax guard torch's SDPA decomposition emits). **Both are emitted
 * from the same generator** MUST — if the reduction order or the rounding points of (2) and (3)
 * differ from softmax by even one word, bit identity with the decomposed path (the parity test)
 * breaks. The variant adds only these three points:
 *
 * 1. The identity of (1) is **−inf** (plain softmax uses −F32_MAX). In a row with at least one
 *    finite element the max is dominated by that element, so amax has **the same bit pattern** in
 *    both, and the only difference is in rows of all −inf — only there does amax become −inf and
 *    tell them apart.
 * 2. If `empty` (row max is −inf), the subtrahend of (2) is replaced with 0. The replacement keeps
 *    `−inf − (−inf) = NaN` from being produced at all; in non-empty rows `select` returns the
 *    original value as is.
 * 3. The write-out of (3) is wrapped in `select(…, 0.0, empty)`. In an empty row (2) has a zero
 *    denominator (the result of division by 0 is indeterminate in WGSL), so 0 is fixed **on the
 *    side that discards the value**.
 *
 * MUST: do not write out empty rows early with a branch (`if`). `amax` comes from workgroup
 * memory, and a `workgroupBarrier` under that condition trips WGSL's uniformity analysis. With
 * `select` the barrier stays at the same position on both paths.
 * MUST: the −inf bit pattern travels in params (the same reason as gather / embedding's NaN — an
 * implementation may reject the constant expression `bitcast<f32>(0xff800000u)` with a shader
 * creation error, "const-expression is inf").
 */

import { IS_NAN_BITS_WGSL, NAN_MAX_WGSL } from "../codegen/numerics-wgsl.ts";
import { CodegenError } from "../codegen/errors.ts";
import { assertRowLoopBound, assertU32Params } from "../codegen/params.ts";

export const SOFTMAX_WORKGROUP_SIZE = 256;

// v2: row max via nan_max (closes the silent zeroing of all-NaN rows — the MUST above)
export const SOFTMAX_KEY = `softmax:v2:f32:lastdim:safe:wg${SOFTMAX_WORKGROUP_SIZE}`;

export const SAFE_SOFTMAX_KEY = `safe_softmax:v2:f32:lastdim:safe:emptyrow0:wg${SOFTMAX_WORKGROUP_SIZE}`;

/** The largest finite f32. WGSL has no infinity literal, so this is amax's identity. */
const F32_MAX = "3.402823466e38";

/** f32 bit pattern of −inf (the third params word of safe_softmax). */
export const SAFE_SOFTMAX_NEG_INF_BITS = 0xff800000;

/**
 * The WGSL of softmax (the safe_softmax variant with `safe`). This one function is the source of
 * truth, and the variant touches only the three points the doc above lists (the shared part
 * MUST NOT be duplicated into two).
 */
const softmaxWgsl = (safe: boolean): string =>
  `// karume ${safe ? "safe_softmax" : "softmax"} (last dim, f32, safe-softmax: exp(x - amax)${
    safe ? ", rows whose max is -inf are all 0" : ""
  })
struct Params {
  rows: u32,
  dim: u32,
${safe ? "  neg_inf: u32,\n" : ""}}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

${IS_NAN_BITS_WGSL}

${NAN_MAX_WGSL}

var<workgroup> scratch: array<f32, ${SOFTMAX_WORKGROUP_SIZE}>;

@compute @workgroup_size(${SOFTMAX_WORKGROUP_SIZE})
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid3: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let lid = lid3.x;
  let dim = params.dim;
${safe ? "  let neg_inf = bitcast<f32>(params.neg_inf);\n" : ""}  var row = wid.x;
  while (row < params.rows) {
    let base = row * dim;

    // ① row maximum (the subtrahend of safe-softmax${safe ? " — identity is -inf" : ""})
    var hi = ${safe ? "neg_inf" : `-${F32_MAX}`};
    var i = lid;
    while (i < dim) {
      hi = nan_max(hi, x[base + i]);
      i = i + ${SOFTMAX_WORKGROUP_SIZE}u;
    }
    scratch[lid] = hi;
    workgroupBarrier();
    var stride = ${SOFTMAX_WORKGROUP_SIZE / 2}u;
    while (stride > 0u) {
      if (lid < stride) {
        scratch[lid] = nan_max(scratch[lid], scratch[lid + stride]);
      }
      workgroupBarrier();
      stride = stride / 2u;
    }
${
  safe
    ? `    let row_max = scratch[0u];
    // a row of all -inf (the row torch's guard returns 0 for) — the subtrahend becomes 0 so no
    // NaN is made
    let empty = row_max == neg_inf;
    let amax = select(row_max, 0.0, empty);
`
    : "    let amax = scratch[0u];\n"
}    // make every read of scratch finish before ② overwrites it
    workgroupBarrier();

    // ② Σ exp(x - amax) (the largest element yields exp(0) = 1, so the denominator is always ≥ 1)
    var acc = 0.0;
    var j = lid;
    while (j < dim) {
      acc = acc + exp(x[base + j] - amax);
      j = j + ${SOFTMAX_WORKGROUP_SIZE}u;
    }
    scratch[lid] = acc;
    workgroupBarrier();
    var stride2 = ${SOFTMAX_WORKGROUP_SIZE / 2}u;
    while (stride2 > 0u) {
      if (lid < stride2) {
        scratch[lid] = scratch[lid] + scratch[lid + stride2];
      }
      workgroupBarrier();
      stride2 = stride2 / 2u;
    }
    let inv = 1.0 / scratch[0u];
    // make every read of scratch[0] finish before the next row overwrites scratch[lid]
    workgroupBarrier();

    // ③ write-out (exp takes the same argument as in ② — it returns the same value, so it is
    // deterministic)
    var o = lid;
    while (o < dim) {
      out[base + o] = ${
        safe ? "select(exp(x[base + o] - amax) * inv, 0.0, empty)" : "exp(x[base + o] - amax) * inv"
      };
      o = o + ${SOFTMAX_WORKGROUP_SIZE}u;
    }
    row = row + nwg.x;
  }
}
`;

export const SOFTMAX_WGSL: string = softmaxWgsl(false);

export const SAFE_SOFTMAX_WGSL: string = softmaxWgsl(true);

/**
 * The uniform Params. 16 bytes are allocated even for two words of content (allocating in 16-byte
 * units is a project convention (the fusion pass fixes the same form in ADR 0040), not a
 * requirement of the specification).
 *
 * With `safe`, the third word carries the −inf bit pattern (safe_softmax `bitcast`s it on the WGSL
 * side and uses it as the identity of ① and for the empty-row test).
 */
export const softmaxParams = (
  rows: number,
  dim: number,
  safe = false,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("softmax params", { rows, dim });
  // MUST: the in-row loops step by the workgroup size.
  assertRowLoopBound("softmax params", { dim }, SOFTMAX_WORKGROUP_SIZE);
  if (dim < 1) {
    throw new CodegenError(`softmax params: dim must be a positive integer (${dim})`);
  }
  const params = new Uint32Array(4);
  params[0] = rows;
  params[1] = dim;
  if (safe) params[2] = SAFE_SOFTMAX_NEG_INF_BITS;
  return params;
};
