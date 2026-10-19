/**
 * The fixed kernel of gather (last dimension fixed, `out[..., j] = src[..., index[..., j]]`, f32
 * values / i32 indices).
 *
 * Output, src and index are all contiguous, and the leading dimensions of src and index match (the
 * contract — src/ops.ts). So with `row = i / J` from the output's flat index i, one read
 * `src[row * D + index[i]]` suffices (D = src's last dimension, J = the output's last dimension).
 *
 * MUST: grid-stride loop. Real models exceed the one-dimensional dispatch limit (spec default
 * 65535).
 *
 * ## Out-of-range indices (the canonical ruling is ADR 0061)
 *
 * The contract is `0 <= index < D` (indices are in range). On a violation, **not silently
 * returning another element** takes priority, and only the affected elements are contaminated with
 * quiet NaN:
 *
 * - **GPU**: writes NaN ({@link GATHER_OOB_BITS}) only to the out-of-range elements. The check is
 *   two comparisons per element and gather is memory-bound, so the effective cost is almost 0.
 *   Leaving it unchecked and relying on WebGPU's bounded access would quietly return "0 or another
 *   in-bounds element", which passes both downstream and the comparisons. The NaN stays in that
 *   kernel's output and propagates to allclose, the golden comparison and later arithmetic.
 *   However, it can disappear in later steps that drop NaN, such as comparisons, where and casts to
 *   int (comparisons return false for NaN — the same as torch).
 * - **Why not an exception**: a kernel cannot throw, and notifying the host would need a per-run
 *   device-side fault flag + readback (a new diagnostic channel). That is not a mechanism to add
 *   along with gather, so it stops at NaN contamination.
 * - **The CPU reference is strict** (throws when out of range). In the step 2 comparison the
 *   oracle side fails first.
 *
 * MUST: the NaN bit pattern travels in params. WGSL has no NaN literal, and some implementations
 * may fail shader creation with "const-expression is NaN" when `bitcast<f32>(...)` is written as a
 * constant expression.
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

export const GATHER_WORKGROUP_SIZE = 256;

/** The quiet NaN bit pattern (f32) written to the output for an out-of-range index. */
export const GATHER_OOB_BITS = 0x7fc00000;

export const GATHER_KEY = `gather:v1:f32:i32:lastdim:wg${GATHER_WORKGROUP_SIZE}`;

export const GATHER_WGSL: string = `// karume gather (out[..., j] = src[..., index[..., j]], last dim fixed, f32 / index i32)
struct Dims {
  n: u32,
  cols: u32,
  src_cols: u32,
  oob: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> index: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(${GATHER_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * ${GATHER_WORKGROUP_SIZE}u;
  var i = gid.x;
  while (i < dims.n) {
    let row = i / dims.cols;
    let pick = index[i];
    // an index outside the contract contaminates with NaN instead of returning another element
    // (the ruling in the kernel doc)
    if (pick < 0 || u32(pick) >= dims.src_cols) {
      out[i] = bitcast<f32>(dims.oob);
    } else {
      out[i] = src[row * dims.src_cols + u32(pick)];
    }
    i = i + stride;
  }
}
`;

/**
 * The uniform Dims (exactly 4 words = 16 bytes — the 16-byte unit of the project convention (the
 * fusion pass fixes the same form in ADR 0040)).
 * `cols` may be 0 (then `n` is 0 too and the kernel's loop never runs).
 */
export const gatherParams = (
  count: number,
  cols: number,
  srcCols: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("gather params", { count, cols, srcCols });
  if (cols === 0 && count !== 0) {
    // The combination that divides by zero (element count > 0 with no columns) cannot arise under
    // the shape contract.
    throw new CodegenError(
      `gather params: column count 0 with element count ${count} is not a valid combination`,
    );
  }
  const params = new Uint32Array(4);
  params[0] = count;
  params[1] = cols;
  params[2] = srcCols;
  params[3] = GATHER_OOB_BITS;
  return params;
};
