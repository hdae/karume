/**
 * The fixed kernel of embedding (`out[…, h] = weight[index[…], h]`, f32 values / i32 indices).
 *
 * Output and weight are both contiguous, so with `row = i / H` and `col = i % H` from the output's
 * flat index `i`, one read `weight[index[row] * H + col]` suffices (H = the weight's last
 * dimension).
 *
 * MUST: grid-stride loop. Real models exceed the one-dimensional dispatch limit (spec default
 * 65535) (measured: V=22012 / H=1024).
 *
 * ## `padding_idx` has no effect on forward (contract — ADR 0012)
 *
 * torch's `padding_idx` is a field for "do not update the padding row through gradients"; the
 * forward pass is exactly a plain row gather (`F.embedding`'s forward does not reference it
 * either). So this kernel takes no attrs at all. The contract table (src/ops.ts) accepts and
 * validates it because dropping it in order to ignore it would open a hole in the "unknown attrs
 * fail loudly" discipline.
 *
 * ## Out-of-range indices (the same ruling as gather — src/kernels/gather.ts)
 *
 * The contract is `0 <= index < V`. A violation is **an error on the model side**, but when it
 * happens, not silently returning another row takes priority: **the GPU writes NaN
 * ({@link EMBEDDING_OOB_BITS}) only to the out-of-range elements**, and **the CPU reference
 * throws**. A kernel cannot throw, and notifying the host would need a per-run device-side fault
 * flag + readback (a new diagnostic channel), so that is not adopted in this phase.
 *
 * MUST: the NaN bit pattern travels in params. WGSL has no NaN literal, and some implementations
 * may fail shader creation with "const-expression is NaN" when `bitcast<f32>(...)` is written as a
 * constant expression.
 *
 * The weight has storage variants (`w=f32` / `w=f16` / `w=i8` / `w=i4` / `w=i2` — ADR 0018 / 0019
 * / 0069 / 0097).
 * They differ only in the one row-read line and in how the scale is read; the out-of-range ruling
 * is shared (src/kernels/weight-storage.ts). The scale read sits **inside** the in-range branch
 * (`wscale` is not read with an out-of-range `pick`).
 *
 * ## The i4 group scale is **per element** (ADR 0069 decision 3)
 *
 * i8's scale is per row (= vocabulary entry) and becomes loop-invariant once the row is known.
 * i4's group scale changes every `group_size` along the quantized axis (the D axis of the weight
 * `[V,D]`), so it is re-read from `col`: `wscale[pick * (D / group_size) + col / group_size]`. One
 * thread has only one output element, so there is nowhere to hoist it, and unlike linear (where
 * gemm.ts's fill reads it per quad) it is read directly here.
 * MUST: the group length is baked into the WGSL as a shift, so it **goes into the key too**
 * (`:wi4g32`).
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";
import {
  i4GroupKeyPart,
  i4GroupShift,
  WEIGHT_SCALE_VAR,
  weightArrayType,
  weightKeyPart,
  weightLoaderWgsl,
  weightNote,
  weightRead,
  weightScaleWgsl,
  type WeightStorage,
} from "./weight-storage.ts";

export const EMBEDDING_WORKGROUP_SIZE = 256;

/** i8 / i4 scale binding (the number after the output — paired with the executor bind entries). */
export const EMBEDDING_SCALE_BINDING = 4;

/** The quiet NaN bit pattern (f32) written to the output for an out-of-range index. */
export const EMBEDDING_OOB_BITS = 0x7fc00000;

/**
 * The line that binds the i4 group scale **for this one thread only** (i8 binds the row scale;
 * the others are the empty string).
 *
 * MUST: the binding name is {@link WEIGHT_SCALE_VAR}, the same as i8 — so the expression passed to
 * {@link weightRead} does not branch on the storage dtype. The line's position (inside the
 * in-range branch, its indentation) also matches i8.
 */
const embeddingScaleWgsl = (weight: WeightStorage, groupShift: number | undefined): string =>
  weight === "i4"
    ? `
      // the group scale depends on the quantized axis (column) — one element per thread, so it is
      // read per element, not hoisted
      let ${WEIGHT_SCALE_VAR} = wscale[u32(pick) * (dims.hidden >> ${groupShift}u) + (col >> ${groupShift}u)];`
    : weightScaleWgsl(weight, "u32(pick)", "      ");

export const embeddingKey = (weight: WeightStorage, groupSize?: number): string => {
  i4GroupShift("embedding", weight, groupSize);
  return `embedding:v1:f32:i32:wg${EMBEDDING_WORKGROUP_SIZE}${weightKeyPart(weight)}${i4GroupKeyPart(
    groupSize,
  )}`;
};

export const embeddingWgsl = (weight: WeightStorage, groupSize?: number): string =>
  `// karume embedding (out[..., h] = weight[index[...], h], f32 / index i32${weightNote(weight)})
struct Dims {
  n: u32,
  hidden: u32,
  vocab: u32,
  oob: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> weight: array<${weightArrayType(weight)}>;
@group(0) @binding(2) var<storage, read> index: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
${weightLoaderWgsl("weight", weight, EMBEDDING_SCALE_BINDING)}
@compute @workgroup_size(${EMBEDDING_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * ${EMBEDDING_WORKGROUP_SIZE}u;
  var i = gid.x;
  while (i < dims.n) {
    let row = i / dims.hidden;
    let col = i % dims.hidden;
    let pick = index[row];
    // an index outside the contract contaminates with NaN instead of returning another row (the
    // ruling in the kernel doc)
    if (pick < 0 || u32(pick) >= dims.vocab) {
      out[i] = bitcast<f32>(dims.oob);
    } else {${embeddingScaleWgsl(weight, i4GroupShift("embedding", weight, groupSize))}
      out[i] = ${weightRead("weight", weight, "u32(pick) * dims.hidden + col", WEIGHT_SCALE_VAR)};
    }
    i = i + stride;
  }
}
`;

/**
 * The uniform Dims (exactly 4 words = 16 bytes — the 16-byte unit of the project convention (the
 * fusion pass fixes the same form in ADR 0040)).
 * `hidden` may be 0 (then `n` is 0 too and the kernel's loop never runs).
 *
 * MUST: the i4 group length divides `hidden` (the same discipline as linear's `k % group_size`
 * check). Otherwise the scale index `pick·(hidden >> shift) + (col >> shift)` shifts across rows,
 * and the bounds clamp of storage reads turns it into **a silent wrong value without an
 * exception**. Through the IR the format layer's check protects it, but for the direct kernel-call
 * path this is the only check.
 */
export const embeddingParams = (
  count: number,
  hidden: number,
  vocab: number,
  groupSize?: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("embedding params", { count, hidden, vocab });
  // MUST: the weight's logical element count `vocab·hidden ≤ 2^32 − 1`. The WGSL builds
  // `u32(pick) * dims.hidden + col` as a u32 flat index, and the packed loaders (i2 / i4) turn it
  // into a word position. If the product wraps, the last row reads the integer codes of the first
  // row — a wrong value without an exception (i2 reaches it at about 1 GiB of storage).
  assertU32Params("embedding params", { "vocab * hidden": vocab * hidden });
  if (hidden === 0 && count !== 0) {
    // The combination that divides by zero (element count > 0 with no hidden) cannot arise under
    // the shape contract.
    throw new CodegenError(
      `embedding params: hidden 0 with element count ${count} is not a valid combination`,
    );
  }
  if (groupSize !== undefined && hidden % groupSize !== 0) {
    throw new CodegenError(
      `embedding params: hidden=${hidden} is not divisible by group_size ${groupSize}`,
    );
  }
  const params = new Uint32Array(4);
  params[0] = count;
  params[1] = hidden;
  params[2] = vocab;
  params[3] = EMBEDDING_OOB_BITS;
  return params;
};
