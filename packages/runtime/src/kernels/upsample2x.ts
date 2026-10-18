/**
 * A fixed f32 kernel that doubles the last two spatial axes of NCHW by nearest neighbour (the body
 * of the fusion rule `upsample2x` — src/runtime/fusion.ts).
 *
 * It reads the input flat as a contiguous `[N*C*H, W]`, and one invocation writes one input
 * element to a 2×2 block of the output. Output destinations do not overlap between input elements,
 * so neither atomics nor workgroup memory are needed. The f32 values are copied through a `u32`
 * bit view without arithmetic, so NaN payloads / subnormals / ±0 are not subject to the backend's
 * floating-point normalisation (this is the only rule that needs no rounding-barrier discussion).
 * Every output element is written exactly once (the full-write invariant — ADR 0014).
 *
 * MUST: use grid-stride. NCHW's batch / channel / height fold into row, and only width stays in
 * params, so no rank-specific coordinate recovery is brought in.
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

export const UPSAMPLE_2X_WORKGROUP_SIZE = 256;

/** MUST: bump the key when the WGSL changes (the pipeline cache does not look at the source). */
export const UPSAMPLE_2X_KEY = `upsample2x:v1:nchw:f32:wg${UPSAMPLE_2X_WORKGROUP_SIZE}`;

export const UPSAMPLE_2X_WGSL: string = `// karume upsample2x (NCHW nearest, f32 bits, full-write)
struct Params {
  n: u32,
  width: u32,
  out_width: u32,
  reserved: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(${UPSAMPLE_2X_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * ${UPSAMPLE_2X_WORKGROUP_SIZE}u;
  var i = gid.x;
  while (i < params.n) {
    let col = i % params.width;
    let row = i / params.width;
    let base = 2u * row * params.out_width + 2u * col;
    let value = x[i];
    out[base] = value;
    out[base + 1u] = value;
    out[base + params.out_width] = value;
    out[base + params.out_width + 1u] = value;
    i = i + stride;
  }
}
`;

/**
 * 16-byte uniform params. `n` is the input's total element count, `width` the last dimension of
 * the NCHW input.
 *
 * MUST: `n % width` is rejected here. Letting a non-divisible value through makes the 2×2 writes
 * of the last incomplete row overlap the next row — a silent wrong value that the total element
 * count alone cannot detect.
 */
export const upsample2xParams = (n: number, width: number): Uint32Array<ArrayBuffer> => {
  assertU32Params("upsample2x params", { n, width });
  if (width === 0) throw new CodegenError("upsample2x params: width は 1 以上");
  if (n % width !== 0) {
    throw new CodegenError(`upsample2x params: n ${n} が width ${width} の整数行でない`);
  }
  const outWidth = width * 2;
  const outN = n * 4;
  assertU32Params("upsample2x params", { "2 * width": outWidth, "4 * n": outN });
  return new Uint32Array([n, width, outWidth, 0]);
};
