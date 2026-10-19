/**
 * A fixed f32 kernel that resamples the last two spatial axes of NCHW to any length by bilinear
 * interpolation (`align_corners = True` only — op `upsample_bilinear2d`).
 *
 * One invocation = **one output element**, which reads the input's four neighbours and writes
 * their weighted sum (output-driven). Every output element is written exactly once (the full-write
 * invariant — ADR 0014). Upscaling and downscaling use the same formula, and the ratio may be
 * non-integer (the weights differ per output position).
 *
 * The numerics follow torch's `aten/src/ATen/native/UpSample.h`:
 *
 * - scale = `(in − 1) / (out − 1)`, **0 for an axis of output length 1**
 *   (`area_pixel_compute_scale`)
 * - source coordinate = `scale · output index` (the `align_corners` side has no 0.5 shift)
 * - `index0 = min(trunc(source coordinate), in − 1)` /
 *   `index1 = index0 + (index0 < in − 1 ? 1 : 0)` / `λ1 = source coordinate − index0` /
 *   `λ0 = 1 − λ1` (`compute_source_index_and_lambda` and `guard_index_and_lambda`)
 * - the expression tree nests **H outside, W inside**:
 *   `λy0·(λx0·v00 + λx1·v01) + λy1·(λx0·v10 + λx1·v11)` (the same form as CUDA's
 *   `UpSampleBilinear2d.cu`)
 *
 * MUST: the upper bound `in − 1` of index0 is applied with **an integer min**. On a large axis the
 * f32 rounding of scale and of the source coordinate lets the last source coordinate reach `in`
 * (real examples: in = 8650802, out = 34 / in = 16777220, out = 2), and without the bound the end
 * point of a plane reads the start of the next plane (a wrong value without an exception). It
 * triggers only on shapes with an axis above roughly 2^23.
 * MUST: the special case of the last tap (`index1 = index0`) is written as **an integer
 * comparison**. Routing it through f32 `min` / `max` would replay the measured NaN swallowing (the
 * same kind as what ADR 0020 fixed at the root).
 * MUST: scale is **divided on the host and carried in params**. WGSL's f32 division is allowed up
 * to 2.5 ULP (addition, subtraction and multiplication are correctly rounded), so dividing in the
 * shader makes the output of the same shape **vary per backend**. Dividing on the host pins it to
 * one `Math.fround` rounding, the same value as torch's `float(in−1)/float(out−1)`.
 * NOTE: even with the division on the host, "the output's edges match the input's edges exactly"
 * does not hold in general. Shapes really exist where `fl(scale · (out−1))` falls below `in−1`
 * (measured: the last element for in=2 → out=42 is 0.9999999403953552). This is not a defect of
 * the implementation — **torch itself produces the same value** (it evaluates
 * `area_pixel_compute_scale` in float and multiplies `scale · dst_index` in float) — a result
 * faithful to the contract "the numerics follow torch" above; clamping the end points to make them
 * exact would instead split the bits from torch (ADR 0058). An exact match is limited to shapes
 * where `(in−1)/(out−1)` round-trips through f32 (powers of two and the like), and the 47
 * real-shape sites of the distributed models are all on that side.
 * MUST: the source coordinate is non-negative, so truncation by `u32(...)` equals floor — the
 * coordinate formula of `align_corners = False` (`scale·(i+0.5) − 0.5`) can go negative and breaks
 * this premise. That is why **the contract has no field** for it; widening the accepted set means
 * redesigning from this truncation.
 * MUST: use grid-stride. A real model's output is on the order of `[1,192,2048,2048]` = 805
 * million elements, and the workgroups needed (3,145,728 at 256 threads) are about 48 times the
 * one-dimensional dispatch limit (default 65535).
 */

import { CodegenError } from "../codegen/errors.ts";
import { assertU32Params } from "../codegen/params.ts";

export const UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE = 256;

/** MUST: bump the key when the WGSL changes (the pipeline cache does not look at the source). */
export const UPSAMPLE_BILINEAR2D_KEY = `upsample_bilinear2d:v2:nchw:f32:align_corners:wg${UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE}`;

export const UPSAMPLE_BILINEAR2D_WGSL: string = `// karume upsample_bilinear2d (NCHW bilinear, align_corners=true, f32, full-write)
struct Params {
  n: u32,
  in_h: u32,
  in_w: u32,
  out_h: u32,
  out_w: u32,
  // (in - 1) / (out - 1) divided by the host in f32 (0 for an axis of output length 1)
  scale_h: f32,
  scale_w: f32,
  reserved: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(${UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let stride = nwg.x * ${UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE}u;
  let out_hw = params.out_h * params.out_w;
  let in_hw = params.in_h * params.in_w;
  var i = gid.x;
  while (i < params.n) {
    // N and C take no part in the interpolation, so they fold into one plane index (rank is not
    // carried in params).
    let ox = i % params.out_w;
    let oy = (i / params.out_w) % params.out_h;
    let plane = i / out_hw;

    let ry = params.scale_h * f32(oy);
    let y0 = min(u32(ry), params.in_h - 1u);
    var y1 = y0;
    if (y0 + 1u < params.in_h) {
      y1 = y0 + 1u;
    }
    let ly1 = ry - f32(y0);
    let ly0 = 1.0 - ly1;

    let rx = params.scale_w * f32(ox);
    let x0 = min(u32(rx), params.in_w - 1u);
    var x1 = x0;
    if (x0 + 1u < params.in_w) {
      x1 = x0 + 1u;
    }
    let lx1 = rx - f32(x0);
    let lx0 = 1.0 - lx1;

    let row0 = plane * in_hw + y0 * params.in_w;
    let row1 = plane * in_hw + y1 * params.in_w;
    out[i] = ly0 * (lx0 * x[row0 + x0] + lx1 * x[row0 + x1])
      + ly1 * (lx0 * x[row1 + x0] + lx1 * x[row1 + x1]);
    i = i + stride;
  }
}
`;

/**
 * torch's `area_pixel_compute_scale` (the `align_corners` side). **0 for an axis of output length
 * 1** (writing `(in−1)/(out−1)` there divides by zero, and torch has the same special case).
 *
 * `Math.fround` rounds the f64 division to f32, but dividend and divisor are both small integers,
 * so no double rounding occurs (with a power-of-two divisor the quotient is exact in f32; otherwise
 * the quotient is an infinite binary fraction and never lands exactly on an f32 rounding boundary).
 * It equals torch's `float(in−1) / float(out−1)` only in the range `in − 1 ≤ 2^24` — beyond that
 * torch rounds the dividend to f32 first, which double-rounds and disagrees (in = 16777218, out = 6
 * gives 3355443.5 in torch against 3355443.25 here).
 */
const alignCornersScale = (lengthIn: number, lengthOut: number): number =>
  lengthOut > 1 ? Math.fround((lengthIn - 1) / (lengthOut - 1)) : 0;

/**
 * 32-byte uniform params (5 u32 words + 2 f32 words + 1 reserved word — allocating in 16-byte units
 * is a project convention (the fusion pass fixes the same form in ADR 0040)). `n` is the output's
 * total element count.
 *
 * MUST: a spatial axis of length 0 is rejected here. With 0 the scale `(0−1)/(out−1)` goes
 * negative and reads go outside the input (on the GPU, a neighbouring value comes out without an
 * exception).
 * MUST: `n % (out_h · out_w)` is rejected here. Letting a non-divisible value through shifts the
 * index decomposition of the last incomplete plane into the next plane — a silent wrong value that
 * the element count alone cannot detect (the same trap as upsample2x).
 */
export const upsampleBilinear2dParams = (
  n: number,
  heightIn: number,
  widthIn: number,
  heightOut: number,
  widthOut: number,
): Uint32Array<ArrayBuffer> => {
  assertU32Params("upsample_bilinear2d params", { n, heightIn, widthIn, heightOut, widthOut });
  for (const [name, value] of Object.entries({ heightIn, widthIn, heightOut, widthOut })) {
    if (value === 0) {
      throw new CodegenError(`upsample_bilinear2d params: ${name} must be at least 1`);
    }
  }
  const plane = heightOut * widthOut;
  if (n % plane !== 0) {
    throw new CodegenError(
      `upsample_bilinear2d params: n ${n} is not an integer multiple of the output plane ${heightOut}×${widthOut}`,
    );
  }
  const params = new Uint32Array([n, heightIn, widthIn, heightOut, widthOut, 0, 0, 0]);
  const floats = new Float32Array(params.buffer);
  floats[5] = alignCornersScale(heightIn, heightOut);
  floats[6] = alignCornersScale(widthIn, widthOut);
  return params;
};
