/**
 * The fixed f32 kernel of DCNv2 (modulated deformable convolution — op `deform_conv2d`, ADR 0055).
 *
 * `out[b, oc, oy, ox] = Σ_{ic,kh,kw} (mask · bilinear(x[b,ic], y, x)) · w[oc,ic,kh,kw] + bias[oc]`,
 * `y = (oy − ph) + kh + off_y` / `x = (ox − pw) + kw + off_x`. stride / dilation / groups /
 * offset_groups **have no field in the contract = fixed at 1**, so they are not in params either.
 *
 * One invocation = **one output element** with a grid-stride loop; every output element is written
 * exactly once (the full-write invariant — ADR 0014). Writes are exclusively owned and the
 * reduction order is static, so "same input → same output" holds even with run-time offsets
 * (ADR 0055 decision 3).
 *
 * MUST: the reduction runs in ascending `(ic, kh, kw)`, **the same nesting as the direct conv2d
 * kernel** (src/kernels/conv2d.ts). The bit match with conv2d in the degenerate case (`offset` all
 * 0, `mask` all 1) is this op's only A/B oracle (tests/gpu_deform_conv2d_test.ts); rearranging the
 * loops changes the reduction order and breaks that check. **The reads of `offset` / `mask`
 * SHOULD NOT be hoisted out of the `ic` loop** (semantically possible since `offset_groups` is 1,
 * but it means rearranging the loops to `(kh,kw)` outside and `ic` inside) (re-graded 2026-10-18
 * — ADR 0055 decision 2). The address does not depend on `ic`, so a hoist by the shader compiler
 * keeps the order and is harmless.
 * MUST: the channel order of `offset` is **even = y / odd = x** (torchvision verbatim). Swapping
 * them can still give matching values with a square kernel and symmetric padding, so the tests pin
 * it with an asymmetric Kh ≠ Kw shape.
 * MUST: `mask` multiplies **after the bilinear interpolation and before the weight**
 * (`(m · v) · w`). Even the bracketing matches torchvision's form "im2col writes
 * `mask · bilinear`, and the GEMM takes `w × col`".
 * MUST: out of bounds is **zero padding, not border clamp**, in two stages (if the centre is
 * outside `(−1, in)` the whole tap is 0; inside, an out-of-range corner is 0 for that corner only).
 * This is **a different rule** from gather / embedding's "out of range = NaN contamination"
 * (docs/limitations.md); deform's out-of-bounds samples are normal semantics.
 * MUST: the range test is written in **the positive form** (`> −1 && < in`), and NaN is detected
 * on the bit pattern before it and propagated to the output. With the negative form
 * (`<= −1 || in <= `) NaN falls "inside" and goes straight to the undefined `i32(floor(NaN))`
 * (docs/limitations.md); with the positive form alone NaN silently becomes a 0 contribution (a
 * silent wrong value). Routing it through `clamp` would replay "the driver's max swallows NaN",
 * which ADR 0020 fixed at the root, so it is not used.
 * MUST: the NaN bit pattern travels in params. WGSL has no NaN literal, and some implementations
 * may fail shader creation with "const-expression is NaN" when `bitcast<f32>(...)` is written as a
 * constant expression (the same discipline as gather / embedding — src/kernels/gather.ts).
 * MUST: use grid-stride. In a real model (BiRefNet_HR's `decoder_block1`) the output is
 * `[1,256,512,512]` = 67.1 million elements, and the workgroups needed (262,144 at 256 threads) are
 * about four times the one-dimensional dispatch limit (default 65535).
 */

import { CodegenError } from "../codegen/errors.ts";
import { IS_NAN_BITS_WGSL } from "../codegen/numerics-wgsl.ts";
import { assertU32Params } from "../codegen/params.ts";

export const DEFORM_CONV2D_WORKGROUP_SIZE = 256;

/**
 * The quiet NaN bit pattern (f32) returned by a sample at a NaN coordinate.
 *
 * NOTE: the field name mirrors gather / embedding's `oob`, but this op's **out of range is zero
 * padding**, not NaN contamination (see the MUST above) — what this carries is "the value to
 * propagate when the offset is NaN".
 */
export const DEFORM_CONV2D_OOB_BITS = 0x7fc00000;

/**
 * MUST: bump the key when the WGSL changes (the pipeline cache does not look at the source).
 *
 * v2: the NaN test moved from a copied `is_nan` to the shared fragment {@link IS_NAN_BITS_WGSL}
 * (the test expression is identical).
 */
export const DEFORM_CONV2D_KEY = `deform_conv2d:v2:nchw:f32:dcnv2:wg${DEFORM_CONV2D_WORKGROUP_SIZE}`;

export const DEFORM_CONV2D_WGSL: string = `// karume deform_conv2d (DCNv2: x[B,Cin,H,W] * W[Cout,Cin,Kh,Kw] + b[Cout], f32, direct convolution)
struct Dims {
  n: u32,
  batch: u32,
  channels_in: u32,
  channels_out: u32,
  height_in: u32,
  width_in: u32,
  height_out: u32,
  width_out: u32,
  kernel_h: u32,
  kernel_w: u32,
  padding_h: u32,
  padding_w: u32,
  oob: u32,
}
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> offsets: array<f32>;
@group(0) @binding(4) var<storage, read> modulator: array<f32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

// NaN test on the bit pattern (ADR 0020) — exponent all ones and mantissa non-zero. Expressed with
// comparisons, the test disappears entirely under the driver's constant folding.
${IS_NAN_BITS_WGSL}

// bilinear sample of the input plane \`plane\`. Out of range is zero padding (per corner); NaN
// propagates.
fn deform_sample(plane: u32, sy: f32, sx: f32) -> f32 {
  if (is_nan_bits(sy) || is_nan_bits(sx)) {
    return bitcast<f32>(dims.oob);
  }
  // range test in the positive form. Past it, floor(sy) is in [-1, H-1], so the i32 conversion is
  // always defined.
  let inside = sy > -1.0 && sy < f32(dims.height_in)
    && sx > -1.0 && sx < f32(dims.width_in);
  if (!inside) {
    return 0.0;
  }
  let base_y = floor(sy);
  let base_x = floor(sx);
  let y0 = i32(base_y);
  let x0 = i32(base_x);
  let ly1 = sy - base_y;
  let lx1 = sx - base_x;
  let ly0 = 1.0 - ly1;
  let lx0 = 1.0 - lx1;
  let last_y = i32(dims.height_in) - 1;
  let last_x = i32(dims.width_in) - 1;
  var v00 = 0.0;
  var v01 = 0.0;
  var v10 = 0.0;
  var v11 = 0.0;
  if (y0 >= 0) {
    let row = plane + u32(y0) * dims.width_in;
    if (x0 >= 0) {
      v00 = x[row + u32(x0)];
    }
    if (x0 + 1 <= last_x) {
      v01 = x[row + u32(x0 + 1)];
    }
  }
  if (y0 + 1 <= last_y) {
    let row = plane + u32(y0 + 1) * dims.width_in;
    if (x0 >= 0) {
      v10 = x[row + u32(x0)];
    }
    if (x0 + 1 <= last_x) {
      v11 = x[row + u32(x0 + 1)];
    }
  }
  // weights and addition order follow torchvision verbatim (w1*v1 + w2*v2 + w3*v3 + w4*v4).
  return ly0 * lx0 * v00 + ly0 * lx1 * v01 + ly1 * lx0 * v10 + ly1 * lx1 * v11;
}

@compute @workgroup_size(${DEFORM_CONV2D_WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let step = nwg.x * ${DEFORM_CONV2D_WORKGROUP_SIZE}u;
  let taps = dims.kernel_h * dims.kernel_w;
  let plane_out = dims.height_out * dims.width_out;
  let plane_in = dims.height_in * dims.width_in;
  var i = gid.x;
  while (i < dims.n) {
    let chunk = dims.channels_out * plane_out;
    let b = i / chunk;
    let rest = i % chunk;
    let oc = rest / plane_out;
    let pixel = rest % plane_out;
    let oy = pixel / dims.width_out;
    let ox = pixel % dims.width_out;
    // the reference position on the input side is signed (it goes negative by the padding).
    // stride / dilation are fixed at 1.
    let origin_y = i32(oy) - i32(dims.padding_h);
    let origin_x = i32(ox) - i32(dims.padding_w);
    // offset / mask are indexed by B and the output pixel (offset_groups == 1, so there is no
    // group axis).
    let offset_base = b * 2u * taps * plane_out + pixel;
    let mask_base = b * taps * plane_out + pixel;
    var acc = bias[oc];
    for (var ic = 0u; ic < dims.channels_in; ic = ic + 1u) {
      let x_base = (b * dims.channels_in + ic) * plane_in;
      let w_base = (oc * dims.channels_in + ic) * taps;
      for (var kh = 0u; kh < dims.kernel_h; kh = kh + 1u) {
        for (var kw = 0u; kw < dims.kernel_w; kw = kw + 1u) {
          let tap = kh * dims.kernel_w + kw;
          // even channel = y / odd channel = x
          let sy = f32(origin_y + i32(kh)) + offsets[offset_base + 2u * tap * plane_out];
          let sx = f32(origin_x + i32(kw)) + offsets[offset_base + (2u * tap + 1u) * plane_out];
          let m = modulator[mask_base + tap * plane_out];
          acc = acc + (m * deform_sample(x_base, sy, sx)) * w[w_base + tap];
        }
      }
    }
    out[i] = acc;
    i = i + step;
  }
}
`;

/** Geometry of deform_conv2d (the one input type of params). */
type DeformConv2dDims = {
  readonly batch: number;
  readonly channelsIn: number;
  readonly channelsOut: number;
  readonly heightIn: number;
  readonly widthIn: number;
  readonly heightOut: number;
  readonly widthOut: number;
  readonly kernelH: number;
  readonly kernelW: number;
  readonly paddingH: number;
  readonly paddingW: number;
};

/**
 * The uniform Dims (13 words of content allocated as **16 words = 64 bytes** — allocating in
 * 16-byte units is a project convention (the fusion pass fixes the same form in ADR 0040), not a
 * requirement of the specification). The first word is the output's total element count, the last
 * field is {@link DEFORM_CONV2D_OOB_BITS}.
 *
 * MUST: the geometry is checked on the direct kernel-call path too (redundant with the contract
 * check = the positive set of the same name in ops/shapes.ts, but the same double check as
 * conv_transpose1d / conv2d). An input with spatial length 0 makes `f32(dims.height_in)` 0, the
 * range test false for every tap, and the output all bias — a silent wrong value.
 */
export const deformConv2dParams = (dims: DeformConv2dDims): Uint32Array<ArrayBuffer> => {
  // The names pair with the WGSL Dims field names (from the one after `n`). The order is the
  // uniform's word order as is.
  const dimensions = {
    batch: dims.batch,
    channels_in: dims.channelsIn,
    channels_out: dims.channelsOut,
    height_in: dims.heightIn,
    width_in: dims.widthIn,
    height_out: dims.heightOut,
    width_out: dims.widthOut,
    kernel_h: dims.kernelH,
    kernel_w: dims.kernelW,
    padding_h: dims.paddingH,
    padding_w: dims.paddingW,
  };
  assertU32Params("deform_conv2d params", dimensions);
  const values = Object.values(dimensions);
  const positive: readonly (readonly [string, number])[] = [
    ["batch", dims.batch],
    ["channels_in", dims.channelsIn],
    ["channels_out", dims.channelsOut],
    ["height_in", dims.heightIn],
    ["width_in", dims.widthIn],
    ["height_out", dims.heightOut],
    ["width_out", dims.widthOut],
    ["kernel_h", dims.kernelH],
    ["kernel_w", dims.kernelW],
  ];
  for (const [name, value] of positive) {
    if (value < 1)
      throw new CodegenError(`deform_conv2d params: ${name} must be a positive integer (${value})`);
  }
  const n = dims.batch * dims.channelsOut * dims.heightOut * dims.widthOut;
  assertU32Params("deform_conv2d params", { n });
  const params = new Uint32Array(16);
  params[0] = n;
  values.forEach((value, index) => {
    params[index + 1] = value;
  });
  params[values.length + 1] = DEFORM_CONV2D_OOB_BITS;
  return params;
};
