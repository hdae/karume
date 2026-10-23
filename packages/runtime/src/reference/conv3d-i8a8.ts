/**
 * **CPU reference** of w8a8 conv3d (per-pixel i8 activations × per-channel i8 weights — ADR 0133).
 *
 * The numeric contract is **agreement with the GPU at atol = 0** (class E of ADR 0110), on the same
 * grounds as w4a8 linear ({@link "./i8a8.ts"}): the dot product over `Cin` of one tap is exact
 * `i32` addition, and the only floating-point operations are `f32(acci)`, one `fma` per tap in
 * ascending tap order and the final `fma` with the per-channel weight scale. `fma` is the one
 * operation without a spec guarantee of single rounding (fused on the development machine; Metal
 * differs by 1 ULP — docs/known-issues.md).
 *
 * MUST: do not copy the GPU side's tile structure. A plain loop over `(co, n, tap, ci)` is the
 * specification; the tap is **the quantisation axis of the activation scale** (one scale per input
 * pixel, and the pixel changes per tap), not a tile.
 * MUST: an out-of-range tap (padding, the time edges) is **skipped** with `fma(0, 0, accf)`, the
 * same operation as the GPU's zero-filled tile with scale 0 (it only normalises −0 to +0 — the
 * comparison treats ±0 as equal).
 * MUST: the activation is quantised with the mirror of `quantize_rows` on the transposed
 * `[T·H·W, Cin]` layout ({@link quantizeRowsReference}), so the tie-margin caveat of that mirror
 * applies (the tests measure it).
 */

import type { Conv3dDims } from "../kernels/conv3d.ts";
import { I8a8Error, quantizeRowsReference } from "./i8a8.ts";

export type Conv3dI8a8Input = {
  /** Activations `[Cin, T, H, W]` (f32, before quantisation). */
  readonly x: ArrayLike<number>;
  /** **Quantised integer values** of the weight `[Cout, Cin, Kt, Kh, Kw]` (i8). */
  readonly weight: ArrayLike<number>;
  /** Per-channel scale of the weight `[Cout]`. */
  readonly weightScale: ArrayLike<number>;
  /** bias `[Cout]` (always f32 — ADR 0006). */
  readonly bias: ArrayLike<number>;
  readonly dims: Conv3dDims;
};

/**
 * Reference of w8a8 conv3d — `out[co, n] = fma(Σ_τ fma(f32(Σ_ci xq·wq), xs[p(n, τ)], ·), wscale[co], bias[co])`.
 *
 * Returns the row-major output `[Cout, Tout, Hout, Wout]`.
 */
export const referenceConv3dI8a8 = (input: Conv3dI8a8Input): Float32Array<ArrayBuffer> => {
  const { x, weight, weightScale, bias, dims } = input;
  const {
    channelsIn,
    channelsOut,
    timeIn,
    heightIn,
    widthIn,
    timeOut,
    heightOut,
    widthOut,
    kernelT,
    kernelH,
    kernelW,
    strideT,
    strideH,
    strideW,
    paddingT,
    paddingH,
    paddingW,
    dilationT,
    dilationH,
    dilationW,
    groups,
  } = dims;
  if (groups !== 1) {
    throw new I8a8Error(`conv3d i8a8 reference: groups must be 1 (${groups})`);
  }
  if (channelsIn % 4 !== 0) {
    throw new I8a8Error(
      `conv3d i8a8 reference: channels_in must be a multiple of 4 (${channelsIn})`,
    );
  }
  const pixelsIn = timeIn * heightIn * widthIn;
  // MUST: check the element counts before computing (missing elements would propagate as NaN and
  // return an out-of-contract expected value).
  const taps = kernelT * kernelH * kernelW;
  for (const [name, actual, expected] of [
    ["x", x.length, channelsIn * pixelsIn],
    ["weight", weight.length, channelsOut * channelsIn * taps],
    ["weightScale", weightScale.length, channelsOut],
    ["bias", bias.length, channelsOut],
  ] as const) {
    if (actual !== expected) {
      throw new I8a8Error(
        `conv3d i8a8 reference: ${name} has ${actual} elements, expected ${expected}`,
      );
    }
  }
  // Per-pixel quantisation = quantize_rows on the transposed [T·H·W, Cin] layout
  const transposed = new Float32Array(pixelsIn * channelsIn);
  for (let ci = 0; ci < channelsIn; ci += 1) {
    for (let pixel = 0; pixel < pixelsIn; pixel += 1) {
      transposed[pixel * channelsIn + ci] = x[ci * pixelsIn + pixel];
    }
  }
  const { q, scale } = quantizeRowsReference(transposed, pixelsIn, channelsIn);
  const out = new Float32Array(channelsOut * timeOut * heightOut * widthOut);
  for (let co = 0; co < channelsOut; co += 1) {
    const weightRow = co * channelsIn * taps;
    for (let ot = 0; ot < timeOut; ot += 1) {
      for (let oy = 0; oy < heightOut; oy += 1) {
        for (let ox = 0; ox < widthOut; ox += 1) {
          let accf = 0;
          for (let kt = 0; kt < kernelT; kt += 1) {
            const it = ot * strideT + kt * dilationT - paddingT;
            for (let kh = 0; kh < kernelH; kh += 1) {
              const iy = oy * strideH + kh * dilationH - paddingH;
              for (let kw = 0; kw < kernelW; kw += 1) {
                const ix = ox * strideW + kw * dilationW - paddingW;
                const tap = (kt * kernelH + kh) * kernelW + kw;
                if (it < 0 || it >= timeIn || iy < 0 || iy >= heightIn || ix < 0 || ix >= widthIn) {
                  // Out of range: the GPU folds a zero-filled tile with scale 0 — fma(0, 0, accf),
                  // which only normalises −0 to +0
                  accf = Math.fround(accf + 0);
                  continue;
                }
                const pixel = (it * heightIn + iy) * widthIn + ix;
                // Exact integer sum (|acci| ≤ Cin·127·128 is far below 2^53)
                let acci = 0;
                for (let ci = 0; ci < channelsIn; ci += 1) {
                  acci += q[pixel * channelsIn + ci] * weight[weightRow + ci * taps + tap];
                }
                accf = Math.fround(Math.fround(acci) * scale[pixel] + accf);
              }
            }
          }
          out[((co * timeOut + ot) * heightOut + oy) * widthOut + ox] = Math.fround(
            accf * weightScale[co] + bias[co],
          );
        }
      }
    }
  }
  return out;
};
