/**
 * torch's uint8 antialiased bicubic resize — a port of the path
 * `torchvision.transforms.v2.functional.resize(uint8, BICUBIC, antialias=True)` takes on the CPU
 * (`torch.nn.functional.interpolate` on a uint8 tensor, `UpSampleKernel.cpp` and
 * `UpSampleKernelAVXAntialias.h`).
 *
 * ## Why this path is separate from `resizeRgb8` and from `pillow-resample.ts`
 *
 * All three are the same separable filter and differ only in their rounding, which is the point of
 * an exact port:
 *
 * | Path | Weights | Accumulation |
 * | ---- | ------- | ------------ |
 * | `resizeRgb8` (`preprocess.ts`) | f64 | f64, rounded per pass |
 * | `pillow-resample.ts` | fixed 22 bits | integers |
 * | here | int16, a precision per axis derived from its largest weight | integers |
 *
 * Measured on noise images (torch 2.13.0): Pillow differs from this path in 0.42–0.81% of the
 * samples, by at most 2. torch's own result does not depend on the CPU: `ATEN_CPU_CAPABILITY` of
 * default, AVX2 and AVX-512 gave the same bytes, and uint8 bicubic is the path torchvision always
 * takes on the CPU, so the port has one answer to match. It matches the upstream processor byte for
 * byte (`gemma4_vision_preprocess_test.ts`).
 *
 * NOTE: `export` serves internal callers in `src/` and the tests (not exposed on `mod.ts` / the
 * subpath surface — ADR 0008).
 */

// Third-party code notice. The weights (`axisTaps`: the filter, the support, the normalisation and
// the int16 rescale with its per-axis precision) and the two passes are ported from
// `HelperInterpCubic` / `_compute_index_ranges_int16_weights` in pytorch/pytorch
// (`aten/src/ATen/native/cpu/UpSampleKernel.cpp`) and `ImagingResampleHorizontal` /
// `ImagingResampleVertical` (`UpSampleKernelAVXAntialias.h`), PyTorch 2.13.0. Specialized to 3
// interleaved channels, and the SIMD loops are written as scalar loops. License: BSD 3-Clause, the
// full notice is in `THIRD_PARTY_NOTICES.md` at the package root. Upstream copyright notice:
//
//   Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)

import { ModelInputError } from "../errors.ts";
import { assertRgb8, type Rgb8Image } from "./preprocess.ts";

/** The bicubic support radius (at scale 1) and its coefficient (`a = -0.5` under antialias). */
const SUPPORT = 2;
const A = -0.5;

/** torch's search stops here: it tries precisions below 22. */
const MAX_PRECISION = 22;

const bicubic = (value: number): number => {
  const x = Math.abs(value);
  if (x < 1) return ((A + 2) * x - (A + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * A;
  return 0;
};

/** One axis's taps: for output `i`, `count[i]` weights at `weights[i * width..]`, from `first[i]`. */
type AxisTaps = {
  readonly first: Int32Array<ArrayBuffer>;
  readonly count: Int32Array<ArrayBuffer>;
  readonly weights: Int32Array<ArrayBuffer>;
  readonly width: number;
  readonly precision: number;
};

const axisTaps = (inSize: number, outSize: number): AxisTaps => {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = SUPPORT * filterScale;
  const width = Math.ceil(support) * 2 + 1;
  const inverse = 1 / filterScale;
  const first = new Int32Array(outSize);
  const count = new Int32Array(outSize);
  const real = new Float64Array(outSize * width);
  let largest = 0;
  for (let i = 0; i < outSize; i++) {
    const center = scale * (i + 0.5);
    // MUST: truncation, as the C cast — `center − support + 0.5` can be negative and is clamped after.
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const size = Math.min(Math.min(Math.trunc(center + support + 0.5), inSize) - xmin, width);
    first[i] = xmin;
    count[i] = size;
    let total = 0;
    for (let j = 0; j < size; j++) {
      const w = bicubic((j + xmin - center + 0.5) * inverse);
      real[i * width + j] = w;
      total += w;
    }
    for (let j = 0; j < size; j++) {
      if (total !== 0) real[i * width + j] = (real[i * width + j] ?? 0) / total;
      largest = Math.max(largest, real[i * width + j] ?? 0);
    }
  }
  // The largest precision whose largest weight still fits int16.
  let precision = 0;
  while (precision < MAX_PRECISION && Math.trunc(0.5 + largest * 2 ** (precision + 1)) < 2 ** 15) {
    precision += 1;
  }
  const weights = new Int32Array(outSize * width);
  for (let k = 0; k < real.length; k++) {
    const v = (real[k] ?? 0) * 2 ** precision;
    weights[k] = Math.trunc(v < 0 ? -0.5 + v : 0.5 + v);
  }
  return { first, count, weights, width, precision };
};

const clip8 = (value: number): number => (value < 0 ? 0 : value > 255 ? 255 : value);

/**
 * Resizes RGB8 pixels. MUST: horizontal first, through a uint8 intermediate, and an axis whose size
 * does not change has no pass (torch's `need_horizontal` / `need_vertical`).
 */
export const resizeRgb8TorchBicubic = (
  image: Rgb8Image,
  width: number,
  height: number,
): Rgb8Image => {
  assertRgb8(image);
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new ModelInputError(`Output size ${width}×${height} is not a positive integer`);
  }
  let source: Uint8Array<ArrayBufferLike> = image.data;
  if (width !== image.width) {
    const taps = axisTaps(image.width, width);
    const round = 1 << (taps.precision - 1);
    const out = new Uint8Array(width * image.height * 3);
    for (let y = 0; y < image.height; y++) {
      const inRow = y * image.width * 3;
      const outRow = y * width * 3;
      for (let x = 0; x < width; x++) {
        const start = inRow + (taps.first[x] ?? 0) * 3;
        const size = taps.count[x] ?? 0;
        const tap = x * taps.width;
        for (let c = 0; c < 3; c++) {
          let sum = round;
          for (let j = 0; j < size; j++) {
            sum += (taps.weights[tap + j] ?? 0) * (source[start + j * 3 + c] ?? 0);
          }
          out[outRow + x * 3 + c] = clip8(sum >> taps.precision);
        }
      }
    }
    source = out;
  }
  if (height !== image.height) {
    const taps = axisTaps(image.height, height);
    const round = 1 << (taps.precision - 1);
    const stride = width * 3;
    const out = new Uint8Array(stride * height);
    for (let y = 0; y < height; y++) {
      const start = (taps.first[y] ?? 0) * stride;
      const size = taps.count[y] ?? 0;
      const tap = y * taps.width;
      for (let k = 0; k < stride; k++) {
        let sum = round;
        for (let j = 0; j < size; j++) {
          sum += (taps.weights[tap + j] ?? 0) * (source[start + j * stride + k] ?? 0);
        }
        out[y * stride + k] = clip8(sum >> taps.precision);
      }
    }
    source = out;
  }
  return { data: source === image.data ? source.slice() : source, width, height };
};
