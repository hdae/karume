// Fixture of gpu_gridstride_test.ts (the DegenerateCase type and the case builders per kernel
// family). The execution harness (runDegenerate / Deno.test) and the assembly of the case list stay
// in gpu_gridstride_test.ts; this file only "produces the reduced-dispatch cases of one family" —
// dependencies go one way, helpers → src.

import {
  ELEMENTWISE_WORKGROUP_SIZE,
  elementwiseKey,
  elementwiseParams,
  elementwiseWgsl,
} from "../../src/codegen/elementwise.ts";
import {
  AXIS_REDUCE_SPLIT,
  axisReduceKey,
  axisReduceOutputsPerGroup,
  axisReduceParams,
  axisReduceWgsl,
  reduceKey,
  reduceParams,
  reduceWgsl,
} from "../../src/codegen/reduce.ts";
import {
  expandSrcStrides,
  permuteSrcStrides,
  STRIDED_WORKGROUP_SIZE,
  STRIDED_WRITE_WORKGROUP_SIZE,
  stridedKey,
  stridedParams,
  stridedWgsl,
  stridedWriteKey,
  stridedWriteParams,
  stridedWriteWgsl,
} from "../../src/codegen/strided.ts";
import { ARGMAX_KEY, ARGMAX_WGSL, argmaxParams } from "../../src/kernels/argmax.ts";
import { topkKey, topkParams, topkWgsl } from "../../src/kernels/topk.ts";
import {
  CONV1D_WORKGROUP_SIZE,
  conv1dKey,
  conv1dParams,
  conv1dWgsl,
} from "../../src/kernels/conv1d.ts";
import {
  CONV2D_WORKGROUP_SIZE,
  conv2dKey,
  conv2dParams,
  conv2dWgsl,
} from "../../src/kernels/conv2d.ts";
import {
  CONV_TRANSPOSE1D_WORKGROUP_SIZE,
  convTranspose1dKey,
  convTranspose1dParams,
  convTranspose1dWgsl,
} from "../../src/kernels/conv-transpose1d.ts";
import {
  CUMSUM_KEY,
  CUMSUM_WGSL,
  CUMSUM_WORKGROUP_SIZE,
  cumsumParams,
} from "../../src/kernels/cumsum.ts";
import {
  EMBEDDING_WORKGROUP_SIZE,
  embeddingKey,
  embeddingParams,
  embeddingWgsl,
} from "../../src/kernels/embedding.ts";
import { FLIP_KEY, FLIP_WGSL, FLIP_WORKGROUP_SIZE, flipParams } from "../../src/kernels/flip.ts";
import {
  GATHER_KEY,
  GATHER_WGSL,
  GATHER_WORKGROUP_SIZE,
  gatherParams,
} from "../../src/kernels/gather.ts";
import { LAYER_NORM_KEY, LAYER_NORM_WGSL, layerNormParams } from "../../src/kernels/layer-norm.ts";
import {
  GROUP_NORM_APPLY_KEY,
  GROUP_NORM_APPLY_WGSL,
  GROUP_NORM_CHUNK,
  GROUP_NORM_PARTIAL_KEY,
  GROUP_NORM_PARTIAL_WGSL,
  GROUP_NORM_WORKGROUP_SIZE,
  groupNormApplyParams,
  groupNormLayout,
  groupNormPartialParams,
} from "../../src/kernels/group-norm.ts";
import {
  RMS_NORM_128_KEY,
  RMS_NORM_128_WGSL,
  RMS_NORM_KEY,
  RMS_NORM_WGSL,
  rmsNormParams,
  rmsNormWorkgroupKey,
  rmsNormWorkgroupWgsl,
} from "../../src/kernels/rms-norm.ts";
import {
  MASKED_FILL_KEY,
  MASKED_FILL_WGSL,
  MASKED_FILL_WORKGROUP_SIZE,
  maskedFillParams,
} from "../../src/kernels/masked-fill.ts";
import { PAD_KEY, PAD_WGSL, PAD_WORKGROUP_SIZE, padParams } from "../../src/kernels/pad.ts";
import {
  UPSAMPLE_BILINEAR2D_KEY,
  UPSAMPLE_BILINEAR2D_WGSL,
  UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE,
  upsampleBilinear2dParams,
} from "../../src/kernels/upsample-bilinear2d.ts";
import { gruScanKey, gruScanParams, gruScanWgsl } from "../../src/kernels/gru-scan.ts";
import {
  DEFORM_CONV2D_KEY,
  DEFORM_CONV2D_WGSL,
  DEFORM_CONV2D_WORKGROUP_SIZE,
  deformConv2dParams,
} from "../../src/kernels/deform-conv2d.ts";
import { ADALN_NORM_KEY, ADALN_NORM_WGSL, adalnNormParams } from "../../src/kernels/adaln-norm.ts";
import { SILU_WORKGROUP_SIZE, siluKey, siluParams, siluWgsl } from "../../src/kernels/silu.ts";
import {
  upsample2xParams,
  UPSAMPLE_2X_KEY,
  UPSAMPLE_2X_WGSL,
  UPSAMPLE_2X_WORKGROUP_SIZE,
} from "../../src/kernels/upsample2x.ts";
import { SOFTMAX_KEY, SOFTMAX_WGSL, softmaxParams } from "../../src/kernels/softmax.ts";
import {
  QUANTIZE_ROWS_KEY,
  QUANTIZE_ROWS_WGSL,
  quantizeRowsChannelMajorKey,
  quantizeRowsChannelMajorWgsl,
  quantizeRowsGeometry,
  quantizeRowsKey,
  quantizeRowsParams,
  quantizeRowsWgsl,
} from "../../src/kernels/quantize-rows.ts";
import { CONV3D_I8A8_QUANTIZE_GEOMETRY } from "../../src/kernels/conv3d-i8a8.ts";
import { quantizeRowsReference } from "../../src/reference/i8a8.ts";
import {
  ATTENTION_STATS_KEY,
  ATTENTION_STATS_STRIDE,
  ATTENTION_STATS_WGSL,
  attentionStatsKey,
  attentionStatsParams,
  attentionStatsWgsl,
} from "../../src/kernels/attention.ts";
import {
  applyReferenceOp,
  applyReferenceOpOutputs,
  type RefTensor,
  refTensor,
} from "../../src/reference/ops.ts";
import { alignF16Payload, roundToF16 } from "../../src/format/f16.ts";
import { alignI8Payload } from "../../src/format/i8.ts";
import type { WeightStorage } from "../../src/kernels/weight-storage.ts";
import { quantizeBf16 } from "./bf16.ts";
import { f32ToF16Bits, quantizeF16 } from "./f16.ts";
import { quantizeI8 } from "./i8.ts";
import { fill, type FilledTensor } from "./model-fixture.ts";

/** Deterministic data (no random numbers — a failure would not reproduce). */
const SIGNED = (i: number): number => ((i % 13) - 6) * 0.75;
const POSITIVE = (i: number): number => 0.125 + (i % 17) * 0.5;

/**
 * The reduced-dispatch case of one kernel family. Bindings are `0 = params` /
 * `1..n = inputs (in order)` / last = output, a layout shared by every kernel family.
 */
export type DegenerateCase = {
  readonly name: string;
  readonly key: string;
  readonly wgsl: string;
  readonly params: Uint32Array<ArrayBuffer>;
  /** Kernels that take params as a uniform (families with a workgroupBarrier in the row loop). */
  readonly uniformParams: boolean;
  /**
   * Contents written to the input buffers. A `Uint8Array` is **raw bytes** (weights stored as f16 /
   * i8 pack 2 / 4 elements into one word and cannot be expressed with the host's numeric array
   * types — ADR 0018 / 0019).
   */
  readonly inputs: readonly (FilledTensor | Uint8Array<ArrayBuffer>)[];
  /**
   * Byte count of **one more write target** that is not compared (bound after the inputs, one
   * before the compared output). It is for families where one dispatch writes two outputs, like
   * `quantize_rows`; its contents cannot be expressed in f32 (4 packed i8), so it is not compared —
   * robustness to a reduced dispatch is carried by the row loop, so comparing the `xs` side, which
   * writes one word per row, exercises the same path.
   *
   * `topk` (ADR 0068 decision 3) is also a family where one dispatch writes two outputs; here the
   * **values** go in this slot and the **indices** are compared (the values could be expressed in
   * f32 too, but the indices detect reduced-dispatch bugs better — the same loop writes both, so a
   * row loop that stops after one pass always shows in the indices, and exact i32 matching is
   * stricter than allclose).
   */
  readonly sideOutputBytes?: number;
  /**
   * Per-channel scale of the i8 variants (ADR 0019). Its binding is **the number after the
   * output**, so it is added after the output (the same position as `*_SCALE_BINDING` on the kernel
   * side).
   */
  readonly weightScale?: Float32Array<ArrayBuffer>;
  /**
   * The expected output from the CPU reference. The read-back view is created **with this dtype**
   * (the f32 families and the i32 families that write indices, such as argmax — storage forms that
   * cannot be compared go into {@link sideOutputBytes}).
   */
  readonly expected: RefTensor;
  /** Workgroups needed with a plain division (pinned as evidence that the dispatch is reduced). */
  readonly natural: number;
  /** The deliberately reduced dispatch count. */
  readonly groups: number;
};

/** Elementwise families (1-D grid-stride over `gid.x`). */
export const elementwiseCase = (): DegenerateCase => {
  const count = 100_000;
  const spec = { op: "neg", rank: 1, dtype: "f32" } as const;
  const input = fill([count], SIGNED);
  return {
    name: "elementwise neg",
    key: elementwiseKey(spec),
    wgsl: elementwiseWgsl(spec),
    params: elementwiseParams(spec, [count], [[count]]),
    uniformParams: false,
    inputs: [input],
    expected: applyReferenceOp("neg", [input], {}, [count]),
    natural: Math.ceil(count / ELEMENTWISE_WORKGROUP_SIZE),
    groups: 3,
  };
};

export const stridedCase = (): DegenerateCase => {
  const srcShape = [400, 250];
  const outShape = [250, 400];
  const dims = [1, 0];
  const input = fill(srcShape, SIGNED);
  return {
    name: "strided copy (permute [1,0])",
    key: stridedKey({ dtype: "f32" }),
    wgsl: stridedWgsl({ dtype: "f32" }),
    params: stridedParams(outShape, permuteSrcStrides(srcShape, dims), 0),
    uniformParams: false,
    inputs: [input],
    expected: applyReferenceOp("permute", [input], { dims }, outShape),
    natural: Math.ceil(100_000 / STRIDED_WORKGROUP_SIZE),
    groups: 3,
  };
};

export const gatherCase = (): DegenerateCase => {
  const rows = 20_000;
  const srcCols = 9;
  const cols = 6;
  const src = fill([rows, srcCols], SIGNED);
  // The indices cycle through 0..srcCols-1 (neither identity nor monotonic — an off-by-one always
  // shows in the values).
  const index = fill([rows, cols], (i) => (i * 5) % srcCols, "i32");
  return {
    name: "gather",
    key: GATHER_KEY,
    wgsl: GATHER_WGSL,
    params: gatherParams(rows * cols, cols, srcCols),
    uniformParams: true,
    inputs: [src, index],
    expected: applyReferenceOp("gather", [src, index], {}, [rows, cols]),
    natural: Math.ceil((rows * cols) / GATHER_WORKGROUP_SIZE),
    groups: 3,
  };
};

/**
 * Small tool for putting the weight-storage variants (ADR 0018 / 0019) on the reduced-dispatch
 * harness. **The f16 / i8 variants have a different WGSL body = a separate detection target**, so
 * each has a reduced-dispatch case of the same shape as the f32 version (the MUST at the top of
 * this file). The CPU reference's weights are **the rounded values** (fake-quant — ADR 0006).
 */
type VariantWeight = {
  /** Contents put into the input buffer (f32 as is / f16 and i8 as aligned raw bytes). */
  readonly payload: FilledTensor | Uint8Array<ArrayBuffer>;
  /** Weights the CPU reference uses (rounded for low-precision storage). */
  readonly reference: RefTensor;
  /** Per-channel scale of the i8 variant (undefined for the other variants). */
  readonly scale?: Float32Array<ArrayBuffer>;
};

/**
 * `channelAxis` is the axis the per-channel scale applies to (1 only for conv_transpose1d —
 * ADR 0019). Unused for f32 / f16.
 */
const variantWeight = (
  weight: FilledTensor,
  storage: WeightStorage,
  channelAxis: number,
): VariantWeight => {
  if (storage === "f32") return { payload: weight, reference: weight };
  if (storage === "f16") {
    const quantized = quantizeF16(weight.data);
    return {
      payload: alignF16Payload(quantized.bytes),
      reference: refTensor(weight.shape, quantized.values),
    };
  }
  if (storage === "bf16") {
    // bf16 is also 2 bytes / element, so it shares the alignment padding with f16 (ADR 0131).
    const quantized = quantizeBf16(weight.data);
    return {
      payload: alignF16Payload(quantized.bytes),
      reference: refTensor(weight.shape, quantized.values),
    };
  }
  const quantized = quantizeI8(weight.data, weight.shape, channelAxis);
  return {
    payload: alignI8Payload(quantized.bytes),
    reference: refTensor(weight.shape, quantized.values),
    scale: quantized.scale,
  };
};

export const embeddingCase = (storage: WeightStorage): DegenerateCase => {
  const vocab = 64;
  // Odd row length (in the f16 variant row starts alternate between the high and low half of a
  // word — a detector for swapped pairs)
  const hidden = 5;
  const rows = 20_000;
  const weight = variantWeight(fill([vocab, hidden], SIGNED), storage, 0);
  const index = fill([rows], (i) => (i * 7) % vocab, "i32");
  return {
    name: `embedding (w=${storage})`,
    key: embeddingKey(storage),
    wgsl: embeddingWgsl(storage),
    params: embeddingParams(rows * hidden, hidden, vocab),
    uniformParams: true,
    inputs: [weight.payload, index],
    weightScale: weight.scale,
    // padding_idx has no effect on forward but is a required attr of the contract
    // (src/ops/attrs.ts) — pass the unset sentinel −1
    expected: applyReferenceOp("embedding", [weight.reference, index], { padding_idx: -1 }, [
      rows,
      hidden,
    ]),
    natural: Math.ceil((rows * hidden) / EMBEDDING_WORKGROUP_SIZE),
    groups: 3,
  };
};

export const maskedFillCase = (): DegenerateCase => {
  const rows = 25_000;
  const cols = 4;
  const outShape = [rows, cols];
  const value = -3.4028234663852886e38;
  const x = fill(outShape, SIGNED);
  // The mask replicates the last dimension with stride 0 (the same shape as the right-aligned
  // broadcast seen in practice).
  const mask = fill([rows, 1], (i) => (i % 3 === 0 ? 1 : 0), "bool");
  return {
    name: "masked_fill",
    key: MASKED_FILL_KEY,
    wgsl: MASKED_FILL_WGSL,
    params: maskedFillParams(outShape, expandSrcStrides(mask.shape, outShape), value),
    uniformParams: false,
    inputs: [x, mask],
    expected: applyReferenceOp("masked_fill", [x, mask], { value }, outShape),
    natural: Math.ceil((rows * cols) / MASKED_FILL_WORKGROUP_SIZE),
    groups: 3,
  };
};

export const conv1dCase = (storage: WeightStorage): DegenerateCase => {
  const batch = 1;
  const channelsIn = 2;
  const channelsOut = 2;
  const lengthIn = 25_000;
  // Odd K (in the f16 variant weight-row starts alternate between the high and low half of a word)
  const kernel = 3;
  const stride = 1;
  const padding = 1;
  const dilation = 1;
  const groups = 1;
  const lengthOut = lengthIn;
  const x = fill([batch, channelsIn, lengthIn], SIGNED);
  const weight = variantWeight(fill([channelsOut, channelsIn, kernel], POSITIVE), storage, 0);
  const bias = fill([channelsOut], SIGNED);
  return {
    name: `conv1d (w=${storage})`,
    key: conv1dKey(storage),
    wgsl: conv1dWgsl(storage),
    params: conv1dParams({
      batch,
      channelsIn,
      channelsOut,
      lengthIn,
      lengthOut,
      kernel,
      stride,
      padding,
      dilation,
      groups,
    }),
    uniformParams: true,
    inputs: [x, weight.payload, bias],
    weightScale: weight.scale,
    expected: applyReferenceOp(
      "conv1d",
      [x, weight.reference, bias],
      { stride, padding, dilation, groups },
      [batch, channelsOut, lengthOut],
    ),
    natural: Math.ceil((batch * channelsOut * lengthOut) / CONV1D_WORKGROUP_SIZE),
    groups: 2,
  };
};

export const convTranspose1dCase = (storage: WeightStorage): DegenerateCase => {
  const batch = 1;
  // Asymmetric channels (reading the weight [Cin,Cout,K] as [Cout,Cin,K] turns red here too)
  const channelsIn = 3;
  const channelsOut = 2;
  const lengthIn = 12_000;
  // K is bound by the contract (2·padding == K − stride) and is even here (the pair selection of
  // odd lengths is covered by tests/gpu_f16_i8_weights_test.ts, which can use stride 1)
  const kernel = 4;
  const stride = 2;
  const padding = 1;
  const lengthOut = lengthIn * stride;
  const x = fill([batch, channelsIn, lengthIn], SIGNED);
  // MUST: the channel axis is 1 (the transposed layout of the weight [Cin, Cout, K] — ADR 0019)
  const weight = variantWeight(fill([channelsIn, channelsOut, kernel], POSITIVE), storage, 1);
  const bias = fill([channelsOut], SIGNED);
  return {
    name: `conv_transpose1d (w=${storage})`,
    key: convTranspose1dKey(storage),
    wgsl: convTranspose1dWgsl(storage),
    params: convTranspose1dParams({
      batch,
      channelsIn,
      channelsOut,
      lengthIn,
      lengthOut,
      kernel,
      stride,
      padding,
    }),
    uniformParams: true,
    inputs: [x, weight.payload, bias],
    weightScale: weight.scale,
    expected: applyReferenceOp(
      "conv_transpose1d",
      [x, weight.reference, bias],
      {
        stride,
        padding,
      },
      [batch, channelsOut, lengthOut],
    ),
    natural: Math.ceil((batch * channelsOut * lengthOut) / CONV_TRANSPOSE1D_WORKGROUP_SIZE),
    groups: 2,
  };
};

export const conv2dCase = (storage: WeightStorage): DegenerateCase => {
  const batch = 1;
  // Cin ≠ Cout, Kh ≠ Kw (a shape where swapped weight axes turn red on the reduced path too)
  const channelsIn = 2;
  const channelsOut = 3;
  const heightIn = 4;
  const widthIn = 8_000;
  // NOTE: Kh·Kw is even, so weight rows always start on a word boundary. Exercising the f16 pair
  // selection (odd lengths) is the job of tests/gpu_f16_i8_weights_test.ts; this only checks
  // robustness to a reduced dispatch.
  const kernelH = 3;
  const kernelW = 2;
  const paddingH = 1;
  const paddingW = 0;
  const heightOut = heightIn;
  const widthOut = widthIn - (kernelW - 1);
  const x = fill([batch, channelsIn, heightIn, widthIn], SIGNED);
  const weight = variantWeight(
    fill([channelsOut, channelsIn, kernelH, kernelW], POSITIVE),
    storage,
    0,
  );
  const bias = fill([channelsOut], SIGNED);
  return {
    name: `conv2d (w=${storage})`,
    key: conv2dKey(storage),
    wgsl: conv2dWgsl(storage),
    params: conv2dParams({
      batch,
      channelsIn,
      channelsOut,
      heightIn,
      widthIn,
      heightOut,
      widthOut,
      kernelH,
      kernelW,
      strideH: 1,
      strideW: 1,
      paddingH,
      paddingW,
      dilationH: 1,
      dilationW: 1,
      groups: 1,
    }),
    uniformParams: true,
    inputs: [x, weight.payload, bias],
    weightScale: weight.scale,
    expected: applyReferenceOp(
      "conv2d",
      [x, weight.reference, bias],
      {
        stride: [1, 1],
        padding: [paddingH, paddingW],
        dilation: [1, 1],
        groups: 1,
      },
      [batch, channelsOut, heightOut, widthOut],
    ),
    natural: Math.ceil((batch * channelsOut * heightOut * widthOut) / CONV2D_WORKGROUP_SIZE),
    groups: 2,
  };
};

export const padCase = (): DegenerateCase => {
  const rows = 4;
  const lengthIn = 25_000;
  // Left and right get **different widths** (with symmetric widths, swapping left / right would not
  // show in the output).
  const left = 3;
  const right = 5;
  const x = fill([rows, lengthIn], SIGNED);
  const outShape = [rows, left + lengthIn + right];
  return {
    name: "pad",
    key: PAD_KEY,
    wgsl: PAD_WGSL,
    params: padParams(rows, lengthIn, left, right),
    uniformParams: true,
    inputs: [x],
    expected: applyReferenceOp("pad", [x], { left, right }, outShape),
    natural: Math.ceil((rows * outShape[1]) / PAD_WORKGROUP_SIZE),
    groups: 2,
  };
};

export const flipCase = (): DegenerateCase => {
  // The flipped axis has length 3 or more (flipping 2 cancels an off-by-one symmetrically and
  // detects nothing).
  const outer = 2;
  const length = 3;
  const inner = 20_000;
  const shape = [outer, length, inner];
  const x = fill(shape, SIGNED);
  return {
    name: "flip",
    key: FLIP_KEY,
    wgsl: FLIP_WGSL,
    params: flipParams(outer, length, inner),
    uniformParams: true,
    inputs: [x],
    expected: applyReferenceOp("flip", [x], { dim: 1 }, shape),
    natural: Math.ceil((outer * length * inner) / FLIP_WORKGROUP_SIZE),
    groups: 2,
  };
};

export const deformConv2dCase = (): DegenerateCase => {
  const batch = 1;
  // Cin ≠ Cout, Kh ≠ Kw (a shape where swapped weight axes and the y/x order of offset turn red on
  // the reduced path too)
  const channelsIn = 2;
  const channelsOut = 3;
  const heightIn = 4;
  const widthIn = 6_000;
  const kernelH = 3;
  const kernelW = 2;
  const paddingH = 1;
  const paddingW = 0;
  const heightOut = heightIn;
  const widthOut = widthIn - (kernelW - 1);
  const taps = kernelH * kernelW;
  const x = fill([batch, channelsIn, heightIn, widthIn], SIGNED);
  const weight = fill([channelsOut, channelsIn, kernelH, kernelW], POSITIVE);
  // offset is the non-integer ±1.5 (crossing in and out of the border); mask is [0,2] as in
  // BiRefNet
  const offset = fill([batch, 2 * taps, heightOut, widthOut], (i) => ((i % 7) - 3) * 0.5);
  const mask = fill([batch, taps, heightOut, widthOut], (i) => (i % 5) * 0.5);
  const bias = fill([channelsOut], SIGNED);
  return {
    name: "deform_conv2d",
    key: DEFORM_CONV2D_KEY,
    wgsl: DEFORM_CONV2D_WGSL,
    params: deformConv2dParams({
      batch,
      channelsIn,
      channelsOut,
      heightIn,
      widthIn,
      heightOut,
      widthOut,
      kernelH,
      kernelW,
      paddingH,
      paddingW,
    }),
    uniformParams: true,
    inputs: [x, weight, offset, mask, bias],
    expected: applyReferenceOp(
      "deform_conv2d",
      [x, weight, offset, mask, bias],
      { padding: [paddingH, paddingW] },
      [batch, channelsOut, heightOut, widthOut],
    ),
    natural: Math.ceil((batch * channelsOut * heightOut * widthOut) / DEFORM_CONV2D_WORKGROUP_SIZE),
    groups: 2,
  };
};

export const upsampleBilinear2dCase = (): DegenerateCase => {
  // The ratio of a real model's shape (BiRefNet's main decoder path is 512² → 2048²) with the
  // element count shrunk. The factors are non-integers (H 39/399, W 41/419), so the reduced path
  // exercises a shape whose weights differ per output position.
  const [heightIn, widthIn] = [40, 42];
  const [heightOut, widthOut] = [400, 420];
  const shape = [1, 1, heightIn, widthIn];
  const outShape = [1, 1, heightOut, widthOut];
  const x = fill(shape, SIGNED);
  return {
    name: "upsample_bilinear2d",
    key: UPSAMPLE_BILINEAR2D_KEY,
    wgsl: UPSAMPLE_BILINEAR2D_WGSL,
    params: upsampleBilinear2dParams(heightOut * widthOut, heightIn, widthIn, heightOut, widthOut),
    uniformParams: true,
    inputs: [x],
    expected: applyReferenceOp(
      "upsample_bilinear2d",
      [x],
      { output_size: [heightOut, widthOut] },
      outShape,
    ),
    natural: Math.ceil((heightOut * widthOut) / UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE),
    groups: 2,
  };
};

/**
 * The execution kernel of cat (the strided **write** family). The real shape of cat, one dispatch =
 * a subregion of the output, leaves the rest unwritten and cannot be compared, so the parameters
 * are given as **a permutation covering the whole output** (the dual of permute [1,0] of the read
 * family). The grid-stride bound `n` of the write family is determined by params[0] (= the input
 * element count) alone, independent of the stride values, so this shape exercises robustness to a
 * reduced dispatch equivalently.
 */
export const stridedWriteCase = (): DegenerateCase => {
  const srcShape = [400, 250];
  const outShape = [250, 400];
  const input = fill(srcShape, SIGNED);
  // Input coordinate (a, b) → output index b·400 + a. Axis 0 advances by 1, axis 1 by 400.
  const outStrides = [1, srcShape[0]];
  return {
    name: "strided write (the write family of cat)",
    key: stridedWriteKey({ dtype: "f32" }),
    wgsl: stridedWriteWgsl({ dtype: "f32" }),
    params: stridedWriteParams(srcShape, outStrides, 0),
    uniformParams: false,
    inputs: [input],
    expected: applyReferenceOp("permute", [input], { dims: [1, 0] }, outShape),
    natural: Math.ceil(100_000 / STRIDED_WRITE_WORKGROUP_SIZE),
    groups: 2,
  };
};

/** Grid-stride over `gid.x` in units of rows (1 invocation = a sequential scan of 1 row). */
export const cumsumCase = (): DegenerateCase => {
  const rows = 20_000;
  const dim = 6;
  // SIGNED values are integer multiples of 0.75, so partial sums of up to 6 terms are exact in f32
  // (no rounding contribution mixed in).
  const input = fill([rows, dim], SIGNED);
  return {
    name: "cumsum",
    key: CUMSUM_KEY,
    wgsl: CUMSUM_WGSL,
    params: cumsumParams(rows, dim),
    uniformParams: true,
    inputs: [input],
    expected: applyReferenceOp("cumsum", [input], { dim: 1 }, [rows, dim]),
    // Each row is a sequential scan by one thread, so the number needed is not the row count but
    // ceil(rows / workgroup size).
    natural: Math.ceil(rows / CUMSUM_WORKGROUP_SIZE),
    groups: 2,
  };
};

/** Row families (a row loop advancing `workgroup_id.x` by `num_workgroups.x`). */
export const reduceCase = (): DegenerateCase => {
  const rows = 5_000;
  const dim = 3;
  const input = fill([rows, dim], POSITIVE);
  return {
    name: "row reduce sum",
    key: reduceKey({ op: "sum", dtype: "f32" }),
    wgsl: reduceWgsl({ op: "sum", dtype: "f32" }),
    params: reduceParams(rows, dim),
    uniformParams: true,
    inputs: [input],
    expected: applyReferenceOp("sum", [input], { dim: 1 }, [rows]),
    // 1 row = 1 workgroup (the row is folded inside the workgroup), so the number needed is the
    // row count itself.
    natural: rows,
    groups: 2,
  };
};

/**
 * argmax (ADR 0068 decision 2). The same row grid-stride as row reduce but **a separate family**
 * (identity −inf, carries (value, index) pairs), so it is listed on its own. The expected values
 * are i32 indices.
 *
 * MUST: the position of the maximum changes from row to row — if the same index were correct for
 * every row, neither "reads another row" nor "the row loop stops after one pass" would show in the
 * values (only unwritten poison would detect anything, and on a run that got a zero-filled buffer
 * some rows would be correct by accident with index 0).
 */
export const argmaxCase = (): DegenerateCase => {
  const rows = 5_000;
  const dim = 3;
  // The maximum of row r is at r % 3 (it cycles because POSITIVE's period 17 and dim 3 are
  // coprime).
  const input = fill([rows, dim], (i) => (i % dim === Math.floor(i / dim) % dim ? 9 : POSITIVE(i)));
  return {
    name: "argmax",
    key: ARGMAX_KEY,
    wgsl: ARGMAX_WGSL,
    params: argmaxParams(rows, dim),
    uniformParams: true,
    inputs: [input],
    expected: applyReferenceOp("argmax", [input], {}, [rows, 1]),
    // 1 row = 1 workgroup (the row is folded inside the workgroup), so the number needed is the
    // row count itself.
    natural: rows,
    groups: 2,
  };
};

/**
 * topk (ADR 0068 decision 3). The same row grid-stride as argmax but **a separate family**
 * (lane-local top-k → a k-round tournament merge, two outputs), so it is listed on its own.
 *
 * **The indices** are compared (exact i32 match), and the values go into
 * {@link DegenerateCase.sideOutputBytes} (bindings: values = after the inputs, indices = the next
 * one, matching the kernel's declaration).
 * MUST: the positions of the top entries change from row to row — if the same indices were correct
 * for every row, neither "reads another row" nor "the row loop stops after one pass" would show in
 * the values.
 */
export const topkCase = (): DegenerateCase => {
  const rows = 5_000;
  const dim = 5;
  const k = 2;
  // The maximum of row r is in column (r % 5) and the second in column ((r+1) % 5) (POSITIVE's
  // period 17 and dim 5 are coprime, so the raw values rotate per row too).
  const input = fill([rows, dim], (i) => {
    const row = Math.floor(i / dim);
    const col = i % dim;
    if (col === row % dim) return 90;
    if (col === (row + 1) % dim) return 80;
    return POSITIVE(i);
  });
  const outputs = applyReferenceOpOutputs("topk", [input], { k });
  return {
    name: "topk",
    key: topkKey(k),
    wgsl: topkWgsl(k),
    params: topkParams(rows, dim),
    uniformParams: true,
    inputs: [input],
    // The values (the side not compared) — the same size as the indices (both rows·k words of 4
    // bytes).
    sideOutputBytes: rows * k * 4,
    expected: outputs[1],
    // 1 row = 1 workgroup (a row is folded by 32 lanes), so the number needed is the row count
    // itself.
    natural: rows,
    groups: 2,
  };
};

/**
 * The axis reduce variant (any dimension but the last). Its grid-stride loop walks blocks of
 * `256 / split` outputs (one block = one workgroup), so the natural count is the output count ÷ the
 * outputs per group.
 */
export const axisReduceCase = (): DegenerateCase => {
  const shape = [40, 3, 500];
  const axis = 1;
  const input = fill(shape, POSITIVE);
  return {
    name: "axis reduce sum",
    key: axisReduceKey({ op: "sum", dtype: "f32" }, AXIS_REDUCE_SPLIT),
    wgsl: axisReduceWgsl({ op: "sum", dtype: "f32" }, AXIS_REDUCE_SPLIT),
    // out_count = 40·500 = 20,000 / axis_len = 3 / inner = 500
    params: axisReduceParams(20_000, shape[axis], 500),
    uniformParams: true,
    inputs: [input],
    expected: applyReferenceOp("sum", [input], { dim: axis }, [40, 500]),
    natural: Math.ceil(20_000 / axisReduceOutputsPerGroup(AXIS_REDUCE_SPLIT)),
    groups: 2,
  };
};

export const softmaxCase = (): DegenerateCase => {
  const rows = 5_000;
  const dim = 4;
  // A different distribution per row (with the same one on every row, "reads another row" would
  // not show in the values).
  const input = fill([rows, dim], (i) => ((i * 3) % 13) * 0.25 - 1.5);
  return {
    name: "softmax",
    key: SOFTMAX_KEY,
    wgsl: SOFTMAX_WGSL,
    params: softmaxParams(rows, dim),
    uniformParams: true,
    inputs: [input],
    expected: applyReferenceOp("softmax", [input], { dim: 1 }, [rows, dim]),
    natural: rows,
    groups: 2,
  };
};

/**
 * ② of fused attention (row statistics — ADR 0023). A row grid-stride family whose expected values
 * are "`amax` and `1/Σexp(x − amax)` per row".
 *
 * MUST: the expected values are not built with the same formula as the GPU. `applyReferenceOp`
 * cannot be used here (row statistics are not a single op), so `inv` is **derived back from the
 * distribution the safe-softmax CPU reference produces** — a broken scan order or reduction stage
 * fails allclose.
 */
export const attentionStatsCase = (): DegenerateCase => {
  const rows = 5_000;
  const dim = 4;
  // A different distribution per row (with the same one on every row, "reads another row" would
  // not show in the values).
  const input = fill([rows, dim], (i) => ((i * 3) % 13) * 0.25 - 1.5);
  const probs = applyReferenceOp("softmax", [input], { dim: 1 }, [rows, dim]);
  const expected = new Float32Array(rows * ATTENTION_STATS_STRIDE);
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let amax = input.data[base];
    for (let i = 1; i < dim; i += 1) amax = Math.max(amax, input.data[base + i]);
    // `p[j] = exp(x[j] − amax) · inv`, so the probability of the maximum element (exp(0) = 1) is
    // inv
    // itself.
    let argmax = 0;
    for (let i = 1; i < dim; i += 1) {
      if (input.data[base + i] > input.data[base + argmax]) argmax = i;
    }
    expected[row * ATTENTION_STATS_STRIDE] = amax;
    expected[row * ATTENTION_STATS_STRIDE + 1] = probs.data[base + argmax];
  }
  return {
    name: "attention_stats",
    key: ATTENTION_STATS_KEY,
    wgsl: ATTENTION_STATS_WGSL,
    params: attentionStatsParams(rows, dim),
    uniformParams: true,
    inputs: [input],
    expected: refTensor([rows, ATTENTION_STATS_STRIDE], expected),
    natural: rows,
    groups: 2,
  };
};

/**
 * **The f16 compute variant** of fused attention ② (reads S as f16 — ADR 0028). It adds one more
 * row grid-stride family, so per the families' MUST it is listed here too.
 *
 * MUST: the expected values are built from the values **after** rounding to f16 (the `values` of
 * `quantizeF16` are what the production `decodeF16` returned). Building them from the values before
 * rounding would eat into the allclose margin by the rounding error, turning "detecting a reduced
 * dispatch" into "detecting rounding".
 */
export const attentionStatsF16Case = (): DegenerateCase => {
  const rows = 5_000;
  const dim = 4;
  const raw = fill([rows, dim], (i) => ((i * 3) % 13) * 0.25 - 1.5);
  const quantized = quantizeF16(raw.data);
  const input = refTensor([rows, dim], quantized.values);
  const probs = applyReferenceOp("softmax", [input], { dim: 1 }, [rows, dim]);
  const expected = new Float32Array(rows * ATTENTION_STATS_STRIDE);
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let amax = input.data[base];
    for (let i = 1; i < dim; i += 1) amax = Math.max(amax, input.data[base + i]);
    let argmax = 0;
    for (let i = 1; i < dim; i += 1) {
      if (input.data[base + i] > input.data[base + argmax]) argmax = i;
    }
    expected[row * ATTENTION_STATS_STRIDE] = amax;
    expected[row * ATTENTION_STATS_STRIDE + 1] = probs.data[base + argmax];
  }
  return {
    name: "attention_stats c16",
    key: attentionStatsKey("f16"),
    wgsl: attentionStatsWgsl("f16"),
    params: attentionStatsParams(rows, dim),
    uniformParams: true,
    inputs: [alignF16Payload(quantized.bytes)],
    expected: refTensor([rows, ATTENTION_STATS_STRIDE], expected),
    natural: rows,
    groups: 2,
  };
};

/**
 * **The S f16 storage variant** of fused attention ② (2 elements per word with `pack2x16float` —
 * option γ, step 1). It adds yet another row grid-stride family, so per the families' MUST it is
 * listed here too.
 *
 * Unlike the `:c16` version (above) it **does not need `shader-f16`** (core WGSL), so it goes into
 * the shared harness instead of a separate device — which is itself the execution-side evidence
 * that "option γ does not depend on a feature".
 * MUST: the expected values are built from the values **after** rounding to f16 (building them from
 * the values before rounding would turn "detecting a reduced dispatch" into "detecting rounding").
 */
export const attentionStatsS16Case = (): DegenerateCase => {
  const rows = 5_000;
  // MUST: dim is a multiple of 4 (the eligible s16 shape is N % 4 == 0) and not a multiple of the
  // workgroup width 256
  const dim = 4;
  const raw = fill([rows, dim], (i) => ((i * 3) % 13) * 0.25 - 1.5);
  const rounded = Float32Array.from(raw.data, roundToF16);
  const input = refTensor([rows, dim], rounded);
  const probs = applyReferenceOp("softmax", [input], { dim: 1 }, [rows, dim]);
  const expected = new Float32Array(rows * ATTENTION_STATS_STRIDE);
  for (let row = 0; row < rows; row += 1) {
    const base = row * dim;
    let amax = input.data[base];
    for (let i = 1; i < dim; i += 1) amax = Math.max(amax, input.data[base + i]);
    let argmax = 0;
    for (let i = 1; i < dim; i += 1) {
      if (input.data[base + i] > input.data[base + argmax]) argmax = i;
    }
    expected[row * ATTENTION_STATS_STRIDE] = amax;
    expected[row * ATTENTION_STATS_STRIDE + 1] = probs.data[base + argmax];
  }
  // Pack into the same storage form as the GPU (the low half holds the even index)
  const words = new Uint32Array(rounded.length / 2);
  for (let i = 0; i < words.length; i += 1) {
    words[i] =
      (f32ToF16Bits(rounded[2 * i]) & 0xffff) | ((f32ToF16Bits(rounded[2 * i + 1]) & 0xffff) << 16);
  }
  return {
    name: "attention_stats s16",
    key: attentionStatsKey("f32", "f16"),
    wgsl: attentionStatsWgsl("f32", "f16"),
    params: attentionStatsParams(rows, dim),
    uniformParams: true,
    inputs: [new Uint8Array(words.buffer)],
    expected: refTensor([rows, ATTENTION_STATS_STRIDE], expected),
    natural: rows,
    groups: 2,
  };
};

/**
 * Per-token quantization of activations (the stage before w8a8 — src/kernels/quantize-rows.ts). A
 * row grid-stride family whose compared output is the per-row scale `xs` (`xq` is 4 packed i8 and
 * cannot be compared as f32, so it is only allocated through {@link
 * DegenerateCase.sideOutputBytes}).
 *
 * MUST: the input magnitudes differ per row (with the same scale on every row, "reads another row"
 * would not show in the values).
 */
export const quantizeRowsCase = (): DegenerateCase => {
  const rows = 5_000;
  // dim is a multiple of 4 (required by packing 4 i8) and not a multiple of the workgroup width 256
  const dim = 20;
  const input = fill([rows, dim], (i) => SIGNED(i) * (1 + (Math.floor(i / dim) % 7) * 0.5));
  return {
    name: "quantize_rows",
    key: QUANTIZE_ROWS_KEY,
    wgsl: QUANTIZE_ROWS_WGSL,
    params: quantizeRowsParams(rows, dim),
    uniformParams: true,
    inputs: [input],
    sideOutputBytes: rows * dim,
    expected: refTensor([rows], quantizeRowsReference(input.data, rows, dim).scale),
    // 1 row = 1 workgroup (the row is folded inside the workgroup), so the number needed is the
    // row count itself
    natural: rows,
    groups: 2,
  };
};

/**
 * The small-D variant of `quantize_rows` (perf-ledger P-1 — D=128 is 8 rows × 32 lanes). The row
 * loop advances several rows per workgroup, so a reduced dispatch exercises both the path where
 * "the last rows of a group are missing" and the grid-stride.
 */
export const quantizeRowsGroupedCase = (): DegenerateCase => {
  const rows = 5_003;
  const dim = 128;
  const geometry = quantizeRowsGeometry(dim);
  const input = fill([rows, dim], (i) => SIGNED(i) * (1 + (Math.floor(i / dim) % 7) * 0.5));
  return {
    name: `quantize_rows r${geometry.rowsPerGroup}w${geometry.lanesPerRow}`,
    key: quantizeRowsKey(geometry),
    wgsl: quantizeRowsWgsl(geometry),
    params: quantizeRowsParams(rows, dim),
    uniformParams: true,
    inputs: [input],
    sideOutputBytes: rows * dim,
    expected: refTensor([rows], quantizeRowsReference(input.data, rows, dim).scale),
    natural: Math.ceil(rows / geometry.rowsPerGroup),
    groups: 2,
  };
};

/**
 * The channel-major form of `quantize_rows` (conv3d w8a8's activation quantiser — perf-ledger V-6)
 * at the production geometry. It walks blocks of pixels with a grid stride, and every pass of that
 * loop runs the word-block loop with its barriers, so a collapsed dispatch covers both. The input
 * is `x[C][P]`; the compared output is the per-pixel scale `xs` (`xq` is the side output).
 *
 * MUST: neighbouring pixels have different scales (with one scale for all pixels, reading another
 * pixel's channels would not show in the values). Every pixel reaches |SIGNED| = 4.5 over its 200
 * channels, so its scale is 4.5·(1 + (p % 7)·0.5) / 127: seven values, and a read of a pixel a
 * multiple of 7 away would not show in `xs`. C = 200 is 50 words per pixel; at the production
 * geometry's 16-word blocks that is four word blocks, the last partial (another geometry changes the
 * count — p8 has 32-word blocks, so two).
 */
export const quantizeRowsChannelMajorCase = (): DegenerateCase => {
  const pixels = 5_003;
  const channels = 200;
  const geometry = CONV3D_I8A8_QUANTIZE_GEOMETRY;
  const input = fill([channels, pixels], (i) => SIGNED(i) * (1 + ((i % pixels) % 7) * 0.5));
  const transposed = new Float32Array(pixels * channels);
  for (let c = 0; c < channels; c += 1) {
    for (let p = 0; p < pixels; p += 1) transposed[p * channels + c] = input.data[c * pixels + p];
  }
  return {
    name: quantizeRowsChannelMajorKey(geometry),
    key: quantizeRowsChannelMajorKey(geometry),
    wgsl: quantizeRowsChannelMajorWgsl(geometry),
    params: quantizeRowsParams(pixels, channels),
    uniformParams: true,
    inputs: [input],
    sideOutputBytes: pixels * channels,
    expected: refTensor([pixels], quantizeRowsReference(transposed, pixels, channels).scale),
    natural: Math.ceil(pixels / geometry.pixels),
    groups: 2,
  };
};

export const layerNormCase = (): DegenerateCase => {
  const rows = 5_000;
  const dim = 4;
  const eps = 1e-5;
  const input = fill([rows, dim], (i) => ((i * 5) % 17) * 0.5 - 4);
  const weight = fill([dim], POSITIVE);
  const bias = fill([dim], SIGNED);
  return {
    name: "layer_norm",
    key: LAYER_NORM_KEY,
    wgsl: LAYER_NORM_WGSL,
    params: layerNormParams(rows, dim, eps),
    uniformParams: true,
    inputs: [input, weight, bias],
    expected: applyReferenceOp(
      "layer_norm",
      [input, weight, bias],
      {
        normalized_shape: [dim],
        eps,
      },
      [rows, dim],
    ),
    natural: rows,
    groups: 2,
  };
};

/**
 * group_norm's partial stage (pass 1 — the squared deviations from a given mean), one dispatch of
 * the node's five: 4 rows × 25 chunks of 1,024. The total stage writes only every other statistic
 * per pass, so it is not a full-write kernel for this harness; its wrap is
 * `gpu_group_norm_test.ts`'s (a device with 3 workgroups per dimension, bit-identical).
 */
export const groupNormPartialCase = (): DegenerateCase => {
  const shape = [1, 8, 12_800];
  const layout = groupNormLayout(shape, 4);
  const x = fill(shape, (i) => ((i * 5) % 17) * 0.5 - 4 + (i % 3));
  const means = [0.25, -1.5, 2, 0.75];
  const stats = Float32Array.from(means.flatMap((mean) => [mean, 0]));
  const partials = new Float32Array(layout.rows * layout.chunks);
  for (let row = 0; row < layout.rows; row += 1) {
    for (let i = 0; i < layout.length; i += 1) {
      const deviation = x.data[row * layout.length + i] - stats[2 * row];
      partials[row * layout.chunks + Math.floor(i / GROUP_NORM_CHUNK)] += deviation * deviation;
    }
  }
  return {
    name: "group_norm partial (pass 1)",
    key: GROUP_NORM_PARTIAL_KEY,
    wgsl: GROUP_NORM_PARTIAL_WGSL,
    params: groupNormPartialParams(layout, 1),
    uniformParams: true,
    inputs: [x, { dtype: "f32", shape: [stats.length], data: stats }],
    expected: { dtype: "f32", shape: [partials.length], data: partials },
    natural: layout.rows * layout.chunks,
    groups: 2,
  };
};

/** group_norm's apply stage (the grid-stride over every element, given the row statistics). */
export const groupNormApplyCase = (): DegenerateCase => {
  const shape = [2, 6, 50, 40];
  const layout = groupNormLayout(shape, 3);
  const x = fill(shape, SIGNED);
  const stats = Float32Array.from({ length: layout.rows * 2 }, (_, i) =>
    i % 2 === 0 ? 0.25 * (i - 5) : 0.5 + 0.125 * i,
  );
  const weight = fill([6], POSITIVE);
  const bias = fill([6], SIGNED);
  const out = new Float32Array(layout.count);
  for (let e = 0; e < layout.count; e += 1) {
    const row = Math.floor(e / layout.length);
    const channel =
      (row % layout.groups) * layout.perGroup + Math.floor((e % layout.length) / layout.inner);
    out[e] =
      (x.data[e] - stats[2 * row]) * stats[2 * row + 1] * weight.data[channel] + bias.data[channel];
  }
  return {
    name: "group_norm apply",
    key: GROUP_NORM_APPLY_KEY,
    wgsl: GROUP_NORM_APPLY_WGSL,
    params: groupNormApplyParams(layout),
    uniformParams: true,
    inputs: [x, { dtype: "f32", shape: [stats.length], data: stats }, weight, bias],
    expected: { dtype: "f32", shape: [...shape], data: out },
    natural: Math.ceil(layout.count / GROUP_NORM_WORKGROUP_SIZE),
    groups: 2,
  };
};

export const gruScanCase = (): DegenerateCase => {
  // The same shape as the row families (1 workgroup = 1 batch element, advanced by
  // `num_workgroups.x`). The time loop is inside the kernel, so only the batch axis is reduced, and
  // the number needed is N itself.
  // MUST: pass a different `h0` per batch (with the same one for every batch, "scans another batch"
  // would not show in the values and the reduced-dispatch check would be tautological).
  const time = 3;
  const batch = 2_000;
  const hidden = 4;
  const gates = 3 * hidden;
  const gi = fill([time, batch, gates], (i) => ((i * 7) % 23) * 0.13 - 1.4);
  const initial = fill([batch, hidden], (i) => ((i * 5) % 19) * 0.09 - 0.8);
  const weight = fill([gates, hidden], (i) => ((i % 11) - 5) * 0.07);
  const bias = fill([gates], SIGNED);
  return {
    name: "gru_scan",
    key: gruScanKey("forward"),
    wgsl: gruScanWgsl("forward"),
    params: gruScanParams({ time, batch, hidden }),
    uniformParams: true,
    inputs: [gi, initial, weight, bias],
    expected: applyReferenceOp("gru_scan", [gi, initial, weight, bias], {}, [time, batch, hidden]),
    natural: batch,
    groups: 2,
  };
};

export const rmsNormCase = (narrow = false): DegenerateCase => {
  const rows = 5_000;
  const dim = 4;
  const eps = 1e-6;
  // A different sum of squares per row (with the same one on every row, "reads another row" would
  // not show in the values)
  const input = fill([rows, dim], (i) => ((i * 5) % 17) * 0.5 - 4);
  const weight = fill([dim], POSITIVE);
  return {
    name: narrow ? "rms_norm 128" : "rms_norm",
    key: narrow ? RMS_NORM_128_KEY : RMS_NORM_KEY,
    wgsl: narrow ? RMS_NORM_128_WGSL : RMS_NORM_WGSL,
    params: rmsNormParams(rows, dim, eps),
    uniformParams: true,
    inputs: [input, weight],
    expected: applyReferenceOp("rms_norm", [input, weight], { eps }, [rows, dim]),
    natural: rows,
    groups: 2,
  };
};

/**
 * The prefetch form of the 256-wide rms_norm (v2, perf-ledger K-99) at 2 trips per thread: the
 * weight words are loaded once per workgroup before the row loop, so a workgroup that strides over
 * many rows must keep using them.
 */
export const rmsNormPrefetchCase = (): DegenerateCase => {
  const rows = 600;
  const dim = 512;
  const eps = 1e-6;
  const input = fill([rows, dim], (i) => ((i * 5) % 17) * 0.5 - 4 + Math.floor(i / dim) * 0.01);
  const weight = fill([dim], POSITIVE);
  return {
    name: "rms_norm prefetch",
    key: rmsNormWorkgroupKey(dim),
    wgsl: rmsNormWorkgroupWgsl(dim),
    params: rmsNormParams(rows, dim, eps),
    uniformParams: true,
    inputs: [input, weight],
    expected: applyReferenceOp("rms_norm", [input, weight], { eps }, [rows, dim]),
    natural: rows,
    groups: 2,
  };
};

// ---------------------------------------------------------------------------
// Fused kernels (outside the op vocabulary; the 3 families src/runtime/fusion.ts inserts by
// rewriting)
//
// MUST: fused kernels claim grid-stride too, so they are listed here. Their own tests
// (gpu_upsample2x_fusion_test.ts / gpu_silu_fusion_test.ts / gpu_adaln_fusion_test.ts) use small
// shapes that get every workgroup they need, so a regression that makes `stride` a constant would
// pass green there.
// ---------------------------------------------------------------------------

/**
 * `upsample2x` (NCHW nearest ×2, fusion rule upsample2x). An elementwise family in which one
 * invocation writes one input element into a 2×2 block of the output. The expected values are
 * **built on the host** (it is not in the op vocabulary, so `applyReferenceOp` cannot be used).
 */
export const upsample2xCase = (): DegenerateCase => {
  const rows = 700;
  const width = 240;
  const count = rows * width;
  const outWidth = width * 2;
  const input = fill([rows, width], SIGNED);
  const expected = new Float32Array(count * 4);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < width; col += 1) {
      const value = input.data[row * width + col];
      const base = 2 * row * outWidth + 2 * col;
      expected[base] = value;
      expected[base + 1] = value;
      expected[base + outWidth] = value;
      expected[base + outWidth + 1] = value;
    }
  }
  return {
    name: "upsample2x (fused)",
    key: UPSAMPLE_2X_KEY,
    wgsl: UPSAMPLE_2X_WGSL,
    params: upsample2xParams(count, width),
    uniformParams: true,
    inputs: [input],
    expected: refTensor([rows * 2, outWidth], expected),
    natural: Math.ceil(count / UPSAMPLE_2X_WORKGROUP_SIZE),
    groups: 2,
  };
};

/**
 * `silu` (`x · sigmoid(x)`, fusion rule silu). A **block loop** family (it has a rounding barrier
 * through a workgroup-memory round trip, so it advances `wid.x` by `nwg.x` instead of `gid.x`). The
 * expected values are built in the same 2 steps as the unfused sequence (sigmoid → mul).
 */
export const siluCase = (): DegenerateCase => {
  const count = 100_000;
  const input = fill([count], SIGNED);
  const sigmoid = applyReferenceOp("sigmoid", [input], {}, [count]);
  const expected = Float32Array.from({ length: count }, (_, i) =>
    Math.fround(input.data[i] * sigmoid.data[i]),
  );
  return {
    name: "silu (fused)",
    key: siluKey("x-sigmoid"),
    wgsl: siluWgsl("x-sigmoid"),
    params: siluParams(count),
    uniformParams: true,
    inputs: [input],
    expected: refTensor([count], expected),
    natural: Math.ceil(count / SILU_WORKGROUP_SIZE),
    groups: 3,
  };
};

/**
 * `adaln_norm` (`layer_norm → (1 + scale) → mul → add`, fusion rule adaln). A row grid-stride
 * family bound with 6 inputs (x / weight / bias / scale_vec / one / shift) + the output.
 *
 * MUST: the statistics differ per row (with the same ones on every row, "reads another row" would
 * not show in the values).
 */
export const adalnNormCase = (): DegenerateCase => {
  const rows = 5_000;
  const dim = 4;
  const eps = 1e-5;
  const input = fill([rows, dim], (i) => ((i * 5) % 17) * 0.5 - 4);
  const weight = fill([dim], POSITIVE);
  const bias = fill([dim], SIGNED);
  const scaleVec = fill([dim], (i) => ((i % 7) - 3) * 0.25);
  const one = fill([1], () => 1);
  const shift = fill([dim], (i) => ((i % 5) - 2) * 0.375);
  const normalized = applyReferenceOp(
    "layer_norm",
    [input, weight, bias],
    {
      normalized_shape: [dim],
      eps,
    },
    [rows, dim],
  );
  const expected = new Float32Array(rows * dim);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < dim; col += 1) {
      const modulation = Math.fround(scaleVec.data[col] + one.data[0]);
      const scaled = Math.fround(normalized.data[row * dim + col] * modulation);
      expected[row * dim + col] = Math.fround(scaled + shift.data[col]);
    }
  }
  return {
    name: "adaln_norm (fused)",
    key: ADALN_NORM_KEY,
    wgsl: ADALN_NORM_WGSL,
    params: adalnNormParams(rows, dim, eps),
    uniformParams: true,
    inputs: [input, weight, bias, scaleVec, one, shift],
    expected: refTensor([rows, dim], expected),
    natural: rows,
    groups: 2,
  };
};
