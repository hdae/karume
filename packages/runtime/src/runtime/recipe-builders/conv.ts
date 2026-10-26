/**
 * Per-op-family derivation — the convolution family (conv1d / conv2d and their igemm variants,
 * conv3d, conv_transpose1d, deform_conv2d) and gru_scan.
 *
 * The entry is `RecipeBuilder` in {@link "../recipe-builder.ts"}; the shared services (Session
 * state, params writes, the weights' storage form and scale bindings) are reached only through
 * {@link RecipeBuildFace}.
 */

import type { BindingSource, StepRecipeBuilder } from "../recipe.ts";
import {
  CONV1D_SCALE_BINDING,
  CONV1D_WORKGROUP_SIZE,
  type Conv1dDims,
  conv1dIgemmKey,
  conv1dIgemmParams,
  conv1dIgemmWgsl,
  conv1dKey,
  conv1dParams,
  conv1dUsesVec4,
  conv1dWgsl,
} from "../../kernels/conv1d.ts";
import {
  CONV2D_SCALE_BINDING,
  CONV2D_WORKGROUP_SIZE,
  type Conv2dDims,
  conv2dIgemmKey,
  conv2dIgemmMTile,
  conv2dIgemmParams,
  conv2dIgemmWgsl,
  conv2dKey,
  conv2dParams,
  conv2dUsesVec4,
  conv2dWgsl,
} from "../../kernels/conv2d.ts";
import {
  CONV3D_SCALE_BINDING,
  type Conv3dDims,
  conv3dIgemmKey,
  conv3dIgemmParams,
  conv3dIgemmWgsl,
  conv3dIgemmWorkgroups,
  conv3dUsesVec4,
} from "../../kernels/conv3d.ts";
import {
  CONV3D_ACT_SCALE_BINDING,
  CONV3D_I8A8_QUANTIZE_GEOMETRY,
  conv3dI8a8Eligible,
  conv3dI8a8Geometry,
  conv3dI8a8Key,
  conv3dI8a8Params,
  conv3dI8a8UsesVec4,
  conv3dI8a8Wgsl,
  conv3dI8a8Workgroups,
} from "../../kernels/conv3d-i8a8.ts";
import {
  quantizeRowsChannelMajorKey,
  quantizeRowsChannelMajorWgsl,
  quantizeRowsChannelMajorWorkgroups,
  quantizeRowsParams,
} from "../../kernels/quantize-rows.ts";
import {
  CONV_TRANSPOSE1D_SCALE_BINDING,
  CONV_TRANSPOSE1D_WORKGROUP_SIZE,
  convTranspose1dKey,
  convTranspose1dParams,
  convTranspose1dWgsl,
} from "../../kernels/conv-transpose1d.ts";
import {
  DEFORM_CONV2D_KEY,
  DEFORM_CONV2D_WGSL,
  DEFORM_CONV2D_WORKGROUP_SIZE,
  deformConv2dParams,
} from "../../kernels/deform-conv2d.ts";
import { ExecutionError, type NodePlan } from "../plan.ts";
import type { WeightStorage } from "../../kernels/weight-storage.ts";
import {
  conv1dAttrs,
  conv2dAttrs,
  conv3dAttrs,
  convTranspose1dAttrs,
  deformConv2dAttrs,
  numel,
} from "../../ops.ts";
import { gemmMTileGeometry } from "../../kernels/gemm.ts";
import { gemmTileM, gemmTileN } from "../../kernels/gemm-geometry.ts";
import { conv2dProfileGeometry } from "../../kernels/geometry-profile.ts";
import { gridStrideWorkgroups, tiledWorkgroups } from "../../codegen/dispatch.ts";
import { gruScanKey, gruScanParams, gruScanWgsl } from "../../kernels/gru-scan.ts";
import type { RecipeBuildFace } from "../recipe-builder.ts";
import { PARAMS_UNIFORM_USAGE } from "./params-usage.ts";

/**
 * deform_conv2d (DCNv2 — ADR 0055). A single grid-stride kernel with one output element = one
 * invocation, no branching (groups / offset_groups / stride / dilation have no field in the
 * contract = fixed to 1).
 *
 * MUST: Kh / Kw are read from **the weight's third / fourth axes**, in that order (the same lesson
 * as conv2d). The contract check has already matched the offset / mask shapes against the output
 * space.
 */
export const buildDeformConv2d = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const [x, weight] = step.inputShapes;
  const outShape = step.outputs[0].shape;
  const { padding } = deformConv2dAttrs(step.node.attrs, `nodes (${step.node.op})`);
  const { pipeline, layout, roles } = await face.state.cache.get(
    DEFORM_CONV2D_KEY,
    DEFORM_CONV2D_WGSL,
  );
  const params = face.writeParams(
    deformConv2dParams({
      batch: outShape[0],
      channelsIn: x[1],
      channelsOut: outShape[1],
      heightIn: x[2],
      widthIn: x[3],
      heightOut: outShape[2],
      widthOut: outShape[3],
      kernelH: weight[2],
      kernelW: weight[3],
      paddingH: padding[0],
      paddingW: padding[1],
    }),
    PARAMS_UNIFORM_USAGE,
  );
  const workgroups = gridStrideWorkgroups(
    numel(outShape),
    DEFORM_CONV2D_WORKGROUP_SIZE,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
  );
  builder.dispatch({
    key: DEFORM_CONV2D_KEY,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      ...binds.map((source, index) => ({ binding: index + 1, source })),
      { binding: 6, source: outs[0] },
    ],
    workgroups: [workgroups, 1, 1],
  });
};

/**
 * gru_scan / gru_scan_reverse (the hidden-side scan of a GRU — ADR 0056).
 *
 * One workgroup = one batch element (only the batch direction crosses the dispatch limit by
 * grid-stride); the time loop is inside the kernel. The scan direction comes from **the op name**
 * (attrs have no field for it).
 *
 * MUST: the geometry comes from h0 and W_hh (T from the head of gi, N and H from h0). Dividing H
 * back out of gi's last dimension 3H would re-create, in one place, the mix-ups the contract check
 * is supposed to catch.
 */
export const buildGruScan = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const direction = step.node.op === "gru_scan_reverse" ? "reverse" : "forward";
  const key = gruScanKey(direction);
  const { pipeline, layout, roles } = await face.state.cache.get(key, gruScanWgsl(direction));
  const [time, batch, hidden] = step.outputs[0].shape;
  const params = face.writeParams(gruScanParams({ time, batch, hidden }), PARAMS_UNIFORM_USAGE);
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      ...binds.map((source, index) => ({ binding: index + 1, source })),
      { binding: 5, source: outs[0] },
    ],
    workgroups: [
      gridStrideWorkgroups(batch, 1, face.state.gpu.limits.maxComputeWorkgroupsPerDimension),
      1,
      1,
    ],
  });
};

/**
 * conv1d (groups / dilation from attrs). **Two kernels, chosen by groups** (same as conv2d):
 * `groups == 1` is the implicit GEMM, `groups > 1` a direct convolution (grid-stride with one
 * thread = one output element).
 */
export const buildConv1d = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const [x, weight] = step.inputShapes;
  const outShape = step.outputs[0].shape;
  const { stride, padding, dilation, groups } = conv1dAttrs(
    step.node.attrs,
    `nodes (${step.node.op})`,
  );
  const dims: Conv1dDims = {
    batch: outShape[0],
    channelsIn: x[1],
    channelsOut: outShape[1],
    lengthIn: x[2],
    lengthOut: outShape[2],
    kernel: weight[2],
    stride,
    padding,
    dilation,
    groups,
  };
  const weightStorage = face.weightStorage(step);
  if (groups === 1) {
    await buildConv1dIgemm(face, step, binds, outs, builder, dims, weightStorage);
    return;
  }
  const key = conv1dKey(weightStorage);
  const { pipeline, layout, roles } = await face.state.cache.get(key, conv1dWgsl(weightStorage));
  const params = face.writeParams(conv1dParams(dims), PARAMS_UNIFORM_USAGE);
  const workgroups = gridStrideWorkgroups(
    numel(outShape),
    CONV1D_WORKGROUP_SIZE,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
  );
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      ...binds.map((source, index) => ({ binding: index + 1, source })),
      { binding: 4, source: outs[0] },
      ...face.weightScaleBindings(step, CONV1D_SCALE_BINDING),
    ],
    workgroups: [workgroups, 1, 1],
  });
};

/**
 * conv1d's implicit GEMM (`C[Cout, N] = W[Cout, K] × Xcol[K, N]` — the 1D version of ADR 0024).
 *
 * MUST: the same "one workgroup = one output tile" as the GEMM skeleton, so exceeding the
 * dispatch limit fails loudly (clamping onto grid-stride would drop tiles and silently break the
 * full-write).
 * MUST: the batch is the **z axis** (as in bmm / conv2d). Folding it into N makes the output
 * `[Cout][B·Lout]`, swapping axes against NCL — it matches only for B = 1, so the measured forms do
 * not reveal it.
 *
 * The m-tile predicate is **the same single one** as conv2d ({@link conv2dIgemmMTile} — a function
 * of M = Cout only, so independent of dimensionality). Either tile shape gives bit-identical
 * output, so this is purely a redistribution of dispatches and does not touch the numeric contract.
 *
 * NOTE: an i4-resident conv1d lands **only on this path** (eligibility is closed to groups == 1 —
 * the conv1d extension of ADR 0069 decision 5, addendum 7). The group length is baked into the
 * WGSL, so it is also part of the key.
 */
const buildConv1dIgemm = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
  dims: Conv1dDims,
  weightStorage: WeightStorage,
): Promise<void> => {
  const m = dims.channelsOut;
  const kFlat = dims.channelsIn * dims.kernel;
  const v4 = conv1dUsesVec4(kFlat, dims.lengthOut, dims.stride);
  const mTile = conv2dIgemmMTile(m);
  // Take the geometry from the same resolution point as generation and key (the dispatch's tile
  // edges match the WGSL's edges structurally).
  const geometry = gemmMTileGeometry(mTile);
  // i4 bakes the group length into the WGSL (paired with the key's g part — the same discipline as
  // linear / embedding, ADR 0069).
  const groupSize = weightStorage === "i4" ? face.weightGroupSize(step) : undefined;
  const key = conv1dIgemmKey(weightStorage, v4, mTile, groupSize);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    conv1dIgemmWgsl(weightStorage, v4, mTile, groupSize),
  );
  const params = face.writeParams(conv1dIgemmParams(dims), PARAMS_UNIFORM_USAGE);
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const where = `conv1d [${step.inputShapes[0].join(",")}] * [${step.inputShapes[1].join(",")}]`;
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      ...binds.map((source, index) => ({ binding: index + 1, source })),
      { binding: 4, source: outs[0] },
      ...face.weightScaleBindings(step, CONV1D_SCALE_BINDING),
    ],
    workgroups: [
      tiledWorkgroups(dims.lengthOut, gemmTileN(geometry), limit, where),
      tiledWorkgroups(m, gemmTileM(geometry), limit, where),
      tiledWorkgroups(dims.batch, 1, limit, where),
    ],
  });
};

/**
 * conv2d (stride / padding / dilation have two components, H/W). **Two kernels, chosen by
 * groups** (ADR 0024): `groups == 1` is the implicit GEMM, `groups > 1` a direct convolution.
 *
 * MUST: Kh / Kw are read from **the weight's third / fourth axes**, in that order. Swapping them
 * still matches numerically for a square kernel, so the tests pin a Kh ≠ Kw shape
 * (src/kernels/conv2d.ts).
 */
export const buildConv2d = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const [x, weight] = step.inputShapes;
  const outShape = step.outputs[0].shape;
  const { stride, padding, dilation, groups } = conv2dAttrs(
    step.node.attrs,
    `nodes (${step.node.op})`,
  );
  const dims: Conv2dDims = {
    batch: outShape[0],
    channelsIn: x[1],
    channelsOut: outShape[1],
    heightIn: x[2],
    widthIn: x[3],
    heightOut: outShape[2],
    widthOut: outShape[3],
    kernelH: weight[2],
    kernelW: weight[3],
    strideH: stride[0],
    strideW: stride[1],
    paddingH: padding[0],
    paddingW: padding[1],
    dilationH: dilation[0],
    dilationW: dilation[1],
    groups,
  };
  const weightStorage = face.weightStorage(step);
  if (groups === 1) {
    await buildConv2dIgemm(face, step, binds, outs, builder, dims, weightStorage);
    return;
  }
  const key = conv2dKey(weightStorage);
  const { pipeline, layout, roles } = await face.state.cache.get(key, conv2dWgsl(weightStorage));
  const params = face.writeParams(conv2dParams(dims), PARAMS_UNIFORM_USAGE);
  const workgroups = gridStrideWorkgroups(
    numel(outShape),
    CONV2D_WORKGROUP_SIZE,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
  );
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      ...binds.map((source, index) => ({ binding: index + 1, source })),
      { binding: 4, source: outs[0] },
      ...face.weightScaleBindings(step, CONV2D_SCALE_BINDING),
    ],
    workgroups: [workgroups, 1, 1],
  });
};

/**
 * conv2d's implicit GEMM (`C[Cout, N] = W[Cout, K] × Xcol[K, N]` — ADR 0024).
 *
 * MUST: the same "one workgroup = one output tile" as the GEMM skeleton, so exceeding the
 * dispatch limit fails loudly (clamping onto grid-stride would drop tiles and silently break the
 * full-write).
 * MUST: the batch is the **z axis** (as in bmm). Folding it into N makes the output
 * `[Cout][B·Hout·Wout]`, swapping axes against NCHW — it matches only for B = 1, so the measured
 * forms do not reveal it.
 *
 * The m tile picks the 64-row / 32-row class as a function of the shape ({@link conv2dIgemmMTile})
 * (next to ADR 0024), and the geometry per class is a field of the adapter's profile
 * (src/kernels/geometry-profile.ts — the default profile keeps the earlier
 * `gemmMTileGeometry(mTile)` = 64×128 tiles for the 64-row class and 32×128 for the 32-row class).
 * Every geometry gives bit-identical output,
 * so this is purely a redistribution of dispatches and does not touch the numeric contract.
 */
const buildConv2dIgemm = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
  dims: Conv2dDims,
  weightStorage: WeightStorage,
): Promise<void> => {
  const m = dims.channelsOut;
  const n = dims.heightOut * dims.widthOut;
  const kFlat = dims.channelsIn * dims.kernelH * dims.kernelW;
  const v4 = conv2dUsesVec4(kFlat, dims.widthOut, dims.strideW);
  const mTile = conv2dIgemmMTile(m);
  // MUST: pass the geometry taken once **unchanged** to all three of key, WGSL and dispatch (the
  // dispatch's tile edges match the WGSL's edges structurally — the source of truth for the actual
  // tile edges is the geometry; mTile is only the class name).
  const geometry = conv2dProfileGeometry(face.state.geometryProfile, mTile);
  const key = conv2dIgemmKey(weightStorage, v4, mTile, geometry);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    conv2dIgemmWgsl(weightStorage, v4, mTile, geometry),
  );
  const params = face.writeParams(conv2dIgemmParams(dims), PARAMS_UNIFORM_USAGE);
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const where = `conv2d [${step.inputShapes[0].join(",")}] * [${step.inputShapes[1].join(",")}]`;
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      ...binds.map((source, index) => ({ binding: index + 1, source })),
      { binding: 4, source: outs[0] },
      ...face.weightScaleBindings(step, CONV2D_SCALE_BINDING),
    ],
    workgroups: [
      tiledWorkgroups(n, gemmTileN(geometry), limit, where),
      tiledWorkgroups(m, gemmTileM(geometry), limit, where),
      tiledWorkgroups(dims.batch, 1, limit, where),
    ],
  });
};

/**
 * conv3d (unbatched, implicit GEMM only — ADR 0118 decision 1).
 *
 * The time axis is folded into N, so the dispatch is `[ceil(N/tileN), ceil(M/tileM), 1]`
 * (unbatched — z is 1). The m-tile predicate is **the same single one** as conv1d / conv2d
 * ({@link conv2dIgemmMTile}), and the geometry is the default (`gemmMTileGeometry`) without a
 * profile lookup — the same treatment as conv1d (ADR 0118 decision 1).
 *
 * MUST: the same "one workgroup = one output tile" as the GEMM skeleton, so exceeding the
 * dispatch limit fails loudly (clamping onto grid-stride would drop tiles and silently break the
 * full-write).
 * MUST: Kt / Kh / Kw are read from **the weight's third / fourth / fifth axes**, in that order (a
 * cubic kernel still matches numerically when swapped, so the tests pin Kt ≠ Kh ≠ Kw).
 * NOTE: groups > 1 is rejected at Session build (`validateGraphContracts` in plan.ts), so it never
 * gets here. If it did, the params layer ({@link conv3dIgemmParams}) would reject it too.
 */
export const buildConv3d = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const [x, weight] = step.inputShapes;
  const outShape = step.outputs[0].shape;
  const { stride, padding, dilation, groups } = conv3dAttrs(
    step.node.attrs,
    `nodes (${step.node.op})`,
  );
  const dims: Conv3dDims = {
    channelsIn: x[0],
    channelsOut: outShape[0],
    timeIn: x[1],
    heightIn: x[2],
    widthIn: x[3],
    timeOut: outShape[1],
    heightOut: outShape[2],
    widthOut: outShape[3],
    kernelT: weight[2],
    kernelH: weight[3],
    kernelW: weight[4],
    strideT: stride[0],
    strideH: stride[1],
    strideW: stride[2],
    paddingT: padding[0],
    paddingH: padding[1],
    paddingW: padding[2],
    dilationT: dilation[0],
    dilationH: dilation[1],
    dilationW: dilation[2],
    groups,
  };
  const weightStorage = face.weightStorage(step);
  // The integer dot-product path (ADR 0133): the upload reordered this weight to the kernel's
  // tap-major layout (only with convCompute "a8", only when every consumer is an eligible conv3d —
  // conv3dTapMajorInitializers in plan.ts). With the default "f32" nothing below changes by a
  // byte; a node whose weight kept the canonical layout shows it by its key.
  // MUST: a tap-major weight never reaches the f32 kernel (its bytes are permuted) — the flag and
  // the predicate are derived from the same graph facts, so a disagreement is a wiring bug.
  if (face.weightLayout(step) === "conv3d-tap-major") {
    if (face.state.convCompute !== "a8" || !conv3dI8a8Eligible(dims, weightStorage)) {
      throw new ExecutionError(
        `conv3d [${x.join(",")}] * [${weight.join(",")}]: the weight was reordered for the w8a8 kernel but the node is not eligible (convCompute ${face.state.convCompute})`,
      );
    }
    await buildConv3dI8a8(face, step, binds, outs, builder, dims);
    return;
  }
  const kFlat = dims.channelsIn * dims.kernelT * dims.kernelH * dims.kernelW;
  const v4 = conv3dUsesVec4(kFlat, dims.widthOut, dims.strideW);
  const mTile = conv2dIgemmMTile(dims.channelsOut);
  const key = conv3dIgemmKey(weightStorage, v4, mTile);
  // Build params first: shapes outside the contract, such as groups > 1, are rejected before the
  // pipeline lookup.
  const paramsData = conv3dIgemmParams(dims);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    conv3dIgemmWgsl(weightStorage, v4, mTile),
  );
  const params = face.writeParams(paramsData, PARAMS_UNIFORM_USAGE);
  const where = `conv3d [${step.inputShapes[0].join(",")}] * [${step.inputShapes[1].join(",")}]`;
  // The dispatch edges are derived in one place from the same resolution point as generation and
  // key (`gemmMTileGeometry(mTile)`) (src/kernels/conv3d.ts — the unit GPU check goes through the
  // same function).
  const workgroups = conv3dIgemmWorkgroups(
    dims,
    mTile,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
    where,
  );
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      ...binds.map((source, index) => ({ binding: index + 1, source })),
      { binding: 4, source: outs[0] },
      ...face.weightScaleBindings(step, CONV3D_SCALE_BINDING),
    ],
    workgroups,
  });
};

/**
 * conv3d's **w8a8 variant** (opt-in — {@link SessionOptions.convCompute}; ADR 0133).
 * **One node = two dispatches**: ① the channel-major form of `quantize_rows`, which reads
 * `x[Cin, T·H·W]` in place and writes per-pixel i8 + scale in the `[T·H·W, Cin]` layout (one row =
 * one input pixel) → ② the integer dot-product implicit GEMM.
 *
 * MUST: the temporary buffers (`xq`, `xs`) are declared → released at the end of the node (the
 * arena's invariant — the same discipline as linear's i8a8 path).
 */
const buildConv3dI8a8 = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
  dims: Conv3dDims,
): Promise<void> => {
  const where = `conv3d i8a8 [${step.inputShapes[0].join(",")}] * [${step.inputShapes[1].join(",")}]`;
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const pixelsIn = dims.timeIn * dims.heightIn * dims.widthIn;
  const pixelsOut = dims.timeOut * dims.heightOut * dims.widthOut;
  const channels = dims.channelsIn;
  // Build params first: shapes outside the contract are rejected before any pipeline lookup.
  const convParams = conv3dI8a8Params(dims);

  const xq = builder.allocTemp(Math.max(4, pixelsIn * (channels / 4) * 4));
  const xs = builder.allocTemp(Math.max(4, pixelsIn * 4));

  // ① Per-pixel quantisation straight from x[Cin, P] (the time, height and width axes are already
  // contiguous as one pixel axis; grid-stride over blocks of pixels)
  const quantizeGeometry = CONV3D_I8A8_QUANTIZE_GEOMETRY;
  const quantizeKeyValue = quantizeRowsChannelMajorKey(quantizeGeometry);
  const quantize = await face.state.cache.get(
    quantizeKeyValue,
    quantizeRowsChannelMajorWgsl(quantizeGeometry),
  );
  builder.dispatch({
    key: quantizeKeyValue,
    pipeline: quantize.pipeline,
    layout: quantize.layout,
    roles: quantize.roles,
    params: face.writeParams(quantizeRowsParams(pixelsIn, channels), PARAMS_UNIFORM_USAGE),
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: xq },
      { binding: 3, source: xs },
    ],
    workgroups: [quantizeRowsChannelMajorWorkgroups(pixelsIn, quantizeGeometry, limit), 1, 1],
  });

  // ② The integer dot-product implicit GEMM (the geometry is in the key — ADR 0133 decision 5)
  const v4 = conv3dI8a8UsesVec4(pixelsOut);
  const geometry = conv3dI8a8Geometry();
  const dp4a = face.state.linearI8a8Dot === "dp4a";
  const key = conv3dI8a8Key(v4, dp4a, geometry);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    conv3dI8a8Wgsl(v4, dp4a, geometry),
  );
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params: face.writeParams(convParams, PARAMS_UNIFORM_USAGE),
    bindings: [
      { binding: 1, source: xq },
      { binding: 2, source: binds[1] },
      { binding: 3, source: binds[2] },
      { binding: 4, source: outs[0] },
      ...face.weightScaleBindings(step, CONV3D_SCALE_BINDING),
      { binding: CONV3D_ACT_SCALE_BINDING, source: xs },
    ],
    workgroups: conv3dI8a8Workgroups(dims, geometry, limit, where),
  });

  // MUST: return the temporary buffers at the node boundary (the arena's invariant).
  builder.releaseTemp(xs);
  builder.releaseTemp(xq);
};

/**
 * conv_transpose1d (gather form). One thread = one output element writes the whole output
 * (ADR 0014).
 *
 * MUST: Cin is **the weight's first axis** (`[Cin, Cout, K]`). The contract check has already
 * matched it against x[1], but weight[0] is used here so as not to break the reading "the source of
 * truth for the transposed layout is the weight".
 */
export const buildConvTranspose1d = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const [x, weight] = step.inputShapes;
  const outShape = step.outputs[0].shape;
  const { stride, padding } = convTranspose1dAttrs(step.node.attrs, `nodes (${step.node.op})`);
  const weightStorage = face.weightStorage(step);
  const key = convTranspose1dKey(weightStorage);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    convTranspose1dWgsl(weightStorage),
  );
  const params = face.writeParams(
    convTranspose1dParams({
      batch: outShape[0],
      channelsIn: weight[0],
      channelsOut: outShape[1],
      lengthIn: x[2],
      lengthOut: outShape[2],
      kernel: weight[2],
      stride,
      padding,
    }),
    PARAMS_UNIFORM_USAGE,
  );
  const workgroups = gridStrideWorkgroups(
    numel(outShape),
    CONV_TRANSPOSE1D_WORKGROUP_SIZE,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
  );
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      ...binds.map((source, index) => ({ binding: index + 1, source })),
      { binding: 4, source: outs[0] },
      ...face.weightScaleBindings(step, CONV_TRANSPOSE1D_SCALE_BINDING),
    ],
    workgroups: [workgroups, 1, 1],
  });
};
