/**
 * Per-op-family derivation — elementwise (unary / binary / cast / where), row reduce, argmax / topk
 * and their split paths, cumsum, softmax, masked_fill.
 *
 * The entry is `RecipeBuilder` in {@link "../recipe-builder.ts"}; the shared services (Session
 * state and params writes) are reached only through {@link RecipeBuildFace}.
 */

import {
  ARGMAX_KEY,
  ARGMAX_SPLIT_MERGE_KEY,
  ARGMAX_SPLIT_MERGE_WGSL,
  ARGMAX_SPLIT_PARTIAL_KEY,
  ARGMAX_SPLIT_PARTIAL_WGSL,
  ARGMAX_WGSL,
  argmaxParams,
  argmaxSplitGroups,
  argmaxSplitParams,
  argmaxSplitPartialBytes,
  TOPK_ONE_SPLIT_MERGE_KEY,
  topkOneSplitGroups,
  topkOneSplitMergeWgsl,
} from "../../kernels/argmax.ts";
import {
  AXIS_REDUCE_WORKGROUP_SIZE,
  axisReduceKey,
  axisReduceParams,
  axisReduceWgsl,
  reduceKey,
  reduceParams,
  reduceWgsl,
} from "../../codegen/reduce.ts";
import type { BindingSource, StepRecipeBuilder } from "../recipe.ts";
import {
  type BinaryOpName,
  CAST_OP,
  maskedFillValue,
  numel,
  reduceDim,
  type ReduceOpName,
  scalarParamValues,
  topkK,
  type UnaryOpName,
  type WHERE_OP,
} from "../../ops.ts";
import {
  CUMSUM_KEY,
  CUMSUM_WGSL,
  CUMSUM_WORKGROUP_SIZE,
  cumsumParams,
} from "../../kernels/cumsum.ts";
import {
  ELEMENTWISE_WORKGROUP_SIZE,
  elementwiseKey,
  elementwiseParams,
  type ElementwiseSpec,
  elementwiseWgsl,
} from "../../codegen/elementwise.ts";
import {
  MASKED_FILL_KEY,
  MASKED_FILL_WGSL,
  MASKED_FILL_WORKGROUP_SIZE,
  maskedFillParams,
} from "../../kernels/masked-fill.ts";
import type { NodePlan } from "../plan.ts";
import {
  SAFE_SOFTMAX_KEY,
  SAFE_SOFTMAX_WGSL,
  SOFTMAX_KEY,
  SOFTMAX_WGSL,
  softmaxParams,
} from "../../kernels/softmax.ts";
import { assertTopkK, topkKey, topkParams, topkWgsl } from "../../kernels/topk.ts";
import { expandSrcStrides } from "../../codegen/strided.ts";
import { gridStrideWorkgroups, tiledWorkgroups } from "../../codegen/dispatch.ts";
import type { IrDtype } from "../../format/ir.ts";
import type { RecipeBuildFace } from "../recipe-builder.ts";
import { PARAMS_STORAGE_USAGE, PARAMS_UNIFORM_USAGE } from "./params-usage.ts";

/** The rank-independent part of the elementwise family's generation input (rank set at encode). */
type ElementwiseOp =
  | { readonly op: UnaryOpName | BinaryOpName | typeof WHERE_OP; readonly dtype: IrDtype }
  | { readonly op: typeof CAST_OP; readonly dtype: IrDtype; readonly to: IrDtype };

export const buildElementwise = async (
  face: RecipeBuildFace,
  step: NodePlan,
  element: ElementwiseOp,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  // Rank 0 (a scalar) is normalised to one dimension of length 1 for codegen's rank ≥ 1 contract.
  const outShape = step.outputs[0].shape.length === 0 ? [1] : [...step.outputs[0].shape];
  const rank = outShape.length;
  const spec: ElementwiseSpec =
    element.op === CAST_OP
      ? { op: CAST_OP, rank, dtype: element.dtype, to: element.to }
      : { op: element.op, rank, dtype: element.dtype };
  const key = elementwiseKey(spec);
  const { pipeline, layout, roles } = await face.state.cache.get(key, elementwiseWgsl(spec));
  // Scalars from attrs (clamp's min/max and so on) go at the end of params as f32 (order per the
  // contract table).
  const params = face.writeParams(
    elementwiseParams(
      spec,
      outShape,
      step.inputShapes,
      scalarParamValues(step.contract, step.node.attrs, `nodes (${step.node.op})`),
    ),
    PARAMS_STORAGE_USAGE,
  );
  const groups = gridStrideWorkgroups(
    numel(outShape),
    ELEMENTWISE_WORKGROUP_SIZE,
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
      { binding: binds.length + 1, source: outs[0] },
    ],
    workgroups: [groups, 1, 1],
  });
};

/**
 * The reduce family (sum / amax / amin). **Two variants, chosen by the reduction axis**:
 *
 * - last dimension → row reduce (the existing kernel, one row = one workgroup + a 256-wide tree)
 * - any other → axis reduce (one thread = one output, coalesced reads. The reduction order
 *   matches the row reduce exactly = bit-identical output — {@link axisReduceWgsl} in
 *   src/codegen/reduce.ts)
 *
 * MUST: the branch is decided **by the axis only**. Leaving room to choose by speed would allow the
 * axis variant to run even on the last dimension, and the default path's bit stability (the PNG
 * sha256 check) would depend on run-time conditions.
 */
export const buildRowReduce = async (
  face: RecipeBuildFace,
  step: NodePlan,
  op: ReduceOpName,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const inputShape = step.inputShapes[0];
  const axis = reduceDim(step.node.attrs, `nodes (${step.node.op})`);
  // The element type is part of the key (a sum over bool reads u32 and writes an i32 count —
  // ADR 0009).
  const spec = { op, dtype: step.inputDtypes[0] };
  const lastDim = axis === inputShape.length - 1;
  const outCount = numel(step.outputs[0].shape);
  const key = lastDim ? reduceKey(spec) : axisReduceKey(spec);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    lastDim ? reduceWgsl(spec) : axisReduceWgsl(spec),
  );
  const inner = numel(inputShape.slice(axis + 1));
  const params = face.writeParams(
    lastDim
      ? reduceParams(outCount, inputShape[axis])
      : axisReduceParams(outCount, inputShape[axis], inner),
    PARAMS_UNIFORM_USAGE,
  );
  // Row reduce is one row = one workgroup, axis reduce one thread = one output. Both are clamped
  // beyond the limit, and the kernel's grid-stride loop covers the rest.
  const groups = gridStrideWorkgroups(
    outCount,
    lastDim ? 1 : AXIS_REDUCE_WORKGROUP_SIZE,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
  );
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: outs[0] },
    ],
    workgroups: [groups, 1, 1],
  });
};

/**
 * argmax (last dimension, rank kept, i32 output — ADR 0068 decision 2). Short rows take **one
 * dispatch**, the same "one row = one workgroup + grid-stride over rows" as the row reduce (the
 * basis of the shape and the tie-break is in src/kernels/argmax.ts). Long rows
 * (`argmaxSplitGroups(dim) > 0` — the vocabulary-length lm_head exit) take **two dispatches**
 * (partial maximum per range → merge per row), bit-identical to the one-dispatch form (the
 * two-phase split section of the same file).
 *
 * MUST: no branching by axis (unlike the reduce family, last dimension only — the contract has no
 * `dim` field).
 * The row count comes from **the product of the input's leading dimensions** (it equals the
 * output's element count, but how much the kernel reads is set by the input's shape).
 * MUST: the path choice is a pure function of the shape (`argmaxSplitGroups`) — the same function
 * as the estimate (the argmax temporary in estimate.ts) goes through.
 */
export const buildArgmax = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const inputShape = step.inputShapes[0];
  const dim = inputShape[inputShape.length - 1];
  const rows = numel(inputShape.slice(0, -1));
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const groups = argmaxSplitGroups(dim);
  // An empty outer extent stays on the existing path (as in topk(k=1) — a 0 B split temporary does
  // not fit the recipe's lifetime contract).
  if (rows > 0 && groups > 0) {
    await buildMaxIndexSplit(face, rows, dim, groups, binds[0], outs[0], builder);
    return;
  }
  const { pipeline, layout, roles } = await face.state.cache.get(ARGMAX_KEY, ARGMAX_WGSL);
  const params = face.writeParams(argmaxParams(rows, dim), PARAMS_UNIFORM_USAGE);
  // One row = one workgroup. Clamped beyond the limit; the kernel's row grid-stride loop covers
  // the rest.
  builder.dispatch({
    key: ARGMAX_KEY,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: outs[0] },
    ],
    workgroups: [gridStrideWorkgroups(rows, 1, limit), 1, 1],
  });
};

/**
 * Shares the reduction of partial maximum and smallest index; only topk(k=1) also copies the value
 * bits of the original input.
 * DECIDED: docs/decisions/0068-decode-exit-multi-output.md#addendum-9-2026-09-11--topk-k1-over-long-rows-also-uses-two-dispatches
 */
const buildMaxIndexSplit = async (
  face: RecipeBuildFace,
  rows: number,
  dim: number,
  groups: number,
  input: BindingSource,
  index: BindingSource,
  builder: StepRecipeBuilder,
  value?: BindingSource,
): Promise<void> => {
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const mergeKey = value === undefined ? ARGMAX_SPLIT_MERGE_KEY : TOPK_ONE_SPLIT_MERGE_KEY;
  const partialPipeline = await face.state.cache.get(
    ARGMAX_SPLIT_PARTIAL_KEY,
    ARGMAX_SPLIT_PARTIAL_WGSL,
  );
  const mergePipeline = await face.state.cache.get(
    mergeKey,
    value === undefined ? ARGMAX_SPLIT_MERGE_WGSL : topkOneSplitMergeWgsl(),
  );
  const params = face.writeParams(argmaxSplitParams(rows, dim, groups), PARAMS_UNIFORM_USAGE);
  // Partial result [rows, groups, 2] (u32). Allocated right before partial, returned right after
  // merge.
  const partial = builder.allocTemp(argmaxSplitPartialBytes(rows, groups));
  builder.dispatch({
    key: ARGMAX_SPLIT_PARTIAL_KEY,
    pipeline: partialPipeline.pipeline,
    layout: partialPipeline.layout,
    roles: partialPipeline.roles,
    params,
    bindings: [
      { binding: 1, source: input },
      { binding: 2, source: partial },
    ],
    // x axis = ranges (one workgroup = one range; a missing one is a silent wrong value, so
    // exceeding the limit fails loudly), y axis = rows (may be clamped onto grid-stride).
    workgroups: [
      tiledWorkgroups(groups, 1, limit, "argmax split partial"),
      gridStrideWorkgroups(rows, 1, limit),
      1,
    ],
  });
  builder.dispatch({
    key: mergeKey,
    pipeline: mergePipeline.pipeline,
    layout: mergePipeline.layout,
    roles: mergePipeline.roles,
    params,
    bindings: [
      { binding: 1, source: partial },
      { binding: 2, source: index },
      ...(value === undefined
        ? []
        : [
            { binding: 3, source: value },
            { binding: 4, source: input },
          ]),
    ],
    workgroups: [gridStrideWorkgroups(rows, 1, limit), 1, 1],
  });
  builder.releaseTemp(partial);
};

/**
 * topk (last dimension, static k, **two outputs** — ADR 0068 decision 3).
 * Long rows with k=1 share argmax's partial maximum and merge; everything else is one dispatch.
 * The values copy the selected original input's u32, keeping NaN payloads and signed zeros.
 *
 * MUST: outputs are **received as a list** (`outs[0]` = values f32, `outs[1]` = indices i32). The
 * order is ascending output slot, the same as `StepRecipe.outputs` in {@link "../recipe.ts"}, and
 * the order of `node.outs` maps directly onto bindings 2 / 3 of the bind surface of the
 * one-dispatch path. Swapping them keeps shape and byteLength the same (both are 4-byte elements of
 * `[…, k]`), so **values and indices trade places with no exception**.
 * MUST: the split path's temporaries are counted in both execution and estimate. They do not share
 * storage with the outputs.
 * The one-dispatch path's scratch stays in workgroup storage, and the implementation limit on k
 * is decided from the device limit ({@link assertTopkK} — no fallback).
 */
export const buildTopk = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const inputShape = step.inputShapes[0];
  const dim = inputShape[inputShape.length - 1];
  const rows = numel(inputShape.slice(0, -1));
  const where = `nodes (${step.node.op})`;
  // `1 ≤ k ≤ last dimension` is already settled by the contract layer (attrs schema + shape
  // rules). What is checked here is only **the device-dependent implementation limit** (workgroup
  // storage — fails loudly with the limit value).
  const k = topkK(step.node.attrs, where);
  assertTopkK(k, face.state.gpu.limits.maxComputeWorkgroupStorageSize, where);
  const splitGroups = topkOneSplitGroups(dim);
  // An empty outer extent stays on the existing path. A 0 B split temporary does not fit the
  // recipe's lifetime contract.
  if (k === 1 && rows > 0 && splitGroups > 0) {
    await buildMaxIndexSplit(face, rows, dim, splitGroups, binds[0], outs[1], builder, outs[0]);
    return;
  }
  const key = topkKey(k);
  const { pipeline, layout, roles } = await face.state.cache.get(key, topkWgsl(k));
  const params = face.writeParams(topkParams(rows, dim), PARAMS_UNIFORM_USAGE);
  // One row = one workgroup. Clamped beyond the limit; the kernel's row grid-stride loop covers
  // the rest.
  const groups = gridStrideWorkgroups(
    rows,
    1,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
  );
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: outs[0] },
      { binding: 3, source: outs[1] },
    ],
    workgroups: [groups, 1, 1],
  });
};

/**
 * cumsum (prefix sum over the last dimension). A sequential scan with **one invocation = one
 * row**, grid-stride over rows (the basis of the shape is in src/kernels/cumsum.ts).
 */
export const buildCumsum = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const shape = step.outputs[0].shape;
  const dim = shape[shape.length - 1];
  const rows = numel(shape.slice(0, -1));
  const { pipeline, layout, roles } = await face.state.cache.get(CUMSUM_KEY, CUMSUM_WGSL);
  const params = face.writeParams(cumsumParams(rows, dim), PARAMS_UNIFORM_USAGE);
  const groups = gridStrideWorkgroups(
    rows,
    CUMSUM_WORKGROUP_SIZE,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
  );
  builder.dispatch({
    key: CUMSUM_KEY,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: outs[0] },
    ],
    workgroups: [groups, 1, 1],
  });
};

/**
 * softmax (last dimension, safe-softmax). The same one row = one workgroup shape as layer_norm.
 *
 * `safe` is the safe_softmax variant (writes 0 to a row whose max is −inf — ADR 0044). The two
 * kernels come from the same generator, and the dispatch shape (bindings, workgroup count) is the
 * same.
 */
export const buildSoftmax = async (
  face: RecipeBuildFace,
  step: NodePlan,
  safe: boolean,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const shape = step.outputs[0].shape;
  const dim = shape[shape.length - 1];
  const rows = numel(shape.slice(0, -1));
  const key = safe ? SAFE_SOFTMAX_KEY : SOFTMAX_KEY;
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    safe ? SAFE_SOFTMAX_WGSL : SOFTMAX_WGSL,
  );
  const params = face.writeParams(softmaxParams(rows, dim, safe), PARAMS_UNIFORM_USAGE);
  const groups = gridStrideWorkgroups(
    rows,
    1,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
  );
  builder.dispatch({
    key,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: outs[0] },
    ],
    workgroups: [groups, 1, 1],
  });
};

/**
 * masked_fill. The output and x have the same shape and are contiguous; only the mask is read with
 * right-aligned broadcast strides. The strides follow the same rule as expand in the strided family
 * ({@link expandSrcStrides}).
 */
export const buildMaskedFill = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const outShape = step.outputs[0].shape;
  // The rule is the same as expand, but the diagnostic's subject is switched to masked_fill (an
  // "input of expand" message on a graph with no expand would point away from the cause).
  const maskStrides = expandSrcStrides(step.inputShapes[1], outShape, {
    src: "masked_fill mask",
    out: "masked_fill output",
  });
  const value = maskedFillValue(step.node.attrs, `nodes (${step.node.op})`);
  const { pipeline, layout, roles } = await face.state.cache.get(MASKED_FILL_KEY, MASKED_FILL_WGSL);
  const params = face.writeParams(
    maskedFillParams(outShape, maskStrides, value),
    PARAMS_STORAGE_USAGE,
  );
  const groups = gridStrideWorkgroups(
    numel(outShape),
    MASKED_FILL_WORKGROUP_SIZE,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
  );
  builder.dispatch({
    key: MASKED_FILL_KEY,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      { binding: 1, source: binds[0] },
      { binding: 2, source: binds[1] },
      { binding: 3, source: outs[0] },
    ],
    workgroups: [groups, 1, 1],
  });
};
