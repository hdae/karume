/**
 * Per-op-family derivation — row normalisation and fixed quantization (layer_norm, group_norm,
 * static_quantize, rms_norm).
 *
 * The entry is `RecipeBuilder` in {@link "../recipe-builder.ts"}; the shared services (Session
 * state and params writes) are reached only through {@link RecipeBuildFace}.
 */

import type { BindingSource, StepRecipeBuilder } from "../recipe.ts";
import { ExecutionError, type NodePlan } from "../plan.ts";
import {
  GROUP_NORM_APPLY_KEY,
  GROUP_NORM_APPLY_WGSL,
  GROUP_NORM_PARTIAL_KEY,
  GROUP_NORM_PARTIAL_WGSL,
  GROUP_NORM_TOTAL_KEY,
  GROUP_NORM_TOTAL_WGSL,
  GROUP_NORM_WORKGROUP_SIZE,
  groupNormApplyParams,
  groupNormLayout,
  groupNormPartialParams,
  groupNormTempBytes,
  groupNormTotalParams,
} from "../../kernels/group-norm.ts";
import { LAYER_NORM_KEY, LAYER_NORM_WGSL, layerNormParams } from "../../kernels/layer-norm.ts";
import type { PackedActivations } from "../fusion.ts";
import {
  RMS_NORM_128_KEY,
  RMS_NORM_128_WGSL,
  rmsNormKernel,
  rmsNormParams,
  rmsNormWorkgroupKey,
  rmsNormWorkgroupWgsl,
} from "../../kernels/rms-norm.ts";
import {
  STATIC_QUANTIZE_KEY,
  STATIC_QUANTIZE_PACKED_KEY,
  STATIC_QUANTIZE_PACKED_WGSL,
  STATIC_QUANTIZE_WGSL,
  STATIC_QUANTIZE_WORKGROUP_SIZE,
  staticQuantizePackedParams,
  staticQuantizeParams,
} from "../../kernels/static-quantize.ts";
import { gridStrideWorkgroups } from "../../codegen/dispatch.ts";
import {
  groupNormAttrs,
  layerNormAttrs,
  numel,
  rmsNormEps,
  staticQuantizeScale,
} from "../../ops.ts";
import { rmsNormSubgroupKey, rmsNormSubgroupWgsl } from "../../kernels/rms-norm-subgroup.ts";
import type { RecipeBuildFace } from "../recipe-builder.ts";
import { PARAMS_UNIFORM_USAGE } from "./params-usage.ts";

/** layer_norm (last dimension, with affine). One row = one workgroup, grid-stride over rows. */
export const buildLayerNorm = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const shape = step.outputs[0].shape;
  const dim = shape[shape.length - 1];
  const rows = numel(shape.slice(0, -1));
  const { eps } = layerNormAttrs(step.node.attrs, `nodes (${step.node.op})`);
  const { pipeline, layout, roles } = await face.state.cache.get(LAYER_NORM_KEY, LAYER_NORM_WGSL);
  const params = face.writeParams(layerNormParams(rows, dim, eps), PARAMS_UNIFORM_USAGE);
  const groups = gridStrideWorkgroups(
    rows,
    1,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
  );
  builder.dispatch({
    key: LAYER_NORM_KEY,
    pipeline,
    layout,
    roles,
    params,
    bindings: [
      ...binds.map((source, index) => ({ binding: index + 1, source })),
      { binding: 4, source: outs[0] },
    ],
    workgroups: [groups, 1, 1],
  });
};

/**
 * group_norm (channel groups, per-channel affine — ADR 0138 decision 7). Five dispatches over two
 * node temporaries — the partial sums and the per-row (mean, rstd) pairs (the stage table of
 * {@link "../../kernels/group-norm.ts"}). Every dispatch is grid-stride, so a node whose partial
 * stage needs more than the per-dimension workgroup limit loops instead of failing.
 */
export const buildGroupNorm = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const [x, weight, bias] = binds;
  const { groups, eps } = groupNormAttrs(step.node.attrs, `nodes (${step.node.op})`);
  const layout = groupNormLayout(step.outputs[0].shape, groups);
  const limit = face.state.gpu.limits.maxComputeWorkgroupsPerDimension;
  const [partial, total, apply] = await Promise.all([
    face.state.cache.get(GROUP_NORM_PARTIAL_KEY, GROUP_NORM_PARTIAL_WGSL),
    face.state.cache.get(GROUP_NORM_TOTAL_KEY, GROUP_NORM_TOTAL_WGSL),
    face.state.cache.get(GROUP_NORM_APPLY_KEY, GROUP_NORM_APPLY_WGSL),
  ]);
  const bytes = groupNormTempBytes(layout);
  const partials = builder.allocTemp(bytes.partials);
  const stats = builder.allocTemp(bytes.stats);
  const partialGroups = gridStrideWorkgroups(layout.rows * layout.chunks, 1, limit);
  const totalGroups = gridStrideWorkgroups(layout.rows, 1, limit);
  for (const pass of [0, 1] as const) {
    builder.dispatch({
      key: GROUP_NORM_PARTIAL_KEY,
      ...partial,
      params: face.writeParams(groupNormPartialParams(layout, pass), PARAMS_UNIFORM_USAGE),
      bindings: [
        { binding: 1, source: x },
        { binding: 2, source: stats },
        { binding: 3, source: partials },
      ],
      workgroups: [partialGroups, 1, 1],
    });
    builder.dispatch({
      key: GROUP_NORM_TOTAL_KEY,
      ...total,
      params: face.writeParams(groupNormTotalParams(layout, pass, eps), PARAMS_UNIFORM_USAGE),
      bindings: [
        { binding: 1, source: partials },
        { binding: 2, source: stats },
      ],
      workgroups: [totalGroups, 1, 1],
    });
  }
  builder.releaseTemp(partials);
  builder.dispatch({
    key: GROUP_NORM_APPLY_KEY,
    ...apply,
    params: face.writeParams(groupNormApplyParams(layout), PARAMS_UNIFORM_USAGE),
    bindings: [
      { binding: 1, source: x },
      { binding: 2, source: stats },
      { binding: 3, source: weight },
      { binding: 4, source: bias },
      { binding: 5, source: outs[0] },
    ],
    workgroups: [gridStrideWorkgroups(layout.count, GROUP_NORM_WORKGROUP_SIZE, limit), 1, 1],
  });
  builder.releaseTemp(stats);
};

/**
 * The fixed SRQ table is held in the existing immutable params cache (ADR 0097).
 *
 * When `packed` is `"write"`, the output goes to a separate key that writes packed int8 (one u32
 * word = 4 elements) (ADR 0105). The output buffer is still allocated with the declared shape
 * (= only 1/4 of it is actually written), so the arena buckets do not move.
 */
export const buildStaticQuantize = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
  packed: PackedActivations | undefined,
): Promise<void> => {
  const count = numel(step.outputs[0].shape);
  const scale = staticQuantizeScale(step.node.attrs, `nodes (${step.node.op})`);
  if (packed !== undefined && packed.role !== "write") {
    throw new ExecutionError(
      `static_quantize: the packed activation role '${packed.role}' does not occur for this op`,
    );
  }
  const writesPacked = packed !== undefined;
  const key = writesPacked ? STATIC_QUANTIZE_PACKED_KEY : STATIC_QUANTIZE_KEY;
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    writesPacked ? STATIC_QUANTIZE_PACKED_WGSL : STATIC_QUANTIZE_WGSL,
  );
  const params = face.writeParams(
    writesPacked ? staticQuantizePackedParams(count, scale) : staticQuantizeParams(count, scale),
    PARAMS_UNIFORM_USAGE,
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
    workgroups: [
      // packed uses the same "one thread, one element" geometry as the f32 path (ADR 0105
      // addendum 1), so the dispatch count comes from the element count (packing goes through
      // workgroup shared memory).
      gridStrideWorkgroups(
        count,
        STATIC_QUANTIZE_WORKGROUP_SIZE,
        face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
      ),
      1,
      1,
    ],
  });
};

/**
 * rms_norm (last dimension, weight only). The same one row = one workgroup shape as layer_norm;
 * the reduction is a single sum-of-squares pass (ADR 0017).
 */
export const buildRmsNorm = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const shape = step.outputs[0].shape;
  const dim = shape[shape.length - 1];
  const rows = numel(shape.slice(0, -1));
  const eps = rmsNormEps(step.node.attrs, `nodes (${step.node.op})`);
  const kernel = rmsNormKernel(dim, face.state.rmsNormReduce);
  const key =
    kernel === "workgroup128"
      ? RMS_NORM_128_KEY
      : kernel === "subgroup32"
        ? rmsNormSubgroupKey()
        : rmsNormWorkgroupKey(dim);
  const wgsl =
    kernel === "workgroup128"
      ? RMS_NORM_128_WGSL
      : kernel === "subgroup32"
        ? rmsNormSubgroupWgsl()
        : rmsNormWorkgroupWgsl(dim);
  const { pipeline, layout, roles } = await face.state.cache.get(key, wgsl);
  const params = face.writeParams(rmsNormParams(rows, dim, eps), PARAMS_UNIFORM_USAGE);
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
      ...binds.map((source, index) => ({ binding: index + 1, source })),
      { binding: 3, source: outs[0] },
    ],
    workgroups: [groups, 1, 1],
  });
};
