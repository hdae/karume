/**
 * Per-op-family derivation — row normalisation and fixed quantization (layer_norm,
 * static_quantize, rms_norm).
 *
 * The entry is `RecipeBuilder` in {@link "../recipe-builder.ts"}; the shared services (Session
 * state and params writes) are reached only through {@link RecipeBuildFace}.
 */

import type { BindingSource, StepRecipeBuilder } from "../recipe.ts";
import { ExecutionError, type NodePlan } from "../plan.ts";
import { LAYER_NORM_KEY, LAYER_NORM_WGSL, layerNormParams } from "../../kernels/layer-norm.ts";
import type { PackedActivations } from "../fusion.ts";
import {
  RMS_NORM_128_KEY,
  RMS_NORM_128_WGSL,
  RMS_NORM_KEY,
  RMS_NORM_WGSL,
  rmsNormParams,
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
import { layerNormAttrs, numel, rmsNormEps, staticQuantizeScale } from "../../ops.ts";
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
  const narrow = dim > 0 && dim <= 128;
  const subgroup = !narrow && face.state.rmsNormReduce === "subgroup32";
  const key = narrow ? RMS_NORM_128_KEY : subgroup ? rmsNormSubgroupKey() : RMS_NORM_KEY;
  const wgsl = narrow ? RMS_NORM_128_WGSL : subgroup ? rmsNormSubgroupWgsl() : RMS_NORM_WGSL;
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
