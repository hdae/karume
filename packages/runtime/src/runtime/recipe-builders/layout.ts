/**
 * Per-op-family derivation — the element-reordering family (strided copy = permute / expand /
 * slice / sym_prefix_slice, cat, pad, flip, gather, upsample_bilinear2d).
 *
 * The entry is `RecipeBuilder` in {@link "../recipe-builder.ts"}; the shared services (Session
 * state and params writes) are reached only through {@link RecipeBuildFace}.
 */

import type { BindingSource, StepRecipeBuilder } from "../recipe.ts";
import { ExecutionError, type NodePlan } from "../plan.ts";
import { FLIP_KEY, FLIP_WGSL, FLIP_WORKGROUP_SIZE, flipParams } from "../../kernels/flip.ts";
import {
  GATHER_KEY,
  GATHER_WGSL,
  GATHER_WORKGROUP_SIZE,
  gatherParams,
} from "../../kernels/gather.ts";
import { PAD_KEY, PAD_WGSL, PAD_WORKGROUP_SIZE, padParams } from "../../kernels/pad.ts";
import {
  catOutOffset,
  catOutStrides,
  expandSrcStrides,
  permuteSrcStrides,
  prefixSliceSrcStrides,
  sliceSrcOffset,
  sliceSrcStrides,
  STRIDED_WORKGROUP_SIZE,
  STRIDED_WRITE_WORKGROUP_SIZE,
  stridedKey,
  stridedParams,
  stridedWgsl,
  stridedWriteKey,
  stridedWriteParams,
  stridedWriteWgsl,
} from "../../codegen/strided.ts";
import {
  UPSAMPLE_BILINEAR2D_KEY,
  UPSAMPLE_BILINEAR2D_WGSL,
  UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE,
  upsampleBilinear2dParams,
} from "../../kernels/upsample-bilinear2d.ts";
import { catDim, flipDim, numel, padAttrs, permuteDims, sliceAttrs } from "../../ops.ts";
import { gridStrideWorkgroups } from "../../codegen/dispatch.ts";
import type { RecipeBuildFace } from "../recipe-builder.ts";
import { PARAMS_STORAGE_USAGE, PARAMS_UNIFORM_USAGE } from "./params-usage.ts";

/**
 * The materialising copy of permute / non-identity expand / slice / sym_prefix_slice (one
 * strided-read kernel family — ADR 0011 / 0010 / 0014). The output is always contiguous; only the
 * input side is read with strides. The replicated axes of expand have stride 0, sym_prefix_slice
 * uses the contiguous strides of **the Tmax-shaped input**, and slice uses the input's contiguous
 * strides plus **a start offset**. An identity expand is aliased and never comes here.
 */
export const buildStridedCopy = async (
  face: RecipeBuildFace,
  step: NodePlan,
  kind: "permute" | "expand" | "slice" | "symPrefixSlice",
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const srcShape = step.inputShapes[0];
  const outShape = step.outputs[0].shape;
  const where = `nodes (${step.node.op})`;
  const srcStrides =
    kind === "permute"
      ? permuteSrcStrides(srcShape, permuteDims(step.node.attrs, where))
      : kind === "expand"
        ? expandSrcStrides(srcShape, outShape)
        : kind === "slice"
          ? sliceSrcStrides(srcShape)
          : // MUST: build from **srcShape (the Tmax shape)**, not the bound outShape (the MUST of
            // prefixSliceSrcStrides — a shrunk stride makes every row after the first read the
            // wrong row).
            prefixSliceSrcStrides(srcShape);
  // offset is exactly the (offset, strides[4]) model of ADR 0011. permute / expand /
  // sym_prefix_slice read from the start of the input, so it is 0; **only slice** takes a non-zero
  // value (this is the "one variable word" ADR 0011 anticipated).
  let offset = 0;
  if (kind === "slice") {
    const { dim, start } = sliceAttrs(step.node.attrs, where);
    offset = sliceSrcOffset(srcShape, dim, start);
  }
  // The element type is part of the key (expands of bool masks / i32 indices are measured —
  // ADR 0009).
  const spec = { dtype: step.outputs[0].dtype };
  const key = stridedKey(spec);
  const { pipeline, layout, roles } = await face.state.cache.get(key, stridedWgsl(spec));
  const params = face.writeParams(
    stridedParams(outShape, srcStrides, offset),
    PARAMS_STORAGE_USAGE,
  );
  const groups = gridStrideWorkgroups(
    numel(outShape),
    STRIDED_WORKGROUP_SIZE,
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
 * cat (the strided-write copy family — ADR 0014). **One dispatch** per input writes into a
 * sub-region of the output. The input is read contiguously and written to the output with
 * `(offset, the output's contiguous strides)` (the dual of the read family).
 *
 * MUST: check that the write positions add up to exactly the length of the concatenation axis
 * (full-write). Otherwise some part of the output is **left unwritten**, and with a reallocated
 * buffer the previous value becomes the result as is. The same fact as the contract's shape rule
 * (axis length = sum of the inputs' axis lengths) is also checked from the encoder side's
 * accumulation, so an error on only one side fails as an internal contradiction.
 */
export const buildCat = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const outShape = step.outputs[0].shape;
  const where = `nodes (${step.node.op})`;
  const dim = catDim(step.node.attrs, where);
  const outStrides = catOutStrides(outShape);
  const spec = { dtype: step.outputs[0].dtype };
  const key = stridedWriteKey(spec);
  const { pipeline, layout, roles } = await face.state.cache.get(key, stridedWriteWgsl(spec));
  let written = 0;
  for (const [index, source] of binds.entries()) {
    const srcShape = step.inputShapes[index];
    const params = face.writeParams(
      stridedWriteParams(srcShape, outStrides, catOutOffset(outShape, dim, written)),
      PARAMS_STORAGE_USAGE,
    );
    const groups = gridStrideWorkgroups(
      numel(srcShape),
      STRIDED_WRITE_WORKGROUP_SIZE,
      face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
    );
    builder.dispatch({
      key,
      pipeline,
      layout,
      roles,
      params,
      bindings: [
        { binding: 1, source },
        { binding: 2, source: outs[0] },
      ],
      workgroups: [groups, 1, 1],
    });
    written += srcShape[dim];
  }
  if (written !== outShape[dim]) {
    throw new ExecutionError(
      `${where}: the cat writes do not cover the whole output (axis ${dim}: ${written} / output ${
        outShape[dim]
      })`,
    );
  }
};

/**
 * pad (constant zero padding of the last dimension). **One dispatch writes every byte of the
 * output** (copied inside the range, 0 outside — the full-write of ADR 0014). It does not assume a
 * zero-initialised buffer.
 */
export const buildPad = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const srcShape = step.inputShapes[0];
  const outShape = step.outputs[0].shape;
  const { left, right } = padAttrs(step.node.attrs, `nodes (${step.node.op})`);
  const { pipeline, layout, roles } = await face.state.cache.get(PAD_KEY, PAD_WGSL);
  const params = face.writeParams(
    padParams(numel(srcShape.slice(0, -1)), srcShape[srcShape.length - 1], left, right),
    PARAMS_UNIFORM_USAGE,
  );
  const groups = gridStrideWorkgroups(
    numel(outShape),
    PAD_WORKGROUP_SIZE,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
  );
  builder.dispatch({
    key: PAD_KEY,
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
 * flip (index reversal on a static axis). The axis position is passed folded into the three parts
 * `[outer, len, inner]` (rank is not carried in params — src/kernels/flip.ts).
 */
export const buildFlip = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const shape = step.outputs[0].shape;
  const dim = flipDim(step.node.attrs, `nodes (${step.node.op})`);
  const { pipeline, layout, roles } = await face.state.cache.get(FLIP_KEY, FLIP_WGSL);
  const params = face.writeParams(
    // MUST: split before and after the axis (`slice(0, dim)` and `slice(dim + 1)`). Shifting the
    // boundary by one moves the reversed axis to its neighbour, and only the values go wrong while
    // shape and element count stay the same.
    flipParams(numel(shape.slice(0, dim)), shape[dim], numel(shape.slice(dim + 1))),
    PARAMS_UNIFORM_USAGE,
  );
  const groups = gridStrideWorkgroups(
    numel(shape),
    FLIP_WORKGROUP_SIZE,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
  );
  builder.dispatch({
    key: FLIP_KEY,
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
 * upsample_bilinear2d (bilinear resampling of the two spatial axes of NCHW, `align_corners = True`
 * only). Grid-stride with one output element = one invocation; params carry only the four spatial
 * lengths (the kernel folds N·C into a plane index — src/kernels/upsample-bilinear2d.ts).
 */
export const buildUpsampleBilinear2d = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const srcShape = step.inputShapes[0];
  const outShape = step.outputs[0].shape;
  const { pipeline, layout, roles } = await face.state.cache.get(
    UPSAMPLE_BILINEAR2D_KEY,
    UPSAMPLE_BILINEAR2D_WGSL,
  );
  const params = face.writeParams(
    upsampleBilinear2dParams(numel(outShape), srcShape[2], srcShape[3], outShape[2], outShape[3]),
    PARAMS_UNIFORM_USAGE,
  );
  const groups = gridStrideWorkgroups(
    numel(outShape),
    UPSAMPLE_BILINEAR2D_WORKGROUP_SIZE,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
  );
  builder.dispatch({
    key: UPSAMPLE_BILINEAR2D_KEY,
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
 * gather over the last dimension. The output is contiguous, reading `src[row * D + index[i]]` with
 * `row = i / J`. Out-of-range indices follow the ruling in src/kernels/gather.ts (the GPU poisons
 * with NaN / the CPU reference throws).
 */
export const buildGather = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const srcShape = step.inputShapes[0];
  const outShape = step.outputs[0].shape;
  const count = numel(outShape);
  const { pipeline, layout, roles } = await face.state.cache.get(GATHER_KEY, GATHER_WGSL);
  const params = face.writeParams(
    gatherParams(count, outShape[outShape.length - 1], srcShape[srcShape.length - 1]),
    PARAMS_UNIFORM_USAGE,
  );
  const groups = gridStrideWorkgroups(
    count,
    GATHER_WORKGROUP_SIZE,
    face.state.gpu.limits.maxComputeWorkgroupsPerDimension,
  );
  builder.dispatch({
    key: GATHER_KEY,
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
