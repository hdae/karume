/**
 * 族別導出 — 行正規化と固定量子化（layer_norm・static_quantize・rms_norm）。
 *
 * 入口は {@link "../recipe-builder.ts"} の `RecipeBuilder` で、共有サービス（Session の状態と
 * params の書き込み）は {@link RecipeBuildFace} 経由でだけ触る。
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

/** layer_norm（最終次元・affine あり）。1 行 = 1 workgroup で、行方向は grid-stride。 */
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
  const { pipeline, layout, roles } = await face.state.cache.get(
    LAYER_NORM_KEY,
    LAYER_NORM_WGSL,
  );
  const params = face.writeParams(
    layerNormParams(rows, dim, eps),
    PARAMS_UNIFORM_USAGE,
  );
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
 * 固定 SRQ の表は既存の不変 params キャッシュで保持する（ADR 0097）。
 *
 * `packed` が `"write"` のときは出力を packed int8（u32 1 語 = 4 要素）で書く別キーへ落とす
 * （ADR 0105）。出力バッファの確保は宣言 shape のまま（= 実際に書くのはその 1/4）で、
 * アリーナのバケットを動かさない。
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
      `static_quantize: packed 活性の役割 '${packed.role}' はこの op には来ない`,
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
    bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: outs[0] }],
    workgroups: [
      // packed も f32 経路と同じ「1 スレッド 1 要素」の幾何（ADR 0105 追記 1）なので、
      // dispatch 数は要素数から引く（詰めるのは workgroup 内の共有メモリ経由）。
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
 * rms_norm（最終次元・weight のみ）。layer_norm と同じ 1 行 = 1 workgroup の形で、
 * 縮約は二乗和 1 パス（ADR 0017）。
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
