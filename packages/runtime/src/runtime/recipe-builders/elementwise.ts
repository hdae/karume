/**
 * 族別導出 — elementwise（単項 / 二項 / cast / where）・行 reduce・argmax / topk とその分割経路・
 * cumsum・softmax・masked_fill。
 *
 * 入口は {@link "../recipe-builder.ts"} の `RecipeBuilder` で、共有サービス（Session の状態と
 * params の書き込み）は {@link RecipeBuildFace} 経由でだけ触る。
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

/** elementwise 族の生成入力のうち rank に依らない部分（rank はエンコード時に決まる）。 */
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
  // rank 0（スカラ）は codegen の rank ≥ 1 契約に合わせて長さ 1 の 1 次元に正規化する。
  const outShape = step.outputs[0].shape.length === 0 ? [1] : [...step.outputs[0].shape];
  const rank = outShape.length;
  const spec: ElementwiseSpec = element.op === CAST_OP
    ? { op: CAST_OP, rank, dtype: element.dtype, to: element.to }
    : { op: element.op, rank, dtype: element.dtype };
  const key = elementwiseKey(spec);
  const { pipeline, layout, roles } = await face.state.cache.get(key, elementwiseWgsl(spec));
  // attrs のスカラ（clamp の min/max など）は params の末尾に f32 で載る（並びは契約表）。
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
 * reduce 族（sum / amax / amin）。**縮約軸で 2 変種に踏み分ける**:
 *
 * - 最終次元 → 行 reduce（既存カーネル・1 行 = 1 workgroup + 256 幅ツリー）
 * - それ以外 → 軸 reduce（1 スレッド = 1 出力・コアレス読み。縮約順序は行 reduce と
 *   厳密に一致 = 出力ビット同一 — src/codegen/reduce.ts の {@link axisReduceWgsl}）
 *
 * MUST: 分岐は**軸だけ**で決める。速度で選ぶ余地を作ると、最終次元でも軸変種が走る形が
 * でき、既定経路のビット不変（PNG sha256 門）が実行時条件に依存してしまう。
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
  // 要素型はキーに載る（bool の sum は u32 を読んで i32 の個数を書く — ADR 0009）。
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
  // 行 reduce は 1 行 = 1 workgroup、軸 reduce は 1 スレッド = 1 出力。どちらも上限を
  // 超えたら縮退させ、カーネル側の grid-stride で回す。
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
    bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: outs[0] }],
    workgroups: [groups, 1, 1],
  });
};

/**
 * argmax（最終次元・rank 保存・出力 i32 — ADR 0068 決定 2）。短い行は**1 dispatch**で、行 reduce と
 * 同じ「1 行 = 1 workgroup + 行方向 grid-stride」（形とタイブレークの根拠は
 * src/kernels/argmax.ts）。長い行（`argmaxSplitGroups(dim) > 0` — 語彙長の lm_head 出口）は
 * **2 dispatch**（区間ごとの部分最大元 → 行ごとの merge）で、結果は 1 dispatch 形と
 * ビット同一（同ファイルの「2 相分割」節）。
 *
 * MUST: 軸で踏み分けない（reduce 族と違い最終次元専業 — 契約に `dim` の欄が無い）。
 * 行数は**入力の先行次元の積**から取る（出力の要素数と一致するが、カーネルが読む量は
 * 入力側の形で決まる）。
 * MUST: 経路の選択は形の純関数（`argmaxSplitGroups`）— 見積り（estimate.ts の argmax の
 * 一時）と同じ関数を通す。
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
  // 外側が空なら従来経路へ残す（topk(k=1) と同じ — 分割一時 0B はレシピの寿命契約に入らない）。
  if (rows > 0 && groups > 0) {
    await buildMaxIndexSplit(
      face,
      rows,
      dim,
      groups,
      binds[0],
      outs[0],
      builder,
    );
    return;
  }
  const { pipeline, layout, roles } = await face.state.cache.get(ARGMAX_KEY, ARGMAX_WGSL);
  const params = face.writeParams(argmaxParams(rows, dim), PARAMS_UNIFORM_USAGE);
  // 1 行 = 1 workgroup。上限を超えたら縮退させ、カーネル側の行 grid-stride で回す。
  builder.dispatch({
    key: ARGMAX_KEY,
    pipeline,
    layout,
    roles,
    params,
    bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: outs[0] }],
    workgroups: [gridStrideWorkgroups(rows, 1, limit), 1, 1],
  });
};

/**
 * 部分最大と最小添字の縮約を共有し、topk(k=1)だけ元入力の値ビットも写す。
 * DECIDED: docs/decisions/0068-decode-exit-multi-output.md#追記-92026-09-11-topk-k1-の長い行も-2-dispatch-へ分割する
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
  const params = face.writeParams(
    argmaxSplitParams(rows, dim, groups),
    PARAMS_UNIFORM_USAGE,
  );
  // 部分結果 [rows, groups, 2]（u32）。partial の直前に確保し merge の直後に返す。
  const partial = builder.allocTemp(argmaxSplitPartialBytes(rows, groups));
  builder.dispatch({
    key: ARGMAX_SPLIT_PARTIAL_KEY,
    pipeline: partialPipeline.pipeline,
    layout: partialPipeline.layout,
    roles: partialPipeline.roles,
    params,
    bindings: [{ binding: 1, source: input }, { binding: 2, source: partial }],
    // x 軸 = 区間（1 workgroup = 1 区間・欠落は沈黙誤値なので上限超過は fail loudly）、
    // y 軸 = 行（grid-stride で縮退可）。
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
        : [{ binding: 3, source: value }, { binding: 4, source: input }]),
    ],
    workgroups: [gridStrideWorkgroups(rows, 1, limit), 1, 1],
  });
  builder.releaseTemp(partial);
};

/**
 * topk（最終次元・static-k・**出力 2 本** — ADR 0068 決定 3）。
 * k=1 の長い行は argmax の部分最大・merge を共用し、それ以外は1 dispatch。
 * 値は選ばれた元入力のu32を写し、NaNのペイロードと符号付きゼロを維持する。
 *
 * MUST: 出力は**列で受ける**（`outs[0]` = 値 f32・`outs[1]` = 添字 i32）。順序は
 * {@link "../recipe.ts"} の `StepRecipe.outputs` と同じ出力 slot 昇順で、`node.outs` の並びがそのまま bind 面の
 * 1 dispatch経路の束縛番号 2 / 3 に対応する。取り違えると shape も byteLength も同じ（どちらも `[…, k]` の
 * 4 バイト要素）なので**例外なしに値と添字が入れ替わる**。
 * MUST: 分割経路の一時は実行・見積りの双方へ載せる。出力と同居させない。
 * 1 dispatch 経路の scratch は workgroup storage に閉じ、k の実装上限は
 * device limit から判定する（{@link assertTopkK} — 縮退しない）。
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
  // `1 ≤ k ≤ 最終次元` は契約層（attrs スキーマ + shape 規則）が済ませている。ここで見るのは
  // **device 依存の実装上限**だけ（workgroup storage — 上限値つきで fail loudly）。
  const k = topkK(step.node.attrs, where);
  assertTopkK(k, face.state.gpu.limits.maxComputeWorkgroupStorageSize, where);
  const splitGroups = topkOneSplitGroups(dim);
  // 外側が空なら従来経路へ残す。分割一時0Bはレシピの寿命契約に入らない。
  if (k === 1 && rows > 0 && splitGroups > 0) {
    await buildMaxIndexSplit(
      face,
      rows,
      dim,
      splitGroups,
      binds[0],
      outs[1],
      builder,
      outs[0],
    );
    return;
  }
  const key = topkKey(k);
  const { pipeline, layout, roles } = await face.state.cache.get(key, topkWgsl(k));
  const params = face.writeParams(topkParams(rows, dim), PARAMS_UNIFORM_USAGE);
  // 1 行 = 1 workgroup。上限を超えたら縮退させ、カーネル側の行 grid-stride で回す。
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
 * cumsum（最終次元の前縁和）。**1 invocation = 1 行**の逐次走査で、行方向を grid-stride で
 * 回す（形の根拠は src/kernels/cumsum.ts）。
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
    bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: outs[0] }],
    workgroups: [groups, 1, 1],
  });
};

/**
 * softmax（最終次元、safe-softmax）。layer_norm と同じ 1 行 = 1 workgroup の形。
 *
 * `safe` は safe_softmax 変種（行 max が −inf の行に 0 を書く — ADR 0044）。カーネルは
 * 同じ生成関数から出た 2 本で、dispatch の形（バインド・workgroup 数）は同じ。
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
    bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: outs[0] }],
    workgroups: [groups, 1, 1],
  });
};

/**
 * masked_fill。出力と x は同形・連続で、mask だけを右詰め broadcast の stride で読む。
 * stride の組み立ては strided 族の expand と同じ規則（{@link expandSrcStrides}）。
 */
export const buildMaskedFill = async (
  face: RecipeBuildFace,
  step: NodePlan,
  binds: readonly BindingSource[],
  outs: readonly BindingSource[],
  builder: StepRecipeBuilder,
): Promise<void> => {
  const outShape = step.outputs[0].shape;
  // 規則は expand と同一だが、診断の主語は masked_fill の側に付け替える（グラフに expand が
  // 無いのに「expand の入力」と出ると原因の当たりを外す）。
  const maskStrides = expandSrcStrides(step.inputShapes[1], outShape, {
    src: "masked_fill の mask",
    out: "masked_fill の出力",
  });
  const value = maskedFillValue(step.node.attrs, `nodes (${step.node.op})`);
  const { pipeline, layout, roles } = await face.state.cache.get(
    MASKED_FILL_KEY,
    MASKED_FILL_WGSL,
  );
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
