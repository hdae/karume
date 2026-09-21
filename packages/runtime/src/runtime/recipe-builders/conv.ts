/**
 * 族別導出 — 畳み込みの族（conv1d / conv2d とその igemm 変種・conv_transpose1d・
 * deform_conv2d）と gru_scan。
 *
 * 入口は {@link "../recipe-builder.ts"} の `RecipeBuilder` で、共有サービス（Session の状態・
 * params の書き込み・重みの格納形と scale 束縛）は {@link RecipeBuildFace} 経由でだけ触る。
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
import type { NodePlan } from "../plan.ts";
import type { WeightStorage } from "../../kernels/weight-storage.ts";
import {
  conv1dAttrs,
  conv2dAttrs,
  convTranspose1dAttrs,
  deformConv2dAttrs,
  numel,
} from "../../ops.ts";
import { gemmMTileGeometry } from "../../kernels/gemm.ts";
import { gemmTileM, gemmTileN } from "../../kernels/gemm-geometry.ts";
import { gridStrideWorkgroups, tiledWorkgroups } from "../../codegen/dispatch.ts";
import { gruScanKey, gruScanParams, gruScanWgsl } from "../../kernels/gru-scan.ts";
import type { RecipeBuildFace } from "../recipe-builder.ts";
import { PARAMS_UNIFORM_USAGE } from "./params-usage.ts";

/**
 * deform_conv2d（DCNv2 — ADR 0055）。出力 1 要素 = 1 invocation の grid-stride 1 本だけで、
 * 踏み分けは無い（groups / offset_groups / stride / dilation は契約に欄が無い = 1 固定）。
 *
 * MUST: Kh / Kw は**重みの第 3 / 第 4 軸**をこの順で読む（conv2d と同じ教訓）。
 * offset / mask の形は契約検査が出力空間と突き合わせ済み。
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
 * gru_scan / gru_scan_reverse（GRU の隠れ側スキャン — ADR 0056）。
 *
 * 1 workgroup = 1 バッチ要素（バッチ方向だけ grid-stride で dispatch 上限を跨ぐ）で、
 * 時間ループはカーネル内。走査方向は **op 名**から引く（attrs に欄は無い）。
 *
 * MUST: 幾何は h0 と W_hh から取る（T は gi の先頭・N と H は h0）。gi の最終次元 3H から
 * H を割り戻すと、契約検査が突き合わせているはずの取り違えを 1 か所で作り直すことになる。
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
  const params = face.writeParams(
    gruScanParams({ time, batch, hidden }),
    PARAMS_UNIFORM_USAGE,
  );
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
 * conv1d（groups / dilation は attrs）。**groups で 2 カーネルを踏み分ける**（conv2d と同型）:
 * `groups == 1` は implicit GEMM、`groups > 1` は直接畳み込み（1 スレッド = 1 出力要素の
 * grid-stride）。
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
 * conv1d の implicit GEMM（`C[Cout, N] = W[Cout, K] × Xcol[K, N]` — ADR 0024 の 1D 版）。
 *
 * MUST: GEMM 骨格と同じ「1 workgroup = 1 出力タイル」なので、dispatch 上限超過は
 * fail loudly（grid-stride で縮退させるとタイルが欠落し、full-write が黙って壊れる）。
 * MUST: バッチは **z 軸**（bmm / conv2d と同じ）。N 側へ畳むと出力が `[Cout][B·Lout]` に
 * なって NCL と軸が入れ替わる — B = 1 でだけ一致するので実測形では露見しない。
 *
 * m タイルの述語は conv2d と**同じ 1 本**（{@link conv2dIgemmMTile} — M = Cout の関数で
 * しかないので次元に依らない）。どちらのタイル形でも出力はビット同一なので、これは純粋な
 * dispatch の割り直しで数値契約に触れない。
 *
 * NOTE: i4 常駐の conv1d が着地するのは**この経路だけ**（適格判定が groups == 1 に閉じる —
 * ADR 0069 決定 5 の conv1d 追補）。group 長は WGSL に焼くのでキーにも乗せる。
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
  // 生成・キーと同じ解決点から幾何を引く（dispatch のタイル辺が WGSL の辺と構造的に一致）。
  const geometry = gemmMTileGeometry(mTile);
  // i4 は group 長を WGSL に焼く（キーの g 部と対 — linear / embedding と同じ規律・ADR 0069）。
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
 * conv2d（stride / padding / dilation は H/W の 2 成分）。**groups で 2 カーネルを踏み分ける**
 * （ADR 0024）: `groups == 1` は implicit GEMM、`groups > 1` は直接畳み込み。
 *
 * MUST: Kh / Kw は**重みの第 3 / 第 4 軸**をこの順で読む。入れ替えても正方カーネルでは
 * 数値が一致するので、テストは Kh ≠ Kw の形で固定する（src/kernels/conv2d.ts）。
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
 * conv2d の implicit GEMM（`C[Cout, N] = W[Cout, K] × Xcol[K, N]` — ADR 0024）。
 *
 * MUST: GEMM 骨格と同じ「1 workgroup = 1 出力タイル」なので、dispatch 上限超過は
 * fail loudly（grid-stride で縮退させるとタイルが欠落し、full-write が黙って壊れる）。
 * MUST: バッチは **z 軸**（bmm と同じ）。N 側へ畳むと出力が `[Cout][B·Hout·Wout]` になり
 * NCHW と軸が入れ替わる — B = 1 でだけ一致するので実測形では露見しない。
 *
 * m タイルは形状の関数（{@link conv2dIgemmMTile}）で 64 行 / 32 行を選ぶ（ADR 0024 隣接）。
 * **n タイルは常に 64**（2048px の n 上限超過の扱いを動かさない）。どちらのタイル形でも
 * 出力はビット同一なので、これは純粋な dispatch の割り直しで数値契約に触れない。
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
  // 生成・キーと同じ解決点から幾何を引く（dispatch のタイル辺が WGSL の辺と構造的に一致）。
  const geometry = gemmMTileGeometry(mTile);
  const key = conv2dIgemmKey(weightStorage, v4, mTile);
  const { pipeline, layout, roles } = await face.state.cache.get(
    key,
    conv2dIgemmWgsl(weightStorage, v4, mTile),
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
 * conv_transpose1d（gather 形）。1 スレッド = 1 出力要素で出力全域を書く（ADR 0014）。
 *
 * MUST: Cin は**重みの第 1 軸**（`[Cin, Cout, K]`）。x[1] と一致することは契約検査済みだが、
 * ここで weight[0] を使うのは「転置レイアウトの正本は重み側」という読みを崩さないため。
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
