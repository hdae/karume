/**
 * 族別導出 — 要素の並べ替え族（strided copy = permute / expand / slice / sym_prefix_slice・
 * cat・pad・flip・gather・upsample_bilinear2d）。
 *
 * 入口は {@link "../recipe-builder.ts"} の `RecipeBuilder` で、共有サービス（Session の状態と
 * params の書き込み）は {@link RecipeBuildFace} 経由でだけ触る。
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
 * permute / 非恒等 expand / slice / sym_prefix_slice の実体化コピー（strided 読み 1 カーネル族 —
 * ADR 0011 / 0010 / 0014）。出力は常に連続で、入力側だけを stride で読む。expand の複製軸は
 * stride 0、sym_prefix_slice は **Tmax 形の入力**の連続 stride、slice は入力の連続 stride と
 * **開始位置 offset**。恒等 expand は別名化されるのでここへは来ない。
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
  const srcStrides = kind === "permute"
    ? permuteSrcStrides(srcShape, permuteDims(step.node.attrs, where))
    : kind === "expand"
    ? expandSrcStrides(srcShape, outShape)
    : kind === "slice"
    ? sliceSrcStrides(srcShape)
    // MUST: 束縛後の outShape ではなく **srcShape（Tmax 形）** から組む（prefixSliceSrcStrides
    // の MUST — 送り幅を縮めると 2 行目以降が別の行を読む）。
    : prefixSliceSrcStrides(srcShape);
  // offset は ADR 0011 の (offset, strides[4]) モデルそのもの。permute / expand /
  // sym_prefix_slice は入力の先頭から読むので 0 で、**slice だけが 0 以外**を取る
  // （ADR 0011 が予告した「可変点 1 語」がここ）。
  let offset = 0;
  if (kind === "slice") {
    const { dim, start } = sliceAttrs(step.node.attrs, where);
    offset = sliceSrcOffset(srcShape, dim, start);
  }
  // 要素型はキーに載る（bool マスク / i32 添字の expand が実測にある — ADR 0009）。
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
    bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: outs[0] }],
    workgroups: [groups, 1, 1],
  });
};

/**
 * cat（strided 書きコピー族 — ADR 0014）。入力ごとに **1 dispatch** で出力の部分領域へ書く。
 * 入力は連続で読み、出力へ `(offset, 出力の連続 strides)` で書く（読み族の双対）。
 *
 * MUST: 書き出し位置の総和が連結軸の長さとちょうど一致することを確かめる（full-write）。
 * 一致しなければ出力のどこかが**未書き込みのまま**残り、配り直しなら前の値がそのまま
 * 結果になる。契約の shape 規則（軸長 = 入力の軸長の総和）と同じ事実をエンコード側の
 * 積み上げからも確かめる形で、片方だけの誤りを内部矛盾として落とす。
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
      bindings: [{ binding: 1, source }, { binding: 2, source: outs[0] }],
      workgroups: [groups, 1, 1],
    });
    written += srcShape[dim];
  }
  if (written !== outShape[dim]) {
    throw new ExecutionError(
      `${where}: cat の書き込みが出力全域を覆わない（軸 ${dim} に ${written} / 出力は ${
        outShape[dim]
      }）`,
    );
  }
};

/**
 * pad（最終次元の定数 0 埋め）。**1 dispatch で出力の全バイトを書く**（範囲内は転写・
 * 範囲外は 0 — ADR 0014 の full-write）。ゼロ初期化されたバッファを前提にしない。
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
    bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: outs[0] }],
    workgroups: [groups, 1, 1],
  });
};

/**
 * flip（静的軸の添字反転）。軸の位置は `[outer, len, inner]` の 3 分割に畳んで渡す
 * （rank を params に載せない — src/kernels/flip.ts）。
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
    // MUST: 軸の前後で分ける（`slice(0, dim)` と `slice(dim + 1)`）。境界を 1 つずらすと
    // 反転する軸が隣にずれ、shape も要素数も変わらないまま値だけが誤る。
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
    bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: outs[0] }],
    workgroups: [groups, 1, 1],
  });
};

/**
 * upsample_bilinear2d（NCHW の空間 2 軸を双線形 resample・`align_corners = True` 専業）。
 * 出力 1 要素 = 1 invocation の grid-stride で、params は空間 4 長だけ運ぶ
 * （N·C はカーネル側が平面添字へ畳む — src/kernels/upsample-bilinear2d.ts）。
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
    upsampleBilinear2dParams(
      numel(outShape),
      srcShape[2],
      srcShape[3],
      outShape[2],
      outShape[3],
    ),
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
    bindings: [{ binding: 1, source: binds[0] }, { binding: 2, source: outs[0] }],
    workgroups: [groups, 1, 1],
  });
};

/**
 * 最終次元の gather。出力は連続で、`row = i / J` から `src[row * D + index[i]]` を引く。
 * 範囲外添字の扱いは src/kernels/gather.ts の裁定（GPU は NaN 汚染 / CPU 参照は throw）。
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
