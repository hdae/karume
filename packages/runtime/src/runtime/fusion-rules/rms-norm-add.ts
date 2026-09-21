/**
 * 融合ルール `rmsNormAdd`（`rms_norm → add` — ADR 0099）。全ルール共通の適格条件と
 * `defineRule` は {@link "../fusion-rule.ts"}、適用順を決める宣言表は
 * {@link "../fusion.ts"} の `FUSION_RULES`。
 */

import { numel, rmsNormEps } from "../../ops.ts";
import {
  rmsNormAddKey,
  type RmsNormAddOrder,
  rmsNormAddWgsl,
  rmsNormParams,
} from "../../kernels/rms-norm.ts";
import { rmsNormSubgroupKey, rmsNormSubgroupWgsl } from "../../kernels/rms-norm-subgroup.ts";
import {
  allF32,
  defineRule,
  type FusionMatch,
  internalsArePrivate,
  sameShape,
} from "../fusion-rule.ts";

type RmsNormAddMatch = FusionMatch & {
  readonly binds: readonly string[];
  readonly shape: readonly number[];
  readonly outputName: string;
  readonly eps: number;
  readonly order: RmsNormAddOrder;
  readonly subgroup: boolean;
};

/** 共有・出力・broadcastは既存経路へ戻す。実測した最終次元に限定する（ADR 0099）。 */
export const RMS_NORM_ADD_RULE = defineRule<RmsNormAddMatch>({
  name: "rmsNormAdd",
  heads: ["rms_norm"],
  match: (nodes, index, context) => {
    if (context.fuseRmsNormAdd !== true) return undefined;
    const norm = nodes[index], add = nodes[index + 1];
    if (norm?.node.op !== "rms_norm" || add?.node.op !== "add") return undefined;
    const chain = [norm, add], shape = norm.outputs[0].shape;
    if (!allF32(chain) || !internalsArePrivate(chain, context)) return undefined;
    if (shape.some((dim) => dim < 1) || ![256, 1536, 2560].includes(shape[shape.length - 1])) {
      return undefined;
    }
    const position = add.node.ins.indexOf(norm.outputs[0].name);
    if (
      position < 0 || !sameShape(shape, norm.inputShapes[0]) ||
      add.inputShapes.some((input) => !sameShape(input, shape)) ||
      !sameShape(shape, add.outputs[0].shape)
    ) return undefined;
    const binds = [...norm.node.ins, add.node.ins[1 - position]];
    // 重複bindingを要求する形は今回の検収外。外部入力の延べ参照は共通の解放簿記が導く。
    if (new Set(binds).size !== 3) return undefined;
    return {
      window: chain,
      chain,
      binds,
      shape,
      outputName: add.outputs[0].name,
      eps: rmsNormEps(norm.node.attrs, "rms_norm fusion"),
      order: position === 0 ? "norm-residual" : "residual-norm",
      subgroup: context.rmsNormReduce === "subgroup32",
    };
  },
  build: (matched) => ({
    binds: matched.binds,
    outputName: matched.outputName,
    outputShape: matched.shape,
    temps: [],
    dispatches: [{
      key: matched.subgroup ? rmsNormSubgroupKey(matched.order) : rmsNormAddKey(matched.order),
      wgsl: () =>
        matched.subgroup ? rmsNormSubgroupWgsl(matched.order) : rmsNormAddWgsl(matched.order),
      params: rmsNormParams(
        numel(matched.shape.slice(0, -1)),
        matched.shape[matched.shape.length - 1],
        matched.eps,
      ),
      workgroups: { kind: "gridStride", items: numel(matched.shape.slice(0, -1)), size: 1 },
    }],
  }),
});
