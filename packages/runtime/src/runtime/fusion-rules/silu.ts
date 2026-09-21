/**
 * 融合ルール `silu`（`sigmoid → mul`）。全ルール共通の適格条件と `defineRule` は
 * {@link "../fusion-rule.ts"}、適用順を決める宣言表は {@link "../fusion.ts"} の
 * `FUSION_RULES`。
 */

import { numel } from "../../ops.ts";
import {
  SILU_WORKGROUP_SIZE,
  siluKey,
  type SiluMulOrder,
  siluParams,
  siluWgsl,
} from "../../kernels/silu.ts";
import {
  allF32,
  defineRule,
  type FusionMatch,
  internalsArePrivate,
  sameShape,
} from "../fusion-rule.ts";

type SiluMatch = FusionMatch & {
  readonly xName: string;
  readonly outputName: string;
  readonly outputShape: readonly number[];
  readonly multiplyOrder: SiluMulOrder;
};

/**
 * SiLU: `sigmoid(x) → mul(x, sigmoid)` の連続 2 ノード。
 *
 * MUST: 中間 sigmoid 値は唯一の consumer が直後の mul で、graph output でないこと。
 * MUST: 全スロットが同 shape の f32 だけ。broadcast SiLU や別入力との gate へ一般化しない
 * （「式が似ている」で受理集合を広げると、fallback が正しいという保証の外へ出る）。
 *
 * 外部入力の延べ回数: x が sigmoid と mul で各 1 回 = 2 回。
 */
export const SILU_RULE = defineRule<SiluMatch>({
  name: "silu",
  heads: ["sigmoid"],
  match: (nodes, index, context) => {
    const sigmoid = nodes[index];
    if (sigmoid?.node.op !== "sigmoid") return undefined;
    const mul = nodes[index + 1];
    if (mul?.node.op !== "mul") return undefined;
    const chain = [sigmoid, mul];
    if (!allF32(chain)) return undefined;

    const xName = sigmoid.node.ins[0];
    const intermediateName = sigmoid.outputs[0].name;
    let multiplyOrder: SiluMulOrder;
    if (mul.node.ins[0] === xName && mul.node.ins[1] === intermediateName) {
      multiplyOrder = "x-sigmoid";
    } else if (mul.node.ins[0] === intermediateName && mul.node.ins[1] === xName) {
      multiplyOrder = "sigmoid-x";
    } else {
      return undefined;
    }

    const shape = sigmoid.inputShapes[0];
    if (shape.length < 1 || shape.some((dim) => dim < 1)) return undefined;
    if (
      !sameShape(sigmoid.outputs[0].shape, shape) ||
      mul.inputShapes.some((inputShape) => !sameShape(inputShape, shape)) ||
      !sameShape(mul.outputs[0].shape, shape)
    ) return undefined;
    if (!internalsArePrivate(chain, context)) return undefined;

    return {
      window: chain,
      chain,
      xName,
      outputName: mul.outputs[0].name,
      outputShape: mul.outputs[0].shape,
      multiplyOrder,
    };
  },
  build: (matched) => {
    const count = numel(matched.outputShape);
    return {
      binds: [matched.xName],
      outputName: matched.outputName,
      outputShape: matched.outputShape,
      temps: [],
      dispatches: [{
        key: siluKey(matched.multiplyOrder),
        wgsl: () => siluWgsl(matched.multiplyOrder),
        params: siluParams(count),
        workgroups: { kind: "gridStride", items: count, size: SILU_WORKGROUP_SIZE },
      }],
    };
  },
});
