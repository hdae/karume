/**
 * Fusion rule `silu` (`sigmoid → mul`). The eligibility conditions shared by every rule and
 * `defineRule` are in {@link "../fusion-rule.ts"}; the declaration table that fixes the
 * application order is `FUSION_RULES` in {@link "../fusion.ts"}.
 */

import { numel } from "../../ops.ts";
import {
  SILU_WORKGROUP_SIZE,
  siluKey,
  type SiluMulOrder,
  siluParams,
  siluWgsl,
} from "../../kernels/silu.ts";
import { allF32, defineRule, type FusionMatch, sameShape } from "../fusion-rule.ts";

type SiluMatch = FusionMatch & {
  readonly xName: string;
  readonly outputName: string;
  readonly outputShape: readonly number[];
  readonly multiplyOrder: SiluMulOrder;
};

/**
 * SiLU: two consecutive nodes `sigmoid(x) → mul(x, sigmoid)`.
 *
 * MUST: the only consumer of the intermediate sigmoid value is the mul right after it, and it is
 * not a graph output.
 * MUST: every slot is f32 with the same shape only. Do not generalise to a broadcast SiLU or to a
 * gate with another input (widening the accepted set because "the formula looks similar" steps
 * outside the guarantee that the fallback is correct).
 *
 * Total uses of external inputs: x once each in sigmoid and mul = 2.
 */
export const SILU_RULE = defineRule<SiluMatch, "silu">({
  name: "silu",
  heads: ["sigmoid"],
  // ADR 0147 decision 2: every built-in fusion rule is class E (u32 equality with the unfused plan).
  contract: "E",
  internals: "private",
  match: (nodes, index) => {
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
    )
      return undefined;

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
      dispatches: [
        {
          key: siluKey(matched.multiplyOrder),
          wgsl: () => siluWgsl(matched.multiplyOrder),
          params: siluParams(count),
          workgroups: { kind: "gridStride", items: count, size: SILU_WORKGROUP_SIZE },
        },
      ],
    };
  },
});
