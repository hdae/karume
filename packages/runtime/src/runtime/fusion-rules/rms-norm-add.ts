/**
 * Fusion rule `rmsNormAdd` (`rms_norm → add` — ADR 0099). The eligibility conditions shared by
 * every rule and `defineRule` are in {@link "../fusion-rule.ts"}; the declaration table that fixes
 * the application order is `FUSION_RULES` in {@link "../fusion.ts"}.
 */

import { numel, rmsNormEps } from "../../ops.ts";
import {
  rmsNormAddKey,
  type RmsNormAddOrder,
  rmsNormAddWgsl,
  rmsNormParams,
} from "../../kernels/rms-norm.ts";
import { rmsNormSubgroupKey, rmsNormSubgroupWgsl } from "../../kernels/rms-norm-subgroup.ts";
import { allF32, defineRule, type FusionMatch, sameShape } from "../fusion-rule.ts";

type RmsNormAddMatch = FusionMatch & {
  readonly binds: readonly string[];
  readonly shape: readonly number[];
  readonly outputName: string;
  readonly eps: number;
  readonly order: RmsNormAddOrder;
  readonly subgroup: boolean;
};

/**
 * Shared, output and broadcast cases go back to the existing path. Limited to the measured last
 * dimensions (ADR 0099).
 */
export const RMS_NORM_ADD_RULE = defineRule<RmsNormAddMatch, "rmsNormAdd">({
  name: "rmsNormAdd",
  heads: ["rms_norm"],
  // ADR 0147 decision 2: every built-in fusion rule is class E (u32 equality with the unfused plan).
  contract: "E",
  internals: "private",
  match: (nodes, index, context) => {
    if (context.fuseRmsNormAdd !== true) return undefined;
    const norm = nodes[index],
      add = nodes[index + 1];
    if (norm?.node.op !== "rms_norm" || add?.node.op !== "add") return undefined;
    const chain = [norm, add],
      shape = norm.outputs[0].shape;
    if (!allF32(chain)) return undefined;
    if (
      shape.some((dim) => dim < 1) ||
      ![256, 1536, 2560, 2816, 3840].includes(shape[shape.length - 1])
    ) {
      return undefined;
    }
    const position = add.node.ins.indexOf(norm.outputs[0].name);
    if (
      position < 0 ||
      !sameShape(shape, norm.inputShapes[0]) ||
      add.inputShapes.some((input) => !sameShape(input, shape)) ||
      !sameShape(shape, add.outputs[0].shape)
    )
      return undefined;
    const binds = [...norm.node.ins, add.node.ins[1 - position]];
    // Shapes that need a duplicated binding are outside this acceptance. The shared release
    // bookkeeping derives the total references to external inputs.
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
    dispatches: [
      {
        key: matched.subgroup
          ? rmsNormSubgroupKey(matched.order)
          : rmsNormAddKey(matched.order, matched.shape[matched.shape.length - 1]),
        wgsl: () =>
          matched.subgroup
            ? rmsNormSubgroupWgsl(matched.order)
            : rmsNormAddWgsl(matched.order, matched.shape[matched.shape.length - 1]),
        params: rmsNormParams(
          numel(matched.shape.slice(0, -1)),
          matched.shape[matched.shape.length - 1],
          matched.eps,
        ),
        workgroups: { kind: "gridStride", items: numel(matched.shape.slice(0, -1)), size: 1 },
      },
    ],
  }),
});
