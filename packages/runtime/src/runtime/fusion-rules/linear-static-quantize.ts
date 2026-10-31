/**
 * Fusion rule `linearStaticQuantize` (`linear → static_quantize` into one parallel GEMV
 * dispatch). The eligibility conditions shared by every rule and `defineRule` are in
 * {@link "../fusion-rule.ts"}; the declaration table that fixes the application order is
 * `FUSION_RULES` in {@link "../fusion.ts"}.
 */

import { numel, staticQuantizeScale } from "../../ops.ts";
import { tiledWorkgroups } from "../../codegen/dispatch.ts";
import {
  linearGemvParallelEligible,
  type LinearGemvParallelLanes,
  linearGemvStaticQuantizeKey,
  linearGemvStaticQuantizePackedKey,
  linearGemvStaticQuantizePackedParams,
  linearGemvStaticQuantizePackedWgsl,
  linearGemvStaticQuantizeParams,
  linearGemvStaticQuantizeWgsl,
} from "../../kernels/linear-gemv.ts";
import type { WeightStorage } from "../../kernels/weight-storage.ts";
import {
  allF32,
  defineRule,
  type FusionMatch,
  internalsArePrivate,
  sameShape,
} from "../fusion-rule.ts";

type LinearStaticQuantizeMatch = FusionMatch & {
  readonly binds: readonly string[];
  readonly shape: readonly number[];
  readonly outputName: string;
  readonly storage: WeightStorage;
  readonly group: number | undefined;
  readonly lanes: LinearGemvParallelLanes;
  readonly m: number;
  readonly n: number;
  readonly k: number;
  readonly scale: number;
  /** The producer's SRQ scale when the input arrives as packed int8 (else `undefined`). */
  readonly packedX: number | undefined;
  readonly workgroups: readonly [number, number, number];
};

// DECIDED: docs/decisions/0103-linear-static-quantize-fusion.md
export const LINEAR_STATIC_QUANTIZE_RULE = defineRule<LinearStaticQuantizeMatch>({
  name: "linearStaticQuantize",
  heads: ["linear"],
  match: (nodes, index, context) => {
    if (
      context.fuseLinearStaticQuantize !== true ||
      context.linearGemvReduce !== "parallel" ||
      context.linearCompute !== "f32"
    )
      return undefined;
    const linear = nodes[index],
      srq = nodes[index + 1];
    if (linear?.node.op !== "linear" || srq?.node.op !== "static_quantize") return undefined;
    const chain = [linear, srq];
    if (!allF32(chain) || !internalsArePrivate(chain, context)) return undefined;
    if (
      linear.node.ins.length !== 3 ||
      new Set(linear.node.ins).size !== 3 ||
      srq.node.ins[0] !== linear.outputs[0].name ||
      !sameShape(linear.outputs[0].shape, srq.outputs[0].shape)
    )
      return undefined;
    const weight = context.weightLayouts?.get(linear.node.ins[1]);
    if (weight === undefined || weight.storage === "f16" || weight.storage === "bf16") {
      return undefined;
    }
    const m = numel(linear.inputShapes[0].slice(0, -1));
    const [n, k] = linear.inputShapes[1];
    const group = weight.storage === "i4" ? weight.groupSize : undefined;
    const lanes = linearGemvParallelEligible(weight.storage, m, n, k, group);
    if (lanes === undefined) return undefined;
    // The tiled form does not fall back to grid-stride; it is rejected at the same device limit
    // as a plain linear.
    const workgroups: readonly [number, number, number] = [
      tiledWorkgroups(
        n,
        128 / lanes,
        context.limits.maxComputeWorkgroupsPerDimension,
        "linear static_quantize N",
      ),
      tiledWorkgroups(
        m,
        1,
        context.limits.maxComputeWorkgroupsPerDimension,
        "linear static_quantize M",
      ),
      1,
    ];
    return {
      window: chain,
      chain,
      binds: linear.node.ins,
      shape: srq.outputs[0].shape,
      outputName: srq.outputs[0].name,
      storage: weight.storage,
      group,
      lanes,
      m,
      n,
      k,
      scale: staticQuantizeScale(srq.node.attrs, "linear static_quantize fusion"),
      // The activation side arrives as packed int8 (ADR 0105). Acceptance is settled in one
      // place, planFusions; this only reads whether it arrives (the output-side SRQ stays f32).
      packedX: context.packedValues.get(linear.node.ins[0]),
      workgroups,
    };
  },
  build: (m) => ({
    binds: m.binds,
    outputName: m.outputName,
    outputShape: m.shape,
    temps: [],
    dispatches: [
      {
        key:
          m.packedX === undefined
            ? linearGemvStaticQuantizeKey(m.storage, m.group, m.lanes)
            : linearGemvStaticQuantizePackedKey(m.storage, m.group, m.lanes),
        wgsl: () =>
          m.packedX === undefined
            ? linearGemvStaticQuantizeWgsl(m.storage, m.group, m.lanes)
            : linearGemvStaticQuantizePackedWgsl(m.storage, m.group, m.lanes),
        params:
          m.packedX === undefined
            ? linearGemvStaticQuantizeParams(m.storage, m.m, m.n, m.k, m.scale, m.group)
            : linearGemvStaticQuantizePackedParams(
                m.storage,
                m.m,
                m.n,
                m.k,
                m.packedX,
                m.scale,
                m.group,
              ),
        operands: [
          { kind: "bind", index: 0 },
          { kind: "bind", index: 1 },
          { kind: "bind", index: 2 },
          { kind: "output" },
          { kind: "weightScale", index: 1 },
        ],
        workgroups: { kind: "tiled", counts: m.workgroups },
      },
    ],
  }),
});
