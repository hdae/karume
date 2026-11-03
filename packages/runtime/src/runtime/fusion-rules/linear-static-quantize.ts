/**
 * Fusion rule `linearStaticQuantize` (`linear → static_quantize` into one parallel GEMV
 * dispatch — or, for an integer GEMV linear whose activation arrives packed, one integer GEMV
 * dispatch with the SRQ epilogue). The eligibility conditions shared by every rule and `defineRule`
 * are in {@link "../fusion-rule.ts"}; the declaration table that fixes the application order is
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
import {
  integerGemvKey,
  integerGemvParams,
  type IntegerGemvVariant,
  integerGemvWgsl,
  integerGemvWorkgroups,
} from "../../kernels/linear-gemv-integer.ts";
import type { WeightStorage } from "../../kernels/weight-storage.ts";
import {
  allF32,
  defineRule,
  type FusedDispatch,
  type FusionMatch,
  internalsArePrivate,
  sameShape,
} from "../fusion-rule.ts";
import { linearIntegerGemvVariant, linearPlanesGemvVariant } from "../linear-route.ts";

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
  /**
   * The integer GEMV variant when the input arrives packed and the linear is an integer GEMV linear
   * (`linearIntegerGemvVariant`), else `undefined` (the parallel GEMV).
   */
  readonly integer: IntegerGemvVariant | undefined;
  readonly workgroups: readonly [number, number, number];
};

/** The one dispatch of a match: the integer GEMV, the packed parallel GEMV or the parallel GEMV. */
const dispatchOf = (
  m: LinearStaticQuantizeMatch,
): Omit<FusedDispatch, "operands" | "workgroups"> => {
  if (m.packedX !== undefined && m.integer !== undefined) {
    const variant = m.integer;
    return {
      key: integerGemvKey(variant, true),
      wgsl: () => integerGemvWgsl(variant, true),
      params: integerGemvParams(variant, m.m, m.n, m.k, m.packedX, m.scale),
    };
  }
  if (m.packedX !== undefined) {
    return {
      key: linearGemvStaticQuantizePackedKey(m.storage, m.group, m.lanes),
      wgsl: () => linearGemvStaticQuantizePackedWgsl(m.storage, m.group, m.lanes),
      params: linearGemvStaticQuantizePackedParams(
        m.storage,
        m.m,
        m.n,
        m.k,
        m.packedX,
        m.scale,
        m.group,
      ),
    };
  }
  return {
    key: linearGemvStaticQuantizeKey(m.storage, m.group, m.lanes),
    wgsl: () => linearGemvStaticQuantizeWgsl(m.storage, m.group, m.lanes),
    params: linearGemvStaticQuantizeParams(m.storage, m.m, m.n, m.k, m.scale, m.group),
  };
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
    // A byte-plane GEMV linear (`linearPlanesGemvVariant` — the function buildLinear reads) keeps
    // its own dispatch and the SRQ stays a node: taking the window would build the parallel GEMV in
    // its place and leave the option silently without effect. The decision reads only the option
    // and the linear, never the packed values, so both scans of planFusions still grab the same
    // windows. No measured parallel row is a byte-plane shape today, so this rejects no window the
    // rule would otherwise take.
    if (linearPlanesGemvVariant(context.linearGemvPlanes, linear, weight.storage) !== undefined) {
      return undefined;
    }
    const m = numel(linear.inputShapes[0].slice(0, -1));
    const [n, k] = linear.inputShapes[1];
    const group = weight.storage === "i4" ? weight.groupSize : undefined;
    const lanes = linearGemvParallelEligible(weight.storage, m, n, k, group);
    if (lanes === undefined) return undefined;
    // The activation side arrives as packed int8 (ADR 0105). Acceptance is settled in one
    // place, planFusions; this only reads whether it arrives (the output-side SRQ stays f32).
    const packedX = context.packedValues.get(linear.node.ins[0]);
    // MUST: whether the integer kernel takes the window enters no acceptance condition above (only
    // the dispatch the build makes) — the two scans of planFusions grab the same windows.
    const integer =
      packedX === undefined
        ? undefined
        : linearIntegerGemvVariant(context.integerGemv, linear, weight.storage, group);
    // The tiled form does not fall back to grid-stride; it is rejected at the same device limit
    // as a plain linear.
    const workgroups: readonly [number, number, number] = [
      integer === undefined
        ? tiledWorkgroups(
            n,
            128 / lanes,
            context.limits.maxComputeWorkgroupsPerDimension,
            "linear static_quantize N",
          )
        : tiledWorkgroups(
            integerGemvWorkgroups(n, integer),
            1,
            context.limits.maxComputeWorkgroupsPerDimension,
            "linear gemv integer static_quantize N",
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
      packedX,
      integer,
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
        ...dispatchOf(m),
        // The integer GEMV keeps the linear's binding numbering (src/kernels/linear-gemv-integer.ts).
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
