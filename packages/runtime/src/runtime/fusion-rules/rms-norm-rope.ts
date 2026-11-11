/**
 * Fusion rule `rmsNormRope` (the Wan DiT's self-attention q / k prologue — perf-ledger F-3, wave a
 * stage 3): `rms_norm` over the whole token row, two reshape aliases, then the pair-form RoPE,
 * folded into one dispatch of {@link "../../kernels/rms-norm-rope.ts"} that never stores the
 * normalised row. The eligibility conditions shared by every rule and `defineRule` are in
 * {@link "../fusion-rule.ts"}; the declaration table that fixes the application order is
 * `FUSION_RULES` in {@link "../fusion.ts"}.
 */

import { RESHAPE_OP, RMS_NORM_OP, rmsNormEps } from "../../ops.ts";
import { rmsNormKernel } from "../../kernels/rms-norm.ts";
import {
  RMS_NORM_ROPE_KEY,
  RMS_NORM_ROPE_WGSL,
  rmsNormRopeParams,
} from "../../kernels/rms-norm-rope.ts";
import { allF32, defineRule, type FusionMatch, sameShape } from "../fusion-rule.ts";
import type { NodePlan } from "../plan.ts";
import { matchRopePairForm } from "./rope.ts";

type RmsNormRopeMatch = FusionMatch & {
  /** x (the rms_norm input), weight, cos, sin — the kernel's bindings 1–4. */
  readonly binds: readonly string[];
  readonly outputName: string;
  readonly outputShape: readonly number[];
  readonly rows: number;
  readonly dim: number;
  readonly eps: number;
  readonly headDim: number;
};

/**
 * The ops the exporter interleaves with one chain in the measured graphs: the other projections'
 * `rms_norm` and `reshape`. Nothing else may sit between the chain's nodes.
 */
const INTERLEAVED_OPS: ReadonlySet<string> = new Set([RMS_NORM_OP, RESHAPE_OP]);

/**
 * The index of the first node from `from` on that reads `name`, when every node before it is of
 * {@link INTERLEAVED_OPS}; otherwise `undefined`.
 */
const nextReader = (nodes: readonly NodePlan[], from: number, name: string): number | undefined => {
  for (let at = from; at < nodes.length; at += 1) {
    const step = nodes[at];
    if (step.node.ins.includes(name)) return at;
    if (!INTERLEAVED_OPS.has(step.node.op)) return undefined;
  }
  return undefined;
};

/**
 * The exact form (ADR 0040 decision 2), from the scan index:
 *
 * ```
 * rms_norm(x, weight)   -> n      [1,S,H·D]       (x [1,S,H·D]: one token row = one norm row)
 *   … rms_norm / reshape nodes of the other projections (deferred) …
 * reshape(n)            -> v      [1,S,H,D]
 *   … (deferred) …
 * reshape(v)            -> pairs  [1,S,H·D/2,2]
 * the 8 nodes of the pair-form RoPE on x = v and pairs, directly after (`matchRopePairForm`)
 * ```
 *
 * - The RoPE window is exactly what the rope rule accepts for the pair / BSHD form (the same
 *   matcher), and `H·D` is the norm's row (D even — the matcher's; a pair never straddles a head).
 * - The rms_norm is last-dim and weight-only by its contract (ADR 0017); x is `[1,S,H·D]` f32. It
 *   folds only where the plain path runs the 256-wide workgroup rms_norm ({@link rmsNormKernel} —
 *   the decision `buildRmsNorm` makes), the kernel the fused one shares its reduction with and is
 *   bit-identical to: a row of at most 128 (the 128-wide form) or `rmsNormReduce: "subgroup32"`
 *   (another summation order) leaves the nodes, and the RoPE then folds alone (rule `rope`).
 * - Every use of the internal values is inside the chain (the declared `internals: "stayInChain"`,
 *   which `defineRule` applies): `n` once (the first reshape), `v` twice (the pair reshape and the
 *   RoPE's direct `mul`), `pairs` twice (the two slices), the RoPE's own values once each (the shared
 *   pair-form matcher checks that one use itself). The matcher pins every slot they occupy, and the
 *   bind surface holds external values only (x, weight, cos, sin — no duplicates).
 * - The nodes between the chain's are only {@link INTERLEAVED_OPS}, and they are **deferred**
 *   ({@link FusionMatch.others}): in the Wan IR the q and k chains interleave (`rms_norm` q,
 *   `rms_norm` k, the reshapes of q / k / v, the q pair reshape and RoPE, the k pair reshape and
 *   RoPE), so q's window holds k's `rms_norm` and reshape. As in-window passthroughs they would be
 *   emitted as plain nodes and k could never fold; deferred, the scan meets k's `rms_norm` again
 *   behind q's fused step, with only v's reshape between k's nodes (deferred once more).
 *
 * NOTE: the head `rms_norm` is shared with rmsNormAdd, and the two never accept the same index: that
 * rule needs the normalised value's single use to be the next node, an `add`; this one needs its
 * every use inside this chain, which reads it only in the first reshape.
 *
 * Total uses of external inputs: x / weight / cos / sin once each.
 */
export const RMS_NORM_ROPE_RULE = defineRule<RmsNormRopeMatch, "rmsNormRope">({
  name: "rmsNormRope",
  heads: [RMS_NORM_OP],
  // ADR 0147 decision 2: every built-in fusion rule is class E (u32 equality with the unfused plan).
  contract: "E",
  internals: "stayInChain",
  match: (nodes, index, context) => {
    const norm = nodes[index];
    if (norm?.node.op !== RMS_NORM_OP) return undefined;
    const viewAt = nextReader(nodes, index + 1, norm.outputs[0].name);
    const view = viewAt === undefined ? undefined : nodes[viewAt];
    if (viewAt === undefined || view?.node.op !== RESHAPE_OP) return undefined;
    const pairsAt = nextReader(nodes, viewAt + 1, view.outputs[0].name);
    const pairs = pairsAt === undefined ? undefined : nodes[pairsAt];
    if (pairsAt === undefined || pairs?.node.op !== RESHAPE_OP) return undefined;
    const rope = matchRopePairForm(nodes, pairsAt + 1, context);
    if (rope === undefined) return undefined;
    const [x, pairsName, cos, sin] = rope.binds;
    if (x !== view.outputs[0].name || pairsName !== pairs.outputs[0].name) return undefined;

    const [, sequence, heads, headDim] = rope.outputShape;
    const dim = heads * headDim;
    const rowShape = [1, sequence, dim];
    if (
      !sameShape(norm.inputShapes[0], rowShape) ||
      !sameShape(norm.outputs[0].shape, rowShape) ||
      !sameShape(view.outputs[0].shape, rope.outputShape)
    )
      return undefined;
    if (rmsNormKernel(dim, context.rmsNormReduce) !== "workgroup256") return undefined;

    // MUST: the actual node order (the release bookkeeping derives the total-use list from it).
    const chain = [norm, view, pairs, ...rope.chain];
    if (!allF32([norm, view, pairs])) return undefined;
    const binds = [norm.node.ins[0], norm.node.ins[1], cos, sin];
    if (new Set(binds).size !== binds.length) return undefined;

    const end = pairsAt + 1 + rope.window.length;
    return {
      window: nodes.slice(index, end),
      chain,
      others: "deferred",
      binds,
      outputName: rope.outputName,
      outputShape: rope.outputShape,
      rows: sequence,
      dim,
      eps: rmsNormEps(norm.node.attrs, "rms_norm → RoPE fusion"),
      headDim,
    };
  },
  build: (matched) => ({
    binds: matched.binds,
    outputName: matched.outputName,
    outputShape: matched.outputShape,
    temps: [],
    dispatches: [
      {
        key: RMS_NORM_ROPE_KEY,
        wgsl: () => RMS_NORM_ROPE_WGSL,
        params: rmsNormRopeParams(matched.rows, matched.dim, matched.eps, matched.headDim),
        // One workgroup per token row, grid-stride over the rows (the plain rms_norm's geometry).
        workgroups: { kind: "gridStride", items: matched.rows, size: 1 },
      },
    ],
  }),
});
