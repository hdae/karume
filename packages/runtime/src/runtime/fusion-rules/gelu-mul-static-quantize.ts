/**
 * Fusion rule `geluMulStaticQuantize` (`gelu_tanh → mul → static_quantize` into one dispatch —
 * perf-ledger K-48 step 1b). The eligibility conditions shared by every rule and `defineRule` are
 * in {@link "../fusion-rule.ts"}; the declaration table that fixes the application order is
 * `FUSION_RULES` in {@link "../fusion.ts"}.
 */

import { numel, STATIC_QUANTIZE_OP, staticQuantizeScale } from "../../ops.ts";
import {
  GELU_MUL_STATIC_QUANTIZE_WORKGROUP_SIZE,
  geluMulStaticQuantizeKey,
  type GeluMulOrder,
  geluMulStaticQuantizeParams,
  geluMulStaticQuantizeWgsl,
} from "../../kernels/gelu-mul-static-quantize.ts";
import { allF32, defineRule, type FusionMatch, sameShape } from "../fusion-rule.ts";

/**
 * The measured form (perf-ledger K-48 step 1b): Gemma 4 QAT's per-layer-input gate,
 * `[1, M, 256]` with M ≤ 8 rows (decode and speculative verification). Identity holds for any
 * shape (one element per thread), but the gain was measured on this one only (ADR 0040 decision
 * 2 — acceptance is fixed to the measured shapes); a prefill plan stays on the plain chain.
 */
export const GELU_MUL_STATIC_QUANTIZE_DIM = 256;
export const GELU_MUL_STATIC_QUANTIZE_MAX_ROWS = 8;

type GeluMulStaticQuantizeMatch = FusionMatch & {
  readonly gate: string;
  readonly other: string;
  readonly order: GeluMulOrder;
  readonly outputName: string;
  readonly outputShape: readonly number[];
  readonly count: number;
  readonly scale: number;
};

/**
 * Three consecutive nodes `gelu_tanh(gate) → mul(gelu, other) → static_quantize(product)` (either
 * operand order of the mul).
 *
 * Accepted only on an exact match (ADR 0040 decision 2):
 *
 * 1. The session option (`fuseGeluMulStaticQuantize`) is set.
 * 2. The gelu value's and the product's single consumers are the next node, and neither is a graph
 *    output (the declared `internals: "private"`, which `defineRule` applies).
 * 3. Every slot is f32, and the gate, the gelu value, the mul's other operand, the product and the
 *    SRQ output have one shape: `[…, 256]` with 1 to 8 rows (no broadcast). The kernel reads the
 *    other operand as `other[i]`, the element order the plain mul reads a same-shape operand in.
 *    An alias the planner makes (a reshape, a permute that keeps the element order, an identity
 *    expand — ADR 0011) is the source's buffer in that same order (the QAT operand is a reshape of
 *    a materialised slice), so any same-shape value reads correctly.
 * 4. The other operand is not the gate (each binding is one value — a duplicate bind is not the
 *    measured form).
 * 5. The SRQ output is not handed over as packed int8 (ADR 0105): the kernel writes the f32 path's
 *    words. NOTE: `planFusions` packs only SRQs left as plain nodes by the first scan, so a folded
 *    SRQ is never packed; the check keeps the invariant local to the rule.
 *
 * Total uses of external inputs: the gate in gelu_tanh, the other operand in mul = 2.
 */
export const GELU_MUL_STATIC_QUANTIZE_RULE = defineRule<
  GeluMulStaticQuantizeMatch,
  "geluMulStaticQuantize"
>({
  name: "geluMulStaticQuantize",
  heads: ["gelu_tanh"],
  internals: "private",
  match: (nodes, index, context) => {
    if (context.fuseGeluMulStaticQuantize !== true) return undefined;
    const gelu = nodes[index],
      mul = nodes[index + 1],
      srq = nodes[index + 2];
    if (
      gelu?.node.op !== "gelu_tanh" ||
      mul?.node.op !== "mul" ||
      srq?.node.op !== STATIC_QUANTIZE_OP
    )
      return undefined;
    const chain = [gelu, mul, srq];
    if (!allF32(chain)) return undefined;
    if (gelu.node.ins.length !== 1 || mul.node.ins.length !== 2 || srq.node.ins.length !== 1) {
      return undefined;
    }
    const act = gelu.outputs[0].name;
    const order: GeluMulOrder | undefined =
      mul.node.ins[0] === act ? "gelu-other" : mul.node.ins[1] === act ? "other-gelu" : undefined;
    if (order === undefined || srq.node.ins[0] !== mul.outputs[0].name) return undefined;
    const otherSlot = order === "gelu-other" ? 1 : 0;
    const gate = gelu.node.ins[0],
      other = mul.node.ins[otherSlot];
    if (other === gate) return undefined;
    const shape = gelu.inputShapes[0];
    const rows = numel(shape.slice(0, -1));
    if (
      shape[shape.length - 1] !== GELU_MUL_STATIC_QUANTIZE_DIM ||
      rows < 1 ||
      rows > GELU_MUL_STATIC_QUANTIZE_MAX_ROWS
    )
      return undefined;
    if (
      !sameShape(gelu.outputs[0].shape, shape) ||
      mul.inputShapes.some((inputShape) => !sameShape(inputShape, shape)) ||
      !sameShape(mul.outputs[0].shape, shape) ||
      !sameShape(srq.outputs[0].shape, shape)
    )
      return undefined;
    if (context.packedValues.has(srq.outputs[0].name)) return undefined;
    return {
      window: chain,
      chain,
      gate,
      other,
      order,
      outputName: srq.outputs[0].name,
      outputShape: srq.outputs[0].shape,
      count: numel(shape),
      scale: staticQuantizeScale(srq.node.attrs, "gelu_tanh mul static_quantize fusion"),
    };
  },
  build: (matched) => ({
    binds: [matched.gate, matched.other],
    outputName: matched.outputName,
    outputShape: matched.outputShape,
    temps: [],
    dispatches: [
      {
        key: geluMulStaticQuantizeKey(matched.order),
        wgsl: () => geluMulStaticQuantizeWgsl(matched.order),
        params: geluMulStaticQuantizeParams(matched.count, matched.scale),
        workgroups: {
          kind: "gridStride",
          items: matched.count,
          size: GELU_MUL_STATIC_QUANTIZE_WORKGROUP_SIZE,
        },
      },
    ],
  }),
});
