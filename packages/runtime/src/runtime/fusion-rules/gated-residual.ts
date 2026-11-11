/**
 * Fusion rule `gatedResidual` (the per-token gated residual of Wan2.2's DiT — 3 nodes). The
 * eligibility conditions shared by every rule and `defineRule` are in {@link "../fusion-rule.ts"};
 * the declaration table that fixes the application order is `FUSION_RULES` in
 * {@link "../fusion.ts"}.
 */

import { numel, WHERE_OP } from "../../ops.ts";
import {
  GATED_RESIDUAL_KEY,
  GATED_RESIDUAL_WGSL,
  GATED_RESIDUAL_WORKGROUP_SIZE,
  gatedResidualParams,
} from "../../kernels/gated-residual.ts";
import { allF32, defineRule, type FusionMatch, isF32Where, sameShape } from "../fusion-rule.ts";

type GatedResidualMatch = FusionMatch & {
  /** mask, gate_c, gate_g, y, residual (= the kernel's bindings 1–5). */
  readonly binds: readonly string[];
  readonly outputName: string;
  readonly outputShape: readonly number[];
  readonly dim: number;
};

/**
 * The gated residual (Wan2.2's DiT, after the self-attention output projection and after the FFN —
 * the gate is picked per token by the condition mask, ADR 0121 decision 3): the **3 consecutive
 * nodes** the exporter emits.
 *
 * ```
 * where(mask, gate_c, gate_g)   -> g     [.., dim]   (mask [..,1] bool, gates [1,..,1,dim])
 * mul(y, g)                     -> p     [.., dim]
 * add(residual, p)              -> out   [.., dim]
 * ```
 *
 * MUST: exactly this order and wiring — `mul(y, g)` and `add(residual, p)` are the measured
 * operand orders (commutative for finite values, but which NaN payload wins may differ between
 * backends, so the other orders are not admitted). The only non-f32 input is the condition slot
 * ({@link isF32Where}); `y` and `residual` are full `[.., dim]` tensors (no broadcast).
 *
 * Total uses of external inputs: mask / gate_c / gate_g / y / residual once each = 5.
 */
export const GATED_RESIDUAL_RULE = defineRule<GatedResidualMatch, "gatedResidual">({
  name: "gatedResidual",
  heads: [WHERE_OP],
  internals: "private",
  match: (nodes, index) => {
    const select = nodes[index];
    if (select?.node.op !== WHERE_OP) return undefined;
    const multiply = nodes[index + 1];
    const add = nodes[index + 2];
    if (multiply?.node.op !== "mul" || add?.node.op !== "add") return undefined;
    if (!isF32Where(select) || !allF32([multiply, add])) return undefined;
    const chain = [select, multiply, add];
    if (
      multiply.node.ins[1] !== select.outputs[0].name ||
      add.node.ins[1] !== multiply.outputs[0].name
    )
      return undefined;

    const rowShape = add.outputs[0].shape;
    if (rowShape.length < 2 || rowShape.some((extent) => extent < 1)) return undefined;
    const dim = rowShape[rowShape.length - 1];
    // One mask element per row (flat index = row), so the kernel reads `mask[i / dim]`.
    const maskShape = [...rowShape.slice(0, -1), 1];
    const gateShape = [...rowShape.slice(0, -1).map(() => 1), dim];
    if (
      !sameShape(select.inputShapes[0], maskShape) ||
      !sameShape(select.inputShapes[1], gateShape) ||
      !sameShape(select.inputShapes[2], gateShape) ||
      !sameShape(select.outputs[0].shape, rowShape) ||
      !sameShape(multiply.inputShapes[0], rowShape) ||
      !sameShape(multiply.outputs[0].shape, rowShape) ||
      !sameShape(add.inputShapes[0], rowShape)
    )
      return undefined;

    // The bind surface maps one-to-one onto the kernel's bindings 1–5, so no duplicates (a value
    // in two slots is not a measured form).
    const binds = [
      select.node.ins[0],
      select.node.ins[1],
      select.node.ins[2],
      multiply.node.ins[0],
      add.node.ins[0],
    ];
    if (new Set(binds).size !== binds.length) return undefined;

    return {
      window: nodes.slice(index, index + chain.length),
      chain,
      binds,
      outputName: add.outputs[0].name,
      outputShape: add.outputs[0].shape,
      dim,
    };
  },
  build: (matched) => {
    const count = numel(matched.outputShape);
    return {
      binds: matched.binds,
      outputName: matched.outputName,
      outputShape: matched.outputShape,
      temps: [],
      dispatches: [
        {
          key: GATED_RESIDUAL_KEY,
          wgsl: () => GATED_RESIDUAL_WGSL,
          params: gatedResidualParams(count, matched.dim),
          workgroups: { kind: "gridStride", items: count, size: GATED_RESIDUAL_WORKGROUP_SIZE },
        },
      ],
    };
  },
});
