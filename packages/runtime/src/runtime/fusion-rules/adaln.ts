/**
 * Fusion rule `adaln` (DiT modulation — a window of 6 / 7 nodes with broadcast modulation, or the
 * 6 nodes of the per-token form). The eligibility conditions shared by every rule and `defineRule`
 * are in {@link "../fusion-rule.ts"}; the declaration table that fixes the application order is
 * `FUSION_RULES` in {@link "../fusion.ts"}.
 */

import { LAYER_NORM_OP, layerNormAttrs, numel, WHERE_OP } from "../../ops.ts";
import {
  ADALN_MODULATION_PACK_KEY,
  ADALN_MODULATION_PACK_WGSL,
  ADALN_MODULATION_PACK_WORKGROUP_SIZE,
  ADALN_NORM_KEY,
  ADALN_NORM_TOKEN_KEY,
  ADALN_NORM_WGSL,
  adalnModulationPackParams,
  type AdalnModulation,
  adalnNormParams,
  adalnNormWgsl,
} from "../../kernels/adaln-norm.ts";
import {
  allF32,
  defineRule,
  type FusedStep,
  type FusionMatch,
  isF32Where,
  sameShape,
} from "../fusion-rule.ts";
import type { NodePlan } from "../plan.ts";

type AdalnMatch = FusionMatch & {
  /**
   * broadcast: x, weight, bias, scale, one, shift (= the kernel's bindings 1–6) / token: x,
   * weight, bias, mask, scale_c, scale_g, one, shift_c, shift_g (the two dispatches pick from it).
   */
  readonly binds: readonly string[];
  readonly outputName: string;
  readonly outputShape: readonly number[];
  readonly rows: number;
  readonly dim: number;
  readonly eps: number;
  readonly modulation: AdalnModulation;
};

/** The broadcast shape of the modulation vector (`[1,…,1,dim]`, every leading axis 1). */
const modulationShape = (rowShape: readonly number[], dim: number): readonly number[] => [
  ...rowShape.slice(0, -1).map(() => 1),
  dim,
];

/**
 * The per-token form (Wan2.2's DiT — the timestep differs per token, ADR 0121 decision 3): the
 * **6 consecutive nodes** the exporter emits.
 *
 * ```
 * layer_norm(x, w, b)              -> t          [.., dim]
 * where(mask, scale_c, scale_g)    -> sel_scale  [.., dim]   (mask [..,1] bool, vectors [1,..,1,dim])
 * add(sel_scale, one[1])           -> s          [.., dim]
 * mul(t, s)                        -> p          [.., dim]
 * where(mask, shift_c, shift_g)    -> sel_shift  [.., dim]   (the same mask)
 * add(p, sel_shift)                -> y          [.., dim]
 * ```
 *
 * MUST: the only non-f32 input is the condition slot of the two `where`s ({@link isF32Where}); the
 * two `where`s read the **same** mask value (the kernel reads one mask per row). The input orders
 * are fixed to the measured form, as in the broadcast form. MUST: `one` is bound without assuming
 * its value (as in the broadcast form).
 *
 * Total uses of external inputs: x / weight / bias / scale_c / scale_g / one / shift_c / shift_g
 * once each and mask twice = 10.
 */
const matchTokenForm = (nodes: readonly NodePlan[], index: number): AdalnMatch | undefined => {
  const norm = nodes[index];
  const selectScale = nodes[index + 1];
  const modulate = nodes[index + 2];
  const multiply = nodes[index + 3];
  const selectShift = nodes[index + 4];
  const offset = nodes[index + 5];
  if (
    norm?.node.op !== LAYER_NORM_OP ||
    selectScale === undefined ||
    modulate?.node.op !== "add" ||
    multiply?.node.op !== "mul" ||
    selectShift === undefined ||
    offset?.node.op !== "add"
  )
    return undefined;
  if (!isF32Where(selectScale) || !isF32Where(selectShift)) return undefined;
  if (!allF32([norm, modulate, multiply, offset])) return undefined;
  const chain = [norm, selectScale, modulate, multiply, selectShift, offset];

  const maskName = selectScale.node.ins[0];
  if (
    selectShift.node.ins[0] !== maskName ||
    modulate.node.ins[0] !== selectScale.outputs[0].name ||
    multiply.node.ins[0] !== norm.outputs[0].name ||
    multiply.node.ins[1] !== modulate.outputs[0].name ||
    offset.node.ins[0] !== multiply.outputs[0].name ||
    offset.node.ins[1] !== selectShift.outputs[0].name
  )
    return undefined;

  const rowShape = norm.inputShapes[0];
  if (rowShape.length < 2 || rowShape.some((extent) => extent < 1)) return undefined;
  const dim = rowShape[rowShape.length - 1];
  const rows = numel(rowShape.slice(0, -1));
  const affineShape = [dim];
  const modShape = modulationShape(rowShape, dim);
  // The mask has one element per row (flat index = row), so the kernel reads `mask[row]`.
  const maskShape = [...rowShape.slice(0, -1), 1];
  const selects = (step: NodePlan): boolean =>
    sameShape(step.inputShapes[0], maskShape) &&
    sameShape(step.inputShapes[1], modShape) &&
    sameShape(step.inputShapes[2], modShape) &&
    sameShape(step.outputs[0].shape, rowShape);
  if (
    !sameShape(norm.outputs[0].shape, rowShape) ||
    !sameShape(norm.inputShapes[1], affineShape) ||
    !sameShape(norm.inputShapes[2], affineShape) ||
    !selects(selectScale) ||
    !selects(selectShift) ||
    !sameShape(modulate.inputShapes[0], rowShape) ||
    !sameShape(modulate.inputShapes[1], [1]) ||
    !sameShape(modulate.outputs[0].shape, rowShape) ||
    !sameShape(multiply.inputShapes[0], rowShape) ||
    !sameShape(multiply.inputShapes[1], rowShape) ||
    !sameShape(multiply.outputs[0].shape, rowShape) ||
    !sameShape(offset.inputShapes[0], rowShape) ||
    !sameShape(offset.inputShapes[1], rowShape) ||
    !sameShape(offset.outputs[0].shape, rowShape)
  )
    return undefined;

  // MUST: no duplicates — the operand lists of the two dispatches index into this surface, and a
  // value in two slots is not a measured form.
  const binds = [
    norm.node.ins[0],
    norm.node.ins[1],
    norm.node.ins[2],
    maskName,
    selectScale.node.ins[1],
    selectScale.node.ins[2],
    modulate.node.ins[1],
    selectShift.node.ins[1],
    selectShift.node.ins[2],
  ];
  if (new Set(binds).size !== binds.length) return undefined;

  const { eps } = layerNormAttrs(norm.node.attrs, `fusion adaln ${norm.node.op}`);
  return {
    window: nodes.slice(index, index + chain.length),
    chain,
    binds,
    outputName: offset.outputs[0].name,
    outputShape: offset.outputs[0].shape,
    rows,
    dim,
    eps,
    modulation: "token",
  };
};

/**
 * The per-token fused step: ① pack the four vectors into one `4 · dim` table (a step temporary),
 * ② the shared staged body reading `mask[row]` and the table. The bind indices follow
 * {@link AdalnMatch.binds} of the token form.
 */
const tokenDispatches = (matched: AdalnMatch): Pick<FusedStep, "temps" | "dispatches"> => ({
  temps: [{ byteLength: 4 * matched.dim * 4, allocBefore: 0, releaseAfter: 1 }],
  dispatches: [
    {
      key: ADALN_MODULATION_PACK_KEY,
      wgsl: () => ADALN_MODULATION_PACK_WGSL,
      params: adalnModulationPackParams(matched.dim),
      operands: [
        { kind: "bind", index: 4 },
        { kind: "bind", index: 5 },
        { kind: "bind", index: 7 },
        { kind: "bind", index: 8 },
        { kind: "temp", id: 0 },
      ],
      workgroups: {
        kind: "gridStride",
        items: matched.dim,
        size: ADALN_MODULATION_PACK_WORKGROUP_SIZE,
      },
    },
    {
      key: ADALN_NORM_TOKEN_KEY,
      wgsl: () => adalnNormWgsl("token"),
      params: adalnNormParams(matched.rows, matched.dim, matched.eps),
      operands: [
        { kind: "bind", index: 0 },
        { kind: "bind", index: 1 },
        { kind: "bind", index: 2 },
        { kind: "bind", index: 3 },
        { kind: "temp", id: 0 },
        { kind: "bind", index: 6 },
        { kind: "output" },
      ],
      // One row = one workgroup, as in the broadcast form.
      workgroups: { kind: "gridStride", items: matched.rows, size: 1 },
    },
  ],
});

/**
 * Two exact forms share the leading `layer_norm`; the next op decides (`where` = the per-token
 * form, {@link matchTokenForm}; anything else is left to the broadcast form below).
 *
 * adaLN (DiT modulation): the **window of 6 / 7 nodes** the exporter emits.
 *
 * ```
 * layer_norm(x, w, b)              -> t          [.., dim]
 * reshape × 2–3                     (in-window passthrough — unsqueeze of the modulation vector)
 * add(scale, one[1])               -> s          [1,..,1,dim]
 * mul(t, s)                        -> p          [.., dim]
 * add(p, shift)                    -> y          [.., dim]
 * ```
 *
 * The four folded nodes are `layer_norm / add / mul / add`; the reshapes in between stay
 * zero-dispatch aliases placed in front of the fused step (`passthroughIsIndependent` in
 * {@link "../fusion-rule.ts"} checks that the reordering is legal).
 *
 * MUST: `one` is bound as a buffer **without assuming its value**, and the kernel reads
 * `one[0]`. The fusion pass cannot see the contents of an IR initializer (and even if it could,
 * baking in `1.0` would step outside "if it does not match, it is always correct").
 * MUST: the input order of mul / add is fixed to the measured form (commutative for finite
 * values, but which NaN payload wins may differ between backends). Unlike SiLU, the order
 * variants are not carried in the key — only one order is measured, so the accepted set is not
 * widened.
 *
 * Total uses of external inputs: x / ln weight / ln bias / scale / one / shift once each = 6.
 */
export const ADALN_RULE = defineRule<AdalnMatch, "adaln">({
  name: "adaln",
  heads: [LAYER_NORM_OP],
  // ADR 0147 decision 2: every built-in fusion rule is class E (u32 equality with the unfused plan).
  contract: "E",
  internals: "private",
  match: (nodes, index) => {
    const norm = nodes[index];
    if (norm?.node.op !== LAYER_NORM_OP) return undefined;
    if (nodes[index + 1]?.node.op === WHERE_OP) return matchTokenForm(nodes, index);
    // The in-window passthrough is only "the consecutive reshapes right after". Two forms are
    // measured: 2 (shift / scale) and 3 with the gate added (the gate is consumed outside the
    // chain).
    let cursor = index + 1;
    while (nodes[cursor]?.node.op === "reshape") cursor += 1;
    const passthroughCount = cursor - index - 1;
    if (passthroughCount < 2 || passthroughCount > 3) return undefined;

    const modulate = nodes[cursor];
    const multiply = nodes[cursor + 1];
    const offset = nodes[cursor + 2];
    if (modulate === undefined || multiply === undefined || offset === undefined) return undefined;
    if (modulate.node.op !== "add" || multiply.node.op !== "mul" || offset.node.op !== "add")
      return undefined;

    // MUST: derive the use counts and the release bookkeeping from **the folded nodes only** (the
    // existing path counts the passthroughs as plain nodes).
    const chain = [norm, modulate, multiply, offset];
    if (!allF32(chain)) return undefined;

    if (
      multiply.node.ins[0] !== norm.outputs[0].name ||
      multiply.node.ins[1] !== modulate.outputs[0].name ||
      offset.node.ins[0] !== multiply.outputs[0].name
    )
      return undefined;

    const rowShape = norm.inputShapes[0];
    if (rowShape.length < 2 || rowShape.some((extent) => extent < 1)) return undefined;
    const dim = rowShape[rowShape.length - 1];
    const rows = numel(rowShape.slice(0, -1));
    const affineShape = [dim];
    const modShape = modulationShape(rowShape, dim);
    if (
      !sameShape(norm.outputs[0].shape, rowShape) ||
      !sameShape(norm.inputShapes[1], affineShape) ||
      !sameShape(norm.inputShapes[2], affineShape) ||
      !sameShape(modulate.inputShapes[0], modShape) ||
      !sameShape(modulate.inputShapes[1], [1]) ||
      !sameShape(modulate.outputs[0].shape, modShape) ||
      !sameShape(multiply.inputShapes[0], rowShape) ||
      !sameShape(multiply.inputShapes[1], modShape) ||
      !sameShape(multiply.outputs[0].shape, rowShape) ||
      !sameShape(offset.inputShapes[0], rowShape) ||
      !sameShape(offset.inputShapes[1], modShape) ||
      !sameShape(offset.outputs[0].shape, rowShape)
    )
      return undefined;

    // The bind surface maps one-to-one onto the kernel's bindings 1–6, so **no duplicates** (a
    // value name in two slots is not measured, and it would break the "no duplicates" of
    // {@link FusedStep.binds}).
    const binds = [
      norm.node.ins[0],
      norm.node.ins[1],
      norm.node.ins[2],
      modulate.node.ins[0],
      modulate.node.ins[1],
      offset.node.ins[1],
    ];
    if (new Set(binds).size !== binds.length) return undefined;

    const { eps } = layerNormAttrs(norm.node.attrs, `fusion adaln ${norm.node.op}`);
    return {
      window: nodes.slice(index, cursor + 3),
      chain,
      binds,
      outputName: offset.outputs[0].name,
      outputShape: offset.outputs[0].shape,
      rows,
      dim,
      eps,
      modulation: "broadcast",
    };
  },
  build: (matched) => ({
    binds: matched.binds,
    outputName: matched.outputName,
    outputShape: matched.outputShape,
    ...(matched.modulation === "token"
      ? tokenDispatches(matched)
      : {
          temps: [],
          dispatches: [
            {
              key: ADALN_NORM_KEY,
              wgsl: () => ADALN_NORM_WGSL,
              params: adalnNormParams(matched.rows, matched.dim, matched.eps),
              // One row = one workgroup (grid-stride over rows), so the divisor is 1 and the row
              // count is the workgroup count. `@workgroup_size` is the separate 256 threads inside
              // a row.
              workgroups: { kind: "gridStride", items: matched.rows, size: 1 },
            },
          ],
        }),
  }),
});
