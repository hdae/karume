/**
 * Fusion rule `adaln` (DiT modulation — a window of 6 / 7 nodes). The eligibility conditions
 * shared by every rule and `defineRule` are in {@link "../fusion-rule.ts"}; the declaration table
 * that fixes the application order is `FUSION_RULES` in {@link "../fusion.ts"}.
 */

import { LAYER_NORM_OP, layerNormAttrs, numel } from "../../ops.ts";
import { ADALN_NORM_KEY, ADALN_NORM_WGSL, adalnNormParams } from "../../kernels/adaln-norm.ts";
import {
  allF32,
  defineRule,
  type FusionMatch,
  internalsArePrivate,
  sameShape,
} from "../fusion-rule.ts";

type AdalnMatch = FusionMatch & {
  readonly binds: readonly string[];
  readonly outputName: string;
  readonly outputShape: readonly number[];
  readonly rows: number;
  readonly dim: number;
  readonly eps: number;
};

/** The broadcast shape of the modulation vector (`[1,…,1,dim]`, every leading axis 1). */
const modulationShape = (rowShape: readonly number[], dim: number): readonly number[] => [
  ...rowShape.slice(0, -1).map(() => 1),
  dim,
];

/**
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
export const ADALN_RULE = defineRule<AdalnMatch>({
  name: "adaln",
  heads: [LAYER_NORM_OP],
  match: (nodes, index, context) => {
    const norm = nodes[index];
    if (norm?.node.op !== LAYER_NORM_OP) return undefined;
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
    if (!internalsArePrivate(chain, context)) return undefined;

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

    const { eps } = layerNormAttrs(norm.node.attrs, `融合 adaln の ${norm.node.op}`);
    return {
      window: nodes.slice(index, cursor + 3),
      chain,
      binds,
      outputName: offset.outputs[0].name,
      outputShape: offset.outputs[0].shape,
      rows,
      dim,
      eps,
    };
  },
  build: (matched) => ({
    binds: matched.binds,
    outputName: matched.outputName,
    outputShape: matched.outputShape,
    temps: [],
    dispatches: [
      {
        key: ADALN_NORM_KEY,
        wgsl: () => ADALN_NORM_WGSL,
        params: adalnNormParams(matched.rows, matched.dim, matched.eps),
        // One row = one workgroup (grid-stride over rows), so the divisor is 1 and the row count
        // is the workgroup count. `@workgroup_size` is the separate 256 threads inside a row.
        workgroups: { kind: "gridStride", items: matched.rows, size: 1 },
      },
    ],
  }),
});
