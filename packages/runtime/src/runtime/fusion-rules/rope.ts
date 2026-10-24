/**
 * Fusion rule `rope` (the 7 nodes of a half-split RoPE, or the 8 nodes of the pair form). The
 * eligibility conditions shared by every rule and `defineRule` are in {@link "../fusion-rule.ts"};
 * the declaration table that fixes the application order is `FUSION_RULES` in
 * {@link "../fusion.ts"}.
 */

import { catDim, numel, sliceAttrs, SYM_PREFIX_SLICE_OP } from "../../ops.ts";
import {
  ROPE_BSHD_KEY,
  ROPE_KEY,
  ROPE_PAIR_BSHD_KEY,
  ROPE_WORKGROUP_SIZE,
  type RopeLayout,
  ropeParams,
  type RopeRotation,
  ropeWgsl,
} from "../../kernels/rope.ts";
import {
  allF32,
  defineRule,
  type FusionMatch,
  type FusionScanContext,
  internalsArePrivate,
  sameShape,
} from "../fusion-rule.ts";
import type { NodePlan } from "../plan.ts";

type RopeMatch = FusionMatch & {
  /**
   * The bind surface in kernel binding order (half: x, cos, sin / pair: x, pairs, cos, sin).
   */
  readonly binds: readonly string[];
  readonly outputName: string;
  readonly outputShape: readonly number[];
  readonly tableAxisSize: number;
  readonly headDim: number;
  readonly layout: RopeLayout;
  readonly rotation: RopeRotation;
};

/**
 * The pair form (Wan's DiT — the complex rotation written in real numbers): the **8 consecutive
 * nodes** the exporter produces.
 *
 * ```
 * slice(pairs, dim 3, 1..2)    -> odd       [1,S,H·D/2,1]   (pairs [1,S,H·D/2,2] — measured: a reshape of x)
 * neg(odd)                     -> negative
 * slice(pairs, dim 3, 0..1)    -> even
 * cat(negative, even, dim 3)   -> rotated   [1,S,H·D/2,2]
 * reshape(rotated)             -> rot       [1,S,H,D]
 * mul(x, cos)                  -> direct    (table [1,S,1,D])
 * mul(rot, sin)                -> cross
 * add(direct, cross)           -> y
 * ```
 *
 * MUST: only this order, BSHD (`x` `[1,S,H,D]`, tables `[1,S,1,D]`, D a positive even number) and
 * these slice boundaries. The reshape that makes `pairs` from `x` lies outside the window; the
 * kernel binds `pairs` and `x` separately ({@link "../../kernels/rope.ts"}), so it computes what
 * the window computes without relying on that reshape.
 * NOTE: the first use of the shared tables in Irodori's DiT / speaker puts two
 * {@link SYM_PREFIX_SLICE_OP}s between the reshape back and the two `mul`s; that spelling is not
 * accepted (this form has no in-window passthrough), while the later uses have exactly this form.
 *
 * Total uses of external inputs: pairs twice (slice×2), x / cos / sin once each.
 */
const matchPairForm = (
  nodes: readonly NodePlan[],
  index: number,
  context: FusionScanContext,
): RopeMatch | undefined => {
  const odd = nodes[index];
  const neg = nodes[index + 1];
  const even = nodes[index + 2];
  const cat = nodes[index + 3];
  const view = nodes[index + 4];
  const direct = nodes[index + 5];
  const cross = nodes[index + 6];
  const add = nodes[index + 7];
  if (
    odd?.node.op !== "slice" ||
    neg?.node.op !== "neg" ||
    even?.node.op !== "slice" ||
    cat?.node.op !== "cat" ||
    view?.node.op !== "reshape" ||
    direct?.node.op !== "mul" ||
    cross?.node.op !== "mul" ||
    add?.node.op !== "add"
  )
    return undefined;
  const chain = [odd, neg, even, cat, view, direct, cross, add];
  if (!allF32(chain)) return undefined;

  const pairsName = odd.node.ins[0];
  const xName = direct.node.ins[0];
  if (
    even.node.ins[0] !== pairsName ||
    neg.node.ins[0] !== odd.outputs[0].name ||
    cat.node.ins[0] !== neg.outputs[0].name ||
    cat.node.ins[1] !== even.outputs[0].name ||
    view.node.ins[0] !== cat.outputs[0].name ||
    cross.node.ins[0] !== view.outputs[0].name ||
    add.node.ins[0] !== direct.outputs[0].name ||
    add.node.ins[1] !== cross.outputs[0].name
  )
    return undefined;

  const xShape = direct.inputShapes[0];
  if (xShape.length !== 4 || xShape[0] !== 1) return undefined;
  const [, sequence, heads, headDim] = xShape;
  if (sequence < 1 || heads < 1 || headDim < 2 || headDim % 2 !== 0) return undefined;
  const pairCount = (heads * headDim) / 2;
  const pairsShape = [1, sequence, pairCount, 2];
  const halfShape = [1, sequence, pairCount, 1];
  const tableShape = [1, sequence, 1, headDim];
  if (
    !sameShape(odd.inputShapes[0], pairsShape) ||
    !sameShape(odd.outputs[0].shape, halfShape) ||
    !sameShape(neg.outputs[0].shape, halfShape) ||
    !sameShape(even.outputs[0].shape, halfShape) ||
    !sameShape(cat.outputs[0].shape, pairsShape) ||
    !sameShape(view.outputs[0].shape, xShape) ||
    !sameShape(direct.inputShapes[1], tableShape) ||
    !sameShape(cross.inputShapes[1], tableShape) ||
    !sameShape(direct.outputs[0].shape, xShape) ||
    !sameShape(cross.outputs[0].shape, xShape) ||
    !sameShape(add.outputs[0].shape, xShape)
  )
    return undefined;

  const oddSlice = sliceAttrs(odd.node.attrs, "RoPE pair odd slice");
  const evenSlice = sliceAttrs(even.node.attrs, "RoPE pair even slice");
  if (
    oddSlice.dim !== 3 ||
    oddSlice.start !== 1 ||
    oddSlice.end !== 2 ||
    evenSlice.dim !== 3 ||
    evenSlice.start !== 0 ||
    evenSlice.end !== 1 ||
    catDim(cat.node.attrs, "RoPE pair cat") !== 3
  )
    return undefined;

  if (!internalsArePrivate(chain, context)) return undefined;
  // The bind surface maps one-to-one onto the kernel's bindings 1–4, so no duplicates (one value
  // in two slots is not a measured form, and it breaks the "no duplicates" of FusedStep.binds).
  const binds = [xName, pairsName, direct.node.ins[1], cross.node.ins[1]];
  if (new Set(binds).size !== binds.length) return undefined;

  return {
    window: nodes.slice(index, index + chain.length),
    chain,
    binds,
    outputName: add.outputs[0].name,
    outputShape: add.outputs[0].shape,
    tableAxisSize: heads,
    headDim,
    layout: "bshd",
    rotation: "pair",
  };
};

/**
 * Two exact forms: the half split below, and the pair form ({@link matchPairForm}).
 *
 * half-split RoPE: the **7 consecutive nodes** the exporter produces. Only two orders are
 * accepted: slice-first (attention) and direct-mul-first (text encoders / the Gemma family).
 *
 * - slice-first: `slice×2, neg, cat, mul(x,cos), mul(cat,sin), add`
 * - direct-first: `mul(x,cos), slice×2, neg, cat, mul(cat,sin), add`
 *
 * MUST: only `[1,H,S,D]` / table `[1,1,S,D]` or `[1,S,H,D]` / table `[1,S,1,D]`
 * (D a positive even number) / dim=3 with
 * `0-D/2` / `D/2-D`. The head width D has **two measured values (128 and 256)**, so it is
 * derived from the slice boundaries; but other broadcasts and other cat axes are not admitted
 * because "the formula looks similar", and the pair form (`slice, neg, slice, cat` — a different
 * rotation) is a separate exact matcher ({@link matchPairForm}) with its own kernel variant — the
 * moment the accepted set widens, the fallback guarantee ("if it does not match, the existing path
 * is always correct") stops holding. The kernel (kernels/rope.ts) takes `head_dim` / `half_dim` as
 * uniforms, so generalising D needs no WGSL change.
 *
 * ## In-window passthrough
 *
 * A measured form has **exactly one** {@link SYM_PREFIX_SLICE_OP} between `cat` and the following
 * `mul`, shrinking the cos / sin table to the runtime T (the Gemma family; the table is built once
 * per θ variant and reused by every layer, so it sits only at the first use of each variant). That
 * one node consumes no value of the chain, so it can move in front of the fused step (legality is
 * decided by `passthroughIsIndependent` in {@link "../fusion-rule.ts"}).
 *
 * Total uses of external inputs: x 3 times (slice×2 and the direct mul), cos / sin once each.
 */
export const ROPE_RULE = defineRule<RopeMatch>({
  name: "rope",
  heads: ["mul", "slice"],
  match: (nodes, index, context) => {
    // Called for every node, so check the leading op before building an array (a short-lived
    // slice per node would eat back on the CPU what the dispatch reduction saves).
    const leading = nodes[index];
    const directFirst = leading?.node.op === "mul";
    if (!directFirst && leading?.node.op !== "slice") return undefined;
    // The pair form shares the leading `slice`; its second op (`neg`) tells it apart from the half
    // split (whose second op is a `slice`) before the cursor walk below.
    if (!directFirst && nodes[index + 1]?.node.op === "neg") {
      return matchPairForm(nodes, index, context);
    }
    let cursor = index + (directFirst ? 1 : 0);
    const first = nodes[cursor];
    const second = nodes[cursor + 1];
    const neg = nodes[cursor + 2];
    const cat = nodes[cursor + 3];
    if (
      first?.node.op !== "slice" ||
      second?.node.op !== "slice" ||
      neg?.node.op !== "neg" ||
      cat?.node.op !== "cat"
    )
      return undefined;
    cursor += 4;
    // The only in-window passthrough is the one sym_prefix_slice of the cos / sin table (the
    // measured form). Widening this to "any number of any op" would accept reorderings across
    // nodes unrelated to the chain.
    if (nodes[cursor]?.node.op === SYM_PREFIX_SLICE_OP) cursor += 1;
    const direct = directFirst ? leading : nodes[cursor];
    if (!directFirst) cursor += 1;
    const cross = nodes[cursor];
    const add = nodes[cursor + 1];
    if (direct?.node.op !== "mul" || cross?.node.op !== "mul" || add?.node.op !== "add")
      return undefined;
    const windowEnd = cursor + 2;

    // MUST: keep the use counts and the release bookkeeping in **the actual node order**. A list
    // reordered by role shifts the set of internal values for direct-first only.
    const chain = directFirst
      ? [direct, first, second, neg, cat, cross, add]
      : [first, second, neg, cat, direct, cross, add];
    if (!allF32(chain)) return undefined;

    const xName = first.node.ins[0];
    if (
      second.node.ins[0] !== xName ||
      neg.node.ins[0] !== second.outputs[0].name ||
      cat.node.ins[0] !== neg.outputs[0].name ||
      cat.node.ins[1] !== first.outputs[0].name ||
      direct.node.ins[0] !== xName ||
      cross.node.ins[0] !== cat.outputs[0].name ||
      add.node.ins[0] !== direct.outputs[0].name ||
      add.node.ins[1] !== cross.outputs[0].name
    )
      return undefined;

    const xShape = first.inputShapes[0];
    if (xShape.length !== 4 || xShape[0] !== 1) return undefined;
    const [, axis1, axis2, headDim] = xShape;
    if (axis1 < 1 || axis2 < 1 || headDim < 2 || headDim % 2 !== 0) return undefined;
    const tablesMatch = (shape: readonly number[]): boolean =>
      sameShape(direct.inputShapes[1], shape) && sameShape(cross.inputShapes[1], shape);
    // In the degenerate form where both match, the existing BHSD wins, keeping the existing key and
    // body.
    const layout = tablesMatch([1, 1, axis2, headDim])
      ? "bhsd"
      : tablesMatch([1, axis1, 1, headDim])
        ? "bshd"
        : undefined;
    if (layout === undefined) return undefined;
    // In both layouts the second axis is the divisor of the table index (BHSD=S, BSHD=H).
    const tableAxisSize = axis2;
    const halfDim = headDim / 2;

    const firstSlice = sliceAttrs(first.node.attrs, "RoPE first slice");
    const secondSlice = sliceAttrs(second.node.attrs, "RoPE second slice");
    if (
      firstSlice.dim !== 3 ||
      firstSlice.start !== 0 ||
      firstSlice.end !== halfDim ||
      secondSlice.dim !== 3 ||
      secondSlice.start !== halfDim ||
      secondSlice.end !== headDim ||
      catDim(cat.node.attrs, "RoPE cat") !== 3
    )
      return undefined;

    const halfShape = [1, axis1, axis2, halfDim];
    const fullShape = [1, axis1, axis2, headDim];
    if (
      !sameShape(first.outputs[0].shape, halfShape) ||
      !sameShape(second.outputs[0].shape, halfShape) ||
      !sameShape(neg.outputs[0].shape, halfShape) ||
      !sameShape(cat.outputs[0].shape, fullShape) ||
      !sameShape(direct.outputs[0].shape, fullShape) ||
      !sameShape(cross.outputs[0].shape, fullShape) ||
      !sameShape(add.outputs[0].shape, fullShape)
    )
      return undefined;

    if (!internalsArePrivate(chain, context)) return undefined;

    return {
      window: nodes.slice(index, windowEnd),
      chain,
      binds: [xName, direct.node.ins[1], cross.node.ins[1]],
      outputName: add.outputs[0].name,
      outputShape: add.outputs[0].shape,
      tableAxisSize,
      headDim,
      layout,
      rotation: "half",
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
          key:
            matched.rotation === "pair"
              ? ROPE_PAIR_BSHD_KEY
              : matched.layout === "bhsd"
                ? ROPE_KEY
                : ROPE_BSHD_KEY,
          wgsl: () => ropeWgsl(matched.layout, matched.rotation),
          params: ropeParams(count, matched.tableAxisSize, matched.headDim, matched.layout),
          workgroups: { kind: "gridStride", items: count, size: ROPE_WORKGROUP_SIZE },
        },
      ],
    };
  },
});
