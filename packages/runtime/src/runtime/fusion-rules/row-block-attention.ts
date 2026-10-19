/**
 * Fusion rule `rowBlockAttention` (the 9 nodes of decomposed attention into **block execution over
 * query rows**) and its splitting pure function {@link planRowBlocks}. The eligibility conditions
 * shared by every rule and `defineRule` are in {@link "../fusion-rule.ts"}; the declaration table
 * that fixes the application order is `FUSION_RULES` in {@link "../fusion.ts"}.
 */

import { numel, SAFE_SOFTMAX_OP, softmaxDim } from "../../ops.ts";
import { tiledWorkgroups } from "../../codegen/dispatch.ts";
import {
  ELEMENTWISE_WORKGROUP_SIZE,
  elementwiseKey,
  elementwiseParams,
  type ElementwiseSpec,
  elementwiseWgsl,
} from "../../codegen/elementwise.ts";
import { bmmKey, bmmParams, bmmRowWindowParams, bmmWgsl } from "../../kernels/bmm.ts";
import { gemmUsesVec4 } from "../../kernels/gemm.ts";
import { gemmGeometryForRows, gemmTileM, gemmTileN } from "../../kernels/gemm-geometry.ts";
import { SAFE_SOFTMAX_KEY, SAFE_SOFTMAX_WGSL, softmaxParams } from "../../kernels/softmax.ts";
import { ExecutionError } from "../plan.ts";
import {
  allF32,
  defineRule,
  type FusedDispatch,
  type FusedTemp,
  type FusionLimits,
  type FusionMatch,
  internalsArePrivate,
  sameShape,
} from "../fusion-rule.ts";

/** One row block (the half-open range of query rows `[offset, offset + rows)`). */
type RowBlock = {
  readonly offset: number;
  readonly rows: number;
};

/**
 * The limit (bytes) **per** row-block temporary, and the side that set it (the result of
 * {@link rowBlockLimit}).
 *
 * The limit is not a total: one S block and one row-statistics block are each within it, and what
 * is alive together may exceed it (ADR 0124 decision 1).
 */
export type RowBlockLimit = {
  readonly bytes: number;
  /**
   * The side that set the limit. The over-limit message names it (the fix differs depending on
   * whether the caller's budget or the device's binding limit was hit).
   */
  readonly source: "maxStorageBufferBindingSize" | "attentionRowBlockMaxBytes";
};

/**
 * The pure function that derives the limit per row-block temporary — the smaller of the device's
 * granted `maxStorageBufferBindingSize` and the budget the caller passes
 * (`SessionOptions.attentionRowBlockMaxBytes`)
 * (DECIDED: docs/decisions/0124-attention-row-block-budget.md decision 1).
 *
 * MUST: the three forms that split by row blocks (fused attention, the states form, the decomposed
 * `rowBlockAttention`) and the estimate all go through this one function. Writing the formula per
 * caller would let the estimate keep claiming a different count against an implementation where
 * only one side was fixed.
 * MUST: with no budget (`undefined`) it returns the binding limit itself — the default count, keys
 * and WGSL do not change by a single byte from before the budget field was added. When the budget
 * equals the binding limit, the binding limit is named as the source (the count is the same).
 * NOTE: the placement limit check for intermediate buffers (transient-plan.ts) does not use this
 * value. The budget is a knob that binds only the row-block temporaries; other bindings such as FFN
 * intermediates are checked against the granted value.
 */
export const rowBlockLimit = (granted: number, budget: number | undefined): RowBlockLimit =>
  budget !== undefined && budget < granted
    ? { bytes: budget, source: "attentionRowBlockMaxBytes" }
    : { bytes: granted, source: "maxStorageBufferBindingSize" };

/**
 * The pure function that splits the query rows evenly into "the **smallest count** where each block
 * fits the limit".
 *
 * Its inputs are resolved static numbers only (the device's granted limit, the caller's budget,
 * bytes per row, row count); no measurement enters — the only form that satisfies the ban on
 * run-time autotuning (ADR 0022).
 * So the same device, the same budget and the same bindings always give the same count, and the
 * prepared-plan key (the resolved bindings) implies the count too (a different S is a different
 * key, so its count is derived separately; the budget does not change over the Session's
 * lifetime).
 *
 * The split is even so that no block at the end is extremely short (that one block would land in a
 * different geometry bucket and grow one extra pipeline). The remainder is dealt one row at a time
 * from the front.
 *
 * MUST: **a shape where even one row does not fit the limit fails loudly**. Silently giving up on
 * the split here would make a graph whose allocation and binding both fail look like "the fusion
 * just dropped out". When the budget set the limit, the message names the budget (raising the
 * budget to at least one row makes it pass; it is not a device-side problem).
 *
 * @param limit The limit per temporary. A number is read as the device's binding limit itself (no
 *   budget). Pass the result of {@link rowBlockLimit} when a budget applies.
 * @param forced Forces the count (test only). A count that does not fit the limit (with a budget,
 *   the smaller of the two) also fails loudly.
 */
export const planRowBlocks = (
  rows: number,
  bytesPerRow: number,
  limit: number | RowBlockLimit,
  forced?: number,
): readonly RowBlock[] => {
  const cap = typeof limit === "number" ? rowBlockLimit(limit, undefined) : limit;
  const byBudget = cap.source === "attentionRowBlockMaxBytes";
  const where =
    `row-block split (rows ${rows} × ${bytesPerRow}B / limit ${cap.bytes}B` +
    `${byBudget ? " — budget attentionRowBlockMaxBytes" : ""})`;
  for (const [name, value] of [
    ["row count", rows],
    ["bytes per row", bytesPerRow],
  ] as const) {
    if (!Number.isSafeInteger(value) || value < 1) {
      throw new ExecutionError(`${where}: ${name} must be a positive integer (${value})`);
    }
  }
  if (bytesPerRow > cap.bytes) {
    throw new ExecutionError(
      byBudget
        ? `${where}: the temporary for one query row, ${bytesPerRow}B, already exceeds the budget attentionRowBlockMaxBytes` +
            " (row blocks cannot split it — set the budget to at least one row)"
        : `${where}: the temporary for one query row, ${bytesPerRow}B, already exceeds the storage binding limit` +
            " (a shape row blocks cannot split)",
    );
  }
  const count = forced ?? Math.ceil(rows / Math.floor(cap.bytes / bytesPerRow));
  if (!Number.isSafeInteger(count) || count < 1 || count > rows) {
    throw new ExecutionError(
      `${where}: the block count ${count} must be an integer from 1 to ${rows}`,
    );
  }
  const base = Math.floor(rows / count);
  const remainder = rows % count;
  const widest = base + (remainder > 0 ? 1 : 0);
  if (widest * bytesPerRow > cap.bytes) {
    throw new ExecutionError(
      `${where}: with ${count} blocks, one block of ${widest * bytesPerRow}B does not fit ` +
        `${byBudget ? "the budget attentionRowBlockMaxBytes" : "the limit"}`,
    );
  }
  const blocks: RowBlock[] = [];
  for (let index = 0, offset = 0; index < count; index += 1) {
    const blockRows = base + (index < remainder ? 1 : 0);
    blocks.push({ offset, rows: blockRows });
    offset += blockRows;
  }
  return blocks;
};

type RowBlockAttentionMatch = FusionMatch & {
  /** `[q, kᵀ, mask, v]` (the bind surface order — no duplicates). */
  readonly binds: readonly string[];
  readonly outputName: string;
  readonly outputShape: readonly number[];
  /** The batch axis with B·H folded. */
  readonly heads: number;
  /** Query row count M (= S). */
  readonly queries: number;
  /** Key column count N (= C). */
  readonly keys: number;
  /** Head width D (shared by q / k / v). */
  readonly headDim: number;
  readonly blocks: readonly RowBlock[];
  readonly limits: FusionLimits;
};

/**
 * **Row-block execution** of decomposed attention: the 9 consecutive nodes the exporter emits
 *
 * ```
 * bmm(q[H,M,D], kᵀ[H,D,N])   -> S     [H,M,N]
 * reshape                     -> S4    [1,H,M,N]
 * add(S4, mask[1,1,1,N])      -> Sm    [1,H,M,N]
 * safe_softmax(Sm, dim=3)     -> P4    [1,H,M,N]
 * expand (identity) → reshape -> P     [H,M,N]
 * expand (identity) → reshape -> V     [H,N,D]
 * bmm(P, V)                   -> O     [H,M,D]
 * ```
 *
 * and replaces them with **4 dispatches of the same form per block of query rows**. What is folded
 * is not the ops but **the materialised width of the intermediates**: `S` / `Sm` / `P` become
 * step-internal temporaries of one block each (materialising all M takes `H·M·N·4` bytes, which
 * exceeds the storage binding limit on some machines — the WebGPU core default
 * `maxStorageBufferBindingSize` is 128MiB).
 *
 * ## Why the result is bit-identical
 *
 * - The two bmms change **only the assignment of rows** (`GemmRowWindow` in
 *   {@link "../../kernels/gemm.ts"}). Neither the K reduction order of an output element nor the
 *   order of rounding moves by a single character (the numeric contract of
 *   src/kernels/gemm-geometry.ts). The geometry bucket may differ per block, but geometry decides
 *   only the assignment.
 * - `add` (mask is a `[1,1,1,N]` broadcast) and `safe_softmax` (an in-row reduction over the last
 *   dimension) are both **closed within a row**, so cutting rows does not change the op sequence
 *   per row. The existing kernels can therefore run as they are on the block buffers (no row
 *   offset needed).
 *
 * ## Always fused, static count
 *
 * A grabbed window is **always** fused, and {@link planRowBlocks} decides the count n from the
 * device's granted limit and the resolved shapes only. **On a machine with n = 1 it is exactly the
 * plain 4-dispatch list** (no row-window variant; the existing `bmmKey(v4, m)` key and existing
 * params), at zero extra cost.
 *
 * ## Application order and head collisions
 *
 * The leading op is `bmm`, disjoint from the other rules' leading ops (`rms_norm` / `sigmoid` /
 * `reshape` / `mul` / `slice` / `layer_norm`), so the declaration order does not affect the
 * result. The window contains five `reshape` / `expand`, but once grabbed the scan advances by the
 * window width, so no other rule can fire inside it (only when it is not grabbed is an inner
 * `reshape` tried as the head of upsample2x, which fails because the 6-node spelling differs).
 *
 * Total uses of external inputs: q / kᵀ / mask / v once each = 4.
 */
export const ROW_BLOCK_ATTENTION_RULE = defineRule<RowBlockAttentionMatch>({
  name: "rowBlockAttention",
  heads: ["bmm"],
  match: (nodes, index, context) => {
    const qk = nodes[index];
    if (qk?.node.op !== "bmm") return undefined;
    const reshapeScores = nodes[index + 1];
    const addMask = nodes[index + 2];
    const softmax = nodes[index + 3];
    const expandP = nodes[index + 4];
    const reshapeP = nodes[index + 5];
    const expandV = nodes[index + 6];
    const reshapeV = nodes[index + 7];
    const pv = nodes[index + 8];
    if (
      reshapeScores?.node.op !== "reshape" ||
      addMask?.node.op !== "add" ||
      softmax?.node.op !== SAFE_SOFTMAX_OP ||
      expandP?.node.op !== "expand" ||
      reshapeP?.node.op !== "reshape" ||
      expandV?.node.op !== "expand" ||
      reshapeV?.node.op !== "reshape" ||
      pv?.node.op !== "bmm"
    )
      return undefined;
    const chain = [qk, reshapeScores, addMask, softmax, expandP, reshapeP, expandV, reshapeV, pv];
    if (!allF32(chain)) return undefined;

    // Wiring (the window order alone cannot rule out "another chain of the same form next to it").
    if (
      reshapeScores.node.ins[0] !== qk.outputs[0].name ||
      addMask.node.ins[0] !== reshapeScores.outputs[0].name ||
      softmax.node.ins[0] !== addMask.outputs[0].name ||
      expandP.node.ins[0] !== softmax.outputs[0].name ||
      reshapeP.node.ins[0] !== expandP.outputs[0].name ||
      reshapeV.node.ins[0] !== expandV.outputs[0].name ||
      pv.node.ins[0] !== reshapeP.outputs[0].name ||
      pv.node.ins[1] !== reshapeV.outputs[0].name
    )
      return undefined;

    const [qShape, ktShape] = qk.inputShapes;
    if (qShape.length !== 3 || ktShape.length !== 3) return undefined;
    const [heads, queries, headDim] = qShape;
    const keys = ktShape[2];
    if (heads < 1 || queries < 1 || headDim < 1 || keys < 1) return undefined;
    const scores3 = [heads, queries, keys];
    const scores4 = [1, heads, queries, keys];
    if (
      !sameShape(ktShape, [heads, headDim, keys]) ||
      !sameShape(qk.outputs[0].shape, scores3) ||
      !sameShape(reshapeScores.outputs[0].shape, scores4) ||
      !sameShape(addMask.inputShapes[1], [1, 1, 1, keys]) ||
      !sameShape(addMask.outputs[0].shape, scores4) ||
      !sameShape(softmax.outputs[0].shape, scores4) ||
      // Unless the expand is an identity (no replicated axis), the plain list emits one
      // materialising copy.
      !sameShape(expandP.inputShapes[0], scores4) ||
      !sameShape(expandP.outputs[0].shape, scores4) ||
      !sameShape(reshapeP.outputs[0].shape, scores3) ||
      !sameShape(expandV.inputShapes[0], [1, heads, keys, headDim]) ||
      !sameShape(expandV.outputs[0].shape, [1, heads, keys, headDim]) ||
      !sameShape(reshapeV.outputs[0].shape, [heads, keys, headDim]) ||
      !sameShape(pv.outputs[0].shape, [heads, queries, headDim])
    )
      return undefined;
    // The reduction axis is fixed to the last dimension (the contract already checks it, but it is
    // stated again as part of the window's accepted set).
    if (softmaxDim(softmax.node.attrs ?? {}, "row-block attention safe_softmax") !== 3) {
      return undefined;
    }
    if (!internalsArePrivate(chain, context)) return undefined;

    const binds = [qk.node.ins[0], qk.node.ins[1], addMask.node.ins[1], expandV.node.ins[0]];
    if (new Set(binds).size !== binds.length) return undefined;

    // From here on is fail-loudly territory (a grabbed window is always fused). Dropping to the
    // plain list only widens the materialisation, so "cannot be split" is never turned into a
    // silent fallback.
    // The limit is the smaller of the binding limit and the budget ({@link rowBlockLimit}).
    // `limits` itself is not lowered — the dispatch-count checks and other bindings keep the
    // granted values.
    const blocks = planRowBlocks(
      queries,
      heads * keys * 4,
      rowBlockLimit(context.limits.maxStorageBufferBindingSize, context.attentionRowBlockMaxBytes),
      context.rowBlockSplit,
    );
    return {
      window: chain,
      chain,
      binds,
      outputName: pv.outputs[0].name,
      outputShape: pv.outputs[0].shape,
      heads,
      queries,
      keys,
      headDim,
      blocks,
      limits: context.limits,
    };
  },
  build: (matched) => {
    const { heads, queries, keys, headDim, blocks, limits } = matched;
    // n = 1 uses no row-window variant (existing key and params = identical to the plain 4-dispatch
    // list).
    const windowed = blocks.length > 1;
    const dispatchLimit = limits.maxComputeWorkgroupsPerDimension;
    const temps: FusedTemp[] = [];
    const dispatches: FusedDispatch[] = [];
    for (const block of blocks) {
      const first = dispatches.length;
      const rows = block.rows;
      const bytes = heads * rows * keys * 4;
      // All three temporaries are "returned once the next dispatch has read them" — exactly two are
      // alive at once, and the third gets the first one's storage by reallocation (across blocks
      // too).
      const scores = temps.length;
      temps.push({ byteLength: bytes, allocBefore: first, releaseAfter: first + 1 });
      const masked = temps.length;
      temps.push({ byteLength: bytes, allocBefore: first + 1, releaseAfter: first + 2 });
      const probabilities = temps.length;
      temps.push({ byteLength: bytes, allocBefore: first + 2, releaseAfter: first + 3 });

      // ① QK: only A (q) is read with the full-M stride + row offset; S is written as a block.
      const qkV4 = gemmUsesVec4(headDim, keys);
      const qkGeometry = gemmGeometryForRows(rows);
      const qkWhere = `row-block bmm(QK) [${heads},${rows},${headDim}] × [${heads},${headDim},${keys}]`;
      dispatches.push({
        key: bmmKey(qkV4, rows, windowed ? "a" : undefined),
        wgsl: () => bmmWgsl(qkV4, rows, windowed ? "a" : undefined),
        params: windowed
          ? bmmRowWindowParams(rows, keys, headDim, block.offset, queries)
          : bmmParams(rows, keys, headDim),
        operands: [
          { kind: "bind", index: 0 },
          { kind: "bind", index: 1 },
          { kind: "temp", id: scores },
        ],
        workgroups: {
          kind: "tiled",
          counts: [
            tiledWorkgroups(keys, gemmTileN(qkGeometry), dispatchLimit, qkWhere),
            tiledWorkgroups(rows, gemmTileM(qkGeometry), dispatchLimit, qkWhere),
            tiledWorkgroups(heads, 1, dispatchLimit, qkWhere),
          ],
        },
      });

      // ② The additive mask. It works on the block buffer, so the plain elementwise runs as is.
      const maskedShape = [1, heads, rows, keys];
      const elementwise: ElementwiseSpec = { op: "add", rank: 4, dtype: "f32" };
      dispatches.push({
        key: elementwiseKey(elementwise),
        wgsl: () => elementwiseWgsl(elementwise),
        params: elementwiseParams(elementwise, maskedShape, [maskedShape, [1, 1, 1, keys]]),
        paramsStorage: true,
        operands: [
          { kind: "temp", id: scores },
          { kind: "bind", index: 2 },
          { kind: "temp", id: masked },
        ],
        workgroups: {
          kind: "gridStride",
          items: numel(maskedShape),
          size: ELEMENTWISE_WORKGROUP_SIZE,
        },
      });

      // ③ safe_softmax (one row = one workgroup). Closed within a row, so cutting rows does not
      // move values.
      dispatches.push({
        key: SAFE_SOFTMAX_KEY,
        wgsl: () => SAFE_SOFTMAX_WGSL,
        params: softmaxParams(heads * rows, keys, true),
        operands: [
          { kind: "temp", id: masked },
          { kind: "temp", id: probabilities },
        ],
        workgroups: { kind: "gridStride", items: heads * rows, size: 1 },
      });

      // ④ PV: A (P) is read as a block; only the output is written with the full-M stride + row
      // offset.
      const pvV4 = gemmUsesVec4(keys, headDim);
      const pvGeometry = gemmGeometryForRows(rows);
      const pvWhere = `row-block bmm(PV) [${heads},${rows},${keys}] × [${heads},${keys},${headDim}]`;
      dispatches.push({
        key: bmmKey(pvV4, rows, windowed ? "c" : undefined),
        wgsl: () => bmmWgsl(pvV4, rows, windowed ? "c" : undefined),
        params: windowed
          ? bmmRowWindowParams(rows, headDim, keys, block.offset, queries)
          : bmmParams(rows, headDim, keys),
        operands: [
          { kind: "temp", id: probabilities },
          { kind: "bind", index: 3 },
          { kind: "output" },
        ],
        workgroups: {
          kind: "tiled",
          counts: [
            tiledWorkgroups(headDim, gemmTileN(pvGeometry), dispatchLimit, pvWhere),
            tiledWorkgroups(rows, gemmTileM(pvGeometry), dispatchLimit, pvWhere),
            tiledWorkgroups(heads, 1, dispatchLimit, pvWhere),
          ],
        },
      });
    }
    return {
      binds: matched.binds,
      outputName: matched.outputName,
      outputShape: matched.outputShape,
      temps,
      dispatches,
    };
  },
});
