/**
 * The row-block execution of decomposed attention (fusion rule `rowBlockAttention` — the runtime's
 * src/runtime/fusion.ts) against the plain 9-node sequence of a spelling the matcher cannot catch
 * (one zero-dispatch alias between bmm and reshape), on the same backend: the fusion split into 1
 * (the default), 2, 3 and 5 blocks by the test-only `ROW_BLOCK_SPLIT`, and split into 3 blocks by
 * the budget `attentionRowBlockMaxBytes` (ADR 0124). u32 identity with NaN by payload,
 * rowBlockAttention at 1 and 0.
 *
 * Row blocking only changes which workgroup handles which row, so a 1 ulp change means the K
 * reduction order or the rounding sequence moved. Separate graphs and separate Sessions on each side
 * (running the same path twice always agrees).
 */

import type { GpuContext } from "../../../runtime/src/gpu/device.ts";
import { ROW_BLOCK_SPLIT, type Session } from "../../../runtime/src/runtime/executor.ts";
import {
  type DeclarationJson,
  fill,
  type FilledTensor,
  openGraphModel,
} from "../../../runtime/tests/helpers/model-fixture.ts";
import { checkFusionTwin } from "../fusion-twin.ts";
import { type CheckOptions, type CheckPart, type ExpectedCases } from "../result.ts";
import { openTwinSession, twinCases } from "./session.ts";

export type RowBlockShape = {
  readonly name: string;
  /** The batch axis with B·H folded. */
  readonly heads: number;
  /** The query row count M. */
  readonly queries: number;
  /** The key column count N. */
  readonly keys: number;
  /** The head width D. */
  readonly headDim: number;
};

/** Deterministic data sequences (no random numbers — a failure would not reproduce). */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

/**
 * The `[1,1,1,N]` additive mask (the same "per column" broadcast as the observed shape).
 *
 * One −inf in every 4 columns exercises safe_softmax's −inf path. MUST: do not block every column —
 * safe_softmax writes all zeros for an all −inf row, so the bit comparison becomes the tautology
 * "0 equals 0".
 */
const columnMask = (keys: number): FilledTensor =>
  fill([1, 1, 1, keys], (i) => (i % 4 === 3 ? Number.NEGATIVE_INFINITY : 0));

/**
 * The observed spelling of decomposed attention (exactly the 9-node window of the irodori DiT).
 *
 * `interpose` only inserts one zero-dispatch alias between bmm and reshape. It breaks only the
 * window's adjacency condition without changing the values or the physical dispatch count, so it
 * becomes **the reference on the same backend** (the path before fusion).
 */
export const rowBlockAttentionGraph = (
  shape: RowBlockShape,
  interpose = false,
): DeclarationJson => {
  const { heads, queries, keys, headDim } = shape;
  const scores3 = [heads, queries, keys];
  const scores4 = [1, heads, queries, keys];
  const qkOut = interpose ? "scores3_alias" : "scores3";
  const nodes: DeclarationJson["nodes"] = [
    { op: "bmm", ins: ["q", "kt"], outs: [qkOut], attrs: {} },
    ...(interpose ? [{ op: "reshape", ins: [qkOut], outs: ["scores3"], attrs: {} }] : []),
    { op: "reshape", ins: ["scores3"], outs: ["scores4"], attrs: {} },
    { op: "add", ins: ["scores4", "mask"], outs: ["masked"], attrs: {} },
    { op: "safe_softmax", ins: ["masked"], outs: ["probs"], attrs: { dim: 3 } },
    { op: "expand", ins: ["probs"], outs: ["probsExpanded"], attrs: {} },
    { op: "reshape", ins: ["probsExpanded"], outs: ["probs3"], attrs: {} },
    { op: "expand", ins: ["v"], outs: ["vExpanded"], attrs: {} },
    { op: "reshape", ins: ["vExpanded"], outs: ["v3"], attrs: {} },
    { op: "bmm", ins: ["probs3", "v3"], outs: ["y"], attrs: {} },
  ];
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs: [
      { name: "q", dtype: "f32", shape: [heads, queries, headDim] },
      { name: "kt", dtype: "f32", shape: [heads, headDim, keys] },
      { name: "mask", dtype: "f32", shape: [1, 1, 1, keys] },
      { name: "v", dtype: "f32", shape: [1, heads, keys, headDim] },
    ],
    outputs: ["y"],
    initializers: {},
    values: {
      ...(interpose ? { scores3_alias: { dtype: "f32", shape: scores3 } } : {}),
      scores3: { dtype: "f32", shape: scores3 },
      scores4: { dtype: "f32", shape: scores4 },
      masked: { dtype: "f32", shape: scores4 },
      probs: { dtype: "f32", shape: scores4 },
      probsExpanded: { dtype: "f32", shape: scores4 },
      probs3: { dtype: "f32", shape: scores3 },
      vExpanded: { dtype: "f32", shape: [1, heads, keys, headDim] },
      v3: { dtype: "f32", shape: [heads, keys, headDim] },
      y: { dtype: "f32", shape: [heads, queries, headDim] },
    },
    nodes,
  };
};

export const rowBlockAttentionInputs = (
  shape: RowBlockShape,
): Readonly<Record<string, FilledTensor>> => ({
  q: fill([shape.heads, shape.queries, shape.headDim], QUERY),
  kt: fill([shape.heads, shape.headDim, shape.keys], KEY),
  mask: columnMask(shape.keys),
  v: fill([1, shape.heads, shape.keys, shape.headDim], VALUE),
});

export const ROW_BLOCK_SHAPES: readonly RowBlockShape[] = [
  // Every axis uneven (none of M / N / D is a multiple of 4 = both bmm take the scalar variant)
  { name: "uneven H3 M17 N19 D13", heads: 3, queries: 17, keys: 19, headDim: 13 },
  // On the v4 path (D % 4 == 0 && N % 4 == 0), M spans two row tiles. Splitting also crosses the
  // geometry buckets (M ≤ 64 / ≤ 512), so this confirms "the values do not move when the geometry
  // changes".
  { name: "v4 H2 M68 N20 D12", heads: 2, queries: 68, keys: 20, headDim: 12 },
  // A shrunken DiT self-attention (D = 128, M / N both multiples of 64 = the real model's path)
  { name: "DiT shape H4 M64 N64 D128", heads: 4, queries: 64, keys: 64, headDim: 128 },
];

/** The forced block counts (1 = the same shape as the default, 2/3 split evenly, 5 unevenly). */
export const ROW_BLOCK_SPLITS: readonly number[] = [1, 2, 3, 5];

/**
 * The budget of ⌈M/3⌉ rows per block (3 blocks for every shape — 17 rows are 6 + 6 + 5, 68 rows are
 * 23 + 23 + 22, 64 rows are 22 + 21 + 21).
 */
export const rowBlockBudget = (shape: RowBlockShape): number =>
  shape.heads * shape.keys * 4 * Math.ceil(shape.queries / 3);

/** Opens the fixture graph (fused, or the plain spelling with `interpose`) with a split or a budget. */
const openRowBlockSession = async (
  gpu: GpuContext,
  where: string,
  shape: RowBlockShape,
  interpose: boolean,
  split?: number,
  budget?: number,
): Promise<Session> =>
  openTwinSession(gpu, where, await openGraphModel(rowBlockAttentionGraph(shape, interpose)), {
    ...(split === undefined ? {} : { [ROW_BLOCK_SPLIT]: split }),
    ...(budget === undefined ? {} : { attentionRowBlockMaxBytes: budget }),
  });

const PART = "fusion-row-block-attention";

/** The cases of {@link checkRowBlockAttentionTwins}: 3 shapes × 4 forced splits; 3 budgets. */
export const ROW_BLOCK_ATTENTION_TWIN_CASES = { [PART]: 15 } as const satisfies ExpectedCases;

/**
 * Every shape at every forced split, then every shape split by its budget: the fused Session
 * against the plain 9-node sequence (checkFusionTwin, NaN by payload), rowBlockAttention at 1 and 0.
 */
export const checkRowBlockAttentionTwins = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const sink = twinCases(PART, ROW_BLOCK_ATTENTION_TWIN_CASES[PART], options);
  const twin = async (
    label: string,
    shape: RowBlockShape,
    fused: Session,
    plain: Session,
  ): Promise<void> => {
    sink.add(
      await checkFusionTwin(gpu, {
        label,
        fused,
        plain,
        feeds: rowBlockAttentionInputs(shape),
        counter: "rowBlockAttention",
        hits: 1,
        nan: "payload",
      }),
    );
  };
  for (const shape of ROW_BLOCK_SHAPES) {
    sink.next();
    // The reference = a spelling the matcher cannot catch (the plain 9-node sequence that never
    // fuses), shared by the splits of the shape.
    const reference = await openRowBlockSession(gpu, `${shape.name} reference`, shape, true);
    try {
      for (const split of ROW_BLOCK_SPLITS) {
        sink.next();
        const label = `${shape.name} / ${split} blocks`;
        const fused = await openRowBlockSession(gpu, label, shape, false, split);
        try {
          await twin(label, shape, fused, reference);
        } finally {
          await fused.dispose();
        }
      }
    } finally {
      await reference.dispose();
    }
  }
  for (const shape of ROW_BLOCK_SHAPES) {
    sink.next();
    const budget = rowBlockBudget(shape);
    const label = `${shape.name} / budget ${budget}B`;
    const reference = await openRowBlockSession(gpu, `${label} reference`, shape, true);
    try {
      const budgeted = await openRowBlockSession(gpu, label, shape, false, undefined, budget);
      try {
        await twin(label, shape, budgeted, reference);
      } finally {
        await budgeted.dispose();
      }
    } finally {
      await reference.dispose();
    }
  }
  return {
    id: PART,
    title: "rowBlockAttention: forced and budgeted row blocks vs the plain 9-node attention",
    cases: sink.cases,
    requirements: [],
  };
};
