/**
 * The exact peephole for RoPE (fusion rule rope — the runtime's src/runtime/fusion.ts) against the
 * primitive sequence on the same backend, u32 identity with NaN by class: the half-split form on
 * `[1,H,S,D]` (H = 1 and H ≥ 2, both orders of the direct product), the half-split form on
 * `[1,S,H,D]` (token / head axes, partial workgroups, widths), and the pair form of Wan's DiT (8
 * dispatches folded into 1).
 */

import type { GpuContext } from "../../../runtime/src/gpu/device.ts";
import {
  type DeclarationJson,
  fill,
  type FilledTensor,
  openModelBytes,
} from "../../../runtime/tests/helpers/model-fixture.ts";
import { checkFusionTwin } from "../fusion-twin.ts";
import { type CheckOptions, type CheckPart, type ExpectedCases, wordsOf } from "../result.ts";
import { twinCases, withTwinSessions } from "./session.ts";

export type RopeOrder = "slice-first" | "direct-first";

export const ROPE_ORDERS: readonly RopeOrder[] = ["slice-first", "direct-first"];

/**
 * `interpose = true` inserts a zero-dispatch reshape right after the first slice, which breaks only
 * the exact matcher without changing the expression. This lets the existing 8-dispatch sequence on
 * the same backend serve as the reference for a direct comparison.
 *
 * `heads` is the H axis of x. The kernel folds the head direction by dividing `row = i / head_dim`
 * by `sequence` to get the token, so that index is never exercised unless H ≥ 2 runs.
 */
export const ropeGraph = (
  order: RopeOrder,
  interpose: boolean,
  heads = 1,
  headDim = 128,
  layout: "bhsd" | "bshd" = "bhsd",
): DeclarationJson => {
  const shape =
    layout === "bshd" ? ([1, "S", heads, headDim] as const) : ([1, heads, "S", headDim] as const);
  const tableShape =
    layout === "bshd" ? ([1, "S", 1, headDim] as const) : ([1, 1, "S", headDim] as const);
  const halfShape =
    layout === "bshd"
      ? ([1, "S", heads, headDim / 2] as const)
      : ([1, heads, "S", headDim / 2] as const);
  const values: DeclarationJson["values"] = {
    first: { dtype: "f32", shape: [...halfShape] },
    second: { dtype: "f32", shape: [...halfShape] },
    negative: { dtype: "f32", shape: [...halfShape] },
    rotated: { dtype: "f32", shape: [...shape] },
    direct: { dtype: "f32", shape: [...shape] },
    cross: { dtype: "f32", shape: [...shape] },
    y: { dtype: "f32", shape: [...shape] },
  };
  if (interpose) values.first_alias = { dtype: "f32", shape: [...halfShape] };
  const direct: DeclarationJson["nodes"][number] = {
    op: "mul",
    ins: ["x", "cos"],
    outs: ["direct"],
    attrs: {},
  };
  const nodes: DeclarationJson["nodes"] = [];
  if (order === "direct-first") nodes.push(direct);
  nodes.push({
    op: "slice",
    ins: ["x"],
    outs: ["first"],
    attrs: { dim: 3, start: 0, end: headDim / 2 },
  });
  if (interpose) {
    nodes.push({ op: "reshape", ins: ["first"], outs: ["first_alias"], attrs: {} });
  }
  nodes.push(
    {
      op: "slice",
      ins: ["x"],
      outs: ["second"],
      attrs: { dim: 3, start: headDim / 2, end: headDim },
    },
    { op: "neg", ins: ["second"], outs: ["negative"], attrs: {} },
    {
      op: "cat",
      ins: ["negative", interpose ? "first_alias" : "first"],
      outs: ["rotated"],
      attrs: { dim: 3 },
    },
  );
  if (order === "slice-first") nodes.push(direct);
  nodes.push(
    { op: "mul", ins: ["rotated", "sin"], outs: ["cross"], attrs: {} },
    { op: "add", ins: ["direct", "cross"], outs: ["y"], attrs: {} },
  );
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: ["S"],
    inputs: [
      { name: "x", dtype: "f32", shape: [...shape] },
      { name: "cos", dtype: "f32", shape: [...tableShape] },
      { name: "sin", dtype: "f32", shape: [...tableShape] },
    ],
    outputs: ["y"],
    initializers: {},
    values,
    nodes,
  };
};

/**
 * The `[1,H,S,128]` input set, with ±0 / subnormals / ±Inf / NaN payloads written at the start of
 * x.
 */
export const ropeInputs = (
  heads: number,
  sequence: number,
): Readonly<Record<string, FilledTensor>> => {
  const x = fill([1, heads, sequence, 128], (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
  wordsOf(x.data).set([
    0x00000000, // +0
    0x80000000, // -0
    0x00000001, // smallest positive subnormal
    0x80000001, // smallest-magnitude negative subnormal
    0x007fffff, // largest positive subnormal
    0x807fffff, // largest-magnitude negative subnormal
    0x7f800000, // +Inf
    0xff800000, // -Inf
    0x7fc01234, // quiet NaN payload
    0xff801234, // negative signaling NaN payload
  ]);
  return {
    x,
    cos: fill([1, 1, sequence, 128], (i) => Math.cos(i * 0.19 + 0.3)),
    sin: fill([1, 1, sequence, 128], (i) => Math.sin(i * 0.23 - 0.2)),
  };
};

/** The `[1,S,H,D]` input set, with the same special words at the start of x. */
export const ropeBshdInputs = (
  heads: number,
  sequence: number,
  dim: number,
): Readonly<Record<string, FilledTensor>> => {
  const inputs = {
    x: fill([1, sequence, heads, dim], (i) => Math.sin(i * 0.37) * 3.1 + i / 997),
    cos: fill([1, sequence, 1, dim], (i) => Math.cos(i * 0.19 + 0.3)),
    sin: fill([1, sequence, 1, dim], (i) => Math.sin(i * 0.23 - 0.2)),
  };
  wordsOf(inputs.x.data).set([
    0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x7f800000, 0xff800000, 0x7fc01234,
    0xff801234,
  ]);
  return inputs;
};

/** (heads, sequence, dim) of the `[1,S,H,D]` form: token / head axes, partial workgroups, widths. */
export const ROPE_BSHD_SHAPES: readonly (readonly [number, number, number])[] = [
  [8, 1, 256],
  [16, 17, 64],
  [3, 17, 6],
  [4, 3, 128],
  [8, 9, 256],
  [2, 5, 512],
];

// ------------------------------------------------------------ pair form (Wan's DiT)

/**
 * The pair-form RoPE as the exporter writes it for Wan's DiT: `x` `[1,S,H,D]` is viewed as pairs
 * `[1,S,H·D/2,2]`, `slice(1..2), neg, slice(0..1), cat`, viewed back and combined with the
 * `[1,S,1,D]` tables.
 *
 * `interpose = true` puts a zero-dispatch reshape between `cat` and the reshape back: the values and
 * the physical dispatches stay the same, but the exact matcher no longer matches, so it is the
 * unfused reference on the same backend.
 * `separatePairs = true` makes `pairs` a graph input of its own, so the kernel's `pairs` binding is a
 * different buffer from `x` (the rotation must come from `pairs`, not from `x`).
 */
export const ropePairGraph = (
  heads: number,
  headDim: number,
  interpose: boolean,
  separatePairs = false,
): DeclarationJson => {
  const full = [1, "S", heads, headDim];
  const pairs = [1, "S", (heads * headDim) / 2, 2];
  const half = [1, "S", (heads * headDim) / 2, 1];
  const values: DeclarationJson["values"] = {
    odd: { dtype: "f32", shape: half },
    negative: { dtype: "f32", shape: half },
    even: { dtype: "f32", shape: half },
    rotated: { dtype: "f32", shape: pairs },
    rot: { dtype: "f32", shape: full },
    direct: { dtype: "f32", shape: full },
    cross: { dtype: "f32", shape: full },
    y: { dtype: "f32", shape: full },
  };
  const nodes: DeclarationJson["nodes"] = [];
  if (!separatePairs) {
    values.pairs = { dtype: "f32", shape: pairs };
    nodes.push({ op: "reshape", ins: ["x"], outs: ["pairs"], attrs: {} });
  }
  nodes.push(
    { op: "slice", ins: ["pairs"], outs: ["odd"], attrs: { dim: 3, start: 1, end: 2 } },
    { op: "neg", ins: ["odd"], outs: ["negative"], attrs: {} },
    { op: "slice", ins: ["pairs"], outs: ["even"], attrs: { dim: 3, start: 0, end: 1 } },
    { op: "cat", ins: ["negative", "even"], outs: ["rotated"], attrs: { dim: 3 } },
  );
  if (interpose) {
    values.rotated_alias = { dtype: "f32", shape: pairs };
    nodes.push({ op: "reshape", ins: ["rotated"], outs: ["rotated_alias"], attrs: {} });
  }
  nodes.push(
    { op: "reshape", ins: [interpose ? "rotated_alias" : "rotated"], outs: ["rot"], attrs: {} },
    { op: "mul", ins: ["x", "cos"], outs: ["direct"], attrs: {} },
    { op: "mul", ins: ["rot", "sin"], outs: ["cross"], attrs: {} },
    { op: "add", ins: ["direct", "cross"], outs: ["y"], attrs: {} },
  );
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: ["S"],
    inputs: [
      { name: "x", dtype: "f32", shape: full },
      ...(separatePairs ? [{ name: "pairs", dtype: "f32", shape: pairs }] : []),
      { name: "cos", dtype: "f32", shape: [1, "S", 1, headDim] },
      { name: "sin", dtype: "f32", shape: [1, "S", 1, headDim] },
    ],
    outputs: ["y"],
    initializers: {},
    values,
    nodes,
  };
};

/** ±0 / subnormals / ±Inf / NaN payloads at the head of a tensor (both signs, both pair slots). */
const SPECIAL_WORDS = [
  0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x7f800000, 0xff800000,
  0x7fc01234, 0xff801234,
];

export const ropePairInputs = (
  heads: number,
  sequence: number,
  headDim: number,
  separatePairs: boolean,
): Readonly<Record<string, FilledTensor>> => {
  const x = fill([1, sequence, heads, headDim], (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
  wordsOf(x.data).set(SPECIAL_WORDS.slice(0, x.data.length));
  const inputs: Record<string, FilledTensor> = {
    x,
    cos: fill([1, sequence, 1, headDim], (i) => Math.cos(i * 0.19 + 0.3)),
    sin: fill([1, sequence, 1, headDim], (i) => Math.sin(i * 0.23 - 0.2)),
  };
  if (separatePairs) {
    // Different values from x (and the special words shifted by one slot), so a kernel that
    // rotated x instead of pairs would differ.
    const pairs = fill([1, sequence, (heads * headDim) / 2, 2], (i) => Math.cos(i * 0.41) * -2.3);
    wordsOf(pairs.data).set(SPECIAL_WORDS.slice(0, pairs.data.length - 1), 1);
    inputs.pairs = pairs;
  }
  return inputs;
};

/**
 * (heads, sequence, headDim, separatePairs): Wan2.2 (24 × 128), Irodori (10 × 64 / 12 × 64), a
 * narrow width, one token, a wide head, lengths that leave a partial last workgroup, and `pairs` as
 * an input of its own.
 */
export const ROPE_PAIR_SHAPES: readonly (readonly [number, number, number, boolean])[] = [
  [24, 5, 128, false],
  [10, 17, 64, false],
  [12, 3, 64, false],
  [3, 17, 6, false],
  [4, 1, 128, false],
  [2, 9, 256, false],
  [3, 7, 128, true],
];

/** The label of a pair-form case. */
export const ropePairLabel = (
  heads: number,
  sequence: number,
  headDim: number,
  separatePairs: boolean,
): string => `H=${heads} S=${sequence} D=${headDim}${separatePairs ? " separate pairs" : ""}`;

const PART = "fusion-rope";

/**
 * The cases of {@link checkRopeTwins}: `[1,H,S,D]` at H = 1 and H = 3 × 2 orders; `[1,S,H,D]` at 6
 * shapes × 2 orders; the pair form at 7 shapes.
 */
export const ROPE_TWIN_CASES = { [PART]: 23 } as const satisfies ExpectedCases;

/**
 * The fused Session against the interposed one (checkFusionTwin, NaN by class), rope at 1 and 0,
 * the fused side in one dispatch (the plain side's 8 where the sequence is the exporter's 8 nodes
 * at H = 1 and in the pair form).
 */
export const checkRopeTwins = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const sink = twinCases(PART, ROPE_TWIN_CASES[PART], options);
  // [1,H,S,D] with H = 1 (RoPE in 1 pass against slice×2 + neg + cat×2 + mul×2 + add) and H = 3,
  // which exercises the head-direction index (the real distribution has H = 16).
  for (const [heads, sequence, plainDispatches] of [
    [1, 3, 8],
    [3, 5, undefined],
  ] as const) {
    const inputs = ropeInputs(heads, sequence);
    for (const order of ROPE_ORDERS) {
      sink.next();
      await withTwinSessions(
        gpu,
        `RoPE H=${heads} ${order}`,
        { model: await openModelBytes(ropeGraph(order, false, heads), []) },
        { model: await openModelBytes(ropeGraph(order, true, heads), []) },
        async ({ fused, plain }) => {
          sink.add(
            await checkFusionTwin(gpu, {
              label: `H=${heads} S=${sequence} ${order}`,
              fused,
              plain,
              feeds: inputs,
              counter: "rope",
              hits: 1,
              nan: "class",
              dispatches:
                plainDispatches === undefined ? { fused: 1 } : { fused: 1, plain: plainDispatches },
            }),
          );
        },
      );
    }
  }
  for (const [heads, sequence, dim] of ROPE_BSHD_SHAPES) {
    const inputs = ropeBshdInputs(heads, sequence, dim);
    for (const order of ROPE_ORDERS) {
      sink.next();
      await withTwinSessions(
        gpu,
        `BSHD RoPE H=${heads} S=${sequence} D=${dim} ${order}`,
        { model: await openModelBytes(ropeGraph(order, false, heads, dim, "bshd"), []) },
        { model: await openModelBytes(ropeGraph(order, true, heads, dim, "bshd"), []) },
        async ({ fused, plain }) => {
          sink.add(
            await checkFusionTwin(gpu, {
              label: `H=${heads} S=${sequence} D=${dim} ${order}`,
              fused,
              plain,
              feeds: inputs,
              counter: "rope",
              hits: 1,
              nan: "class",
              dispatches: { fused: 1 },
            }),
          );
        },
      );
    }
  }
  for (const [heads, sequence, headDim, separatePairs] of ROPE_PAIR_SHAPES) {
    sink.next();
    const label = ropePairLabel(heads, sequence, headDim, separatePairs);
    await withTwinSessions(
      gpu,
      `pair RoPE ${label}`,
      { model: await openModelBytes(ropePairGraph(heads, headDim, false, separatePairs), []) },
      { model: await openModelBytes(ropePairGraph(heads, headDim, true, separatePairs), []) },
      async ({ fused, plain }) => {
        sink.add(
          await checkFusionTwin(gpu, {
            label,
            fused,
            plain,
            feeds: ropePairInputs(heads, sequence, headDim, separatePairs),
            counter: "rope",
            hits: 1,
            nan: "class",
            // One pass against slice + neg + slice + cat×2 + mul×2 + add.
            dispatches: { fused: 1, plain: 8 },
          }),
        );
      },
    );
  }
  return {
    id: PART,
    title: "rope: fused vs slice → neg → cat → mul → add (half-split BHSD / BSHD, pair form)",
    cases: sink.cases,
    requirements: [],
  };
};
