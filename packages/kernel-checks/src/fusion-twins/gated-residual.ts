/**
 * The gated residual `add(residual, mul(y, where(mask, gate_c, gate_g)))` as one dispatch (fusion
 * rule gatedResidual — the runtime's src/runtime/fusion.ts) against the unfused primitive sequence
 * on the same backend: u32 identity with NaN by class, and 3 dispatches folded into 1.
 *
 * The fused kernel drops the storage round trips of the selected gate and of the product, so a
 * rounding point that moved (an fma contraction of `residual + y * g`) shows up directly as a bit
 * difference of finite values. Whether the workgroup u32 staging works as a rounding barrier on a
 * real backend is visible only on that backend.
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

/**
 * `interpose = true` puts a zero-dispatch reshape on the selected gate: the values and the physical
 * dispatches stay the same and only the exact matcher stops matching, so it is the unfused
 * reference on the same backend.
 */
export const gatedResidualGraph = (
  rows: number,
  dim: number,
  interpose: boolean,
): DeclarationJson => {
  const row = [1, rows, dim];
  const vector = [1, 1, dim];
  const values: DeclarationJson["values"] = {
    g: { dtype: "f32", shape: row },
    p: { dtype: "f32", shape: row },
    out: { dtype: "f32", shape: row },
  };
  const nodes: DeclarationJson["nodes"] = [
    { op: "where", ins: ["mask", "gate_c", "gate_g"], outs: ["g"], attrs: {} },
  ];
  if (interpose) {
    values.g_alias = { dtype: "f32", shape: row };
    nodes.push({ op: "reshape", ins: ["g"], outs: ["g_alias"], attrs: {} });
  }
  nodes.push(
    { op: "mul", ins: ["y", interpose ? "g_alias" : "g"], outs: ["p"], attrs: {} },
    { op: "add", ins: ["residual", "p"], outs: ["out"], attrs: {} },
  );
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs: [
      { name: "mask", dtype: "bool", shape: [1, rows, 1] },
      { name: "gate_c", dtype: "f32", shape: vector },
      { name: "gate_g", dtype: "f32", shape: vector },
      { name: "y", dtype: "f32", shape: row },
      { name: "residual", dtype: "f32", shape: row },
    ],
    outputs: ["out"],
    initializers: {},
    values,
    nodes,
  };
};

/**
 * Inputs whose product and residual are of similar magnitude with both signs (so a contracted
 * `residual + y * g` would round differently), the two gates differ in value and sign (so picking
 * the wrong side changes the bits), and ±0 / subnormals / ±Inf / NaN payloads sit in fixed slots.
 */
export const gatedResidualInputs = (
  rows: number,
  dim: number,
  mask: (row: number) => number,
): Readonly<Record<string, FilledTensor>> => {
  const gateCondition = fill([1, 1, dim], (i) => Math.cos(i * 0.29) * 1.7 - 0.3);
  const gateGeneration = fill([1, 1, dim], (i) => -Math.sin(i * 0.13) * 2.1 + 0.2);
  const y = fill([1, rows, dim], (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
  const residual = fill([1, rows, dim], (i) => -Math.cos(i * 0.23) * 2.9 + i / 1301);
  wordsOf(gateCondition.data).set([0x80000000, 0x00000001], 0);
  wordsOf(gateGeneration.data).set([0x807fffff, 0x7fc01234], 2);
  wordsOf(y.data).set(
    [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x7f800000, 0xff800000, 0x7fc01234],
    dim,
  );
  wordsOf(residual.data).set([0x80000000, 0x007fffff, 0xff800000, 0x00000000, 0xff801234], dim + 2);
  return {
    mask: fill([1, rows, 1], mask, "bool"),
    gate_c: gateCondition,
    gate_g: gateGeneration,
    y,
    residual,
  };
};

/** Mixed, all condition, all generation. */
export const GATED_RESIDUAL_MASKS: readonly (readonly [string, (row: number) => number])[] = [
  ["mixed mask", (row) => (row % 3 === 1 ? 0 : 1)],
  ["all condition", () => 1],
  ["all generation", () => 0],
];

/**
 * (rows, dim): 257 = a row that is not a multiple of the workgroup; 3072 = Wan2.2's width; 8 = rows
 * shorter than one workgroup.
 */
export const GATED_RESIDUAL_SHAPES: readonly (readonly [number, number])[] = [
  [5, 257],
  [7, 3072],
  [4, 8],
];

const PART = "fusion-gated-residual";

/** The cases of {@link checkGatedResidualTwins}: 3 shapes × 3 masks. */
export const GATED_RESIDUAL_TWIN_CASES = { [PART]: 9 } as const satisfies ExpectedCases;

/**
 * Every shape and mask: the fused Session against the interposed one (checkFusionTwin, NaN by
 * class), gatedResidual at 1 and 0, one dispatch against where + mul + add.
 */
export const checkGatedResidualTwins = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const sink = twinCases(PART, GATED_RESIDUAL_TWIN_CASES[PART], options);
  for (const [rows, dim] of GATED_RESIDUAL_SHAPES) {
    sink.next();
    await withTwinSessions(
      gpu,
      `gatedResidual rows=${rows} dim=${dim}`,
      { model: await openModelBytes(gatedResidualGraph(rows, dim, false), []) },
      { model: await openModelBytes(gatedResidualGraph(rows, dim, true), []) },
      async ({ fused, plain }) => {
        for (const [maskLabel, mask] of GATED_RESIDUAL_MASKS) {
          sink.next();
          sink.add(
            await checkFusionTwin(gpu, {
              label: `rows=${rows} dim=${dim} ${maskLabel}: fused vs primitive`,
              fused,
              plain,
              feeds: gatedResidualInputs(rows, dim, mask),
              counter: "gatedResidual",
              hits: 1,
              nan: "class",
              // One pass against where + mul + add.
              dispatches: { fused: 1, plain: 3 },
            }),
          );
        }
      },
    );
  }
  return {
    id: PART,
    title: "gatedResidual: fused vs where → mul → add",
    cases: sink.cases,
    requirements: [],
  };
};
