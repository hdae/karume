/**
 * adaLN = the strict peephole for `layer_norm → (1 + scale) → mul → add` (fusion rule adaln — the
 * runtime's src/runtime/fusion.ts) against the plain sequence on the same backend: the broadcast
 * form in its windows of 6 / 7 nodes (4 dispatches folded into 1) and Wan2.2's per-token form (6
 * dispatches folded into 2), u32 identity with NaN by class.
 *
 * The fused kernel removes the storage round trips of the 3 intermediates (normalized result /
 * `1 + scale` / product), so if its rounding points drift from the plain sequence the bits of finite
 * values move. Whether the rounding barrier of the workgroup u32 staging works on a real backend can
 * only be seen on that backend.
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
 * The row length exercises a remainder of the 256-thread workgroup; the row count makes grid-stride
 * loop at least once.
 */
const ROWS = 5;
const DIM = 257;
const ROW_SHAPE = [1, ROWS, DIM] as const;
const MOD_SHAPE = [1, 1, DIM] as const;

export type AdalnGraphOptions = {
  /** Also place the gate's reshape, as measured (window 7). false gives window 6. */
  readonly gate?: boolean;
  /**
   * Interpose one 0-dispatch alias on the normalized result. It removes only the wiring condition
   * of the strict matcher without changing values or the physical dispatch count (= builds the
   * reference on the same backend).
   */
  readonly interpose?: boolean;
  /**
   * Make the normalized result a graph output (a counterexample removing the private-internal-value
   * condition).
   */
  readonly normOutput?: boolean;
  /** A counterexample adding another consumer to `1 + scale`. */
  readonly modulationExtraConsumer?: boolean;
};

export const adalnGraph = (options: AdalnGraphOptions = {}): DeclarationJson => {
  const values: DeclarationJson["values"] = {
    t: { dtype: "f32", shape: [...ROW_SHAPE] },
    shift: { dtype: "f32", shape: [...MOD_SHAPE] },
    scale: { dtype: "f32", shape: [...MOD_SHAPE] },
    s: { dtype: "f32", shape: [...MOD_SHAPE] },
    p: { dtype: "f32", shape: [...ROW_SHAPE] },
    y: { dtype: "f32", shape: [...ROW_SHAPE] },
  };
  const inputs: DeclarationJson["inputs"] = [
    { name: "x", dtype: "f32", shape: [...ROW_SHAPE] },
    { name: "ln_weight", dtype: "f32", shape: [DIM] },
    { name: "ln_bias", dtype: "f32", shape: [DIM] },
    { name: "one", dtype: "f32", shape: [1] },
    { name: "shift_src", dtype: "f32", shape: [1, DIM] },
    { name: "scale_src", dtype: "f32", shape: [1, DIM] },
  ];
  const nodes: DeclarationJson["nodes"] = [
    {
      op: "layer_norm",
      ins: ["x", "ln_weight", "ln_bias"],
      outs: ["t"],
      attrs: { normalized_shape: [DIM], eps: 0.000001 },
    },
  ];
  let normForMul = "t";
  if (options.interpose) {
    values.t_alias = { dtype: "f32", shape: [...ROW_SHAPE] };
    nodes.push({ op: "reshape", ins: ["t"], outs: ["t_alias"], attrs: {} });
    normForMul = "t_alias";
  }
  nodes.push(
    { op: "reshape", ins: ["shift_src"], outs: ["shift"], attrs: {} },
    { op: "reshape", ins: ["scale_src"], outs: ["scale"], attrs: {} },
  );
  if (options.gate !== false) {
    inputs.push({ name: "gate_src", dtype: "f32", shape: [1, DIM] });
    values.gate = { dtype: "f32", shape: [...MOD_SHAPE] };
    nodes.push({ op: "reshape", ins: ["gate_src"], outs: ["gate"], attrs: {} });
  }
  nodes.push(
    { op: "add", ins: ["scale", "one"], outs: ["s"], attrs: {} },
    { op: "mul", ins: [normForMul, "s"], outs: ["p"], attrs: {} },
    { op: "add", ins: ["p", "shift"], outs: ["y"], attrs: {} },
  );
  if (options.modulationExtraConsumer) {
    values.s_copy = { dtype: "f32", shape: [...MOD_SHAPE] };
    nodes.push({ op: "neg", ins: ["s"], outs: ["s_copy"], attrs: {} });
  }
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs,
    outputs: [
      ...(options.normOutput ? ["t"] : []),
      "y",
      // gate is a value consumed outside the chain (in real IR the next block consumes it).
      ...(options.gate !== false ? ["gate"] : []),
      ...(options.modulationExtraConsumer ? ["s_copy"] : []),
    ],
    initializers: {},
    values,
    nodes,
  };
};

/**
 * Each row has a different character. Rows 0–2 are finite (small and large variance, signed zeros),
 * row 3 pushes the magnitude to hit cancellation in the normalization, and row 4 mixes in ±Inf /
 * NaN to make the whole row NaN.
 */
const adalnX = (): FilledTensor => {
  const x = fill(ROW_SHAPE, (i) => {
    const row = Math.floor(i / DIM);
    const at = i % DIM;
    if (row === 0) return Math.sin(at * 0.37) * 3.1 + at / 997;
    if (row === 1) return Math.cos(at * 0.11) * 1e-4;
    if (row === 2) return (at % 2 === 0 ? 1 : -1) * (at / 3);
    if (row === 3) return Math.sin(at * 0.017) * 1e5 + 1e6;
    return Math.tan(at * 0.05);
  });
  // Inject non-finite values into row 4 (mean becomes NaN and the whole row falls into the NaN
  // class).
  wordsOf(x.data).set([0x7f800000, 0xff800000, 0x7fc01234], 4 * DIM);
  // The head of row 2 is ±0 / subnormal / the smallest normal number.
  wordsOf(x.data).set([0x00000000, 0x80000000, 0x00000001, 0x807fffff, 0x00800000], 2 * DIM);
  return x;
};

export const adalnInputs = (
  options: AdalnGraphOptions = {},
): Readonly<Record<string, FilledTensor>> => ({
  x: adalnX(),
  ln_weight: fill([DIM], (i) => 1 + Math.sin(i * 0.21) * 0.4),
  ln_bias: fill([DIM], (i) => Math.cos(i * 0.13) * 0.2),
  one: fill([1], () => 1),
  // scale ranges from near ±0 to fairly large (if `t · (scale + 1)` gets distributed, finite values
  // break).
  shift_src: fill([1, DIM], (i) => Math.sin(i * 0.07) * 2.5),
  scale_src: fill([1, DIM], (i) => Math.cos(i * 0.29) * 3 - 1),
  ...(options.gate !== false ? { gate_src: fill([1, DIM], (i) => i / 31) } : {}),
});

/** With the gate's reshape (window 7) and without it (window 6). */
export const ADALN_GATES: readonly boolean[] = [true, false];

// ------------------------------------------------------------ per-token form (Wan2.2's DiT)

/**
 * Wan2.2's per-token adaLN: `layer_norm, where(mask, scale_c, scale_g), add(+one), mul,
 * where(mask, shift_c, shift_g), add` — 6 contiguous nodes.
 *
 * `interpose = true` puts a zero-dispatch reshape on the normalised value (the same trick as the
 * broadcast form): the values and the physical dispatches stay the same and only the exact matcher
 * stops matching, so it is the unfused reference on the same backend.
 */
export const adalnTokenGraph = (rows: number, dim: number, interpose: boolean): DeclarationJson => {
  const row = [1, rows, dim];
  const vector = [1, 1, dim];
  const values: DeclarationJson["values"] = {
    t: { dtype: "f32", shape: row },
    sel_scale: { dtype: "f32", shape: row },
    s: { dtype: "f32", shape: row },
    p: { dtype: "f32", shape: row },
    sel_shift: { dtype: "f32", shape: row },
    y: { dtype: "f32", shape: row },
  };
  const nodes: DeclarationJson["nodes"] = [
    {
      op: "layer_norm",
      ins: ["x", "ln_weight", "ln_bias"],
      outs: ["t"],
      attrs: { normalized_shape: [dim], eps: 0.000001 },
    },
  ];
  if (interpose) {
    values.t_alias = { dtype: "f32", shape: row };
    nodes.push({ op: "reshape", ins: ["t"], outs: ["t_alias"], attrs: {} });
  }
  nodes.push(
    { op: "where", ins: ["mask", "scale_c", "scale_g"], outs: ["sel_scale"], attrs: {} },
    { op: "add", ins: ["sel_scale", "one"], outs: ["s"], attrs: {} },
    { op: "mul", ins: [interpose ? "t_alias" : "t", "s"], outs: ["p"], attrs: {} },
    { op: "where", ins: ["mask", "shift_c", "shift_g"], outs: ["sel_shift"], attrs: {} },
    { op: "add", ins: ["p", "sel_shift"], outs: ["y"], attrs: {} },
  );
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))] },
    symbols: [],
    inputs: [
      { name: "x", dtype: "f32", shape: row },
      { name: "ln_weight", dtype: "f32", shape: [dim] },
      { name: "ln_bias", dtype: "f32", shape: [dim] },
      { name: "mask", dtype: "bool", shape: [1, rows, 1] },
      { name: "one", dtype: "f32", shape: [1] },
      { name: "scale_c", dtype: "f32", shape: vector },
      { name: "scale_g", dtype: "f32", shape: vector },
      { name: "shift_c", dtype: "f32", shape: vector },
      { name: "shift_g", dtype: "f32", shape: vector },
    ],
    outputs: ["y"],
    initializers: {},
    values,
    nodes,
  };
};

/**
 * Rows of different character (as {@link adalnX}): wide / tiny variance, signed alternation, a large
 * offset that cancels in the mean, a row with ±Inf / NaN, and ±0 / subnormals at the head of row 2.
 */
const tokenX = (rows: number, dim: number): FilledTensor => {
  const x = fill([1, rows, dim], (i) => {
    const row = Math.floor(i / dim);
    const at = i % dim;
    switch (row % 5) {
      case 0:
        return Math.sin(at * 0.37 + row) * 3.1 + at / 997;
      case 1:
        return Math.cos(at * 0.11) * 1e-4;
      case 2:
        return (at % 2 === 0 ? 1 : -1) * (at / 3);
      case 3:
        return Math.sin(at * 0.017) * 1e5 + 1e6;
      default:
        return Math.tan(at * 0.05);
    }
  });
  if (rows > 4) wordsOf(x.data).set([0x7f800000, 0xff800000, 0x7fc01234], 4 * dim);
  if (rows > 2) {
    wordsOf(x.data).set([0x00000000, 0x80000000, 0x00000001, 0x807fffff, 0x00800000], 2 * dim);
  }
  return x;
};

/**
 * The four vectors have different values and signs (so picking the wrong side changes the bits),
 * with -0, a subnormal and a NaN payload in fixed slots.
 */
export const adalnTokenInputs = (
  rows: number,
  dim: number,
  mask: (row: number) => number,
): Readonly<Record<string, FilledTensor>> => {
  const scaleCondition = fill([1, 1, dim], (i) => Math.cos(i * 0.29) * 3 - 1);
  const scaleGeneration = fill([1, 1, dim], (i) => Math.sin(i * 0.31) * 2.5 + 0.5);
  const shiftCondition = fill([1, 1, dim], (i) => Math.sin(i * 0.07) * 2.5);
  const shiftGeneration = fill([1, 1, dim], (i) => -Math.cos(i * 0.05) * 1.5);
  wordsOf(scaleCondition.data).set([0x80000000, 0x00000001], 0);
  wordsOf(scaleGeneration.data).set([0x807fffff], 1);
  wordsOf(shiftCondition.data).set([0x7fc01234], 2);
  wordsOf(shiftGeneration.data).set([0x80000000, 0x00000003], 3);
  return {
    x: tokenX(rows, dim),
    ln_weight: fill([dim], (i) => 1 + Math.sin(i * 0.21) * 0.4),
    ln_bias: fill([dim], (i) => Math.cos(i * 0.13) * 0.2),
    mask: fill([1, rows, 1], mask, "bool"),
    one: fill([1], () => 1),
    scale_c: scaleCondition,
    scale_g: scaleGeneration,
    shift_c: shiftCondition,
    shift_g: shiftGeneration,
  };
};

/** Mixed (the I2V shape: the first latent frame is the condition side), all condition, all generation. */
export const ADALN_TOKEN_MASKS: readonly (readonly [string, (row: number) => number])[] = [
  ["mixed mask", (row) => (row % 3 === 1 ? 0 : 1)],
  ["all condition", () => 1],
  ["all generation", () => 0],
];

/** (rows, dim): 257 = a partial last block; 3072 = Wan2.2's width; 8 = narrower than one block. */
export const ADALN_TOKEN_SHAPES: readonly (readonly [number, number])[] = [
  [5, 257],
  [7, 3072],
  [4, 8],
];

const PART = "fusion-adaln";

/** The cases of {@link checkAdalnTwins}: windows 7 and 6; 3 per-token shapes × 3 masks. */
export const ADALN_TWIN_CASES = { [PART]: 11 } as const satisfies ExpectedCases;

/**
 * The broadcast form in both windows (y; one dispatch against layer_norm + add + mul + add) and the
 * per-token form at every shape and mask (two dispatches against six): the fused Session against
 * the interposed one (checkFusionTwin, NaN by class), adaln at 1 and 0.
 */
export const checkAdalnTwins = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const sink = twinCases(PART, ADALN_TWIN_CASES[PART], options);
  for (const gate of ADALN_GATES) {
    sink.next();
    const label = `window ${gate ? 7 : 6}`;
    await withTwinSessions(
      gpu,
      `adaLN ${label}`,
      { model: await openModelBytes(adalnGraph({ gate }), []) },
      { model: await openModelBytes(adalnGraph({ gate, interpose: true }), []) },
      async ({ fused, plain }) => {
        // If the fusion silently drops off, the values stay correct and only dispatches grow. Pin it
        // directly with the counter.
        sink.add(
          await checkFusionTwin(gpu, {
            label: `${label}: fused vs primitive`,
            fused,
            plain,
            feeds: adalnInputs({ gate }),
            counter: "adaln",
            hits: 1,
            nan: "class",
            outputs: ["y"],
            // One dispatch against layer_norm + add + mul + add.
            dispatches: { fused: 1, plain: 4 },
          }),
        );
      },
    );
  }
  for (const [rows, dim] of ADALN_TOKEN_SHAPES) {
    sink.next();
    await withTwinSessions(
      gpu,
      `per-token adaLN rows=${rows} dim=${dim}`,
      { model: await openModelBytes(adalnTokenGraph(rows, dim, false), []) },
      { model: await openModelBytes(adalnTokenGraph(rows, dim, true), []) },
      async ({ fused, plain }) => {
        for (const [maskLabel, mask] of ADALN_TOKEN_MASKS) {
          sink.next();
          sink.add(
            await checkFusionTwin(gpu, {
              label: `rows=${rows} dim=${dim} ${maskLabel}: fused vs primitive`,
              fused,
              plain,
              feeds: adalnTokenInputs(rows, dim, mask),
              counter: "adaln",
              hits: 1,
              nan: "class",
              // pack + body against layer_norm + where + add + mul + where + add.
              dispatches: { fused: 2, plain: 6 },
            }),
          );
        }
      },
    );
  }
  return {
    id: PART,
    title: "adaln: fused vs layer_norm → (1 + scale) → mul → add (broadcast and per-token)",
    cases: sink.cases,
    requirements: [],
  };
};
