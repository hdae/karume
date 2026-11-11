/**
 * gelu_tanh and the erf gelu folded into the step ① of the a8 linear that reads them (perf-ledger
 * F-4 — the linear prologue of the runtime's src/runtime/fusion.ts, executed by `buildLinearI8a8`)
 * against the unfused pair on the same backend, for both integer dot forms: u32 identity with NaN by
 * class, linearPrologue at 1 and 0, 2 dispatches against 3.
 *
 * The fused Session never stores `op(x)`; its quantize reads `x` and applies the op in registers.
 * The reference is the same graph with a zero-dispatch reshape between the two nodes: the values and
 * the physical kernels stay the production pair, only the exact match stops matching.
 *
 * The dp4a cases force `dot4I8Packed` (WGSL `packed_4x8_integer_dot_product`), so the part needs that
 * language feature ({@link linearPrologueShortfall}).
 */

import type { PrologueUnaryOp } from "../../../runtime/src/codegen/elementwise.ts";
import { perChannelGroupSize } from "../../../runtime/src/format/container/codecs.ts";
import type { GpuContext } from "../../../runtime/src/gpu/device.ts";
import { DP4A_WGSL_FEATURE, dp4aAvailable } from "../../../runtime/src/kernels/linear-i8a8.ts";
import { I8A8_DOT, type I8a8Dot } from "../../../runtime/src/runtime/executor.ts";
import { quantizeI8 } from "../../../runtime/tests/helpers/i8.ts";
import {
  type DeclarationJson,
  f32Bytes,
  fill,
  type FilledTensor,
  GRAPH_NAME,
  openModelBytes,
} from "../../../runtime/tests/helpers/model-fixture.ts";
import { checkFusionTwin } from "../fusion-twin.ts";
import { type CheckOptions, type CheckPart, type ExpectedCases } from "../result.ts";
import { twinCases, withTwinSessions } from "./session.ts";

/**
 * `y = linear(op(x), w, b)` with `w` resident as i8. `interpose = true` puts a zero-dispatch reshape
 * on the op's output (the unfused reference on the same backend).
 */
export const linearPrologueGraph = (
  op: PrologueUnaryOp,
  rows: number,
  k: number,
  n: number,
  interpose: boolean,
): DeclarationJson => {
  const row = [1, rows, k];
  const values: DeclarationJson["values"] = {
    w: { dtype: "f32", shape: [n, k] },
    b: { dtype: "f32", shape: [n] },
    g: { dtype: "f32", shape: row },
    y: { dtype: "f32", shape: [1, rows, n] },
  };
  const nodes: DeclarationJson["nodes"] = [{ op, ins: ["x"], outs: ["g"], attrs: {} }];
  if (interpose) {
    values.g_alias = { dtype: "f32", shape: row };
    nodes.push({ op: "reshape", ins: ["g"], outs: ["g_alias"], attrs: {} });
  }
  nodes.push({
    op: "linear",
    ins: [interpose ? "g_alias" : "g", "w", "b"],
    outs: ["y"],
    attrs: {},
  });
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))].sort() },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: row }],
    outputs: ["y"],
    initializers: { w: {}, b: {} },
    values,
    nodes,
  };
};

/** The i8 weight (per-output-channel scales of different sizes) and the bias. */
export const linearPrologueWeights = (k: number, n: number) => {
  const weight = fill([n, k], (i) => Math.sin(i * 0.61) * (1 + (Math.floor(i / k) % 5) * 0.5));
  const quantized = quantizeI8(weight.data, [n, k], 0);
  const bias = fill([n], (i) => Math.cos(i * 0.7) * 0.5);
  return [
    {
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: new Uint8Array(bias.data.buffer.slice(0)),
      encoding: { codec: "f32" },
    },
    {
      graph: GRAPH_NAME,
      initializer: "w",
      bytes: quantized.bytes,
      encoding: {
        codec: "int8-sym",
        groupSize: perChannelGroupSize(k),
        scale: { bytes: f32Bytes(quantized.scale), dtype: "f32" },
      },
    },
  ] as const;
};

/**
 * Activations of the DiT's pre-gelu magnitude (both signs, up to |x| ≈ 14.7: past the point where
 * both ops saturate to x and to −0), one all-zero row and one row with a NaN (its output row is NaN
 * in both forms).
 */
export const linearPrologueActivations = (rows: number, k: number): FilledTensor => {
  const x = fill([1, rows, k], (i) => Math.sin(i * 0.37) * 3.1 + ((i % 101) - 50) / 4.3);
  x.data.fill(0, k, 2 * k);
  if (rows > 2) x.data[2 * k + 5] = Number.NaN;
  return x;
};

/**
 * (rows, K, N): a remainder row length, Wan2.2's text embedder (M 512 → 300 here), DeBERTa's
 * intermediate (K 4096), anima's FFN down projection (K 8192), Wan2.2's FFN down projection.
 */
export const LINEAR_PROLOGUE_SHAPES: readonly (readonly [number, number, number])[] = [
  [7, 1028, 68],
  [300, 3072, 16],
  [11, 4096, 24],
  [9, 8192, 16],
  [5, 14336, 8],
];

/** The ops the fusion pass folds into the linear's step ① (`LINEAR_PROLOGUE_OPS`). */
export const LINEAR_PROLOGUE_OPS = [
  "gelu_tanh",
  "gelu",
] as const satisfies readonly PrologueUnaryOp[];

/** Every op with both integer dot forms, in the order the cases run. */
export const LINEAR_PROLOGUE_FORMS: readonly (readonly [PrologueUnaryOp, I8a8Dot])[] =
  LINEAR_PROLOGUE_OPS.flatMap((op) =>
    (["dp4a", "emu"] satisfies I8a8Dot[]).map((dot) => [op, dot] as const),
  );

/** The case label of a shape, op and dot form. */
export const linearPrologueLabel = (
  op: PrologueUnaryOp,
  [rows, k, n]: readonly [number, number, number],
  dot: I8a8Dot,
): string => `${op} rows=${rows} K=${k} N=${n} ${dot}`;

/** What the device lacks for the part (the dp4a cases force `dot4I8Packed`), or undefined. */
export const linearPrologueShortfall = (
  gpu: Pick<GpuContext, "wgslLanguageFeatures">,
): string | undefined =>
  dp4aAvailable(gpu.wgslLanguageFeatures)
    ? undefined
    : `the dp4a cases need the WGSL language feature ${DP4A_WGSL_FEATURE}, which this device lacks`;

const PART = "fusion-linear-prologue";

/** The cases of {@link checkLinearPrologueTwins}: 5 shapes × 2 ops × 2 dot forms. */
export const LINEAR_PROLOGUE_TWIN_CASES = { [PART]: 20 } as const satisfies ExpectedCases;

/**
 * Every shape, op and dot form on the a8 seat: the fused Session against the interposed one
 * (checkFusionTwin, NaN by class), linearPrologue at 1 and 0, the prologue quantize + the GEMM
 * against the op + quantize_rows + the GEMM. Throws on a device without dp4a.
 */
export const checkLinearPrologueTwins = async (
  gpu: GpuContext,
  options: CheckOptions = {},
): Promise<CheckPart> => {
  const shortfall = linearPrologueShortfall(gpu);
  if (shortfall !== undefined) throw new Error(`${PART}: ${shortfall}`);
  const sink = twinCases(PART, LINEAR_PROLOGUE_TWIN_CASES[PART], options);
  for (const shape of LINEAR_PROLOGUE_SHAPES) {
    const [rows, k, n] = shape;
    const tensors = linearPrologueWeights(k, n);
    const x = linearPrologueActivations(rows, k);
    for (const [op, dot] of LINEAR_PROLOGUE_FORMS) {
      sink.next();
      const label = linearPrologueLabel(op, shape, dot);
      const sessionOptions = { linearCompute: "a8", [I8A8_DOT]: dot } as const;
      await withTwinSessions(
        gpu,
        label,
        {
          model: await openModelBytes(linearPrologueGraph(op, rows, k, n, false), tensors),
          options: sessionOptions,
        },
        {
          model: await openModelBytes(linearPrologueGraph(op, rows, k, n, true), tensors),
          options: sessionOptions,
        },
        async ({ fused, plain }) => {
          sink.add(
            await checkFusionTwin(gpu, {
              label: `${label}: fused vs pair`,
              fused,
              plain,
              feeds: { x },
              counter: "linearPrologue",
              hits: 1,
              nan: "class",
              // The prologue quantize + the GEMM against the op + quantize_rows + the GEMM.
              dispatches: { fused: 2, plain: 3 },
            }),
          );
        },
      );
    }
  }
  return {
    id: PART,
    title: "linearPrologue: gelu folded into the a8 linear vs the pair (dp4a and emulated dot)",
    cases: sink.cases,
    requirements: [],
  };
};
