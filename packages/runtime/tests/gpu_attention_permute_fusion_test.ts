// A fused attention that absorbs the [0, 2, 1, 3] permutes around it (perf-ledger F-3 A4 / F-8 —
// `attentionPermutesAt` of src/runtime/fusion.ts, honoured by `buildAttention` on every route): bit
// parity with the same graph where the permutes stay nodes, the counter, and the dispatches each
// route saves, on the online route, the three-dispatch a8 route with S in f16 (s16), the dense f32
// route and the mixed a8 route whose ①QK falls back to f32 (D % 4 != 0).
//
// The reference is the same graph with a zero-dispatch reshape between each permute and the
// attention and between the attention and the output permute: the values and the physical kernels
// stay the graph's, only the exact match stops matching.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { stridedKey } from "../src/codegen/strided.ts";
import { acquireGpu, type GpuContext } from "../src/gpu/device.ts";
import { QUANTIZE_ROWS_KEY } from "../src/kernels/quantize-rows.ts";
import { TRANSPOSE_TILED_KEY } from "../src/kernels/transpose-tiled.ts";
import { attentionDeviceFacts, attentionRoute } from "../src/runtime/attention-route.ts";
import {
  createSessionFromContainer,
  type SessionOptions,
  type Tensor,
} from "../src/runtime/executor.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openGraphModel } from "./helpers/model-fixture.ts";
import { type CensusRow, countDispatches, requireCensus } from "./helpers/pipeline-census.ts";

type Shape = {
  readonly b: number;
  readonly h: number;
  readonly s: number;
  readonly n: number;
  readonly d: number;
};

/**
 * Token-major q / k / v → permute to head-major → attention → permute O back (the Wan DiT form).
 * `interpose` puts a zero-dispatch reshape after each input permute and after the attention.
 */
const permutedAttentionGraph = (shape: Shape, interpose: boolean): DeclarationJson => {
  const { b, h, s, n, d } = shape;
  const dtype = "f32";
  const token = { q: [b, s, h, d], k: [b, n, h, d], v: [b, n, h, d] } as const;
  const head = { q: [b, h, s, d], k: [b, h, n, d], v: [b, h, n, d] } as const;
  const values: DeclarationJson["values"] = {
    o: { dtype, shape: [b, h, s, d] },
    ot: { dtype, shape: [b, s, h, d] },
  };
  const nodes: DeclarationJson["nodes"] = [];
  for (const operand of ["q", "k", "v"] as const) {
    values[operand] = { dtype, shape: [...head[operand]] };
    nodes.push({
      op: "permute",
      ins: [`${operand}t`],
      outs: [operand],
      attrs: { dims: [0, 2, 1, 3] },
    });
  }
  const read = (name: string, shape: readonly number[]): string => {
    if (!interpose) return name;
    values[`${name}_r`] = { dtype, shape: [...shape] };
    nodes.push({ op: "reshape", ins: [name], outs: [`${name}_r`], attrs: {} });
    return `${name}_r`;
  };
  const ins = (["q", "k", "v"] as const).map((operand) => read(operand, head[operand]));
  const scale = Math.fround(1 / Math.sqrt(d));
  nodes.push({ op: "attention", ins, outs: ["o"], attrs: { scale } });
  const o = read("o", [b, h, s, d]);
  nodes.push({ op: "permute", ins: [o], outs: ["ot"], attrs: { dims: [0, 2, 1, 3] } });
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: [...new Set(nodes.map((node) => node.op))].sort() },
    symbols: [],
    inputs: (["q", "k", "v"] as const).map((operand) => ({
      name: `${operand}t`,
      dtype,
      shape: [...token[operand]],
    })),
    outputs: ["ot"],
    initializers: {},
    values,
    nodes,
  };
};

const words = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

type Observed = {
  readonly output: Uint32Array;
  readonly census: readonly CensusRow[];
  readonly absorbed: number | undefined;
};

const runGraph = async (
  gpu: GpuContext,
  shape: Shape,
  interpose: boolean,
  options: SessionOptions,
): Promise<Observed> => {
  const { b, h, s, n, d } = shape;
  const session = await createSessionFromContainer(
    gpu,
    await openGraphModel(permutedAttentionGraph(shape, interpose)),
    GRAPH_NAME,
    options,
  );
  try {
    const outputs = await session.run({
      qt: fill([b, s, h, d], (i) => Math.sin(i * 0.37) * 1.7 + ((i % 13) - 6) / 9),
      kt: fill([b, n, h, d], (i) => Math.cos(i * 0.23) * 1.3 + ((i % 7) - 3) / 7),
      vt: fill([b, n, h, d], (i) => Math.sin(i * 0.11 + 1) * 2.1 + ((i % 5) - 2) / 3),
    });
    const ot = outputs["ot"];
    assert(ot !== undefined, "the attention output is missing");
    const diagnostics = session.diagnostics();
    return {
      output: words(ot),
      census: requireCensus(diagnostics.lastRunPipelines, interpose ? "reference" : "absorbed"),
      absorbed: diagnostics.lastRunFusions?.attentionPermute,
    };
  } finally {
    await session.dispose();
  }
};

/** The dispatches of the kernels the absorption moves, and the total. */
type Dispatches = {
  readonly strided: number;
  readonly transpose: number;
  readonly quantizeBshd: number;
  readonly quantizeRows: number;
  readonly total: number;
};

const dispatchesOf = (census: readonly CensusRow[]): Dispatches => ({
  strided: countDispatches(census, stridedKey({ dtype: "f32" })),
  transpose: countDispatches(census, TRANSPOSE_TILED_KEY),
  quantizeBshd: countDispatches(
    census,
    (key) => key.startsWith(QUANTIZE_ROWS_KEY) && key.endsWith(":bshd"),
  ),
  quantizeRows: countDispatches(
    census,
    (key) => key.startsWith(QUANTIZE_ROWS_KEY) && !key.endsWith(":bshd"),
  ),
  total: countDispatches(census, () => true),
});

const assertSameWords = (actual: Uint32Array, expected: Uint32Array, where: string): void => {
  assertEquals(actual.length, expected.length, `${where}: length`);
  let differing = 0;
  for (let i = 0; i < actual.length; i += 1) if (actual[i] !== expected[i]) differing += 1;
  assertEquals(differing, 0, `${where}: differing words`);
};

type Route = {
  readonly name: string;
  readonly options: SessionOptions;
  readonly shape: Shape;
  /** The census of the absorbed graph. */
  readonly absorbed: Dispatches;
  /** The census of the reference (the permutes as nodes). */
  readonly reference: Dispatches;
};

const WAN_LIKE: Shape = { b: 2, h: 3, s: 150, n: 200, d: 128 };

const ROUTES: readonly Route[] = [
  {
    // a8 + "none": q / k read token-major by the quantisation, Vᵀ by the tiled transpose, O
    // stored token-major — no strided copy left. The reference: the three input permutes, the
    // strided Vᵀ, the output permute.
    name: "online",
    options: { attentionCompute: "a8", attentionScoreStorage: "none" },
    shape: WAN_LIKE,
    absorbed: { strided: 0, transpose: 1, quantizeBshd: 2, quantizeRows: 1, total: 5 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 3, total: 9 },
  },
  {
    // a8 + S in f16 (the three dispatches): the same quantisation; ③PV writes a head-major O
    // temporary that the output permute's copy moves into the output.
    name: "three-dispatch s16",
    options: { attentionCompute: "a8", attentionScoreStorage: "f16" },
    shape: WAN_LIKE,
    absorbed: { strided: 1, transpose: 1, quantizeBshd: 2, quantizeRows: 1, total: 8 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 3, total: 11 },
  },
  {
    // The f32 reference preset: every absorbed permute is made by the attention itself — the
    // same four copies, the same dispatch count.
    name: "dense f32",
    options: {},
    shape: WAN_LIKE,
    absorbed: { strided: 4, transpose: 0, quantizeBshd: 0, quantizeRows: 0, total: 7 },
    reference: { strided: 4, transpose: 0, quantizeBshd: 0, quantizeRows: 0, total: 7 },
  },
  {
    // a8 with D = 14 (D % 4 != 0): ①QK falls back to f32 and reads head-major copies of q / k,
    // ③PV stays i8a8 and transposes the token-major V; O goes through the temporary.
    name: "a8 with ①QK on f32 (D = 14)",
    options: { attentionCompute: "a8" },
    shape: { b: 2, h: 3, s: 37, n: 44, d: 14 },
    absorbed: { strided: 3, transpose: 1, quantizeBshd: 0, quantizeRows: 1, total: 8 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 1, total: 9 },
  },
];

describe.skipIf(!GPU_AVAILABLE)(
  "a fused attention that absorbs its permutes (perf-ledger F-3 / F-8)",
  () => {
    for (const route of ROUTES) {
      it(`${route.name}: bit-identical to the permutes as nodes, attentionPermute 4, and the route's own dispatches`, async (context) => {
        const gpu = await acquireGpu({ subgroups: "available" });
        try {
          const { d, n } = route.shape;
          if (
            route.options.attentionScoreStorage === "none" &&
            attentionRoute(
              { attentionCompute: "a8", attentionScoreStorage: "none" },
              d,
              n,
              attentionDeviceFacts(gpu),
            ).online === undefined
          ) {
            // The device holds neither online form: the route falls back (the s16 case covers it).
            context.skip();
            return;
          }
          const absorbed = await runGraph(gpu, route.shape, false, route.options);
          const reference = await runGraph(gpu, route.shape, true, route.options);
          assertEquals(absorbed.absorbed, 4, `${route.name}: absorbed counter`);
          assertEquals(reference.absorbed, 0, `${route.name}: reference counter`);
          assert(new Set(reference.output).size > 1, `${route.name}: the output is constant`);
          assertSameWords(
            absorbed.output,
            reference.output,
            `${route.name}: absorbed vs reference`,
          );
          const keys = (census: readonly CensusRow[]): string =>
            census.map((row) => `${row.key}×${row.dispatchCount}`).join(" / ");
          assertEquals(
            dispatchesOf(absorbed.census),
            route.absorbed,
            `${route.name}: absorbed census — ${keys(absorbed.census)}`,
          );
          assertEquals(
            dispatchesOf(reference.census),
            route.reference,
            `${route.name}: reference census — ${keys(reference.census)}`,
          );
          if (route.options.attentionScoreStorage === "none") {
            assert(
              absorbed.census.some(
                (row) => row.key.startsWith("attention_online:") && row.key.endsWith(":obshd"),
              ),
              `${route.name}: the online kernel stores O token-major — ${keys(absorbed.census)}`,
            );
          }
          console.log(
            `[attention permutes] ${route.name}: ${route.reference.total} → ${route.absorbed.total} dispatches, ${absorbed.output.length} words identical`,
          );
        } finally {
          gpu.destroy();
        }
      });
    }
  },
);
