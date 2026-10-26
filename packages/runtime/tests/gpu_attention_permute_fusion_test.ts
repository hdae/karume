// A fused attention on an a8 seat that absorbs the [0, 2, 1, 3] permutes of its q / k / v
// (perf-ledger F-3 A4 / F-8 — `attentionPermutesAt` of src/runtime/fusion.ts, honoured by
// `buildAttention` on every route an a8 seat can take) and, on the online route only, the permute of
// O after it (the kernel stores O token-major): bit parity with the same graph where the permutes
// stay nodes, the counter, and the dispatches each route saves, on the online route (sg32 at D 128
// and 64, wg at D 128), the three-dispatch a8 route with S in f16 (s16), an online seat whose shape
// (D 96) falls back to the three dispatches with S in f32, and the two a8 shapes whose stage falls
// back to f32 (①QK at D % 4 != 0, ③PV at N % 4 != 0 — the head-major copy path). The f32 seat
// absorbs nothing. Off the online route the output permute stays a node.
//
// The reference is the same graph with a zero-dispatch reshape between each input permute and the
// attention, and between the attention and the output permute: the values and the physical kernels
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
  const attended = interpose ? "o_a" : "o";
  if (interpose) values[attended] = { dtype, shape: [b, h, s, d] };
  nodes.push({ op: "attention", ins, outs: [attended], attrs: { scale } });
  if (interpose) nodes.push({ op: "reshape", ins: [attended], outs: ["o"], attrs: {} });
  nodes.push({ op: "permute", ins: ["o"], outs: ["ot"], attrs: { dims: [0, 2, 1, 3] } });
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
  /**
   * The subgroup request of the device (default "available"); `false` leaves the online kernel only
   * its wg form.
   */
  readonly subgroups?: false;
  /** The online kernel's reduction form the route must take (online routes only). */
  readonly reduce?: "sg32" | "wg";
  /**
   * An online seat whose shape the online kernel does not admit: the route must fall back to the
   * three a8 dispatches with S in f32, and the run must be the a8 × "f32" Session's.
   */
  readonly fallsBack?: true;
  /** The attentionPermute count of the graph without the reshapes. */
  readonly absorbedPermutes: number;
  /** The census of the absorbed graph. */
  readonly absorbed: Dispatches;
  /** The census of the reference (the permutes as nodes). */
  readonly reference: Dispatches;
};

const WAN_LIKE: Shape = { b: 2, h: 3, s: 150, n: 200, d: 128 };

const ROUTES: readonly Route[] = [
  {
    // a8 + "none": q / k read token-major by the quantisation, Vᵀ by the tiled transpose, O stored
    // token-major by the online kernel; no strided copy is left. The reference: the three input
    // permutes, the strided Vᵀ, the output permute.
    name: "online (sg32, D 128)",
    options: { attentionCompute: "a8", attentionScoreStorage: "none" },
    shape: WAN_LIKE,
    reduce: "sg32",
    absorbedPermutes: 4,
    absorbed: { strided: 0, transpose: 1, quantizeBshd: 2, quantizeRows: 1, total: 5 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 3, total: 9 },
  },
  {
    // The same at D 64.
    name: "online (sg32, D 64)",
    options: { attentionCompute: "a8", attentionScoreStorage: "none" },
    shape: { b: 2, h: 3, s: 150, n: 200, d: 64 },
    reduce: "sg32",
    absorbedPermutes: 4,
    absorbed: { strided: 0, transpose: 1, quantizeBshd: 2, quantizeRows: 1, total: 5 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 3, total: 9 },
  },
  {
    // A device without subgroups: the online kernel's wg form stores O token-major the same way.
    name: "online (wg, D 128)",
    options: { attentionCompute: "a8", attentionScoreStorage: "none" },
    shape: WAN_LIKE,
    subgroups: false,
    reduce: "wg",
    absorbedPermutes: 4,
    absorbed: { strided: 0, transpose: 1, quantizeBshd: 2, quantizeRows: 1, total: 5 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 3, total: 9 },
  },
  {
    // a8 + S in f16 (the three dispatches): the same quantisation, ③PV writes O head-major for the
    // output permute node.
    name: "three-dispatch s16",
    options: { attentionCompute: "a8", attentionScoreStorage: "f16" },
    shape: WAN_LIKE,
    absorbedPermutes: 3,
    absorbed: { strided: 1, transpose: 1, quantizeBshd: 2, quantizeRows: 1, total: 8 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 3, total: 11 },
  },
  {
    // a8 + "none" at D = 96 (not 64 / 128): the online kernel does not admit the shape, so the route
    // is the three a8 dispatches with S in f32 — the same dispatches as s16, O stored head-major and
    // its permute a node.
    name: "online seat falling back (D 96)",
    options: { attentionCompute: "a8", attentionScoreStorage: "none" },
    shape: { b: 2, h: 3, s: 150, n: 200, d: 96 },
    fallsBack: true,
    absorbedPermutes: 3,
    absorbed: { strided: 1, transpose: 1, quantizeBshd: 2, quantizeRows: 1, total: 8 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 3, total: 11 },
  },
  {
    // a8 with D = 14 (D % 4 != 0): ①QK falls back to f32 and reads head-major copies of q / k,
    // ③PV stays i8a8 and transposes the token-major V.
    name: "a8 with ①QK on f32 (D = 14)",
    options: { attentionCompute: "a8" },
    shape: { b: 2, h: 3, s: 37, n: 44, d: 14 },
    absorbedPermutes: 3,
    absorbed: { strided: 3, transpose: 1, quantizeBshd: 0, quantizeRows: 1, total: 8 },
    reference: { strided: 5, transpose: 0, quantizeBshd: 0, quantizeRows: 1, total: 9 },
  },
  {
    // a8 with N = 43 (N % 4 != 0): ③PV falls back to f32 and reads a head-major copy of V, ①QK
    // stays i8a8 and quantises the token-major q / k.
    name: "a8 with ③PV on f32 (N = 43)",
    options: { attentionCompute: "a8" },
    shape: { b: 2, h: 3, s: 37, n: 43, d: 16 },
    absorbedPermutes: 3,
    absorbed: { strided: 2, transpose: 0, quantizeBshd: 2, quantizeRows: 0, total: 7 },
    reference: { strided: 4, transpose: 0, quantizeBshd: 0, quantizeRows: 2, total: 9 },
  },
  {
    // The f32 seat absorbs nothing: both graphs run the four permutes as nodes.
    name: "f32 seat",
    options: {},
    shape: WAN_LIKE,
    absorbedPermutes: 0,
    absorbed: { strided: 4, transpose: 0, quantizeBshd: 0, quantizeRows: 0, total: 7 },
    reference: { strided: 4, transpose: 0, quantizeBshd: 0, quantizeRows: 0, total: 7 },
  },
];

describe.skipIf(!GPU_AVAILABLE)(
  "a fused attention that absorbs the permutes of its q / k / v (perf-ledger F-3 / F-8)",
  () => {
    for (const route of ROUTES) {
      it(`${route.name}: bit-identical to the permutes as nodes, attentionPermute ${route.absorbedPermutes}, and the route's own dispatches`, async (context) => {
        const gpu = await acquireGpu(route.subgroups === false ? {} : { subgroups: "available" });
        try {
          const { d, n } = route.shape;
          if (
            route.reduce !== undefined &&
            attentionRoute(
              { attentionCompute: "a8", attentionScoreStorage: "none" },
              d,
              n,
              attentionDeviceFacts(gpu),
            ).online !== route.reduce
          ) {
            // The device does not take this online form (the other online cases and s16 cover it).
            context.skip();
            return;
          }
          if (route.fallsBack) {
            // Precondition (asserted, not skipped — the shape alone rules the online kernel out).
            assertEquals(
              attentionRoute(
                { attentionCompute: "a8", attentionScoreStorage: "none" },
                d,
                n,
                attentionDeviceFacts(gpu),
              ),
              { qk: "i8a8", pv: "i8a8", score: "f32" },
              `${route.name}: route`,
            );
          }
          const absorbed = await runGraph(gpu, route.shape, false, route.options);
          const reference = await runGraph(gpu, route.shape, true, route.options);
          assertEquals(
            absorbed.absorbed,
            route.absorbedPermutes,
            `${route.name}: absorbed counter`,
          );
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
          if (route.fallsBack) {
            // The run is the three-dispatch route's: the same kernels and counts as a8 × "f32".
            const threeDispatch = await runGraph(gpu, route.shape, false, {
              attentionCompute: "a8",
              attentionScoreStorage: "f32",
            });
            assertEquals(
              keys(absorbed.census),
              keys(threeDispatch.census),
              `${route.name}: the a8 × "f32" census`,
            );
            assertSameWords(
              absorbed.output,
              threeDispatch.output,
              `${route.name}: absorbed vs a8 × "f32"`,
            );
          } else if (route.options.attentionScoreStorage === "none") {
            assert(
              absorbed.census.some((row) => row.key.startsWith("attention_online:")),
              `${route.name}: the online kernel runs — ${keys(absorbed.census)}`,
            );
            // The absorbed run stores O token-major; the reference stores it head-major.
            const tokenMajorStores = (census: readonly CensusRow[]): number =>
              countDispatches(
                census,
                (key) => key.startsWith("attention_online:") && key.endsWith(":obshd"),
              );
            assertEquals(tokenMajorStores(absorbed.census), 1, `${route.name}: absorbed O store`);
            assertEquals(tokenMajorStores(reference.census), 0, `${route.name}: reference O store`);
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
