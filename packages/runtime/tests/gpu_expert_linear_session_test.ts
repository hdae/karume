// expert_linear through a Session on a real GPU (ADR 0145 decisions 2 to 4): one routed expert FFN
// built from existing ops around two expert_linear nodes, with the weights written into a container
// as f32, int4-sym-g (f32 steps) and int4-sym-g-f16 (f16 steps):
//
//   scores = linear(x, router)  →  topk(scores, k) = (weights, index)
//   h = expert_linear(x, index, gate_up)            (x shared by the k picks)
//   m = gelu_tanh(h[..., :F]) · h[..., F:]          (slice / gelu_tanh / mul)
//   y = expert_linear(m, index, down)               (one row per pick)
//   out = sum(y · weights[..., None], dim 1)        (reshape / mul / sum, in pick order)
//
// The CPU reference walks the same ops with the weights decoded from the stored bytes by the
// production decoders (the f16 steps widened exactly — ADR 0069 decision 3). The routing ids MUST
// match exactly; the output is compared within the GEMM band the GEMV tests use. The token axis is
// symbolic and the Session runs two lengths. The compressed weights MUST stay resident (no host
// expansion — ADR 0145 decision 2). Its weight rows are short, so every run takes the one-row kernel.
//
// A second graph has long weight rows (64 words) and nothing but the routing and one expert_linear:
//
//   topk(scores, k) = (picked, index)  →  h = expert_linear(x, index, weight)
//
// A run of five tokens takes the grouped kernel (rows that picked one expert share the walk of its
// weight row), a run of one token the one-row kernel, and every word of a token's rows MUST be the
// same in both: the two kernels are one numeric contract.

import { assert, assertEquals } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import { expertLinearKey } from "../src/kernels/expert-linear.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import {
  applyReferenceOp,
  applyReferenceOpOutputs,
  decodeExpertLinearWeight,
  type ExpertLinearStoredWeight,
  type RefTensor,
  refTensor,
} from "../src/reference/ops.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import type { EncodingInput } from "./helpers/container-write.ts";
import { f16BytesFromBits, f32ToF16Bits } from "./helpers/f16.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { quantizeI4 } from "./helpers/i4.ts";
import {
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";
import { GEMM_TOLERANCE } from "../src/reference/op-tolerance.ts";

const E = 6;
const K = 2;
const IN = 64;
const F = 32;
const H = 64;
const GROUP = 32;

/** A deterministic value in [-scale, scale) (no random numbers — a failure must reproduce). */
const wave = (i: number, seed: number, scale: number): number =>
  (((Math.imul(i + seed, 0x9e3779b9) >>> 8) / 2 ** 24) * 2 - 1) * scale;

const values = (count: number, seed: number, scale: number): Float32Array<ArrayBuffer> =>
  Float32Array.from({ length: count }, (_, i) => wave(i, seed, scale));

const GATE_UP_SHAPE = [E, 2 * F, IN];
const DOWN_SHAPE = [E, H, F];
const ROUTER = { weight: values(E * IN, 3, 0.3), bias: values(E, 5, 0.2) };
const GATE_UP = values(E * 2 * F * IN, 7, 0.15);
const DOWN = values(E * H * F, 11, 0.15);

type Storage = ExpertLinearStoredWeight["storage"];

/** The stored form of one expert weight and its container encoding. */
const stored = (
  storage: Storage,
  data: Float32Array<ArrayBuffer>,
  shape: readonly number[],
): {
  readonly weight: ExpertLinearStoredWeight;
  readonly bytes: Uint8Array<ArrayBuffer>;
  readonly encoding: EncodingInput;
} => {
  if (storage === "f32") {
    return { weight: { storage, values: data }, bytes: f32Bytes(data), encoding: { codec: "f32" } };
  }
  const quantized = quantizeI4(data, shape, GROUP);
  if (storage === "i4") {
    return {
      weight: { storage, bytes: quantized.bytes, steps: quantized.scale, groupSize: GROUP },
      bytes: quantized.bytes,
      encoding: {
        codec: "int4-sym-g",
        groupSize: GROUP,
        scale: { bytes: f32Bytes(quantized.scale), dtype: "f32" },
      },
    };
  }
  // The f16 step table: each step rounded once, two per u32 word (step 2i in the low half).
  const bits = Array.from(quantized.scale, (step) => f32ToF16Bits(step));
  const steps = f16BytesFromBits(bits.length % 2 === 0 ? bits : [...bits, 0]);
  return {
    weight: { storage, bytes: quantized.bytes, steps, groupSize: GROUP },
    bytes: quantized.bytes,
    encoding: { codec: "int4-sym-g-f16", groupSize: GROUP, scale: { bytes: steps, dtype: "f16" } },
  };
};

const DECLARATION: DeclarationJson = {
  format: "karume-ir",
  version: 2,
  requires: {
    ops: ["expert_linear", "gelu_tanh", "linear", "mul", "reshape", "slice", "sum", "topk"],
  },
  symbols: ["T"],
  inputs: [{ name: "x", dtype: "f32", shape: ["T", IN] }],
  outputs: ["out", "index"],
  initializers: { router_w: {}, router_b: {}, gate_up: {}, down: {} },
  values: {
    router_w: { dtype: "f32", shape: [E, IN] },
    router_b: { dtype: "f32", shape: [E] },
    gate_up: { dtype: "f32", shape: GATE_UP_SHAPE },
    down: { dtype: "f32", shape: DOWN_SHAPE },
    scores: { dtype: "f32", shape: ["T", E] },
    picked: { dtype: "f32", shape: ["T", K] },
    index: { dtype: "i32", shape: ["T", K] },
    h: { dtype: "f32", shape: ["T", K, 2 * F] },
    gate: { dtype: "f32", shape: ["T", K, F] },
    up: { dtype: "f32", shape: ["T", K, F] },
    act: { dtype: "f32", shape: ["T", K, F] },
    m: { dtype: "f32", shape: ["T", K, F] },
    y: { dtype: "f32", shape: ["T", K, H] },
    w3: { dtype: "f32", shape: ["T", K, 1] },
    yw: { dtype: "f32", shape: ["T", K, H] },
    out: { dtype: "f32", shape: ["T", H] },
  },
  nodes: [
    { op: "linear", ins: ["x", "router_w", "router_b"], outs: ["scores"], attrs: {} },
    { op: "topk", ins: ["scores"], outs: ["picked", "index"], attrs: { k: K } },
    { op: "expert_linear", ins: ["x", "index", "gate_up"], outs: ["h"], attrs: {} },
    { op: "slice", ins: ["h"], outs: ["gate"], attrs: { dim: 2, start: 0, end: F } },
    { op: "slice", ins: ["h"], outs: ["up"], attrs: { dim: 2, start: F, end: 2 * F } },
    { op: "gelu_tanh", ins: ["gate"], outs: ["act"], attrs: {} },
    { op: "mul", ins: ["act", "up"], outs: ["m"], attrs: {} },
    { op: "expert_linear", ins: ["m", "index", "down"], outs: ["y"], attrs: {} },
    { op: "reshape", ins: ["picked"], outs: ["w3"], attrs: {} },
    { op: "mul", ins: ["y", "w3"], outs: ["yw"], attrs: {} },
    { op: "sum", ins: ["yw"], outs: ["out"], attrs: { dim: 1 } },
  ],
};

/** The same ops on the CPU reference (the expert weights decoded from their stored bytes). */
const reference = (
  x: RefTensor,
  gateUp: RefTensor,
  down: RefTensor,
): { readonly out: RefTensor; readonly index: RefTensor; readonly scores: RefTensor } => {
  const tokens = x.shape[0];
  const scores = applyReferenceOp("linear", [
    x,
    refTensor([E, IN], ROUTER.weight),
    refTensor([E], ROUTER.bias),
  ]);
  const [picked, index] = applyReferenceOpOutputs("topk", [scores], { k: K });
  const h = applyReferenceOp("expert_linear", [x, index, gateUp]);
  const gate = applyReferenceOp("slice", [h], { dim: 2, start: 0, end: F });
  const up = applyReferenceOp("slice", [h], { dim: 2, start: F, end: 2 * F });
  const m = applyReferenceOp("mul", [applyReferenceOp("gelu_tanh", [gate]), up]);
  const y = applyReferenceOp("expert_linear", [m, index, down]);
  const w3 = applyReferenceOp("reshape", [picked], {}, [tokens, K, 1]);
  const out = applyReferenceOp("sum", [applyReferenceOp("mul", [y, w3])], { dim: 1 });
  return { out, index, scores };
};

/**
 * MUST (precondition): the reference's scores leave a margin around the k picks of every token,
 * so the GPU's router linear (another summation order) picks the same experts in the same order.
 */
const assertClearRouting = (scores: RefTensor): void => {
  for (let t = 0; t < scores.shape[0]; t += 1) {
    const row = [...scores.data.slice(t * E, (t + 1) * E)].sort((a, b) => b - a);
    for (let rank = 0; rank < K; rank += 1) {
      assert(row[rank] - row[rank + 1] > 1e-3, `token ${t}: scores ${row.join(", ")} too close`);
    }
  }
};

/** Compressed bytes the two expert weights hold on the GPU (payload + steps), by hand. */
const RESIDENT_BYTES: Readonly<Record<Storage, number>> = {
  f32: 0,
  // gate_up 24,576 codes = 12,288 B + 768 steps; down 12,288 codes = 6,144 B + 384 steps.
  i4: 12288 + 768 * 4 + 6144 + 384 * 4,
  i4f16: 12288 + 768 * 2 + 6144 + 384 * 2,
};

/** The expert_linear rows of a run's census, as [key, dispatches]. */
const expertKernels = (
  rows: readonly { readonly key: string; readonly dispatchCount: number }[] | undefined,
): [string, number][] =>
  (rows ?? [])
    .filter((row) => row.key.startsWith("expert_linear:"))
    .map((row) => [row.key, row.dispatchCount]);

const LONG_PICKS = 3;
const LONG_OUT = 40;
/**
 * The picks of five tokens in pick order: expert 4 by every token (five rows: a walk of four and a
 * walk of one), expert 1 by four, experts 2 and 5 by two, experts 0 and 3 by one.
 */
const LONG_ROUTING = [
  [4, 1, 5],
  [1, 4, 0],
  [4, 2, 1],
  [4, 1, 3],
  [2, 4, 5],
];

/** Scores whose top three are `picks` in order (every score distinct, so there is no tie). */
const scoresOf = (picks: readonly number[]): number[] =>
  Array.from({ length: E }, (_, expert) => {
    const rank = picks.indexOf(expert);
    return rank < 0 ? -1 - expert : LONG_PICKS - rank;
  });

/** The routing and one expert_linear over weight rows of `width` elements. */
const longRowDeclaration = (width: number): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["expert_linear", "topk"] },
  symbols: ["T"],
  inputs: [
    { name: "x", dtype: "f32", shape: ["T", width] },
    { name: "scores", dtype: "f32", shape: ["T", E] },
  ],
  outputs: ["h", "index"],
  initializers: { weight: {} },
  values: {
    weight: { dtype: "f32", shape: [E, LONG_OUT, width] },
    picked: { dtype: "f32", shape: ["T", LONG_PICKS] },
    index: { dtype: "i32", shape: ["T", LONG_PICKS] },
    h: { dtype: "f32", shape: ["T", LONG_PICKS, LONG_OUT] },
  },
  nodes: [
    { op: "topk", ins: ["scores"], outs: ["picked", "index"], attrs: { k: LONG_PICKS } },
    { op: "expert_linear", ins: ["x", "index", "weight"], outs: ["h"], attrs: {} },
  ],
});

describe.skipIf(!GPU_AVAILABLE)(
  "expert_linear over long weight rows through a Session (real GPU)",
  () => {
    for (const storage of ["f32", "i4", "i4f16"] as const) {
      it(`gives each token of a five-token run the words of its own one-token run, with ${storage} weights: the grouped kernel against the one-row kernel`, async () => {
        // 64 words of a weight row: 4 f32 elements or 32 i4 elements to a word.
        const width = storage === "f32" ? 256 : 2048;
        const shape = [E, LONG_OUT, width];
        const weight = stored(storage, values(E * LONG_OUT * width, 19, 0.15), shape);
        const opened = await openModelBytes(longRowDeclaration(width), [
          { graph: GRAPH_NAME, initializer: "weight", ...weight },
        ]);
        const group = storage === "f32" ? undefined : GROUP;
        const tokens = LONG_ROUTING.length;
        const x = values(tokens * width, 23, 1.5);
        const scores = Float32Array.from(LONG_ROUTING.flatMap((picks) => scoresOf(picks)));
        const gpu = await acquireGpu();
        try {
          const session = await createSessionFromContainer(gpu, opened, GRAPH_NAME);
          try {
            const all = await session.run({
              x: refTensor([tokens, width], x),
              scores: refTensor([tokens, E], scores),
            });
            assertEquals(
              [...all["index"].data],
              LONG_ROUTING.flat(),
              "the routing of the five tokens",
            );
            assertEquals(
              expertKernels(session.diagnostics().lastRunPipelines),
              [[expertLinearKey(storage, group, false, true), 1]],
              "five tokens: the grouped kernel",
            );
            const span = LONG_PICKS * LONG_OUT;
            const words = new Uint32Array(
              all["h"].data.buffer,
              all["h"].data.byteOffset,
              tokens * span,
            );
            for (let token = 0; token < tokens; token += 1) {
              const one = await session.run({
                x: refTensor([1, width], x.slice(token * width, (token + 1) * width)),
                scores: refTensor([1, E], scores.slice(token * E, (token + 1) * E)),
              });
              assertEquals(
                expertKernels(session.diagnostics().lastRunPipelines),
                [[expertLinearKey(storage, group), 1]],
                `token ${token} alone: the one-row kernel`,
              );
              assertEquals(
                [...words.subarray(token * span, (token + 1) * span)],
                [...new Uint32Array(one["h"].data.buffer, one["h"].data.byteOffset, span)],
                `token ${token}: the words of its rows in the five-token run and alone`,
              );
            }
            // MUST (precondition): the rows hold sums, not zeros — an unbound weight would compare equal.
            assert(
              words.filter((word) => word !== 0).length > words.length / 2,
              "h is mostly zero",
            );
          } finally {
            await session.dispose();
          }
        } finally {
          gpu.destroy();
        }
      });
    }
  },
);

describe.skipIf(!GPU_AVAILABLE)("expert_linear through a Session (real GPU)", () => {
  for (const storage of ["f32", "i4", "i4f16"] as const) {
    it(`runs a routed expert FFN with ${storage} expert weights like the CPU reference, at two token counts`, async () => {
      const gateUp = stored(storage, GATE_UP, GATE_UP_SHAPE);
      const down = stored(storage, DOWN, DOWN_SHAPE);
      const opened = await openModelBytes(DECLARATION, [
        {
          graph: GRAPH_NAME,
          initializer: "router_w",
          bytes: f32Bytes(ROUTER.weight),
          encoding: { codec: "f32" },
        },
        {
          graph: GRAPH_NAME,
          initializer: "router_b",
          bytes: f32Bytes(ROUTER.bias),
          encoding: { codec: "f32" },
        },
        { graph: GRAPH_NAME, initializer: "gate_up", ...gateUp },
        { graph: GRAPH_NAME, initializer: "down", ...down },
      ]);
      const gateUpValues = decodeExpertLinearWeight(GATE_UP_SHAPE, gateUp.weight);
      const downValues = decodeExpertLinearWeight(DOWN_SHAPE, down.weight);
      const gpu = await acquireGpu();
      try {
        const session = await createSessionFromContainer(gpu, opened, GRAPH_NAME);
        try {
          const storageDiagnostics = session.diagnostics().storage;
          assertEquals(storageDiagnostics.residentCompressedBytes, RESIDENT_BYTES[storage]);
          assertEquals(storageDiagnostics.hostExpandedBytes, 0, "an expert weight was expanded");
          for (const [tokens, seed] of [
            [5, 13],
            [1, 17],
          ] as const) {
            const x = refTensor([tokens, IN], values(tokens * IN, seed, 1.5));
            const expected = reference(x, gateUpValues, downValues);
            assertClearRouting(expected.scores);
            // MUST (precondition): the routing reaches several experts, not one pair for every token.
            const hit = new Set(expected.index.data).size;
            assert(tokens === 1 || hit >= 4, `${storage} T=${tokens}: only ${hit} experts hit`);
            const outputs = await session.run({ x });
            const where = `${storage} T=${tokens}`;
            assertEquals(outputs["index"].shape, [tokens, K], where);
            assertEquals([...outputs["index"].data], [...expected.index.data], `${where}: routing`);
            const report = compareTensors(outputs["out"], expected.out, GEMM_TOLERANCE);
            assertEquals(report.pass, true, `${where}: ${formatAllclose(report)}`);
            // Short weight rows: both expert_linear nodes on the one-row kernel at every length.
            assertEquals(
              expertKernels(session.diagnostics().lastRunPipelines),
              [[expertLinearKey(storage, storage === "f32" ? undefined : GROUP), 2]],
              `${where}: the expert_linear kernels of the run`,
            );
          }
        } finally {
          await session.dispose();
        }
      } finally {
        gpu.destroy();
      }
    });
  }
});
