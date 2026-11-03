/**
 * The integer GEMV (`SessionOptions.integerGemv`) on a real GPU:
 *
 * 1. The kernel against its CPU mirror: the shared check of `@karume/kernel-checks` (the same cases
 *    the playground's kernel-check page runs), every output word equal.
 * 2. Through a Session: `static_quantize → linear (i2) → static_quantize` at a measured decode shape
 *    (M = 1). With the option on, the integer key runs (with the SRQ epilogue when the linear→SRQ
 *    fusion takes the window, plain otherwise) and the output words equal `referenceIntegerGemv`;
 *    with it off, the parallel GEMV's key runs and no integer key does.
 */

import { assertEquals, assertRejects } from "@std/assert";
import { checkIntegerGemv, GEMV_INTEGER_CASES } from "@karume/kernel-checks";
import { describe, it } from "vite-plus/test";
import { acquireGpu } from "../src/gpu/device.ts";
import { integerGemvKey, type IntegerGemvVariant } from "../src/kernels/linear-gemv-integer.ts";
import {
  linearGemvParallelKey,
  linearGemvParallelPackedKey,
  linearGemvStaticQuantizeKey,
  linearGemvStaticQuantizePackedKey,
} from "../src/kernels/linear-gemv.ts";
import { STATIC_QUANTIZE_PACKED_KEY } from "../src/kernels/static-quantize.ts";
import { referenceIntegerGemv } from "../src/reference/linear-gemv-integer.ts";
import type { BoundContainer } from "../src/format/container/bind.ts";
import { createSessionFromContainer } from "../src/runtime/executor.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import type { SessionOptions } from "../src/runtime/session-types.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckPart } from "./helpers/kernel-check-assert.ts";
import {
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  memoryModel,
} from "./helpers/model-fixture.ts";
import { type CensusRow, countDispatches, requireCensus } from "./helpers/pipeline-census.ts";

describe.skipIf(!GPU_AVAILABLE)("integer GEMV against its CPU mirror (real GPU)", () => {
  it("writes every output word of every case of the shared check as the mirror does", async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkIntegerGemv(gpu);
      // i2 K = 16 / 80 / 1536 / 12288, i4 g16x1 / g16x3 / g32x2 and i8 K = 16 / 80 / 1536 × 15
      // variants, i4 g512x3 / g2048x3 / g4096x1 × 5 corners, and 2 saturated per storage's largest
      // K (i4: two), each at M = 1 / 8, with and without the SRQ epilogue.
      assertEquals(part.cases.length, GEMV_INTEGER_CASES["gemv-integer"]);
      assertEquals(part.cases.length, 692);
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  });
});

/** A power of two: an input `code × scale` quantizes back to exactly `code`. */
const X_SCALE = 2 ** -7;
const OUTPUT_SCALE = Math.fround(0.0013);

type Shape = {
  readonly name: string;
  readonly n: number;
  readonly k: number;
  readonly variant: IntegerGemvVariant;
  /** The parallel GEMV's key of the shape with the option off (fused with the output SRQ or not). */
  readonly parallelKey: (fused: boolean) => string;
};

/** The measured shapes (E2B QAT, the KV-shared layers' MLP). */
const SHAPES: readonly Shape[] = [
  {
    name: "down 1536×12288",
    n: 1536,
    k: 12288,
    variant: { weight: { storage: "i2" }, lanes: 16, rows: 4 },
    // The parallel GEMV's table packs this shape (ADR 0105 addendum 1).
    parallelKey: (fused) =>
      fused
        ? linearGemvStaticQuantizePackedKey("i2", undefined, 32)
        : linearGemvParallelPackedKey("i2", undefined, 32),
  },
  {
    name: "gate / up 12288×1536",
    n: 12288,
    k: 1536,
    variant: { weight: { storage: "i2" }, lanes: 8, rows: 4 },
    // ... and not this one: without the option its SRQ stays f32.
    parallelKey: (fused) =>
      fused
        ? linearGemvStaticQuantizeKey("i2", undefined, 2)
        : linearGemvParallelKey("i2", undefined, 2),
  },
];

/** `x → static_quantize → linear (i2) → static_quantize → z`, one decode row. */
const sessionGraph = ({ n, k }: Shape): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["linear", "static_quantize"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [1, k] }],
  outputs: ["z"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [n, k] },
    b: { dtype: "f32", shape: [n] },
    xq: { dtype: "f32", shape: [1, k] },
    y: { dtype: "f32", shape: [1, n] },
    z: { dtype: "f32", shape: [1, n] },
  },
  nodes: [
    { op: "static_quantize", ins: ["x"], outs: ["xq"], attrs: { scale: X_SCALE } },
    { op: "linear", ins: ["xq", "w", "b"], outs: ["y"], attrs: {} },
    { op: "static_quantize", ins: ["y"], outs: ["z"], attrs: { scale: OUTPUT_SCALE } },
  ],
});

type Fixture = {
  readonly model: BoundContainer;
  readonly codes: Int8Array<ArrayBuffer>;
  readonly weights: Uint32Array<ArrayBuffer>;
  readonly wscale: Float32Array<ArrayBuffer>;
  readonly bias: Float32Array<ArrayBuffer>;
};

/** The model with its weights, and the same weights as the mirror reads them. */
const fixtureOf = (shape: Shape): Fixture => {
  const { n, k } = shape;
  const bytes = Uint8Array.from(
    { length: (n * k) / 4 },
    (_, i) => Math.imul(i + 1, 0x9e3779b9) >>> 24,
  );
  const wscale = Float32Array.from({ length: n }, (_, i) => ((i % 17) + 1) * 0.00017);
  const bias = Float32Array.from({ length: n }, (_, i) => ((i % 7) - 3) * 0.11);
  const tensors: readonly TensorInput[] = [
    {
      graph: GRAPH_NAME,
      initializer: "w",
      bytes,
      // One scale per output channel (the i2 codec's grouping is the whole row).
      encoding: {
        codec: "int2-off",
        groupSize: k,
        scale: { bytes: f32Bytes(wscale), dtype: "f32" },
      },
    },
    { graph: GRAPH_NAME, initializer: "b", bytes: f32Bytes(bias), encoding: { codec: "f32" } },
  ];
  return {
    model: memoryModel(sessionGraph(shape), tensors),
    codes: Int8Array.from({ length: k }, (_, i) => ((i * 37 + 11) % 256) - 128),
    // The i2 payload is bound as is: element i at bits 2·(i mod 16) of word i / 16.
    weights: new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4),
    wscale,
    bias,
  };
};

const packCodes = (codes: Int8Array<ArrayBuffer>): Uint32Array<ArrayBuffer> => {
  const words = new Uint32Array(codes.length / 4);
  codes.forEach((code, index) => {
    words[Math.floor(index / 4)] += (code & 0xff) * 2 ** (8 * (index % 4));
  });
  return words;
};

const wordsOf = (data: ArrayBufferView<ArrayBuffer>): Uint32Array<ArrayBuffer> =>
  new Uint32Array(data.buffer, data.byteOffset, data.byteLength / 4);

const GEMV_INTEGER_FAMILY = "linear_gemv_integer";
const isInteger = (key: string): boolean => key.split(":", 1)[0] === GEMV_INTEGER_FAMILY;

/** Runs the fixture twice (the second run hits the prepared plan); returns the words and census. */
const runTwice = async (
  fixture: Fixture,
  options: SessionOptions,
  where: string,
): Promise<{ readonly words: Uint32Array<ArrayBuffer>; readonly census: readonly CensusRow[] }> => {
  const gpu = await acquireGpu();
  try {
    const session = await createSessionFromContainer(gpu, fixture.model, GRAPH_NAME, options);
    try {
      const input = {
        x: {
          dtype: "f32" as const,
          shape: [1, fixture.codes.length],
          data: Float32Array.from(fixture.codes, (code) => code * X_SCALE),
        },
      };
      const first = wordsOf((await session.run(input)).z.data).slice();
      const second = wordsOf((await session.run(input)).z.data).slice();
      assertEquals(second, first, `${where}: the prepared plan's run differs from the first`);
      return {
        words: first,
        census: requireCensus(session.diagnostics().lastRunPipelines, where),
      };
    } finally {
      await session.dispose();
    }
  } finally {
    gpu.destroy();
  }
};

/** The mirror's output words for a fixture (with the output-side SRQ). */
const expectedWords = (shape: Shape, fixture: Fixture): Uint32Array<ArrayBuffer> =>
  referenceIntegerGemv({
    weight: shape.variant.weight,
    m: 1,
    n: shape.n,
    k: shape.k,
    codes: packCodes(fixture.codes),
    xScale: X_SCALE,
    weights: fixture.weights,
    wscale: fixture.wscale,
    bias: fixture.bias,
    outputScale: OUTPUT_SCALE,
  });

describe.skipIf(!GPU_AVAILABLE)("integerGemv through a Session (real GPU)", () => {
  for (const shape of SHAPES) {
    describe(shape.name, () => {
      // Built on first use, so collecting the skipped tests on a machine without a GPU costs nothing.
      let fixture: Fixture | undefined;
      const fixtureFor = (): Fixture => (fixture ??= fixtureOf(shape));

      for (const fused of [true, false]) {
        const form = fused ? "with the SRQ epilogue" : "plain, then the SRQ node";
        it(`runs the integer key ${form} and writes the mirror's words`, async () => {
          const where = `${shape.name} ${form}`;
          const { words, census } = await runTwice(
            fixtureFor(),
            {
              linearGemvReduce: "parallel",
              packedStaticQuantize: true,
              integerGemv: true,
              fuseLinearStaticQuantize: fused,
            },
            where,
          );
          assertEquals(countDispatches(census, integerGemvKey(shape.variant, fused)), 1, where);
          assertEquals(countDispatches(census, STATIC_QUANTIZE_PACKED_KEY), 1, where);
          assertEquals(countDispatches(census, shape.parallelKey(fused)), 0, where);
          assertEquals(words, expectedWords(shape, fixtureFor()), where);
        });

        it(`runs the parallel GEMV's key ${form} with the option off`, async () => {
          const where = `${shape.name} ${form} (option off)`;
          const { census } = await runTwice(
            fixtureFor(),
            {
              linearGemvReduce: "parallel",
              packedStaticQuantize: true,
              fuseLinearStaticQuantize: fused,
            },
            where,
          );
          assertEquals(countDispatches(census, shape.parallelKey(fused)), 1, where);
          assertEquals(countDispatches(census, isInteger), 0, where);
        });
      }
    });
  }

  it("rejects integerGemv without packed activations, and a non-boolean, at construction", async () => {
    const gpu = await acquireGpu();
    try {
      const { model } = fixtureOf(SHAPES[0]);
      await assertRejects(
        () =>
          createSessionFromContainer(gpu, model, GRAPH_NAME, {
            linearGemvReduce: "parallel",
            integerGemv: true,
          }),
        ExecutionError,
        "integerGemv requires packedStaticQuantize: true",
      );
      const options: SessionOptions = { linearGemvReduce: "parallel", packedStaticQuantize: true };
      Object.defineProperty(options, "integerGemv", { value: 1, enumerable: true });
      await assertRejects(
        () => createSessionFromContainer(gpu, model, GRAPH_NAME, options),
        ExecutionError,
        "options.integerGemv must be a boolean",
      );
    } finally {
      gpu.destroy();
    }
  });
});
