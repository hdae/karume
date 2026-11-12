// End-to-end check of the execution path of i4 storage (group-symmetric packed 4-bit along K —
// ADR 0069): the eligibility decision (limited to linear / embedding / conv1d(groups==1)), the
// kernel variants, the group scale indexing and the diagnostics.
//
// It exercises the same 2 paths as f16 / i8:
//   eligible (consumed **only by the weight slots of linear / embedding / conv1d with groups == 1**
//     — narrower than i8) → stays packed and resident on the GPU, dequantized inside the kernel
//     (`unpack4xU8` + mask/shift + group scale)
//   ineligible (shared with a weight slot that has no expansion path [conv2d / conv_transpose1d /
//     conv1d with groups > 1], a graph output, etc.) → expanded to f32 on the CPU at load time
//
// MUST: the weights **vary in magnitude per group** (if the scale is the same in every group, a
// mix-up of the scale index [swapping row/group, a wrong shift] never shows in the values — the
// same trap as the per-row scale of i8).
// MUST: the values include **an asymmetric pattern where all adjacent elements differ** (a
// symmetric pattern matches even with the upper and lower nibbles of the pack swapped — ADR 0069
// decision 4 ①).
// MUST: the expected values are built from **the rounded weights** (fake-quant — ADR 0006).

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import type { OpenedContainer } from "../src/format/container/open.ts";
import { decodeI4 } from "../src/format/i4.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { compareTensors, formatAllclose } from "../src/reference/allclose.ts";
import { GEMM_TOLERANCE } from "./helpers/op-tolerance.ts";
import { applyReferenceOp, type RefTensor, refTensor } from "../src/reference/ops.ts";
import { conv1dUsesVec4 } from "../src/kernels/conv1d.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { i4EligibleInitializers } from "../src/runtime/plan.ts";
import { linearI8a8Key, linearI8a8UsesVec4 } from "../src/kernels/linear-i8a8.ts";
import { quantizeRowsTieMargin, referenceLinearW4a8 } from "../src/reference/i8a8.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import { deviceGeometryProfile, expectedW4a8LinearGeometry } from "./helpers/geometry.ts";
import { quantizeI4 } from "./helpers/i4.ts";
import {
  type DeclarationJson,
  f32Bytes,
  fill,
  type FilledTensor,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

const SIGNED = (i: number): number => ((i % 13) - 6) * 0.75;

/**
 * Weights whose amplitude differs per group (a form where the group scale index shows in the values
 * — the safeguard against a vacuous pass). Adjacent elements also alternate in sign, so swapping
 * the upper and lower nibbles of the pack shows in the values too.
 */
const groupVarying =
  (cols: number, groupSize: number) =>
  (i: number): number => {
    const group = Math.floor((i % cols) / groupSize);
    const row = Math.floor(i / cols);
    const base = (0.125 + (i % 11) * 0.5) * (i % 2 === 0 ? 1 : -1);
    return base * (1 + group * 0.75 + (row % 5) * 0.25);
  };

type I4Case = {
  readonly name: string;
  readonly x: FilledTensor;
  readonly weight: FilledTensor;
  readonly bias: FilledTensor;
  readonly groupSize: number;
  readonly outShape: readonly number[];
};

/** A graph with one `linear(x, w, b)` (w is i4 + group scale). `wAsOutput` makes w ineligible. */
const i4LinearModel = (
  testCase: I4Case,
  quantized: ReturnType<typeof quantizeI4>,
  { wAsOutput = false }: { wAsOutput?: boolean } = {},
): Promise<OpenedContainer> => {
  const declaration: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["linear"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [...testCase.x.shape] }],
    outputs: wAsOutput ? ["y", "w"] : ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [...testCase.weight.shape] },
      b: { dtype: "f32", shape: [...testCase.bias.shape] },
      y: { dtype: "f32", shape: [...testCase.outShape] },
    },
    nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
  };
  return openModelBytes(declaration, [
    {
      graph: GRAPH_NAME,
      initializer: "w",
      bytes: quantized.bytes,
      encoding: {
        codec: "int4-sym-g",
        groupSize: testCase.groupSize,
        scale: { bytes: f32Bytes([...quantized.scale]), dtype: "f32" },
      },
    },
    {
      graph: GRAPH_NAME,
      initializer: "b",
      bytes: f32Bytes([...testCase.bias.data]),
      encoding: { codec: "f32" },
    },
  ]);
};

const expectedLinear = (testCase: I4Case, quantized: ReturnType<typeof quantizeI4>): RefTensor =>
  applyReferenceOp(
    "linear",
    [
      testCase.x as RefTensor,
      refTensor(testCase.weight.shape, quantized.values),
      testCase.bias as RefTensor,
    ],
    {},
    testCase.outShape,
  );

/** Shapes that exercise GEMM tile boundaries, group boundaries and both the v4 and scalar paths. */
const CASES: readonly I4Case[] = [
  {
    // v4 path (k and n both multiples of 4). k = 48 is 3 groups of 16 = the scale changes twice
    // within a row, and it also crosses the K tile (16 elements). n = 68 leaves a partial last N
    // tile; m = 65 gives 2 row tiles.
    name: "linear v4 [65,48] × W[68,48] g16",
    x: fill([65, 48], SIGNED),
    weight: fill([68, 48], groupVarying(48, 16)),
    bias: fill([68], SIGNED),
    groupSize: 16,
    outShape: [65, 68],
  },
  {
    // Scalar path (n = 19 is not a multiple of 4). k = 32, group 16 = 2 groups per row.
    name: "linear scalar [17,32] × W[19,32] g16",
    x: fill([17, 32], SIGNED),
    weight: fill([19, 32], groupVarying(32, 16)),
    bias: fill([19], SIGNED),
    groupSize: 16,
    outShape: [17, 19],
  },
  {
    // group 32 (the default — ADR 0069 addendum 1). k = 96 gives 3 groups per row.
    name: "linear v4 [9,96] × W[20,96] g32",
    x: fill([9, 96], SIGNED),
    weight: fill([20, 96], groupVarying(96, 32)),
    bias: fill([20], SIGNED),
    groupSize: 32,
    outShape: [9, 20],
  },
];

test.skipIf(!GPU_AVAILABLE)(
  "linear with w=i4 matches the CPU reference (the rounded weights) (v4 / scalar / 2 group sizes, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of CASES) {
        const quantized = quantizeI4(
          testCase.weight.data,
          testCase.weight.shape,
          testCase.groupSize,
        );
        const session = await createSessionFromContainer(
          gpu,
          await i4LinearModel(testCase, quantized),
          GRAPH_NAME,
        );
        let output: Tensor;
        let residentBytes: number;
        try {
          output = (await session.run({ x: testCase.x }))["y"];
          residentBytes = session.diagnostics().storage.residentCompressedBytes;
        } finally {
          await session.dispose();
        }
        // Stays packed and resident (= did not fall back to CPU expansion). i4 needs no padding by
        // construction (byte length = numel / 2 is always a multiple of 8), so payload + scale fit
        // exactly.
        assertEquals(
          residentBytes,
          quantized.bytes.byteLength + quantized.scale.byteLength,
          `${testCase.name}: compressed bytes resident on the GPU (scale included)`,
        );
        const expected = expectedLinear(testCase, quantized);
        assertEquals(output.shape, expected.shape, testCase.name);
        const report = compareTensors(output, expected, GEMM_TOLERANCE);
        assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "group 16 / 32 in the same graph run correctly in separate pipelines (the g part of the key keeps them apart, real GPU)",
  async () => {
    // 2 linears with the same m / n / k / v4 — keys that differ only in the g part. If the key did
    // not carry the group, one WGSL (with its baked-in shift) would run with the other's assets and
    // the scale index would be off, a silent wrong value.
    const x = fill([9, 32], SIGNED);
    const w16 = fill([20, 32], groupVarying(32, 16));
    const w32 = fill([20, 32], groupVarying(32, 32));
    const bias = fill([20], SIGNED);
    const q16 = quantizeI4(w16.data, w16.shape, 16);
    const q32 = quantizeI4(w32.data, w32.shape, 32);
    const declaration: DeclarationJson = {
      format: "karume-ir",
      version: 2,
      requires: { ops: ["linear"] },
      symbols: [],
      inputs: [{ name: "x", dtype: "f32", shape: [9, 32] }],
      outputs: ["y16", "y32"],
      initializers: { w16: {}, w32: {}, b: {} },
      values: {
        w16: { dtype: "f32", shape: [20, 32] },
        w32: { dtype: "f32", shape: [20, 32] },
        b: { dtype: "f32", shape: [20] },
        y16: { dtype: "f32", shape: [9, 20] },
        y32: { dtype: "f32", shape: [9, 20] },
      },
      nodes: [
        { op: "linear", ins: ["x", "w16", "b"], outs: ["y16"], attrs: {} },
        { op: "linear", ins: ["x", "w32", "b"], outs: ["y32"], attrs: {} },
      ],
    };
    const opened = await openModelBytes(declaration, [
      {
        graph: GRAPH_NAME,
        initializer: "w16",
        bytes: q16.bytes,
        encoding: {
          codec: "int4-sym-g",
          groupSize: 16,
          scale: { bytes: f32Bytes([...q16.scale]), dtype: "f32" },
        },
      },
      {
        graph: GRAPH_NAME,
        initializer: "w32",
        bytes: q32.bytes,
        encoding: {
          codec: "int4-sym-g",
          groupSize: 32,
          scale: { bytes: f32Bytes([...q32.scale]), dtype: "f32" },
        },
      },
      {
        graph: GRAPH_NAME,
        initializer: "b",
        bytes: f32Bytes([...bias.data]),
        encoding: { codec: "f32" },
      },
    ]);
    const gpu = await acquireGpu();
    try {
      const session = await createSessionFromContainer(gpu, opened, GRAPH_NAME);
      try {
        const outputs = await session.run({ x });
        for (const [name, weight, quantized] of [
          ["y16", w16, q16] as const,
          ["y32", w32, q32] as const,
        ]) {
          const expected = applyReferenceOp(
            "linear",
            [x as RefTensor, refTensor(weight.shape, quantized.values), bias as RefTensor],
            {},
            [9, 20],
          );
          const report = compareTensors(outputs[name], expected, GEMM_TOLERANCE);
          assertEquals(report.pass, true, `${name}: ${formatAllclose(report)}`);
        }
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the outputs of eligible (GPU dequant) and ineligible (CPU expansion) match bit for bit (real GPU)",
  async () => {
    // Adding w as a graph output makes it ineligible (treated as consumed by something besides a
    // weight slot), so it is expanded to f32 by decodeI4 at load time. If the dequant matches bit
    // for bit, the same GEMM kernel folds the same f32 MAC sequence in the same order, so y matches
    // **bit for bit** (the e2e form of acceptance ④ of ADR 0069).
    const testCase = CASES[1];
    const quantized = quantizeI4(testCase.weight.data, testCase.weight.shape, testCase.groupSize);
    const gpu = await acquireGpu();
    try {
      const run = async (wAsOutput: boolean) => {
        const session = await createSessionFromContainer(
          gpu,
          await i4LinearModel(testCase, quantized, { wAsOutput }),
          GRAPH_NAME,
        );
        try {
          const outputs = await session.run({ x: testCase.x });
          return {
            y: Float32Array.from(outputs["y"].data as Float32Array),
            diagnostics: session.diagnostics().storage,
          };
        } finally {
          await session.dispose();
        }
      };
      const resident = await run(false);
      const expanded = await run(true);
      assert(resident.diagnostics.residentCompressedBytes > 0, "the eligible side is not resident");
      assertEquals(
        expanded.diagnostics.residentCompressedBytes,
        0,
        "the ineligible side is resident",
      );
      assert(
        expanded.diagnostics.hostExpandedBytes > 0,
        "the ineligible side was not expanded on the CPU",
      );
      assertEquals(
        resident.y,
        expanded.y,
        "the outputs of GPU dequant and CPU expansion differ in bits",
      );
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * A graph holding the embedding vocabulary table `[V,D]` as i4 (`wAsOutput` makes it ineligible).
 */
const i4EmbeddingModel = (
  weight: FilledTensor,
  index: FilledTensor,
  quantized: ReturnType<typeof quantizeI4>,
  groupSize: number,
  { wAsOutput = false }: { wAsOutput?: boolean } = {},
): Promise<OpenedContainer> => {
  const [vocab, hidden] = weight.shape;
  const declaration: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["embedding"] },
    symbols: [],
    inputs: [{ name: "index", dtype: "i32", shape: [...index.shape] }],
    outputs: wAsOutput ? ["y", "w"] : ["y"],
    initializers: { w: {} },
    values: {
      w: { dtype: "f32", shape: [vocab, hidden] },
      y: { dtype: "f32", shape: [index.shape[0], hidden] },
    },
    nodes: [{ op: "embedding", ins: ["w", "index"], outs: ["y"], attrs: { padding_idx: -1 } }],
  };
  return openModelBytes(declaration, [
    {
      graph: GRAPH_NAME,
      initializer: "w",
      bytes: quantized.bytes,
      encoding: {
        codec: "int4-sym-g",
        groupSize,
        scale: { bytes: f32Bytes([...quantized.scale]), dtype: "f32" },
      },
    },
  ]);
};

test.skipIf(!GPU_AVAILABLE)(
  "embedding with w=i4 matches the CPU reference (the rounded table) (2 group sizes, real GPU)",
  async () => {
    // The vocabulary table is `[V,D]` (`channel_rows` is the identity), and groups are cut along
    // the D axis. 1 thread is 1 output element, so the scale is looked up per element — a row /
    // group mix-up shows directly in the values through the amplitude differences of
    // {@link groupVarying}. Only shapes where the group divides D (i4 MUST NOT make partial
    // groups — ADR 0069 decision 2), and both put several groups in a row.
    const cases = [
      { name: "embedding [7,48] g16", vocab: 7, hidden: 48, groupSize: 16, picks: 11 },
      { name: "embedding [5,96] g32", vocab: 5, hidden: 96, groupSize: 32, picks: 9 },
    ] as const;
    const gpu = await acquireGpu();
    try {
      for (const testCase of cases) {
        const weight = fill(
          [testCase.vocab, testCase.hidden],
          groupVarying(testCase.hidden, testCase.groupSize),
        );
        const quantized = quantizeI4(weight.data, weight.shape, testCase.groupSize);
        // Indices that do not walk the vocabulary in order (including forms that expose a row
        // mix-up and that look up the same row twice)
        const index = fill([testCase.picks], (i) => (i * 3 + 1) % testCase.vocab, "i32");
        const session = await createSessionFromContainer(
          gpu,
          await i4EmbeddingModel(weight, index, quantized, testCase.groupSize),
          GRAPH_NAME,
        );
        let output: Tensor;
        let residentBytes: number;
        try {
          output = (await session.run({ index }))["y"];
          residentBytes = session.diagnostics().storage.residentCompressedBytes;
        } finally {
          await session.dispose();
        }
        assertEquals(
          residentBytes,
          quantized.bytes.byteLength + quantized.scale.byteLength,
          `${testCase.name}: compressed bytes resident on the GPU (scale included)`,
        );
        const expected = applyReferenceOp(
          "embedding",
          [refTensor(weight.shape, quantized.values), index as RefTensor],
          { padding_idx: -1 },
          [testCase.picks, testCase.hidden],
        );
        assertEquals(output.shape, expected.shape, testCase.name);
        const report = compareTensors(output, expected, GEMM_TOLERANCE);
        assertEquals(report.pass, true, `${testCase.name}: ${formatAllclose(report)}`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * GPU `unpack4xU8` + group scale and the CPU {@link decodeI4} **match bit for bit for all 15 nibble
 * values × 8 positions within a word** (the i4 mirror of "all 256 values × 4 lanes" on the i8 side,
 * `gpu_f16_i8_weights_test.ts`).
 *
 * How the values are carried: the embedding weight is `[16, 32]` (g16 = 2 groups per row), and the
 * quantized value at row r, column c is `((r + c) % 15) − 7`. The flat index is `r·32 + c`, so
 * **every nibble value appears at all 8 positions within a word** (32 is a multiple of 8, so a row
 * always starts on a word boundary — an invariant that follows from i4's storage constraint [the
 * quantization axis is divisible by a power-of-2 group ≥ 16]; the "row length not a multiple of the
 * word" trap of i8 / f16 does not exist for i4).
 * The amplitude varies by row × group, so a wrong scale index is detected at the same time.
 */
test.skipIf(!GPU_AVAILABLE)(
  "GPU unpack4xU8 and CPU expansion match bit for bit for all 15 nibble values × 8 positions within a word (real GPU)",
  async () => {
    const vocab = 16;
    const hidden = 32;
    const groupSize = 16;
    const weight = fill([vocab, hidden], (i) => {
      const row = Math.floor(i / hidden);
      const col = i % hidden;
      const q = ((row + col) % 15) - 7;
      // Amplitudes that give a different scale per row × group (with equal scales an index mix-up
      // does not show in the values)
      return q * (0.1 + row * 0.017 + Math.floor(col / groupSize) * 0.31);
    });
    const quantized = quantizeI4(weight.data, weight.shape, groupSize);
    // The check against a vacuous pass: all 15 nibble values appear at each of the 8 positions
    // within a word
    for (let position = 0; position < 8; position += 1) {
      const seen = new Set<number>();
      for (let i = position; i < vocab * hidden; i += 8) {
        const byte = quantized.bytes[i >> 1];
        seen.add((i & 1) === 1 ? byte >> 4 : byte & 0x0f);
      }
      assertEquals(
        seen.size,
        15,
        `position ${position} within the word: not all 15 nibble values are present`,
      );
    }
    const index = fill([vocab], (i) => i, "i32");
    const gpu = await acquireGpu();
    let actual: Float32Array<ArrayBuffer>;
    try {
      const session = await createSessionFromContainer(
        gpu,
        await i4EmbeddingModel(weight, index, quantized, groupSize),
        GRAPH_NAME,
      );
      try {
        actual = (await session.run({ index }))["y"].data as Float32Array<ArrayBuffer>;
        assertEquals(
          session.diagnostics().storage.residentCompressedBytes,
          quantized.bytes.byteLength + quantized.scale.byteLength,
        );
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
    const expected = decodeI4(
      quantized.bytes,
      [vocab, hidden],
      quantized.scale,
      quantized.scaleShape,
      groupSize,
    );
    const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
    const expectedBits = new Uint32Array(expected.buffer, expected.byteOffset, expected.length);
    assertEquals(actualBits, expectedBits, "GPU expansion and CPU expansion differ in bits");
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "embedding also matches bit for bit between eligible (GPU dequant) and ineligible (CPU expansion) (real GPU)",
  async () => {
    // The ineligible fallback is the f32 expansion of `[V,D]` by `decodeI4`. If the expansion
    // matches bit for bit, the same gather kernel copies the same bytes, so the output matches
    // **bit for bit**.
    const groupSize = 32;
    const weight = fill([9, 64], groupVarying(64, groupSize));
    const quantized = quantizeI4(weight.data, weight.shape, groupSize);
    const index = fill([13], (i) => (i * 5 + 2) % 9, "i32");
    const gpu = await acquireGpu();
    try {
      const run = async (wAsOutput: boolean) => {
        const session = await createSessionFromContainer(
          gpu,
          await i4EmbeddingModel(weight, index, quantized, groupSize, { wAsOutput }),
          GRAPH_NAME,
        );
        try {
          const outputs = await session.run({ index });
          return {
            y: Float32Array.from(outputs["y"].data as Float32Array),
            diagnostics: session.diagnostics().storage,
          };
        } finally {
          await session.dispose();
        }
      };
      const resident = await run(false);
      const expanded = await run(true);
      assert(resident.diagnostics.residentCompressedBytes > 0, "the eligible side is not resident");
      assertEquals(
        expanded.diagnostics.residentCompressedBytes,
        0,
        "the ineligible side is resident",
      );
      assert(
        expanded.diagnostics.hostExpandedBytes > 0,
        "the ineligible side was not expanded on the CPU",
      );
      assertEquals(
        resident.y,
        expanded.y,
        "the outputs of GPU dequant and CPU expansion differ in bits",
      );
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// Implicit GEMM of conv1d (the conv1d supplement to ADR 0069 decision 5, wave J-5b)
// ---------------------------------------------------------------------------

type ConvAttrs = {
  readonly stride: number;
  readonly padding: number;
  readonly dilation: number;
};

/** A graph with one `conv1d(x, w, b)`. `storage` switches between i4 resident and the f32 twin. */
const conv1dModel = (
  x: FilledTensor,
  weight: FilledTensor,
  bias: FilledTensor,
  quantized: ReturnType<typeof quantizeI4>,
  groupSize: number,
  attrs: ConvAttrs,
  outShape: readonly number[],
  {
    storage = "i4",
    wAsOutput = false,
    groups = 1,
  }: { storage?: "i4" | "f32"; wAsOutput?: boolean; groups?: number } = {},
): Promise<OpenedContainer> => {
  const declaration: DeclarationJson = {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["conv1d"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [...x.shape] }],
    outputs: wAsOutput ? ["y", "w"] : ["y"],
    initializers: { w: {}, b: {} },
    values: {
      w: { dtype: "f32", shape: [...weight.shape] },
      b: { dtype: "f32", shape: [...bias.shape] },
      y: { dtype: "f32", shape: [...outShape] },
    },
    nodes: [{ op: "conv1d", ins: ["x", "w", "b"], outs: ["y"], attrs: { ...attrs, groups } }],
  };
  const biasTensor: TensorInput = {
    graph: GRAPH_NAME,
    initializer: "b",
    bytes: f32Bytes([...bias.data]),
    encoding: { codec: "f32" },
  };
  return openModelBytes(
    declaration,
    storage === "i4"
      ? [
          {
            graph: GRAPH_NAME,
            initializer: "w",
            bytes: quantized.bytes,
            encoding: {
              codec: "int4-sym-g",
              groupSize,
              scale: { bytes: f32Bytes([...quantized.scale]), dtype: "f32" },
            },
          },
          biasTensor,
        ]
      : // The twin (f32 resident) multiplies **the rounded weights** as f32 directly. The same igemm
        // kernel folds the same f32 MAC sequence in the same order, so if the GPU dequant of i4
        // matches the CPU expansion bit for bit, the output matches bit for bit too.
        [
          {
            graph: GRAPH_NAME,
            initializer: "w",
            bytes: f32Bytes([...quantized.values]),
            encoding: { codec: "f32" },
          },
          biasTensor,
        ],
  );
};

const outLength = (
  lengthIn: number,
  { stride, padding, dilation }: ConvAttrs,
  kernel: number,
): number => Math.floor((lengthIn + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1;

type ConvCase = {
  readonly name: string;
  readonly batch: number;
  readonly channelsIn: number;
  readonly channelsOut: number;
  readonly lengthIn: number;
  readonly kernel: number;
  readonly groupSize: number;
  readonly attrs: ConvAttrs;
  /** The A-side variant it should exercise (quad expansion or expansion of 4 separate elements). */
  readonly expectVec4: boolean;
};

/**
 * Shapes for the i4 variant of conv1d igemm.
 *
 * MUST: every shape **puts several groups in the row length `Cin·K`** (with 1 group per row, a
 * mix-up of the scale index [swapping row and group, a wrong shift] never shows in the values).
 * NOTE: the A side of an eligible i4 conv has `kFlat % group_size == 0` (power of 2 ≥ 16), so
 * `kFlat % 4 == 0` always holds — only `Lout % 4` and `stride` separate v4 / scalar
 * (conv1dUsesVec4).
 */
const CONV_CASES: readonly ConvCase[] = [
  {
    // v4 path (Lout = 16, stride 1). Cout = 70 has `70 % 64 = 6`, so it picks **the 32-row m tile**
    // and the rows split into 3 tiles (a detector for the mix-up of reading the tile-relative row
    // instead of `arow`). Row length Cin·K = 96 holds 3 g32 groups. B = 2 also exercises the batch
    // axis (z).
    name: "conv1d v4 [2,32,16] * W[70,32,3] pad1 g32",
    batch: 2,
    channelsIn: 32,
    channelsOut: 70,
    lengthIn: 16,
    kernel: 3,
    groupSize: 32,
    attrs: { stride: 1, padding: 1, dilation: 1 },
    expectVec4: true,
  },
  {
    // Scalar path (Lout = 13 is not a multiple of 4). Row length 32 holds 2 g16 groups; Cout = 19.
    name: "conv1d scalar [1,16,14] * W[19,16,2] pad0 g16",
    batch: 1,
    channelsIn: 16,
    channelsOut: 19,
    lengthIn: 14,
    kernel: 2,
    groupSize: 16,
    attrs: { stride: 1, padding: 0, dilation: 1 },
    expectVec4: false,
  },
  {
    // stride / padding / dilation all non-trivial. Lout = 10 (scalar), row length Cin·K = 48 holds
    // 3 g16 groups, Cout = 36 (`36 % 64 = 36`, so the 64-row m tile).
    name: "conv1d stride2 dil3 pad2 [1,16,21] * W[36,16,3] g16",
    batch: 1,
    channelsIn: 16,
    channelsOut: 36,
    lengthIn: 21,
    kernel: 3,
    groupSize: 16,
    attrs: { stride: 2, padding: 2, dilation: 3 },
    expectVec4: false,
  },
];

test.skipIf(!GPU_AVAILABLE)(
  "conv1d(igemm) with w=i4 matches the f32 twin (the rounded weights) bit for bit (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      for (const testCase of CONV_CASES) {
        const { batch, channelsIn, channelsOut, lengthIn, kernel, groupSize, attrs } = testCase;
        const lengthOut = outLength(lengthIn, attrs, kernel);
        // Pin the variant it exercises explicitly (so a wrong decision does not hide as "both use
        // the same variant")
        assertEquals(
          conv1dUsesVec4(channelsIn * kernel, lengthOut, attrs.stride),
          testCase.expectVec4,
          `${testCase.name}: the variant exercised is not the expected one`,
        );
        const x = fill([batch, channelsIn, lengthIn], SIGNED);
        const weight = fill(
          [channelsOut, channelsIn, kernel],
          groupVarying(channelsIn * kernel, groupSize),
        );
        const bias = fill([channelsOut], SIGNED);
        const quantized = quantizeI4(weight.data, weight.shape, groupSize);
        // The scale has the rank-independent rank-2 form (`[Cout, (Cin·K)/g]` — the only new form
        // of wave J-5b)
        assertEquals(
          quantized.scaleShape,
          [channelsOut, (channelsIn * kernel) / groupSize],
          `${testCase.name}: the group form of the scale`,
        );
        const outShape = [batch, channelsOut, lengthOut];
        const run = async (storage: "i4" | "f32") => {
          const session = await createSessionFromContainer(
            gpu,
            await conv1dModel(x, weight, bias, quantized, groupSize, attrs, outShape, { storage }),
            GRAPH_NAME,
          );
          try {
            const outputs = await session.run({ x });
            return {
              y: Float32Array.from(outputs["y"].data as Float32Array),
              shape: outputs["y"].shape,
              resident: session.diagnostics().storage.residentCompressedBytes,
            };
          } finally {
            await session.dispose();
          }
        };
        const packed = await run("i4");
        const twin = await run("f32");
        // Stays packed and resident (= did not fall back to CPU expansion)
        assertEquals(
          packed.resident,
          quantized.bytes.byteLength + quantized.scale.byteLength,
          `${testCase.name}: compressed bytes resident on the GPU (scale included)`,
        );
        assertEquals(twin.resident, 0, `${testCase.name}: the f32 twin is resident compressed`);
        assertEquals(packed.shape, outShape, testCase.name);
        assertEquals(
          new Uint32Array(packed.y.buffer),
          new Uint32Array(twin.y.buffer),
          `${testCase.name}: the outputs of i4 and the f32 twin differ in bits`,
        );
        // The check against a vacuous pass (with a constant output the match checks nothing)
        assert(new Set(packed.y).size > 1, `${testCase.name}: constant output`);
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "conv1d(i4) also matches bit for bit between eligible (GPU dequant) and ineligible (CPU expansion, rank 3) (real GPU)",
  async () => {
    // Adding w as a graph output makes it ineligible, so it is expanded to f32 by `decodeI4` at
    // load time.
    // Checks that the CPU side reads **rank-3 weights × a rank-2 scale** with the same index
    // expression as the GPU.
    const testCase = CONV_CASES[1];
    const { batch, channelsIn, channelsOut, lengthIn, kernel, groupSize, attrs } = testCase;
    const x = fill([batch, channelsIn, lengthIn], SIGNED);
    const weight = fill(
      [channelsOut, channelsIn, kernel],
      groupVarying(channelsIn * kernel, groupSize),
    );
    const bias = fill([channelsOut], SIGNED);
    const quantized = quantizeI4(weight.data, weight.shape, groupSize);
    const outShape = [batch, channelsOut, outLength(lengthIn, attrs, kernel)];
    const gpu = await acquireGpu();
    try {
      const run = async (wAsOutput: boolean) => {
        const session = await createSessionFromContainer(
          gpu,
          await conv1dModel(x, weight, bias, quantized, groupSize, attrs, outShape, { wAsOutput }),
          GRAPH_NAME,
        );
        try {
          const outputs = await session.run({ x });
          return {
            y: Float32Array.from(outputs["y"].data as Float32Array),
            diagnostics: session.diagnostics().storage,
          };
        } finally {
          await session.dispose();
        }
      };
      const resident = await run(false);
      const expanded = await run(true);
      assert(resident.diagnostics.residentCompressedBytes > 0, "the eligible side is not resident");
      assertEquals(
        expanded.diagnostics.residentCompressedBytes,
        0,
        "the ineligible side is resident",
      );
      assert(
        expanded.diagnostics.hostExpandedBytes > 0,
        "the ineligible side was not expanded on the CPU",
      );
      assertEquals(
        resident.y,
        expanded.y,
        "the outputs of GPU dequant and CPU expansion differ in bits",
      );
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "conv1d with groups > 1 does not keep i4 resident and falls back to CPU expansion (real GPU)",
  async () => {
    // The direct kernel (groups > 1) has no expansion path. The eligibility decision drops it, so
    // it does not stay packed and resident but goes to the CPU-expansion fallback (correctness is
    // kept; the VRAM saving is zero).
    const groupSize = 16;
    const groups = 2;
    const x = fill([1, 8, 12], SIGNED);
    // The weights' second axis is Cin/groups, so the row length is 4·4 = 16 (one g16 group)
    const weight = fill([6, 4, 4], groupVarying(16, groupSize));
    const bias = fill([6], SIGNED);
    const quantized = quantizeI4(weight.data, weight.shape, groupSize);
    const attrs: ConvAttrs = { stride: 1, padding: 0, dilation: 1 };
    const outShape = [1, 6, outLength(12, attrs, 4)];
    const gpu = await acquireGpu();
    try {
      const session = await createSessionFromContainer(
        gpu,
        await conv1dModel(x, weight, bias, quantized, groupSize, attrs, outShape, { groups }),
        GRAPH_NAME,
      );
      try {
        const output = await session.run({ x });
        const diagnostics = session.diagnostics().storage;
        assertEquals(diagnostics.residentCompressedBytes, 0, "resident although groups > 1");
        assert(diagnostics.hostExpandedBytes > 0, "not expanded on the CPU");
        const expected = applyReferenceOp(
          "conv1d",
          [x as RefTensor, refTensor(weight.shape, quantized.values), bias as RefTensor],
          { ...attrs, groups },
          outShape,
        );
        const report = compareTensors(output["y"], expected, GEMM_TOLERANCE);
        assertEquals(report.pass, true, formatAllclose(report));
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * The i4 expansion on the GEMM **A side** (the weight tile of conv1d igemm) **matches the CPU
 * {@link decodeI4} bit for bit for all 15 nibble values × 8 positions within a word** (the conv
 * mirror of the B side = the embedding version for linear [the gather check above]).
 *
 * How the values are carried: with `K = 1` and a **one-hot** input (`x[b, ic, l] = (ic == b) ? 1 :
 * 0`), `out[b, oc, l] = bias[oc] + Σ_ic x[b,ic,l]·w[oc,ic,0] = w[oc, b, 0]` (bias = 0), which is
 * the identity **1 output element = 1 weight element**. A term multiplied by 0 is `+0.0`, and
 * adding it to a finite acc does not move bits (a signed zero is kept too: `+0.0 + 0.0 = +0.0`).
 * With the quantized value of the weights `[16, 32, 1]` set to `((r + c) % 15) − 7`, the 32 of the
 * flat index `r·32 + c` is a multiple of 8, so **every nibble value appears at all 8 positions
 * within a word**. The amplitude varies by row × group, so a wrong scale index is detected at the
 * same time.
 * MUST: exercise **both** v4 (`dequant4`) and scalar (`dequant`) — only `Lout % 4` separates them.
 */
test.skipIf(!GPU_AVAILABLE)(
  "the A-side i4 expansion of conv1d matches CPU expansion bit for bit for all 15 nibble values × 8 positions within a word (real GPU)",
  async () => {
    const channelsOut = 16;
    const channelsIn = 32;
    const groupSize = 16;
    const weight = fill([channelsOut, channelsIn, 1], (i) => {
      const row = Math.floor(i / channelsIn);
      const col = i % channelsIn;
      const q = ((row + col) % 15) - 7;
      // Amplitudes that give a different scale per row × group (with equal scales an index mix-up
      // does not show in the values)
      return q * (0.1 + row * 0.017 + Math.floor(col / groupSize) * 0.31);
    });
    const quantized = quantizeI4(weight.data, weight.shape, groupSize);
    // The check against a vacuous pass: all 15 nibble values appear at each of the 8 positions
    // within a word
    for (let position = 0; position < 8; position += 1) {
      const seen = new Set<number>();
      for (let i = position; i < channelsOut * channelsIn; i += 8) {
        const byte = quantized.bytes[i >> 1];
        seen.add((i & 1) === 1 ? byte >> 4 : byte & 0x0f);
      }
      assertEquals(
        seen.size,
        15,
        `position ${position} within the word: not all 15 nibble values are present`,
      );
    }
    const expected = decodeI4(
      quantized.bytes,
      [...weight.shape],
      quantized.scale,
      quantized.scaleShape,
      groupSize,
    );
    const bias = fill([channelsOut], () => 0);
    const attrs: ConvAttrs = { stride: 1, padding: 0, dilation: 1 };
    const gpu = await acquireGpu();
    try {
      // Exercise both Lout = 4 (v4 = `dequant4`) and Lout = 3 (scalar = `dequant`)
      for (const lengthIn of [4, 3]) {
        const x = fill([channelsIn, channelsIn, lengthIn], (i) =>
          Math.floor(i / (channelsIn * lengthIn)) === Math.floor(i / lengthIn) % channelsIn ? 1 : 0,
        );
        const outShape = [channelsIn, channelsOut, lengthIn];
        assertEquals(
          conv1dUsesVec4(channelsIn, lengthIn, 1),
          lengthIn % 4 === 0,
          `Lin=${lengthIn}: variant exercised`,
        );
        const session = await createSessionFromContainer(
          gpu,
          await conv1dModel(x, weight, bias, quantized, groupSize, attrs, outShape),
          GRAPH_NAME,
        );
        let actual: Float32Array;
        try {
          actual = (await session.run({ x }))["y"].data as Float32Array;
        } finally {
          await session.dispose();
        }
        // out[b][oc][l] = w[oc][b][0] (the one-hot identity) — compare after mapping back to the
        // weight-side index
        const actualBits = new Uint32Array(actual.buffer, actual.byteOffset, actual.length);
        const expectedBits = new Uint32Array(
          Float32Array.from({ length: channelsIn * channelsOut * lengthIn }, (_, i) => {
            const batch = Math.floor(i / (channelsOut * lengthIn));
            const channel = Math.floor(i / lengthIn) % channelsOut;
            return expected[channel * channelsIn + batch];
          }).buffer,
        );
        assertEquals(
          actualBits,
          expectedBits,
          `Lin=${lengthIn}: A-side expansion differs from CPU expansion`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------
// Eligibility decision (the narrowing for i4 — GPU independent)
// ---------------------------------------------------------------------------

/** conv1d attrs (`groups` is not filled with a default — ADR 0015). */
const convAttrs = (groups: number) => ({ stride: 1, padding: 0, dilation: 1, groups });

test("i4 eligibility is limited to the weight slots of linear / embedding / conv1d with groups==1", () => {
  // The only kernels with an expansion path (`unpack4xU8` + group scale) are linear (the GEMM B
  // tile), embedding and the implicit GEMM of conv1d (the GEMM A tile). Keeping a weight resident
  // that is also consumed elsewhere makes that other kernel read the packed bytes as f32 (a silent
  // wrong value with no exception).
  // This looks **only at the eligibility decision**, so it passes forms directly that work even
  // where the shape contract would not hold (the same tensor placed in both a rank-2 and a rank-3
  // weight slot).
  const eligible = (nodes: readonly unknown[]): readonly string[] =>
    [...i4EligibleInitializers({ initializers: { w: {} }, nodes } as never)].sort();
  assertEquals(eligible([{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }]), ["w"]);
  assertEquals(eligible([{ op: "embedding", ins: ["w", "x"], outs: ["y"], attrs: {} }]), ["w"]);
  // A weight consumed by both linear and embedding (tied lm_head) is eligible, since both have an
  // expansion path
  assertEquals(
    eligible([
      { op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} },
      { op: "embedding", ins: ["w", "x"], outs: ["z"], attrs: {} },
    ]),
    ["w"],
  );
  // For conv1d **only groups == 1 (implicit GEMM)** is eligible. groups > 1 goes to the direct
  // kernel, which has no expansion path (wave J-5b).
  assertEquals(
    eligible([{ op: "conv1d", ins: ["x", "w", "b"], outs: ["y"], attrs: convAttrs(1) }]),
    ["w"],
  );
  assertEquals(
    eligible([{ op: "conv1d", ins: ["x", "w", "b"], outs: ["y"], attrs: convAttrs(2) }]),
    [],
  );
  // Sharing between linear and a groups==1 conv1d stays eligible, since both have an expansion path
  assertEquals(
    eligible([
      { op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} },
      { op: "conv1d", ins: ["x", "w", "b"], outs: ["z"], attrs: convAttrs(1) },
    ]),
    ["w"],
  );
  // Exclusive narrowing for mixed consumers: a single side without an expansion path drops it
  assertEquals(
    eligible([
      { op: "conv1d", ins: ["x", "w", "b"], outs: ["y"], attrs: convAttrs(1) },
      { op: "conv1d", ins: ["x", "w", "b"], outs: ["z"], attrs: convAttrs(4) },
    ]),
    [],
    "shared by groups 1 and groups 4",
  );
  for (const op of ["conv2d", "conv_transpose1d"]) {
    assertEquals(eligible([{ op, ins: ["x", "w", "b"], outs: ["y"], attrs: {} }]), [], op);
    assertEquals(
      eligible([
        { op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} },
        { op, ins: ["x", "w", "b"], outs: ["z"], attrs: {} },
      ]),
      [],
      `shared by linear and ${op}`,
    );
  }
  // MUST: conv1d's `groups` is not filled with a default (missing means fail loudly — silently
  // assuming 1 keeps depthwise weights resident as i4 and the direct kernel reads the packed bytes
  // as f32)
  assertThrows(
    () =>
      eligible([
        {
          op: "conv1d",
          ins: ["x", "w", "b"],
          outs: ["y"],
          attrs: { stride: 1, padding: 0, dilation: 1 },
        },
      ]),
    Error,
    "attrs.groups",
  );
  // Consumers **other than** weight slots are not looked at here (used together with
  // `eligibleCompressedInitializers`)
  assertEquals(eligible([{ op: "add", ins: ["w", "w"], outs: ["y"], attrs: {} }]), []);
});

test.skipIf(!GPU_AVAILABLE)(
  "i4 resident weights × linearCompute 'f16' (w4a16) is fail loudly (real GPU)",
  async () => {
    const testCase = CASES[1];
    const quantized = quantizeI4(testCase.weight.data, testCase.weight.shape, testCase.groupSize);
    // The f16 compute variants require 'shader-f16' at device creation (without it a different
    // check fails first and the w4a16 check is never reached).
    const gpu = await acquireGpu({ shaderF16: true });
    try {
      const session = await createSessionFromContainer(
        gpu,
        await i4LinearModel(testCase, quantized),
        GRAPH_NAME,
        { linearCompute: "f16" },
      );
      try {
        await assertRejects(() => session.run({ x: testCase.x }), Error, "w4a16");
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "linearCompute 'a8' × i4 resident runs on the w4a8 path (integer dot products) (real GPU)",
  async () => {
    // The knob `linearCompute: "a8"` means "make the activations i8 and compute with integer dot
    // products"; the weight storage is a separate axis (i8 resident → w8a8 / i4 resident → w4a8 —
    // perf-ledger Q-8). The intent of **pinning this combination's behaviour here** is unchanged,
    // and the expectation has been rewritten to the w4a8 path
    // (it used to pin "i4 resident goes to the f32 compute path" — that behaviour no longer
    // exists).
    // Full coverage of the path itself (v4 / tile edges / 2 group sizes / dp4a comparison) is in
    // the w4a8 section of tests/gpu_i8a8_test.ts.
    const testCase = CASES[1];
    const { m, n, k, groupSize } = {
      m: testCase.x.shape[0],
      n: testCase.weight.shape[0],
      k: testCase.weight.shape[1],
      groupSize: testCase.groupSize,
    };
    const quantized = quantizeI4(testCase.weight.data, testCase.weight.shape, testCase.groupSize);
    // MUST: the activations are a sequence **far enough from the rounding boundaries**. This
    // section's SIGNED (steps of 0.75) puts x/s on half-integers, and the 2.5 ULP allowed for GPU
    // division makes the quantized value wobble by ±1 step (measured margin 3.8e-6). atol=0 is "a
    // contract that holds only for data away from the boundaries", so this uses the same
    // non-commensurate step as the i8a8 side and makes the margin, measured every time, a check
    // (the docstring of src/reference/i8a8.ts).
    const x = fill([m, k], (i) => ((i % 29) - 14) * 0.3717 + 0.0131);
    const margin = quantizeRowsTieMargin(x.data, m, k);
    assert(margin > 1e-3, `the margin from the rounding boundaries is only ${margin}`);
    const expected = referenceLinearW4a8({
      x: x.data,
      weight: quantized.q,
      weightScale: quantized.scale,
      bias: testCase.bias.data,
      m,
      n,
      k,
      groupSize,
    });
    const gpu = await acquireGpu();
    const profile = deviceGeometryProfile(gpu);
    try {
      const session = await createSessionFromContainer(
        gpu,
        await i4LinearModel(testCase, quantized),
        GRAPH_NAME,
        { linearCompute: "a8" },
      );
      try {
        const output = (await session.run({ x }))["y"];
        // MUST: atol=0 (inside a group it is an exact i32 dot product, with only k/g + 1 floating
        // point operations). There is structurally no room for a tolerance to absorb, so allclose
        // would be too loose.
        assertEquals(output.shape, [m, n], "output shape");
        for (let i = 0; i < expected.length; i += 1) {
          assertEquals(output.data[i], expected[i], `w4a8 output[${i}]`);
        }
        // The diagnostics show that w4a8 ran (storage discriminator + group length — ADR 0021 /
        // 0069)
        const diagnostics = session.diagnostics();
        assertEquals(diagnostics.pipelineCount, 2, "2 pipelines: quantize_rows + w4a8 GEMM");
        // MUST: an empty census is red — guarding with `keys.length > 0` makes the key check miss
        // silently.
        const keys = requireCensus(diagnostics.lastRunPipelines, "w4a8").map((row) => row.key);
        assertEquals(
          keys.includes(
            linearI8a8Key(
              linearI8a8UsesVec4(n),
              true,
              expectedW4a8LinearGeometry(profile, m, groupSize),
              "i4",
              groupSize,
            ),
          ),
          true,
          `keys that ran: ${keys.join(" / ")}`,
        );
        assert(
          keys.some((key) => key.endsWith(`:wi4g${groupSize}`)),
          "the diagnostic keys do not carry the group length of the i4 resident weights",
        );
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);
