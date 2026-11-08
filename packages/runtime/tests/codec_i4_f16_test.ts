// The codec `int4-sym-g-f16` (ADR 0145 decision 3): the packed bytes, nibble order, grouping and
// row axis of `int4-sym-g` with one f16 step per group (two per u32 word, step 2i in the low half),
// under its own layout `i4f16`. Only `expert_linear` reads it; no op does yet, so every existing
// consumer refuses it loudly.
//
// The expected byte counts and step bits are written out by hand (deriving them from `scaleBytes`
// or the decode would make the checks tautological).

import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { mergedGraph } from "../src/format/container/bind.ts";
import {
  CODEC_NAMES,
  codecEntry,
  codecLayout,
  scaleBytes,
} from "../src/format/container/codecs.ts";
import { ContainerFormatError } from "../src/format/container/header.ts";
import { openMemoryContainer } from "../src/format/container/memory.ts";
import { openContainer } from "../src/format/container/open.ts";
import { decodeI4, decodeI4F16Steps, I4Error } from "../src/format/i4.ts";
import { type IrDeclaration, parseIrDeclarationValue } from "../src/format/ir.ts";
import { RuntimeSupportError } from "../src/ops/support.ts";
import { ExecutionError } from "../src/runtime/plan.ts";
import { planWeightResidency } from "../src/runtime/weight-residency.ts";
import { type EncodingInput, writeModelContainer } from "./helpers/container-write.ts";
import { openGraph, weight } from "./helpers/estimate-graphs.ts";
import { mergeGraph } from "./helpers/merged-graph.ts";
import type { DeclarationJson } from "./helpers/model-fixture.ts";

const REFUSAL = "int4-sym-g-f16 (f16 steps) is read by expert_linear only — ADR 0145";

const zeros = (length: number): Uint8Array<ArrayBuffer> => new Uint8Array(new ArrayBuffer(length));

/** Bytes counting up from `seed` (distinct per tensor, so a mix-up shows). */
const patternBytes = (length: number, seed: number): Uint8Array<ArrayBuffer> =>
  Uint8Array.from({ length }, (_, i) => (seed + i) & 0xff);

/** One graph `y = matmul(x, w)` with `w` of `shape` (the consumer does not matter to the merge). */
const weightGraph = (shape: readonly number[]): IrDeclaration =>
  parseIrDeclarationValue({
    format: "karume-ir",
    version: 2,
    requires: { ops: ["matmul"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [2, shape[0]] }],
    outputs: ["y"],
    initializers: { w: {} },
    values: {
      w: { dtype: "f32", shape: [...shape] },
      y: { dtype: "f32", shape: [2, shape.slice(1).reduce((count, dim) => count * dim, 1)] },
    },
    nodes: [{ op: "matmul", ins: ["x", "w"], outs: ["y"], attrs: {} }],
  });

const PROVENANCE = { license: "apache-2.0", writer: "test" } as const;

/** Writes `w` (packed i4 bytes of `shape`) with `encoding` and opens the container. */
const openWeight = async (
  shape: readonly number[],
  encoding: EncodingInput,
  options: { readonly blockBytes?: number } = {},
) => {
  const numel = shape.reduce((count, dim) => count * dim, 1);
  const written = await writeModelContainer(
    {
      graphs: { g: weightGraph(shape) },
      consts: [],
      weights: [{ graph: "g", initializer: "w", bytes: patternBytes(numel / 2, 11), encoding }],
      assets: [],
      provenance: PROVENANCE,
    },
    { partBytes: 4096, blockBytes: options.blockBytes ?? 512 },
  );
  return await openContainer({ kind: "parts", parts: written.parts });
};

describe("codec ledger: int4-sym-g-f16", () => {
  it("has the packing, grouping, zero point and row axes of int4-sym-g", () => {
    const sibling = codecEntry("int4-sym-g-f16");
    const base = codecEntry("int4-sym-g");
    assertEquals(
      { ...sibling, layout: base.layout, scaleDtype: base.scaleDtype },
      base,
      "only the layout and the step dtype differ",
    );
  });

  it("has its own layout i4f16, so the step dtype travels in the codec name", () => {
    assertEquals(codecLayout("int4-sym-g-f16"), "i4f16");
    assertEquals(codecLayout("int4-sym-g"), "i4");
  });

  it("implies f16 steps, while int4-sym-g and every other quantized codec keep f32", () => {
    const dtypes = Object.fromEntries(
      CODEC_NAMES.map((codec) => [codec, codecEntry(codec).scaleDtype ?? "none"]),
    );
    assertEquals(dtypes, {
      f32: "none",
      f16: "none",
      bf16: "none",
      i32: "none",
      "int8-sym": "f32",
      "int4-sym-g": "f32",
      "int4-sym-g-f16": "f16",
      "int2-off": "f32",
      ternary: "f32",
    });
  });
});

describe("scaleBytes", () => {
  it("counts 4 bytes per f32 step of int4-sym-g", () => {
    assertEquals(
      [0, 1, 2, 3, 4].map((count) => scaleBytes("int4-sym-g", count, "test")),
      [0, 4, 8, 12, 16],
    );
  });

  it("counts 2 bytes per f16 step, an odd step count rounded up to a whole u32 word", () => {
    assertEquals(
      [0, 1, 2, 3, 4, 5].map((count) => scaleBytes("int4-sym-g-f16", count, "test")),
      [0, 4, 4, 8, 8, 12],
    );
  });

  it("refuses a codec that has no scale", () => {
    assertThrows(() => scaleBytes("f32", 1, "test"), Error, "is not quantized");
  });
});

describe("descriptor and merge layer: int4-sym-g-f16", () => {
  for (const { shape, steps, bytes } of [
    { shape: [3, 16], steps: 3, bytes: 8 },
    { shape: [4, 16], steps: 4, bytes: 8 },
    { shape: [5, 16], steps: 5, bytes: 12 },
  ]) {
    it(`accepts an f16 step table of ${steps} steps as ${bytes} bytes (padded to a whole word)`, async () => {
      const opened = await openWeight(shape, {
        codec: "int4-sym-g-f16",
        groupSize: 16,
        scale: { bytes: patternBytes(bytes, 21), dtype: "f16" },
      });
      const bound = opened.graphs["g"];
      const supply = bound.supplies.get("w");
      assertEquals(supply?.encoding.scale?.dtype, "f16");
      assertEquals(supply?.scale?.payloadBytes, bytes);
      assertEquals(await opened.readBlock(supply?.scale?.id ?? ""), patternBytes(bytes, 21));
      // The merged graph carries the codec name, and with it the step dtype.
      assertEquals(mergedGraph(bound, "g").initializers["w"].storage, {
        codec: "int4-sym-g-f16",
        groupSize: 16,
        rowAxis: 0,
      });
    });
  }

  it("refuses a scale block sized for f32 steps", async () => {
    // 3 steps: 8 bytes of f16 steps, not the 12 of f32 steps.
    await assertRejects(
      () =>
        openWeight([3, 16], {
          codec: "int4-sym-g-f16",
          groupSize: 16,
          scale: { bytes: zeros(12), dtype: "f16" },
        }),
      ContainerFormatError,
      "is not payload 8 bytes + padding",
    );
  });

  it("refuses int4-sym-g-f16 declared with an f32 scale", async () => {
    await assertRejects(
      () =>
        openWeight([4, 16], {
          codec: "int4-sym-g-f16",
          groupSize: 16,
          scale: { bytes: zeros(16), dtype: "f32" },
        }),
      ContainerFormatError,
      "is not in the accepted set of codec 'int4-sym-g-f16' (f16 steps): f32",
    );
  });

  it("refuses int4-sym-g declared with an f16 scale", async () => {
    await assertRejects(
      () =>
        openWeight([4, 16], {
          codec: "int4-sym-g",
          groupSize: 16,
          scale: { bytes: zeros(8), dtype: "f16" },
        }),
      ContainerFormatError,
      "is not in the accepted set of codec 'int4-sym-g' (f32 steps): f16",
    );
  });

  it("splits an odd f16 step table on the data's pieces by whole rows, padding only the last", async () => {
    // [9, 48] with group 16: 3 steps (6 bytes) per row, 27 steps = 56 bytes once padded. At a
    // 48-byte block limit the data (24 bytes per row) is cut 2 rows per piece, and the scale (over
    // the limit) into 12-byte pieces on the same rows, the last one row of 6 bytes + 2 of padding.
    // A row of the scale is 6 bytes — not 56 / 9.
    const steps = patternBytes(9 * 6, 31);
    const opened = await openWeight(
      [9, 48],
      { codec: "int4-sym-g-f16", groupSize: 16, scale: { bytes: steps, dtype: "f16" } },
      { blockBytes: 48 },
    );
    const scale = opened.graphs["g"].supplies.get("w")?.scale;
    assert(scale?.pieces !== undefined, "the scale is split");
    assertEquals(
      scale.pieces.map((piece) => [piece.rows, piece.payloadBytes, piece.length]),
      [
        [[0, 2], 12, 12],
        [[2, 4], 12, 12],
        [[4, 6], 12, 12],
        [[6, 8], 12, 12],
        [[8, 9], 6, 8],
      ],
    );
    assertEquals(scale.payloadBytes, 56);
    const joined = await opened.readBlock(scale.id);
    assertEquals(joined, Uint8Array.from([...steps, 0, 0]));
  });
});

describe("in-memory container: int4-sym-g-f16", () => {
  it("declares the f16 scale dtype the codec implies", () => {
    const bound = openMemoryContainer({
      graphs: { g: weightGraph([3, 16]) },
      tensors: {
        g: {
          w: {
            bytes: zeros(24),
            encoding: { codec: "int4-sym-g-f16", groupSize: 16, scale: zeros(8) },
          },
        },
      },
    });
    const supply = bound.graphs["g"].supplies.get("w");
    assertEquals(supply?.encoding.scale?.dtype, "f16");
    assertEquals(supply?.scale?.payloadBytes, 8);
  });
});

// ---------------------------------------------------------------------------
// Every existing consumer refuses it
// ---------------------------------------------------------------------------

/** `y = linear(x, w, b)` with `w [3, 16]` (3 groups of 16 — an odd step count). */
const linearGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 16] }],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [3, 16] },
    b: { dtype: "f32", shape: [3] },
    y: { dtype: "f32", shape: [2, 3] },
  },
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
});

/** `y = embedding(w, ids)` with the table `w [4, 32]`. */
const embeddingGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["embedding"] },
  symbols: [],
  inputs: [{ name: "ids", dtype: "i32", shape: [2] }],
  outputs: ["y"],
  initializers: { w: {} },
  values: {
    w: { dtype: "f32", shape: [4, 32] },
    y: { dtype: "f32", shape: [2, 32] },
  },
  nodes: [{ op: "embedding", ins: ["w", "ids"], outs: ["y"], attrs: { padding_idx: -1 } }],
});

/** `y = conv1d(x, w, b)` with `groups == 1` (row length 8 · 2 = 16, one group per output channel). */
const conv1dGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["conv1d"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [1, 8, 6] }],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [4, 8, 2] },
    b: { dtype: "f32", shape: [4] },
    y: { dtype: "f32", shape: [1, 4, 5] },
  },
  nodes: [
    {
      op: "conv1d",
      ins: ["x", "w", "b"],
      outs: ["y"],
      attrs: { stride: 1, padding: 0, dilation: 1, groups: 1 },
    },
  ],
});

/** `y = add(x, w)` — `w` consumed outside any weight slot (the CPU-expansion path). */
const addGraph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["add"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, 16] }],
  outputs: ["y"],
  initializers: { w: {} },
  values: {
    w: { dtype: "f32", shape: [2, 16] },
    y: { dtype: "f32", shape: [2, 16] },
  },
  nodes: [{ op: "add", ins: ["x", "w"], outs: ["y"], attrs: {} }],
});

const CONSUMERS = [
  { op: "linear", graph: linearGraph },
  { op: "embedding", graph: embeddingGraph },
  { op: "conv1d (groups == 1)", graph: conv1dGraph },
  { op: "add (outside a weight slot)", graph: addGraph },
] as const;

describe("every existing consumer refuses int4-sym-g-f16 (ADR 0145 decision 3)", () => {
  for (const { op, graph } of CONSUMERS) {
    it(`the residency plan refuses it for ${op}, before any residency class`, () => {
      const merged = mergeGraph(graph(), { w: { codec: "int4-sym-g-f16", groupSize: 16 } });
      assertThrows(
        () => planWeightResidency(merged),
        ExecutionError,
        `initializer 'w': ${REFUSAL}`,
      );
    });

    it(`the same weight as int4-sym-g is still planned for ${op} (control)`, () => {
      const merged = mergeGraph(graph(), { w: { codec: "int4-sym-g", groupSize: 16 } });
      assert(planWeightResidency(merged).get("w") !== undefined);
    });
  }

  it("a Session and the estimate refuse it at the capability check, before any weight is read", () => {
    // linear's `w [3, 16]`: 24 packed bytes and 3 f16 steps (8 bytes once padded).
    assertThrows(
      () =>
        openGraph(linearGraph(), [
          weight("w", zeros(24), {
            codec: "int4-sym-g-f16",
            groupSize: 16,
            scale: { bytes: zeros(8), dtype: "f16" },
          }),
          weight("b", zeros(12)),
        ]),
      RuntimeSupportError,
      "unsupported storage 'i4f16' (1): w",
    );
  });
});

// ---------------------------------------------------------------------------
// CPU decode
// ---------------------------------------------------------------------------

describe("decodeI4F16Steps", () => {
  it("equals decodeI4 with the f16 steps widened to f32 (normal, subnormal and the largest f16)", () => {
    // f16 bits 0x3c00 = 1, 0x0001 = 2^-24 (the smallest subnormal), 0x7bff = 65504 (the largest
    // finite), little endian, padded with one zero step to a whole word.
    const steps = Uint8Array.from([0x00, 0x3c, 0x01, 0x00, 0xff, 0x7b, 0x00, 0x00]);
    // [3, 16] with group 16; codes cover u = 0..15 in an order where neighbours differ.
    const packed = Uint8Array.from(
      { length: 24 },
      (_, i) => ((i * 7) & 0x0f) | (((i * 5 + 3) & 0x0f) << 4),
    );
    const decoded = decodeI4F16Steps(packed, [3, 16], steps, [3, 1], 16);
    assertEquals(
      [...decoded],
      [...decodeI4(packed, [3, 16], Float32Array.from([1, 2 ** -24, 65504]), [3, 1], 16)],
    );
  });

  it("reads step 2i from the low half and step 2i+1 from the high half of a u32 word", () => {
    // One word: low half 0x3c00 = 1 (row 0's step), high half 0x4000 = 2 (row 1's step). Every
    // code is q = 1 (u = 9), so a swapped half shows as row 0 decoding to 2.
    const steps = Uint8Array.from([0x00, 0x3c, 0x00, 0x40]);
    const packed = new Uint8Array(16).fill(0x99);
    const decoded = decodeI4F16Steps(packed, [2, 16], steps, [2, 1], 16);
    assertEquals([...decoded], [...new Array(16).fill(1), ...new Array(16).fill(2)]);
  });

  it("refuses a step table that is not padded to whole words", () => {
    // 3 steps take 8 bytes (6 + 2 of padding), not 6.
    assertThrows(
      () => decodeI4F16Steps(new Uint8Array(24), [3, 16], new Uint8Array(6), [3, 1], 16),
      I4Error,
      "f16 step table of 6 bytes does not match scale [3,1]",
    );
  });
});
