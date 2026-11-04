// The strict peephole for the VAE's nearest-exact x2 (fusion rule upsample2x —
// src/runtime/fusion.ts). Pins that replacing the exporter's sequence of two expands with a private
// 1-pass copy still duplicates every bit unchanged.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import { acquireGpu } from "../src/gpu/device.ts";
import { UPSAMPLE_2X_KEY } from "../src/kernels/upsample2x.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { type DeclarationJson, fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

// B>1 / C>1, odd H/W, and 2*3*5*7=210 is not a multiple of the workgroup width 256.
const SHAPE = [2, 3, 5, 7] as const;
const FIRST_RESHAPE = [30, 7, 1] as const;
const WIDE = [30, 7, 2] as const;
const OUT = [2, 3, 10, 14] as const;

/** Builds the real exporter's reshape→expand→reshape→reshape→expand→reshape verbatim. */
const upsampleGraph = (interpose: boolean, scaleWidth = 2): DeclarationJson => {
  const wide = [FIRST_RESHAPE[0], FIRST_RESHAPE[1], scaleWidth];
  const wideReshape = [6, 5, 7 * scaleWidth];
  const tallReshape = [6, 5, 1, 7 * scaleWidth];
  const tall = [6, 5, 2, 7 * scaleWidth];
  const out = [2, 3, 10, 7 * scaleWidth];
  const values: DeclarationJson["values"] = {
    first_reshape: { dtype: "f32", shape: [...FIRST_RESHAPE] },
    wide: { dtype: "f32", shape: wide },
    wide_reshape: { dtype: "f32", shape: wideReshape },
    tall_reshape: { dtype: "f32", shape: tallReshape },
    tall: { dtype: "f32", shape: tall },
    y: { dtype: "f32", shape: out },
  };
  if (interpose) values.first_alias = { dtype: "f32", shape: [...FIRST_RESHAPE] };
  const nodes: DeclarationJson["nodes"] = [
    { op: "reshape", ins: ["x"], outs: ["first_reshape"], attrs: {} },
  ];
  if (interpose) {
    // Inserts only a same-shape alias: neither the values nor the physical dispatch count change,
    // only the exact matcher stops matching.
    nodes.push({ op: "reshape", ins: ["first_reshape"], outs: ["first_alias"], attrs: {} });
  }
  nodes.push(
    { op: "expand", ins: [interpose ? "first_alias" : "first_reshape"], outs: ["wide"], attrs: {} },
    { op: "reshape", ins: ["wide"], outs: ["wide_reshape"], attrs: {} },
    { op: "reshape", ins: ["wide_reshape"], outs: ["tall_reshape"], attrs: {} },
    { op: "expand", ins: ["tall_reshape"], outs: ["tall"], attrs: {} },
    { op: "reshape", ins: ["tall"], outs: ["y"], attrs: {} },
  );
  return {
    format: "karume-ir",
    version: 2,
    requires: { ops: ["reshape", "expand"] },
    symbols: [],
    inputs: [{ name: "x", dtype: "f32", shape: [...SHAPE] }],
    outputs: ["y"],
    initializers: {},
    values,
    nodes,
  };
};

const words = (tensor: Tensor): Uint32Array =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

const isNanWord = (word: number): boolean => (word & 0x7fffffff) > 0x7f800000;

/** The u32 copy kernel itself preserves every bit, NaN payloads included. */
const assertExactWords = (actual: Uint32Array, expected: Uint32Array, label: string): void => {
  assertEquals(actual.length, expected.length, label + ": length");
  for (let i = 0; i < actual.length; i++) {
    assertEquals(actual[i], expected[i], `${label}: word index=${i}`);
  }
};

/**
 * Against the existing f32 primitives, only NaNs (which a backend may canonicalize) are compared by
 * classification.
 */
const assertSemanticWords = (actual: Uint32Array, expected: Uint32Array, label: string): void => {
  assertEquals(actual.length, expected.length, label + ": length");
  for (let i = 0; i < actual.length; i++) {
    if (isNanWord(expected[i])) {
      assertEquals(isNanWord(actual[i]), true, `${label}: NaN classification index=${i}`);
    } else {
      assertEquals(actual[i], expected[i], `${label}: word index=${i}`);
    }
  }
};

const expectedUpsampleWords = (input: Uint32Array): Uint32Array => {
  const out = new Uint32Array(OUT.reduce((total, dim) => total * dim, 1));
  for (let b = 0; b < SHAPE[0]; b++) {
    for (let c = 0; c < SHAPE[1]; c++) {
      for (let h = 0; h < SHAPE[2]; h++) {
        for (let w = 0; w < SHAPE[3]; w++) {
          const source = ((b * SHAPE[1] + c) * SHAPE[2] + h) * SHAPE[3] + w;
          for (let dh = 0; dh < 2; dh++) {
            for (let dw = 0; dw < 2; dw++) {
              const target = ((b * OUT[1] + c) * OUT[2] + h * 2 + dh) * OUT[3] + w * 2 + dw;
              out[target] = input[source];
            }
          }
        }
      }
    }
  }
  return out;
};

test.skipIf(!GPU_AVAILABLE)(
  "the VAE nearest x2 fusion writes every element the real exported sequence writes and folds 2 → 1 dispatch (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const x = fill(SHAPE, (i) => Math.sin(i * 0.37) * 3.1 + i / 997);
    // The copy fusion does no arithmetic, so all of the following bit patterns must be preserved
    // too.
    new Uint32Array(x.data.buffer).set([
      0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x7f800000,
      0xff800000, 0x7fc01234, 0xff801234,
    ]);
    const fused = await createSessionFromContainer(
      gpu,
      await openModelBytes(upsampleGraph(false), []),
      GRAPH_NAME,
    );
    const primitive = await createSessionFromContainer(
      gpu,
      await openModelBytes(upsampleGraph(true), []),
      GRAPH_NAME,
    );
    try {
      const fusedOut = (await fused.run({ x })).y;
      const primitiveOut = (await primitive.run({ x })).y;
      const expected = expectedUpsampleWords(words(x));
      assertExactWords(words(fusedOut), expected, "fused full write");
      assertSemanticWords(words(primitiveOut), expected, "primitive full write");
      assertSemanticWords(words(fusedOut), words(primitiveOut), "fused vs primitive");
      assertEquals(fused.diagnostics().submit.dispatchCount, 1, "fused copy 1 pass");
      assertEquals(primitive.diagnostics().submit.dispatchCount, 2, "expand x2 primitive");
      assertEquals(fused.diagnostics().lastRunFusions?.upsample2x, 1, "fusion counter");
      assertEquals(
        primitive.diagnostics().lastRunFusions?.upsample2x,
        0,
        "the counter of the counterexample is 0",
      );
      assertEquals(
        requireCensus(fused.diagnostics().lastRunPipelines, "upsample2x").map((row) => row.key),
        [UPSAMPLE_2X_KEY],
      );
    } finally {
      await fused.dispose();
      await primitive.dispose();
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "upsample x2 does not fuse when an internal value is an output or has another consumer, nor for a near shape (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const x = fill(SHAPE, (i) => (i % 13) - 6.25);
    try {
      const outputGraph = upsampleGraph(false);
      outputGraph.outputs = ["wide", "y"];
      const outputSession = await createSessionFromContainer(
        gpu,
        await openModelBytes(outputGraph, []),
        GRAPH_NAME,
      );
      try {
        await outputSession.run({ x });
        assertEquals(
          outputSession.diagnostics().submit.dispatchCount,
          2,
          "internal output fallback",
        );
        assertEquals(outputSession.diagnostics().lastRunFusions?.upsample2x, 0, "fusion counter 0");
      } finally {
        await outputSession.dispose();
      }

      const consumerGraph = upsampleGraph(false);
      consumerGraph.values.wide_copy = { dtype: "f32", shape: [...WIDE] };
      consumerGraph.nodes.push({ op: "neg", ins: ["wide"], outs: ["wide_copy"], attrs: {} });
      consumerGraph.requires.ops.push("neg");
      consumerGraph.outputs = ["wide_copy", "y"];
      const consumerSession = await createSessionFromContainer(
        gpu,
        await openModelBytes(consumerGraph, []),
        GRAPH_NAME,
      );
      try {
        await consumerSession.run({ x });
        assertEquals(
          consumerSession.diagnostics().submit.dispatchCount,
          3,
          "extra consumer fallback",
        );
        assertEquals(
          consumerSession.diagnostics().lastRunFusions?.upsample2x,
          0,
          "fusion counter 0",
        );
      } finally {
        await consumerSession.dispose();
      }

      const nearGraph = upsampleGraph(false, 3);
      const nearSession = await createSessionFromContainer(
        gpu,
        await openModelBytes(nearGraph, []),
        GRAPH_NAME,
      );
      try {
        const output = (await nearSession.run({ x })).y;
        assertEquals(
          nearSession.diagnostics().submit.dispatchCount,
          2,
          "anything but x2 falls back",
        );
        assertEquals(nearSession.diagnostics().lastRunFusions?.upsample2x, 0, "fusion counter 0");
        assertEquals(output.shape, [2, 3, 10, 21]);
      } finally {
        await nearSession.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);
