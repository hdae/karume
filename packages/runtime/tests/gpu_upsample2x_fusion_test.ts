// The strict peephole for the VAE's nearest-exact x2 (fusion rule upsample2x —
// src/runtime/fusion.ts). Pins that replacing the exporter's sequence of two expands with a private
// 1-pass copy still duplicates every bit unchanged.
//
// The twin case lives in @karume/kernel-checks (checkUpsample2xTwins), which the playground's
// /kernel-check page runs on the browser's backend too; the comparisons with the host-built
// expectation, the census and the fallback cases stay here.

import { test } from "vite-plus/test";
import { assertEquals } from "@std/assert";
import {
  checkUpsample2xTwins,
  UPSAMPLE2X_SHAPE,
  UPSAMPLE2X_TWIN_CASES,
  upsample2xGraph,
  upsample2xInput,
} from "@karume/kernel-checks";
import { acquireGpu } from "../src/gpu/device.ts";
import { UPSAMPLE_2X_KEY } from "../src/kernels/upsample2x.ts";
import { createSessionFromContainer, type Tensor } from "../src/runtime/executor.ts";
import { fill, GRAPH_NAME, openModelBytes } from "./helpers/model-fixture.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { assertCheckPart, assertWordIdentity } from "./helpers/kernel-check-assert.ts";
import { requireCensus } from "./helpers/pipeline-census.ts";

const SHAPE = UPSAMPLE2X_SHAPE;
const WIDE = [30, 7, 2] as const;
const OUT = [2, 3, 10, 14] as const;

const words = (tensor: Tensor): Uint32Array<ArrayBuffer> =>
  new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length);

const expectedUpsampleWords = (input: Uint32Array): Uint32Array<ArrayBuffer> => {
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
  "the VAE nearest x2 fusion matches the real exported sequence bit for bit (finite) / by NaN class and folds 2 → 1 dispatch (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    try {
      const part = await checkUpsample2xTwins(gpu);
      assertEquals([part.id, part.cases.length], Object.entries(UPSAMPLE2X_TWIN_CASES)[0]);
      assertCheckPart(part);
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "the VAE nearest x2 fusion writes every element the real exported sequence writes, with only the copy kernel (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const x = upsample2xInput();
    const fused = await createSessionFromContainer(
      gpu,
      await openModelBytes(upsample2xGraph(false), []),
      GRAPH_NAME,
    );
    const primitive = await createSessionFromContainer(
      gpu,
      await openModelBytes(upsample2xGraph(true), []),
      GRAPH_NAME,
    );
    try {
      const fusedOut = (await fused.run({ x })).y;
      const primitiveOut = (await primitive.run({ x })).y;
      const expected = expectedUpsampleWords(words(x));
      // The u32 copy kernel itself preserves every bit, NaN payloads included.
      assertWordIdentity(words(fusedOut), expected, "payload", "fused full write");
      // Against the existing f32 primitives, only NaNs (which a backend may canonicalize) are
      // compared by classification.
      assertWordIdentity(words(primitiveOut), expected, "class", "primitive full write");
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
      const outputGraph = upsample2xGraph(false);
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

      const consumerGraph = upsample2xGraph(false);
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

      const nearGraph = upsample2xGraph(false, 3);
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
