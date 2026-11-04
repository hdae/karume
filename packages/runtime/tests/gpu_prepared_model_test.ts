// Real-GPU check of the two-phase boundary (ADR 0070 decision 5 / graph-first) — `prepareContainer(opened, graph) →
// estimate() → createContainerSession(gpu)` must build **the same Session** as the combined
// createSessionFromContainer.
//
// Two detectors: bit-identical output of a run that goes through every weight, and matching storage
// diagnostics. The fixture mixes 4 codecs (f32 / f16 / int8-sym / int4-sym-g) — so that dropping any one
// codec on the two-phase path does not pass silently.

import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import type { BoundContainer } from "../src/format/container/bind.ts";
import { openMemoryContainer } from "../src/format/container/memory.ts";
import { parseIrDeclarationValue } from "../src/format/ir.ts";
import { acquireGpu } from "../src/gpu/device.ts";
import { RuntimeSupportError } from "../src/ops/support.ts";
import {
  createSessionFromContainer,
  prepareContainer,
  type Tensor,
} from "../src/runtime/executor.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import { buildFixture, countingContainer } from "./helpers/mixed-codec-fixture.ts";
import {
  baseDeclaration,
  baseTensors,
  f32Bytes,
  GRAPH_NAME,
  openModelBytes,
} from "./helpers/model-fixture.ts";

const bitsOf = (tensor: Tensor): readonly number[] => [
  ...new Uint32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length),
];

test.skipIf(!GPU_AVAILABLE)(
  "the two-phase boundary matches createSessionFromContainer in bit-identical output and storage diagnostics (real GPU)",
  async () => {
    const fixture = buildFixture();
    const gpu = await acquireGpu();
    try {
      // Phase 1: admission from the declarations of the opened container only (not a single weight block fetched)
      const counted = countingContainer(await openModelBytes(fixture.declaration, fixture.tensors));
      const prepared = prepareContainer(counted.container, GRAPH_NAME);
      const estimate = prepared.estimate();
      assertEquals(counted.reads(), 0, "admission fetched a weight block");
      // Phase 2: fetch the blocks and build
      const twoPhase = await prepared.createContainerSession(gpu);
      const composed = await createSessionFromContainer(
        gpu,
        await openModelBytes(fixture.declaration, fixture.tensors),
        GRAPH_NAME,
      );
      try {
        const twoPhaseOut = (await twoPhase.run({ x: fixture.x }))["y"];
        const composedOut = (await composed.run({ x: fixture.x }))["y"];
        assertEquals(bitsOf(twoPhaseOut), bitsOf(composedOut), "the outputs are not bit-identical");
        assertEquals(twoPhase.diagnostics().storage, composed.diagnostics().storage);
        const storage = twoPhase.diagnostics().storage;
        assert(storage.residentCompressedBytes > 0, "no compressed resident weight at all");
        // The number given before a single byte of weights was fetched matches the measured compressed
        // residency exactly (the mapping table of ADR 0070 decision 5 — this is where the two-phase boundary pays off)
        assertEquals(estimate.resident.weights.compressedBytes, storage.residentCompressedBytes);
      } finally {
        await twoPhase.dispose();
        await composed.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "admission fails before a single weight block is fetched (real GPU)",
  async () => {
    // A semantic dtype that cannot run even on a supported op (matmul weights in i32 — missing capability).
    // NOTE: it cannot be made on the storage-codec side — every codec of the ledger can run (bf16 was opened by ADR 0131).
    const unsupportedDtype = baseDeclaration();
    unsupportedDtype.values["w"] = { dtype: "i32", shape: [4, 3] };
    const counted = countingContainer(
      await openModelBytes(unsupportedDtype, [
        {
          graph: GRAPH_NAME,
          initializer: "w",
          bytes: new Uint8Array(new ArrayBuffer(48)),
          encoding: { codec: "i32" },
        },
        baseTensors()[1],
      ]),
    );
    const gpu = await acquireGpu();
    try {
      await assertRejects(
        () => createSessionFromContainer(gpu, counted.container, GRAPH_NAME),
        RuntimeSupportError,
        "i32",
      );
      assertEquals(counted.reads(), 0, "fetched a weight block for a model that cannot run");
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * An in-memory container that supplies the weight `w` (`[4,3]` f32) as 2 pieces and fails **only the read of
 * the second one**.
 *
 * MUST: the failure has to happen "in the middle of construction". Failing at admission (capability / op
 * contract) returns before a single GPU resource is created, so the claim "no half-built resources remain"
 * would test nothing (the check at `:70` covers that phase separately). Piece 1 returns real bytes and lets
 * construction go as far as `queue.writeBuffer`; only the `read()` of piece 2 fails.
 */
const halfUploadedContainer = (): { readonly container: BoundContainer; reads(): number } => {
  let reads = 0;
  const half = f32Bytes(new Array(6).fill(0.5));
  return {
    container: openMemoryContainer({
      graphs: { [GRAPH_NAME]: parseIrDeclarationValue(baseDeclaration()) },
      tensors: {
        [GRAPH_NAME]: {
          w: {
            encoding: { codec: "f32" },
            pieces: [
              {
                rows: [0, 2],
                read: () => {
                  reads += 1;
                  return Promise.resolve(half);
                },
              },
              {
                rows: [2, 4],
                read: () => {
                  reads += 1;
                  return Promise.reject(new Error("the supply of piece 2 failed"));
                },
              },
            ],
          },
          b: { encoding: { codec: "f32" }, bytes: f32Bytes([1, 2, 3]) },
        },
      },
    }),
    reads: () => reads,
  };
};

test.skipIf(!GPU_AVAILABLE)(
  "a failure in the middle of construction can be rebuilt on the same device (no partial Session remains; real GPU)",
  async () => {
    // When Session construction fails partway, it must leave no half-built resources or broken residency on the GPU.
    // An invariant that must hold with a container as the source too; separate from the other 2, which look only at the admission phase.
    const broken = halfUploadedContainer();
    const healthy = await openModelBytes();
    const gpu = await acquireGpu();
    try {
      await assertRejects(
        () => createSessionFromContainer(gpu, broken.container, GRAPH_NAME),
        Error,
        "the supply of piece 2 failed",
      );
      // Not vacuous — the first piece was actually uploaded, and the failure came after the transfer had started.
      assertEquals(
        broken.reads(),
        2,
        "the piece reads did not reach 2 (the failure came before the transfer started)",
      );
      // Rebuild from a healthy container on the same device and run to the end (x[1,4]·w[4,3] + b).
      const session = await createSessionFromContainer(gpu, healthy, GRAPH_NAME);
      try {
        const out = (
          await session.run({
            x: { dtype: "f32", shape: [1, 4], data: Float32Array.from([1, 1, 1, 1]) },
          })
        )["y"];
        // Every element of w is 0.5, so each row sum is 2; b is [1,2,3].
        assertEquals([...out.data], [3, 4, 5]);
      } finally {
        await session.dispose();
      }
    } finally {
      gpu.destroy();
    }
  },
);
