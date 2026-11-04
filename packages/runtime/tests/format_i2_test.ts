import { test } from "vite-plus/test";
import { assertEquals, assertThrows } from "@std/assert";
import { ContainerFormatError } from "../src/format/container/header.ts";
import { decodeI2, I2Error } from "../src/format/i2.ts";
import { prepareContainer } from "../src/runtime/executor.ts";
import { planWeightBuffers, planWeightResidency } from "../src/runtime/weight-residency.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import {
  type DeclarationJson,
  GRAPH_NAME,
  memoryModel,
  openModelBytes,
} from "./helpers/model-fixture.ts";

const graph = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["embedding"] },
  symbols: [],
  inputs: [{ name: "ids", dtype: "i32", shape: [2] }],
  outputs: ["y"],
  initializers: { w: {} },
  values: { w: { dtype: "f32", shape: [2, 16] }, y: { dtype: "f32", shape: [2, 16] } },
  nodes: [{ op: "embedding", ins: ["w", "ids"], outs: ["y"], attrs: { padding_idx: -1 } }],
});
const bytes = Uint8Array.of(0xe4, 0x1b, 0x72, 0x8d, 0x1b, 0xe4, 0x8d, 0x72);
const scales = Float32Array.of(0.375, 1.25);

/**
 * One `int2-off` supply. It is per-channel, so `groupSize` is the row length (16) and there is one
 * scale per row (`[rows, 1]` worth of f32 — the same bytes as the v1 keepdim shape).
 */
const i2Weight = (
  scaleBytes: Uint8Array<ArrayBuffer> = new Uint8Array(scales.buffer),
  groupSize = 16,
): TensorInput => ({
  graph: GRAPH_NAME,
  initializer: "w",
  bytes,
  encoding: { codec: "int2-off", groupSize, scale: { bytes: scaleBytes, dtype: "f32" } },
});

test("INT2 decodes every code value, every in-byte position and per-row scales", async () => {
  const expected = [
    -2, -1, 0, 1, 1, 0, -1, -2, 0, -2, 1, -1, -1, 1, -2, 0, 1, 0, -1, -2, -2, -1, 0, 1, -1, 1, -2,
    0, 0, -2, 1, -1,
  ].map((q, i) => Math.fround(q * scales[Math.floor(i / 16)]));
  assertEquals(decodeI2(bytes, [2, 16], scales, [2, 1]), new Float32Array(expected));
  // Round-trip through the container (`krm`) once and check that the reader side also gets the i2
  // `seat`.
  const loaded = prepareContainer(await openModelBytes(graph(), [i2Weight()]), GRAPH_NAME);
  const buffers = planWeightBuffers(planWeightResidency(loaded.graph));
  assertEquals(
    buffers.map((b) => b.declaredBytes),
    [8, 8],
  );
  assertEquals(
    buffers.map((b) => b.seat),
    ["i2", "i2"],
  );
});

test("INT2 rejects invalid shapes, scales and partial bytes", () => {
  // The declared-shape check lives in the **merge layer** (declared shape × codec) — an IR v2
  // declaration carries no storage, so "i2 but rank 1 / row length not a multiple of 16" is decided
  // only when it is matched against the supply.
  for (const shape of [[32], [2, 15], [2, 20], [0, 16], [2, 0], [1, 2, 16]]) {
    const g = graph();
    g.values.w.shape = shape;
    assertThrows(
      () => memoryModel(g, [i2Weight()]),
      ContainerFormatError,
      "positive rank-2 declared shape with a row length that is a multiple of 16",
    );
    assertThrows(() => decodeI2(bytes, shape, scales, [2, 1]), I2Error);
  }
  // A per-channel codec's `groupSize` MUST equal the row length (where v1's `group_size`
  // declaration went).
  assertThrows(
    () => memoryModel(graph(), [i2Weight(new Uint8Array(scales.buffer), 8)]),
    ContainerFormatError,
    "per-channel",
  );
  // An `int2-off` without a scale fails at supply time (where v1's missing `storage.scale` went).
  assertThrows(
    () =>
      memoryModel(graph(), [
        {
          graph: GRAPH_NAME,
          initializer: "w",
          bytes,
          encoding: { codec: "int2-off" },
        },
      ]),
    ContainerFormatError,
    "needs scale and groupSize",
  );
  // The scale byte count is `rows × groups × 4` — a list that does not match the row count fails
  // (where v1's scale shape check went). The message names the byte-count match itself (with only
  // `"scale"` the test would stay green even if the check regressed to "is there a scale" — the
  // messages of the two cases above also contain `scale`).
  assertThrows(
    () => memoryModel(graph(), [i2Weight(new Uint8Array(Float32Array.of(0.375).buffer))]),
    ContainerFormatError,
    "length 4 is not payload 8 bytes",
  );
  assertThrows(() => decodeI2(bytes.subarray(1), [2, 16], scales, [2, 1]), I2Error, "payload");
  assertThrows(() => decodeI2(bytes, [2, 16], scales, [1, 2]), I2Error, "scale");
});

test("INT2: routing the same weight to a graph output estimates the f32 expansion", () => {
  const g = graph();
  g.outputs.push("w");
  const plan = planWeightResidency(
    prepareContainer(memoryModel(g, [i2Weight()]), GRAPH_NAME).graph,
  );
  assertEquals(plan.get("w"), { seat: "expanded", payloadBytes: 8, expandedBytes: 128 });
});
