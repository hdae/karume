/**
 * The `ternary` codec (container-v1 §6.3, ADR 0108 decision 13) — a codec that declares a subset of
 * the `int2-off` value range under another name.
 *
 * Three things are pinned:
 * - The packing and the decode formula are identical to i2, so the decode path **shares i2 as is**
 *   (acceptance and layout).
 * - The checkable content of the claim "it is ternary" = every 2-bit code of the payload is in
 *   `{1, 2, 3}`. The check is a mask over the 4 lanes of a byte and rejects code 0 in any lane
 *   (missing one lane lets the element at that position pass silently as `q = −2`, a value outside
 *   the ternary set).
 * - That check actually takes effect at the entry of Session construction (the items of
 *   `containerBatches`).
 *
 * No real GPU is used (the check runs on the host right after the block is read).
 */

import { assertEquals, assertRejects, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { assertTernaryCodes } from "../src/format/container/bind.ts";
import { codecLayout } from "../src/format/container/codecs.ts";
import { ContainerFormatError } from "../src/format/container/header.ts";
import { containerBatches } from "../src/runtime/executor.ts";
import type { TensorInput } from "./helpers/container-write.ts";
import {
  type DeclarationJson,
  f32Bytes,
  GRAPH_NAME,
  memoryModel,
  openModelBytes,
} from "./helpers/model-fixture.ts";

const ROWS = 4;
const ROW_LENGTH = 16;

/**
 * y = linear(x, w, b) — `w` is [4,16] (the declared shape of the i2 path: positive rank 2, row
 * length a multiple of 16).
 */
const declaration = (): DeclarationJson => ({
  format: "karume-ir",
  version: 2,
  requires: { ops: ["linear"] },
  symbols: [],
  inputs: [{ name: "x", dtype: "f32", shape: [2, ROW_LENGTH] }],
  outputs: ["y"],
  initializers: { w: {}, b: {} },
  values: {
    w: { dtype: "f32", shape: [ROWS, ROW_LENGTH] },
    b: { dtype: "f32", shape: [ROWS] },
    y: { dtype: "f32", shape: [2, ROWS] },
  },
  nodes: [{ op: "linear", ins: ["x", "w", "b"], outs: ["y"], attrs: {} }],
});

/** Supplies `w` as ternary (per-channel, f32 scale) with `payload` as is (16 B = 64 elements). */
const tensors = (payload: Uint8Array<ArrayBuffer>): readonly TensorInput[] => [
  {
    graph: GRAPH_NAME,
    initializer: "w",
    bytes: payload,
    encoding: {
      codec: "ternary",
      groupSize: ROW_LENGTH,
      scale: { bytes: f32Bytes(new Array(ROWS).fill(0.5)), dtype: "f32" },
    },
  },
  {
    graph: GRAPH_NAME,
    initializer: "b",
    bytes: f32Bytes(new Array(ROWS).fill(0)),
    encoding: { codec: "f32" },
  },
];

/** A payload whose codes are all 1 / 2 / 3 (0x55 = code 1 × 4, 0xFF = 3 × 4, 0xAA = 2 × 4). */
const validPayload = (): Uint8Array<ArrayBuffer> =>
  Uint8Array.from({ length: (ROWS * ROW_LENGTH) / 4 }, (_, i) => [0x55, 0xff, 0xaa][i % 3]);

describe("ternary: sharing the i2 path", () => {
  it("a ternary declaration in krm is accepted and decodes through the i2 path", async () => {
    const opened = await openModelBytes(declaration(), tensors(validPayload()));
    assertEquals(opened.graphs[GRAPH_NAME].supplies.get("w")?.encoding.codec, "ternary");
    assertEquals(codecLayout("ternary"), "i2");
  });
});

describe("assertTernaryCodes: every code of the payload is in {1, 2, 3}", () => {
  it("passes a payload without code 0", () => {
    assertTernaryCodes(Uint8Array.of(0x55, 0xff, 0xaa), "w");
  });

  it("rejects a byte containing code 0, naming its position", () => {
    // 0x54 = 01 01 01 00 — only the lowest lane is code 0.
    assertThrows(
      () => assertTernaryCodes(Uint8Array.of(0x55, 0x54), "w"),
      ContainerFormatError,
      "(byte 1)",
    );
  });

  it("rejects code 0 at any of the 4 lane positions", () => {
    // 4 variants with only one lane set to 0 (the other 3 lanes are code 1).
    for (const byte of [0x54, 0x51, 0x45, 0x15]) {
      assertThrows(
        () => assertTernaryCodes(Uint8Array.of(byte), "w"),
        ContainerFormatError,
        "has code 0",
        `0x${byte.toString(16)}`,
      );
    }
  });
});

describe("ternary: the entry of Session construction (containerBatches)", () => {
  it("a supply containing code 0 is rejected while iterating items", async () => {
    const payload = validPayload();
    payload[9] = 0xa8; // 10 10 10 00 — only the lowest lane is code 0.
    const opened = memoryModel(declaration(), tensors(payload));
    await assertRejects(
      async () => {
        for await (const batch of containerBatches(opened, GRAPH_NAME)) {
          for await (const _item of batch.items) {
            /* each pull reads the block and checks it */
          }
        }
      },
      ContainerFormatError,
      "(byte 9)",
    );
  });

  it("a supply without code 0 lets items be pulled to the end", async () => {
    const opened = memoryModel(declaration(), tensors(validPayload()));
    const names: string[] = [];
    for await (const batch of containerBatches(opened, GRAPH_NAME)) {
      for await (const item of batch.items) names.push(item.name);
    }
    assertEquals(names.sort(), ["b", "w"]);
  });
});
