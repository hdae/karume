// Behaviour of the pure functions that build umT5's graph inputs and outputs
// (src/umt5/session-io.ts — ADR 0119 decisions 3 and 4). Needs neither a GPU nor real assets. The
// graph input names are compared with the spelling the recipe (`INPUT_NAMES` in
// `wan/umt5_patch.py`) wrote into the fixture `wan-umt5/relative-position.json`.

import { readTextFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import type { Tensor } from "@karume/runtime";
import { buildUmt5RelativePositionBuckets, type I32Tensor } from "../src/umt5/relative-position.ts";
import {
  padUmt5Context,
  UMT5_INPUT_IDS,
  UMT5_RELATIVE_POSITION_BUCKETS,
  umt5HiddenStates,
  umt5SessionInputs,
} from "../src/umt5/session-io.ts";

const meta = JSON.parse(
  await readTextFile(new URL("./fixtures/wan-umt5/relative-position.json", import.meta.url)),
) as { readonly input_names: readonly string[] };

const idsOf = (...values: number[]): Int32Array<ArrayBuffer> => Int32Array.from(values);

test("umT5's graph input names match the recipe's declaration (the export's input order)", () => {
  assertEquals([UMT5_INPUT_IDS, UMT5_RELATIVE_POSITION_BUCKETS], meta.input_names);
});

test("umt5SessionInputs: makes the id sequence [L] into [1, L] and passes the bucket table as is (no copy)", () => {
  const ids = idsOf(289, 748, 1);
  const buckets = buildUmt5RelativePositionBuckets(ids.length);

  const inputs = umt5SessionInputs(ids, buckets);

  assertEquals(Object.keys(inputs).sort(), [...meta.input_names].sort());
  assertEquals(inputs[UMT5_INPUT_IDS].dtype, "i32");
  assertEquals(inputs[UMT5_INPUT_IDS].shape, [1, 3]);
  assertStrictEquals(inputs[UMT5_INPUT_IDS].data, ids);
  assertStrictEquals(inputs[UMT5_RELATIVE_POSITION_BUCKETS], buckets);
});

test("umt5SessionInputs: fails when the table length differs from the id sequence (L is one value across both inputs)", () => {
  const ids = idsOf(289, 748, 1);
  for (const length of [2, 4]) {
    assertThrows(
      () => umt5SessionInputs(ids, buildUmt5RelativePositionBuckets(length)),
      Error,
      "is not",
      `table length ${length}`,
    );
  }
  const flat: I32Tensor = { dtype: "i32", shape: [9], data: new Int32Array(9) };
  assertThrows(() => umt5SessionInputs(ids, flat), Error, "is not");
  const short: I32Tensor = { dtype: "i32", shape: [3, 3], data: new Int32Array(8) };
  assertThrows(() => umt5SessionInputs(ids, short), Error, "element count");
});

test("umt5SessionInputs: fails when the id sequence length is outside 2 to 512", () => {
  const buckets = buildUmt5RelativePositionBuckets(2);
  assertThrows(() => umt5SessionInputs(idsOf(1), buckets), Error, "outside");
  assertThrows(() => umt5SessionInputs(new Int32Array(513), buckets), Error, "outside");
});

/**
 * An output of `[1, tokens, width]` (values = serial numbers from 1 — a position mix-up shows in
 * the values).
 */
const outputOf = (tokens: number, width: number): Tensor => ({
  dtype: "f32",
  shape: [1, tokens, width],
  data: Float32Array.from({ length: tokens * width }, (_, index) => index + 1),
});

test("umt5HiddenStates: returns the contents of [1, tokens, width] as [tokens, width] without copying", () => {
  const output = outputOf(3, 4);

  const states = umt5HiddenStates(output, 3, 4);

  assertStrictEquals(states, output.data);
});

test("umt5HiddenStates: rejects mismatched shape, dtype and element count (no padding, so no row limit)", () => {
  const output = outputOf(3, 4);
  // The output row count differs from the length passed as input (an output mixed up with another
  // run's, for example).
  assertThrows(() => umt5HiddenStates(output, 2, 4), Error, "is not");
  // Differs from the width of the graph declaration.
  assertThrows(() => umt5HiddenStates(output, 3, 8), Error, "is not");
  const integer: Tensor = { dtype: "i32", shape: [1, 3, 4], data: new Int32Array(12) };
  assertThrows(() => umt5HiddenStates(integer, 3, 4), Error, "f32");
  const short: Tensor = { dtype: "f32", shape: [1, 3, 4], data: new Float32Array(11) };
  assertThrows(() => umt5HiddenStates(short, 3, 4), Error, "element count");
});

test("padUmt5Context: rows within the valid length stay as is, and the rows after them are zero", () => {
  const tokens = 3;
  const width = 4;
  const rows = 6;

  const padded = padUmt5Context(outputOf(tokens, width), tokens, rows, width);

  assertEquals(padded.length, rows * width);
  assertEquals(
    [...padded.subarray(0, tokens * width)],
    Array.from({ length: tokens * width }, (_, index) => index + 1),
  );
  assertEquals([...padded.subarray(tokens * width)], new Array((rows - tokens) * width).fill(0));
});

test("padUmt5Context: when the valid length equals the context's row count, every row without padding", () => {
  const padded = padUmt5Context(outputOf(5, 2), 5, 5, 2);

  assertEquals(
    [...padded],
    Array.from({ length: 10 }, (_, index) => index + 1),
  );
});

test("padUmt5Context: rejects mismatched shape, dtype and row count", () => {
  const output = outputOf(3, 4);
  // The output row count differs from the length passed as input (positive / negative mixed up, for
  // example).
  assertThrows(() => padUmt5Context(output, 2, 6, 4), Error, "is not");
  // Differs from the DiT's context width.
  assertThrows(() => padUmt5Context(output, 3, 6, 8), Error, "is not");
  // Exceeds the context's row count.
  assertThrows(() => padUmt5Context(output, 3, 2, 4), Error, "exceeds");
  const integer: Tensor = { dtype: "i32", shape: [1, 3, 4], data: new Int32Array(12) };
  assertThrows(() => padUmt5Context(integer, 3, 6, 4), Error, "f32");
  const rank2: Tensor = { dtype: "f32", shape: [3, 4], data: new Float32Array(12) };
  assertThrows(() => padUmt5Context(rank2, 3, 6, 4), Error, "is not");
  const short: Tensor = { dtype: "f32", shape: [1, 3, 4], data: new Float32Array(11) };
  assertThrows(() => padUmt5Context(short, 3, 6, 4), Error, "element count");
});
