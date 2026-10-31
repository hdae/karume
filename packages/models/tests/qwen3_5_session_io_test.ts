// The host-built inputs of the Qwen3.5 text tower (src/qwen3_5/rope.ts, src/qwen3_5/session-io.ts —
// ADR 0137 decision 2). No GPU and no local assets.
//
// The fixture `fixtures/qwen3_5-encoder/rope-rows.json` is the recipe's output, not this module's: it
// holds the sha256 of `rope_rows(range(4096), config)` (f32 little-endian, cos and sin), five of its
// rows word for word, the recipe's input names and length bounds, and the checkpoint the config was
// read from. It was written on 2026-10-31 by
//
//     cd tools/export-recipes && uv run --with 'transformers==5.14.1' --inexact python -c '<generator>'
//
// where <generator> reads the pinned checkpoint's text config (`export.text_config(snapshot_dir(
// SOURCES[DEFAULT_SOURCE]))`), calls `patch.rope_rows(range(patch.MAX_TOKENS), config)`, and prints
// the JSON (sha256 of `.numpy().astype("<f4").tobytes()`; sampled rows as `.view("<u4")` words).

import { readTextFile } from "@karume/test-support/fs";
import { describe, it } from "vite-plus/test";
import { assertEquals, assertStrictEquals, assertThrows } from "@std/assert";
import type { Tensor } from "@karume/runtime";
import {
  buildQwen3_5Positions,
  buildQwen3_5RopeRows,
  QWEN3_5_ROPE_THETA,
  QWEN3_5_ROTARY_DIM,
} from "../src/qwen3_5/rope.ts";
import {
  QWEN3_5_INPUT_IDS,
  QWEN3_5_MAX_TOKENS,
  QWEN3_5_MIN_TOKENS,
  QWEN3_5_POSITIONS,
  QWEN3_5_ROPE_COS,
  QWEN3_5_ROPE_SIN,
  qwen3_5HiddenStates,
  qwen3_5SessionInputs,
} from "../src/qwen3_5/session-io.ts";

type RopeFixture = {
  readonly inputNames: readonly string[];
  readonly minTokens: number;
  readonly maxTokens: number;
  readonly theta: number;
  readonly rotaryDim: number;
  readonly length: number;
  readonly sha256: { readonly cos: string; readonly sin: string };
  readonly samples: {
    readonly positions: readonly number[];
    readonly cos: readonly (readonly number[])[];
    readonly sin: readonly (readonly number[])[];
  };
};

const fixture = JSON.parse(
  await readTextFile(new URL("./fixtures/qwen3_5-encoder/rope-rows.json", import.meta.url)),
) as RopeFixture;

/** The embedding's row count of Qwen3.5-9B (the graph declaration's `embed_tokens.weight`). */
const VOCABULARY = 248_320;

const sha256Hex = async (data: Float32Array<ArrayBuffer>): Promise<string> =>
  Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", data)))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

/** The f32 words of one row of a `[1, T, width]` table. */
const rowWords = (data: Float32Array<ArrayBuffer>, position: number, width: number): number[] => [
  ...new Uint32Array(data.buffer, position * width * 4, width),
];

/** How many f32 words of the sampled rows differ from the recipe's. */
const differingWords = (
  data: Float32Array<ArrayBuffer>,
  expected: readonly (readonly number[])[],
): number =>
  fixture.samples.positions.reduce(
    (count, position, index) =>
      count +
      rowWords(data, position, fixture.rotaryDim).filter(
        (word, column) => word !== expected[index][column],
      ).length,
    0,
  );

describe("Qwen3.5 host constants: the recipe's graph interface", () => {
  it("names the graph inputs as the recipe's INPUT_NAMES, in its order", () => {
    assertEquals(
      [QWEN3_5_INPUT_IDS, QWEN3_5_POSITIONS, QWEN3_5_ROPE_COS, QWEN3_5_ROPE_SIN],
      fixture.inputNames,
    );
  });

  it("bounds the length by the recipe's MIN_TOKENS / MAX_TOKENS", () => {
    assertEquals([QWEN3_5_MIN_TOKENS, QWEN3_5_MAX_TOKENS], [fixture.minTokens, fixture.maxTokens]);
  });

  it("pins theta and the rotary width of the checkpoint the recipe reads", () => {
    assertEquals([QWEN3_5_ROPE_THETA, QWEN3_5_ROTARY_DIM], [fixture.theta, fixture.rotaryDim]);
  });
});

describe("buildQwen3_5RopeRows: the recipe's rope_rows, bit for bit", () => {
  const rows = buildQwen3_5RopeRows(fixture.length);

  it("returns f32 [1, T, rotaryDim] tables for cos and sin", () => {
    for (const table of [rows.cos, rows.sin]) {
      assertEquals(table.dtype, "f32");
      assertEquals(table.shape, [1, fixture.length, QWEN3_5_ROTARY_DIM]);
      assertEquals(table.data.length, fixture.length * QWEN3_5_ROTARY_DIM);
    }
  });

  it("matches every f32 word of the sampled rows (positions 0, 1, 62, 1023, 4095)", () => {
    assertEquals(differingWords(rows.cos.data, fixture.samples.cos), 0, "cos words that differ");
    assertEquals(differingWords(rows.sin.data, fixture.samples.sin), 0, "sin words that differ");
  });

  it("matches the sha256 of the whole tables (every position up to the length bound)", async () => {
    assertEquals(await sha256Hex(rows.cos.data), fixture.sha256.cos, "cos");
    assertEquals(await sha256Hex(rows.sin.data), fixture.sha256.sin, "sin");
  });

  it("gives a shorter sequence the leading rows of the long one", () => {
    const short = buildQwen3_5RopeRows(63);
    const width = QWEN3_5_ROTARY_DIM;
    assertEquals([...short.cos.data], [...rows.cos.data.subarray(0, 63 * width)]);
    assertEquals([...short.sin.data], [...rows.sin.data.subarray(0, 63 * width)]);
  });
});

describe("buildQwen3_5Positions", () => {
  it("is 0 … T−1 as f32 [T]", () => {
    const positions = buildQwen3_5Positions(5);
    assertEquals(positions.dtype, "f32");
    assertEquals(positions.shape, [5]);
    assertEquals([...positions.data], [0, 1, 2, 3, 4]);
  });
});

const idsOf = (...values: number[]): Int32Array<ArrayBuffer> => Int32Array.from(values);

describe("qwen3_5SessionInputs", () => {
  it("bundles the ids as [1, T] without copying, with positions and the RoPE rows of the same T", () => {
    const ids = idsOf(248045, 846, 198);

    const inputs = qwen3_5SessionInputs(ids, VOCABULARY);

    assertEquals(Object.keys(inputs).sort(), [...fixture.inputNames].sort());
    assertEquals(inputs[QWEN3_5_INPUT_IDS].shape, [1, 3]);
    assertStrictEquals(inputs[QWEN3_5_INPUT_IDS].data, ids);
    assertEquals([...inputs[QWEN3_5_POSITIONS].data], [0, 1, 2]);
    const rope = buildQwen3_5RopeRows(3);
    assertEquals([...inputs[QWEN3_5_ROPE_COS].data], [...rope.cos.data]);
    assertEquals([...inputs[QWEN3_5_ROPE_SIN].data], [...rope.sin.data]);
    assertEquals(inputs[QWEN3_5_ROPE_COS].shape, [1, 3, QWEN3_5_ROTARY_DIM]);
  });

  it("accepts the length bounds and rejects one past either", () => {
    assertEquals(qwen3_5SessionInputs(idsOf(7), VOCABULARY)[QWEN3_5_INPUT_IDS].shape, [1, 1]);
    const longest = new Int32Array(QWEN3_5_MAX_TOKENS);
    assertEquals(qwen3_5SessionInputs(longest, VOCABULARY)[QWEN3_5_POSITIONS].shape, [4096]);
    assertThrows(
      () => qwen3_5SessionInputs(idsOf(), VOCABULARY),
      Error,
      "length 0 is outside 1..4096",
    );
    assertThrows(
      () => qwen3_5SessionInputs(new Int32Array(QWEN3_5_MAX_TOKENS + 1), VOCABULARY),
      Error,
      "length 4097 is outside 1..4096",
    );
  });

  it("rejects an id outside the embedding's rows, naming it and its index", () => {
    assertEquals(
      qwen3_5SessionInputs(idsOf(0, VOCABULARY - 1), VOCABULARY)[QWEN3_5_INPUT_IDS].shape,
      [1, 2],
    );
    assertThrows(
      () => qwen3_5SessionInputs(idsOf(5, VOCABULARY), VOCABULARY),
      Error,
      "id 248320 at index 1 is outside the embedding's 248320 rows",
    );
    assertThrows(
      () => qwen3_5SessionInputs(idsOf(-1, 5), VOCABULARY),
      Error,
      "id -1 at index 0 is outside",
    );
  });
});

/** `[1, tokens, width]` output whose values are a 1-based running index (a mix-up shows in values). */
const outputOf = (tokens: number, width: number): Tensor => ({
  dtype: "f32",
  shape: [1, tokens, width],
  data: Float32Array.from({ length: tokens * width }, (_, index) => index + 1),
});

describe("qwen3_5HiddenStates", () => {
  it("returns the contents of [1, tokens, width] as [tokens, width] without copying", () => {
    const output = outputOf(3, 4);
    assertStrictEquals(qwen3_5HiddenStates(output, 3, 4), output.data);
  });

  it("rejects another dtype, shape or element count", () => {
    assertThrows(() => qwen3_5HiddenStates(outputOf(3, 4), 2, 4), Error, "is not [1,2,4]");
    assertThrows(() => qwen3_5HiddenStates(outputOf(3, 4), 3, 8), Error, "is not [1,3,8]");
    const ids: Tensor = { dtype: "i32", shape: [1, 3, 4], data: new Int32Array(12) };
    assertThrows(() => qwen3_5HiddenStates(ids, 3, 4), Error, "dtype i32 is not f32");
    const short: Tensor = { dtype: "f32", shape: [1, 3, 4], data: new Float32Array(11) };
    assertThrows(() => qwen3_5HiddenStates(short, 3, 4), Error, "element count 11");
  });
});
