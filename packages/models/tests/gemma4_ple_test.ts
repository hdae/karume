// Checks of the lifetime, budget and reading strategy of the PLE host gather (`src/gemma/ple.ts` — ADR 0085 / 0109
// decision 4). Needs neither a GPU nor real assets (the synthetic index and blocks come from `helpers/ple-fixture.ts`).
//
// Six things are pinned here:
//
// - **The index's accepted set** (`parseGemma4PleIndex`): schema 3 only, unknown keys, contiguous ranges, and
//   consistency of `rowBytes`. A combination where only the index was rebuilt fails only here.
// - **The comparison of the index and the container's assets** (`assertGemma4PleAssets` / `readGemma4PleIndex`):
//   the existence, role and logical length of the blocks are checked by **listing every mismatch**. A distribution
//   where only one side was rebuilt drifts in dozens of places at once, so it should not take one round trip per fix.
// - **Row translation**: token → (block, row offset) is correct at block boundaries and across blocks. A shift reads
//   not OOB but **a valid row of another token** (the silent wrong value of ADR 0085 decision 5).
// - **The policy table**: the number of unique rows per block and the free budget decide "full read / row read". The
//   values are bit-identical regardless of the path, and the only way to confirm that is "read the same ids on both
//   paths and compare them as u32" (f32 `===` misses NaN / ±0).
// - **The byte budget**: the residency limit is **in bytes, not in block count** (the block width changes with the
//   asset generation). It freezes the default, the LRU eviction order, 0 (no residency), and the rejection of a
//   budget that cannot hold even one block.
// - **Release**: `dispose()` empties the residency, and any later gather fails loudly.

import { beforeAll, describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { ModelInputError } from "../src/errors.ts";
import { createGemma4Ple } from "../src/gemma/ple.ts";
import {
  assertGemma4PleAssets,
  defaultGemma4PleResidentBytes,
  gemma4PleBlockBytes,
  gemma4PleTotalBytes,
  parseGemma4PleIndex,
  readGemma4PleIndex,
} from "../src/gemma/ple-index.ts";
import {
  expectedValue,
  pleComponentOf,
  pleFixture,
  type PleFixtureSpec,
} from "./helpers/ple-fixture.ts";

/**
 * The default dimensions. One row of `values` and of `scales` has **the same byte count** (8 B), so the blocks are
 * the same size too (2 rows = 16 B) and the budget accounting reads directly as a block count.
 */
const SPEC: PleFixtureSpec = {
  tokens: 6,
  layers: 2,
  dim: 4,
  // Powers of 2 (f32 multiplication is exact — the synthetic side keeps the same property as ADR 0085 decision 4).
  embedScale: 4,
  valueRows: 2,
  scaleRows: 2,
};

/** One block (both tables of this index have the same width). */
const BLOCK_BYTES = SPEC.layers * SPEC.dim * (SPEC.valueRows as number);
/** One row (values + scales) — the unit of the row cache. */
const ROW_BYTES = SPEC.layers * SPEC.dim + SPEC.layers * 4;

const fixture = (options: { readonly defer?: boolean } = {}) => pleFixture(SPEC, options);

const pleOf = (rig: ReturnType<typeof fixture>, maxResidentBytes?: number) =>
  createGemma4Ple({
    index: rig.index,
    openBlock: rig.openBlock,
    vocabSize: SPEC.tokens,
    ...(maxResidentBytes === undefined ? {} : { maxResidentBytes }),
  });

/** Compares a `[1, ids, layers, dim]` f32 with the expected values. */
const assertRows = (tensor: { readonly data: Float32Array }, ids: readonly number[]): void => {
  const stride = SPEC.layers * SPEC.dim;
  for (const [position, id] of ids.entries()) {
    for (let layer = 0; layer < SPEC.layers; layer += 1) {
      for (let column = 0; column < SPEC.dim; column += 1) {
        assertEquals(
          tensor.data[position * stride + layer * SPEC.dim + column],
          expectedValue(SPEC, id, layer, column),
          `position ${position} (token ${id}) / layer ${layer} / column ${column}`,
        );
      }
    }
  }
};

const tensorOf = async (
  ple: ReturnType<typeof pleOf>,
  ids: readonly number[],
): Promise<{ readonly data: Float32Array }> => {
  const value = await ple.gather(ids);
  assertEquals(value.dtype, "f32");
  assert("data" in value && value.data instanceof Float32Array);
  assertEquals(value.shape, [1, ids.length, SPEC.layers, SPEC.dim]);
  return { data: value.data };
};

// ---- The index's accepted set --------------------------------------------------------

describe("parseGemma4PleIndex: a broken index is not silently read", () => {
  let raw: ReturnType<typeof fixture>["document"];
  beforeAll(() => {
    raw = fixture().document;
  });

  it("normal case (negative control — the check does not always fail)", () => {
    assertEquals(parseGemma4PleIndex(raw).tokens, SPEC.tokens);
  });

  it("the old sidecar version is not read (no dual reading)", () => {
    // Schema 4 (grouped scales) is read since the q4_0 QAT line — `gemma4_ple_grouped_test.ts` —
    // and schema 5 (a bf16 table) since the original-weights preset — `gemma4_ple_bf16_test.ts`;
    // 6 stands for a version this reader does not know.
    for (const schema of [1, 2, 6]) {
      assertThrows(
        () => parseGemma4PleIndex({ ...raw, schema }),
        Error,
        `ple_index.schema ${schema} is not 3`,
      );
    }
  });

  it("unknown key (a misspelling does not silently fall back to a default)", () => {
    assertThrows(() => parseGemma4PleIndex({ ...raw, shards: [] }), Error, "unknown key 'shards'");
  });

  it("storage is i8 / i4 / i2", () => {
    assertThrows(
      () => parseGemma4PleIndex({ ...raw, storage: "i3" }),
      Error,
      "ple_index.storage i3 is not i8 / i4 / i2",
    );
  });

  it("a missing storage is not read as i8 (required in schema 3)", () => {
    const { storage: _, ...withoutStorage } = raw;
    assertThrows(
      () => parseGemma4PleIndex(withoutStorage),
      Error,
      "ple_index.storage undefined is not i8 / i4 / i2",
    );
  });

  it("embedScale is a positive finite number (0 makes every row 0)", () => {
    assertThrows(
      () => parseGemma4PleIndex({ ...raw, embedScale: 0 }),
      Error,
      "ple_index.embedScale 0 is not a positive finite number",
    );
  });

  it("rowBytes matches the value determined by the declaration (layers / dimension / storage)", () => {
    const values = raw.values as { rowBytes: number; blocks: unknown[] };
    assertThrows(
      () => parseGemma4PleIndex({ ...raw, values: { ...values, rowBytes: values.rowBytes + 4 } }),
      Error,
      `ple_index.values.rowBytes ${values.rowBytes + 4} differs from ${values.rowBytes}, which the declaration determines`,
    );
  });

  it("duplicate block names (two ranges naming the same data)", () => {
    assertThrows(
      () =>
        parseGemma4PleIndex({
          ...raw,
          tokens: 4,
          values: {
            rowBytes: SPEC.layers * SPEC.dim,
            blocks: [
              { asset: "ple.values.0", start: 0, stop: 2 },
              { asset: "ple.values.0", start: 2, stop: 4 },
            ],
          },
          scales: {
            rowBytes: SPEC.layers * 4,
            blocks: [{ asset: "ple.scales.0", start: 0, stop: 4 }],
          },
        }),
      Error,
      ".asset 'ple.values.0' is a duplicate",
    );
  });

  it("non-contiguous ranges (ids that cannot be read appear silently)", () => {
    assertThrows(
      () =>
        parseGemma4PleIndex({
          ...raw,
          tokens: 5,
          values: {
            rowBytes: SPEC.layers * SPEC.dim,
            blocks: [
              { asset: "ple.values.0", start: 0, stop: 2 },
              { asset: "ple.values.1", start: 3, stop: 5 },
            ],
          },
          scales: {
            rowBytes: SPEC.layers * 4,
            blocks: [{ asset: "ple.scales.0", start: 0, stop: 5 }],
          },
        }),
      Error,
      ".start 3 does not continue from the end 2 of the previous block",
    );
  });

  it("total row count mismatch (a combination where only the index was rebuilt)", () => {
    assertThrows(
      () => parseGemma4PleIndex({ ...raw, tokens: SPEC.tokens + 1 }),
      Error,
      `the blocks total ${SPEC.tokens} rows, which differs from tokens ${SPEC.tokens + 1}`,
    );
  });

  it("the 2 tables can be split independently (block boundaries need not be aligned)", () => {
    const index = pleFixture({ ...SPEC, valueRows: 2, scaleRows: 3 }).index;
    assertEquals(index.values.blocks.length, 3);
    assertEquals(index.scales.blocks.length, 2);
  });
});

// ---- Comparison of the index and the container's assets (the admission step) -------------------------------

describe("readGemma4PleIndex: reads the index from the container's assets and compares its targets, listing every mismatch", () => {
  let rig: ReturnType<typeof fixture>;
  beforeAll(() => {
    rig = fixture();
  });

  it("normal case (negative control)", async () => {
    const index = await readGemma4PleIndex("test", pleComponentOf(rig));
    assertEquals(index, rig.index);
  });

  it("the index itself is missing", async () => {
    await assertRejects(
      () => readGemma4PleIndex("test", pleComponentOf(rig, { ple_index: undefined })),
      Error,
      "the container does not declare asset 'ple_index'",
    );
  });

  it("the index has a different role", async () => {
    await assertRejects(
      () => readGemma4PleIndex("test", pleComponentOf(rig, { ple_index: { role: "rope-base" } })),
      Error,
      "asset 'ple_index' has role 'rope-base'",
    );
  });

  it("a target block is missing from the container / has a different role / a different length — every mismatch listed", async () => {
    const error = await assertRejects(
      () =>
        readGemma4PleIndex(
          "test",
          pleComponentOf(rig, {
            "ple.values.0": undefined,
            "ple.values.1": { role: "ple-scales" },
            "ple.scales.2": { length: 1 },
          }),
        ),
      Error,
      "the PLE index and the container's assets disagree (3 problems)",
    );
    assert(error.message.includes("ple.values.0: not declared by the container"), error.message);
    assert(error.message.includes("ple.values.1: role 'ple-scales'"), error.message);
    assert(error.message.includes("ple.scales.2: logical length 1"), error.message);
  });

  it("a block in the container the index does not point at (the shape that runs without ever reading it)", () => {
    assertThrows(
      () =>
        assertGemma4PleAssets("test", rig.index, {
          assets: {
            ...pleComponentOf(rig).assets,
            "ple.values.99": "ple-values",
          },
          asset: pleComponentOf(rig).asset,
        }),
      Error,
      "ple.values.99: an asset with role 'ple-values' that the index does not point at",
    );
  });
});

// ---- Row translation (block boundaries and across blocks) ------------------------------------

test("Gemma4Ple.gather: the token → (block, row offset) translation is correct at boundaries", async () => {
  const rig = fixture();
  const ple = pleOf(rig, 0);
  // A sequence across the boundaries of 3 blocks (first, last, across).
  const ids = [0, 1, 2, 3, 4, 5];
  assertRows(await tensorOf(ple, ids), ids);

  // The read range is exactly "row (id - start) inside that block". Measuring the range is the only check that is not
  // always true (with value agreement alone, reading another block would also match through the same formula).
  const rowBytes = { values: rig.index.values.rowBytes, scales: rig.index.scales.rowBytes };
  const expected = ids.flatMap((id) => [
    {
      asset: `ple.values.${Math.floor(id / 2)}`,
      offset: (id % 2) * rowBytes.values,
      length: rowBytes.values,
    },
    {
      asset: `ple.scales.${Math.floor(id / 2)}`,
      offset: (id % 2) * rowBytes.scales,
      length: rowBytes.scales,
    },
  ]);
  assertEquals(
    [...rig.reads].sort((a, b) => a.asset.localeCompare(b.asset) || a.offset - b.offset),
    expected.sort((a, b) => a.asset.localeCompare(b.asset) || a.offset - b.offset),
  );
});

test("Gemma4Ple.gather: reads the same row even when the 2 tables' block boundaries differ", async () => {
  // `values` is cut every 2 rows and `scales` every 3 (in real assets the byte count of one row differs, so this happens normally).
  const spec: PleFixtureSpec = { ...SPEC, valueRows: 2, scaleRows: 3 };
  const rig = pleFixture(spec);
  const ple = createGemma4Ple({
    index: rig.index,
    openBlock: rig.openBlock,
    vocabSize: spec.tokens,
    maxResidentBytes: 0,
  });
  const ids = [2, 3];
  const tensor = await ple.gather(ids);
  assert("data" in tensor && tensor.data instanceof Float32Array);
  const stride = spec.layers * spec.dim;
  for (const [position, id] of ids.entries()) {
    for (let layer = 0; layer < spec.layers; layer += 1) {
      for (let column = 0; column < spec.dim; column += 1) {
        assertEquals(
          tensor.data[position * stride + layer * spec.dim + column],
          expectedValue(spec, id, layer, column),
        );
      }
    }
  }
  // Token 2 is in block 1 of values (row 0) and block 0 of scales (row 2) — a different block per table.
  assert(
    rig.reads.some((read) => read.asset === "ple.values.1" && read.offset === 0),
    JSON.stringify(rig.reads),
  );
  assert(
    rig.reads.some(
      (read) => read.asset === "ple.scales.0" && read.offset === 2 * rig.index.scales.rowBytes,
    ),
    JSON.stringify(rig.reads),
  );
});

test("Gemma4Ple.gather: each row of a sequence with repeated ids is bit-identical to a single gather", async () => {
  const rig = fixture();
  const ple = pleOf(rig, 0);
  const single = await tensorOf(ple, [3]);
  const repeated = await tensorOf(ple, [3, 3, 3]);
  const stride = SPEC.layers * SPEC.dim;
  for (let position = 0; position < 3; position += 1) {
    assertEquals(
      [...new Uint32Array(repeated.data.buffer, position * stride * 4, stride)],
      [...new Uint32Array(single.data.buffer)],
      `position ${position} is not bit-identical to a single gather`,
    );
  }
});

// ---- Policy table (full read ↔ row read) -------------------------------------------

test("Gemma4Ple.gather: a single-row gather does not read a whole block (decode of 1 token)", async () => {
  const rig = fixture();
  const ple = pleOf(rig);
  assertRows(await tensorOf(ple, [0]), [0]);
  const stats = ple.stats();
  assertEquals(stats.loads, 0, "reads a whole block for a single row");
  assertEquals(stats.rowReads, 2, "row reads are the 2 ranges of values / scales");
  assertEquals(stats.resident, 0);
  assertEquals(
    rig.reads.map((read) => read.length),
    [rig.index.values.rowBytes, rig.index.scales.rowBytes],
  );
});

test("Gemma4Ple.gather: a gather that touches every row of a block switches to a full read", async () => {
  const rig = fixture();
  const ple = pleOf(rig);
  // A block has 2 rows, so touching 2 rows reaches the threshold (= min(32, row count)).
  assertRows(await tensorOf(ple, [0, 1]), [0, 1]);
  const stats = ple.stats();
  assertEquals(stats.loads, 2, "reads one block each of values / scales in full");
  assertEquals(stats.rowReads, 0);
  assertEquals(stats.resident, 2);
  // Two blocks + copies of the two rows looked up (the row cache is outside the block budget —
  // ADR 0085 addendum 2026-10-27).
  assertEquals(stats.residentBytes, 2 * BLOCK_BYTES + 2 * ROW_BYTES);
  assertEquals(
    rig.reads.map((read) => read.offset),
    [0, 0],
    "reads from the middle despite a full read",
  );
});

test("Gemma4Ple.gather: the values of row reads and full reads are bit-identical", async () => {
  const ids = [0, 1];
  const rows = await tensorOf(pleOf(fixture(), 0), ids);
  const blocks = await tensorOf(pleOf(fixture()), ids);
  assertEquals(
    [...new Uint32Array(rows.data.buffer)],
    [...new Uint32Array(blocks.data.buffer)],
    "the row-read values are not bit-identical to the full read",
  );
  assertRows(rows, ids);
});

test("Gemma4Ple.gather: a full read happens only when it fits the budget without eviction", async () => {
  const rig = fixture();
  // The budget is one block. Once values is loaded, scales does not fit (a gather does not evict what it loaded
  // itself).
  const ple = pleOf(rig, BLOCK_BYTES);
  assertRows(await tensorOf(ple, [0, 1]), [0, 1]);
  const stats = ple.stats();
  assertEquals(stats.loads, 1, "reads 2 blocks with a budget of one");
  assertEquals(stats.resident, 1);
  assertEquals(stats.rowReads, 2, "the table that did not fit falls back to row reads");
});

test("Gemma4Ple.gather: below 32 rows it still switches to a full read when the block is small (a remainder block)", async () => {
  // One block of 64 rows (the threshold is 32 rows) — 31 rows are row reads, 32 rows a full read.
  const wide: PleFixtureSpec = { tokens: 64, layers: 1, dim: 4, embedScale: 4 };
  const ids = Array.from({ length: 31 }, (_row, index) => index);
  const narrow = pleFixture(wide);
  const rowPle = createGemma4Ple({
    index: narrow.index,
    openBlock: narrow.openBlock,
    vocabSize: wide.tokens,
    maxResidentBytes: gemma4PleTotalBytes(narrow.index),
  });
  await rowPle.gather(ids);
  assertEquals(rowPle.stats().loads, 0, "switches to a full read at 31 rows (the threshold is 32)");
  assertEquals(rowPle.stats().rowReads, 62);

  const full = pleFixture(wide);
  const fullPle = createGemma4Ple({
    index: full.index,
    openBlock: full.openBlock,
    vocabSize: wide.tokens,
    maxResidentBytes: gemma4PleTotalBytes(full.index),
  });
  await fullPle.gather([...ids, 31]);
  assertEquals(fullPle.stats().loads, 2, "still row reads at 32 rows");
  assertEquals(fullPle.stats().rowReads, 0);
});

test("Gemma4Ple.gather: concurrent row reads do not exceed the limit (do not exhaust fds)", async () => {
  // Reads 128 rows from an index with 1 row per block (= always row reads).
  const spec: PleFixtureSpec = { tokens: 128, layers: 1, dim: 4, embedScale: 4, valueRows: 1 };
  const rig = pleFixture(spec, { defer: true });
  const ple = createGemma4Ple({
    index: rig.index,
    openBlock: rig.openBlock,
    vocabSize: spec.tokens,
    maxResidentBytes: 0,
  });
  await ple.gather(Array.from({ length: 128 }, (_row, index) => index));
  assertEquals(ple.stats().rowReads, 256);
  assert(
    rig.peak.inFlight <= 16,
    `the concurrency peak ${rig.peak.inFlight} exceeded the limit of 16`,
  );
  assert(
    rig.peak.inFlight > 1,
    "reads were never issued concurrently (the limit check is vacuous)",
  );
});

// ---- Row cache ----------------------------------------------------------

test("Gemma4Ple.gather: a row read by row reads stays in the row cache (headroom outside the block budget)", async () => {
  const rig = fixture();
  const ple = pleOf(rig);
  await ple.gather([0]);
  const first = rig.reads.length;
  assertEquals(
    ple.stats().residentBytes,
    ROW_BYTES,
    "the kept row is not counted in residentBytes",
  );
  await ple.gather([0]);
  assertEquals(rig.reads.length, first, "misses the row cache and reads again");
  assertEquals(ple.stats().rowReads, 2);
});

test("Gemma4Ple.gather: a budget of 0 keeps no rows either (reads again every time)", async () => {
  const rig = fixture();
  const ple = pleOf(rig, 0);
  await ple.gather([0]);
  await ple.gather([0]);
  assertEquals(ple.stats().residentBytes, 0);
  assertEquals(ple.stats().rowReads, 4, "keeps rows despite a budget of 0");
});

// ---- Budget and LRU ------------------------------------------------------------

test("Gemma4Ple: the resident bytes are decided by the index alone (a meaning independent of the block width)", () => {
  const rig = fixture();
  assertEquals(gemma4PleBlockBytes(rig.index.values, rig.index.values.blocks[0]), BLOCK_BYTES);
  assertEquals(gemma4PleTotalBytes(rig.index), 6 * BLOCK_BYTES);
  // The default is "the 2 largest blocks" — the meaning (any 2 blocks fit) holds when the width changes.
  assertEquals(defaultGemma4PleResidentBytes(rig.index), 2 * BLOCK_BYTES);
  const wide = pleFixture({ ...SPEC, valueRows: 3, scaleRows: 3 }).index;
  assertEquals(defaultGemma4PleResidentBytes(wide), 2 * 3 * SPEC.layers * SPEC.dim);
});

test("Gemma4Ple: eviction is LRU (the blocks an earlier gather loaded go first)", async () => {
  // Rows looked up stay in the row cache (ADR 0085 addendum 2026-10-27), so the block LRU is only
  // observable through rows not looked up yet. Blocks have 96 rows (three times the whole-read
  // threshold of 32), and the re-read uses another 32 rows of the same block.
  const spec: PleFixtureSpec = {
    tokens: 192,
    layers: 1,
    dim: 4,
    embedScale: 4,
    valueRows: 96,
    scaleRows: 96,
  };
  const block = 96 * 4;
  const rig = pleFixture(spec);
  const ple = createGemma4Ple({
    index: rig.index,
    openBlock: rig.openBlock,
    vocabSize: spec.tokens,
    maxResidentBytes: 2 * block,
  });
  const range = (from: number) => Array.from({ length: 32 }, (_row, row) => from + row);
  await ple.gather(range(0));
  assertEquals(ple.stats().resident, 2);
  // Loading another pair of blocks in full evicts the earlier pair (the budget is 2 blocks).
  await ple.gather(range(96));
  assertEquals(ple.stats().resident, 2);
  assertEquals(
    ple.stats().residentBytes,
    2 * block + 64 * (4 + 4),
    "two blocks + the 64 rows looked up",
  );
  const loads = ple.stats().loads;
  // The older ones were evicted — reading them again adds reads.
  await ple.gather(range(32));
  assertEquals(ple.stats().loads, loads + 2, "the evicted blocks were not read again");
  // The blocks touched just before are still alive.
  const after = ple.stats().loads;
  const rowReads = ple.stats().rowReads;
  await ple.gather(range(64));
  assertEquals(ple.stats().loads, after, "re-reads a resident block");
  assertEquals(ple.stats().rowReads, rowReads, "rows of a resident block were read by row reads");
});

test("Gemma4Ple.gather: a single gather does not evict its own hits", async () => {
  // Rows looked up stay in the row cache (ADR 0085 addendum 2026-10-27), so the block hit has to
  // come from a row not looked up yet. Blocks have 33 rows, and the first gather takes only 32 (the
  // whole-read threshold).
  const spec: PleFixtureSpec = {
    tokens: 66,
    layers: 1,
    dim: 4,
    embedScale: 4,
    valueRows: 33,
    scaleRows: 33,
  };
  const rig = pleFixture(spec);
  const ple = createGemma4Ple({
    index: rig.index,
    openBlock: rig.openBlock,
    vocabSize: spec.tokens,
    maxResidentBytes: 2 * 33 * 4,
  });
  await ple.gather(Array.from({ length: 32 }, (_row, row) => row));
  const loads = ple.stats().loads;
  const rowReads = ple.stats().rowReads;
  // One gather touches a hit (row 32 of block pair 0, not looked up yet) and a miss (the 33 rows of
  // block pair 1). The budget holds two blocks, so loading the miss whole would evict the hit — the
  // policy counts the hits first, so it does not.
  const ids = Array.from({ length: 34 }, (_row, row) => 32 + row);
  const tensor = await ple.gather(ids);
  assert("data" in tensor && tensor.data instanceof Float32Array);
  for (const [position, id] of ids.entries()) {
    for (let column = 0; column < spec.dim; column += 1) {
      assertEquals(
        tensor.data[position * spec.dim + column],
        expectedValue(spec, id, 0, column),
        `position ${position} (token ${id}) / column ${column}`,
      );
    }
  }
  assertEquals(ple.stats().loads, loads, "re-reads a block that should be a hit");
  assertEquals(ple.stats().resident, 2);
  assertEquals(
    ple.stats().rowReads,
    rowReads + 33 * 2,
    "misses that did not fit fall back to row reads",
  );
});

test("Gemma4Ple: a budget that cannot hold even one block fails loudly at build time", () => {
  const rig = fixture();
  const error = assertThrows(
    () => pleOf(rig, BLOCK_BYTES - 1),
    ModelInputError,
    `maxResidentBytes ${BLOCK_BYTES - 1} is less than one PLE block, ${BLOCK_BYTES} bytes`,
  );
  assert(
    error instanceof ModelInputError,
    "the caller's setting, so caused by the input (ADR 0107 decision 2)",
  );
  // 0 is a legitimate setting meaning "no residency" (negative control).
  assertEquals(pleOf(rig, 0).stats().residentBytes, 0);
  assertThrows(() => pleOf(rig, -1), ModelInputError, "is not a non-negative integer");
});

// ---- Cross-checking the id space and ranges -----------------------------------------------

test("createGemma4Ple: the vocab cross-check fails before the reader is touched", () => {
  const rig = fixture();
  const error = assertThrows(
    () =>
      createGemma4Ple({
        index: rig.index,
        openBlock: rig.openBlock,
        vocabSize: SPEC.tokens + 1,
      }),
    Error,
    `PLE index's row count ${SPEC.tokens} differs from the main embedding's vocab row count ${SPEC.tokens + 1}`,
  );
  assertEquals(rig.opens.length, 0, "opens a block before the cross-check");
  // A mismatch in the exported combination, so unlike the budget check of the same function it is **not** caused by
  // the input (no change to `maxResidentPleBytes` by the caller fixes it — ADR 0107 decision 2).
  assert(!(error instanceof ModelInputError));
});

test("Gemma4Ple.gather: an id out of range and an empty sequence fail loudly (no valid row of another token is read)", async () => {
  const rig = fixture();
  const ple = pleOf(rig);
  for (const bad of [SPEC.tokens, -1, 1.5]) {
    await assertRejects(
      () => ple.gather([bad]),
      Error,
      `token id[0] ${bad} is outside the PLE index's 0..${SPEC.tokens - 1}`,
    );
  }
  await assertRejects(() => ple.gather([]), Error, "token sequence of the PLE gather is empty");
  assertEquals(rig.opens.length, 0, "opens a block before the range check");
});

// ---- Abort and release ------------------------------------------------------------

test("Gemma4Ple.gather: an already aborted signal opens not a single block", async () => {
  const rig = fixture();
  const ple = pleOf(rig);
  const controller = new AbortController();
  controller.abort(new Error("test abort"));
  await assertRejects(() => ple.gather([0], { signal: controller.signal }), Error, "test abort");
  assertEquals(rig.opens.length, 0);
});

test("Gemma4Ple.gather: an abort during the run takes effect at a step boundary and keeps no rows that were read", async () => {
  // An index with 1 row per block (= always row reads). The abort takes effect at the boundary after the first read settles.
  const spec: PleFixtureSpec = { tokens: 64, layers: 1, dim: 4, embedScale: 4, valueRows: 1 };
  const rig = pleFixture(spec, { defer: true });
  const ple = createGemma4Ple({
    index: rig.index,
    openBlock: rig.openBlock,
    vocabSize: spec.tokens,
    maxResidentBytes: 0,
  });
  const controller = new AbortController();
  const pending = ple.gather(
    Array.from({ length: 64 }, (_row, index) => index),
    { signal: controller.signal },
  );
  controller.abort(new Error("test abort"));
  await assertRejects(() => pending, Error, "test abort");
  assert(
    rig.reads.length < 128,
    `reads every range even after the abort (${rig.reads.length} reads)`,
  );
  assert(rig.reads.length > 0, "not a single read happened (the abort check is vacuous)");
});

test("Gemma4Ple.gather: when one row read fails, it stops issuing new reads and fails after the started reads settle", async () => {
  // 1 row per block, budget 0 (= all 40 ranges are row reads). Only the first read rejects. If the other workers kept
  // issuing after the failure, they would overlap with the caller's retry and exceed the concurrency limit of 16.
  const spec: PleFixtureSpec = { tokens: 20, layers: 1, dim: 4, embedScale: 4, valueRows: 1 };
  const rig = pleFixture(spec, { defer: true });
  let calls = 0;
  let inFlight = 0;
  const openBlock = (asset: string) => {
    const reader = rig.openBlock(asset);
    return {
      ...reader,
      read: (offset: number, length: number) => {
        calls += 1;
        inFlight += 1;
        const pending =
          calls === 1
            ? new Promise<Uint8Array<ArrayBuffer>>((_resolve, reject) =>
                setTimeout(() => reject(new Error("injected read failure")), 0),
              )
            : reader.read(offset, length);
        return pending.finally(() => {
          inFlight -= 1;
        });
      },
    };
  };
  const ple = createGemma4Ple({
    index: rig.index,
    openBlock,
    vocabSize: spec.tokens,
    maxResidentBytes: 0,
  });
  await assertRejects(
    () => ple.gather(Array.from({ length: spec.tokens }, (_row, index) => index)),
    Error,
    "injected read failure",
  );
  assertEquals(inFlight, 0, "rejected before the started reads settled");
  const callsAtReject = calls;
  assert(callsAtReject < 40, `issues every range even after the failure (${callsAtReject} reads)`);
  // No new read is issued after the failure arrives (observed after waiting a few macrotasks).
  for (let tick = 0; tick < 8; tick += 1) await new Promise((resolve) => setTimeout(resolve, 0));
  assertEquals(calls, callsAtReject, "a new read was issued after the reject");
});

test("Gemma4Ple: dispose empties the residency, and any later gather fails loudly", async () => {
  const rig = fixture();
  const ple = pleOf(rig);
  await ple.gather([0, 1]);
  assert(ple.stats().residentBytes > 0);
  ple.dispose();
  assertEquals(ple.stats(), { loads: 2, rowReads: 0, resident: 0, residentBytes: 0 });
  await assertRejects(() => ple.gather([0]), Error, "a disposed index cannot be looked up");
  // Idempotent.
  ple.dispose();
  assertEquals(ple.stats().residentBytes, 0);
});

test("Gemma4Ple: a read-ahead that completes after dispose does not bring the residency back", async () => {
  const rig = fixture({ defer: true });
  const ple = pleOf(rig);
  const pending = ple.gather([0, 1]);
  ple.dispose();
  await pending;
  assertEquals(ple.stats().resident, 0);
  assertEquals(ple.stats().residentBytes, 0);
});
