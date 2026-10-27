// The row cache of the host PLE gather (`src/gemma/ple.ts` — ADR 0085 addenda 2026-09-12 and
// 2026-10-27, perf-ledger H-41). No GPU and no real assets (`helpers/ple-fixture.ts` builds the
// synthetic index and blocks).
//
// What this pins:
//
// - **Rows from resident whole blocks are kept too.** The real E2B layout makes scales block 0
//   (91% of the vocabulary) whole-resident after the first prompt with 32 or more distinct ids;
//   from then on every row has its scales from a block view. A cache that kept only rows read by
//   row reads from both tables kept nothing, and decode re-read every row.
// - **The cache is headroom outside the block budget.** With a values block and the scales block
//   resident, the budget's free space is less than one row; a cache bounded by that space keeps
//   nothing even when it copies rows out of blocks.
// - **The bound:** blocks ≤ budget, rows ≤ 256, budget 0 keeps no rows; LRU; nothing is kept after
//   an abort, a failed read or dispose.
//
// Value checks compare u32 bit patterns against a gather with a budget of 0 (row reads only, no
// cache) — f32 `===` would miss NaN / ±0.

import { describe, it } from "vite-plus/test";
import { assert, assertEquals, assertRejects } from "@std/assert";
import type { AssetReader } from "@karume/runtime";
import { createGemma4Ple } from "../src/gemma/ple.ts";
import { expectedValue, pleFixture, type PleFixtureSpec } from "./helpers/ple-fixture.ts";

/** Kept rows per owner (`ROW_CACHE_CAPACITY` in `src/gemma/ple.ts`). */
const ROW_CACHE_CAPACITY = 256;

/** Quantized values per byte of each storage. */
const PER_BYTE = { i8: 1, i4: 2, i2: 4 } as const;

/** Bytes of one row of both tables (values + scales) — the unit of the row cache. */
const rowBytesOf = (spec: PleFixtureSpec): number =>
  (spec.layers * spec.dim) / PER_BYTE[spec.storage ?? "i8"] + spec.layers * 4;

type Rig = ReturnType<typeof pleFixture>;

const pleOf = (
  rig: Rig,
  spec: PleFixtureSpec,
  maxResidentBytes?: number,
  openBlock: (asset: string) => AssetReader = rig.openBlock,
) =>
  createGemma4Ple({
    index: rig.index,
    openBlock,
    vocabSize: spec.tokens,
    ...(maxResidentBytes === undefined ? {} : { maxResidentBytes }),
  });

const bitsOf = async (ple: ReturnType<typeof pleOf>, ids: readonly number[]): Promise<number[]> => {
  const tensor = await ple.gather(ids);
  assert("data" in tensor && tensor.data instanceof Float32Array);
  return [...new Uint32Array(tensor.data.buffer)];
};

/** The same ids through a fresh owner with a budget of 0 (row reads, nothing kept). */
const freshBits = (spec: PleFixtureSpec, ids: readonly number[]): Promise<number[]> =>
  bitsOf(pleOf(pleFixture(spec), spec, 0), ids);

const assertExpected = async (
  ple: ReturnType<typeof pleOf>,
  spec: PleFixtureSpec,
  ids: readonly number[],
): Promise<void> => {
  const tensor = await ple.gather(ids);
  assert("data" in tensor && tensor.data instanceof Float32Array);
  const stride = spec.layers * spec.dim;
  for (const [position, id] of ids.entries()) {
    for (let layer = 0; layer < spec.layers; layer += 1) {
      for (let column = 0; column < spec.dim; column += 1) {
        assertEquals(
          tensor.data[position * stride + layer * spec.dim + column],
          expectedValue(spec, id, layer, column),
          `position ${position} (token ${id}) / layer ${layer} / column ${column}`,
        );
      }
    }
  }
};

describe("Gemma4Ple row cache: a scales block made resident by a wide prompt", () => {
  // values: 16 blocks of 4 rows (a whole read needs all 4 rows); scales: one block of 64 rows (a
  // whole read needs 32 rows) — the shape of the real E2B index, where one scales block covers
  // most of the vocabulary and a values block covers a few thousand tokens.
  const spec: PleFixtureSpec = {
    tokens: 64,
    layers: 2,
    dim: 4,
    embedScale: 4,
    valueRows: 4,
    scaleRows: 64,
  };
  // 32 distinct ids, two per values block: the scales block reaches the whole-read threshold, no
  // values block does.
  const prompt = Array.from({ length: 32 }, (_row, n) => Math.floor(n / 2) * 4 + (n % 2));

  it("loads the scales block whole and reads the values by row reads (the precondition)", async () => {
    const rig = pleFixture(spec);
    const ple = pleOf(rig, spec);
    await ple.gather(prompt);
    assertEquals(ple.stats().loads, 1, "only the scales block is read whole");
    assertEquals(ple.stats().resident, 1);
    assertEquals(ple.stats().rowReads, prompt.length, "one values row read per id");
  });

  it("serves later gathers of ids seen before from the row cache, without reading", async () => {
    const rig = pleFixture(spec);
    const ple = pleOf(rig, spec);
    await ple.gather(prompt);
    const rowReads = ple.stats().rowReads;
    const reads = rig.reads.length;
    // Decode-like gathers: one id at a time, ids the prompt already looked up.
    const seen = [prompt[0], prompt[7], prompt[31], prompt[7]];
    for (const id of seen) await ple.gather([id]);
    assertEquals(ple.stats().rowReads, rowReads, "a seen id was read again (the row was not kept)");
    assertEquals(rig.reads.length, reads);
  });

  it("returns values bit-identical to a fresh read, and the expected dequantization", async () => {
    const rig = pleFixture(spec);
    const ple = pleOf(rig, spec);
    await ple.gather(prompt);
    const ids = [prompt[3], prompt[30], prompt[3]];
    assertEquals(await bitsOf(ple, ids), await freshBits(spec, ids));
    await assertExpected(ple, spec, prompt);
  });

  it("still reads ids it has not seen (the cache does not answer for other tokens)", async () => {
    const rig = pleFixture(spec);
    const ple = pleOf(rig, spec);
    await ple.gather(prompt);
    const rowReads = ple.stats().rowReads;
    const unseen = 2; // values block 0, row 2 — not in the prompt.
    assert(!prompt.includes(unseen));
    await assertExpected(ple, spec, [unseen]);
    assertEquals(ple.stats().rowReads, rowReads + 1, "the values row of an unseen id is read");
  });

  it("counts each kept row as exactly one row, whichever path supplied it", async () => {
    // residentBytes measures the kept arrays' own buffers, so a row that aliased a block (a view
    // instead of a copy) or an over-long copy breaks this equality.
    const rig = pleFixture(spec);
    const ple = pleOf(rig, spec);
    const valuesBlockBytes = 4 * spec.layers * spec.dim;
    const scalesBlockBytes = spec.tokens * spec.layers * 4;
    // Values block 1 (rows 4..7) touched in full: values copied out of the whole block, scales by
    // row reads.
    await ple.gather([4, 5, 6, 7]);
    assertEquals(ple.stats().loads, 1, "values block 1 must be read whole");
    assertEquals(ple.stats().rowReads, 4, "scales must be row reads");
    assertEquals(ple.stats().residentBytes, valuesBlockBytes + 4 * rowBytesOf(spec));
    // Rows 0..2 of the other 15 values blocks: values by row reads, scales copied out of the whole
    // scales block (45 distinct ids reach its threshold of 32).
    const others = Array.from({ length: 16 }, (_block, block) => block)
      .filter((block) => block !== 1)
      .flatMap((block) => [block * 4, block * 4 + 1, block * 4 + 2]);
    await ple.gather(others);
    assertEquals(ple.stats().loads, 2, "the scales block must be read whole");
    assertEquals(ple.stats().rowReads, 4 + 45, "values must be row reads");
    assertEquals(
      ple.stats().residentBytes,
      valuesBlockBytes + scalesBlockBytes + (4 + 45) * rowBytesOf(spec),
    );
  });

  it("counts the kept rows in residentBytes on top of the resident block", async () => {
    const rig = pleFixture(spec);
    const ple = pleOf(rig, spec);
    await ple.gather(prompt);
    const scalesBlockBytes = spec.tokens * spec.layers * 4;
    assertEquals(ple.stats().residentBytes, scalesBlockBytes + prompt.length * rowBytesOf(spec));
  });
});

for (const storage of ["i8", "i4"] as const) {
  describe(`Gemma4Ple row cache (${storage}): a values block and the scales block both resident`, () => {
    // values: 2 blocks of 32 rows (i8: 64 B per row → 2,048 B each; i4 packs two per byte);
    // scales: one block of 64 rows (8 B per row → 512 B). The budget holds exactly one values block
    // and the scales block, so its free space (0 B) is less than one row — the default E2B budget
    // leaves 8,120 B against a 9,100 B row. The i4 case catches a copy offset that is right only for
    // one value per byte.
    const spec: PleFixtureSpec = {
      storage,
      tokens: 64,
      layers: 2,
      dim: 32,
      embedScale: 4,
      valueRows: 32,
      scaleRows: 64,
    };
    const valuesBlockBytes = (32 * spec.layers * spec.dim) / PER_BYTE[storage];
    const scalesBlockBytes = spec.tokens * spec.layers * 4;
    const budget = valuesBlockBytes + scalesBlockBytes;
    const first = Array.from({ length: 32 }, (_row, n) => n);
    const second = Array.from({ length: 32 }, (_row, n) => 32 + n);

    it("keeps rows from whole blocks even when the budget has no free space", async () => {
      const rig = pleFixture(spec);
      const ple = pleOf(rig, spec, budget);
      await ple.gather(first);
      assertEquals(ple.stats().loads, 2, "values block 0 and the scales block are read whole");
      assertEquals(ple.stats().rowReads, 0);
      // The second values block evicts the first (the scales block is a hit and stays).
      await ple.gather(second);
      assertEquals(ple.stats().loads, 3);
      assertEquals(ple.stats().resident, 2);
      const rowReads = ple.stats().rowReads;
      const reads = rig.reads.length;
      // Ids of the evicted values block come from the row cache, not from row reads.
      const ids = [5, 0, 31];
      assertEquals(await bitsOf(ple, ids), await freshBits(spec, ids));
      assertEquals(ple.stats().rowReads, rowReads, "a row of the evicted block was read again");
      assertEquals(rig.reads.length, reads);
    });

    it("keeps blocks within the budget and rows within the 256-row headroom", async () => {
      const rig = pleFixture(spec);
      const ple = pleOf(rig, spec, budget);
      await ple.gather(first);
      await ple.gather(second);
      const stats = ple.stats();
      assertEquals(stats.residentBytes, budget + 64 * rowBytesOf(spec));
      assert(stats.residentBytes <= budget + ROW_CACHE_CAPACITY * rowBytesOf(spec));
    });
  });
}

describe("Gemma4Ple row cache: the bound", () => {
  // values and scales: 5 blocks of 64 rows (a whole read needs 32 rows, so single-id gathers are
  // row reads).
  const spec: PleFixtureSpec = {
    tokens: 320,
    layers: 1,
    dim: 4,
    embedScale: 4,
    valueRows: 64,
    scaleRows: 64,
  };

  it("keeps at most 256 rows and drops the least recently used first", async () => {
    const rig = pleFixture(spec);
    const ple = pleOf(rig, spec);
    for (let id = 0; id < 300; id += 1) await ple.gather([id]);
    assertEquals(ple.stats().loads, 0, "single-id gathers must stay row reads");
    assertEquals(ple.stats().residentBytes, ROW_CACHE_CAPACITY * rowBytesOf(spec));
    // Kept: 44..299. The oldest kept row hits and moves to the tail.
    const rowReads = ple.stats().rowReads;
    await ple.gather([300 - ROW_CACHE_CAPACITY]);
    assertEquals(ple.stats().rowReads, rowReads, "the oldest kept row was read again");
    // An evicted row is read (and kept, which now evicts 45 — 44 was just touched).
    await ple.gather([300 - ROW_CACHE_CAPACITY - 1]);
    assertEquals(ple.stats().rowReads, rowReads + 2, "an evicted row was answered from the cache");
    await ple.gather([300 - ROW_CACHE_CAPACITY]);
    assertEquals(
      ple.stats().rowReads,
      rowReads + 2,
      "the touched row was evicted before older ones",
    );
    await ple.gather([300 - ROW_CACHE_CAPACITY + 1]);
    assertEquals(ple.stats().rowReads, rowReads + 4, "the least recently used row was kept");
    assertEquals(ple.stats().residentBytes, ROW_CACHE_CAPACITY * rowBytesOf(spec));
  });

  it("keeps the last 256 distinct ids of a gather wider than the cache, in first-seen order", async () => {
    // 300 distinct ids, at most 22 per 64-row block (below the whole-read threshold of 32), so
    // every row is a row read and no block is resident to answer for an evicted row. The trailing
    // repeat of the first id does not make it recent: ids are ordered by their first position.
    const wide: PleFixtureSpec = { ...spec, tokens: 960 };
    const rig = pleFixture(wide);
    const ple = pleOf(rig, wide);
    const ids = Array.from({ length: 300 }, (_row, n) => n * 3);
    await ple.gather([...ids, ids[0]]);
    assertEquals(ple.stats().loads, 0, "the wide gather must stay row reads");
    assertEquals(ple.stats().resident, 0);
    assertEquals(ple.stats().residentBytes, ROW_CACHE_CAPACITY * rowBytesOf(wide));
    const rowReads = ple.stats().rowReads;
    // Kept: the last 256 (ids[44..299]) — hits move to the tail without evicting anything.
    for (const id of ids.slice(300 - ROW_CACHE_CAPACITY)) await ple.gather([id]);
    assertEquals(ple.stats().rowReads, rowReads, "a row of the last 256 ids was not kept");
    // Not kept: the first 44, the repeated first id included.
    await ple.gather(ids.slice(0, 300 - ROW_CACHE_CAPACITY));
    assertEquals(
      ple.stats().rowReads,
      rowReads + 2 * (300 - ROW_CACHE_CAPACITY),
      "a row of the first 44 ids was kept",
    );
  });

  it("keeps no rows with a budget of 0, whichever path a row took", async () => {
    const rig = pleFixture(spec);
    const ple = pleOf(rig, spec, 0);
    const ids = Array.from({ length: 40 }, (_row, n) => n);
    await ple.gather(ids);
    await ple.gather(ids);
    assertEquals(ple.stats().residentBytes, 0);
    assertEquals(ple.stats().rowReads, 4 * ids.length, "budget 0 kept a row");
  });
});

describe("Gemma4Ple row cache: abort, failed reads and dispose", () => {
  // One block per table of 64 rows (256 B each): 32 ids reach the whole-read threshold.
  const spec: PleFixtureSpec = { tokens: 64, layers: 1, dim: 4, embedScale: 4 };
  const tableBlockBytes = spec.tokens * 4;
  const ids = Array.from({ length: 32 }, (_row, n) => n);

  it("keeps the rows of a whole-block gather (control for the abort case)", async () => {
    const rig = pleFixture(spec, { defer: true });
    const ple = pleOf(rig, spec);
    await ple.gather(ids);
    assertEquals(ple.stats().residentBytes, 2 * tableBlockBytes + ids.length * rowBytesOf(spec));
  });

  it("keeps no row from a gather aborted while its whole reads were in flight", async () => {
    const rig = pleFixture(spec, { defer: true });
    const ple = pleOf(rig, spec);
    const controller = new AbortController();
    const pending = ple.gather(ids, { signal: controller.signal });
    controller.abort(new Error("test abort"));
    // A whole read cannot be stopped once started (abort is best-effort, between steps), so this
    // gather still returns its values — but it must not keep rows.
    await pending;
    assertEquals(ple.stats().resident, 2);
    assertEquals(ple.stats().residentBytes, 2 * tableBlockBytes, "rows were kept after the abort");
  });

  it("keeps no row when the other table's row read fails, while the whole block stays", async () => {
    // A budget of one block: values is read whole, scales falls to row reads — which fail.
    const control = pleOf(pleFixture(spec), spec, tableBlockBytes);
    await control.gather(ids);
    assertEquals(control.stats().loads, 1, "the control must read values whole");
    assertEquals(
      control.stats().residentBytes,
      tableBlockBytes + ids.length * rowBytesOf(spec),
      "control: rows from a values block and scales row reads are kept",
    );

    const rig = pleFixture(spec);
    const failing = (asset: string): AssetReader => {
      const reader = rig.openBlock(asset);
      if (!asset.startsWith("ple.scales.")) return reader;
      return { ...reader, read: () => Promise.reject(new Error("injected scales read failure")) };
    };
    const ple = pleOf(rig, spec, tableBlockBytes, failing);
    await assertRejects(() => ple.gather(ids), Error, "injected scales read failure");
    assertEquals(ple.stats().resident, 1);
    assertEquals(ple.stats().residentBytes, tableBlockBytes, "rows were kept after a failed read");
  });

  it("empties the row cache on dispose and keeps nothing a late read returns", async () => {
    const rig = pleFixture(spec);
    const ple = pleOf(rig, spec);
    await ple.gather([1, 2, 3]);
    assertEquals(ple.stats().residentBytes, 3 * rowBytesOf(spec));
    ple.dispose();
    assertEquals(ple.stats().residentBytes, 0);

    const deferred = pleFixture(spec, { defer: true });
    const late = pleOf(deferred, spec);
    const pending = late.gather([1, 2, 3]);
    late.dispose();
    await pending;
    assertEquals(late.stats().residentBytes, 0, "a gather finishing after dispose kept rows");
  });
});
