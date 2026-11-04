/**
 * Comparison of fixed packed PLE (i2 / i4) with an independent Torch CPU fixture (ADR 0097).
 *
 * The assets are an in-repository fixture, a `krm` container of **exactly the same shape as the series output**
 * (the part list `model-NNNNN-of-NNNNN.krm` + the assets `ple_index` / `ple.values.*` / `ple.scales.*`). The
 * generator is `tools/export-recipes/gemma4/tests/ple_fixture.py`, written by the same functions as the recipe's
 * production path (`karume.ple.ple_assets` + `karume.pipeline.publish_model`). The reader is also the single path
 * used for series output (`seriesPleHandle` in `helpers/ple-series.ts`), so this check reads **the shipped shape**,
 * not a "fixture-only reader".
 *
 * Five checks:
 * ① The values of **row reads** (budget 0) are **bit-identical** to torch's CPU reference
 * ② The values of **full reads** (budget = everything) are bit-identical to ①, and the resident blocks and
 *    residentBytes match the index
 * ③ If the index's storage declaration disagrees with the data, the values diverge (a combination where only one
 *    side was rebuilt)
 * ④ The index's accepted set (schema 3) — an old version, a misspelt storage, or a dimension not divisible by the
 *    packing count is not read
 * ⑤ The comparison of the index and the container's assets **in both directions** (`readGemma4PleIndex`) rejects a
 *    container missing either side
 *
 * NOTE: `values` and `scales` have **separate block boundaries** (the generator cuts at `BLOCK_BYTES = 144`, so 3 : 1
 * for i4 and 2 : 1 for i2). A fixture with aligned boundaries cannot catch a reader that wrongly reads a different
 * block per table. ②'s assertion on the block count is derived from the index (it may move on a rebuild), but "the
 * boundaries are not aligned" itself gets a named assertion below.
 */

import { readFile } from "@karume/test-support/fs";
import { test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { type OpenedContainer, parseSafetensors } from "@karume/runtime";
import { openSeriesContainer } from "../../runtime/tests/helpers/container-files.ts";
import { createGemma4Ple } from "../src/gemma/ple.ts";
import {
  gemma4PleAssetSource,
  gemma4PleTotalBytes,
  parseGemma4PleIndex,
  PLE_INDEX_ASSET,
  readGemma4PleIndex,
} from "../src/gemma/ple-index.ts";
import { SERIES_MODEL_FILE, seriesPleHandle } from "./helpers/ple-series.ts";

/** Reads the index asset raw (the accepted-set checks pass doctored documents to `parseGemma4PleIndex`). */
const readIndexDocument = async (
  opened: Pick<OpenedContainer, "asset">,
): Promise<Record<string, unknown>> => {
  const reader = opened.asset(PLE_INDEX_ASSET);
  return JSON.parse(new TextDecoder().decode(await reader.read(0, reader.length)));
};

for (const dtype of ["i2", "i4"] as const) {
  test(`PLE ${dtype}: row reads and full reads are bit-identical to the independent CPU reference`, async () => {
    const dir = new URL(`./fixtures/gemma4-ple-packed/${dtype}/`, import.meta.url);
    const opened = await openSeriesContainer(new URL(SERIES_MODEL_FILE, dir));
    const { index, openBlock } = await seriesPleHandle(opened, `test: ${dtype} fixture`);
    const oracle = parseSafetensors((await readFile(new URL("oracle.safetensors", dir))).buffer);
    const iv = oracle.tensors.get("ids")!,
      ev = oracle.tensors.get("expected")!;
    const ids = Array.from(new Int32Array(oracle.buffer, iv.byteOffset, iv.byteLength / 4));
    const expected = new Uint32Array(oracle.buffer, ev.byteOffset, ev.byteLength / 4);
    const distinct = new Set(ids).size;
    const factor = dtype === "i2" ? 4 : 2;
    assertEquals(index.storage, dtype);
    assert(index.storage !== "bf16"); // narrows to the quantized form (the line above pins which)
    assertEquals(index.values.rowBytes, (index.layers * index.dim) / factor);
    assertEquals(
      gemma4PleTotalBytes(index),
      index.tokens * (index.layers * (index.dim / factor + 4)),
    );
    // The boundaries differ per table (the NOTE in the module doc). With an aligned fixture, a reader that wrongly
    // reads scales by the values block number would slip through.
    assert(
      index.values.blocks.length > 1 && index.values.blocks.length !== index.scales.blocks.length,
      `values ${index.values.blocks.length} block / scales ${index.scales.blocks.length} block` +
        " — the fixture no longer exercises boundaries that differ per table",
    );

    // ① Row reads (budget 0 = no residency and no row cache).
    const rows = createGemma4Ple({
      index,
      openBlock,
      vocabSize: index.tokens,
      maxResidentBytes: 0,
    });
    const byRows = await rows.gather(ids);
    assert(byRows.dtype === "f32");
    assertEquals(new Uint32Array(byRows.data.buffer), expected);
    assertEquals(rows.stats().loads, 0, "reads whole blocks despite a budget of 0");
    assertEquals(
      rows.stats().rowReads,
      distinct * 2,
      "row reads cover the 2 tables values / scales",
    );
    rows.dispose();

    // ② Full reads (budget = everything — the referenced ids cover every token, so every block reaches the threshold).
    const blocks = createGemma4Ple({
      index,
      openBlock,
      vocabSize: index.tokens,
      maxResidentBytes: gemma4PleTotalBytes(index),
    });
    const byBlocks = await blocks.gather(ids);
    assert(byBlocks.dtype === "f32");
    assertEquals(
      [...new Uint32Array(byBlocks.data.buffer)],
      [...new Uint32Array(byRows.data.buffer)],
      "the full-read values are not bit-identical to the row reads",
    );
    const loaded = index.values.blocks.length + index.scales.blocks.length;
    assertEquals(blocks.stats().loads, loaded);
    assertEquals(blocks.stats().rowReads, 0);
    assertEquals(blocks.stats().resident, loaded);
    // Resident = every block + the row cache (the distinct rows looked up, in the 256-row headroom
    // outside the block budget — ADR 0085 addendum 2026-10-27).
    assertEquals(
      blocks.stats().residentBytes,
      gemma4PleTotalBytes(index) +
        Math.min(256, distinct) * (index.values.rowBytes + index.scales.rowBytes),
    );
    // They are resident, so reading again does not add reads.
    const again = await blocks.gather(ids);
    assertEquals(blocks.stats().loads, loaded, "re-reads a resident block");
    // The second gather is answered from the row cache (rows copied out of the packed blocks), so
    // a packed-specific copy offset error shows up here, not in the first gather.
    assert(again.dtype === "f32");
    assertEquals(
      new Uint32Array(again.data.buffer),
      expected,
      "a gather answered from the row cache differs from the reference",
    );
    blocks.dispose();
    assertEquals(blocks.stats().residentBytes, 0);
    await assertRejects(() => blocks.gather(ids));

    // ③ Swapping only the index's storage declaration gives different values with matching shape and dtype — they diverge from the reference.
    const mismatched = createGemma4Ple({
      index: { ...index, storage: dtype === "i2" ? "i4" : "i2" },
      openBlock,
      vocabSize: index.tokens,
      maxResidentBytes: 0,
    });
    try {
      const wrong = await mismatched.gather(ids);
      assert(wrong.dtype === "f32");
      assert(
        [...new Uint32Array(wrong.data.buffer)].some((word, at) => word !== expected[at]),
        "values do not change with a mixed-up storage declaration (the index's storage has no effect)",
      );
    } finally {
      mismatched.dispose();
    }

    // ④ The accepted set (the schema 3 check) — an old version, a misspelt storage, or a dimension not divisible by the packing count is not read.
    const raw = await readIndexDocument(opened);
    assertEquals(raw.schema, 3, "the container's index is not schema 3");
    for (const invalid of [{ schema: 1 }, { schema: 2 }, { storage: "i8" }, { dim: 31 }]) {
      assertThrows(() => parseGemma4PleIndex({ ...raw, ...invalid }));
    }
  });

  /**
   * ⑤ Comparison of the index and the container's assets (`readGemma4PleIndex` → `assertGemma4PleAssets`).
   *
   * The old sidecar reader did not apply this check (the test assembled the index itself, so "an asset the index
   * points at is missing from the container" could not happen by definition). Now that the container is read, a
   * distribution where only one side was rebuilt can be rejected **before a single byte of the weight blocks is
   * fetched**. Here a surface with thinned-out asset declarations is passed to confirm it is not always true.
   */
  test(`PLE ${dtype}: a mismatch between the index and the container's assets fails before reading`, async () => {
    const dir = new URL(`./fixtures/gemma4-ple-packed/${dtype}/`, import.meta.url);
    const opened = await openSeriesContainer(new URL(SERIES_MODEL_FILE, dir));
    const source = gemma4PleAssetSource(opened);
    const dropped = Object.keys(source.assets).find((name) => name.startsWith("ple.values."));
    assert(dropped !== undefined, "the container has no values asset");

    // (a) The container does not declare a block the index points at.
    const missing = { ...source.assets };
    delete missing[dropped];
    await assertRejects(
      () => readGemma4PleIndex("test: missing asset", { ...source, assets: missing }),
      Error,
      `${dropped}: not declared by the container`,
    );

    // (b) The container has a PLE asset the index does not point at (the shape where a declared asset is never read).
    await assertRejects(
      () =>
        readGemma4PleIndex("test: extra asset", {
          ...source,
          assets: { ...source.assets, "ple.values.999": "ple-values" },
        }),
      Error,
      "ple.values.999: an asset with role 'ple-values' that the index does not point at",
    );

    // (c) A different role (a rebuild that mixed up values and scales).
    await assertRejects(
      () =>
        readGemma4PleIndex("test: wrong role", {
          ...source,
          assets: { ...source.assets, [dropped]: "ple-scales" },
        }),
      Error,
      `${dropped}: role 'ple-scales'`,
    );
  });
}
