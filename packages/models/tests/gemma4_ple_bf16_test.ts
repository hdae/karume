/**
 * PLE index schema 5 — a bf16 `values` table with no `scales` (the original-weights form of Gemma
 * 4's BF16 checkpoints; writer: the exporter's `karume/ple.py`, ADR 0085 addendum 2026-11-04).
 *
 * The cases:
 * ① the accepted set: schema 5 needs storage bf16 and spells neither `scales` nor `groupSize`, its
 *   `values` row is 2 bytes per value, and bf16 is not a schema 3 storage
 * ② the cross-check against the container's assets: a stray scale block or a missing values block
 *   fails before a block byte is read
 * ③ the host gather (row reads, whole blocks, the row cache) equals the fixture's reference bit for
 *   bit, reading the one table only
 * ④ the GPU-resident container binds `values` with the runtime's `bf16` codec and no scale (no GPU)
 * ⑤ the GPU-resident gather equals the reference and the host gather bit for bit (real GPU; SKIPs
 *   explicitly without an adapter)
 */

import { describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { acquireGpu, prepareContainer, type Tensor } from "@karume/runtime";
import { createGemma4Ple } from "../src/gemma/ple.ts";
import {
  buildGemma4PleGatherContainer,
  createGemma4PleResident,
  gemma4PleGpuBytes,
} from "../src/gemma/ple-gpu.ts";
import {
  gemma4PleTotalBytes,
  parseGemma4PleIndex,
  readGemma4PleIndex,
} from "../src/gemma/ple-index.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  expectedBf16Value,
  pleBf16Fixture,
  type PleBf16FixtureSpec,
  pleComponentOf,
  pleFixture,
} from "./helpers/ple-fixture.ts";

/**
 * A bf16 table scaled down: three layers of 8 values (one layer = 16 bytes, 4-aligned for the GPU
 * row), blocks of 3 tokens with a short last block.
 */
const SPEC: PleBf16FixtureSpec = {
  tokens: 7,
  layers: 3,
  dim: 8,
  embedScale: 16,
  valueRows: 3,
};

/** Bytes of one token row (`layers × dim × 2`). */
const ROW_BYTES = SPEC.layers * SPEC.dim * 2;

/** Every token twice, out of order (duplicates take the copy path of the gather). */
const IDS = [6, 0, 3, 3, 1, 5, 2, 4, 0, 6];

/** The fixture's reference gather of `ids` (`[1, ids, layers, dim]` in row-major order). */
const referenceOf = (ids: readonly number[]): Float32Array => {
  const out = new Float32Array(ids.length * SPEC.layers * SPEC.dim);
  let cursor = 0;
  for (const id of ids) {
    for (let layer = 0; layer < SPEC.layers; layer += 1) {
      for (let column = 0; column < SPEC.dim; column += 1) {
        out[cursor] = expectedBf16Value(SPEC, id, layer, column);
        cursor += 1;
      }
    }
  }
  return out;
};

/** u32 words (f32 `===` misses NaN and ±0). */
const bitsOf = (values: Float32Array): number[] => [
  ...new Uint32Array(values.buffer, values.byteOffset, values.length),
];

const dataOf = (tensor: Tensor): Float32Array => {
  assert(tensor.dtype === "f32", `dtype ${tensor.dtype}`);
  return tensor.data;
};

describe("parseGemma4PleIndex: schema 5", () => {
  const { document } = pleBf16Fixture(SPEC);

  it("accepts a bf16 table with a values row of 2 bytes per value and no scales", () => {
    assertEquals(document.schema, 5);
    const index = parseGemma4PleIndex(document);
    assert(index.storage === "bf16", `storage ${index.storage}`);
    assertEquals(index.values.rowBytes, ROW_BYTES);
    assertEquals(Object.hasOwn(index, "scales"), false);
    assertEquals(Object.hasOwn(index, "groupSize"), false);
    assertEquals(gemma4PleTotalBytes(index), SPEC.tokens * ROW_BYTES);
  });

  it("refuses a scale table or a group size", () => {
    const quantized = pleFixture({ ...SPEC, storage: "i8" }).document;
    assertThrows(
      () => parseGemma4PleIndex({ ...document, scales: quantized.scales }),
      Error,
      "unknown key 'scales'",
    );
    assertThrows(
      () => parseGemma4PleIndex({ ...document, groupSize: SPEC.dim }),
      Error,
      "unknown key 'groupSize'",
    );
  });

  it("needs storage bf16 spelled", () => {
    const { storage: _dropped, ...withoutStorage } = document;
    assertThrows(
      () => parseGemma4PleIndex(withoutStorage),
      Error,
      "ple_index.storage undefined is not bf16",
    );
    assertThrows(
      () => parseGemma4PleIndex({ ...document, storage: "i8" }),
      Error,
      "ple_index.storage i8 is not bf16",
    );
  });

  it("refuses a values row of another width", () => {
    const values = document.values as { rowBytes: number };
    assertThrows(
      () => parseGemma4PleIndex({ ...document, values: { ...values, rowBytes: ROW_BYTES / 2 } }),
      Error,
      `rowBytes ${ROW_BYTES / 2} differs from ${ROW_BYTES}`,
    );
  });

  it("bf16 is not a schema 3 storage (schema 3 carries a scale table)", () => {
    const schema3 = pleFixture({ ...SPEC, storage: "i8" }).document;
    assertThrows(
      () => parseGemma4PleIndex({ ...schema3, storage: "bf16" }),
      Error,
      "ple_index.storage bf16 is not i8 / i4 / i2",
    );
  });
});

describe("readGemma4PleIndex: a bf16 index against the container's assets", () => {
  it("reads an index whose values blocks are all declared", async () => {
    const rig = pleBf16Fixture(SPEC);
    const index = await readGemma4PleIndex("test", pleComponentOf(rig));
    assertEquals(index.storage, "bf16");
    assertEquals(rig.reads, [], "a block byte was read by the cross-check");
  });

  it("refuses a scale block the index does not point at", async () => {
    const rig = pleBf16Fixture(SPEC);
    await assertRejects(
      () =>
        readGemma4PleIndex(
          "test",
          pleComponentOf(rig, { "ple.scales.0": { role: "ple-scales", length: 4 } }),
        ),
      Error,
      "ple.scales.0: an asset with role 'ple-scales' that the index does not point at",
    );
  });

  it("refuses a missing values block", async () => {
    const rig = pleBf16Fixture(SPEC);
    await assertRejects(
      () => readGemma4PleIndex("test", pleComponentOf(rig, { "ple.values.1": undefined })),
      Error,
      "ple.values.1: not declared by the container",
    );
  });
});

describe("createGemma4Ple: a bf16 index", () => {
  it("row reads, whole blocks and the row cache equal the reference bit for bit", async () => {
    const rig = pleBf16Fixture(SPEC);
    const expected = bitsOf(referenceOf(IDS));
    const distinct = new Set(IDS).size;
    for (const maxResidentBytes of [0, gemma4PleTotalBytes(rig.index)]) {
      const ple = createGemma4Ple({
        index: rig.index,
        openBlock: rig.openBlock,
        vocabSize: SPEC.tokens,
        maxResidentBytes,
      });
      try {
        const first = await ple.gather(IDS);
        assertEquals(first.shape, [1, IDS.length, SPEC.layers, SPEC.dim]);
        assertEquals(bitsOf(dataOf(first)), expected, `budget ${maxResidentBytes}`);
        const stats = ple.stats();
        if (maxResidentBytes === 0) {
          // One table: one row read per distinct id (a quantized index reads two).
          assertEquals(stats.rowReads, distinct, "row reads");
          assertEquals(stats.loads, 0, "whole reads with a budget of 0");
        } else {
          assertEquals(stats.loads, rig.index.values.blocks.length, "whole reads");
          assertEquals(stats.residentBytes, gemma4PleTotalBytes(rig.index) + distinct * ROW_BYTES);
          // The second gather is answered from the row cache (rows copied out of the blocks).
          assertEquals(bitsOf(dataOf(await ple.gather(IDS))), expected, "row cache");
          assertEquals(ple.stats().loads, rig.index.values.blocks.length, "a block was re-read");
        }
        assert(
          rig.reads.every((read) => read.asset.startsWith("ple.values.")),
          "a read touched a table other than values",
        );
      } finally {
        ple.dispose();
      }
    }
  });

  it("the reference spans both signs and several exponents (a dropped bit would show)", () => {
    const values = referenceOf(IDS);
    assert(values.some((value) => value < 0) && values.some((value) => value > 0), "one sign only");
    const exponents = new Set(
      [...new Uint32Array(values.buffer)].map((bits) => (bits >>> 23) & 0xff),
    );
    assert(exponents.size >= 6, `${exponents.size} exponents`);
  });
});

describe("buildGemma4PleGatherContainer: a bf16 index (no GPU)", () => {
  it("binds values with the bf16 codec and no scale", async () => {
    const rig = pleBf16Fixture(SPEC);
    const bound = await buildGemma4PleGatherContainer(rig.index, rig.openBlock, "T");
    const supply = bound.graphs.ple.supplies.get("ple");
    assertEquals(supply?.encoding.codec, "bf16");
    assertEquals(supply?.encoding.scale, undefined);
    assertEquals(prepareContainer(bound, "ple").graph.initializers.ple.storage, {
      codec: "bf16",
    });
    assertEquals(gemma4PleGpuBytes(rig.index), {
      values: SPEC.tokens * ROW_BYTES,
      scales: 0,
      total: SPEC.tokens * ROW_BYTES,
    });
  });
});

test.skipIf(!GPU_AVAILABLE)(
  "createGemma4PleResident: a bf16 index gathers the reference and the host's bits (real GPU)",
  async () => {
    const rig = pleBf16Fixture(SPEC);
    const expected = bitsOf(referenceOf(IDS));
    const gpu = await acquireGpu();
    const host = createGemma4Ple({
      index: rig.index,
      openBlock: rig.openBlock,
      vocabSize: SPEC.tokens,
      maxResidentBytes: 0,
    });
    try {
      const resident = await createGemma4PleResident({
        gpu,
        index: rig.index,
        openBlock: rig.openBlock,
        vocabSize: SPEC.tokens,
        inputName: "per_layer_inputs",
        idsName: "input_ids",
        rows: [IDS.length],
        entry: "T",
      });
      try {
        assertEquals(resident.residentBytes, SPEC.tokens * ROW_BYTES);
        const batch = await gpu.beginBatch();
        let output;
        try {
          output = await resident.enqueue(batch, Int32Array.from(IDS));
        } finally {
          await batch.finish();
        }
        const actual = bitsOf(new Float32Array(await output.read()));
        assertEquals(actual, expected, "GPU gather against the reference");
        assertEquals(actual, bitsOf(dataOf(await host.gather(IDS))), "GPU gather against the host");
      } finally {
        await resident.dispose();
      }
    } finally {
      host.dispose();
      gpu.destroy();
    }
  },
);
