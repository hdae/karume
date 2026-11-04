/**
 * PLE index schema 4 (one f32 scale per `groupSize` consecutive values of a (token, layer) row —
 * the form of Gemma 4's q4_0 QAT line; writer: the exporter's `karume/ple.py`).
 *
 * The cases:
 * ① the accepted set: schema 4 needs a valid `groupSize`, schema 3 may not spell one, and the scales
 *   row length follows the group count
 * ② the host gather (row reads and whole blocks) equals the fixture's TS reference decode bit for
 *   bit, and a decode with one scale per row would not (the grouping is observable)
 * ③ the GPU-resident container binds the scale table as the i4 group scale matrix (no GPU), and a
 *   grouped i8 / i2 index is refused before a block byte is read
 * ④ the GPU-resident gather equals the reference and the host gather bit for bit (real GPU; SKIPs
 *   explicitly without an adapter)
 */

import { describe, it, test } from "vite-plus/test";
import { assert, assertEquals, assertRejects, assertThrows } from "@std/assert";
import { acquireGpu, prepareContainer, type Tensor } from "@karume/runtime";
import { createGemma4Ple } from "../src/gemma/ple.ts";
import { buildGemma4PleGatherContainer, createGemma4PleResident } from "../src/gemma/ple-gpu.ts";
import { gemma4PleTotalBytes, parseGemma4PleIndex, SCALE_BYTES } from "../src/gemma/ple-index.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";
import {
  expectedValue,
  pleFixture,
  type PleFixtureSpec,
  scaleOf,
  storedValue,
} from "./helpers/ple-fixture.ts";

/**
 * A grouped table in the real line's shape, scaled down: four groups per layer, and `values` /
 * `scales` blocks with different token boundaries (a reader that looks a scale up by the values
 * block would read another token's steps).
 */
const SPEC: PleFixtureSpec = {
  storage: "i4",
  tokens: 7,
  layers: 3,
  dim: 64,
  groupSize: 16,
  embedScale: 16,
  valueRows: 3,
  scaleRows: 2,
};

/** Every token twice, out of order (duplicates take the copy path of the gather). */
const IDS = [6, 0, 3, 3, 1, 5, 2, 4, 0, 6];

/** The fixture's reference decode of `ids` (`[1, ids, layers, dim]` in row-major order). */
const referenceOf = (spec: PleFixtureSpec, ids: readonly number[]): Float32Array => {
  const out = new Float32Array(ids.length * spec.layers * spec.dim);
  let cursor = 0;
  for (const id of ids) {
    for (let layer = 0; layer < spec.layers; layer += 1) {
      for (let column = 0; column < spec.dim; column += 1) {
        out[cursor] = expectedValue(spec, id, layer, column);
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

describe("parseGemma4PleIndex: schema 4", () => {
  const { document } = pleFixture(SPEC);

  it("accepts groupSize and derives the scales row from the group count", () => {
    assertEquals(document.schema, 4);
    const index = parseGemma4PleIndex(document);
    assertEquals(index.groupSize, SPEC.groupSize);
    assertEquals(index.scales.rowBytes, SPEC.layers * (SPEC.dim / 16) * SCALE_BYTES);
  });

  it("refuses schema 4 without groupSize", () => {
    const { groupSize: _dropped, ...withoutGroupSize } = document;
    assertThrows(
      () => parseGemma4PleIndex(withoutGroupSize),
      Error,
      "groupSize undefined is not a positive integer",
    );
  });

  it("refuses an invalid groupSize", () => {
    for (const groupSize of [0, -16, 1.5, "16", null]) {
      assertThrows(
        () => parseGemma4PleIndex({ ...document, groupSize }),
        Error,
        "is not a positive integer",
      );
    }
    assertThrows(
      () => parseGemma4PleIndex({ ...document, groupSize: 24 }),
      Error,
      "groupSize 24 does not divide dim 64",
    );
  });

  it("refuses a scales row length of another group count", () => {
    // groupSize 32 keeps the document valid in itself but halves the scales row the index states.
    assertThrows(
      () => parseGemma4PleIndex({ ...document, groupSize: 32 }),
      Error,
      "rowBytes 48 differs from 24",
    );
  });

  it("schema 3 stays without groupSize (one scale per row) and may not spell one", () => {
    const { groupSize: _dropped, ...rowSpec } = SPEC;
    const schema3 = pleFixture(rowSpec).document;
    assertEquals(schema3.schema, 3);
    assertEquals(Object.hasOwn(schema3, "groupSize"), false);
    const index = parseGemma4PleIndex(schema3);
    // The accepted form carries the row as the one group.
    assertEquals(index.groupSize, SPEC.dim);
    assertEquals(index.scales.rowBytes, SPEC.layers * SCALE_BYTES);
    assertThrows(
      () => parseGemma4PleIndex({ ...schema3, groupSize: SPEC.dim }),
      Error,
      "unknown key 'groupSize'",
    );
  });
});

describe("createGemma4Ple: a grouped index applies each group's own scale", () => {
  for (const storage of ["i4", "i8", "i2"] as const) {
    it(`${storage}: row reads and whole blocks equal the reference decode`, async () => {
      const spec = { ...SPEC, storage };
      const rig = pleFixture(spec);
      const expected = bitsOf(referenceOf(spec, IDS));
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
          // The second gather is answered from the row cache (the copied rows).
          assertEquals(
            bitsOf(dataOf(await ple.gather(IDS))),
            expected,
            `budget ${maxResidentBytes}, row cache`,
          );
        } finally {
          ple.dispose();
        }
      }
    });
  }

  it("the reference differs from a decode with one scale per row (the grouping is observable)", () => {
    let differing = 0;
    for (let id = 0; id < SPEC.tokens; id += 1) {
      for (let layer = 0; layer < SPEC.layers; layer += 1) {
        for (let column = 0; column < SPEC.dim; column += 1) {
          const rowScaled =
            Math.fround(storedValue("i4", id, layer, column) * scaleOf(id, layer)) *
            SPEC.embedScale;
          if (rowScaled !== expectedValue(SPEC, id, layer, column)) differing += 1;
        }
      }
    }
    assert(differing > 0, "every value decodes the same with the row's first scale");
  });
});

describe("buildGemma4PleGatherContainer: a grouped index (no GPU)", () => {
  it("binds the scale table as the i4 group scale matrix", async () => {
    const rig = pleFixture(SPEC);
    const bound = await buildGemma4PleGatherContainer(rig.index, rig.openBlock, "T");
    const encoding = bound.graphs.ple.supplies.get("ple")?.encoding;
    assertEquals(encoding?.codec, "int4-sym-g");
    assertEquals(encoding?.groupSize, SPEC.groupSize);
    assertEquals(prepareContainer(bound, "ple").graph.initializers.ple.storage, {
      codec: "int4-sym-g",
      groupSize: SPEC.groupSize,
      rowAxis: 0,
    });
  });

  for (const storage of ["i8", "i2"] as const) {
    it(`refuses a grouped ${storage} index before reading a block`, async () => {
      const rig = pleFixture({ ...SPEC, storage });
      await assertRejects(
        () => buildGemma4PleGatherContainer(rig.index, rig.openBlock, "T"),
        Error,
        'pleResidency: "gpu" needs storage i4',
      );
      assertEquals(rig.reads, [], "a block byte was read before the refusal");
    });
  }
});

test.skipIf(!GPU_AVAILABLE)(
  "createGemma4PleResident: a grouped i4 index gathers the reference and the host's bits (real GPU)",
  async () => {
    const rig = pleFixture(SPEC);
    const expected = bitsOf(referenceOf(SPEC, IDS));
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
        const batch = await gpu.beginBatch();
        let output;
        try {
          output = await resident.enqueue(batch, Int32Array.from(IDS));
        } finally {
          await batch.finish();
        }
        const actual = bitsOf(new Float32Array(await output.read()));
        assertEquals(actual, expected, "GPU gather against the reference decode");
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
