// Behaviour of the style / speaker **table lookup** (`src/sbv2/style.ts`). Written with small
// synthetic tables only; depends on neither real assets nor a GPU.
//
// The design starts from this being a nest of silent wrong values (ADR 0039 decision 4): both style
// IDs and speaker IDs are **the table's physical rows**, so a one-row shift keeps the shapes valid,
// loading and running succeed, and a different style's or speaker's voice comes out with no
// exception. Three behaviours are pinned:
//
//  ① The row and column counts are decided by **the actual table** (not baked into the
//   implementation)
//  ② `mean` in the mixing formula `mean + (picked − mean)·weight` is **row 0** (not the per-column
//   mean)
//  ③ A mismatch between the table and the row index or weight fails loudly
//
// NOTE: only the comparison "pipelineConfig's `styles` count equals the table's row count" cannot
// be written here — the check (`assertTableFits` in `pipeline.ts`) needs the static dimensions of
// the graph inputs, so it needs real assets and a GPU. `e2e_sbv2_wav_test.ts` holds it.

import { assert, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import {
  assertFiniteStyleWeight,
  parseSbv2Table,
  SPEAKER_TENSOR,
  speakerEmbedding,
  STYLE_TENSOR,
  styleVector,
} from "../src/sbv2/style.ts";
import { ModelInputError } from "../src/errors.ts";
import { InputError } from "../src/sbv2/errors.ts";
import { buildSafetensors, f32Bytes } from "./helpers/safetensors.ts";

const COLS = 3;
/**
 * A synthetic table of 4 styles × 3 columns. **Row 0 is the mean style** (the SBV2 convention) and
 * is **deliberately** different from the per-column means (1.5 / 7 / 14.5) — so that the mixing
 * formula's fault injection can tell apart an implementation that recomputes the mean from the
 * columns.
 */
const STYLE_ROWS: readonly (readonly number[])[] = [
  [0, 10, 20],
  [2, 14, 26],
  [-4, 6, 12],
  [8, -2, 0],
];
const STYLE_TABLE = Float32Array.from(STYLE_ROWS.flat());

/**
 * 2 speakers × 4 columns (a column count different from the style table, so a mix-up is
 * detectable).
 */
const SPEAKER_ROWS: readonly (readonly number[])[] = [
  [1, 2, 3, 4],
  [-1, -2, -3, -4],
];
const SPEAKER_TABLE = Float32Array.from(SPEAKER_ROWS.flat());

const styleTableAsset = (
  rows: readonly (readonly number[])[],
  name: string = STYLE_TENSOR,
): ArrayBuffer =>
  buildSafetensors([
    {
      name,
      dtype: "F32",
      shape: [rows.length, rows[0].length],
      data: f32Bytes(rows.flat()),
    },
  ]);

describe("reading the style table: the actual safetensors decide the row and column counts", () => {
  it("reads the shape as [rows, columns] as is, with values in row-major order", () => {
    const table = parseSbv2Table(styleTableAsset(STYLE_ROWS), STYLE_TENSOR);
    assertEquals(table.rows, 4);
    assertEquals(table.cols, COLS);
    assertEquals([...table.data], STYLE_ROWS.flat());
  });

  it("a table with a different row count is read accordingly (the table is not baked in)", () => {
    // A different ckpt changes the number and order of styles (ADR 0039 decision 3). Holding the
    // count in the implementation gives "a different style with valid shapes", so this pins that
    // only the actual table is authoritative.
    const table = parseSbv2Table(styleTableAsset(STYLE_ROWS.slice(0, 2)), STYLE_TENSOR);
    assertEquals(table.rows, 2);
    assertEquals(table.cols, COLS);
    assertEquals([...table.data], STYLE_ROWS.slice(0, 2).flat());
  });

  it("the speaker table is read by the same reader (only the tensor name differs)", () => {
    const table = parseSbv2Table(styleTableAsset(SPEAKER_ROWS, SPEAKER_TENSOR), SPEAKER_TENSOR);
    assertEquals([table.rows, table.cols], [2, 4]);
    assertEquals([...table.data], SPEAKER_ROWS.flat());
  });

  it("fails, listing what is there, when the requested tensor is missing", () => {
    // An asset mix-up (reading the style asset as the speaker asset, for example) can only be
    // detected by name.
    assertThrows(
      () => parseSbv2Table(styleTableAsset(STYLE_ROWS), SPEAKER_TENSOR),
      Error,
      `The asset has no tensor '${SPEAKER_TENSOR}'`,
    );
  });

  it("rejects a table that is not F32 (a different byte interpretation makes every row something else)", () => {
    const buffer = buildSafetensors([
      {
        name: STYLE_TENSOR,
        dtype: "F16",
        shape: [4, COLS],
        data: new Uint8Array(4 * COLS * 2),
      },
    ]);
    assertThrows(() => parseSbv2Table(buffer, STYLE_TENSOR), Error, "is not F32");
  });

  it("rejects a table that is not 2-D (the row slice width is undefined)", () => {
    const buffer = buildSafetensors([
      {
        name: STYLE_TENSOR,
        dtype: "F32",
        shape: [4 * COLS],
        data: f32Bytes(STYLE_TABLE),
      },
    ]);
    assertThrows(() => parseSbv2Table(buffer, STYLE_TENSOR), Error, "not the 2-D");
  });
});

describe("style mixing mean + (picked − mean)·weight: mean is row 0", () => {
  const mix = (index: number, weight: number): number[] => [
    ...styleVector(STYLE_TABLE, STYLE_ROWS.length, COLS, index, weight).data,
  ];

  it("weight 1 gives exactly the chosen row (an f32 tensor of [1, cols])", () => {
    const tensor = styleVector(STYLE_TABLE, STYLE_ROWS.length, COLS, 2, 1);
    assertEquals(tensor.dtype, "f32");
    assertEquals(tensor.shape, [1, COLS]);
    assertEquals([...tensor.data], STYLE_ROWS[2]);
  });

  it("weight 0 gives the mean row (row 0) whichever row is chosen", () => {
    for (let index = 0; index < STYLE_ROWS.length; index += 1) {
      assertEquals(mix(index, 0), STYLE_ROWS[0], `row ${index}`);
    }
  });

  it("an intermediate weight interpolates linearly between row 0 and the chosen row", () => {
    // The midpoint of row 0 [0,10,20] and row 1 [2,14,26].
    assertEquals(mix(1, 0.5), [1, 12, 23]);
    // A negative row (on the opposite side as seen from row 0) follows the same formula.
    assertEquals(mix(2, 0.5), [-2, 8, 16]);
  });

  it("weight > 1 extrapolates (the implementation does not cap the strength)", () => {
    assertEquals(mix(1, 2), [4, 18, 32]);
  });

  it("fault injection: differs from an implementation that recomputes the mean per column", () => {
    // The column means are [1.5, 7, 14.5]. The difference from the implementation that reads row 0
    // as the mean shows only here (at the two ends, weight 1 and weight 0, both implementations
    // agree).
    const columnMeans = Array.from(
      { length: COLS },
      (_, col) => STYLE_ROWS.reduce((sum, row) => sum + row[col], 0) / STYLE_ROWS.length,
    );
    const byColumnMean = columnMeans.map((mean, col) => mean + (STYLE_ROWS[1][col] - mean) * 0.5);
    assertEquals(mix(1, 0.5), [1, 12, 23]);
    assertNotEquals(mix(1, 0.5), byColumnMean);
  });

  it("choosing row 0 itself gives row 0 for any weight (picked = mean)", () => {
    for (const weight of [0, 0.3, 1, 2]) {
      assertEquals(mix(0, weight), STYLE_ROWS[0], `weight ${weight}`);
    }
  });
});

describe("speaker embedding: looks up a row of the table and makes it [1, cols, 1]", () => {
  const pick = (index: number) => speakerEmbedding(SPEAKER_TABLE, SPEAKER_ROWS.length, 4, index);

  it("returns the chosen row as is (the shape of g that front / voice take)", () => {
    const tensor = pick(1);
    assertEquals(tensor.dtype, "f32");
    assertEquals(tensor.shape, [1, 4, 1]);
    assertEquals([...tensor.data], SPEAKER_ROWS[1]);
  });

  it("a one-row shift gives a different voice (with valid shapes)", () => {
    const [first, second] = [pick(0), pick(1)];
    assertEquals(first.shape, second.shape, "not detectable by shape");
    assert(
      [...first.data].some((value, at) => value !== second.data[at]),
      "the same vector comes out for a different row",
    );
  });

  it("returns a copy of the table (writing to the result leaves the table unchanged)", () => {
    // The table is reused for the pipeline's lifetime, so returning a view would let one generation
    // corrupt every later one.
    const tensor = pick(0);
    tensor.data[0] = 999;
    assertEquals([...SPEAKER_TABLE], SPEAKER_ROWS.flat());
  });
});

describe("a mismatch between row index, weight and table fails loudly", () => {
  const rows = STYLE_ROWS.length;

  it("rejects an out-of-range row index (for both styles and speakers)", () => {
    assertThrows(
      () => styleVector(STYLE_TABLE, rows, COLS, rows, 1),
      Error,
      `Row index ${rows} is outside the table (0..${rows - 1})`,
    );
    assertThrows(() => styleVector(STYLE_TABLE, rows, COLS, -1, 1), Error, "outside the table");
    assertThrows(
      () => speakerEmbedding(SPEAKER_TABLE, SPEAKER_ROWS.length, 4, 2),
      Error,
      "outside the table",
    );
  });

  it("rejects a non-integer row index (the slice would start in the middle of a row)", () => {
    assertThrows(() => styleVector(STYLE_TABLE, rows, COLS, 1.5, 1), Error, "outside the table");
  });

  it("rejects a table whose element count differs from rows × columns", () => {
    // With only the row count wrong, even an in-range row index reads outside the table.
    assertThrows(
      () => styleVector(STYLE_TABLE, rows + 1, COLS, 4, 1),
      Error,
      `Table element count ${STYLE_TABLE.length} differs from ${rows + 1}×${COLS}`,
    );
  });

  it("rejects a non-finite weight (NaN would propagate to the waveform)", () => {
    for (const weight of [Number.NaN, Number.POSITIVE_INFINITY]) {
      // The type is `InputError` (400-like) — the request passes once the caller fixes the weight,
      // so it branches differently from the row-index and table mismatches the same function throws
      // (plain `Error` = 500).
      assertThrows(
        () => styleVector(STYLE_TABLE, rows, COLS, 1, weight),
        InputError,
        "not a finite number",
      );
    }
  });
});

describe("accepted values of styleWeight", () => {
  const rows = STYLE_ROWS.length;

  it("the table lookup and the leaf-side check give the same type and the same diagnostic", () => {
    const viaGate = assertThrows(
      () => assertFiniteStyleWeight(Number.NaN),
      InputError,
      "styleWeight NaN is not a finite number",
    );
    const viaTable = assertThrows(
      () => styleVector(STYLE_TABLE, rows, COLS, 1, Number.NaN),
      InputError,
    );
    // What can be observed is only that the two paths give the same diagnostic (an implementation
    // that copied the condition and the message would also pass this assert, so it is no proof that
    // both go through the same function). Still, if a copy creeps in and one side loosens or its
    // message drifts, this fails.
    assertEquals(viaTable.message, viaGate.message);
    assertFiniteStyleWeight(0);
    assertFiniteStyleWeight(-1.5);
  });

  it("is an input error, so it is caught by the cross-family ModelInputError", () => {
    const error = assertThrows(() => assertFiniteStyleWeight(Number.POSITIVE_INFINITY));
    assert(error instanceof ModelInputError, "a rejected styleWeight is 400-like");
  });
});
