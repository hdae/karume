/**
 * The row-blocked parallel GEMV (M = 2..8 of the parallel family): how many rows a thread holds,
 * where the kernel exists, which shapes take it, its key and the structure of its WGSL. No GPU —
 * the word-for-word identity with the parallel GEMV is tests/gpu_linear_gemv_parallel_rows_test.ts.
 */

import { assert, assertEquals, assertStringIncludes, assertThrows } from "@std/assert";
import { describe, it } from "vite-plus/test";
import { CodegenError } from "../src/codegen/errors.ts";
import {
  assertGeometryProfile,
  DEFAULT_GEOMETRY_PROFILE,
} from "../src/kernels/geometry-profile.ts";
import {
  LINEAR_GEMV_PARALLEL_ROWS,
  type LinearGemvChoices,
  linearGemvParallelEligible,
  type LinearGemvParallelLanes,
  linearGemvParallelRowBlock,
  type LinearGemvParallelRows,
  linearGemvParallelRowsFits,
  linearGemvParallelRowsFor,
  linearGemvParallelRowsKey,
  linearGemvParallelRowsWgsl,
  type LinearGemvParallelShape,
  linearGemvParallelStagedKey,
  linearGemvParallelStagedWgsl,
  linearGemvParallelWgsl,
} from "../src/kernels/linear-gemv.ts";

const LANES: readonly LinearGemvParallelLanes[] = [2, 4, 8, 16, 32];

describe("linearGemvParallelRowsFor", () => {
  it("takes the largest of 2 / 4 / 8 rows that is at most M and at most the row's cap", () => {
    const table: readonly [number, LinearGemvParallelRows, number | undefined][] = [
      // The verify buckets under each cap
      [4, 2, 2],
      [4, 4, 4],
      [4, 8, 4],
      [8, 2, 2],
      [8, 4, 4],
      [8, 8, 8],
      // Row counts between the buckets: the last y tile holds fewer rows than the block
      [2, 8, 2],
      [3, 8, 2],
      [5, 8, 4],
      [7, 4, 4],
    ];
    for (const [m, cap, rows] of table) {
      assertEquals(linearGemvParallelRowsFor(m, cap), rows, `M=${m} cap ${cap}`);
    }
  });

  it("has no row block for one row (M = 1 keeps the decode kernel)", () => {
    for (const cap of LINEAR_GEMV_PARALLEL_ROWS) {
      assertEquals(linearGemvParallelRowsFor(1, cap), undefined);
    }
  });
});

describe("linearGemvParallelRowsFits", () => {
  it("holds the staged copies of every row and the partial sums in the workgroup storage WebGPU core guarantees", () => {
    const fits = (storage: "i2" | "i4" | "i8"): string[] =>
      LANES.flatMap((lanes) =>
        LINEAR_GEMV_PARALLEL_ROWS.filter((rows) =>
          linearGemvParallelRowsFits(storage, lanes, rows),
        ).map((rows) => `l${lanes}r${rows}`),
      );
    // i4: 8 quads a word. 32 lanes stage 4 KiB a row, so two rows are the most
    assertEquals(fits("i4"), [
      "l2r2",
      "l2r4",
      "l2r8",
      "l4r2",
      "l4r4",
      "l4r8",
      "l8r2",
      "l8r4",
      "l8r8",
      "l16r2",
      "l16r4",
      "l32r2",
    ]);
    // i8: 4 quads a word
    assertEquals(fits("i8"), [
      "l2r2",
      "l2r4",
      "l2r8",
      "l4r2",
      "l4r4",
      "l4r8",
      "l8r2",
      "l8r4",
      "l8r8",
      "l16r2",
      "l16r4",
      "l16r8",
      "l32r2",
      "l32r4",
    ]);
    // i2: 16 quads a word; 32 lanes stage 8 KiB a row, which leaves no room for a second
    assertEquals(fits("i2"), [
      "l2r2",
      "l2r4",
      "l2r8",
      "l4r2",
      "l4r4",
      "l4r8",
      "l8r2",
      "l8r4",
      "l16r2",
    ]);
  });

  it("has no kernel for a float storage or a row count that is not 2 / 4 / 8", () => {
    for (const storage of ["f32", "f16", "bf16"] as const) {
      assertEquals(linearGemvParallelRowsFits(storage, 4, 2), false, storage);
    }
    for (const rows of [0, 1, 3, 6, 16, 2.5, Number.NaN]) {
      assertEquals(linearGemvParallelRowsFits("i4", 4, rows), false, `${rows}`);
    }
  });
});

/** The default profile's GEMV choices. */
const GEMV = DEFAULT_GEOMETRY_PROFILE.linearGemv;

/** The default choices with one row in front (the lookup takes the first row that names a shape). */
const withRow = (row: LinearGemvParallelShape): LinearGemvChoices => ({
  ...GEMV,
  parallelShapes: [row, ...GEMV.parallelShapes],
});

/** A shape of no model: a row of its own, so the default table's rows play no part. */
const SHAPE = { storage: "i4", n: 1280, k: 1024, group: 32 } as const;
const ROW: LinearGemvParallelShape = { ...SHAPE, lanes: 4, packedActivations: false };

const rowBlockOf = (choices: LinearGemvChoices, m: number): LinearGemvParallelRows | undefined =>
  linearGemvParallelRowBlock(choices, SHAPE.storage, m, SHAPE.n, SHAPE.k, SHAPE.group);

describe("linearGemvParallelRowBlock", () => {
  it("gives a shape whose row carries a row block the rows of its cap at M = 2..8, and none at M = 1", () => {
    const choices = withRow({ ...ROW, rowBlock: 4 });
    assertEquals(
      [1, 2, 3, 4, 5, 8].map((m) => rowBlockOf(choices, m)),
      [undefined, 2, 2, 4, 4, 4],
    );
    assertEquals(rowBlockOf(withRow({ ...ROW, rowBlock: 8 }), 8), 8);
    assertEquals(rowBlockOf(withRow({ ...ROW, rowBlock: 2 }), 8), 2);
  });

  it("gives none past the parallel range of M, where the sequential row block runs", () => {
    assertEquals(rowBlockOf(withRow({ ...ROW, rowBlock: 4 }), 9), undefined);
    assertEquals(rowBlockOf(withRow({ ...ROW, rowBlock: 4 }), 64), undefined);
  });

  it("gives none to a row without the field: every row count keeps the kernel of M = 1 (fault: the field removed)", () => {
    assertEquals(
      [2, 4, 8].map((m) => rowBlockOf(withRow(ROW), m)),
      [undefined, undefined, undefined],
    );
  });

  it("gives none to a shape the table does not name, or names with another group length", () => {
    const choices = withRow({ ...ROW, rowBlock: 4 });
    assertEquals(linearGemvParallelRowBlock(choices, "i4", 4, 1281, 1024, 32), undefined);
    assertEquals(linearGemvParallelRowBlock(choices, "i4", 4, 1280, 1024, 64), undefined);
    assertEquals(linearGemvParallelRowBlock(choices, "i8", 4, 1280, 1024), undefined);
  });

  it("never gives a packed-activation row a row block (ADR 0105: its binding holds int8 codes)", () => {
    // A row a hand-built table could hold; the profile's guard refuses it before a Session sees it
    const choices = withRow({ ...ROW, packedActivations: true, rowBlock: 4 });
    assertEquals(rowBlockOf(choices, 4), undefined);
  });

  it("is independent of the staged flag, which is the form at M = 1", () => {
    assertEquals(rowBlockOf(withRow({ ...ROW, stagedActivations: true, rowBlock: 4 }), 4), 4);
    assertEquals(rowBlockOf(withRow({ ...ROW, rowBlock: 4 }), 4), 4);
  });
});

/** The lines of a kernel's text that match `pattern`. */
const linesOf = (code: string, pattern: RegExp): string[] =>
  code.split("\n").filter((line) => pattern.test(line));

describe("a row block on a float-stored parallel row", () => {
  const FLOATS = ["bf16", "f16", "f32"] as const;
  /** A float row as the profile takes it: no group, no packed activations. */
  const floatRow = (storage: (typeof FLOATS)[number]): LinearGemvParallelShape => ({
    storage,
    n: 1536,
    k: 2048,
    lanes: 4,
    packedActivations: false,
  });
  const profileOf = (row: LinearGemvParallelShape) => ({
    ...DEFAULT_GEOMETRY_PROFILE,
    id: "float-row-block",
    linearGemv: { ...GEMV, parallelShapes: [row] },
  });

  it("has no kernel: at no lane count and no height", () => {
    for (const storage of FLOATS) {
      for (const lanes of LANES) {
        for (const rows of LINEAR_GEMV_PARALLEL_ROWS) {
          assertEquals(linearGemvParallelRowsFits(storage, lanes, rows), false);
        }
      }
    }
  });

  it("is refused by the profile's guard, naming the row and the storage (fault: the field added to a valid float row)", () => {
    for (const storage of FLOATS) {
      assertGeometryProfile(profileOf(floatRow(storage)));
      for (const rowBlock of LINEAR_GEMV_PARALLEL_ROWS) {
        assertThrows(
          () => assertGeometryProfile(profileOf({ ...floatRow(storage), rowBlock })),
          CodegenError,
          `Geometry profile 'float-row-block': linearGemv.parallelShapes[0].rowBlock is set on a row of ${storage} storage (the row-blocked kernel exists for i2 / i4 / i8 storage)`,
        );
      }
    }
  });

  it("is not taken by a valid float row: every row count of 2 to 8 runs the kernel of M = 1", () => {
    for (const storage of FLOATS) {
      const row = floatRow(storage);
      const choices = withRow(row);
      for (const m of [2, 3, 4, 5, 6, 7, 8]) {
        assertEquals(linearGemvParallelEligible(choices, storage, m, row.n, row.k), row.lanes);
        assertEquals(linearGemvParallelRowBlock(choices, storage, m, row.n, row.k), undefined);
      }
    }
  });

  it("fails at the kernel's text when a table that skipped the guard carries one: it is not read as no row block", () => {
    const row = floatRow("bf16");
    const choices = withRow({ ...row, rowBlock: 2 });
    const rows = linearGemvParallelRowBlock(choices, "bf16", 4, row.n, row.k);
    assertEquals(rows, 2);
    assertThrows(
      () => linearGemvParallelRowsWgsl("bf16", undefined, row.lanes, 2),
      CodegenError,
      "no row block for bf16 storage",
    );
  });
});

describe("the row-blocked parallel GEMV's key and WGSL", () => {
  it("appends the rows to the staged key, so the storage, lane and staged discriminators stay where they are", () => {
    assertEquals(
      linearGemvParallelRowsKey("i4", 32, 4, 4),
      `${linearGemvParallelStagedKey("i4", 32, 4)}:r4`,
    );
    assertEquals(
      linearGemvParallelRowsKey("i8", undefined, 32, 2),
      "linear_gemv_parallel:wi8:l32:staged-x:r2",
    );
    assertEquals(
      linearGemvParallelRowsKey("i2", undefined, 2, 8),
      "linear_gemv_parallel:wi2:l2:staged-x:r8",
    );
  });

  it("gives the same text for the same parameters, and another for another row count, lane count, storage or group", () => {
    const text = linearGemvParallelRowsWgsl("i4", 32, 4, 4);
    assertEquals(linearGemvParallelRowsWgsl("i4", 32, 4, 4), text);
    for (const other of [
      linearGemvParallelRowsWgsl("i4", 32, 4, 2),
      linearGemvParallelRowsWgsl("i4", 32, 8, 4),
      linearGemvParallelRowsWgsl("i8", undefined, 4, 4),
      linearGemvParallelRowsWgsl("i4", 512, 4, 4),
    ]) {
      assert(other !== text);
    }
  });

  it("decodes every weight element once and multiply-adds it once per row, in the parallel kernel's element order", () => {
    for (const [storage, group, lanes, rows] of [
      ["i4", 32, 4, 4],
      ["i4", 32, 32, 2],
      ["i8", undefined, 16, 8],
      ["i2", undefined, 2, 8],
    ] as const) {
      const where = `${storage} L${lanes} R${rows}`;
      const plain = linearGemvParallelWgsl(storage, group, lanes);
      const blocked = linearGemvParallelRowsWgsl(storage, group, lanes, rows);
      const perWord = linesOf(plain, /acc = fma\(/).length;
      for (let row = 0; row < rows; row += 1) {
        assertEquals(linesOf(blocked, new RegExp(`acc${row} = fma\\(`)).length, perWord, where);
      }
      // The decode: one `let` per element for i4 / i8, one per four elements for i2 — never per row
      const decodes = linesOf(blocked, /^ {4}let dt_\w+ = /).length;
      assertEquals(decodes, storage === "i2" ? perWord / 4 : perWord, where);
      // The weight word and, for i4, its group scale are read once a word
      assertEquals(linesOf(blocked, /= w\[row_base \+ /).length, 1, where);
      assertEquals(
        linesOf(blocked, /= wscale\[scale_base \+ /).length,
        storage === "i4" ? 1 : 0,
        where,
      );
    }
  });

  it("uses the i4 decode expression of the parallel kernel (fault: another spelling)", () => {
    const expressions = (code: string): string[] =>
      [...code.matchAll(/\((f32\([^;]*?\)) \* wst\)/g)].map((match) => match[1]);
    const plain = expressions(linearGemvParallelWgsl("i4", 32, 4));
    assertEquals(plain.length, 32);
    assertEquals(expressions(linearGemvParallelRowsWgsl("i4", 32, 4, 4)), plain);
  });

  it("reads every activation from the staged copy of its row, as the staged kernel does for one row", () => {
    const staged = linearGemvParallelStagedWgsl("i4", 32, 4);
    const blocked = linearGemvParallelRowsWgsl("i4", 32, 4, 4);
    const reads = linesOf(staged, /= sx\[/).length;
    assertEquals(reads, 8);
    assertEquals(linesOf(blocked, /= sx\[/).length, reads * 4);
    assert(!/= x\[xq/.test(blocked));
    // 4 lanes × 8 quads a word, four rows of them; four rows of 128 partial sums
    assertStringIncludes(blocked, "var<workgroup> sx: array<vec4<f32>, 128u>;");
    assertStringIncludes(blocked, "var<workgroup> partial: array<f32, 512>;");
    // The staged kernel's barriers: two a chunk, one before the tree and one a level
    assertEquals(
      blocked.match(/workgroupBarrier\(\)/g)?.length,
      staged.match(/workgroupBarrier\(\)/g)?.length,
    );
  });

  it("stores only the rows inside M and re-reads the last row for the rest of a block", () => {
    const blocked = linearGemvParallelRowsWgsl("i8", undefined, 8, 4);
    for (let row = 0; row < 4; row += 1) {
      assertStringIncludes(blocked, `let xr${row} = min(row0 + ${row}u, dims.m - 1u) * kq;`);
      assertStringIncludes(blocked, `if (row0 + ${row}u < dims.m) {`);
    }
    assertStringIncludes(blocked, "let row0 = wg.y * 4u;");
    assertEquals(linesOf(blocked, /out\[\(row0 \+ \du\) \* dims\.n \+ col\] = /).length, 4);
  });

  it("refuses a row count the kernel does not exist at and a float storage", () => {
    assertThrows(
      () => linearGemvParallelRowsWgsl("i4", 32, 32, 4),
      CodegenError,
      "no row block of 4 rows for i4 at 32 lanes",
    );
    assertThrows(
      () => linearGemvParallelRowsWgsl("i2", undefined, 32, 2),
      CodegenError,
      "no row block of 2 rows for i2 at 32 lanes",
    );
    assertThrows(
      () => linearGemvParallelRowsWgsl("bf16", undefined, 4, 2),
      CodegenError,
      "no row block for bf16 storage",
    );
  });
});
