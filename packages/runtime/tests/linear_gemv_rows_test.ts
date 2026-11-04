// Pure-function checks of the linear GEMV family's **row-block variant selection** (ADR 0082
// addendum 5 / perf-ledger K-21). No GPU needed (only the generated code and the keys are checked).
//
// The row-block height `rows` is decided by `linearGemvRowsForShape`, **a pure function of
// (storage, m, n, parallelism target)** (the target is a static knob fixed when the Session is
// created; when omitted it is 16384, the saturation point of the reference device). This purity
// backs both "same key → byte-identical WGSL" and the ban on run-time autotuning (ADR 0022), so
// three things are checked:
//
// 1. **The selection table itself** — pins the (storage, m, n) → rows mapping taken from the sweep
//    (RTX 3080 Ti, docs/research/2026-09-07-gemv-rows-k21.md). MUST: the table is **a copy of the
//    implementation**, not an independent source of truth — when the sweep is redone and the
//    implementation changes, replace this table together with the measurements (this being red is
//    not by itself proof of a defect in the implementation). Conversely, if it turns red without
//    the implementation being touched, that is evidence the selection has state.
// 2. **The fail-loudly edges** — outside the gate (m=0 / m > limit / n=0 / non-integer) the call
//    fails instead of silently rounding. Rounding would be an entry point for silent wrong values:
//    "a shape the gate should have closed runs with different rows".
// 3. **`rows` is in the key** — without it, a rows-4 asset would be handed to a rows-16 dispatch,
//    the accumulator count and the y-tile height would disagree, and the values would be wrong at
//    once.
//
// Bit identity of the numbers (exact u32 match with the default GEMM path) is on the real-GPU side
// = tests/gpu_linear_gemv_test.ts.

import { describe, it, test } from "vite-plus/test";
import { assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { CodegenError } from "../src/codegen/errors.ts";
import {
  defaultLinearGemvRowsVariant,
  defaultLinearGemvVariant,
  LINEAR_GEMV_MAX_ROWS,
  linearGemvKey,
  linearGemvParallelKey,
  linearGemvParallelLanes,
  linearGemvParallelWgsl,
  linearGemvParams,
  linearGemvRowsForShape,
  linearGemvRowsKey,
  linearGemvRowsWgsl,
  linearGemvSubgroupKey,
  linearGemvSubgroupWgsl,
  linearGemvWgsl,
} from "../src/kernels/linear-gemv.ts";
import type { WeightStorage } from "../src/kernels/weight-storage.ts";

/** One row of the selection table (the M → rows mapping for one (storage, n)). */
type RowsTable = {
  readonly storage: WeightStorage;
  readonly n: number;
  /** What shape this is (which layer of the census it corresponds to). */
  readonly note: string;
  /** `[m, rows]` pairs. M always includes 1 and the limit 64. */
  readonly picks: readonly (readonly [m: number, rows: number])[];
};

/**
 * The (storage, m, n) → rows table.
 *
 * MUST: **span orders of magnitude of n** — the selection halves rows "until the thread count = n ×
 * ceil(m / rows) reaches the target 16384", so without both small-n layers (always rows=1) and
 * large-n layers (growing up to the ceiling), one cannot tell whether the target side or the
 * ceiling side is in effect.
 * MUST: include both storage types — how the ceiling (512 elements a thread holds per word) acts on
 * rows differs by storage (i4 has 32 elements per word → up to 8 rows; i8 has 16 → up to 16 rows).
 */
const ROWS_TABLES: readonly RowsTable[] = [
  {
    storage: "i4",
    n: 256,
    note: "small n (the smallest layer of gemma4 E2B) — the thread count never reaches the target, so rows=1 for every M",
    picks: [
      [1, 1],
      [2, 1],
      [4, 1],
      [7, 1],
      [8, 1],
      [16, 1],
      [32, 1],
      [64, 1],
    ],
  },
  {
    storage: "i4",
    n: 2048,
    note: "medium n — the target is reached only as M grows, and rows rises step by step",
    picks: [
      [1, 1],
      [2, 1],
      [4, 1],
      [7, 1],
      [8, 1],
      [16, 2],
      [32, 4],
      [64, 8],
    ],
  },
  {
    storage: "i4",
    n: 4096,
    note: "medium n (double 2048) — rows is exactly one step deeper at the same M, and hits the i4 ceiling 8 at M=64",
    picks: [
      [1, 1],
      [2, 1],
      [4, 1],
      [7, 2],
      [8, 2],
      [16, 4],
      [32, 8],
      [64, 8],
    ],
  },
  {
    storage: "i4",
    n: 12288,
    note: "large n — hits the i4 ceiling of 8 rows at M=16, and stays at 8 for M=32 / 64",
    picks: [
      [1, 1],
      [2, 1],
      [4, 2],
      [7, 4],
      [8, 4],
      [16, 8],
      [32, 8],
      [64, 8],
    ],
  },
  {
    storage: "i8",
    n: 262144,
    note: "lm_head (i8) — every M reaches the target, so rows is M itself, capped at the ceiling 16",
    picks: [
      [1, 1],
      [2, 2],
      [4, 4],
      [7, 4],
      [8, 8],
      [16, 16],
      [32, 16],
      [64, 16],
    ],
  },
  {
    storage: "i4",
    n: 1,
    note: "degenerate (a single column) — the target can never be reached, so rows=1",
    picks: [
      [1, 1],
      [2, 1],
      [64, 1],
    ],
  },
  {
    storage: "i8",
    n: 1,
    note: "degenerate (a single column, i8)",
    picks: [
      [1, 1],
      [2, 1],
      [64, 1],
    ],
  },
];

test("the row-block rows is a pure function of (storage, m, n) and follows the table taken from the sweep", () => {
  for (const { storage, n, note, picks } of ROWS_TABLES) {
    for (const [m, rows] of picks) {
      const actual = linearGemvRowsForShape(storage, m, n);
      assertEquals(actual, rows, `${storage} n=${n} m=${m}（${note}）`);
      // The same input looked up twice gives the same value (the selection has no state = the
      // premise of the key-to-WGSL mapping)
      assertEquals(
        linearGemvRowsForShape(storage, m, n),
        actual,
        `${storage} n=${n} m=${m}: reproduced`,
      );
      // The default variant inherits cols / unroll from the M=1 variant; only rows comes from this
      // pure function
      assertEquals(
        defaultLinearGemvRowsVariant(storage, m, n),
        { ...defaultLinearGemvVariant(), rows },
        `${storage} n=${n} m=${m}: default variant`,
      );
    }
  }
});

test("lowering the parallelism target raises rows (a static knob for other devices)", () => {
  // The target is the limit `SessionOptions.linearGemvRowsThreadTarget` fixes when the Session is
  // created; on a GPU with a smaller saturation point it is lowered to raise `rows` (fewer weight
  // re-reads). For i4, m=4, n=2048 the thread count = 2048 × ceil(4 / rows), so with the target
  // 16384 it drops all the way to rows=1, and each halving of the target moves the stopping point
  // up one step.
  assertEquals(linearGemvRowsForShape("i4", 4, 2048, 16384), 1, "default target");
  assertEquals(linearGemvRowsForShape("i4", 4, 2048, 4096), 2, "target 1/4");
  assertEquals(linearGemvRowsForShape("i4", 4, 2048, 2048), 4, "target 1/8");
  // Lowering the target does not exceed the ceiling (256 elements per word = 8 rows for i4).
  assertEquals(linearGemvRowsForShape("i4", 64, 2048, 1), 8, "the i4 ceiling 8 even with target 1");
  // Omitting the fourth argument = passing the default target (16384) (existing calls give the same
  // result unchanged).
  for (const { storage, n, picks } of ROWS_TABLES) {
    for (const [m] of picks) {
      assertEquals(
        linearGemvRowsForShape(storage, m, n),
        linearGemvRowsForShape(storage, m, n, 16384),
        `${storage} n=${n} m=${m}: omitted = default 16384`,
      );
      assertEquals(
        defaultLinearGemvRowsVariant(storage, m, n),
        defaultLinearGemvRowsVariant(storage, m, n, 16384),
        `${storage} n=${n} m=${m}: the variant matches too`,
      );
    }
  }
  // The target also reaches the variant side (the key's `r<rows>` moves = the generated code is a
  // different text).
  assertEquals(defaultLinearGemvRowsVariant("i4", 4, 2048, 2048).rows, 4);
});

test("M=1 gives rows=1 regardless of n (decode's generated code is not row-blocked)", () => {
  for (const n of [1, 256, 2048, 16384, 262144]) {
    assertEquals(linearGemvRowsForShape("i4", 1, n), 1, `i4 n=${n}`);
    assertEquals(linearGemvRowsForShape("i8", 1, n), 1, `i8 n=${n}`);
  }
});

test("(m, n) outside the gate fail instead of being silently rounded", () => {
  // The range of m: 1..LINEAR_GEMV_MAX_ROWS. Beyond the limit the default GEMM skeleton takes the
  // shape, so silently rounding to the limit here would be a silent wrong value: "a shape the gate
  // closed runs as a row block".
  assertThrows(() => linearGemvRowsForShape("i4", 0, 2048), CodegenError, "row count 0");
  assertThrows(
    () => linearGemvRowsForShape("i4", LINEAR_GEMV_MAX_ROWS + 1, 2048),
    CodegenError,
    `row count ${LINEAR_GEMV_MAX_ROWS + 1}`,
  );
  assertThrows(() => linearGemvRowsForShape("i8", -1, 2048), CodegenError, "row count -1");
  assertThrows(() => linearGemvRowsForShape("i4", 1.5, 2048), CodegenError, "row count 1.5");
  // The range of n: positive integers. For 0 / a non-integer the "thread count" computation has no
  // meaning at all.
  assertThrows(() => linearGemvRowsForShape("i4", 8, 0), CodegenError, "column count 0");
  assertThrows(() => linearGemvRowsForShape("i8", 8, -256), CodegenError, "column count -256");
  assertThrows(() => linearGemvRowsForShape("i4", 8, 2048.5), CodegenError, "column count 2048.5");
  // Only the two compressed storage types (staying within the measured range — ADR 0082 decision 4)
  assertThrows(() => linearGemvRowsForShape("f32", 8, 2048), CodegenError, "f32 storage");
});

test("rows is in the key, and a rows=1 row block has a different key from the M=1 variant", () => {
  assertEquals(
    linearGemvRowsKey("i4", 32, defaultLinearGemvRowsVariant("i4", 32, 2048)),
    "linear_gemv:v2:f32:c32u4r4:wi4g32",
  );
  assertEquals(
    linearGemvRowsKey("i4", 64, defaultLinearGemvRowsVariant("i4", 32, 12288)),
    "linear_gemv:v2:f32:c32u4r8:wi4g64",
  );
  assertEquals(
    linearGemvRowsKey("i8", undefined, defaultLinearGemvRowsVariant("i8", 32, 262144)),
    "linear_gemv:v2:f32:c32u4r16:wi8",
  );
  // MUST: `r1` differs from the M=1 variant's key (the text differs too — a y-tiled row block). If
  // they mapped to the same key, the M=1 asset would be handed to an M ≥ 2 dispatch and only one
  // row would be written.
  assertNotEquals(
    linearGemvRowsKey("i4", 32, defaultLinearGemvRowsVariant("i4", 2, 64)),
    linearGemvKey("i4", 32),
  );
  assertEquals(
    linearGemvRowsKey("i4", 32, defaultLinearGemvRowsVariant("i4", 2, 64)),
    "linear_gemv:v2:f32:c32u4r1:wi4g32",
  );
  // rows has a range just like cols / unroll (if key generation let an out-of-range value through,
  // it would first fail on the WGSL side)
  assertThrows(
    () => linearGemvRowsKey("i4", 32, { cols: 32, unroll: 4, rows: 0 }),
    CodegenError,
    "rows must be",
  );
  assertThrows(
    () => linearGemvRowsKey("i4", 32, { cols: 32, unroll: 4, rows: LINEAR_GEMV_MAX_ROWS + 1 }),
    CodegenError,
    "rows must be",
  );
});

test("the uniform m is 1..limit (the range the row block's y tiles take)", () => {
  // Exactly the limit passes (dims.m is the very value the row block's write-back guard reads)
  assertEquals(
    [...linearGemvParams("i4", LINEAR_GEMV_MAX_ROWS, 2048, 128, 32)],
    [LINEAR_GEMV_MAX_ROWS, 2048, 128, 0],
  );
  assertEquals([...linearGemvParams("i8", 2, 64, 64)], [2, 64, 64, 0]);
  // Outside the range (m=0 / limit + 1) it fails. Letting it through would make the dispatch's y
  // count and dims.m disagree, so the write-back guard silently drops every row / writes rows out
  // of range.
  assertThrows(
    () => linearGemvParams("i4", 0, 2048, 128, 32),
    CodegenError,
    "m must be an integer in 1..",
  );
  assertThrows(
    () => linearGemvParams("i4", LINEAR_GEMV_MAX_ROWS + 1, 2048, 128, 32),
    CodegenError,
    "m must be an integer in 1..",
  );
  assertThrows(
    () => linearGemvParams("i8", LINEAR_GEMV_MAX_ROWS + 1, 64, 64),
    CodegenError,
    "m must be an integer in 1..",
  );
});

test("the f16 GEMV accepts only M=1 with 8-element alignment and refuses to generate row blocks", () => {
  assertEquals([...linearGemvParams("f16", 1, 36, 40)], [1, 36, 40, 0]);
  assertEquals([...linearGemvParams("f16", 1, 4, 0)], [1, 4, 0, 0]);
  assertThrows(() => linearGemvParams("f16", 2, 36, 40), CodegenError, "only m=1");
  assertThrows(() => linearGemvParams("f16", 1, 36, 36), CodegenError, "k");
  assertThrows(() => linearGemvParams("f16", 1, 36, 40, 32), CodegenError);
  assertThrows(
    () => linearGemvRowsForShape("f16", 1, 36),
    CodegenError,
    "row blocks are not supported",
  );
  const variant = { cols: 32, unroll: 4, rows: 1 };
  assertThrows(
    () => linearGemvRowsKey("f16", undefined, variant),
    CodegenError,
    "row blocks are not supported",
  );
  assertThrows(
    () => linearGemvRowsWgsl("f16", undefined, variant),
    CodegenError,
    "row blocks are not supported",
  );
});

test("the f32 GEMV accepts only M=1 with 4-element alignment and refuses to generate row blocks", () => {
  assertEquals([...linearGemvParams("f32", 1, 36, 40)], [1, 36, 40, 0]);
  assertEquals([...linearGemvParams("f32", 1, 4, 0)], [1, 4, 0, 0]);
  assertThrows(() => linearGemvParams("f32", 2, 36, 40), CodegenError, "only m=1");
  assertThrows(() => linearGemvParams("f32", 1, 36, 38), CodegenError, "k");
  assertThrows(() => linearGemvParams("f32", 1, 36, 40, 32), CodegenError);
  assertThrows(
    () => linearGemvRowsForShape("f32", 1, 36),
    CodegenError,
    "row blocks are not supported",
  );
  const variant = { cols: 32, unroll: 4, rows: 1 };
  assertThrows(
    () => linearGemvRowsKey("f32", undefined, variant),
    CodegenError,
    "row blocks are not supported",
  );
  assertThrows(
    () => linearGemvRowsWgsl("f32", undefined, variant),
    CodegenError,
    "row blocks are not supported",
  );
});

test("the INT2 row block keeps the shader under 10 KB at every default height", () => {
  // Detects a regression in first-parse cost. The former per-row unrolling was 76 KB at r4 (ADR
  // 0082 addendum 8).
  for (const rows of [1, 2, 4]) {
    const variant = { cols: 32, unroll: 4, rows };
    const source = linearGemvRowsWgsl("i2", undefined, variant);
    assertEquals(source.length < 10_000, true, `${rows} rows: ${source.length} bytes`);
    assertEquals(
      linearGemvRowsKey("i2", undefined, variant),
      `linear_gemv:v2:f32:c32u4r${rows}:wi2`,
    );
  }
});

test("the INT4 and INT8 row blocks keep the shader under 12 KB at every default height", () => {
  // The former maximum height was about 70 KB. Detects a regression back to row unrolling that
  // increases first compilation.
  for (const storage of ["i4", "i8"] as const) {
    for (const rows of storage === "i4" ? [1, 2, 4, 8] : [1, 2, 4, 8, 16]) {
      const variant = { cols: 32, unroll: 4, rows };
      const source = linearGemvRowsWgsl(storage, storage === "i4" ? 32 : undefined, variant);
      assertEquals(
        source.length < 12_000,
        true,
        `${storage} ${rows} rows: ${source.length} characters`,
      );
    }
  }
});

// The M=1 variant choice and the parallel-add (lanes) selection share the selection-function
// subject of the row-block gates above: pure functions of the shape, keys that tell every
// generated text apart, and fail-loudly rejection before codegen.

describe("M=1 GEMV shape selection", () => {
  it("selects c16 for the measured large-vocabulary INT8 and tells the WGSL apart by a separate key", () => {
    const selected = defaultLinearGemvVariant({ storage: "i8", n: 262144, k: 1536 });
    assertEquals(selected, { cols: 16, unroll: 4 });
    assertEquals(linearGemvKey("i8", undefined, selected), "linear_gemv:v1:f32:c16u4:wi8");
    assertNotEquals(linearGemvKey("i8", undefined, selected), linearGemvKey("i8"));
    assertEquals(
      linearGemvWgsl("i8", undefined, selected),
      linearGemvWgsl("i8")
        .replace("32 cols / wg", "16 cols / wg")
        .replace("@workgroup_size(32)", "@workgroup_size(16)"),
    );
  });

  it("keeps the former variant for a different storage type or matrix shape and for generation without a shape", () => {
    assertEquals(defaultLinearGemvVariant(), { cols: 32, unroll: 4 });
    const storageTypes: readonly WeightStorage[] = ["i2", "i4", "f16", "f32"];
    for (const storage of storageTypes) {
      assertEquals(defaultLinearGemvVariant({ storage, n: 262144, k: 1536 }), {
        cols: 32,
        unroll: 4,
      });
    }
    for (const [n, k] of [
      [262140, 1536],
      [262148, 1536],
      [262144, 1520],
      [262144, 1552],
      [6144, 1536],
    ]) {
      assertEquals(defaultLinearGemvVariant({ storage: "i8", n, k }), { cols: 32, unroll: 4 });
    }
  });
});

describe("GEMV parallel-sum selection and code generation", () => {
  it("selects the same geometry for decode and small-row verification, and does not extend it to prefill or unmeasured shapes", () => {
    for (const m of [1, 4, 8]) {
      assertEquals(linearGemvParallelLanes("i4", m, 256, 1536, 32), 32);
      assertEquals(linearGemvParallelLanes("i4", m, 6144, 1536, 512), 4);
      assertEquals(linearGemvParallelLanes("i4", m, 1536, 4096, 4096), 32);
      assertEquals(linearGemvParallelLanes("i2", m, 12288, 1536), 2);
      assertEquals(linearGemvParallelLanes("i8", m, 262144, 1536), 16);
    }
    for (const m of [0, 9, 64, 65]) {
      assertEquals(linearGemvParallelLanes("i4", m, 256, 1536, 32), undefined);
    }
    assertEquals(linearGemvParallelLanes("i2", 1, 262144, 1536), undefined);
    assertEquals(linearGemvParallelLanes("i4", 1, 256, 1536, 64), undefined);
    assertEquals(linearGemvParallelLanes("i4", 1, 260, 1536, 32), undefined);
    assertEquals(linearGemvParallelLanes("f16", 1, 256, 1536), undefined);
  });
  it("selects the lanes of the matching E2B row for the regular E4B shapes (i4 g32 + i8 lm_head) (an unmeasured copy)", () => {
    // The shapes are real shapes derived from the upstream config (hidden 2560, intermediate 10240,
    // kv 2, 42 layers, PLE 256). QAT E4B is covered by the census test rooted at the distribution
    // mirror's descriptor.
    const rows: readonly [number, number, 4 | 16 | 32][] = [
      [10752, 2560, 4],
      [2048, 2560, 4],
      [512, 2560, 32],
      [256, 2560, 32],
      [2560, 2048, 32],
      [10240, 2560, 4],
      [2560, 10240, 32],
      [2560, 256, 4],
      [4096, 2560, 4],
      [1024, 2560, 32],
      [2560, 4096, 32],
    ];
    for (const m of [1, 4, 8]) {
      for (const [n, k, lanes] of rows) {
        assertEquals(linearGemvParallelLanes("i4", m, n, k, 32), lanes, `i4 ${n}×${k} M${m}`);
      }
      assertEquals(linearGemvParallelLanes("i8", m, 262144, 2560), 16);
    }
    assertEquals(linearGemvParallelLanes("i4", 9, 10240, 2560, 32), undefined);
    // The i2 lm_head (QAT E4B) has no row in E2B either, so it is not added.
    assertEquals(linearGemvParallelLanes("i2", 1, 262144, 2560), undefined);
  });
  it("the key distinguishes the sum form, storage, group and lane", () => {
    const keys = new Map<string, string>();
    for (const storage of ["i2", "i4", "i8"] as const) {
      for (const group of storage === "i4" ? [32, 512, 2048, 4096] : [undefined]) {
        for (const lanes of [2, 4, 8, 16, 32] as const) {
          const key = linearGemvParallelKey(storage, group, lanes);
          const shader = linearGemvParallelWgsl(storage, group, lanes);
          assertNotEquals(key, linearGemvKey(storage, group));
          assertEquals(keys.has(key), false);
          keys.set(key, shader);
          assertEquals(linearGemvParallelWgsl(storage, group, lanes), shader);
          const subgroupKey = linearGemvSubgroupKey(storage, group, lanes);
          const subgroupShader = linearGemvSubgroupWgsl(storage, group, lanes);
          assertNotEquals(subgroupKey, key);
          assertEquals(keys.has(subgroupKey), false);
          keys.set(subgroupKey, subgroupShader);
          assertEquals(linearGemvSubgroupWgsl(storage, group, lanes), subgroupShader);
        }
      }
    }
    assertEquals(keys.size, 60);
  });
  it("rejects unsupported storage and an invalid group before code generation", () => {
    assertThrows(() => linearGemvParallelWgsl("f16", undefined, 4));
    assertThrows(() => linearGemvParallelWgsl("f32", undefined, 4));
    assertThrows(() => linearGemvParallelWgsl("i4", 16, 4));
    assertThrows(() => linearGemvParallelWgsl("i8", 32, 4));
    assertThrows(() => linearGemvSubgroupWgsl("f16", undefined, 4));
    assertThrows(() => linearGemvSubgroupWgsl("f32", undefined, 4));
    assertThrows(() => linearGemvSubgroupWgsl("i4", 16, 4));
    assertThrows(() => linearGemvSubgroupWgsl("i8", 32, 4));
  });
});

describe("the multiply-adds of the subgroup32 variant are spelled the same as parallel", () => {
  // Binds ADR 0101 decision 3 (the multiply-adds are identical to parallel) at the spelling level.
  // If only the parallel family is turned into fma and subgroup keeps `acc + x * d`, Metal changes
  // how it contracts depending on the expression form and the u32 match breaks (ADR 0105 addendum
  // 4). The real-GPU u32 match check (checkGemvSubgroup of @karume/kernel-checks) SKIPs on Deno, so
  // this check is the always-on side.
  const macLines = (wgsl: string): string[] =>
    wgsl
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.startsWith("acc = "));
  it("the fma lines match for all 60 shapes and no acc + x * d remains in the multiply-adds", () => {
    let compared = 0;
    for (const storage of ["i2", "i4", "i8"] as const) {
      for (const group of storage === "i4" ? [32, 512, 2048, 4096] : [undefined]) {
        for (const lanes of [2, 4, 8, 16, 32] as const) {
          const parallel = macLines(linearGemvParallelWgsl(storage, group, lanes));
          const subgroup = macLines(linearGemvSubgroupWgsl(storage, group, lanes));
          const fma = (lines: string[]): string[] =>
            lines.filter((l) => l.startsWith("acc = fma("));
          assertEquals(fma(subgroup), fma(parallel), `${storage} g${group} l${lanes}`);
          assertEquals(fma(subgroup).length > 0, true);
          // The subgroup value exchange (`if (lane < width) { acc = acc + other; }`) is an if line,
          // so having not a single `acc + x * d` among the lines starting with `acc = ` is the
          // multiply-add check.
          assertEquals(
            subgroup.filter((l) => l.startsWith("acc = acc +")),
            [],
            `${storage} g${group} l${lanes}: acc + x * d remains in the multiply-adds`,
          );
          compared++;
        }
      }
    }
    assertEquals(compared, 30);
  });
});
