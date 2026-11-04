// The **GPU-free surface** of the three states-form attention kernels + `state_append` (ADR 0067
// decisions 4 to 7): the variant bits of the keys, determinism of generation, the shared read/write
// formula, the value-range checks of params, the pure functions of dispatch geometry, and the
// semantics of the CPU reference. The comparison with the real GPU is gpu_state_attention_test.ts.
//
// MUST: the semantics of the CPU reference are pinned with **hand-computed oracles** (comparing
// with the GPU alone would leave both green if the reference and the GPU shared the same mistake).
// The probes make the window set visible in the value (V of an in-window column is its col number,
// so the output is an average), folding the predicate's upper bound, lower bound and the ring
// mapping into one number.

import { describe, it, test } from "vite-plus/test";
import { assertAlmostEquals, assertEquals, assertNotEquals, assertThrows } from "@std/assert";
import { CodegenError } from "../src/codegen/errors.ts";
import { DispatchLimitError } from "../src/codegen/errors.ts";
import {
  STATE_ATTENTION_TILE_M,
  STATE_ATTENTION_TILE_X,
  STATE_NEG_INF_BITS,
  STATE_QK_D_LANES,
  STATE_STATS_WORKGROUP_SIZE,
  stateAttentionParams,
  stateColumnBase,
  stateEffectiveRows,
  stateLiveColumns,
  statePvKey,
  statePvParallelKey,
  statePvParallelReadonlyKey,
  statePvParallelReadonlyWgsl,
  statePvParallelWgsl,
  statePvTiledEligible,
  statePvTiledKey,
  statePvTiledParams,
  statePvTiledWorkgroups,
  statePvWgsl,
  statePvWorkgroups,
  STATE_PV_UNROLL,
  STATE_QK_FOLD_TILE_X,
  assertStateQkFold,
  statePvScanWgsl,
  stateQkFoldedKey,
  stateQkFoldedWgsl,
  stateQkFoldedWorkgroups,
  stateQkFoldHeads,
  stateQkKey,
  stateQkParallelEligible,
  stateQkParallelKey,
  stateQkParallelReadonlyKey,
  stateQkParallelWgsl,
  stateQkParallelWorkgroups,
  stateQkTiledEligible,
  stateQkTiledKey,
  stateQkTiledParams,
  stateQkTiledWorkgroups,
  stateQkWgsl,
  stateQkWorkgroups,
  stateSliding,
  stateSlotRowWgsl,
  stateStatsKey,
  stateStatsParams,
  stateStatsReadonlyKey,
  stateStatsWgsl,
  stateStatsWorkgroups,
} from "../src/kernels/state-attention.ts";
import { stateStatsPvKey, stateStatsPvWgsl } from "../src/kernels/state-attention-stats-pv.ts";
import { statePvTiledWgsl, stateQkTiledWgsl } from "../src/kernels/gemm.ts";
import {
  STATE_APPEND_WORKGROUP_SIZE,
  stateAppendKey,
  stateAppendParams,
  stateAppendWgsl,
  stateAppendWorkgroups,
} from "../src/kernels/state-append.ts";
import { ReferenceOpError } from "../src/reference/ops.ts";
import {
  referenceStateAppend,
  referenceStateAttention,
  referenceStateAttentionReadonly,
  type StateAttentionRefInput,
} from "../src/reference/state-attention.ts";

const VARIANTS: readonly (readonly [boolean, boolean])[] = [
  [false, false],
  [false, true],
  [true, false],
  [true, true],
];

test("states-form keys differ only by the two bits :sliding / :gqa (their order lives in one place)", () => {
  assertEquals(stateQkKey(false, false), "attention_state_qk:v1:f32:wg16x4");
  assertEquals(stateQkKey(true, false), "attention_state_qk:v1:f32:wg16x4:sliding");
  assertEquals(stateQkKey(false, true), "attention_state_qk:v1:f32:wg16x4:gqa");
  assertEquals(stateQkKey(true, true), "attention_state_qk:v1:f32:wg16x4:sliding:gqa");
  assertEquals(statePvKey(true, true), "attention_state_pv:v1:f32:wg16x4:sliding:gqa");
  assertEquals(stateStatsKey(false), "attention_state_stats:v2:f32:wg256");
  assertEquals(stateStatsKey(true), "attention_state_stats:v2:f32:wg256:sliding");
  // The fused ②+③' (ADR 0102) also splits by the same two bits. This is the cache identity, so the
  // surface where a moved spelling gives "a different pipeline for the same shape" is pinned at the
  // same level as the other variants.
  // v2 = the unrolled PV loop (perf-ledger K-91 — the WGSL bytes changed, so the key moved).
  assertEquals(stateStatsPvKey(false, false), "attention_state_stats_pv:v2:f32:wg16x16");
  assertEquals(stateStatsPvKey(true, false), "attention_state_stats_pv:v2:f32:wg16x16:sliding");
  assertEquals(stateStatsPvKey(false, true), "attention_state_stats_pv:v2:f32:wg16x16:gqa");
  assertEquals(stateStatsPvKey(true, true), "attention_state_stats_pv:v2:f32:wg16x16:sliding:gqa");
  assertEquals(stateAppendKey(false), "state_append:v1:f32:wg256");
  assertEquals(stateAppendKey(true), "state_append:v1:f32:wg256:sliding");
  // The geometry constants appear in the key (moving the workgroup moves the key = a different
  // pipeline)
  assertEquals(STATE_ATTENTION_TILE_X, 16);
  assertEquals(STATE_ATTENTION_TILE_M, 4);
  assertEquals(STATE_STATS_WORKGROUP_SIZE, 256);
  assertEquals(STATE_APPEND_WORKGROUP_SIZE, 256);
  // Every variant of the 5 families has its own key (one configuration does not have two keys / two
  // configurations do not share a key)
  const keys = [
    ...VARIANTS.map(([sliding, gqa]) => stateQkKey(sliding, gqa)),
    ...VARIANTS.map(([sliding, gqa]) => statePvKey(sliding, gqa)),
    ...VARIANTS.map(([sliding, gqa]) => stateStatsPvKey(sliding, gqa)),
    stateStatsKey(false),
    stateStatsKey(true),
    stateAppendKey(false),
    stateAppendKey(true),
  ];
  assertEquals(new Set(keys).size, keys.length);
});

test("readonly-form keys are spelled differently from the live form and collide with no states-form key", () => {
  // readonly has one binding fewer, and the live formula of ② differs too. The drafter and the
  // target share the pipeline cache of the same device, so if a key collided with the live form the
  // other's WGSL would be handed out and the window would shift without an exception. The spelling
  // is pinned here.
  assertEquals(
    stateQkParallelReadonlyKey(false, false),
    "attention_state_qk:v1:f32:wg16x16:par:ro",
  );
  assertEquals(stateStatsReadonlyKey(true), "attention_state_stats:v2:f32:wg256:ro:sliding");
  assertEquals(
    statePvParallelReadonlyKey(true, true),
    "attention_state_pv:v1:f32:wg16x16:par:ro:sliding:gqa",
  );
  const keys = [
    ...VARIANTS.flatMap(([sliding, gqa]) => [
      stateQkKey(sliding, gqa),
      stateQkParallelKey(sliding, gqa),
      stateQkParallelReadonlyKey(sliding, gqa),
      statePvKey(sliding, gqa),
      statePvParallelKey(sliding, gqa),
      statePvParallelReadonlyKey(sliding, gqa),
      stateStatsPvKey(sliding, gqa),
    ]),
    ...[false, true].flatMap((sliding) => [stateStatsKey(sliding), stateStatsReadonlyKey(sliding)]),
    stateAppendKey(false),
    stateAppendKey(true),
    stateQkFoldedKey(4),
    stateQkFoldedKey(8),
  ];
  assertEquals(keys.length, 4 * 7 + 2 * 2 + 2 + 2);
  assertEquals(new Set(keys).size, keys.length);
});

test("the logical col → slot physical row mapping is the same string for reads (①③) and writes (state_append)", () => {
  for (const sliding of [false, true]) {
    const fragment = stateSlotRowWgsl(sliding);
    // MUST: it is contained verbatim. If only one side used a different formula, every read after
    // the ring wrapped around would silently point at another row (the mechanical proof of the
    // "same formula for reads and writes MUST" of ADR 0067 decision 4)
    for (const [name, wgsl] of [
      ["qk", stateQkWgsl(sliding, false)],
      ["qk gqa", stateQkWgsl(sliding, true)],
      ["pv", statePvWgsl(sliding, false)],
      ["append", stateAppendWgsl(sliding)],
    ] as const) {
      assertEquals(wgsl.includes(fragment), true, `${name}: the slot_row fragment is not shared`);
    }
  }
  // Only sliding has the modulo (the full mapping is the identity)
  assertEquals(stateSlotRowWgsl(true).includes("col % params.capacity"), true);
  assertEquals(stateSlotRowWgsl(false).includes("%"), false);
});

test("the same generation input always gives the same WGSL (determinism of the states form)", () => {
  for (const [sliding, gqa] of VARIANTS) {
    assertEquals(stateQkWgsl(sliding, gqa), stateQkWgsl(sliding, gqa));
    assertEquals(statePvWgsl(sliding, gqa), statePvWgsl(sliding, gqa));
  }
  for (const sliding of [false, true]) {
    assertEquals(stateStatsWgsl(sliding), stateStatsWgsl(sliding));
    assertEquals(stateAppendWgsl(sliding), stateAppendWgsl(sliding));
    // The variants are really different (keys split but identical contents = the variant has no
    // effect)
    assertNotEquals(stateQkWgsl(sliding, false), stateQkWgsl(sliding, true));
  }
  assertNotEquals(stateStatsWgsl(false), stateStatsWgsl(true));
  assertNotEquals(stateAppendWgsl(false), stateAppendWgsl(true));
});

test("the sliding variant ANDs a **lower bound** onto the causal upper bound (rejects an upper-bound-only implementation by string)", () => {
  // ADR 0067 decision 4, round 5 high: with only the upper bound, row > 0 silently mixes in row
  // out-of-window keys
  assertEquals(stateQkWgsl(true, false).includes("(limit - col) < params.window"), true);
  assertEquals(stateQkWgsl(false, false).includes("(limit - col) < params.window"), false);
  // MUST: the subtraction of the lower bound is on the `limit − col` side (`col + W` wraps around
  // when P is near the u32 maximum, puts even the causal diagonal out of the window and quietly
  // empties the row). The short-circuiting `col <= limit` comes first, so `limit − col` does not
  // underflow
  for (const [sliding, gqa] of VARIANTS) {
    assertEquals(
      stateQkWgsl(sliding, gqa).includes("col + params.window"),
      false,
      `sliding=${sliding} gqa=${gqa}: the addition form that wraps around in u32 remains`,
    );
  }
  // Empty-row guard (②) — built from the -inf identity + a denominator guard, not a finite sentinel
  assertEquals(stateStatsWgsl(false).includes("let empty = amax == neg_inf;"), true);
  assertEquals(STATE_NEG_INF_BITS, 0xff800000);
});

test("the effective-row formula is one for the WGSL of ①②③ and the host (pad rows are not handled differently by the three)", () => {
  // Host (dispatch count): `clamp(Q − rowOffset, 0, rowsBlock)`
  assertEquals(stateEffectiveRows(4, 0, 1), 1, "decode has a single row");
  assertEquals(stateEffectiveRows(4, 0, 4), 4, "prefill without pad has every row");
  assertEquals(stateEffectiveRows(4, 0, 9), 4, "Q > rowsBlock is capped at the block");
  assertEquals(stateEffectiveRows(4, 4, 6), 2, "pad starts in the middle of the block");
  assertEquals(stateEffectiveRows(4, 4, 4), 0, "a block that is entirely pad is 0");
  assertEquals(stateEffectiveRows(4, 8, 4), 0, "underflow does not give a huge value");
  // WGSL (row range): all three kernels contain **the same string** (rejects fixing only one side
  // by string)
  const fragment = "fn effective_rows(query: u32) -> u32 {";
  for (const [sliding, gqa] of VARIANTS) {
    assertEquals(stateQkWgsl(sliding, gqa).includes(fragment), true, "qk");
    assertEquals(statePvWgsl(sliding, gqa).includes(fragment), true, "pv");
  }
  for (const sliding of [false, true]) {
    assertEquals(stateStatsWgsl(sliding).includes(fragment), true, "stats");
  }
  // ③PV **writes 0 and returns** for pad rows (keeps full-write without scanning live)
  assertEquals(statePvWgsl(false, false).includes("out[at] = 0.0;"), true);
});

test("the three PV variants of the KV parallel reduction write the pad rows' 0 from lane 0 only (rejects unsynchronised writes to the same element)", () => {
  // The KV_LANES lanes of one workgroup point at the same `out[at]`. If every lane wrote 0 without
  // a lane guard, that is a data race in the WGSL memory model even though the value is the same —
  // comparing output values cannot detect it, so it is bound by the contents of the pad branch in
  // the generated string (the same guard as the write of a regular row).
  const padBranch = (wgsl: string): string => {
    const start = wgsl.indexOf("if (local_row >= effective_rows(lengths.query)) {");
    assertNotEquals(start, -1, "the pad branch was not found");
    return wgsl.slice(start, wgsl.indexOf("return;", start));
  };
  for (const [sliding, gqa] of VARIANTS) {
    for (const [name, wgsl] of [
      ["③'", statePvParallelWgsl(sliding, gqa)],
      ["③' readonly", statePvParallelReadonlyWgsl(sliding, gqa)],
      ["②③ fused", stateStatsPvWgsl(sliding, gqa)],
    ] as const) {
      const where = `${name} sliding=${sliding} gqa=${gqa}`;
      const branch = padBranch(wgsl);
      assertEquals(branch.includes("out[at] = 0.0;"), true, `${where}: the pad rows' 0 write`);
      assertEquals(
        branch.includes("if (lane == 0u && in_depth) {"),
        true,
        `${where}: the lane 0 guard`,
      );
      assertEquals(branch.includes("if (in_depth) {"), false, `${where}: a guard without the lane`);
    }
  }
});

const GEOMETRY = {
  rowsBlock: 4,
  rowOffset: 0,
  chunkRows: 8,
  depth: 6,
  kvRepeat: 2,
  window: 0,
  capacity: 32,
  colCap: 32,
  scale: 0.5,
};

test("the params of ①③ are packed in word order (the same geometry is not held in two tables)", () => {
  const params = stateAttentionParams({ ...GEOMETRY, rowOffset: 4 });
  assertEquals(params.length, 12, "48 bytes with uniform struct alignment");
  assertEquals([...params.subarray(0, 9)], [4, 4, 8, 6, 2, 0, 32, 32, STATE_NEG_INF_BITS]);
  assertEquals(new Float32Array(params.buffer)[9], 0.5);
  // ② has a separate struct (6 words: batch_heads / rows_block / row_offset / col_cap / window /
  // neg_inf)
  const stats = stateStatsParams(6, 4, 4, 32, 0);
  assertEquals(stats.length, 8, "32 bytes with uniform struct alignment");
  assertEquals([...stats.subarray(0, 6)], [6, 4, 4, 32, 0, STATE_NEG_INF_BITS]);
});

test("the params check of ①③ rejects every geometry that would give a silent wrong value", () => {
  // r = 0 (a u32 division by zero in WGSL does not trap but gives an implementation-defined value =
  // the kv plane goes wrong)
  assertThrows(() => stateAttentionParams({ ...GEOMETRY, kvRepeat: 0 }), CodegenError, "kv_repeat");
  // The row block overflows the chunk (rows of q / O wrap into another batch)
  assertThrows(
    () => stateAttentionParams({ ...GEOMETRY, rowOffset: 6, rowsBlock: 4 }),
    CodegenError,
    "row block",
  );
  // full: col_cap < C (a shape that passed the context-side check P + Q ≤ C overflows S)
  assertThrows(() => stateAttentionParams({ ...GEOMETRY, colCap: 31 }), CodegenError, "col_cap");
  // sliding: W > C
  assertThrows(
    () => stateAttentionParams({ ...GEOMETRY, window: 33, colCap: 40 }),
    CodegenError,
    "window",
  );
  // sliding: col_cap < (W−1) + M
  assertThrows(
    () => stateAttentionParams({ ...GEOMETRY, window: 8, colCap: 14 }),
    CodegenError,
    "col_cap",
  );
  assertEquals(stateAttentionParams({ ...GEOMETRY, window: 8, colCap: 15 }).length, 12);
  // Non-finite scale / outside the u32 range
  assertThrows(
    () => stateAttentionParams({ ...GEOMETRY, scale: Number.NaN }),
    CodegenError,
    "scale",
  );
  // MUST: finiteness is judged as f32 (1e39, finite in f64, becomes +Inf as an f32 word, and `0 *
  // Inf = NaN`)
  assertThrows(
    () => stateAttentionParams({ ...GEOMETRY, scale: 1e39 }),
    CodegenError,
    "scale must be a finite number as f32",
  );
  assertThrows(() => stateAttentionParams({ ...GEOMETRY, depth: 2 ** 32 }), CodegenError, "depth");
  assertThrows(() => stateStatsParams(0, 4, 0, 32, 0), CodegenError, "batch_heads");
  assertThrows(() => stateStatsParams(6, 0, 0, 32, 0), CodegenError, "rows_block");
  // ② also has the sliding value-range check (② does not take the sliding flag as a separate
  // argument, so it checks the relation W ≤ col_cap — if broken, live crosses col_cap and eats S of
  // the adjacent row)
  assertThrows(() => stateStatsParams(6, 4, 0, 8, 9), CodegenError, "col_cap");
  assertEquals(stateStatsParams(6, 4, 0, 8, 8).length, 8);
  // full (window = 0) is outside ②'s relation (the upper bound is held by the C check on the ①③
  // side)
  assertEquals(stateStatsParams(6, 4, 0, 8, 0).length, 8);
});

test("the params check of state_append (capacity, positive integers, u32 range)", () => {
  const base = { kvPlanes: 2, chunkRows: 4, depth: 3, capacity: 16, window: 0 };
  assertEquals([...stateAppendParams(base)], [2, 4, 3, 16, 0, 0, 0, 0]);
  assertEquals([...stateAppendParams({ ...base, window: 8 }).subarray(0, 5)], [2, 4, 3, 16, 8]);
  assertThrows(() => stateAppendParams({ ...base, window: 17 }), CodegenError, "window");
  assertThrows(() => stateAppendParams({ ...base, depth: 0 }), CodegenError, "positive integer");
  assertThrows(() => stateAppendParams({ ...base, capacity: -1 }), CodegenError, "capacity");
});

const DISPATCH = { batchHeads: 6, rowsBlock: 4, rowOffset: 0, depth: 32, window: 0 };
const LIMIT = 65535;

test("the dispatch geometry is proportional to the live columns and independent of capacity (the mechanism of ADR 0066 decision 3)", () => {
  // The x axis of ①QK = ⌈live / 16⌉. full grows in proportion to P
  assertEquals(stateQkWorkgroups(DISPATCH, 0, 16, LIMIT, "t")[0], 1);
  assertEquals(stateQkWorkgroups(DISPATCH, 16, 16, LIMIT, "t")[0], 2);
  assertEquals(stateQkWorkgroups(DISPATCH, 1008, 16, LIMIT, "t")[0], 64);
  // Monotonic (does not decrease with P or with Q)
  let previous = 0;
  for (const past of [0, 1, 7, 64, 512, 4096]) {
    const groups = stateQkWorkgroups(DISPATCH, past, 1, LIMIT, "t")[0];
    assertEquals(groups >= previous, true, `not monotonic at past=${past}`);
    previous = groups;
  }
  // sliding is **capped** at W (multiplying P by 8 keeps live at (W−1)+Q)
  const sliding = { ...DISPATCH, window: 64 };
  const cap = stateQkWorkgroups(sliding, 63, 16, LIMIT, "t")[0];
  assertEquals(stateQkWorkgroups(sliding, 512, 16, LIMIT, "t")[0], cap);
  assertEquals(stateQkWorkgroups(sliding, 131072, 16, LIMIT, "t")[0], cap);
  // The z axis is the static geometry itself (the row axis is checked by the workload check below)
  assertEquals(stateQkWorkgroups(DISPATCH, 100, 4, LIMIT, "t")[2], 6);
  // ③ is independent of the logical length (it writes pad rows too = full-write; the kernel cuts
  // the scan)
  assertEquals(statePvWorkgroups(DISPATCH, LIMIT, "t"), [2, 1, 6]);
  // ② is grid-stride, so exceeding the limit **degrades** (the separate handling of ② and ③ is
  // intact)
  assertEquals(stateStatsWorkgroups({ ...DISPATCH, batchHeads: 4096 }, 4, 100, "t"), [100, 1, 1]);
});

test("the row-direction dispatch of ①② is proportional to Q, not M (a block of pad only is 0)", () => {
  // A fixed M=8 split into 2 blocks of 4 rows, with only Q moving (neither capacity nor colCap
  // moves a single word)
  const first = { ...DISPATCH, rowsBlock: 4, rowOffset: 0 };
  const second = { ...DISPATCH, rowsBlock: 4, rowOffset: 4 };
  // The y axis of ①QK = ⌈effective rows / 4⌉
  assertEquals(stateQkWorkgroups(first, 0, 1, LIMIT, "t")[1], 1, "Q=1 is one tile");
  assertEquals(stateQkWorkgroups(first, 0, 4, LIMIT, "t")[1], 1);
  assertEquals(stateQkWorkgroups(second, 0, 4, LIMIT, "t")[1], 0, "a block of pad only is 0");
  assertEquals(stateQkWorkgroups(second, 0, 6, LIMIT, "t")[1], 1, "a block that is half effective");
  assertEquals(stateQkWorkgroups(second, 0, 8, LIMIT, "t")[1], 1);
  // ② is 1 row = 1 workgroup, so the row count comes out as is (B·H = 6)
  assertEquals(stateStatsWorkgroups(first, 1, LIMIT, "t"), [6, 1, 1], "Q=1 is 1 row × B·H");
  assertEquals(stateStatsWorkgroups(first, 4, LIMIT, "t"), [24, 1, 1], "Q=M is 4 rows × B·H");
  assertEquals(stateStatsWorkgroups(second, 4, LIMIT, "t"), [0, 1, 1], "a block of pad only is 0");
  assertEquals(stateStatsWorkgroups(second, 6, LIMIT, "t"), [12, 1, 1]);
  // MUST: only ③PV stays proportional to M (writing pad rows too is the full-write invariant)
  assertEquals(statePvWorkgroups(first, LIMIT, "t")[1], 1);
  assertEquals(statePvWorkgroups(second, LIMIT, "t")[1], 1);
});

test("the dispatch geometry does not take capacity C / col_cap as an argument (identical from 64 to 65536)", () => {
  // The measured version of "the type has no capacity field": building params that differ only in
  // capacity for the same (P, Q) changes not a single word of the geometry passed to dispatch (=
  // there is no way to write a workgroup count proportional to capacity)
  const small = stateAttentionParams({ ...GEOMETRY, capacity: 64, colCap: 64 });
  const large = stateAttentionParams({ ...GEOMETRY, capacity: 65536, colCap: 65536 });
  assertNotEquals([...small], [...large], "capacity does move on the params side");
  // A table **with** the capacity field is passed to the three families as is (structural typing
  // lets the extra field through). Today the geometry does not read capacity, so the 1024×
  // difference does not show in the output — the moment capacity enters the geometry (making the
  // column axis ⌈C/16⌉, for example), small / large split here.
  const smallCapacity = { ...DISPATCH, capacity: 64, colCap: 64 };
  const largeCapacity = { ...DISPATCH, capacity: 65536, colCap: 65536 };
  for (const [past, query] of [
    [0, 1],
    [7, 4],
    [513, 16],
  ] as const) {
    const where = `(P,Q)=(${past},${query})`;
    assertEquals(
      stateQkWorkgroups(smallCapacity, past, query, LIMIT, "t"),
      stateQkWorkgroups(largeCapacity, past, query, LIMIT, "t"),
      `①QK ${where}`,
    );
    assertEquals(
      stateStatsWorkgroups(smallCapacity, query, LIMIT, "t"),
      stateStatsWorkgroups(largeCapacity, query, LIMIT, "t"),
      `②stats ${where}`,
    );
  }
  assertEquals(
    statePvWorkgroups(smallCapacity, LIMIT, "t"),
    statePvWorkgroups(largeCapacity, LIMIT, "t"),
    "③PV",
  );
  assertEquals(Object.keys(DISPATCH).includes("capacity"), false);
  assertEquals(Object.keys(DISPATCH).includes("colCap"), false);
  assertEquals(Object.keys(DISPATCH).includes("chunkRows"), false, "M is not passed either");
});

test("exceeding the dispatch limit fails loudly for the tile kinds and degrades for the grid-stride kinds", () => {
  assertThrows(() => stateQkWorkgroups(DISPATCH, 100000, 1, 16, "t"), DispatchLimitError, "limit");
  assertThrows(
    () => statePvWorkgroups({ ...DISPATCH, batchHeads: 70000 }, 65535, "t"),
    DispatchLimitError,
    "limit",
  );
  // The value-range check of the logical length (Q ≥ 1, u32)
  assertThrows(() => stateQkWorkgroups(DISPATCH, 0, 0, LIMIT, "t"), CodegenError, "queryLength");
  assertThrows(() => stateQkWorkgroups(DISPATCH, -1, 1, LIMIT, "t"), CodegenError, "past");
});

test("the state_append dispatch is proportional to Q and independent of P", () => {
  const geometry = { kvPlanes: 4, depth: 64 };
  assertEquals(stateAppendWorkgroups(geometry, 1, LIMIT, "t"), [1, 1, 1]);
  assertEquals(stateAppendWorkgroups(geometry, 16, LIMIT, "t"), [16, 1, 1]);
  assertEquals(stateAppendWorkgroups(geometry, 64, LIMIT, "t"), [64, 1, 1]);
  // Grid-stride, so the limit degrades
  assertEquals(stateAppendWorkgroups(geometry, 64, 8, "t"), [8, 1, 1]);
  assertThrows(
    () => stateAppendWorkgroups(geometry, 0, LIMIT, "t"),
    CodegenError,
    "positive integer",
  );
});

test("pure functions of the live range (full is P+Q; sliding is capped at W)", () => {
  assertEquals(stateSliding(0), false);
  assertEquals(stateSliding(1), true);
  // full
  assertEquals(stateColumnBase(0, 100), 0);
  assertEquals(stateLiveColumns(0, 100, 8), 108);
  // sliding: with P < W−1, every past row is resident
  assertEquals(stateColumnBase(4, 2), 0);
  assertEquals(stateLiveColumns(4, 2, 1), 3);
  // sliding: with P ≥ W−1, only the last W−1 rows are resident (the boundary where row 0's window
  // fits entirely)
  assertEquals(stateColumnBase(4, 9), 6);
  assertEquals(stateLiveColumns(4, 9, 1), 4);
  assertEquals(stateLiveColumns(4, 1000, 1), 4);
});

/**
 * Input for the D=1 hand-computed oracles (with V of an in-window column set to its col number, the
 * output is "the window's average").
 */
const windowProbe = (options: {
  readonly window: number;
  readonly past: number;
  readonly query: number;
  readonly chunkRows: number;
  readonly capacity: number;
  /**
   * The poison value placed in physical rows that are not resident (the output jumps if it is
   * read).
   */
  readonly poison: number;
}): StateAttentionRefInput => {
  const { window, past, query, chunkRows, capacity, poison } = options;
  const slotK = new Float32Array(capacity);
  const slotV = new Float32Array(capacity).fill(poison);
  const insK = new Float32Array(chunkRows);
  const insV = new Float32Array(chunkRows).fill(poison);
  // K is all 1 (= every in-window column has the same score → uniform weights)
  slotK.fill(1);
  insK.fill(1);
  // V(col) = col, placed through the **logical col → physical row** mapping. Physical rows that are
  // not resident keep the poison
  const base = stateColumnBase(window, past);
  for (let col = base; col < past; col += 1) {
    slotV[stateSliding(window) ? col % capacity : col] = col;
  }
  for (let row = 0; row < query; row += 1) insV[row] = past + row;
  return {
    batch: 1,
    heads: 1,
    kvHeads: 1,
    chunkRows,
    depth: 1,
    capacity,
    window,
    past,
    query,
    q: new Float32Array(chunkRows).fill(1),
    insK,
    insV,
    slotK,
    slotV,
    scale: 1,
  };
};

test("CPU reference: the sliding window is cut on both sides (hand-computed — with only the upper bound the value jumps)", () => {
  // W=4 / P=5 / Q=1: row 0 sees col ∈ {2,3,4,5} (lower bound = 5−4+1 = 2). Average = 3.5. An
  // upper-bound-only implementation would mix in the equivalent of col 0,1 (= leftovers of physical
  // rows 0,1) and the value would jump
  const probe = windowProbe({
    window: 4,
    past: 5,
    query: 1,
    chunkRows: 1,
    capacity: 4,
    poison: 1e6,
  });
  const out = referenceStateAttention(probe);
  assertEquals(out.shape, [1, 1, 1, 1]);
  assertAlmostEquals(out.data[0], 3.5, 1e-6);
});

test("CPU reference: full runs from the start to the causal upper bound (pad rows are exactly 0)", () => {
  // P=3 / Q=2 / M=3: the logical col space is [0, P+Q) = {0..4} (the keys of pad rows **do not
  // enter the logical col space** — only Q are added). Row 0 → upper bound 3, {0,1,2,3} (average
  // 1.5); row 1 → upper bound 4, {0..4} (average 2); row 2 is a pad row, so **exactly 0** (it is
  // written = full-write, but the value is fixed at 0 — the value contract of ADR 0066 addendum 6)
  const probe = windowProbe({
    window: 0,
    past: 3,
    query: 2,
    chunkRows: 3,
    capacity: 8,
    poison: 0,
  });
  const out = referenceStateAttention(probe);
  assertAlmostEquals(out.data[0], 1.5, 1e-6);
  assertAlmostEquals(out.data[1], 2, 1e-6);
  assertEquals(Object.is(out.data[2], 0), true, "a pad row is exactly 0 (neither −0 nor NaN)");
});

test("CPU reference: the output of an empty row (a pad row with no col satisfying the predicate) is exactly 0", () => {
  // W=2 / P=0 / Q=1 / M=3: row 2 needs col ≥ 1 as its lower bound, but live is only {0} = an empty
  // row
  const probe = windowProbe({
    window: 2,
    past: 0,
    query: 1,
    chunkRows: 3,
    capacity: 2,
    poison: 1e6,
  });
  const out = referenceStateAttention(probe);
  assertEquals(Object.is(out.data[2], 0), true, "an empty row is exactly 0 (neither −0 nor NaN)");
  // Row 0 sees only col 0 (V(0) = 0); for row 1, col 0 of cols 0,1 drops at the lower bound…
  // live = {0}, so row 1 also sees only col 0 (P=0, so col 0 is ins row 0)
  assertAlmostEquals(out.data[0], 0, 1e-6);
});

test("CPU reference: V of columns outside the predicate is not read (NaN outside the causal upper bound / the window's lower bound does not change the value)", () => {
  // As in the formula O[row] = Σ_{col in the predicate} P · V(col), V outside the predicate is not
  // in the sum. A form that walks the live range adding 0 × V would turn that row into NaN through
  // 0 × NaN of a NaN outside the predicate.
  // Common: K / q are all 1 (uniform weights inside the predicate), V(col) = col, D=1. NaN is also
  // placed in old physical rows that have wrapped and in physical rows past the logical length (no
  // form may read them).
  const cases = [
    {
      // full, P=3 / Q=2: row 0's predicate is {0..3} (average 1.5). col 4 (ins row 1) is outside
      // the upper bound
      name: "outside the full causal upper bound",
      window: 0,
      capacity: 8,
      nanColumn: 4,
      row: 0,
      expected: 1.5,
    },
    {
      // W=2, C=4, P=3 / Q=2: row 0 is {2,3} (average 2.5). col 4 is outside the upper bound
      name: "outside the sliding causal upper bound",
      window: 2,
      capacity: 4,
      nanColumn: 4,
      row: 0,
      expected: 2.5,
    },
    {
      // W=2, C=4, P=3 / Q=2: row 1 is {3,4} (average 3.5). col 2 is outside the window's lower
      // bound (a column row 0 reads)
      name: "outside the sliding window's lower bound",
      window: 2,
      capacity: 4,
      nanColumn: 2,
      row: 1,
      expected: 3.5,
    },
  ];
  const [past, query, chunkRows] = [3, 2, 2];
  for (const { name, window, capacity, nanColumn, row, expected } of cases) {
    const slotRowOf = (col: number) => (stateSliding(window) ? col % capacity : col);
    const slotV = new Float32Array(capacity).fill(Number.NaN);
    const insV = new Float32Array(chunkRows);
    for (let col = 0; col < past + query; col += 1) {
      const value = col === nanColumn ? Number.NaN : col;
      if (col < past) {
        // For sliding, the physical rows of columns that left the window (col < P − W + 1) stay NaN
        if (!stateSliding(window) || col > past - window) slotV[slotRowOf(col)] = value;
      } else {
        insV[col - past] = value;
      }
    }
    const out = referenceStateAttention({
      batch: 1,
      heads: 1,
      kvHeads: 1,
      chunkRows,
      depth: 1,
      capacity,
      window,
      past,
      query,
      q: new Float32Array(chunkRows).fill(1),
      insK: new Float32Array(chunkRows).fill(1),
      insV,
      slotK: new Float32Array(capacity).fill(1),
      slotV,
      scale: 1,
    });
    assertEquals(out.data[row], expected, name);
  }
});

/**
 * The GPU's scan form (walk the live range `[stateColumnBase, +stateLiveColumns)` and add weight
 * 0 × V for columns outside the predicate too), written out on the CPU. The reference walks every
 * logical column and skips those outside the predicate, so the two being bit-identical for finite V
 * is the basis for "changing the scan form does not move the value".
 */
const liveRangeAttention = (input: StateAttentionRefInput): Float32Array<ArrayBuffer> => {
  const { batch, heads, kvHeads, chunkRows, depth, capacity, window, past, query } = input;
  const scale = Math.fround(input.scale);
  const base = stateColumnBase(window, past);
  const live = stateLiveColumns(window, past, query);
  const slotRowOf = (col: number) => (stateSliding(window) ? col % capacity : col);
  const rowOf = (kvPlane: number, col: number) =>
    col < past
      ? { source: "slot" as const, offset: (kvPlane * capacity + slotRowOf(col)) * depth }
      : { source: "ins" as const, offset: (kvPlane * chunkRows + (col - past)) * depth };
  const out = new Float32Array(batch * heads * chunkRows * depth);
  const scores = new Float32Array(live);
  const weights = new Float32Array(live);
  for (let plane = 0; plane < batch * heads; plane += 1) {
    const kvPlane = Math.floor(plane / (heads / kvHeads));
    for (let row = 0; row < query; row += 1) {
      const qBase = (plane * chunkRows + row) * depth;
      const limit = past + row;
      for (let cl = 0; cl < live; cl += 1) {
        const col = base + cl;
        if (!(col <= limit && (!stateSliding(window) || limit - col < window))) {
          scores[cl] = Number.NEGATIVE_INFINITY;
          continue;
        }
        const { source, offset } = rowOf(kvPlane, col);
        const k = source === "slot" ? input.slotK : input.insK;
        let acc = 0;
        for (let d = 0; d < depth; d += 1) {
          acc += input.q[qBase + d] * scale * (k[offset + d] * scale);
        }
        scores[cl] = Math.fround(acc);
      }
      let amax = Number.NEGATIVE_INFINITY;
      for (let cl = 0; cl < live; cl += 1) amax = Math.max(amax, scores[cl]);
      if (amax === Number.NEGATIVE_INFINITY) continue;
      let total = 0;
      for (let cl = 0; cl < live; cl += 1) total += Math.exp(scores[cl] - amax);
      for (let cl = 0; cl < live; cl += 1) {
        weights[cl] = Math.fround(Math.exp(scores[cl] - amax) / total);
      }
      for (let d = 0; d < depth; d += 1) {
        let acc = 0;
        for (let cl = 0; cl < live; cl += 1) {
          const { source, offset } = rowOf(kvPlane, base + cl);
          acc += weights[cl] * (source === "slot" ? input.slotV : input.insV)[offset + d];
        }
        out[qBase + d] = Math.fround(acc);
      }
    }
  }
  return out;
};

test("CPU reference: for finite V, bit-identical to the form that adds 0 × V outside the predicate within the live range", () => {
  // Sweep: full / sliding (capacity = window, and window + 2 to create old wrapped rows) × P × Q.
  // Every physical row (including outside the window and past the logical length) gets a finite
  // random value — this binds that V outside the predicate does not show in the value.
  // P also takes values near the top of u32 (logical column numbers above 2^31, a shape where one
  // cannot walk [0, P+Q) down to 0).
  let seed = 0x2545f491;
  const random = (): number => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return (seed / 2 ** 32) * 4 - 2;
  };
  const filled = (count: number) => Float32Array.from({ length: count }, random);
  const [batch, heads, kvHeads, chunkRows, depth] = [1, 2, 1, 3, 3];
  let compared = 0;
  for (const window of [0, 1, 2, 3, 5]) {
    const capacities = stateSliding(window) ? [window, window + 2] : [12];
    for (const capacity of capacities) {
      for (const past of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0xfffffffe]) {
        for (let query = 1; query <= chunkRows; query += 1) {
          if (!stateSliding(window) && past + query > capacity) continue;
          const input: StateAttentionRefInput = {
            batch,
            heads,
            kvHeads,
            chunkRows,
            depth,
            capacity,
            window,
            past,
            query,
            q: filled(batch * heads * chunkRows * depth),
            insK: filled(batch * kvHeads * chunkRows * depth),
            insV: filled(batch * kvHeads * chunkRows * depth),
            slotK: filled(batch * kvHeads * capacity * depth),
            slotV: filled(batch * kvHeads * capacity * depth),
            scale: 0.5,
          };
          const actual = referenceStateAttention(input).data;
          const expected = liveRangeAttention(input);
          assertEquals(
            new Uint32Array(actual.buffer),
            new Uint32Array(expected.buffer),
            `W=${window} C=${capacity} P=${past} Q=${query}`,
          );
          compared += 1;
        }
      }
    }
  }
  // The sweep is not running empty (full is 10 P × 3 Q within capacity 12; sliding is 8 capacities
  // × 11 P × 3 Q)
  assertEquals(compared, 30 + 264);
});

test("CPU reference: GQA maps the kv plane by ⌊z/r⌋ (a modulo mapping changes the value)", () => {
  const [batch, heads, kvHeads, chunkRows, depth, capacity] = [2, 4, 2, 1, 1, 1];
  const repeat = heads / kvHeads;
  // A different V per kv plane (z=0,1 → plane 0 / z=2,3 → plane 1 / …)
  const insV = Float32Array.from([10, 20, 30, 40]);
  const out = referenceStateAttention({
    batch,
    heads,
    kvHeads,
    chunkRows,
    depth,
    capacity,
    window: 0,
    past: 0,
    query: 1,
    q: new Float32Array(batch * heads * chunkRows * depth).fill(1),
    insK: new Float32Array(batch * kvHeads * chunkRows * depth).fill(1),
    insV,
    slotK: new Float32Array(batch * kvHeads * capacity * depth),
    slotV: new Float32Array(batch * kvHeads * capacity * depth),
    scale: 1,
  });
  // For z = b·H + h, ⌊z/r⌋ = b·Hkv + ⌊h/r⌋ (the B term matters — the modulo z%Hkv would drop b)
  for (let z = 0; z < batch * heads; z += 1) {
    assertAlmostEquals(out.data[z], insV[Math.floor(z / repeat)], 1e-6, `z=${z}`);
  }
});

test("CPU reference state_append: writes only the first Q rows, overwrites old rows on wrap, and does not destroy the input", () => {
  const spec = { kvPlanes: 1, chunkRows: 4, depth: 1, capacity: 4, window: 4 };
  const slot = Float32Array.from([-1, -2, -3, -4]);
  const x = Float32Array.from([100, 200, 300, 400]);
  // P=2 / Q=2 → logical rows 2,3 = physical rows 2,3. Pad rows (2,3) are not written
  const after = referenceStateAppend({ ...spec, past: 2, query: 2, x, slot });
  assertEquals([...after.data], [-1, -2, 100, 200]);
  assertEquals([...slot], [-1, -2, -3, -4], "the input slot is not destroyed");
  // wrap: P=3 / Q=3 → logical rows 3,4,5 = physical rows 3,0,1
  const wrapped = referenceStateAppend({ ...spec, past: 3, query: 3, x, slot });
  assertEquals([...wrapped.data], [200, 300, -3, 100]);
  // Q > W: several logical rows map to the same physical row → **the last logical row remains**
  const collide = referenceStateAppend({
    ...spec,
    window: 2,
    capacity: 2,
    past: 0,
    query: 4,
    x,
    slot: Float32Array.from([-1, -2]),
  });
  assertEquals([...collide.data], [300, 400]);
});

test("CPU reference: mixed-up shapes fail before the comparison", () => {
  const probe = windowProbe({
    window: 0,
    past: 1,
    query: 1,
    chunkRows: 1,
    capacity: 4,
    poison: 0,
  });
  assertThrows(
    () => referenceStateAttention({ ...probe, q: new Float32Array(2) }),
    ReferenceOpError,
    "q",
  );
  assertThrows(
    () => referenceStateAttention({ ...probe, heads: 3, kvHeads: 2 }),
    ReferenceOpError,
    "not divisible",
  );
  assertThrows(
    () => referenceStateAttention({ ...probe, past: 4, capacity: 4 }),
    ReferenceOpError,
    "capacity",
  );
  assertThrows(
    () => referenceStateAttention({ ...probe, query: 2 }),
    ReferenceOpError,
    "queryLength",
  );
});

test("CPU reference: a negative or non-integer pastLength / window fails before the comparison instead of turning into full or an empty row", () => {
  // Without the check, window = -1 makes stateSliding false and is silently treated as full, and
  // past = -1 gives zero live columns and returns exactly 0 (append silently writes to the
  // out-of-range physical row -1 and discards it).
  const probe = windowProbe({
    window: 0,
    past: 1,
    query: 1,
    chunkRows: 1,
    capacity: 4,
    poison: 0,
  });
  const append = {
    kvPlanes: 1,
    chunkRows: 1,
    depth: 1,
    capacity: 4,
    window: 0,
    past: 1,
    query: 1,
    x: new Float32Array(1),
    slot: new Float32Array(4),
  };
  const readonly = {
    batch: 1,
    heads: 1,
    kvHeads: 1,
    depth: 1,
    capacity: 4,
    window: 0,
    past: 1,
    q: new Float32Array(1),
    slotK: new Float32Array(4),
    slotV: new Float32Array(4),
    scale: 1,
  };
  const cases: readonly [string, () => unknown, string][] = [
    ["attention window=-1", () => referenceStateAttention({ ...probe, window: -1 }), "window"],
    ["attention window=1.5", () => referenceStateAttention({ ...probe, window: 1.5 }), "window"],
    ["attention past=-1", () => referenceStateAttention({ ...probe, past: -1 }), "pastLength"],
    ["attention past=0.5", () => referenceStateAttention({ ...probe, past: 0.5 }), "pastLength"],
    ["append window=-1", () => referenceStateAppend({ ...append, window: -1 }), "window"],
    ["append past=-1", () => referenceStateAppend({ ...append, past: -1 }), "pastLength"],
    [
      "readonly window=-1",
      () => referenceStateAttentionReadonly({ ...readonly, window: -1 }),
      "window",
    ],
    [
      "readonly past=-1",
      () => referenceStateAttentionReadonly({ ...readonly, past: -1 }),
      "pastLength",
    ],
  ];
  for (const [name, run, message] of cases) {
    assertThrows(run, ReferenceOpError, message, name);
  }
});

test("CPU reference: a NaN or non-integer queryLength fails before the comparison instead of turning into an all-0 or NaN expectation", () => {
  // With range comparisons alone, NaN slips through because every comparison is false (the row loop
  // runs 0 times → the output is all 0), and 1.5 returns NaN with a fractional live column count —
  // both are "the oracle silently returns a different number".
  const probe = {
    batch: 1,
    heads: 1,
    kvHeads: 1,
    chunkRows: 2,
    depth: 1,
    capacity: 2,
    window: 0,
    past: 0,
    query: 1,
    q: Float32Array.from([1, 1]),
    insK: Float32Array.from([1, 1]),
    insV: Float32Array.from([4, 8]),
    slotK: new Float32Array(2),
    slotV: new Float32Array(2),
    scale: 1,
  };
  const append = {
    kvPlanes: 1,
    chunkRows: 2,
    depth: 1,
    capacity: 2,
    window: 0,
    past: 0,
    query: 1,
    x: Float32Array.from([4, 8]),
    slot: new Float32Array(2),
  };
  for (const query of [Number.NaN, 1.5]) {
    assertThrows(
      () => referenceStateAttention({ ...probe, query }),
      ReferenceOpError,
      "queryLength",
      `attention query=${query}`,
    );
    assertThrows(
      () => referenceStateAppend({ ...append, query }),
      ReferenceOpError,
      "queryLength",
      `append query=${query}`,
    );
  }
});

test("CPU reference: NaN or non-integer dimensions fail as shape errors instead of turning into an expectation with a fractional shape", () => {
  // A non-integer passes even the array-length check when the product of the dimensions is an
  // integer (with D=1.5 and M=2, q has 3 elements). NaN is caught by the array-length check by
  // chance, but the message does not say it is a shape error — both fail at the dimension check.
  const filled = (count: number): Float32Array<ArrayBuffer> => new Float32Array(count).fill(1);
  const base = {
    batch: 1,
    heads: 1,
    kvHeads: 1,
    chunkRows: 2,
    depth: 1,
    capacity: 2,
    window: 0,
    past: 0,
    query: 1,
    scale: 1,
  };
  const cases = [
    { name: "D=1.5", dims: { depth: 1.5 }, insLength: 3, slotLength: 3 },
    { name: "C=2.5", dims: { depth: 2, capacity: 2.5 }, insLength: 4, slotLength: 5 },
    { name: "B=NaN", dims: { batch: Number.NaN }, insLength: 2, slotLength: 2 },
  ];
  for (const { name, dims, insLength, slotLength } of cases) {
    assertThrows(
      () =>
        referenceStateAttention({
          ...base,
          ...dims,
          q: filled(insLength),
          insK: filled(insLength),
          insV: filled(insLength),
          slotK: filled(slotLength),
          slotV: filled(slotLength),
        }),
      ReferenceOpError,
      "not all positive integers",
      name,
    );
  }
});

test("the parallel-reduction variant of ①' has `:par` in its key and shares the contract fragments of ① as is", () => {
  // The variant's marker (the census check looks for this `:par`). The geometry `wg16x16` = TILE_X
  // × D lanes
  assertEquals(stateQkParallelKey(false, false), "attention_state_qk:v1:f32:wg16x16:par");
  assertEquals(stateQkParallelKey(true, true), "attention_state_qk:v1:f32:wg16x16:par:sliding:gqa");
  assertEquals(STATE_QK_D_LANES, 16);
  for (const [sliding, gqa] of VARIANTS) {
    // ① and ①' have different keys (one configuration does not have two keys / two configurations
    // do not share a key)
    assertNotEquals(stateQkParallelKey(sliding, gqa), stateQkKey(sliding, gqa));
    assertEquals(
      stateQkParallelWgsl(sliding, gqa),
      stateQkParallelWgsl(sliding, gqa),
      "determinism",
    );
    const wgsl = stateQkParallelWgsl(sliding, gqa);
    // MUST: only the arrangement of the reduction differs. The shared read/write formula, the
    // effective rows, the −inf write and avoiding the u32 wrap-around are **the same fragments** as
    // ① (rejects fixing only one side by string)
    assertEquals(wgsl.includes(stateSlotRowWgsl(sliding)), true, "slot_row");
    assertEquals(wgsl.includes("fn effective_rows(query: u32) -> u32 {"), true, "effective_rows");
    assertEquals(wgsl.includes("bitcast<f32>(params.neg_inf)"), true, "−inf outside the predicate");
    assertEquals(
      wgsl.includes("col + params.window"),
      false,
      "the addition form that wraps around in u32",
    );
  }
  assertEquals(stateQkParallelWgsl(true, false).includes("(limit - col) < params.window"), true);
  assertEquals(stateQkParallelWgsl(false, false).includes("(limit - col) < params.window"), false);
});

test("the dispatch geometry of ①' has the same two axes as ① (columns = live, rows = effective rows), only the row tile width is 1", () => {
  // The column axis is ⌈live / TILE_X⌉ as in ① (the D direction is lanes inside the workgroup, so
  // it does not appear in dispatch)
  assertEquals(stateQkParallelWorkgroups(DISPATCH, 1008, 16, LIMIT, "t")[0], 64);
  const first = { ...DISPATCH, rowsBlock: 4, rowOffset: 0 };
  // The row axis is 1 row = 1 workgroup (for ①, ⌈effective rows / TILE_M⌉ = 1)
  assertEquals(stateQkParallelWorkgroups(first, 0, 4, LIMIT, "t")[1], 4);
  assertEquals(stateQkWorkgroups(first, 0, 4, LIMIT, "t")[1], 1);
  assertEquals(stateQkParallelWorkgroups(first, 0, 1, LIMIT, "t")[1], 1, "decode is one row");
  // A block of pad only is 0 (the caller does not record the dispatch at all)
  const second = { ...DISPATCH, rowsBlock: 4, rowOffset: 4 };
  assertEquals(stateQkParallelWorkgroups(second, 0, 4, LIMIT, "t")[1], 0);
  // The z axis is B·H, and the column axis does not depend on capacity (only live matters = the
  // workload acceptance condition)
  assertEquals(stateQkParallelWorkgroups(DISPATCH, 100, 4, LIMIT, "t")[2], 6);
  // MUST: exceeding the limit fails loudly (a tile kind — degrading would leave S tiles missing and
  // ②③ would read leftovers)
  assertThrows(() => stateQkParallelWorkgroups(DISPATCH, 4096, 1, 8, "t"), DispatchLimitError);
});

test("the condition for ①' is M (chunkRows) ≤ 8 — no extrapolation beyond the measured range", () => {
  // The decode plan (M=1) and the speculative verify plan (M = k+1 ≤ 8) qualify. This is the only
  // applicability condition, and the runtime side ANDs it with the option check. M=1 was measured
  // to help (2026-09-06) and M=4 to be neutral (2026-09-09; with the default option, row 0 of
  // verify and decode share the reduction order); 8 is the verify bucket limit
  for (const chunkRows of [1, 2, 3, 4, 8]) {
    assertEquals(stateQkParallelEligible(chunkRows), true, `M=${chunkRows} uses ①'`);
  }
  // MUST: reject the extrapolation "a small M should pay off". At prefill width (M=768) the GPU
  // time of ①QK grew 1.5 to 1.9 times (2026-09-06). M ≥ 9 is unmeasured, so it stays on ① — even if
  // loosened, the values stay inside the tolerance band and neither the numerics check nor the
  // decode side of the census check would fire
  for (const chunkRows of [9, 16, 768]) {
    assertEquals(stateQkParallelEligible(chunkRows), false, `M=${chunkRows} stays on ①`);
  }
});

describe("the folded ①' — the F query heads of one kv plane per workgroup (perf-ledger K-55)", () => {
  it("is selected with F = r only on the measured full-attention classes (r, D, M)", () => {
    // [r, D, M, F]. The classes measured not slower than ①' at P = 1,024 / 4,000 / 16,000 and
    // observed bit-identical to ①' on every backend checked (Vulkan RTX 3080 Ti, D3D12 RTX 5070 Ti
    // — 2026-10-27): the Gemma 4 E2B full layer (r = 8) at decode M = 1 and verify M = 4 / 8, and
    // the MiniCPM5 full layer (r = 8, D = 128) at M = 1
    const folded: readonly (readonly [number, number, number, number])[] = [
      [8, 512, 1, 8],
      [8, 512, 4, 8],
      [8, 512, 8, 8],
      [8, 128, 1, 8],
    ];
    for (const [kvRepeat, depth, chunkRows, heads] of folded) {
      const label = `full r=${kvRepeat} D=${depth} M=${chunkRows}`;
      assertEquals(stateQkFoldHeads(false, kvRepeat, depth, chunkRows), heads, label);
      // The fold was neutral to slower on the sliding layers, so they never fold
      assertEquals(stateQkFoldHeads(true, kvRepeat, depth, chunkRows), 1, `sliding ${label}`);
    }
    // MUST: no extrapolation — every unmeasured class keeps the plain ①' (F = 1)
    const plain: readonly (readonly [number, number, number])[] = [
      // other repeat counts
      [1, 512, 1],
      [2, 512, 1],
      [3, 512, 1],
      [16, 512, 1],
      // chunk row counts between the measured ones
      [8, 512, 2],
      [8, 512, 3],
      [4, 512, 5],
      [8, 512, 7],
      // MiniCPM5 verify-sized plans (only M = 1 was measured at D = 128)
      [8, 128, 4],
      [8, 128, 8],
      // other head dims
      [8, 256, 1],
      [4, 128, 1],
      [8, 64, 1],
      // The Gemma 4 E4B full layer (r = 4) was measured faster, but its F = 4 kernel differs from
      // ①' on D3D12 (perf-ledger K-95), so it keeps ①'
      [4, 512, 1],
      [4, 512, 4],
      [4, 512, 8],
    ];
    for (const [kvRepeat, depth, chunkRows] of plain) {
      assertEquals(
        stateQkFoldHeads(false, kvRepeat, depth, chunkRows),
        1,
        `full r=${kvRepeat} D=${depth} M=${chunkRows}`,
      );
    }
  });

  it("fails loudly when F does not divide kv_repeat or has no lane to write it", () => {
    assertStateQkFold(8, 8, "t");
    assertStateQkFold(4, 4, "t");
    assertStateQkFold(4, 8, "t");
    // A group of F heads straddling two kv planes would read another plane's K silently
    assertThrows(() => assertStateQkFold(8, 4, "t"), CodegenError, "divide kv_repeat");
    assertThrows(() => assertStateQkFold(3, 8, "t"), CodegenError, "divide kv_repeat");
    assertThrows(() => assertStateQkFold(4, 6.5, "t"), CodegenError, "divide kv_repeat");
    // Lane f writes head f, so F is at most D_LANES; F = 1 is the plain ①', not a fold
    assertThrows(() => assertStateQkFold(1, 8, "t"), CodegenError, "heads per workgroup");
    assertThrows(() => assertStateQkFold(32, 32, "t"), CodegenError, "heads per workgroup");
    // Every F the selection returns passes the check for its own r
    for (let kvRepeat = 1; kvRepeat <= 32; kvRepeat += 1) {
      for (const depth of [128, 512]) {
        for (let chunkRows = 1; chunkRows <= 8; chunkRows += 1) {
          const heads = stateQkFoldHeads(false, kvRepeat, depth, chunkRows);
          if (heads > 1)
            assertStateQkFold(heads, kvRepeat, `r=${kvRepeat} D=${depth} M=${chunkRows}`);
        }
      }
    }
  });

  it("carries F in the key and keeps the `:par` census marker and the `:gqa` bit", () => {
    assertEquals(stateQkFoldedKey(8), "attention_state_qk:v1:f32:wg8x16:par:fold8:gqa");
    assertEquals(stateQkFoldedKey(4), "attention_state_qk:v1:f32:wg8x16:par:fold4:gqa");
    assertEquals(STATE_QK_FOLD_TILE_X, 8);
    // r travels in the uniform, so one key per F: two F must not share a key, nor F with ①'
    assertNotEquals(stateQkFoldedKey(4), stateQkFoldedKey(8));
    for (const [sliding, gqa] of VARIANTS) {
      assertNotEquals(stateQkFoldedKey(8), stateQkParallelKey(sliding, gqa));
    }
    assertEquals(`${stateQkFoldedKey(8)}:`.includes(":par:"), true, "census marker");
  });

  it("builds every head's terms from ①''s one-term expression and keeps ①''s write guards", () => {
    // The term ①' sums, in the score function the folded kernel must match
    const parallel = stateQkParallelWgsl(false, true);
    assertEquals(
      parallel.includes(
        "acc = acc + (q[q_base + d] * params.scale) * (slot_k[k_base + d] * params.scale);",
      ),
      true,
      "①' term",
    );
    for (const heads of [4, 8]) {
      const wgsl = stateQkFoldedWgsl(heads);
      assertEquals(stateQkFoldedWgsl(heads), wgsl, "determinism");
      // One K load per d, the same half-scaled k as ①'
      assertEquals(wgsl.includes("let kd = slot_k[k_base + d] * params.scale;"), true, "slot k");
      assertEquals(wgsl.includes("let kd = ins_k[k_base + d] * params.scale;"), true, "ins k");
      for (let f = 0; f < heads; f += 1) {
        assertEquals(
          wgsl.includes(`acc${f} = acc${f} + (q[q_base${f} + d] * params.scale) * kd;`),
          true,
          `F=${heads}: the term of head ${f}`,
        );
      }
      assertEquals(wgsl.includes(`var acc${heads} =`), false, `F=${heads}: exactly F heads`);
      // MUST: no fused multiply-add written by hand (it would change the rounding against ①')
      assertEquals(wgsl.includes("fma("), false, `F=${heads}: no fma`);
      // The same per-lane step and tree as ①'
      assertEquals(wgsl.includes("d = d + 16u"), true, `F=${heads}: lane step`);
      assertEquals(wgsl.includes("var stride = 8u;"), true, `F=${heads}: tree`);
      // The plane of the group (the shared head → kv plane fragment on z = the group's first head)
      // and ①''s write predicates (lane f writes head f)
      assertEquals(wgsl.includes(`let z = wid.z * ${heads}u;`), true, "first head");
      assertEquals(wgsl.includes("let kv_plane = z / params.kv_repeat;"), true, "kv plane");
      assertEquals(wgsl.includes(`if (lane < ${heads}u && cl < live) {`), true, "writers");
      assertEquals(wgsl.includes("bitcast<f32>(params.neg_inf)"), true, "-inf outside");
      assertEquals(wgsl.includes(stateSlotRowWgsl(false)), true, "slot_row");
      assertEquals(wgsl.includes("fn effective_rows(query: u32) -> u32 {"), true, "rows");
      assertEquals(wgsl.includes("@workgroup_size(8, 16)"), true, "workgroup");
      assertEquals(wgsl.includes(`array<f32, ${heads * 16 * 8}>`), true, "shared storage");
    }
    assertThrows(() => stateQkFoldedWgsl(1), CodegenError, "heads per workgroup");
    assertThrows(() => stateQkFoldedWgsl(17), CodegenError, "heads per workgroup");
  });

  it("dispatches ⌈live / 8⌉ columns × valid rows × B·H / F workgroups", () => {
    const geometry = { batchHeads: 16, rowsBlock: 4, rowOffset: 0, depth: 512, window: 0 };
    // Column axis: the live columns in tiles of 8 (①' uses 16); z axis: one workgroup per F heads
    assertEquals(stateQkFoldedWorkgroups(geometry, 8, 1000, 1, LIMIT, "t"), [126, 1, 2]);
    assertEquals(stateQkParallelWorkgroups(geometry, 1000, 1, LIMIT, "t"), [63, 1, 16]);
    assertEquals(stateQkFoldedWorkgroups(geometry, 4, 1000, 1, LIMIT, "t"), [126, 1, 4]);
    assertEquals(stateQkFoldedWorkgroups(geometry, 8, 0, 1, LIMIT, "t"), [1, 1, 2]);
    // Row axis: one workgroup per valid row; a block of pad rows only gets 0 (no dispatch)
    assertEquals(stateQkFoldedWorkgroups(geometry, 8, 0, 4, LIMIT, "t")[1], 4);
    const second = { ...geometry, rowOffset: 4 };
    assertEquals(stateQkFoldedWorkgroups(second, 8, 0, 4, LIMIT, "t")[1], 0);
    // MUST: B·H not a multiple of F fails loudly (a partial group would leave heads unwritten)
    assertThrows(
      () => stateQkFoldedWorkgroups({ ...geometry, batchHeads: 12 }, 8, 0, 1, LIMIT, "t"),
      CodegenError,
      "multiple",
    );
    // MUST: exceeding the limit fails loudly (tiled, as ①')
    assertThrows(() => stateQkFoldedWorkgroups(geometry, 8, 4096, 1, 8, "t"), DispatchLimitError);
  });
});

describe("the unrolled KV loop of ③' and the fused stats + PV (perf-ledger K-91)", () => {
  it("keys ③' and the fused kernel as v2 while ③ and the readonly ③' stay v1", () => {
    assertEquals(statePvParallelKey(false, false), "attention_state_pv:v2:f32:wg16x16:par");
    assertEquals(
      statePvParallelKey(true, true),
      "attention_state_pv:v2:f32:wg16x16:par:sliding:gqa",
    );
    assertEquals(statePvKey(false, false), "attention_state_pv:v1:f32:wg16x4");
    assertEquals(
      statePvParallelReadonlyKey(false, false),
      "attention_state_pv:v1:f32:wg16x16:par:ro",
    );
    assertEquals(stateStatsPvKey(false, false), "attention_state_stats_pv:v2:f32:wg16x16");
  });

  it("shares one scan, built from v1's one-term expression, between ③' and the fused kernel", () => {
    const scan = statePvScanWgsl();
    assertEquals(STATE_PV_UNROLL, 8);
    for (const [sliding, gqa] of VARIANTS) {
      assertEquals(statePvParallelWgsl(sliding, gqa).includes(scan), true, "③'");
      assertEquals(stateStatsPvWgsl(sliding, gqa).includes(scan), true, "fused");
      // The readonly ③' keeps its single-column loop (not part of this change)
      assertEquals(statePvParallelReadonlyWgsl(sliding, gqa).includes(scan), false, "readonly");
    }
    // A batch of 8 columns of one lane (cl, cl + 16, …, cl + 112), then the single-column tail
    assertEquals(scan.includes("while (cl + 112u < live) {"), true, "batch bound");
    assertEquals(scan.includes("cl = cl + 128u;"), true, "batch step");
    assertEquals(scan.includes("cl = cl + 16u;"), true, "tail step");
    // MUST: one addition per column with v1's expression, in ascending column order
    const terms = Array.from({ length: 8 }, (_, u) => {
      const at = u === 0 ? "cl" : `cl + ${16 * u}u`;
      return `acc = acc + (exp(s[s_base + ${at}] - amax) * inv) * v${u};`;
    });
    let from = 0;
    for (const term of terms) {
      const index = scan.indexOf(term, from);
      assertNotEquals(index, -1, `term in order: ${term}`);
      from = index + term.length;
    }
    assertEquals(scan.includes("fma("), false, "no fma");
  });
});

const TILED_DISPATCH = { batchHeads: 6, rowsBlock: 40, rowOffset: 0, depth: 32, window: 0 };

test("the key of ①ₜ carries the M geometry bucket (the geometry itself, not just the edge)", () => {
  // The spelling is identical to gemm's geometry fragment
  // (`reg{tileM}x{tileN}r{regM}x{regN}w{wgX}`). The variant bits are appended in the same order as
  // for ①
  assertEquals(stateQkTiledKey(false, false, 16), "attention_state_qk:v1:f32:reg16x16r1x4w4");
  assertEquals(
    stateQkTiledKey(true, true, 16),
    "attention_state_qk:v1:f32:reg16x16r1x4w4:sliding:gqa",
  );
  // Crossing a bucket boundary (64 / 512) **changes the key** (= a different pipeline). Otherwise,
  // WGSL of a different geometry would be handed out of the cache under the same key, would not
  // mesh with the dispatch edge, and tiles would go missing
  assertEquals(stateQkTiledKey(false, false, 64), stateQkTiledKey(false, false, 16));
  assertEquals(stateQkTiledKey(false, false, 65), "attention_state_qk:v1:f32:reg64x32r4x4w8");
  assertEquals(stateQkTiledKey(false, false, 512), stateQkTiledKey(false, false, 65));
  assertEquals(stateQkTiledKey(false, false, 513), "attention_state_qk:v1:f32:reg128x128r8x8w16");
  assertEquals(stateQkTiledKey(false, false, 768), stateQkTiledKey(false, false, 513));
  // All three paths have different keys (one configuration does not have two keys / two
  // configurations do not share a key)
  const keys = [
    ...VARIANTS.map(([sliding, gqa]) => stateQkKey(sliding, gqa)),
    ...VARIANTS.map(([sliding, gqa]) => stateQkParallelKey(sliding, gqa)),
    ...VARIANTS.flatMap(([sliding, gqa]) =>
      [16, 100, 768].map((rows) => stateQkTiledKey(sliding, gqa, rows)),
    ),
  ];
  assertEquals(new Set(keys).size, keys.length);
});

test("①ₜ shares the contract fragments of ① (shared read/write formula, predicate, effective rows, −inf) with only the uniform name changed", () => {
  for (const [sliding, gqa] of VARIANTS) {
    const wgsl = stateQkTiledWgsl(sliding, gqa, 16);
    assertEquals(stateQkTiledWgsl(sliding, gqa, 16), wgsl, "determinism");
    // MUST: the fragments come from **the same generator** as ① (only the uniform name changes from
    // `params` to `dims`). A copied implementation would let through a form where only one side of
    // the ring mapping, the window's lower bound or the effective rows is fixed
    assertEquals(wgsl.includes(stateSlotRowWgsl(sliding, "dims")), true, "slot_row");
    assertEquals(wgsl.includes("fn effective_rows(query: u32) -> u32 {"), true, "effective_rows");
    assertEquals(wgsl.includes("bitcast<f32>(dims.neg_inf)"), true, "−inf outside the predicate");
    // MUST: no addition form that wraps around in u32 (same as ①)
    assertEquals(
      wgsl.includes("col + dims.window"),
      false,
      "the addition form that wraps around in u32",
    );
    // The effective N is live. Cutting the B tile at `col_cap` (= dims.n) would read leftovers
    assertEquals(wgsl.includes("if (wcol0 < live) {"), true, "the B loader's live cut");
    assertEquals(wgsl.includes("if (ocol < live) {"), true, "the store's live cut");
    // The half scale is on **both** the q side and the k side (the same rounding sequence as ①'s
    // single-term formula)
    assertEquals(wgsl.includes("* dims.scale;"), true, "the A side's half scale");
    assertEquals(wgsl.includes("wv0 = wv0 * dims.scale;"), true, "the B side's half scale");
    // The two sources of K are folded into one place (the same split as ①'s score_slot / score_ins)
    assertEquals(wgsl.includes("fn k_read(from_slot: bool, index: u32) -> f32 {"), true, "k_read");
  }
  assertEquals(stateQkTiledWgsl(true, false, 16).includes("(limit - col) < dims.window"), true);
  assertEquals(stateQkTiledWgsl(false, false, 16).includes("(limit - col) < dims.window"), false);
  // The variants and buckets are really different (keys split but identical contents = the variant
  // / geometry has no effect)
  assertNotEquals(stateQkTiledWgsl(false, false, 16), stateQkTiledWgsl(false, true, 16));
  assertNotEquals(stateQkTiledWgsl(false, false, 16), stateQkTiledWgsl(true, false, 16));
  assertNotEquals(stateQkTiledWgsl(false, false, 16), stateQkTiledWgsl(false, false, 768));
});

test("the params of ①ₜ are packed in word order (the skeleton's m/n/k are rows_block / col_cap / depth)", () => {
  const params = stateQkTiledParams({ ...GEOMETRY, rowOffset: 4 });
  assertEquals(params.length, 12, "48 bytes with uniform alignment");
  assertEquals(
    [...params.slice(0, 9)],
    [
      GEOMETRY.rowsBlock,
      GEOMETRY.colCap,
      GEOMETRY.depth,
      4,
      GEOMETRY.chunkRows,
      GEOMETRY.kvRepeat,
      GEOMETRY.window,
      GEOMETRY.capacity,
      STATE_NEG_INF_BITS,
    ],
  );
  assertEquals(new Float32Array(params.buffer)[9], GEOMETRY.scale);
  // One value-range check shared with ① (holding a separate one would allow a form that passes only
  // one of them)
  assertThrows(() => stateQkTiledParams({ ...GEOMETRY, kvRepeat: 0 }), CodegenError, "kv_repeat");
  assertThrows(
    () => stateQkTiledParams({ ...GEOMETRY, rowOffset: 6, rowsBlock: 4 }),
    CodegenError,
    "row block",
  );
  assertThrows(() => stateQkTiledParams({ ...GEOMETRY, colCap: 31 }), CodegenError, "col_cap");
});

test("the dispatch edges of ①ₜ come from the M geometry bucket (not carried around as constants)", () => {
  // M ≤ 64 → M16N16. Columns ⌈40/16⌉ = 3, rows ⌈40/16⌉ = 3
  assertEquals(stateQkTiledWorkgroups(TILED_DISPATCH, 40, 0, 40, LIMIT, "t"), [3, 3, 6]);
  // 65..512 → M64N32. Columns ⌈40/32⌉ = 2, rows ⌈40/64⌉ = 1
  assertEquals(stateQkTiledWorkgroups(TILED_DISPATCH, 100, 0, 40, LIMIT, "t"), [2, 1, 6]);
  // 513.. → the default M128N128. One tile for both columns and rows
  assertEquals(stateQkTiledWorkgroups(TILED_DISPATCH, 768, 0, 40, LIMIT, "t"), [1, 1, 6]);
  // The column axis is live (independent of capacity); the row axis is the effective rows (does not
  // cover pad rows)
  assertEquals(stateQkTiledWorkgroups(TILED_DISPATCH, 40, 88, 40, LIMIT, "t")[0], 8, "live 128");
  const tail = { ...TILED_DISPATCH, rowsBlock: 40, rowOffset: 40 };
  assertEquals(stateQkTiledWorkgroups(tail, 40, 0, 40, LIMIT, "t")[1], 0, "entirely pad is 0");
  assertEquals(stateQkTiledWorkgroups(tail, 40, 0, 41, LIMIT, "t")[1], 1, "only one row effective");
  // MUST: exceeding the limit fails loudly (a tile kind — degrading would leave S tiles missing and
  // ②③ would read leftovers)
  assertThrows(
    () => stateQkTiledWorkgroups(TILED_DISPATCH, 40, 4096, 1, 8, "t"),
    DispatchLimitError,
  );
});

test("the condition for ①ₜ is M ≥ 16 (the smallest row-tile edge) — it does not overlap with M=1 of ①'", () => {
  // The tileM of the smallest bucket M16N16 is 16. Below that, most of the tile does no work: the
  // same traffic as ①, plus the cost of barriers and shared memory
  for (const chunkRows of [1, 2, 4, 8, 15]) {
    assertEquals(stateQkTiledEligible(chunkRows), false, `M=${chunkRows} stays on ①`);
  }
  for (const chunkRows of [16, 17, 40, 100, 768]) {
    assertEquals(stateQkTiledEligible(chunkRows), true, `M=${chunkRows} uses ①ₜ`);
  }
  // MUST: the two applicability conditions do not overlap (if they did, the runtime side's priority
  // order would affect the values)
  for (const chunkRows of [1, 2, 15, 16, 768]) {
    assertEquals(
      stateQkTiledEligible(chunkRows) && stateQkParallelEligible(chunkRows),
      false,
      `the conditions of ①ₜ and ①' overlapped at M=${chunkRows}`,
    );
  }
});

test("the key of ③ₜ carries the M geometry bucket (the geometry itself, not just the edge)", () => {
  // The spelling is identical to gemm's geometry fragment
  // (`reg{tileM}x{tileN}r{regM}x{regN}w{wgX}`) and differs from ①ₜ only in the family name. The
  // variant bits are appended in the same order as for ③
  assertEquals(statePvTiledKey(false, false, 16), "attention_state_pv:v1:f32:reg16x16r1x4w4");
  assertEquals(
    statePvTiledKey(true, true, 16),
    "attention_state_pv:v1:f32:reg16x16r1x4w4:sliding:gqa",
  );
  // Crossing a bucket boundary (64 / 512) **changes the key** (= a different pipeline). Otherwise,
  // WGSL of a different geometry would be handed out of the cache under the same key, would not
  // mesh with the dispatch edge, and tiles would go missing
  assertEquals(statePvTiledKey(false, false, 64), statePvTiledKey(false, false, 16));
  assertEquals(statePvTiledKey(false, false, 65), "attention_state_pv:v1:f32:reg64x32r4x4w8");
  assertEquals(statePvTiledKey(false, false, 512), statePvTiledKey(false, false, 65));
  assertEquals(statePvTiledKey(false, false, 513), "attention_state_pv:v1:f32:reg128x128r8x8w16");
  assertEquals(statePvTiledKey(false, false, 768), statePvTiledKey(false, false, 513));
  // All three paths of ③ have different keys (one configuration does not have two keys / two
  // configurations do not share a key)
  const keys = [
    ...VARIANTS.map(([sliding, gqa]) => statePvKey(sliding, gqa)),
    ...VARIANTS.map(([sliding, gqa]) => statePvParallelKey(sliding, gqa)),
    ...VARIANTS.flatMap(([sliding, gqa]) =>
      [16, 100, 768].map((rows) => statePvTiledKey(sliding, gqa, rows)),
    ),
  ];
  assertEquals(new Set(keys).size, keys.length);
  // No collision with ①ₜ either (obvious, since the family names differ, but sharing the geometry
  // fragment makes a mix-up possible)
  assertNotEquals(statePvTiledKey(false, false, 16), stateQkTiledKey(false, false, 16));
});

test("③ₜ shares the contract fragments of ③ (shared read/write formula, effective rows, the probability formula) with only the uniform name changed", () => {
  for (const [sliding, gqa] of VARIANTS) {
    const wgsl = statePvTiledWgsl(sliding, gqa, 16);
    assertEquals(statePvTiledWgsl(sliding, gqa, 16), wgsl, "determinism");
    // MUST: the fragments come from **the same generator** as ③ (only the uniform name changes from
    // `params` to `dims`). A copied implementation would let through a form where only one side of
    // the ring mapping, the live cut or the effective rows is fixed
    assertEquals(wgsl.includes(stateSlotRowWgsl(sliding, "dims")), true, "slot_row");
    assertEquals(wgsl.includes("fn effective_rows(query: u32) -> u32 {"), true, "effective_rows");
    // MUST: it has no predicate (`in_window`) — outside the predicate, exp turns the −inf that ①
    // burned into S into 0
    assertEquals(
      wgsl.includes("fn in_window("),
      false,
      "the ③ kernels do not evaluate the predicate",
    );
    assertEquals(wgsl.includes("neg_inf"), false, "the ③ kernels do not read −inf");
    assertEquals(wgsl.includes("dims.scale"), false, "the ③ kernels do not read the half scale");
    // The single-term formula has the same text as ③ (`exp(S − m) * inv`)
    assertEquals(wgsl.includes(") * row_inv0;"), true, "the probability's inv");
    assertEquals(
      wgsl.includes("exp(s[arow_base0 + ak0] - row_max0)"),
      true,
      "the probability's exp",
    );
    // The effective K is live. Cutting A / B at `col_cap` (= dims.k) would read the leftovers of
    // `[live, col_cap)`
    assertEquals(wgsl.includes("if (brow0 < live) {"), true, "the B loader's live cut");
    assertEquals(
      wgsl.includes("let tiles = (k_live + 15u) / 16u;"),
      true,
      "the K loop's live limit",
    );
    // Pad rows stay full-write and are **exactly 0** (writing acc would turn them NaN with a
    // non-finite V)
    assertEquals(wgsl.includes("if (orow0 < dims.m) {"), true, "the store's full-write");
    assertEquals(wgsl.includes("select(0.0, acc0_0.x, live_row0);"), true, "the pad rows' exact 0");
    // The two sources of V are folded into one place (the same split as ③'s slot_v / ins_v)
    assertEquals(wgsl.includes("fn v_read(from_slot: bool, index: u32) -> f32 {"), true, "v_read");
  }
  // The variants and buckets are really different (keys split but identical contents = the variant
  // / geometry has no effect)
  assertNotEquals(statePvTiledWgsl(false, false, 16), statePvTiledWgsl(false, true, 16));
  assertNotEquals(statePvTiledWgsl(false, false, 16), statePvTiledWgsl(true, false, 16));
  assertNotEquals(statePvTiledWgsl(false, false, 16), statePvTiledWgsl(false, false, 768));
});

test("the params of ③ₜ are packed in word order (the skeleton's m/n/k are rows_block / depth / col_cap)", () => {
  const params = statePvTiledParams({ ...GEOMETRY, rowOffset: 4 });
  assertEquals(params.length, 8, "32 bytes with uniform alignment");
  assertEquals(
    [...params],
    [
      GEOMETRY.rowsBlock,
      GEOMETRY.depth,
      GEOMETRY.colCap,
      4,
      GEOMETRY.chunkRows,
      GEOMETRY.kvRepeat,
      GEOMETRY.window,
      GEOMETRY.capacity,
    ],
  );
  // One value-range check shared with ③ (holding a separate one would allow a form that passes only
  // one of them)
  assertThrows(() => statePvTiledParams({ ...GEOMETRY, kvRepeat: 0 }), CodegenError, "kv_repeat");
  assertThrows(
    () => statePvTiledParams({ ...GEOMETRY, rowOffset: 6, rowsBlock: 4 }),
    CodegenError,
    "row block",
  );
  assertThrows(() => statePvTiledParams({ ...GEOMETRY, colCap: 31 }), CodegenError, "col_cap");
});

test("the dispatch edges of ③ₜ come from the M geometry bucket, and the row axis is the whole rows_block", () => {
  // M ≤ 64 → M16N16. Columns ⌈32/16⌉ = 2, rows ⌈40/16⌉ = 3
  assertEquals(statePvTiledWorkgroups(TILED_DISPATCH, 40, LIMIT, "t"), [2, 3, 6]);
  // 65..512 → M64N32. Columns ⌈32/32⌉ = 1, rows ⌈40/64⌉ = 1
  assertEquals(statePvTiledWorkgroups(TILED_DISPATCH, 100, LIMIT, "t"), [1, 1, 6]);
  // 513.. → the default M128N128. One tile for both columns and rows
  assertEquals(statePvTiledWorkgroups(TILED_DISPATCH, 768, LIMIT, "t"), [1, 1, 6]);
  // MUST: the row axis is **the whole rows_block** (unlike the ① kernels, it is not cut at the
  // effective rows — writing pad rows too is the full-write invariant). Not taking the logical
  // length at all is its structural backing
  const tail = { ...TILED_DISPATCH, rowsBlock: 40, rowOffset: 40 };
  assertEquals(
    statePvTiledWorkgroups(tail, 40, LIMIT, "t")[1],
    3,
    "3 tiles even when entirely pad",
  );
  // MUST: exceeding the limit fails loudly (a tile kind — degrading would leave unwritten parts of
  // O)
  assertThrows(
    () => statePvTiledWorkgroups({ ...TILED_DISPATCH, depth: 4096 }, 40, 8, "t"),
    DispatchLimitError,
  );
});

test("the condition for ③ₜ is M ≥ 16 (the smallest row-tile edge) — ③' remains only for plans with M < 16", () => {
  // The tileM of the smallest bucket M16N16 is 16. Below that, most of the tile does no work: the
  // same traffic as ③, plus the cost of barriers and shared memory
  for (const chunkRows of [1, 2, 4, 8, 15]) {
    assertEquals(
      statePvTiledEligible(chunkRows),
      false,
      `M=${chunkRows} uses ③ / ③' as the option says`,
    );
  }
  for (const chunkRows of [16, 17, 40, 100, 768]) {
    assertEquals(statePvTiledEligible(chunkRows), true, `M=${chunkRows} uses ③ₜ`);
  }
  // The same threshold as ①ₜ but a separate function (keeps the form where either one's range can
  // move alone)
  for (const chunkRows of [1, 15, 16, 768]) {
    assertEquals(
      statePvTiledEligible(chunkRows),
      stateQkTiledEligible(chunkRows),
      `the thresholds of ①ₜ and ③ₜ diverged at M=${chunkRows}`,
    );
  }
  // decode (M=1) is outside ③ₜ, so ③' of the option value `"parallel"` stays there
  assertEquals(statePvTiledEligible(1), false);
});
