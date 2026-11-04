// Gates for the two row-tile-sharing (GEMM skeleton) variants of the states-form attention. Both
// enter the default path on one ground only — bit identity with their sequential reference — so
// both are judged by word (u32) equality of the buffer they write, over one shared case grid, one
// shared input builder and one shared fault-injection harness:
//   ①ₜ = ①QK with K-row tile sharing (perf-ledger K-13), compared on the S buffer
//   ③ₜ = ③PV with V-row tile sharing (perf-ledger K-13 stage 2), compared on the O buffer
//
// --- ①ₜ ---
// Checks for the **K-row tile-sharing variant ①ₜ** of the states-form ①QK (GEMM skeleton —
// perf-ledger K-13).
//
// ①ₜ's entry condition is "**bit-identical** to ①" (unlike ①', which changes the reduction order,
// this alone is the ground for it becoming the default path regardless of the preset). So what is
// checked here is not a band but **exact word (u32) equality of the S buffer**; the correctness of
// the values themselves (comparison with the f64 reference) stays with the checks on the ① side
// (gpu_state_attention_test.ts) — ①ₜ inherits that by "not differing from ① by a single bit".
//
// The words compared are of three kinds, none of which f32 `===` can compare:
//   ① the **−inf** outside the predicate (`0xff800000`)
//   ② the **leftovers** in `[live, col_cap)` and the **non-writes** of pad rows (the poison 2¹⁰⁰
//      still in place)
//   ③ the real values inside the predicate
// "The number of words written" is counted independently from the spec and compared, so a shape
// where both paths write nothing cannot pass this check (the structural evidence that the check
// fires).
//
// --- ③ₜ ---
// Checks for the **V-row tile-sharing variant ③ₜ** of the states-form ③PV (GEMM skeleton —
// perf-ledger K-13 stage 2).
//
// ③ₜ's entry condition is "**bit-identical** to ③" (unlike ③', which changes the reduction order,
// this alone is the ground for it becoming the default path regardless of the preset). So what is
// checked here is not a band but **exact word (u32) equality of the O buffer**; the correctness of
// the values themselves (comparison with the f64 reference) stays with the checks on the ③ side
// (gpu_state_attention_test.ts) — ③ₜ inherits that by "not differing from ③ by a single bit".
//
// Words are compared because values that f32 `===` cannot compare occur legitimately (the exact 0
// of pad rows, and the NaN / ±0 distinction a broken implementation produces). In addition, O is
// **full-write** (all `rows_block` rows × D), so "not a single poison word remains" and "pad rows
// are exactly 0" can be counted independently from the spec alone — a shape where both paths write
// nothing cannot pass this check.
//
// --- shared grid ---
// Grid: {full, sliding} × {r=1, r=2} × M ∈ {16, 40, 100, 768} (spanning the three geometry
// buckets) × {P=0 (ins only), P>0 (2 sources), P≥W (ring wrap)} × {with / without pad rows} ×
// {1 row block / several (including blocks with no valid row)} × {live not a multiple of the
// K tile 16} × {D not a multiple of the column tile edge}.
// Both variants run every case of the grid.
//
// MUST: show in the same file that the checks fail under fault injection (①ₜ: removing the B
// loader's half scale, mixing up the two sources of the column base / ③ₜ: removing the
// probabilities' inv, mixing up the two sources of V). A check that never fails verifies nothing.

import { test } from "vite-plus/test";
import { assert, assertEquals } from "@std/assert";
import { acquireGpu } from "../src/gpu/device.ts";
import {
  STATE_NEG_INF_BITS,
  stateColumnBase,
  stateEffectiveRows,
  stateLiveColumns,
  stateSliding,
} from "../src/kernels/state-attention.ts";
import {
  assertMutated,
  caseColCap,
  runStatePv,
  runStateQk,
  seeded,
  STATE_S_POISON,
  type StateCase,
  type StateInputs,
  type StateMutation,
  type StatePipelineCache,
} from "./helpers/state-dispatch.ts";
import { GPU_AVAILABLE } from "./helpers/gpu.ts";

/**
 * Deterministic input sequences (the same generation rules as gpu_state_attention_test.ts — no
 * random numbers).
 */
const QUERY = (i: number): number => (((i * 7) % 23) - 11) * 0.17;
const KEY = (i: number): number => (((i * 11) % 19) - 9) * 0.23;
const VALUE = (i: number): number => (((i * 5) % 17) - 8) * 0.31;

/** Poison values placed where nothing may read (the values jump if they are read). */
const SLOT_POISON_K = 9;
const SLOT_POISON_V = 400;

/** The bit pattern of the S / O poison (2¹⁰⁰ is exact in f32, so word comparison works). */
const POISON_BITS = new Uint32Array(new Float32Array([STATE_S_POISON]).buffer)[0];

/** A word as `0x%08x` for failure messages. */
const hex = (word: number): string => `0x${word.toString(16).padStart(8, "0")}`;

/**
 * Builds the five buffers ①③ read (the same poison rules as `makeInputs` in
 * gpu_state_attention_test.ts).
 *
 * MUST: put poison in **non-resident physical slot rows** and in **the pad rows of ins**. ①ₜ reads
 * K even for columns outside the predicate (the rows it reads are within the live range, so this is
 * safe by contract), so a "read beyond the range" error shows in the values only through this
 * poison. ③ₜ reads V by loading it into row tiles, so a "loaded out-of-range V rows at the tile
 * edge" error shows in the values only through this poison (even where the probability p is 0,
 * `0 · 400` is 0, so **a different row** always moves the values).
 */
const makeInputs = (spec: StateCase): StateInputs => {
  const { batch, heads, kvHeads, chunkRows, depth, capacity, window, past, query } = spec;
  const kvPlanes = batch * kvHeads;
  const q = seeded(batch * heads * chunkRows * depth, QUERY);
  const insK = seeded(kvPlanes * chunkRows * depth, KEY);
  const insV = seeded(kvPlanes * chunkRows * depth, VALUE);
  const slotK = seeded(kvPlanes * capacity * depth, (i) => KEY(i + 3));
  const slotV = seeded(kvPlanes * capacity * depth, (i) => VALUE(i + 5));
  const resident = new Set<number>();
  const base = stateColumnBase(window, past);
  for (let col = base; col < past; col += 1) {
    resident.add(stateSliding(window) ? col % capacity : col);
  }
  for (let plane = 0; plane < kvPlanes; plane += 1) {
    for (let row = 0; row < capacity; row += 1) {
      if (resident.has(row)) continue;
      for (let d = 0; d < depth; d += 1) {
        slotK[(plane * capacity + row) * depth + d] = SLOT_POISON_K;
        slotV[(plane * capacity + row) * depth + d] = SLOT_POISON_V;
      }
    }
    for (let row = query; row < chunkRows; row += 1) {
      for (let d = 0; d < depth; d += 1) {
        insK[(plane * chunkRows + row) * depth + d] = SLOT_POISON_K;
        insV[(plane * chunkRows + row) * depth + d] = SLOT_POISON_V;
      }
    }
  }
  return { q, insK, insV, slotK, slotV };
};

/**
 * The single-case probe for the invariance and fault-injection tests: P > 0 (2 sources), not
 * sliding, GQA.
 */
const PROBE_CASE: StateCase = {
  name: "full r2 M16 P13 (GQA, 2 sources)",
  batch: 1,
  heads: 4,
  kvHeads: 2,
  chunkRows: 16,
  depth: 16,
  capacity: 64,
  window: 0,
  past: 13,
  query: 16,
};

/**
 * The grid. It spans **the three geometry buckets** (M ≤ 64 = M16N16 / 65..512 = M64N32 / above
 * that = the default M128N128), and in every bucket mixes in shapes where `live` and the valid rows
 * (in ③ₜ, `live` (= K) and `D` (= N)) are not multiples of the tile edge.
 *
 * The axes swap roles between the variants: in ①ₜ `live` is the column (N) axis and `D` the
 * reduction (K) axis, in ③ₜ `live` is the reduction axis and `D` the column axis. A case name that
 * speaks of "K tile" / "column edge" describes the variant it was first written for; every case
 * runs on both.
 *
 * MUST: full satisfies `P + Q ≤ C` (the same condition as the pre-run validation) (violating it
 * leaves `col_cap` short of live, and the value-range checks on the ①③ side fail).
 */
const CASES: readonly StateCase[] = [
  // --- smallest bucket M16N16 (tileM = tileN = 16) ---
  {
    name: "full r1 M16 P0 (ins only, live = exactly the tile edge)",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 16,
    depth: 16,
    capacity: 64,
    window: 0,
    past: 0,
    query: 16,
  },
  {
    name: "full r1 M16 P13 (2 sources, live 29 = not a multiple of the edge)",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 16,
    depth: 16,
    capacity: 64,
    window: 0,
    past: 13,
    query: 16,
  },
  PROBE_CASE,
  {
    name: "full r2 M16 P13 B2（batch > 1）",
    batch: 2,
    heads: 4,
    kvHeads: 2,
    chunkRows: 16,
    depth: 16,
    capacity: 64,
    window: 0,
    past: 13,
    query: 16,
  },
  {
    name: "full r1 M16 D24 (D not a multiple of the K tile 16)",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 16,
    depth: 24,
    capacity: 64,
    window: 0,
    past: 13,
    query: 16,
  },
  {
    name: "full r2 M16 P13 D24 (D not a multiple of the column edge 16)",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 16,
    depth: 24,
    capacity: 64,
    window: 0,
    past: 13,
    query: 16,
  },
  {
    name: "full r1 M40 P0 (40 valid rows = not a multiple of the row edge 16)",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 40,
    depth: 16,
    capacity: 128,
    window: 0,
    past: 0,
    query: 40,
  },
  {
    name: "full r2 M40 P37 Q17 (with pad rows, 2 sources)",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 40,
    depth: 16,
    capacity: 128,
    window: 0,
    past: 37,
    query: 17,
  },
  {
    name: "full r2 M40 P37 Q17 row block 12 (including blocks with no valid row)",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 40,
    depth: 16,
    capacity: 128,
    window: 0,
    past: 37,
    query: 17,
    rowsBlock: 12,
  },
  {
    name: "full r2 M40 P37 row block 12 (split, no pad rows)",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 40,
    depth: 16,
    capacity: 128,
    window: 0,
    past: 37,
    query: 40,
    rowsBlock: 12,
  },
  // --- sliding (lower-bound predicate + ring mapping; −inf outside the predicate as p = 0) ---
  {
    name: "sliding r1 M16 W8 P0 (window < Q, no past)",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 16,
    depth: 16,
    capacity: 8,
    window: 8,
    past: 0,
    query: 16,
  },
  {
    name: "sliding r1 M16 W8 P20 (across a ring wrap, live = col_cap)",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 16,
    depth: 16,
    capacity: 8,
    window: 8,
    past: 20,
    query: 16,
  },
  {
    name: "sliding r2 M16 W8 P5 (GQA, past < W−1)",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 16,
    depth: 16,
    capacity: 8,
    window: 8,
    past: 5,
    query: 16,
  },
  {
    name: "sliding r2 M40 W12 P30 Q23 (pad rows + ring wrap)",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 40,
    depth: 16,
    capacity: 12,
    window: 12,
    past: 30,
    query: 23,
  },
  // --- medium-M bucket M64N32 (tileM 64 / tileN 32) ---
  {
    name: "full r2 M100 P0（live 100）",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 100,
    depth: 16,
    capacity: 256,
    window: 0,
    past: 0,
    query: 100,
  },
  {
    name: "full r2 M100 P37 Q60 (pad rows + 2 sources)",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 100,
    depth: 16,
    capacity: 256,
    window: 0,
    past: 37,
    query: 60,
  },
  {
    name: "full r2 M100 P37 Q60 D40 (pad rows + D not a multiple of the column edge 32)",
    batch: 1,
    heads: 4,
    kvHeads: 2,
    chunkRows: 100,
    depth: 40,
    capacity: 256,
    window: 0,
    past: 37,
    query: 60,
  },
  {
    name: "sliding r1 M100 W20 P64 row block 33 (split)",
    batch: 1,
    heads: 2,
    kvHeads: 2,
    chunkRows: 100,
    depth: 16,
    capacity: 20,
    window: 20,
    past: 64,
    query: 100,
    rowsBlock: 33,
  },
  // --- default bucket M128N128 (the observed prefill shape M = 768) ---
  {
    name: "full r1 M768 P0 (default geometry, 6 row tiles)",
    batch: 1,
    heads: 1,
    kvHeads: 1,
    chunkRows: 768,
    depth: 16,
    capacity: 800,
    window: 0,
    past: 0,
    query: 768,
  },
  {
    name: "full r1 M768 Q100 (100 valid rows = only the first row tile of 128)",
    batch: 1,
    heads: 1,
    kvHeads: 1,
    chunkRows: 768,
    depth: 16,
    capacity: 800,
    window: 0,
    past: 0,
    query: 100,
  },
  {
    name: "sliding r2 M768 W64 P500 row block 256 (split, 2 sources)",
    batch: 1,
    heads: 2,
    kvHeads: 1,
    chunkRows: 768,
    depth: 16,
    capacity: 64,
    window: 64,
    past: 500,
    query: 768,
    rowsBlock: 256,
  },
];

/**
 * Inputs for two runs with different capacities C (the flip side of the work condition of ADR 0066
 * decision 3 — the output does not depend on the capacity).
 *
 * MUST: the large-capacity slots get **the same values in the same physical rows** (the generator
 * depends on the index, so regenerating at another capacity would shift each plane's origin and
 * compare "two runs with different inputs").
 */
const capacityPair = (
  spec: StateCase,
): {
  readonly small: StateCase;
  readonly large: StateCase;
  readonly smallInputs: StateInputs;
  readonly largeInputs: StateInputs;
} => {
  const small: StateCase = { ...spec, name: `${spec.name} C64`, capacity: 64 };
  const large: StateCase = { ...spec, name: `${spec.name} C160`, capacity: 160 };
  const smallInputs = makeInputs(small);
  const relayout = (source: Float32Array<ArrayBuffer>, poison: number) => {
    const planes = small.batch * small.kvHeads;
    const grown = seeded(planes * large.capacity * small.depth, () => poison);
    for (let plane = 0; plane < planes; plane += 1) {
      for (let row = 0; row < small.capacity; row += 1) {
        for (let d = 0; d < small.depth; d += 1) {
          grown[(plane * large.capacity + row) * small.depth + d] =
            source[(plane * small.capacity + row) * small.depth + d];
        }
      }
    }
    return grown;
  };
  const largeInputs: StateInputs = {
    ...smallInputs,
    slotK: relayout(smallInputs.slotK, SLOT_POISON_K),
    slotV: relayout(smallInputs.slotV, SLOT_POISON_V),
  };
  return { small, large, smallInputs, largeInputs };
};

/**
 * Exact word-level equality (not a single difference from ① / ③ is allowed). `symbols` is the
 * pair of variant marks printed on a mismatch (tiled first, reference second).
 */
const assertSameWords = (
  actual: readonly Uint32Array<ArrayBuffer>[],
  expected: readonly Uint32Array<ArrayBuffer>[],
  label: string,
  symbols: readonly [tiled: string, reference: string],
): void => {
  assertEquals(actual.length, expected.length, `${label}: number of row blocks`);
  for (let block = 0; block < actual.length; block += 1) {
    const got = actual[block];
    const want = expected[block];
    assertEquals(got.length, want.length, `${label}: word count of block ${block}`);
    for (let index = 0; index < got.length; index += 1) {
      if (got[index] === want[index]) continue;
      throw new Error(
        `${label}: block ${block}, word ${index} differs (${symbols[0]} ${hex(got[index])} / ${symbols[1]} ${hex(want[index])})`,
      );
    }
  }
};

/**
 * Fault injection shared by both variants: each mutation is applied to `kernel` only and MUST make
 * the tiled run differ from the reference in at least one word.
 *
 * MUST: apply the mutation only to the target kernel (dragging ①② into a ③ₜ mutation would pass
 * as "S differs, so O differs too", which says nothing about an error in ③ₜ itself).
 */
const assertMutationsBreakIdentity = async (
  kernel: "qk" | "pv",
  reference: readonly Uint32Array<ArrayBuffer>[],
  runTiled: (mutate: StateMutation) => Promise<readonly Uint32Array<ArrayBuffer>[]>,
  mutations: readonly (readonly [label: string, mutate: (wgsl: string) => string])[],
  missed: (label: string) => string,
): Promise<void> => {
  for (const [label, mutate] of mutations) {
    let mutated = false;
    const broken = await runTiled((target, wgsl) => {
      if (target !== kernel) return wgsl;
      const after = mutate(wgsl);
      assertMutated(wgsl, after, label);
      mutated = true;
      return after;
    });
    assert(mutated, `${label}: the mutation was never called`);
    const differs = broken.some((words, block) =>
      words.some((word, index) => word !== reference[block][index]),
    );
    assert(differs, missed(label));
  }
};

// ---------------------------------------------------------------------------------------------
// ①ₜ — S buffer
// ---------------------------------------------------------------------------------------------

const QK_SYMBOLS = ["①ₜ", "①"] as const;

/** The row-block split of one case (the same expression as `runStateQk`'s traversal). */
const blocksOf = (spec: StateCase): readonly { rows: number; offset: number }[] => {
  const rowsBlock = spec.rowsBlock ?? spec.chunkRows;
  const blocks: { rows: number; offset: number }[] = [];
  for (let offset = 0; offset < spec.chunkRows; offset += rowsBlock) {
    blocks.push({ rows: Math.min(rowsBlock, spec.chunkRows - offset), offset });
  }
  return blocks;
};

/**
 * Counts S's words in two groups, **written / leftover** (the structural evidence that the check
 * fires).
 *
 * The number of words written is determined by the spec alone (`B·H × valid rows × live`), so the
 * expected value needs no reference to the implementation at all. The rest must still be poison.
 */
const assertWriteCoverage = (
  spec: StateCase,
  words: Uint32Array<ArrayBuffer>,
  block: { rows: number; offset: number },
  label: string,
): void => {
  const batchHeads = spec.batch * spec.heads;
  const live = stateLiveColumns(spec.window, spec.past, spec.query);
  const rows = stateEffectiveRows(block.rows, block.offset, spec.query);
  let poison = 0;
  for (let index = 0; index < words.length; index += 1) {
    if (words[index] === POISON_BITS) poison += 1;
  }
  assertEquals(
    words.length - poison,
    batchHeads * rows * live,
    `${label}: the number of words written differs from B·H × valid rows × live`,
  );
};

/**
 * Folds into a form independent of the row-block split — a word sequence of **only the rows that
 * are written**, in the order `(z, global row within the chunk, live column)`. Splitting into row
 * blocks changes S's row stride (`rows_block`), so comparing two runs with different splits needs
 * this normalization.
 */
const liveRows = (spec: StateCase, blocks: readonly Uint32Array<ArrayBuffer>[]): number[] => {
  const colCap = caseColCap(spec);
  const batchHeads = spec.batch * spec.heads;
  const live = stateLiveColumns(spec.window, spec.past, spec.query);
  const out: number[] = [];
  for (let z = 0; z < batchHeads; z += 1) {
    blocksOf(spec).forEach((block, index) => {
      const rows = stateEffectiveRows(block.rows, block.offset, spec.query);
      for (let local = 0; local < rows; local += 1) {
        const base = (z * block.rows + local) * colCap;
        for (let col = 0; col < live; col += 1) out.push(blocks[index][base + col]);
      }
    });
  }
  return out;
};

test.skipIf(!GPU_AVAILABLE)(
  "①ₜ (K-row tile sharing) matches ① word for word in the S buffer (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    // Count "was it seen" over the whole grid. For both −inf and leftovers, **zero words is correct
    // in some cases** (e.g. live = col_cap with no pad rows), so check over the grid, not per case.
    let negInf = 0;
    let poison = 0;
    let value = 0;
    try {
      for (const spec of CASES) {
        const inputs = makeInputs(spec);
        const reference = await runStateQk(gpu.device, spec, inputs, "sequential", { cache });
        const tiled = await runStateQk(gpu.device, spec, inputs, "tiled", { cache });
        assertSameWords(tiled, reference, spec.name, QK_SYMBOLS);
        // The check fires: the number of words written matches the spec
        // (`B·H × valid rows × live`). A shape where both paths write nothing fails here.
        blocksOf(spec).forEach((block, index) => {
          assertWriteCoverage(spec, tiled[index], block, `${spec.name} ①ₜ block ${index}`);
          assertWriteCoverage(spec, reference[index], block, `${spec.name} ① block ${index}`);
        });
        for (const words of tiled) {
          for (const word of words) {
            if (word === STATE_NEG_INF_BITS) negInf += 1;
            else if (word === POISON_BITS) poison += 1;
            else value += 1;
          }
        }
      }
    } finally {
      gpu.destroy();
    }
    assert(negInf > 0, "no case in the grid produced −inf outside the predicate");
    assert(poison > 0, "no case in the grid produced leftovers ([live, col_cap) / pad rows)");
    assert(value > 0, "no case in the grid wrote a single real value");
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "①ₜ's written words do not move across two dispatches, capacity C or the row-block split (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      const spec = PROBE_CASE;
      const inputs = makeInputs(spec);
      const first = await runStateQk(gpu.device, spec, inputs, "tiled", { cache });
      const second = await runStateQk(gpu.device, spec, inputs, "tiled", { cache });
      assertSameWords(first, second, `${spec.name} (second run)`, QK_SYMBOLS);

      // Changing the capacity C does not move the values (S's contents do not depend on the
      // capacity). For full, `col_cap = C`, so the row stride changes → extract only the live range
      // and compare.
      const { small, large, smallInputs, largeInputs } = capacityPair(spec);
      const live = stateLiveColumns(spec.window, spec.past, spec.query);
      const rows = spec.batch * spec.heads * spec.chunkRows;
      const liveRange = (words: Uint32Array<ArrayBuffer>, colCap: number): number[] => {
        const out: number[] = [];
        for (let row = 0; row < rows; row += 1) {
          for (let col = 0; col < live; col += 1) out.push(words[row * colCap + col]);
        }
        return out;
      };
      const smallWords = await runStateQk(gpu.device, small, smallInputs, "tiled", { cache });
      const largeWords = await runStateQk(gpu.device, large, largeInputs, "tiled", { cache });
      assertEquals(
        liveRange(smallWords[0], caseColCap(small)),
        liveRange(largeWords[0], caseColCap(large)),
        "changing the capacity C moved S in the live range",
      );

      // The row-block split does not move the values either (ADR 0067 decision 7). S's row stride
      // changes, so fold into `(z, global row, live column)` before comparing. Three ways: 1 block
      // / 3 blocks / with a partial block.
      const whole: StateCase = { ...spec, name: `${spec.name} 1 block` };
      const wholeWords = await runStateQk(gpu.device, whole, inputs, "tiled", { cache });
      assert(
        liveRows(whole, wholeWords).length > 0,
        "the folded word sequence is empty (the comparison never fires)",
      );
      for (const rowsBlock of [8, 5, 1]) {
        const split: StateCase = { ...spec, name: `${spec.name} split ${rowsBlock}`, rowsBlock };
        const splitWords = await runStateQk(gpu.device, split, inputs, "tiled", { cache });
        assertEquals(
          liveRows(split, splitWords),
          liveRows(whole, wholeWords),
          `splitting into row blocks of ${rowsBlock} moved S's words`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "①ₜ fault injection (half scale removed / two sources mixed up) fails the word-equality check",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    const spec = PROBE_CASE;
    const inputs = makeInputs(spec);
    try {
      await assertMutationsBreakIdentity(
        "qk",
        await runStateQk(gpu.device, spec, inputs, "sequential", { cache }),
        (mutate) => runStateQk(gpu.device, spec, inputs, "tiled", { cache, mutate }),
        [
          // ① Remove the B loader's half scale (scale is not applied on the k side only = one
          // term's expression differs from ①)
          [
            "remove the B loader's half scale",
            (wgsl) => wgsl.replace(/^ *(wv\d+) = \1 \* dims\.scale;\n/gm, ""),
          ],
          // ② Mix up the two sources of the column base (read every column from ins)
          [
            "mix up the two sources of the column base",
            (wgsl) => wgsl.replace(/(let kpast\d+ = kcol\d+) < past;/g, "$1 < 0u;"),
          ],
        ],
        (label) => `${label}: S still matched ① with the mutation (the check never fires)`,
      );
    } finally {
      gpu.destroy();
    }
  },
);

// ---------------------------------------------------------------------------------------------
// ③ₜ — O buffer
// ---------------------------------------------------------------------------------------------

const PV_SYMBOLS = ["③ₜ", "③"] as const;

/** The word count of O (`B·H × M × D` — independent of the row-block split). */
const outWords = (spec: StateCase): number => spec.batch * spec.heads * spec.chunkRows * spec.depth;

/** The per-word breakdown (the structural evidence that the check fires). */
type Coverage = {
  /** The word count of pad rows (`row ≥ Q`) — fails unless every one is exactly 0. */
  readonly pad: number;
  /**
   * The non-zero word count of valid rows (0 means an implementation that "computes nothing"
   * passes).
   */
  readonly liveNonZero: number;
};

/**
 * Checks O on two points: **full-write and the exact 0 of pad rows**.
 *
 * The expectation is determined by the spec alone (`B·H × M × D` and `Q`), so it can be written
 * without any reference to the implementation.
 * MUST: not a single poison word remains (③'s full-write invariant — ADR 0066 addendum 6).
 * MUST: pad rows are **exactly the +0.0 bit pattern** (③ writes 0 without scanning a single live
 * column). Neither `0x80000000` (−0.0) nor `NaN` is allowed — an implementation that mixes in
 * `0 · non-finite` fails here.
 */
const assertCoverage = (
  spec: StateCase,
  words: Uint32Array<ArrayBuffer>,
  label: string,
): Coverage => {
  assertEquals(words.length, outWords(spec), `${label}: word count of O`);
  const batchHeads = spec.batch * spec.heads;
  let pad = 0;
  let liveNonZero = 0;
  for (let z = 0; z < batchHeads; z += 1) {
    for (let row = 0; row < spec.chunkRows; row += 1) {
      for (let d = 0; d < spec.depth; d += 1) {
        const word = words[(z * spec.chunkRows + row) * spec.depth + d];
        assertEquals(
          word === POISON_BITS,
          false,
          `${label}: (z=${z}, row=${row}, d=${d}) is unwritten (full-write is broken)`,
        );
        if (row >= spec.query) {
          assertEquals(
            word,
            0,
            `${label}: pad row (z=${z}, row=${row}, d=${d}) is not exactly +0.0 (${hex(word)})`,
          );
          pad += 1;
        } else if (word !== 0) liveNonZero += 1;
      }
    }
  }
  return { pad, liveNonZero };
};

test.skipIf(!GPU_AVAILABLE)(
  "③ₜ (V-row tile sharing) matches ③ word for word in the O buffer (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    // Count "was it seen" over the whole grid. For the zeros of pad rows, **zero words is correct
    // in some cases** (shapes with Q = M), so check over the grid, not per case.
    let pad = 0;
    let liveNonZero = 0;
    try {
      for (const spec of CASES) {
        const inputs = makeInputs(spec);
        const reference = await runStatePv(gpu.device, spec, inputs, "sequential", { cache });
        const tiled = await runStatePv(gpu.device, spec, inputs, "tiled", { cache });
        assertSameWords([tiled], [reference], spec.name, PV_SYMBOLS);
        // The check fires: **both paths** satisfy full-write and the exact 0 of pad rows.
        const seen = assertCoverage(spec, tiled, `${spec.name} ③ₜ`);
        assertCoverage(spec, reference, `${spec.name} ③`);
        pad += seen.pad;
        liveNonZero += seen.liveNonZero;
      }
    } finally {
      gpu.destroy();
    }
    assert(pad > 0, "no case in the grid has a pad row (the zero-write check never fires)");
    assert(
      liveNonZero > 0,
      "the valid rows are all zero in every case of the grid (nothing computed)",
    );
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "③ₜ's O does not move across two dispatches, capacity C or the row-block split (real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    try {
      const spec = PROBE_CASE;
      const inputs = makeInputs(spec);
      const first = await runStatePv(gpu.device, spec, inputs, "tiled", { cache });
      const second = await runStatePv(gpu.device, spec, inputs, "tiled", { cache });
      assertSameWords([first], [second], `${spec.name} (second run)`, PV_SYMBOLS);

      // Changing the capacity C does not move O. O's shape is `[B·H, M, D]`, which does not
      // include C, so the words can be compared directly.
      const { small, large, smallInputs, largeInputs } = capacityPair(spec);
      assertSameWords(
        [await runStatePv(gpu.device, large, largeInputs, "tiled", { cache })],
        [await runStatePv(gpu.device, small, smallInputs, "tiled", { cache })],
        "changing the capacity C moved O",
        PV_SYMBOLS,
      );

      // The row-block split does not move the values either (ADR 0067 decision 7). O's shape does
      // not depend on the split, so compare directly. Four ways: 1 block / 3 blocks / with a
      // partial block / one row at a time.
      const whole = await runStatePv(gpu.device, spec, inputs, "tiled", { cache });
      for (const rowsBlock of [8, 5, 1]) {
        const split: StateCase = { ...spec, name: `${spec.name} split ${rowsBlock}`, rowsBlock };
        assertSameWords(
          [await runStatePv(gpu.device, split, inputs, "tiled", { cache })],
          [whole],
          `splitting into row blocks of ${rowsBlock} moved O`,
          PV_SYMBOLS,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

/**
 * A case that checks the exact 0 of pad rows **on its own**.
 *
 * In the grid above, ③ₜ writing `acc` into pad rows would pass unnoticed — the A tile of pad rows
 * is filled with 0, and with finite V `Σ 0 · v` is exactly +0.0, so whether `select(0.0, …)` is
 * there does not show in the values. A difference appears only when **V in a live column is
 * non-finite** (`0 · Inf = NaN`), which is the only point where it parts from ③'s "pad rows write 0
 * without scanning a single live column".
 *
 * MUST: use a shape where a row tile **spans valid rows and pad rows** (a row tile that is entirely
 * pad never runs the K loop, so acc stays 0 and again no difference shows). With M=40 / Q=17, the
 * second tile `[16, 32)` spans both against the smallest bucket's row tile edge 16.
 */
const PAD_INF_CASE: StateCase = {
  name: "full r2 M40 Q17 P0 (+Inf in V of a live column)",
  batch: 1,
  heads: 4,
  kvHeads: 2,
  chunkRows: 40,
  depth: 16,
  capacity: 128,
  window: 0,
  past: 0,
  query: 17,
};

test.skipIf(!GPU_AVAILABLE)(
  "③ₜ's pad rows are exactly 0 even with non-finite V in a live column (③'s contract, real GPU)",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    const spec = PAD_INF_CASE;
    const inputs = makeInputs(spec);
    // Set V of valid row 3 (= live column 3) to +Inf. The A of pad rows is 0, so an implementation
    // that writes `acc` puts `0 · Inf = NaN` into the pad rows.
    const infRow = 3;
    for (let plane = 0; plane < spec.batch * spec.kvHeads; plane += 1) {
      for (let d = 0; d < spec.depth; d += 1) {
        inputs.insV[(plane * spec.chunkRows + infRow) * spec.depth + d] = Number.POSITIVE_INFINITY;
      }
    }
    try {
      for (const variant of ["sequential", "tiled"] as const) {
        const words = await runStatePv(gpu.device, spec, inputs, variant, { cache });
        let nonFinite = 0;
        for (let z = 0; z < spec.batch * spec.heads; z += 1) {
          for (let row = 0; row < spec.chunkRows; row += 1) {
            for (let d = 0; d < spec.depth; d += 1) {
              const word = words[(z * spec.chunkRows + row) * spec.depth + d];
              if (row >= spec.query) {
                assertEquals(
                  word,
                  0,
                  `${variant}: pad row (z=${z}, row=${row}, d=${d}) is not exactly +0.0 (${hex(word)})`,
                );
              } else if ((word & 0x7f800000) === 0x7f800000) nonFinite += 1;
            }
          }
        }
        // The injection takes effect (if no non-finite value shows in the valid rows, the +Inf
        // never arrived).
        assert(
          nonFinite > 0,
          `${variant}: not a single non-finite word in the valid rows (the injection never fired)`,
        );
      }
    } finally {
      gpu.destroy();
    }
  },
);

test.skipIf(!GPU_AVAILABLE)(
  "③ₜ fault injection (probabilities' inv removed / two sources of V mixed up) fails the word-equality check",
  async () => {
    const gpu = await acquireGpu();
    const cache: StatePipelineCache = new Map();
    const spec = PROBE_CASE;
    const inputs = makeInputs(spec);
    try {
      await assertMutationsBreakIdentity(
        "pv",
        [await runStatePv(gpu.device, spec, inputs, "sequential", { cache })],
        async (mutate) => [await runStatePv(gpu.device, spec, inputs, "tiled", { cache, mutate })],
        [
          // ① Remove the probabilities' inv (`1/Σexp(S−m)`) (one term of the A loader takes a
          // different form from ③)
          ["remove the probabilities' inv", (wgsl) => wgsl.replace(/ \* row_inv\d+;/g, ";")],
          // ② Mix up the two sources of the columns (read every column from ins)
          [
            "mix up the two sources of V",
            (wgsl) => wgsl.replace(/(let vpast\d+ = vcol\d+) < past;/g, "$1 < 0u;"),
          ],
        ],
        (label) => `${label}: O still matched ③ with the mutation (the check never fires)`,
      );
    } finally {
      gpu.destroy();
    }
  },
);
